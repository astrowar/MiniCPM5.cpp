#include "ops_scalar_optimized.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

// =============================================================================
// PORTABLE SCALAR REFERENCE FOR Q4_K/Q6_K x Q8_K
// =============================================================================
//
// Design goals:
//   1. Never dequantize every weight to FP32 in the inner loop.
//   2. Quantize the activation x to Q8_K once.
//   3. Accumulate weight * activation products in integer arithmetic.
//   4. Apply floating-point block scales only once per 256-value super-block.
//   5. Use Q8_K.bsums for Q4_K minima and Q6_K's -32 zero-point correction.
//   6. Expose row kernels with no allocation/threading so another ISA backend
//      can translate the exact same math into vectors.
//   7. Reuse one Q8_K activation across Q/K/V and gate/up projections.
//
// IMPORTANT:
//   This is a reference implementation, not an attempt to beat SIMD with scalar
//   C++.  Keep the math equivalent when writing NEON/SVE/RVV/etc. kernels.
// =============================================================================

namespace {

inline int8_t quantize_q8_value(float x, float inv_d) {
    // AVX2 _mm256_cvtps_epi32 uses the active FP rounding mode (normally
    // round-to-nearest, ties-to-even). nearbyintf follows that same model.
    int q = static_cast<int>(std::nearbyintf(x * inv_d));
    q = std::max(-127, std::min(127, q));
    return static_cast<int8_t>(q);
}

inline float dot_matrix_row_q8_scalar(
    const QuantMatrixK& matrix,
    const block_q8_K* xq,
    int row) {

    assert(matrix.data != nullptr);
    assert(matrix.cols > 0 && matrix.cols % QK_K == 0);
    assert(row >= 0 && row < matrix.rows);

    const int nb = matrix.cols / QK_K;

    if (matrix.type == KQuantType::Q4_K) {
        const auto* blocks = reinterpret_cast<const block_q4_K*>(matrix.data);
        return dot_row_q4_K_q8_K_scalar(
            blocks + static_cast<std::size_t>(row) * nb, xq, nb);
    }

    const auto* blocks = reinterpret_cast<const block_q6_K*>(matrix.data);
    return dot_row_q6_K_q8_K_scalar(
        blocks + static_cast<std::size_t>(row) * nb, xq, nb);
}

} // namespace

block_q8_K* Q8KWorkspace::resize_for(int n) {
    assert(n > 0);
    assert(n % QK_K == 0);
    blocks.resize(static_cast<std::size_t>(n / QK_K));
    return blocks.data();
}

// =============================================================================
// 1. FP32 -> Q8_K
// =============================================================================
void quantize_row_q8_K_scalar(const float* x, block_q8_K* y, int n) {
    assert(x != nullptr);
    assert(y != nullptr);
    assert(n > 0 && n % QK_K == 0);

    const int nb = n / QK_K;

    for (int b = 0; b < nb; ++b) {
        const float* xb = x + static_cast<std::size_t>(b) * QK_K;
        block_q8_K& dst = y[b];

        // One scale for the whole 256-value activation block.
        float amax = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            amax = std::max(amax, std::fabs(xb[i]));
        }

        if (amax == 0.0f) {
            dst.d = 0.0f;
            std::memset(dst.qs, 0, sizeof(dst.qs));
            std::memset(dst.bsums, 0, sizeof(dst.bsums));
            continue;
        }

        dst.d = amax / 127.0f;
        const float inv_d = 127.0f / amax;

        // bsums[g] = sum of each consecutive group of 16 quantized activations.
        // Q4_K uses pairs of these (32 values) for the dmin term.
        // Q6_K uses them directly for the encoded (q - 32) correction.
        for (int g = 0; g < QK_K / 16; ++g) {
            int32_t sum = 0;
            const int base = g * 16;
            for (int i = 0; i < 16; ++i) {
                const int8_t q = quantize_q8_value(xb[base + i], inv_d);
                dst.qs[base + i] = q;
                sum += static_cast<int32_t>(q);
            }
            dst.bsums[g] = static_cast<int16_t>(sum);
        }
    }
}

// =============================================================================
// 2. Q4_K x Q8_K row dot product
// =============================================================================
float dot_row_q4_K_q8_K_scalar(
    const block_q4_K* w,
    const block_q8_K* xq,
    int num_blocks) {

    assert(w != nullptr);
    assert(xq != nullptr);
    assert(num_blocks > 0);

    float result = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_K& wb = w[b];
        const block_q8_K& ab = xq[b];

        // The Q4_K block stores 8 six-bit scales and 8 six-bit minima.
        uint8_t scales[8];
        uint8_t mins[8];
        for (int g = 0; g < 8; ++g) {
            get_scale_min_k4(g, wb.scales, &scales[g], &mins[g]);
        }

        // Integer part of:
        //   sum_g scale[g] * sum_i(q4[i] * q8[i])
        //
        // q4 is kept as unsigned [0,15].  We do NOT construct FP32 weights.
        int32_t weighted_dot = 0;

        // Physical Q4_K layout: each 32-byte chunk contains 64 weights:
        //   low nibble  -> first  32-value group
        //   high nibble -> second 32-value group
        for (int chunk = 0; chunk < 4; ++chunk) {
            const uint8_t* packed = wb.qs + chunk * 32;
            const int xbase = chunk * 64;

            int32_t dot_lo = 0;
            int32_t dot_hi = 0;

            for (int i = 0; i < 32; ++i) {
                const uint8_t byte = packed[i];
                const int32_t qlo = static_cast<int32_t>(byte & 0x0F);
                const int32_t qhi = static_cast<int32_t>(byte >> 4);

                dot_lo += qlo * static_cast<int32_t>(ab.qs[xbase + i]);
                dot_hi += qhi * static_cast<int32_t>(ab.qs[xbase + 32 + i]);
            }

            weighted_dot += static_cast<int32_t>(scales[2 * chunk + 0]) * dot_lo;
            weighted_dot += static_cast<int32_t>(scales[2 * chunk + 1]) * dot_hi;
        }

        // Q4_K real weight is approximately:
        //   weight = d * scale[g] * q4 - dmin * min[g]
        //
        // Therefore the min contribution needs only sum(q8) for each group.
        // Q8_K already stores sums of 16 values, so two bsums = one Q4_K
        // 32-value group. No per-weight subtraction is required.
        int32_t min_dot = 0;
        for (int g = 0; g < 8; ++g) {
            const int32_t sum_x =
                static_cast<int32_t>(ab.bsums[2 * g + 0]) +
                static_cast<int32_t>(ab.bsums[2 * g + 1]);
            min_dot += static_cast<int32_t>(mins[g]) * sum_x;
        }

        const float xd = ab.d;
        const float wd = fp16_to_fp32(wb.d);
        const float wm = fp16_to_fp32(wb.dmin);

        // x ~= xd * q8
        // w ~= wd * scale * q4 - wm * min
        result += xd * (
            wd * static_cast<float>(weighted_dot) -
            wm * static_cast<float>(min_dot));
    }

    return result;
}

// =============================================================================
// 3. Q6_K x Q8_K row dot product
// =============================================================================
float dot_row_q6_K_q8_K_scalar(
    const block_q6_K* w,
    const block_q8_K* xq,
    int num_blocks) {

    assert(w != nullptr);
    assert(xq != nullptr);
    assert(num_blocks > 0);

    float result = 0.0f;

    for (int b = 0; b < num_blocks; ++b) {
        const block_q6_K& wb = w[b];
        const block_q8_K& ab = xq[b];

        // Q6_K stores an unsigned reconstructed q in [0,63], but the actual
        // signed quantized value is (q - 32).  Keep q unsigned in the hot loop:
        //
        //   scale * (q - 32) * x
        // = scale * q * x - 32 * scale * x
        //
        // The second term is obtained from Q8_K.bsums.  This is important for
        // SIMD ports because unsigned-byte dot instructions can consume q
        // directly without a per-element q -= 32 conversion.
        int32_t weighted_unsigned_dot = 0;

        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = wb.ql + half * 64;
            const uint8_t* qh = wb.qh + half * 32;
            const int8_t* sc  = wb.scales + half * 8;
            const int8_t* aq  = ab.qs + half * 128;

            for (int l = 0; l < 32; ++l) {
                // Four Q6 values are encoded together. This is exactly the same
                // physical mapping as the original scalar GGUF decoder, except
                // we intentionally do NOT subtract 32 here.
                const uint8_t h = qh[l];

                const int32_t q0 =
                    static_cast<int32_t>(ql[l +  0] & 0x0F) |
                    (static_cast<int32_t>((h >> 0) & 0x03) << 4);

                const int32_t q1 =
                    static_cast<int32_t>(ql[l + 32] & 0x0F) |
                    (static_cast<int32_t>((h >> 2) & 0x03) << 4);

                const int32_t q2 =
                    static_cast<int32_t>(ql[l +  0] >> 4) |
                    (static_cast<int32_t>((h >> 4) & 0x03) << 4);

                const int32_t q3 =
                    static_cast<int32_t>(ql[l + 32] >> 4) |
                    (static_cast<int32_t>((h >> 6) & 0x03) << 4);

                // Each scale covers 16 consecutive logical weights.
                const int parity = (l >= 16) ? 1 : 0;
                const int32_t s0 = static_cast<int32_t>(sc[0 + parity]);
                const int32_t s1 = static_cast<int32_t>(sc[2 + parity]);
                const int32_t s2 = static_cast<int32_t>(sc[4 + parity]);
                const int32_t s3 = static_cast<int32_t>(sc[6 + parity]);

                weighted_unsigned_dot += s0 * q0 * static_cast<int32_t>(aq[l +  0]);
                weighted_unsigned_dot += s1 * q1 * static_cast<int32_t>(aq[l + 32]);
                weighted_unsigned_dot += s2 * q2 * static_cast<int32_t>(aq[l + 64]);
                weighted_unsigned_dot += s3 * q3 * static_cast<int32_t>(aq[l + 96]);
            }
        }

        // Correct all q values from [0,63] to [-32,31] in one compact pass:
        //   sum_g 32 * scale[g] * sum(q8[g])
        int32_t offset_dot = 0;
        for (int g = 0; g < 16; ++g) {
            offset_dot +=
                static_cast<int32_t>(wb.scales[g]) *
                static_cast<int32_t>(ab.bsums[g]);
        }
        offset_dot *= 32;

        const int32_t signed_dot = weighted_unsigned_dot - offset_dot;
        const float d = fp16_to_fp32(wb.d) * ab.d;
        result += d * static_cast<float>(signed_dot);
    }

    return result;
}

// =============================================================================
// 4. Pre-quantized GEMV wrappers
// =============================================================================
void gemv_q4_K_q8_K_scalar(
    const char* matrix_weights,
    const block_q8_K* xq,
    float* out,
    int num_rows,
    int num_cols) {

    assert(matrix_weights != nullptr);
    assert(xq != nullptr);
    assert(out != nullptr);
    assert(num_cols > 0 && num_cols % QK_K == 0);

    const auto* blocks = reinterpret_cast<const block_q4_K*>(matrix_weights);
    const int nb = num_cols / QK_K;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < num_rows; ++r) {
        out[r] = dot_row_q4_K_q8_K_scalar(
            blocks + static_cast<std::size_t>(r) * nb, xq, nb);
    }
}

void gemv_q6_K_q8_K_scalar(
    const char* matrix_weights,
    const block_q8_K* xq,
    float* out,
    int num_rows,
    int num_cols) {

    assert(matrix_weights != nullptr);
    assert(xq != nullptr);
    assert(out != nullptr);
    assert(num_cols > 0 && num_cols % QK_K == 0);

    const auto* blocks = reinterpret_cast<const block_q6_K*>(matrix_weights);
    const int nb = num_cols / QK_K;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < num_rows; ++r) {
        out[r] = dot_row_q6_K_q8_K_scalar(
            blocks + static_cast<std::size_t>(r) * nb, xq, nb);
    }
}

void gemv_k_q8_K_scalar(
    const QuantMatrixK& matrix,
    const block_q8_K* xq,
    float* out) {

    assert(matrix.data != nullptr);
    assert(xq != nullptr);
    assert(out != nullptr);
    assert(matrix.cols > 0 && matrix.cols % QK_K == 0);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < matrix.rows; ++r) {
        out[r] = dot_matrix_row_q8_scalar(matrix, xq, r);
    }
}

void gemv_q4_K_scalar_q8_path(
    const char* matrix_weights,
    const float* x,
    float* out,
    int num_rows,
    int num_cols,
    Q8KWorkspace& workspace) {

    block_q8_K* xq = workspace.resize_for(num_cols);
    quantize_row_q8_K_scalar(x, xq, num_cols);
    gemv_q4_K_q8_K_scalar(matrix_weights, xq, out, num_rows, num_cols);
}

void gemv_q6_K_scalar_q8_path(
    const char* matrix_weights,
    const float* x,
    float* out,
    int num_rows,
    int num_cols,
    Q8KWorkspace& workspace) {

    block_q8_K* xq = workspace.resize_for(num_cols);
    quantize_row_q8_K_scalar(x, xq, num_cols);
    gemv_q6_K_q8_K_scalar(matrix_weights, xq, out, num_rows, num_cols);
}

// =============================================================================
// 5. Structural fusion: reuse one Q8_K activation
// =============================================================================
void project_qkv_scalar(
    const float* x,
    int x_size,
    const QuantMatrixK& wq,
    const QuantMatrixK& wk,
    const QuantMatrixK& wv,
    float* q,
    float* k,
    float* v,
    Q8KWorkspace& workspace) {

    assert(x != nullptr && q != nullptr && k != nullptr && v != nullptr);
    assert(x_size > 0 && x_size % QK_K == 0);
    assert(wq.cols == x_size);
    assert(wk.cols == x_size);
    assert(wv.cols == x_size);

    // Optimization #1: quantize x exactly once for all three projections.
    block_q8_K* xq = workspace.resize_for(x_size);
    quantize_row_q8_K_scalar(x, xq, x_size);

#if defined(_OPENMP)
    // Optimization #2: one parallel team for Q, K and V rather than three
    // independent parallel-region launches. The barriers are still explicit
    // scheduling points; architecture backends may replace this with a
    // persistent thread pool.
    #pragma omp parallel
    {
        #pragma omp for nowait schedule(static)
        for (int r = 0; r < wq.rows; ++r) {
            q[r] = dot_matrix_row_q8_scalar(wq, xq, r);
        }

        #pragma omp for nowait schedule(static)
        for (int r = 0; r < wk.rows; ++r) {
            k[r] = dot_matrix_row_q8_scalar(wk, xq, r);
        }

        #pragma omp for schedule(static)
        for (int r = 0; r < wv.rows; ++r) {
            v[r] = dot_matrix_row_q8_scalar(wv, xq, r);
        }
    }
#else
    for (int r = 0; r < wq.rows; ++r) q[r] = dot_matrix_row_q8_scalar(wq, xq, r);
    for (int r = 0; r < wk.rows; ++r) k[r] = dot_matrix_row_q8_scalar(wk, xq, r);
    for (int r = 0; r < wv.rows; ++r) v[r] = dot_matrix_row_q8_scalar(wv, xq, r);
#endif
}

void project_gate_up_scalar(
    const float* x,
    int x_size,
    const QuantMatrixK& w_gate,
    const QuantMatrixK& w_up,
    float* gate,
    float* up,
    Q8KWorkspace& workspace) {

    assert(x != nullptr && gate != nullptr && up != nullptr);
    assert(x_size > 0 && x_size % QK_K == 0);
    assert(w_gate.cols == x_size);
    assert(w_up.cols == x_size);
    assert(w_gate.rows == w_up.rows);

    // Optimization #1: one Q8_K conversion shared by both MLP projections.
    block_q8_K* xq = workspace.resize_for(x_size);
    quantize_row_q8_K_scalar(x, xq, x_size);

    // Optimization #2: each worker computes gate[r] and up[r] while it owns r.
    // This avoids a second scheduling pass / parallel-region launch.
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < w_gate.rows; ++r) {
        gate[r] = dot_matrix_row_q8_scalar(w_gate, xq, r);
        up[r]   = dot_matrix_row_q8_scalar(w_up,   xq, r);
    }
}

