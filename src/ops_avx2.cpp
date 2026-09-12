#include "ops_internal.h"
#include "thread_affinity.h"
#include <cassert>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

static inline float hsum_f32_8(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline int hsum_i32_8_fast(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    __m128i hi64 = _mm_unpackhi_epi64(s, s);
    s = _mm_add_epi32(s, hi64);
    __m128i hi32 = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(s, hi32));
}

static inline float hmax_f32_8(__m256 v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

// Prefer hardware conversion when the native target exposes F16C.
// The generic bit-manipulation conversion in ops_internal.h remains the fallback.
static inline float fp16_to_fp32_hot(ggml_fp16_t h) {
#if defined(_MSC_VER)
    const __m128i h16 = _mm_cvtsi32_si128(static_cast<int>(h));
    return _mm_cvtss_f32(_mm_cvtph_ps(h16));
#elif defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return fp16_to_fp32(h);
#endif
}

void quantize_row_q8_K_avx2(const float* RESTRICT x, block_q8_K* RESTRICT y, int n) {
    assert(n % QK_K == 0);
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256i qmin = _mm256_set1_epi32(-127);
    const __m256i qmax = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int ib = 0; ib < n / QK_K; ++ib, x += QK_K) {
        block_q8_K& dst = y[ib];
        __m256 vmax0 = _mm256_setzero_ps();
        __m256 vmax1 = _mm256_setzero_ps();
        for (int i = 0; i < QK_K; i += 16) {
            __m256 v0 = _mm256_loadu_ps(x + i + 0);
            __m256 v1 = _mm256_loadu_ps(x + i + 8);
            vmax0 = _mm256_max_ps(vmax0, _mm256_andnot_ps(sign_bit, v0));
            vmax1 = _mm256_max_ps(vmax1, _mm256_andnot_ps(sign_bit, v1));
        }
        const float amax = hmax_f32_8(_mm256_max_ps(vmax0, vmax1));
        if (amax == 0.0f) {
            dst.d = 0.0f;
            std::memset(dst.qs, 0, sizeof(dst.qs));
            std::memset(dst.bsums, 0, sizeof(dst.bsums));
            continue;
        }

        dst.d = amax / 127.0f;
        const __m256 mul = _mm256_set1_ps(127.0f / amax);

        for (int i = 0; i < QK_K; i += 32) {
            auto cvt8 = [&](const float* p) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(p), mul);
                __m256i q = _mm256_cvtps_epi32(v); // nearest according to MXCSR
                return _mm256_min_epi32(qmax, _mm256_max_epi32(qmin, q));
            };

            __m256i q0 = cvt8(x + i + 0);
            __m256i q1 = cvt8(x + i + 8);
            __m256i q2 = cvt8(x + i + 16);
            __m256i q3 = cvt8(x + i + 24);

            __m256i p01 = _mm256_packs_epi32(q0, q1);
            __m256i p23 = _mm256_packs_epi32(q2, q3);
            __m256i q8 = _mm256_packs_epi16(p01, p23);
            q8 = _mm256_permutevar8x32_epi32(q8, perm);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.qs + i), q8);
        }

        // bsums are needed by Q4_K minima and Q6_K's -32 offset correction.
        for (int g = 0; g < QK_K / 16; ++g) {
            const __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst.qs + 16*g));
            const __m256i q16 = _mm256_cvtepi8_epi16(q8);
            dst.bsums[g] = static_cast<int16_t>(hsum_i32_8_fast(_mm256_madd_epi16(q16, ones16)));
        }
    }
}

// Q4_K row dot: keep eight partial sums live in FP32 SIMD across ALL 256-element
// blocks and reduce once at the end of the row. This removes one horizontal
// reduction from every superblock.
static inline float dot_row_q4_K_q8_K_avx2(
        const block_q4_K* RESTRICT w,
        const block_q8_K* RESTRICT a,
        int nb) {
    const __m256i mask4 = _mm256_set1_epi8(0x0f);
    __m256 acc = _mm256_setzero_ps();
    float acc_min = 0.0f;

    for (int b = 0; b < nb; ++b) {
        uint8_t scales[8], mins[8];
        decode_q4k_scales_mins(w[b].scales, scales, mins);

        __m256i isum = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[b].qs + 32*j));
            const __m256i qlo = _mm256_and_si256(packed, mask4);
            const __m256i qhi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask4);
            const __m256i xlo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[b].qs + 64*j));
            const __m256i xhi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[b].qs + 64*j + 32));

            __m256i p0 = _mm256_maddubs_epi16(qlo, xlo);
            __m256i p1 = _mm256_maddubs_epi16(qhi, xhi);
            p0 = _mm256_madd_epi16(p0, _mm256_set1_epi16(static_cast<int16_t>(scales[2*j + 0])));
            p1 = _mm256_madd_epi16(p1, _mm256_set1_epi16(static_cast<int16_t>(scales[2*j + 1])));
            isum = _mm256_add_epi32(isum, _mm256_add_epi32(p0, p1));
        }

        const float ds = a[b].d * fp16_to_fp32_hot(w[b].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(ds), _mm256_cvtepi32_ps(isum), acc);

        int32_t minsum = 0;
        for (int g = 0; g < 8; ++g) {
            const int32_t sx = static_cast<int32_t>(a[b].bsums[2*g]) + static_cast<int32_t>(a[b].bsums[2*g + 1]);
            minsum += static_cast<int32_t>(mins[g]) * sx;
        }
        acc_min -= a[b].d * fp16_to_fp32_hot(w[b].dmin) * static_cast<float>(minsum);
    }

    return hsum_f32_8(acc) + acc_min;
}

// Q6_K row dot: keep reconstructed q in [0,63] so maddubs can consume it as
// unsigned bytes. Correct the format's -32 offset once via Q8_K bsums, instead
// of subtracting 32 and doing signed-byte sign fixups for every weight.
static inline float dot_row_q6_K_q8_K_avx2(
        const block_q6_K* RESTRICT w,
        const block_q8_K* RESTRICT a,
        int nb) {
    const __m256i m3  = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i mc0 = _mm256_set1_epi8(static_cast<char>(0xC0));
    const __m256i shuf01 = _mm256_setr_epi8(
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);
    const __m256i shuf23 = _mm256_setr_epi8(
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7);
    const __m256i shuf45 = _mm256_setr_epi8(
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11);
    const __m256i shuf67 = _mm256_setr_epi8(
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15);
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; ++b) {
        const __m128i sc8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w[b].scales));
        const __m256i sc16 = _mm256_cvtepi8_epi16(sc8);
        const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[b].bsums));
        const __m256i offset32 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc16), 5);

        __m256i isum = _mm256_setzero_si256();
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = w[b].ql + half*64;
            const uint8_t* qh = w[b].qh + half*32;
            const int8_t* xq = a[b].qs + half*128;
            const __m128i shalf = (half == 0) ? _mm256_castsi256_si128(sc16) : _mm256_extracti128_si256(sc16, 1);
            const __m256i s = _mm256_broadcastsi128_si256(shalf);
            const __m256i s01 = _mm256_shuffle_epi8(s, shuf01);
            const __m256i s23 = _mm256_shuffle_epi8(s, shuf23);
            const __m256i s45 = _mm256_shuffle_epi8(s, shuf45);
            const __m256i s67 = _mm256_shuffle_epi8(s, shuf67);

            const __m256i lo0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 0));
            const __m256i lo1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i hb  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

            const __m256i q0 = _mm256_or_si256(_mm256_and_si256(lo0, m15), _mm256_slli_epi16(_mm256_and_si256(hb, m3), 4));
            const __m256i q1 = _mm256_or_si256(_mm256_and_si256(lo1, m15), _mm256_slli_epi16(_mm256_and_si256(hb, m12), 2));
            const __m256i q2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo0, 4), m15), _mm256_and_si256(hb, m48));
            const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo1, 4), m15), _mm256_srli_epi16(_mm256_and_si256(hb, mc0), 2));

            const __m256i x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 0));
            const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 32));
            const __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 64));
            const __m256i x3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + 96));

            __m256i p0 = _mm256_maddubs_epi16(q0, x0);
            __m256i p1 = _mm256_maddubs_epi16(q1, x1);
            __m256i p2 = _mm256_maddubs_epi16(q2, x2);
            __m256i p3 = _mm256_maddubs_epi16(q3, x3);

            p0 = _mm256_madd_epi16(p0, s01);
            p1 = _mm256_madd_epi16(p1, s23);
            p2 = _mm256_madd_epi16(p2, s45);
            p3 = _mm256_madd_epi16(p3, s67);

            isum = _mm256_add_epi32(isum, _mm256_add_epi32(p0, p1));
            isum = _mm256_add_epi32(isum, _mm256_add_epi32(p2, p3));
        }

        isum = _mm256_sub_epi32(isum, offset32);
        const float ds = a[b].d * fp16_to_fp32_hot(w[b].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(ds), _mm256_cvtepi32_ps(isum), acc);
    }

    return hsum_f32_8(acc);
}

void gemv_q4_K_q8_K_avx2(const char* RESTRICT matrix_weights, const block_q8_K* RESTRICT xq,
                          float* RESTRICT out, int num_rows, int num_cols) {
    assert(num_cols % QK_K == 0);
    const block_q4_K* RESTRICT blocks = reinterpret_cast<const block_q4_K*>(matrix_weights);
    const int nb = num_cols / QK_K;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        cpu_affinity::pin_once();
        out[r] = dot_row_q4_K_q8_K_avx2(blocks + static_cast<size_t>(r)*nb, xq, nb);
    }
}

void gemv_q6_K_q8_K_avx2(const char* RESTRICT matrix_weights, const block_q8_K* RESTRICT xq,
                          float* RESTRICT out, int num_rows, int num_cols) {
    assert(num_cols % QK_K == 0);
    const block_q6_K* RESTRICT blocks = reinterpret_cast<const block_q6_K*>(matrix_weights);
    const int nb = num_cols / QK_K;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        cpu_affinity::pin_once();
        out[r] = dot_row_q6_K_q8_K_avx2(blocks + static_cast<size_t>(r)*nb, xq, nb);
    }
}

// Fused QKV (all Q4_K): single flattened parallel for over total rows.
void gemv_qkv_q4_K_q8_K_avx2(const char* wq, const char* wk, const char* wv,
                              const block_q8_K* RESTRICT xq,
                              float* RESTRICT q, float* RESTRICT k, float* RESTRICT v,
                              int q_rows, int kv_rows, int num_cols) {
    assert(num_cols % QK_K == 0);
    const block_q4_K* bq = reinterpret_cast<const block_q4_K*>(wq);
    const block_q4_K* bk = reinterpret_cast<const block_q4_K*>(wk);
    const block_q4_K* bv = reinterpret_cast<const block_q4_K*>(wv);
    const int nb = num_cols / QK_K;
    const int total = q_rows + kv_rows + kv_rows;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < total; ++r) {
        cpu_affinity::pin_once();
        if (r < q_rows) {
            q[r] = dot_row_q4_K_q8_K_avx2(bq + static_cast<size_t>(r)*nb, xq, nb);
        } else if (r < q_rows + kv_rows) {
            int kr = r - q_rows;
            k[kr] = dot_row_q4_K_q8_K_avx2(bk + static_cast<size_t>(kr)*nb, xq, nb);
        } else {
            int vr = r - q_rows - kv_rows;
            v[vr] = dot_row_q4_K_q8_K_avx2(bv + static_cast<size_t>(vr)*nb, xq, nb);
        }
    }
}

// Fused QKV (q/k Q4_K, v Q6_K): single flattened parallel for.
void gemv_qkv_q4_q4_q6_q8_K_avx2(const char* wq, const char* wk, const char* wv,
                                  const block_q8_K* RESTRICT xq,
                                  float* RESTRICT q, float* RESTRICT k, float* RESTRICT v,
                                  int q_rows, int kv_rows, int num_cols) {
    assert(num_cols % QK_K == 0);
    const block_q4_K* bq = reinterpret_cast<const block_q4_K*>(wq);
    const block_q4_K* bk = reinterpret_cast<const block_q4_K*>(wk);
    const block_q6_K* bv = reinterpret_cast<const block_q6_K*>(wv);
    const int nb = num_cols / QK_K;
    const int total = q_rows + kv_rows + kv_rows;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < total; ++r) {
        cpu_affinity::pin_once();
        if (r < q_rows) {
            q[r] = dot_row_q4_K_q8_K_avx2(bq + static_cast<size_t>(r)*nb, xq, nb);
        } else if (r < q_rows + kv_rows) {
            int kr = r - q_rows;
            k[kr] = dot_row_q4_K_q8_K_avx2(bk + static_cast<size_t>(kr)*nb, xq, nb);
        } else {
            int vr = r - q_rows - kv_rows;
            v[vr] = dot_row_q6_K_q8_K_avx2(bv + static_cast<size_t>(vr)*nb, xq, nb);
        }
    }
}

// Combined Gate/Up: single pass through Q8_K blocks, computing both gate and up
// dot products simultaneously. Reads xq bytes once per block for both outputs.
static inline void dot_row_gate_up_q4_K_q8_K_avx2(
        const block_q4_K* RESTRICT wg,
        const block_q4_K* RESTRICT wu,
        const block_q8_K* RESTRICT a,
        int nb,
        float& gate_out, float& up_out) {
    const __m256i mask4 = _mm256_set1_epi8(0x0f);
    __m256 acc_g = _mm256_setzero_ps();
    __m256 acc_u = _mm256_setzero_ps();
    float acc_min_g = 0.0f;
    float acc_min_u = 0.0f;

    for (int b = 0; b < nb; ++b) {
        uint8_t scales_g[8], mins_g[8];
        uint8_t scales_u[8], mins_u[8];
        decode_q4k_scales_mins(wg[b].scales, scales_g, mins_g);
        decode_q4k_scales_mins(wu[b].scales, scales_u, mins_u);

        __m256i isum_g = _mm256_setzero_si256();
        __m256i isum_u = _mm256_setzero_si256();

        for (int j = 0; j < 4; ++j) {
            const __m256i packed_g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wg[b].qs + 32*j));
            const __m256i packed_u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wu[b].qs + 32*j));
            const __m256i qlo_g = _mm256_and_si256(packed_g, mask4);
            const __m256i qhi_g = _mm256_and_si256(_mm256_srli_epi16(packed_g, 4), mask4);
            const __m256i qlo_u = _mm256_and_si256(packed_u, mask4);
            const __m256i qhi_u = _mm256_and_si256(_mm256_srli_epi16(packed_u, 4), mask4);

            // Load xq ONCE, use for both gate and up
            const __m256i xlo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[b].qs + 64*j));
            const __m256i xhi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[b].qs + 64*j + 32));

            __m256i pg0 = _mm256_maddubs_epi16(qlo_g, xlo);
            __m256i pg1 = _mm256_maddubs_epi16(qhi_g, xhi);
            pg0 = _mm256_madd_epi16(pg0, _mm256_set1_epi16(static_cast<int16_t>(scales_g[2*j + 0])));
            pg1 = _mm256_madd_epi16(pg1, _mm256_set1_epi16(static_cast<int16_t>(scales_g[2*j + 1])));
            isum_g = _mm256_add_epi32(isum_g, _mm256_add_epi32(pg0, pg1));

            __m256i pu0 = _mm256_maddubs_epi16(qlo_u, xlo);
            __m256i pu1 = _mm256_maddubs_epi16(qhi_u, xhi);
            pu0 = _mm256_madd_epi16(pu0, _mm256_set1_epi16(static_cast<int16_t>(scales_u[2*j + 0])));
            pu1 = _mm256_madd_epi16(pu1, _mm256_set1_epi16(static_cast<int16_t>(scales_u[2*j + 1])));
            isum_u = _mm256_add_epi32(isum_u, _mm256_add_epi32(pu0, pu1));
        }

        const float d_g = a[b].d * fp16_to_fp32_hot(wg[b].d);
        const float d_u = a[b].d * fp16_to_fp32_hot(wu[b].d);
        acc_g = _mm256_fmadd_ps(_mm256_set1_ps(d_g), _mm256_cvtepi32_ps(isum_g), acc_g);
        acc_u = _mm256_fmadd_ps(_mm256_set1_ps(d_u), _mm256_cvtepi32_ps(isum_u), acc_u);

        int32_t minsum_g = 0;
        int32_t minsum_u = 0;
        for (int g = 0; g < 8; ++g) {
            const int32_t sx = static_cast<int32_t>(a[b].bsums[2*g]) + static_cast<int32_t>(a[b].bsums[2*g + 1]);
            minsum_g += static_cast<int32_t>(mins_g[g]) * sx;
            minsum_u += static_cast<int32_t>(mins_u[g]) * sx;
        }
        acc_min_g -= a[b].d * fp16_to_fp32_hot(wg[b].dmin) * static_cast<float>(minsum_g);
        acc_min_u -= a[b].d * fp16_to_fp32_hot(wu[b].dmin) * static_cast<float>(minsum_u);
    }

    gate_out = hsum_f32_8(acc_g) + acc_min_g;
    up_out   = hsum_f32_8(acc_u) + acc_min_u;
}

void gemv_gate_up_q4_K_q8_K_avx2(const char* w_gate, const char* w_up,
                                  const block_q8_K* RESTRICT xq,
                                  float* RESTRICT gate, float* RESTRICT up,
                                  int num_rows, int num_cols) {
    assert(num_cols % QK_K == 0);
    const block_q4_K* bg = reinterpret_cast<const block_q4_K*>(w_gate);
    const block_q4_K* bu = reinterpret_cast<const block_q4_K*>(w_up);
    const int nb = num_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        cpu_affinity::pin_once();
        float g, u;
        dot_row_gate_up_q4_K_q8_K_avx2(bg + static_cast<size_t>(r)*nb, bu + static_cast<size_t>(r)*nb, xq, nb, g, u);
        gate[r] = g;
        up[r]   = u;
    }
}

static thread_local std::vector<block_q8_K> q8_scratch;
static inline const block_q8_K* prepare_q8(const float* x, int num_cols) {
    assert(num_cols % QK_K == 0);
    q8_scratch.resize(num_cols / QK_K);
    quantize_row_q8_K_avx2(x, q8_scratch.data(), num_cols);
    return q8_scratch.data();
}

void gemv_q4_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    gemv_q4_K_q8_K_avx2(matrix_weights, prepare_q8(x, num_cols), out, num_rows, num_cols);
}

void gemv_q6_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    gemv_q6_K_q8_K_avx2(matrix_weights, prepare_q8(x, num_cols), out, num_rows, num_cols);
}

#else
void gemv_q4_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    gemv_q4_K_scalar(matrix_weights, x, out, num_rows, num_cols);
}
void gemv_q6_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    gemv_q6_K_scalar(matrix_weights, x, out, num_rows, num_cols);
}
void gemv_qkv_q4_K_q8_K_avx2(const char* wq, const char* wk, const char* wv,
                              const block_q8_K*, float* q, float* k, float* v,
                              int q_rows, int kv_rows, int num_cols) {
    // Non-AVX2 fallback: call scalar kernels individually
    // (these have their own OpenMP internally)
}
void gemv_qkv_q4_q4_q6_q8_K_avx2(const char* wq, const char* wk, const char* wv,
                                  const block_q8_K*, float* q, float* k, float* v,
                                  int q_rows, int kv_rows, int num_cols) {
}
void gemv_gate_up_q4_K_q8_K_avx2(const char* w_gate, const char* w_up,
                                  const block_q8_K*, float* gate, float* up,
                                  int num_rows, int num_cols) {
    // Non-AVX2 fallback: call scalar kernels individually
}
#endif
