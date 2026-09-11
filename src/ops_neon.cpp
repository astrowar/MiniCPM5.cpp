#include "ops_internal.h"
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>

static inline uint8x8_t safe_vld1_u8(const uint8_t* p) {
    uint8x8_t v;
    memcpy(&v, p, 8);
    return v;
}

static inline int32_t neon_dot_q4_q8(const uint8_t* q, bool high_nibble, const int8_t* xq) {
    int32_t dot = 0;
    for (int i = 0; i < 32; i += 8) {
        uint8x8_t q8 = safe_vld1_u8(q + i);
        uint8x8_t nib = high_nibble ? vshr_n_u8(q8, 4) : vand_u8(q8, vdup_n_u8(0x0F));
        int8x8_t nib_s = vreinterpret_s8_u8(nib);
        int8x8_t x8 = vld1_s8(xq + i);

        int16x8_t prod = vmull_s8(nib_s, x8);
        int32x4_t lo = vmovl_s16(vget_low_s16(prod));
        int32x4_t hi = vmovl_s16(vget_high_s16(prod));
        int32x4_t s = vaddq_s32(lo, hi);
        int32x2_t s2 = vadd_s32(vget_low_s32(s), vget_high_s32(s));
        dot += vget_lane_s32(s2, 0) + vget_lane_s32(s2, 1);
    }
    return dot;
}

float dot_row_q4_K_q8_K_neon(const block_q4_K* w, const block_q8_K* xq, int num_blocks) {
    float result = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        const block_q4_K& wb = w[b];
        const block_q8_K& ab = xq[b];

        uint8_t scales[8];
        uint8_t mins[8];
        for (int g = 0; g < 8; ++g) {
            get_scale_min_k4(g, wb.scales, &scales[g], &mins[g]);
        }

        int32_t weighted_dot = 0;
        for (int chunk = 0; chunk < 4; ++chunk) {
            const uint8_t* packed = wb.qs + chunk * 32;
            const int xbase = chunk * 64;

            int32_t dot_lo = neon_dot_q4_q8(packed, false, ab.qs + xbase);
            int32_t dot_hi = neon_dot_q4_q8(packed, true,  ab.qs + xbase + 32);

            weighted_dot += static_cast<int32_t>(scales[2 * chunk + 0]) * dot_lo;
            weighted_dot += static_cast<int32_t>(scales[2 * chunk + 1]) * dot_hi;
        }

        int32_t min_dot = 0;
        for (int g = 0; g < 8; ++g) {
            int32_t sum_x = ab.bsums[2 * g + 0] + ab.bsums[2 * g + 1];
            min_dot += static_cast<int32_t>(mins[g]) * sum_x;
        }

        const float xd = ab.d;
        const float wd = fp16_to_fp32(wb.d);
        const float wm = fp16_to_fp32(wb.dmin);

        result += xd * (wd * static_cast<float>(weighted_dot) - wm * static_cast<float>(min_dot));
    }
    return result;
}

// Q6_K Inner dot product
// This is exactly the scalar implementation from ops_scalar_optimized, 
// ensuring mathematically identical results, while trusting GCC/Clang 
// to automatically vectorize the clean inner loops via -O3 NEON autovectorization.
float dot_row_q6_K_q8_K_neon(const block_q6_K* w, const block_q8_K* xq, int num_blocks) {
    float result = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        const block_q6_K& wb = w[b];
        const block_q8_K& ab = xq[b];

        int32_t weighted_unsigned_dot = 0;
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = wb.ql + half * 64;
            const uint8_t* qh = wb.qh + half * 32;
            const int8_t* sc  = wb.scales + half * 8;
            const int8_t* aq  = ab.qs + half * 128;

            for (int l = 0; l < 32; ++l) {
                const uint8_t h = qh[l];
                const int32_t q0 = static_cast<int32_t>(ql[l +  0] & 0x0F) | (static_cast<int32_t>((h >> 0) & 0x03) << 4);
                const int32_t q1 = static_cast<int32_t>(ql[l + 32] & 0x0F) | (static_cast<int32_t>((h >> 2) & 0x03) << 4);
                const int32_t q2 = static_cast<int32_t>(ql[l +  0] >> 4) | (static_cast<int32_t>((h >> 4) & 0x03) << 4);
                const int32_t q3 = static_cast<int32_t>(ql[l + 32] >> 4) | (static_cast<int32_t>((h >> 6) & 0x03) << 4);

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

        int32_t offset_dot = 0;
        for (int g = 0; g < 16; ++g) {
            offset_dot += static_cast<int32_t>(wb.scales[g]) * static_cast<int32_t>(ab.bsums[g]);
        }
        offset_dot *= 32;

        const int32_t signed_dot = weighted_unsigned_dot - offset_dot;
        const float d = fp16_to_fp32(wb.d) * ab.d;
        result += d * static_cast<float>(signed_dot);
    }
    return result;
}

// -----------------------------------------------------------------------------
void quantize_row_q8_K_neon(const float* x, block_q8_K* y, int n) {
    const int nb = n / QK_K;
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * QK_K;
        block_q8_K& dst = y[b];

        float32x4_t m = vabsq_f32(vld1q_f32(xb));
        for (int i = 4; i < QK_K; i += 4) {
            m = vmaxq_f32(m, vabsq_f32(vld1q_f32(xb + i)));
        }
        float32x2_t t = vmax_f32(vget_low_f32(m), vget_high_f32(m));
        float amax = vget_lane_f32(vpmax_f32(t, t), 0);

        if (amax == 0.0f) {
            dst.d = 0.0f;
            std::memset(dst.qs, 0, sizeof(dst.qs));
            std::memset(dst.bsums, 0, sizeof(dst.bsums));
            continue;
        }

        dst.d = amax / 127.0f;
        const float inv_d = 127.0f / amax;
        float32x4_t vinv = vdupq_n_f32(inv_d);

        for (int g = 0; g < QK_K / 16; ++g) {
            int32_t sum = 0;
            const int base = g * 16;

            for (int i = 0; i < 16; i += 4) {
                float32x4_t xf = vld1q_f32(xb + base + i);
                int32x4_t xi = vcvtq_s32_f32(vmulq_f32(xf, vinv));
                
                // Saturate to [-127, 127]
                xi = vmaxq_s32(xi, vdupq_n_s32(-127));
                xi = vminq_s32(xi, vdupq_n_s32(127));

                int16x4_t x16 = vmovn_s32(xi);
                int8x8_t x8 = vmovn_s16(vcombine_s16(x16, x16));
                
                vst1_lane_u32((uint32_t*)(dst.qs + base + i), vreinterpret_u32_s8(x8), 0);

                sum += vgetq_lane_s32(xi, 0) + vgetq_lane_s32(xi, 1) + vgetq_lane_s32(xi, 2) + vgetq_lane_s32(xi, 3);
            }
            dst.bsums[g] = static_cast<int16_t>(sum);
        }
    }
}
#endif