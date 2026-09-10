#include "ops_internal.h"
#include <omp.h>

#if defined(__AVX2__)
#include <immintrin.h>

// GEMV para Tensores Q4_K (Int4) otimizado com intrínsecos AVX2
void gemv_q4_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    #pragma omp parallel for
    for (int r = 0; r < num_rows; ++r) {
        int row_block_offset = r * super_blocks_per_row;
        
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256i mask_0f = _mm256_set1_epi32(0xF);

        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q4_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            float dmin = fp16_to_fp32(bloco.dmin);
            int x_offset = sb * QK_K;
            
            int is = 0;
            uint8_t sc, m;
            const uint8_t* q = bloco.qs;

            // Processa em chunks de 64 (32 bytes empacotados = 64 pesos)
            for (int j = 0; j < QK_K; j += 64) {
                get_scale_min_k4(is + 0, bloco.scales, &sc, &m);
                float d1 = d * sc; float m1 = dmin * m;
                
                get_scale_min_k4(is + 1, bloco.scales, &sc, &m);
                float d2 = d * sc; float m2 = dmin * m;

                __m256 vd1 = _mm256_set1_ps(d1);
                __m256 vm1 = _mm256_set1_ps(m1);
                __m256 vd2 = _mm256_set1_ps(d2);
                __m256 vm2 = _mm256_set1_ps(m2);

                // Processa 8 bytes de 'q' (16 pesos) por iteração AVX2
                for (int l = 0; l < 32; l += 8) {
                    // Carrega 8 bytes (uint8) para registrador XMM
                    __m128i q_8bytes = _mm_loadl_epi64((const __m128i*)(q + l));
                    
                    // Expande 8x uint8 para 8x uint32 no registrador YMM (256-bit)
                    __m256i q_epi32 = _mm256_cvtepu8_epi32(q_8bytes);
                    
                    // Isola e dequantiza os 4 bits inferiores (w1)
                    __m256i q1_epi32 = _mm256_and_si256(q_epi32, mask_0f);
                    __m256 f1 = _mm256_cvtepi32_ps(q1_epi32);
                    __m256 w1 = _mm256_sub_ps(_mm256_mul_ps(vd1, f1), vm1);
                    
                    // Isola e dequantiza os 4 bits superiores (w2)
                    __m256i q2_epi32 = _mm256_srli_epi32(q_epi32, 4);
                    __m256 f2 = _mm256_cvtepi32_ps(q2_epi32);
                    __m256 w2 = _mm256_sub_ps(_mm256_mul_ps(vd2, f2), vm2);

                    // Carrega vetor de entrada 'x' (8 floats cada)
                    __m256 vx1 = _mm256_loadu_ps(&x[x_offset + j + l]);
                    __m256 vx2 = _mm256_loadu_ps(&x[x_offset + j + l + 32]);

                    // Acumula (w * x) em blocos paralelos (FMA será inferido com -ffast-math)
                    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(w1, vx1));
                    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(w2, vx2));
                }
                q += 32;
                is += 2;
            }
        }
        
        // Redução horizontal rápida dos dois acumuladores YMM de 256-bits para 1 float
        __m256 acc_total = _mm256_add_ps(acc1, acc2);
        __m128 sum_hi = _mm256_extractf128_ps(acc_total, 1);
        __m128 sum_lo = _mm256_castps256_ps128(acc_total);
        __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[r] = _mm_cvtss_f32(sum128);
    }
}

// GEMV para Tensores Q6_K (Int6) otimizado com intrínsecos AVX2
void gemv_q6_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q6_K* blocks = reinterpret_cast<const block_q6_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    #pragma omp parallel for
    for (int r = 0; r < num_rows; ++r) {
        int row_block_offset = r * super_blocks_per_row;
        
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();

        __m256i mask_0f = _mm256_set1_epi32(0xF);
        __m256i mask_03 = _mm256_set1_epi32(3);
        __m256i offset_32 = _mm256_set1_epi32(32);

        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q6_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            int x_offset = sb * QK_K;

            const uint8_t* ql = bloco.ql;
            const uint8_t* qh = bloco.qh;
            const int8_t* sc = bloco.scales;

            for (int n = 0; n < QK_K; n += 128) {
                // Primeira metade do loop (l = 0..15)
                __m256 vs0 = _mm256_set1_ps(d * sc[0]);
                __m256 vs2 = _mm256_set1_ps(d * sc[2]);
                __m256 vs4 = _mm256_set1_ps(d * sc[4]);
                __m256 vs6 = _mm256_set1_ps(d * sc[6]);

                for (int l = 0; l < 16; l += 8) {
                    __m128i m_ql0 = _mm_loadl_epi64((const __m128i*)(ql + l));
                    __m128i m_ql32 = _mm_loadl_epi64((const __m128i*)(ql + l + 32));
                    __m128i m_qh = _mm_loadl_epi64((const __m128i*)(qh + l));

                    __m256i y_ql0 = _mm256_cvtepu8_epi32(m_ql0);
                    __m256i y_ql32 = _mm256_cvtepu8_epi32(m_ql32);
                    __m256i y_qh = _mm256_cvtepu8_epi32(m_qh);

                    // q1
                    __m256i q1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(y_ql0, mask_0f), _mm256_slli_epi32(_mm256_and_si256(y_qh, mask_03), 4)), offset_32);
                    __m256 w1 = _mm256_cvtepi32_ps(q1);
                    
                    // q2
                    __m256i q2 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(y_ql32, mask_0f), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(y_qh, 2), mask_03), 4)), offset_32);
                    __m256 w2 = _mm256_cvtepi32_ps(q2);

                    // q3
                    __m256i q3 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(y_ql0, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(y_qh, 4), mask_03), 4)), offset_32);
                    __m256 w3 = _mm256_cvtepi32_ps(q3);

                    // q4
                    __m256i q4 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(y_ql32, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(y_qh, 6), mask_03), 4)), offset_32);
                    __m256 w4 = _mm256_cvtepi32_ps(q4);

                    __m256 vx1 = _mm256_loadu_ps(&x[x_offset + n + l + 0]);
                    __m256 vx2 = _mm256_loadu_ps(&x[x_offset + n + l + 32]);
                    __m256 vx3 = _mm256_loadu_ps(&x[x_offset + n + l + 64]);
                    __m256 vx4 = _mm256_loadu_ps(&x[x_offset + n + l + 96]);

                    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_mul_ps(w1, vs0), vx1));
                    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_mul_ps(w2, vs2), vx2));
                    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_mul_ps(w3, vs4), vx3));
                    acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(_mm256_mul_ps(w4, vs6), vx4));
                }

                // Segunda metade do loop (l = 16..31)
                __m256 vs1 = _mm256_set1_ps(d * sc[1]);
                __m256 vs3 = _mm256_set1_ps(d * sc[3]);
                __m256 vs5 = _mm256_set1_ps(d * sc[5]);
                __m256 vs7 = _mm256_set1_ps(d * sc[7]);

                for (int l = 16; l < 32; l += 8) {
                    __m128i m_ql0 = _mm_loadl_epi64((const __m128i*)(ql + l));
                    __m128i m_ql32 = _mm_loadl_epi64((const __m128i*)(ql + l + 32));
                    __m128i m_qh = _mm_loadl_epi64((const __m128i*)(qh + l));

                    __m256i y_ql0 = _mm256_cvtepu8_epi32(m_ql0);
                    __m256i y_ql32 = _mm256_cvtepu8_epi32(m_ql32);
                    __m256i y_qh = _mm256_cvtepu8_epi32(m_qh);

                    // q1
                    __m256i q1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(y_ql0, mask_0f), _mm256_slli_epi32(_mm256_and_si256(y_qh, mask_03), 4)), offset_32);
                    __m256 w1 = _mm256_cvtepi32_ps(q1);
                    
                    // q2
                    __m256i q2 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(y_ql32, mask_0f), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(y_qh, 2), mask_03), 4)), offset_32);
                    __m256 w2 = _mm256_cvtepi32_ps(q2);

                    // q3
                    __m256i q3 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(y_ql0, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(y_qh, 4), mask_03), 4)), offset_32);
                    __m256 w3 = _mm256_cvtepi32_ps(q3);

                    // q4
                    __m256i q4 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(y_ql32, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(y_qh, 6), mask_03), 4)), offset_32);
                    __m256 w4 = _mm256_cvtepi32_ps(q4);

                    __m256 vx1 = _mm256_loadu_ps(&x[x_offset + n + l + 0]);
                    __m256 vx2 = _mm256_loadu_ps(&x[x_offset + n + l + 32]);
                    __m256 vx3 = _mm256_loadu_ps(&x[x_offset + n + l + 64]);
                    __m256 vx4 = _mm256_loadu_ps(&x[x_offset + n + l + 96]);

                    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_mul_ps(w1, vs1), vx1));
                    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_mul_ps(w2, vs3), vx2));
                    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_mul_ps(w3, vs5), vx3));
                    acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(_mm256_mul_ps(w4, vs7), vx4));
                }

                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        
        // Redução horizontal dos quatro acumuladores YMM de 256-bits para 1 float
        __m256 acc_total = _mm256_add_ps(_mm256_add_ps(acc1, acc2), _mm256_add_ps(acc3, acc4));
        __m128 sum_hi = _mm256_extractf128_ps(acc_total, 1);
        __m128 sum_lo = _mm256_castps256_ps128(acc_total);
        __m128 sum128 = _mm_add_ps(sum_lo, sum_hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[r] = _mm_cvtss_f32(sum128);
    }
}
#else
// Fallback para compilação sem suporte a AVX2 (delega ao kernel escalar genérico)
void gemv_q4_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    gemv_q4_K_scalar(matrix_weights, x, out, num_rows, num_cols);
}

void gemv_q6_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    gemv_q6_K_scalar(matrix_weights, x, out, num_rows, num_cols);
}
#endif
