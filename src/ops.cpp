#include "ops.h"
#include "ops_internal.h"
#include <cmath>
#include <iostream>
#include <cassert>
#include <cstring>

// ============================================================================
// 2. KERNELS DE PRODUTO ESCALAR "ON-THE-FLY" (GEMV) MULTI-THREAD (OPENMP)
// ============================================================================

// GEMV para Tensores Float32
void gemv_f32(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const float* w = reinterpret_cast<const float*>(matrix_weights);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        float row_sum = 0.0f;
        int row_offset = r * num_cols;
        for (int c = 0; c < num_cols; ++c) {
            row_sum += w[row_offset + c] * x[c];
        }
        out[r] = row_sum;
    }
}

// GEMV para Tensores Q8_0 (Int8)
void gemv_q8_0(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(matrix_weights);
    int blocks_per_row = num_cols / QK8_0;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        float row_sum = 0.0f;
        int row_block_offset = r * blocks_per_row;

        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q8_0& bloco = blocks[row_block_offset + b];
            float scale = fp16_to_fp32(bloco.d);
            
            float sum_local = 0.0f;
            for (int i = 0; i < QK8_0; ++i) {
                sum_local += bloco.qs[i] * x[b * QK8_0 + i];
            }
            row_sum += sum_local * scale;
        }
        out[r] = row_sum;
    }
}

// GEMV para Tensores Q4_K (Int4) - Escalar Puro
void gemv_q4_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        int row_block_offset = r * super_blocks_per_row;

        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q4_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            float dmin = fp16_to_fp32(bloco.dmin);
            int x_offset = sb * QK_K;
            
            int is = 0;
            uint8_t sc, m;
            const uint8_t* q = bloco.qs;

            // Processa em chunks de 64
            for (int j = 0; j < QK_K; j += 64) {
                get_scale_min_k4(is + 0, bloco.scales, &sc, &m);
                float d1 = d * sc; float m1 = dmin * m;
                
                get_scale_min_k4(is + 1, bloco.scales, &sc, &m);
                float d2 = d * sc; float m2 = dmin * m;

                for (int l = 0; l < 32; ++l) {
                    float w1 = d1 * (q[l] & 0xF) - m1;
                    float w2 = d2 * (q[l] >> 4) - m2;
                    sum1 += w1 * x[x_offset + j + l];
                    sum2 += w2 * x[x_offset + j + l + 32];
                }
                q += 32;
                is += 2;
            }
        }
        out[r] = sum1 + sum2;
    }
}

// Dispatcher para GEMV Q4_K
void gemv_q4_K(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
#if defined(__AVX2__)
    gemv_q4_K_avx2(matrix_weights, x, out, num_rows, num_cols);
#else
    gemv_q4_K_scalar(matrix_weights, x, out, num_rows, num_cols);
#endif
}

// GEMV para Tensores Q6_K (Int6) - Escalar Puro
void gemv_q6_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q6_K* blocks = reinterpret_cast<const block_q6_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sum4 = 0.0f;
        int row_block_offset = r * super_blocks_per_row;

        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q6_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            int x_offset = sb * QK_K;

            const uint8_t* ql = bloco.ql;
            const uint8_t* qh = bloco.qh;
            const int8_t* sc = bloco.scales;

            for (int n = 0; n < QK_K; n += 128) {
                // Primeira metade do loop (l = 0..15, is = 0)
                float s0 = d * sc[0];
                float s2 = d * sc[2];
                float s4 = d * sc[4];
                float s6 = d * sc[6];

                for (int l = 0; l < 16; ++l) {
                    // Reconstroi os 6-bits signed inteiros (offset -32)
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    sum1 += s0 * q1 * x[x_offset + n + l +  0];
                    sum2 += s2 * q2 * x[x_offset + n + l + 32];
                    sum3 += s4 * q3 * x[x_offset + n + l + 64];
                    sum4 += s6 * q4 * x[x_offset + n + l + 96];
                }

                // Segunda metade do loop (l = 16..31, is = 1)
                float s1 = d * sc[1];
                float s3 = d * sc[3];
                float s5 = d * sc[5];
                float s7 = d * sc[7];

                for (int l = 16; l < 32; ++l) {
                    // Reconstroi os 6-bits signed inteiros (offset -32)
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    sum1 += s1 * q1 * x[x_offset + n + l +  0];
                    sum2 += s3 * q2 * x[x_offset + n + l + 32];
                    sum3 += s5 * q3 * x[x_offset + n + l + 64];
                    sum4 += s7 * q4 * x[x_offset + n + l + 96];
                }

                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        out[r] = sum1 + sum2 + sum3 + sum4;
    }
}

// Dispatcher para GEMV Q6_K
void gemv_q6_K(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
#if defined(__AVX2__)
    gemv_q6_K_avx2(matrix_weights, x, out, num_rows, num_cols);
#else
    gemv_q6_K_scalar(matrix_weights, x, out, num_rows, num_cols);
#endif
}

// ============================================================================
// 3. DISPATCHER E OPERAÇÕES DA REDE NEURAL
// ============================================================================

void matmul(std::vector<float>& output, const std::vector<float>& input, const Tensor& tensor) {
    uint64_t num_cols = tensor.dims[0];
    uint64_t num_rows = tensor.dims.size() > 1 ? tensor.dims[1] : 1;
    
    // Ajusta o tamanho da saída se necessário
    if (output.size() != num_rows) {
        output.resize(num_rows);
    }

    if (tensor.type_str == "F32") {
        gemv_f32(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q8_0") {
        gemv_q8_0(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q4_K") {
        gemv_q4_K(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q6_K") {
        gemv_q6_K(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else {
        // Fallback para tipos não implementados
        static bool warned = false;
        if (!warned) {
            std::cout << "[AVISO] Kernel para o tipo " << tensor.type_str << " ainda nao foi implementado. Usando zeros." << std::endl;
            warned = true;
        }
        std::fill(output.begin(), output.end(), 0.001f);
    }
}

void quantize_row_q8_K(const float* x, int n, block_q8_K* y) {
#if defined(__AVX2__)
    quantize_row_q8_K_avx2(x, y, n);
#else
    for (int ib = 0; ib < n / QK_K; ++ib) {
        const float* xb = x + ib * QK_K;
        block_q8_K& dst = y[ib];
        float amax = 0.0f;
        for (int i = 0; i < QK_K; ++i) amax = std::max(amax, std::fabs(xb[i]));
        if (amax == 0.0f) { dst.d = 0.0f; std::memset(dst.qs, 0, QK_K); std::memset(dst.bsums, 0, sizeof(dst.bsums)); continue; }
        dst.d = amax / 127.0f;
        float mul = 127.0f / amax;
        for (int i = 0; i < QK_K; ++i) dst.qs[i] = (int8_t)std::max(-127.0f, std::min(127.0f, (float)std::lround(xb[i] * mul)));
        for (int g = 0; g < QK_K / 16; ++g) {
            int32_t s = 0;
            for (int i = 0; i < 16; ++i) s += dst.qs[g * 16 + i];
            dst.bsums[g] = (int16_t)s;
        }
    }
#endif
}

void matmul_q8k(std::vector<float>& output, const block_q8_K* input_q8k, int num_cols, const Tensor& tensor) {
    uint64_t num_rows = tensor.dims.size() > 1 ? tensor.dims[1] : 1;
    if (output.size() != num_rows) {
        output.resize(num_rows);
    }
#if defined(__AVX2__)
    if (tensor.type_str == "Q4_K") {
        gemv_q4_K_q8_K_avx2(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q6_K") {
        gemv_q6_K_q8_K_avx2(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    } else
#endif
    {
        static bool warned = false;
        if (!warned) {
            std::cout << "[AVISO] matmul_q8k: tipo " << tensor.type_str << " nao suportado." << std::endl;
            warned = true;
        }
        std::fill(output.begin(), output.end(), 0.0f);
    }
}

void matmul_qkv_q8k(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v,
                    const block_q8_K* input_q8k, int num_cols,
                    const Tensor& tensor_q, const Tensor& tensor_k, const Tensor& tensor_v) {
    const int q_rows = static_cast<int>(tensor_q.dims.size() > 1 ? tensor_q.dims[1] : 1);
    const int kv_rows = static_cast<int>(tensor_k.dims.size() > 1 ? tensor_k.dims[1] : 1);
    if (q.size() != q_rows) q.resize(q_rows);
    if (k.size() != kv_rows) k.resize(kv_rows);
    if (v.size() != kv_rows) v.resize(kv_rows);

#if defined(__AVX2__)
    if (tensor_v.type_str == "Q4_K") {
        gemv_qkv_q4_K_q8_K_avx2(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                 input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    } else if (tensor_v.type_str == "Q6_K") {
        gemv_qkv_q4_q4_q6_q8_K_avx2(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                     input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    } else
#endif
    {
        static bool warned = false;
        if (!warned) {
            std::cout << "[AVISO] matmul_qkv_q8k: tipo v=" << tensor_v.type_str << " nao suportado." << std::endl;
            warned = true;
        }
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
    }
}

void matmul_gate_up_q8k(std::vector<float>& gate, std::vector<float>& up,
                        const block_q8_K* input_q8k, int num_cols,
                        const Tensor& tensor_gate, const Tensor& tensor_up) {
    const int num_rows = static_cast<int>(tensor_gate.dims.size() > 1 ? tensor_gate.dims[1] : 1);
    if (gate.size() != num_rows) gate.resize(num_rows);
    if (up.size() != num_rows) up.resize(num_rows);

#if defined(__AVX2__)
    if (tensor_gate.type_str == "Q4_K" && tensor_up.type_str == "Q4_K") {
        gemv_gate_up_q4_K_q8_K_avx2(tensor_gate.data_ptr, tensor_up.data_ptr,
                                     input_q8k, gate.data(), up.data(), num_rows, num_cols);
    } else
#endif
    {
        static bool warned = false;
        if (!warned) {
            std::cout << "[AVISO] matmul_gate_up_q8k: tipos nao suportados." << std::endl;
            warned = true;
        }
        std::fill(gate.begin(), gate.end(), 0.0f);
        std::fill(up.begin(), up.end(), 0.0f);
    }
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void rmsnorm(std::vector<float>& out, const std::vector<float>& x, const Tensor& weight_tensor, float eps) {
    const float* w = reinterpret_cast<const float*>(weight_tensor.data_ptr);
    int dim = x.size();
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt(sum / dim + eps);
    if (out.size() != dim) out.resize(dim);
    
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * rms * w[i];
    }
}

void apply_rope(std::vector<float>& vec, int pos, int head_idx, int head_dim, float rope_base) {
    for (int i = 0; i < head_dim; i += 2) {
        int idx = head_idx * head_dim + i;
        float theta = 1.0f / std::pow(rope_base, (float)i / head_dim);
        float m_theta = pos * theta;
        float cos_val = std::cos(m_theta);
        float sin_val = std::sin(m_theta);
        
        float v0 = vec[idx];
        float v1 = vec[idx + 1];
        
        vec[idx]     = v0 * cos_val - v1 * sin_val;
        vec[idx + 1] = v0 * sin_val + v1 * cos_val;
    }
}

// Atenção GQA (Grouped-Query Attention) com suporte a KV Cache
void execute_attention(std::vector<float>& attn_out, 
                       const std::vector<float>& q, 
                       const std::vector<float>& k_curr, 
                       const std::vector<float>& v_curr,
                       KVCacheLayer& kv_cache,
                       int pos,
                       int max_seq_len,
                       int num_heads, int num_kv_heads, int head_dim) {
    
    if (attn_out.size() != num_heads * head_dim) {
        attn_out.resize(num_heads * head_dim);
    }

    if (pos >= max_seq_len) {
        std::cerr << "[AVISO] Contexto máximo excedido (pos >= max_seq_len)." << std::endl;
        return;
    }

    // 1. Salva o Key e Value atuais no KV Cache da camada
    int kv_size_per_token = num_kv_heads * head_dim;
    int kv_offset = pos * kv_size_per_token;
    
    for (int i = 0; i < kv_size_per_token; ++i) {
        kv_cache.k[kv_offset + i] = k_curr[i];
        kv_cache.v[kv_offset + i] = v_curr[i];
    }

    // 2. Calcula a Atenção Escalonada: Softmax(Q * K_T) * V
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int num_queries_per_kv = num_heads / num_kv_heads;

    std::vector<float> all_scores(num_heads * (pos + 1));

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / num_queries_per_kv;
        const float* q_h = q.data() + h * head_dim;
        
        float* scores = all_scores.data() + h * (pos + 1);
        float max_score = -1e9f;

        // Q * K_T (Produto Escalar com todos os tokens passados)
        for (int t = 0; t <= pos; ++t) {
            const float* k_h_t = kv_cache.k.data() + (t * kv_size_per_token) + (kv_h * head_dim);
            
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                dot += q_h[i] * k_h_t[i];
            }
            dot *= scale;
            
            scores[t] = dot;
            if (dot > max_score) {
                max_score = dot;
            }
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t <= pos; ++t) {
            scores[t] /= sum_exp;
        }

        // Output = Softmax_Scores * V
        float* out_h = attn_out.data() + h * head_dim;

        // Inicializa com t = 0 diretamente para evitar std::fill
        const float* v_h_0 = kv_cache.v.data() + (0 * kv_size_per_token) + (kv_h * head_dim);
        float score_0 = scores[0];
        for (int i = 0; i < head_dim; ++i) {
            out_h[i] = score_0 * v_h_0[i];
        }

        for (int t = 1; t <= pos; ++t) {
            const float* v_h_t = kv_cache.v.data() + (t * kv_size_per_token) + (kv_h * head_dim);
            float score = scores[t];
            
            for (int i = 0; i < head_dim; ++i) {
                out_h[i] += score * v_h_t[i];
            }
        }
    }
}

// Extração de Embeddings baseada no formato do Tensor
void get_embedding(std::vector<float>& out, const Tensor& embd_tensor, int token_id) {
    int dim = embd_tensor.dims[0];
    if (out.size() != dim) out.resize(dim);

    if (embd_tensor.type_str == "Q4_K") {
        const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(embd_tensor.data_ptr);
        int super_blocks_per_row = dim / QK_K;
        
        // Pula para os super-blocos do token_id solicitado
        int row_block_offset = token_id * super_blocks_per_row;
        
        // Esta operação lê uma única linha (2048 elementos / 8 blocos), então OpenMP não é tão necessário aqui
        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q4_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            float dmin = fp16_to_fp32(bloco.dmin);
            int x_offset = sb * QK_K;
            
            int is = 0;
            uint8_t sc, m;
            const uint8_t* q = bloco.qs;

            // Processa em chunks de 64 (mesma matemática da dequantização q4_k)
            for (int j = 0; j < QK_K; j += 64) {
                get_scale_min_k4(is + 0, bloco.scales, &sc, &m);
                float d1 = d * sc; float m1 = dmin * m;
                
                get_scale_min_k4(is + 1, bloco.scales, &sc, &m);
                float d2 = d * sc; float m2 = dmin * m;

                for (int l = 0; l < 32; ++l) {
                    float w1 = d1 * (q[l] & 0xF) - m1;
                    float w2 = d2 * (q[l] >> 4) - m2;
                    out[x_offset + j + l] = w1;
                    out[x_offset + j + l + 32] = w2;
                }
                q += 32;
                is += 2;
            }
        }
    } else {
        std::cerr << "[AVISO] get_embedding so foi implementado para Q4_K." << std::endl;
        std::fill(out.begin(), out.end(), 0.0f);
    }
}

// Operações fundidas de alta performance para otimização de sincronização OpenMP e localidade de cache
void rmsnorm_and_quantize_q8k(block_q8_K* out_q8k, const std::vector<float>& x, const Tensor& weight_tensor, float eps) {
    int dim = x.size();
    const float* w = reinterpret_cast<const float*>(weight_tensor.data_ptr);

    // Compute sum of squares
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt(sum / dim + eps);

    int num_blocks = dim / QK_K;
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < num_blocks; ib++) {
        const float* xb = x.data() + ib * QK_K;
        const float* wb = w + ib * QK_K;
        block_q8_K& dst = out_q8k[ib];

        // 1. Compute rmsnorm on-the-fly for 256 elements
        float local_x[QK_K];
        float amax = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            local_x[i] = xb[i] * rms * wb[i];
            amax = std::max(amax, std::fabs(local_x[i]));
        }

        // 2. Quantize on-the-fly
        if (amax == 0.0f) {
            dst.d = 0.0f;
            std::memset(dst.qs, 0, QK_K);
            std::memset(dst.bsums, 0, sizeof(dst.bsums));
            continue;
        }

        dst.d = amax / 127.0f;
        float mul = 127.0f / amax;

        int32_t bsums[QK_K / 16] = {0};
        for (int i = 0; i < QK_K; ++i) {
            dst.qs[i] = (int8_t)std::max(-127.0f, std::min(127.0f, (float)std::lround(local_x[i] * mul)));
            bsums[i / 16] += dst.qs[i];
        }
        for (int g = 0; g < QK_K / 16; ++g) {
            dst.bsums[g] = (int16_t)bsums[g];
        }
    }
}

void swiglu_and_quantize_q8k(block_q8_K* out_q8k, const std::vector<float>& gate, const std::vector<float>& up) {
    int dim = gate.size();
    int num_blocks = dim / QK_K;

    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < num_blocks; ib++) {
        const float* gb = gate.data() + ib * QK_K;
        const float* ub = up.data() + ib * QK_K;
        block_q8_K& dst = out_q8k[ib];

        // 1. SwiGLU on-the-fly
        float local_x[QK_K];
        float amax = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            float g = gb[i];
            float silu_g = g / (1.0f + std::exp(-g));
            local_x[i] = silu_g * ub[i];
            amax = std::max(amax, std::fabs(local_x[i]));
        }

        // 2. Quantize on-the-fly
        if (amax == 0.0f) {
            dst.d = 0.0f;
            std::memset(dst.qs, 0, QK_K);
            std::memset(dst.bsums, 0, sizeof(dst.bsums));
            continue;
        }

        dst.d = amax / 127.0f;
        float mul = 127.0f / amax;

        int32_t bsums[QK_K / 16] = {0};
        for (int i = 0; i < QK_K; ++i) {
            dst.qs[i] = (int8_t)std::max(-127.0f, std::min(127.0f, (float)std::lround(local_x[i] * mul)));
            bsums[i / 16] += dst.qs[i];
        }
        for (int g = 0; g < QK_K / 16; ++g) {
            dst.bsums[g] = (int16_t)bsums[g];
        }
    }
}
