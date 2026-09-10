#include "ops.h"
#include <cmath>
#include <iostream>
#include <cassert>
#include <omp.h>

// ============================================================================
// 1. UTILITÁRIOS MATEMÁTICOS & ESTRUTURAS FÍSICAS DOS BLOCOS
// ============================================================================

typedef uint16_t ggml_fp16_t;

// Conversor portátil de Half-Float (16-bits) para Float padrão (32-bits)
static inline float fp16_to_fp32(ggml_fp16_t h) {
    union { uint32_t u; float f; } o;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;
    if (exp == 0x1F) {
        o.u = sign | 0x7F800000 | mant;
    } else if (exp == 0) {
        if (mant == 0) o.u = sign;
        else {
            while (!(mant & 0x00800000)) { mant <<= 1; exp--; }
            o.u = sign | ((exp + 113) << 23) | (mant & 0x007FFFFF);
        }
    } else {
        o.u = sign | ((exp + 112) << 23) | mant;
    }
    return o.f;
}

// Estrutura física Q8_0 (34 bytes)
#define QK8_0 32
struct block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[QK8_0];
};

// Estrutura física Q4_K (144 bytes)
#define QK_K 256
#define K_SCALE_SIZE 12
struct block_q4_K {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

// Estrutura física Q6_K (210 bytes)
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // 128 bytes (lower 4 bits)
    uint8_t qh[QK_K / 4];      // 64 bytes  (upper 2 bits)
    int8_t  scales[QK_K / 16]; // 16 bytes  (sub-block scales)
    ggml_fp16_t d;             // 2 bytes   (super-block scale)
};

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// ============================================================================
// 2. KERNELS DE PRODUTO ESCALAR "ON-THE-FLY" (GEMV) MULTI-THREAD (OPENMP)
// ============================================================================

// GEMV para Tensores Float32
void gemv_f32(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const float* w = reinterpret_cast<const float*>(matrix_weights);
    #pragma omp parallel for
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

    #pragma omp parallel for
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

// GEMV para Tensores Q4_K (Int4)
void gemv_q4_K(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    #pragma omp parallel for
    for (int r = 0; r < num_rows; ++r) {
        float row_sum = 0.0f;
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
                    row_sum += w1 * x[x_offset + j + l];
                    row_sum += w2 * x[x_offset + j + l + 32];
                }
                q += 32;
                is += 2;
            }
        }
        out[r] = row_sum;
    }
}

// GEMV para Tensores Q6_K (Int6)
void gemv_q6_K(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q6_K* blocks = reinterpret_cast<const block_q6_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    #pragma omp parallel for
    for (int r = 0; r < num_rows; ++r) {
        float row_sum = 0.0f;
        int row_block_offset = r * super_blocks_per_row;

        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q6_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            int x_offset = sb * QK_K;

            const uint8_t* ql = bloco.ql;
            const uint8_t* qh = bloco.qh;
            const int8_t* sc = bloco.scales;

            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    
                    // Reconstroi os 6-bits signed inteiros (offset -32)
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    row_sum += d * sc[is + 0] * q1 * x[x_offset + n + l +  0];
                    row_sum += d * sc[is + 2] * q2 * x[x_offset + n + l + 32];
                    row_sum += d * sc[is + 4] * q3 * x[x_offset + n + l + 64];
                    row_sum += d * sc[is + 6] * q4 * x[x_offset + n + l + 96];
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        out[r] = row_sum;
    }
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
    
    #pragma omp parallel for
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

    #pragma omp parallel for
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / num_queries_per_kv;
        const float* q_h = q.data() + h * head_dim;
        
        std::vector<float> scores(pos + 1);
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
        std::fill(out_h, out_h + head_dim, 0.0f);

        for (int t = 0; t <= pos; ++t) {
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
