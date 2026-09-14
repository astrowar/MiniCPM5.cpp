#pragma once

#include <vector>
#include <string>
#include <cstdint>

// ============================================================================
// GGML TENSOR MEMORY LAYOUTS & UTILITIES
// ============================================================================

typedef uint16_t ggml_fp16_t;

// Portable Half-Float (16-bits) to Float (32-bits) converter
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

// Q8_0 physical structure (34 bytes)
#define QK8_0 32
struct block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[QK8_0];
};

// Q4_K physical structure (144 bytes)
#define QK_K 256
#define K_SCALE_SIZE 12
struct block_q4_K {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

// Q6_K physical structure (210 bytes)
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // 128 bytes (lower 4 bits)
    uint8_t qh[QK_K / 4];      // 64 bytes  (upper 2 bits)
    int8_t  scales[QK_K / 16]; // 16 bytes  (sub-block scales)
    ggml_fp16_t d;             // 2 bytes   (super-block scale)
};

// Intermediate Q8_K: used to quantize activations before dot-product.
// 4 + 256 + 32 = 292 bytes per 256-element block.
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

static inline void decode_q4k_scales_mins(const uint8_t* q, uint8_t* sc, uint8_t* mn) {
    sc[0] = q[0] & 63;
    sc[1] = q[1] & 63;
    sc[2] = q[2] & 63;
    sc[3] = q[3] & 63;

    mn[0] = q[4] & 63;
    mn[1] = q[5] & 63;
    mn[2] = q[6] & 63;
    mn[3] = q[7] & 63;

    sc[4] = (q[8]  & 0x0F) | ((q[0] >> 6) << 4);
    sc[5] = (q[9]  & 0x0F) | ((q[1] >> 6) << 4);
    sc[6] = (q[10] & 0x0F) | ((q[2] >> 6) << 4);
    sc[7] = (q[11] & 0x0F) | ((q[3] >> 6) << 4);

    mn[4] = (q[8]  >> 4) | ((q[4] >> 6) << 4);
    mn[5] = (q[9]  >> 4) | ((q[5] >> 6) << 4);
    mn[6] = (q[10] >> 4) | ((q[6] >> 6) << 4);
    mn[7] = (q[11] >> 4) | ((q[7] >> 6) << 4);
}

// ============================================================================
// NEURAL NETWORK OPERATORS
// ============================================================================

// Base Structure for a Network Tensor
struct Tensor {
    std::string name;
    std::string type_str;
    std::vector<uint64_t> dims;     // [0] = colunas, [1] = linhas
    const char* data_ptr = nullptr; // Ponteiro para os dados brutos na RAM
    uint64_t size_bytes = 0;
};

// Funções de Ativação e Normalização
float silu(float x);
void rmsnorm(std::vector<float>& out, const std::vector<float>& x, const Tensor& weight_tensor, float eps = 1e-6f);

// Operações fundidas de alta performance (otimização de sincronização e localidade de cache)
void rmsnorm_and_quantize_q8k(block_q8_K* out_q8k, const std::vector<float>& x, const Tensor& weight_tensor, float eps = 1e-6f);
void swiglu_and_quantize_q8k(block_q8_K* out_q8k, const std::vector<float>& gate, const std::vector<float>& up);

// Rotary Position Embeddings (RoPE)
void apply_rope(std::vector<float>& vec, int pos, int head_idx, int head_dim, float rope_base);

// Multiplicação de Matrizes (GEMV) - Faz o dispatch automático com base no `type_str` do Tensor
// Suporta descompactação on-the-fly para Q4_K, Q8_0, Q6_K e F32
void matmul(std::vector<float>& output, const std::vector<float>& input, const Tensor& tensor);

// Quantiza uma linha FP32 para Q8_K (usada para reutilizar a quantização entre projeções)
void quantize_row_q8_K(const float* x, int n, block_q8_K* y);

// GEMV com entrada já quantizada Q8_K (pula a quantização interna)
void matmul_q8k(std::vector<float>& output, const block_q8_K* input_q8k, int num_cols, const Tensor& tensor);

// GEMVs fundidos: QKV em uma unica regiao OpenMP
void matmul_qkv_q8k(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v,
                    const block_q8_K* input_q8k, int num_cols,
                    const Tensor& tensor_q, const Tensor& tensor_k, const Tensor& tensor_v);

// GEMVs fundidos: Gate+Up em uma unica regiao OpenMP
void matmul_gate_up_q8k(std::vector<float>& gate, std::vector<float>& up,
                        const block_q8_K* input_q8k, int num_cols,
                        const Tensor& tensor_gate, const Tensor& tensor_up);

// Extração de linha de Embeddings
void get_embedding(std::vector<float>& out, const Tensor& embd_tensor, int token_id);

// KV Cache para uma única camada
struct KVCacheLayer {
    std::vector<float> k; // Armazena Keys de todos os tokens passados
    std::vector<float> v; // Armazena Values de todos os tokens passados
};

// Mecanismo de Atenção (GQA 8:1) com suporte a KV Cache
void execute_attention(std::vector<float>& attn_out, 
                       const std::vector<float>& q, 
                       const std::vector<float>& k_curr, 
                       const std::vector<float>& v_curr,
                       KVCacheLayer& kv_cache,
                       int pos,
                       int max_seq_len,
                       int num_heads, int num_kv_heads, int head_dim);
