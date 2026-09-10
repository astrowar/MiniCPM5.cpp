#pragma once

#include <vector>
#include <string>
#include <cstdint>

// Estrutura Base de um Tensor
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

// Rotary Position Embeddings (RoPE)
void apply_rope(std::vector<float>& vec, int pos, int head_idx, int head_dim, float rope_base);

// Multiplicação de Matrizes (GEMV) - Faz o dispatch automático com base no `type_str` do Tensor
// Suporta descompactação on-the-fly para Q4_K, Q8_0, Q6_K e F32
void matmul(std::vector<float>& output, const std::vector<float>& input, const Tensor& tensor);

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
