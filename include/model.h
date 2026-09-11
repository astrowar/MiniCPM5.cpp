#pragma once

#include <string>
#include <vector>
#include <memory>
#include "minicpm_metadata.h"
#include "ops.h"

// ============================================================================
// ESTRUTURAS DE DADOS DE ARQUITETURA DO MODELO
// ============================================================================

struct LayerWeights {
    int layer_id;
    Tensor attn_norm;
    Tensor attn_q;
    Tensor attn_k;
    Tensor attn_v;
    Tensor attn_output;
    Tensor ffn_norm;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
};

struct ModelWeights {
    Tensor token_embd;
    Tensor output_norm;
    Tensor output; // LM Head
    std::vector<LayerWeights> layers;
};

// ============================================================================
// MOTOR PRINCIPAL DE INFERÊNCIA DO MINICPM5
// ============================================================================

class MiniCPM5Engine {
public:
    MiniCPM5Engine() {
        weights_.layers.resize(MODEL_LAYERS);
    }

    void init_kv_cache(int max_seq_len);
    bool load_model(const std::string& gguf_path, bool verbose = false);
    const std::vector<float>& forward(int token_id, int pos, int max_seq_len);

private:
    std::unique_ptr<std::vector<char>> raw_gguf_buffer_;
    ModelWeights weights_;
    std::vector<KVCacheLayer> kv_caches_;

    std::vector<float> x_;
    std::vector<float> x_norm_;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> attn_out_temp_;
    std::vector<float> attn_output_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> ffn_intermediate_;
    std::vector<float> ffn_output_;
    std::vector<float> residual_;
    std::vector<float> logits_;
    std::vector<block_q8_K> x_norm_q8k_;
    std::vector<block_q8_K> ffn_intermediate_q8k_;
};
