#include "model.h"
#include "minicpm_metadata.h"
#include "ops.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>

void MiniCPM5Engine::init_kv_cache(int max_seq_len) {
    kv_caches_.resize(MODEL_LAYERS);
    int kv_size = max_seq_len * MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS);
    for (int i = 0; i < MODEL_LAYERS; ++i) {
        kv_caches_[i].k.assign(kv_size, 0.0f);
        kv_caches_[i].v.assign(kv_size, 0.0f);
    }
}

bool MiniCPM5Engine::load_model(const std::string& gguf_path, bool verbose) {
    if (verbose) std::cout << "[Engine] Abrindo arquivo do modelo GGUF..." << std::endl;
    std::ifstream file(gguf_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[Engine] Falha ao abrir o arquivo GGUF: " << gguf_path << std::endl;
        return false;
    }

    uint64_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (verbose) {
        std::cout << "[Engine] Alocando buffer único de GGUF na RAM ("
                  << std::fixed << std::setprecision(2) << (double)file_size / (1024.0 * 1024.0) << " MB)..." << std::endl;
    }

    raw_gguf_buffer_ = std::make_unique<std::vector<char>>(file_size);
    file.read(raw_gguf_buffer_->data(), file_size);
    if (!file) {
        std::cerr << "[Engine] Erro ao ler os dados do arquivo GGUF para a RAM." << std::endl;
        return false;
    }

    if (verbose) std::cout << "[Engine] GGUF carregado em RAM. Mapeando tensores da arquitetura..." << std::endl;

    const char* base_ptr = raw_gguf_buffer_->data();
    int mapped_count = 0;

    for (int i = 0; i < NUM_TENSORS; i++) {
        const auto& t_meta = TENSORS_INFO[i];

        Tensor tensor;
        tensor.name = t_meta.name;
        tensor.type_str = t_meta.type_str;
        tensor.size_bytes = t_meta.size_bytes;
        tensor.data_ptr = base_ptr + t_meta.absolute_offset;

        for (uint32_t d = 0; d < t_meta.n_dims; d++) {
            tensor.dims.push_back(t_meta.dims[d]);
        }

        if (tensor.name == "token_embd.weight") {
            weights_.token_embd = tensor;
            mapped_count++;
        } else if (tensor.name == "output_norm.weight") {
            weights_.output_norm = tensor;
            mapped_count++;
        } else if (tensor.name == "output.weight") {
            weights_.output = tensor;
            mapped_count++;
        } else if (tensor.name.rfind("blk.", 0) == 0) {
            size_t first_dot = tensor.name.find('.');
            size_t second_dot = tensor.name.find('.', first_dot + 1);
            std::string layer_id_str = tensor.name.substr(first_dot + 1, second_dot - first_dot - 1);
            int layer_id = std::stoi(layer_id_str);

            if (layer_id >= 0 && layer_id < MODEL_LAYERS) {
                LayerWeights& layer = weights_.layers[layer_id];
                layer.layer_id = layer_id;

                std::string sub_name = tensor.name.substr(second_dot + 1);
                if (sub_name == "attn_norm.weight") layer.attn_norm = tensor;
                else if (sub_name == "attn_q.weight") layer.attn_q = tensor;
                else if (sub_name == "attn_k.weight") layer.attn_k = tensor;
                else if (sub_name == "attn_v.weight") layer.attn_v = tensor;
                else if (sub_name == "attn_output.weight") layer.attn_output = tensor;
                else if (sub_name == "ffn_norm.weight") layer.ffn_norm = tensor;
                else if (sub_name == "ffn_gate.weight") layer.ffn_gate = tensor;
                else if (sub_name == "ffn_up.weight") layer.ffn_up = tensor;
                else if (sub_name == "ffn_down.weight") layer.ffn_down = tensor;

                mapped_count++;
            }
        }
    }

    if (verbose) std::cout << "[Engine] Mapeamento de " << mapped_count << " tensores concluido com sucesso!" << std::endl;
    return true;
}

const std::vector<float>& MiniCPM5Engine::forward(int token_id, int pos, int max_seq_len) {
    if (x_.size() != MODEL_DIM) x_.resize(MODEL_DIM);
    if (x_norm_.size() != MODEL_DIM) x_norm_.resize(MODEL_DIM);
    if (q_.size() != MODEL_DIM) q_.resize(MODEL_DIM);

    int kv_len = MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS);
    if (k_.size() != kv_len) k_.resize(kv_len);
    if (v_.size() != kv_len) v_.resize(kv_len);

    if (attn_out_temp_.size() != MODEL_DIM) attn_out_temp_.resize(MODEL_DIM);
    if (attn_output_.size() != MODEL_DIM) attn_output_.resize(MODEL_DIM);
    if (gate_.size() != MODEL_FFN_DIM) gate_.resize(MODEL_FFN_DIM);
    if (up_.size() != MODEL_FFN_DIM) up_.resize(MODEL_FFN_DIM);
    if (ffn_intermediate_.size() != MODEL_FFN_DIM) ffn_intermediate_.resize(MODEL_FFN_DIM);
    if (ffn_output_.size() != MODEL_DIM) ffn_output_.resize(MODEL_DIM);
    if (residual_.size() != MODEL_DIM) residual_.resize(MODEL_DIM);

    // 1. TOKEN EMBEDDING LOOKUP
    get_embedding(x_, weights_.token_embd, token_id);

    // 2. ITERAÇÃO DE CAMADAS TRANSFORMER
    for (int l = 0; l < MODEL_LAYERS; l++) {
        const LayerWeights& layer = weights_.layers[l];

        // --- BLOCO DE ATENÇÃO (GQA + RoPE) ---
        residual_ = x_;

        rmsnorm(x_norm_, x_, layer.attn_norm);

        if (x_norm_q8k_.size() != MODEL_DIM / QK_K) x_norm_q8k_.resize(MODEL_DIM / QK_K);
        quantize_row_q8_K(x_norm_.data(), MODEL_DIM, x_norm_q8k_.data());

        matmul_qkv_q8k(q_, k_, v_, x_norm_q8k_.data(), MODEL_DIM, layer.attn_q, layer.attn_k, layer.attn_v);

        int head_dim = MODEL_DIM / MODEL_HEADS;
        for (int h = 0; h < MODEL_HEADS; h++) {
            apply_rope(q_, pos, h, head_dim, MODEL_ROPE_BASE);
        }
        for (int h = 0; h < MODEL_KV_HEADS; h++) {
            apply_rope(k_, pos, h, head_dim, MODEL_ROPE_BASE);
        }

        execute_attention(attn_out_temp_, q_, k_, v_, kv_caches_[l], pos, max_seq_len, MODEL_HEADS, MODEL_KV_HEADS, head_dim);

        matmul(attn_output_, attn_out_temp_, layer.attn_output);

        for (int i = 0; i < MODEL_DIM; i++) {
            x_[i] = residual_[i] + attn_output_[i];
        }

        // --- BLOCO FEED-FORWARD (SwiGLU) ---
        residual_ = x_;

        rmsnorm(x_norm_, x_, layer.ffn_norm);

        if (x_norm_q8k_.size() != MODEL_DIM / QK_K) x_norm_q8k_.resize(MODEL_DIM / QK_K);
        quantize_row_q8_K(x_norm_.data(), MODEL_DIM, x_norm_q8k_.data());

        matmul_gate_up_q8k(gate_, up_, x_norm_q8k_.data(), MODEL_DIM, layer.ffn_gate, layer.ffn_up);

        for (int i = 0; i < MODEL_FFN_DIM; i++) {
            ffn_intermediate_[i] = silu(gate_[i]) * up_[i];
        }

        if (ffn_intermediate_q8k_.size() != MODEL_FFN_DIM / QK_K) ffn_intermediate_q8k_.resize(MODEL_FFN_DIM / QK_K);
        quantize_row_q8_K(ffn_intermediate_.data(), MODEL_FFN_DIM, ffn_intermediate_q8k_.data());

        matmul_q8k(ffn_output_, ffn_intermediate_q8k_.data(), MODEL_FFN_DIM, layer.ffn_down);

        for (int i = 0; i < MODEL_DIM; i++) {
            x_[i] = residual_[i] + ffn_output_[i];
        }
    }

    // 3. FINAL NORMALIZATION
    rmsnorm(x_norm_, x_, weights_.output_norm);

    // 4. LM HEAD PROJECTION
    matmul(logits_, x_norm_, weights_.output);

    return logits_;
}
