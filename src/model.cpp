#include "model.h"
#include "minicpm_metadata.h"
#include "ops.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>

// ----------------------------------------------------------------------------
// KV CACHE INITIALIZATION
//
// The Key-Value (KV) Cache is pre-allocated once during engine startup.
// It stores the historical Key and Value projections for all past tokens,
// preventing the model from recalculating attention scores for the entire
// sequence during each autoregressive decoding step.
// ----------------------------------------------------------------------------
void MiniCPM5Engine::init_kv_cache(int max_seq_len) {
    kv_caches_.resize(MODEL_LAYERS);
    int kv_size = max_seq_len * MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS);
    for (int i = 0; i < MODEL_LAYERS; ++i) {
        kv_caches_[i].k.assign(kv_size, 0.0f);
        kv_caches_[i].v.assign(kv_size, 0.0f);
    }
}

// ----------------------------------------------------------------------------
// GGUF MODEL LOADER
//
// The GGUF file format concatenates hundreds of tensor arrays into a single
// binary blob. This function orchestrates reading the file into RAM and 
// subsequently mapping its contents to our logical weight structures.
// ----------------------------------------------------------------------------
bool MiniCPM5Engine::load_model(const std::string& gguf_path, bool verbose) {
    if (!load_gguf_file(gguf_path, verbose)) {
        return false;
    }
    map_tensors(verbose);
    return true;
}

// ----------------------------------------------------------------------------
// STEP 1: FILE I/O TO RAM
// ----------------------------------------------------------------------------
bool MiniCPM5Engine::load_gguf_file(const std::string& gguf_path, bool verbose) {
    if (verbose) std::cout << "[Engine] Opening GGUF model file..." << std::endl;
    std::ifstream file(gguf_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[Engine] Failed to open GGUF file: " << gguf_path << std::endl;
        return false;
    }

    uint64_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (verbose) {
        std::cout << "[Engine] Allocating single monolithic RAM buffer ("
                  << std::fixed << std::setprecision(2) << static_cast<double>(file_size) / (1024.0 * 1024.0) << " MB)..." << std::endl;
    }

    raw_gguf_buffer_ = std::make_unique<std::vector<char>>(file_size);
    file.read(raw_gguf_buffer_->data(), file_size);
    if (!file) {
        std::cerr << "[Engine] Error reading GGUF data into RAM." << std::endl;
        return false;
    }

    if (verbose) std::cout << "[Engine] GGUF loaded into RAM successfully." << std::endl;
    return true;
}

// ----------------------------------------------------------------------------
// STEP 2: TENSOR MAPPING
//
// Translates the monolithic binary buffer into structural tensor pointers based
// on the precomputed metadata (TENSORS_INFO). It matches strings like
// "blk.0.attn_q.weight" to the appropriate layer struct fields.
// ----------------------------------------------------------------------------
void MiniCPM5Engine::map_tensors(bool verbose) {
    if (verbose) std::cout << "[Engine] Mapping architecture tensors..." << std::endl;

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

    if (verbose) std::cout << "[Engine] Successfully mapped " << mapped_count << " tensors!" << std::endl;
}

// ----------------------------------------------------------------------------
// STEP 3: FORWARD PASS (INFERENCE GRAPH)
//
// This is the beating heart of the Large Language Model. The forward pass takes
// a single token ID and propagates it sequentially through the entire neural
// network architecture to predict the next token (returning the logits array).
// ----------------------------------------------------------------------------
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

    // ------------------------------------------------------------------------
    // A) TOKEN EMBEDDING LOOKUP
    // Converts the discrete vocabulary token ID into a continuous high-dimensional
    // floating-point vector (x) representing its semantic meaning.
    // ------------------------------------------------------------------------
    get_embedding(x_, weights_.token_embd, token_id);

    // ------------------------------------------------------------------------
    // B) TRANSFORMER LAYERS ITERATION
    // The signal passes sequentially through all hidden layers. Each layer consists
    // of two main blocks: The Attention Block and the Feed-Forward Block.
    // ------------------------------------------------------------------------
    for (int l = 0; l < MODEL_LAYERS; l++) {
        const LayerWeights& layer = weights_.layers[l];

        // --- ATTENTION BLOCK (GQA + RoPE) ---
        // Learns contextual relationships between tokens.
        residual_ = x_; // Save input for residual connection

        // Pre-normalization (RMSNorm) and dynamic 8-bit quantization to accelerate GEMV
        if (x_norm_q8k_.size() != MODEL_DIM / QK_K) x_norm_q8k_.resize(MODEL_DIM / QK_K);
        rmsnorm_and_quantize_q8k(x_norm_q8k_.data(), x_, layer.attn_norm);

        // Fused QKV Projection: map normalized input into Query, Key, and Value spaces
        matmul_qkv_q8k(q_, k_, v_, x_norm_q8k_.data(), MODEL_DIM, layer.attn_q, layer.attn_k, layer.attn_v);

        // Rotary Positional Embeddings (RoPE): inject relative positional information
        int head_dim = MODEL_DIM / MODEL_HEADS;
        for (int h = 0; h < MODEL_HEADS; h++) {
            apply_rope(q_, pos, h, head_dim, MODEL_ROPE_BASE);
        }
        for (int h = 0; h < MODEL_KV_HEADS; h++) {
            apply_rope(k_, pos, h, head_dim, MODEL_ROPE_BASE);
        }

        // Grouped-Query Attention computation using the historical KV Cache
        execute_attention(attn_out_temp_, q_, k_, v_, kv_caches_[l], pos, max_seq_len, MODEL_HEADS, MODEL_KV_HEADS, head_dim);

        // Final output projection of the attention block
        matmul(attn_output_, attn_out_temp_, layer.attn_output);

        // Add residual connection: x = x + attention_output
        for (int i = 0; i < MODEL_DIM; i++) {
            x_[i] = residual_[i] + attn_output_[i];
        }

        // --- FEED-FORWARD BLOCK (SwiGLU) ---
        // Acts as the model's factual memory and non-linear feature extractor.
        residual_ = x_;

        if (x_norm_q8k_.size() != MODEL_DIM / QK_K) x_norm_q8k_.resize(MODEL_DIM / QK_K);
        rmsnorm_and_quantize_q8k(x_norm_q8k_.data(), x_, layer.ffn_norm);

        // Fused Gate and Up projections
        matmul_gate_up_q8k(gate_, up_, x_norm_q8k_.data(), MODEL_DIM, layer.ffn_gate, layer.ffn_up);

        // Apply SiLU activation and element-wise multiplication (SwiGLU)
        if (ffn_intermediate_q8k_.size() != MODEL_FFN_DIM / QK_K) ffn_intermediate_q8k_.resize(MODEL_FFN_DIM / QK_K);
        swiglu_and_quantize_q8k(ffn_intermediate_q8k_.data(), gate_, up_);

        // Down projection back to the model's main dimension
        matmul_q8k(ffn_output_, ffn_intermediate_q8k_.data(), MODEL_FFN_DIM, layer.ffn_down);

        // Add residual connection: x = x + ffn_output
        for (int i = 0; i < MODEL_DIM; i++) {
            x_[i] = residual_[i] + ffn_output_[i];
        }
    }

    // ------------------------------------------------------------------------
    // C) FINAL NORMALIZATION
    // Stabilizes the vector before generating probability distributions.
    // ------------------------------------------------------------------------
    rmsnorm(x_norm_, x_, weights_.output_norm);

    // ------------------------------------------------------------------------
    // D) LM HEAD PROJECTION
    // Projects the final hidden state back into the vocabulary space to generate
    // the raw unnormalized scores (logits) for predicting the next token.
    // ------------------------------------------------------------------------
    matmul(logits_, x_norm_, weights_.output);

    return logits_;
}
