#include "minicpm_metadata.h"
#include "ops.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cassert>
#include <memory>
#include <random>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// 1. ESTRUTURAS DE DADOS DE ARQUITETURA DO MODELO (LAYERS & WEIGHTS)
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
// 2. TOKENIZADOR E DETOKENIZADOR (Conversão Texto <-> IDs)
// ============================================================================

class Tokenizer {
public:
    Tokenizer() {
        init_unicode_mappings();
    }

    bool load(const std::string& gguf_path, bool verbose = false) {
        std::ifstream f(gguf_path, std::ios::binary);
        if (!f) return false;

        uint32_t magic, version;
        f.read((char*)&magic, 4);
        f.read((char*)&version, 4);
        
        uint64_t tensor_count, kv_count;
        f.read((char*)&tensor_count, 8);
        f.read((char*)&kv_count, 8);

        bool found = false;
        for (uint64_t i = 0; i < kv_count; i++) {
            uint64_t key_len;
            f.read((char*)&key_len, 8);
            std::string key(key_len, '\0');
            f.read(&key[0], key_len);

            uint32_t val_type;
            f.read((char*)&val_type, 4);

            if (key == "tokenizer.ggml.tokens") {
                uint32_t arr_type;
                f.read((char*)&arr_type, 4); // Deve ser 8 (STRING)

                uint64_t arr_len;
                f.read((char*)&arr_len, 8);

                vocab_.resize(arr_len);
                vocab_map_.reserve(arr_len);
                for (uint64_t j = 0; j < arr_len; j++) {
                    uint64_t str_len;
                    f.read((char*)&str_len, 8);
                    std::string token(str_len, '\0');
                    f.read(&token[0], str_len);

                    // MiniCPM/GPT-2 vocab tokens are already in correct byte format.
                    // Store them directly without BPE decoding.
                    vocab_[j] = token;
                    vocab_map_[vocab_[j]] = j;
                }
                found = true;
                break;
            } else {
                skip_gguf_value(f, val_type);
            }
        }
        
        if (found && verbose) {
            std::cout << "[Tokenizer] Vocabulario lido dinamicamente (" << vocab_.size() << " tokens) direto do arquivo GGUF." << std::endl;
        }
        return found;
    }

    std::string decode(int token_id) {
        if (token_id < 0 || token_id >= vocab_.size()) return "<UNK>";
        
        std::string token = vocab_[token_id];
        
        // Byte fallback tokens <0xXX>
        if (token.size() == 6 && token.substr(0, 3) == "<0x" && token.back() == '>') {
            std::string hex_str = token.substr(3, 2);
            char byte_val = (char)std::stoi(hex_str, nullptr, 16);
            token = std::string(1, byte_val);
        }

        std::string raw_bytes = map_unicode_to_bytes(token);
        byte_buffer_ += raw_bytes;

        std::string valid_output = "";
        size_t i = 0;
        
        while (i < byte_buffer_.size()) {
            unsigned char c = byte_buffer_[i];
            size_t bytes_needed = 0;
            
            if ((c & 0x80) == 0x00) { bytes_needed = 1; }
            else if ((c & 0xE0) == 0xC0) { bytes_needed = 2; }
            else if ((c & 0xF0) == 0xE0) { bytes_needed = 3; }
            else if ((c & 0xF8) == 0xF0) { bytes_needed = 4; }
            else { bytes_needed = 1; }
            
            if (i + bytes_needed > byte_buffer_.size()) {
                break;
            }
            
            valid_output += byte_buffer_.substr(i, bytes_needed);
            i += bytes_needed;
        }
        
        byte_buffer_.erase(0, i);
        return valid_output;
    }

    int get_id(const std::string& str) const {
        auto it = vocab_map_.find(str);
        if (it != vocab_map_.end()) {
            return it->second;
        }
        return -1;
    }

    std::vector<int> tokenize(std::string text) const {
        std::vector<int> tokens;

        size_t start_idx = 0;
        if (!text.empty() && text[0] == '\x01') {
            tokens.push_back(0); // BOS token is 0
            start_idx = 1;
        }

        std::string mapped_text = "";
        for (size_t i = start_idx; i < text.size(); ++i) {
            unsigned char c = text[i];
            mapped_text += byte_to_unicode_[c];
        }

        size_t i = 0;
        while (i < mapped_text.length()) {
            int best_id = -1;
            size_t best_len = 0;

            for (size_t len = mapped_text.length() - i; len > 0; len--) {
                std::string sub = mapped_text.substr(i, len);
                auto it = vocab_map_.find(sub);
                if (it != vocab_map_.end()) {
                    best_id = it->second;
                    best_len = len;
                    break;
                }
            }

            if (best_id != -1) {
                tokens.push_back(best_id);
                i += best_len;
            } else {
                i++;
            }
        }
        return tokens;
    }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> vocab_map_;
    std::string byte_buffer_;

    std::vector<std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    void init_unicode_mappings() {
        byte_to_unicode_.resize(256);
        unicode_to_byte_.clear();

        std::vector<int> bs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        auto unicode_to_utf8 = [](uint32_t cp) -> std::string {
            std::string s;
            if (cp <= 0x7F) {
                s += (char)cp;
            } else if (cp <= 0x7FF) {
                s += (char)(0xC0 | ((cp >> 6) & 0x1F));
                s += (char)(0x80 | (cp & 0x3F));
            } else if (cp <= 0xFFFF) {
                s += (char)(0xE0 | ((cp >> 12) & 0x0F));
                s += (char)(0x80 | ((cp >> 6) & 0x3F));
                s += (char)(0x80 | (cp & 0x3F));
            } else if (cp <= 0x10FFFF) {
                s += (char)(0xF0 | ((cp >> 18) & 0x07));
                s += (char)(0x80 | ((cp >> 12) & 0x3F));
                s += (char)(0x80 | ((cp >> 6) & 0x3F));
                s += (char)(0x80 | (cp & 0x3F));
            }
            return s;
        };

        for (size_t i = 0; i < bs.size(); ++i) {
            std::string utf8_char = unicode_to_utf8(cs[i]);
            byte_to_unicode_[bs[i]] = utf8_char;
            unicode_to_byte_[utf8_char] = bs[i];
        }
    }

    std::string map_unicode_to_bytes(const std::string& text) const {
        std::string raw_bytes = "";
        size_t i = 0;
        while (i < text.size()) {
            unsigned char c = text[i];
            size_t len = 1;
            if ((c & 0x80) == 0x00) { len = 1; }
            else if ((c & 0xE0) == 0xC0) { len = 2; }
            else if ((c & 0xF0) == 0xE0) { len = 3; }
            else if ((c & 0xF8) == 0xF0) { len = 4; }
            
            if (i + len > text.size()) {
                raw_bytes += (char)c;
                i++;
                continue;
            }
            
            std::string utf8_char = text.substr(i, len);
            auto it = unicode_to_byte_.find(utf8_char);
            if (it != unicode_to_byte_.end()) {
                raw_bytes += (char)it->second;
            } else {
                raw_bytes += utf8_char;
            }
            i += len;
        }
        return raw_bytes;
    }

    // Utilitário para pular os metadados do GGUF no stream
    void skip_gguf_value(std::ifstream& f, uint32_t type) {
        uint64_t dummy64; uint32_t dummy32;
        switch (type) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
            case 2: case 3: f.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
            case 8: // STRING
                f.read((char*)&dummy64, 8);
                f.seekg(dummy64, std::ios::cur);
                break;
            case 9: // ARRAY
                f.read((char*)&dummy32, 4);
                f.read((char*)&dummy64, 8);
                for (uint64_t k = 0; k < dummy64; k++) skip_gguf_value(f, dummy32);
                break;
        }
    }
};

struct ChatMessage {
    std::string role;
    std::string content;
};

// Implementação C++ do chat_template.jinja do MiniCPM5
std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_think = true) {
    // BOS token is 0x01 in the GPT-2 style tokenizer (token id 0)
    std::string prompt = "\001";
    
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            prompt += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "user") {
            prompt += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "assistant") {
            if (enable_think) {
                prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + msg.content + "<|im_end|>\n";
            } else {
                prompt += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
    }
    
    if (add_generation_prompt) {
        if (enable_think) {
            prompt += "<|im_start|>assistant\n<think>\n";
        } else {
            prompt += "<|im_start|>assistant\n<think>\nNo think is needed for this response.\n\n</think>\n\n";
        }
    }
    return prompt;
}

// ============================================================================
// 3. MOTOR PRINCIPAL DE INFERÊNCIA DO MINICPM5
// ============================================================================

class MiniCPM5Engine {
public:
    MiniCPM5Engine() {
        weights_.layers.resize(MODEL_LAYERS);
    }
    
    // Inicializa o KV Cache para todas as camadas com tamanho dinamico
    void init_kv_cache(int max_seq_len) {
        kv_caches_.resize(MODEL_LAYERS);
        int kv_size = max_seq_len * MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS);
        for (int i = 0; i < MODEL_LAYERS; ++i) {
            kv_caches_[i].k.assign(kv_size, 0.0f);
            kv_caches_[i].v.assign(kv_size, 0.0f);
        }
    }

    // Carrega e mapeia todo o arquivo GGUF para as estruturas do C++
    bool load_model(const std::string& gguf_path, bool verbose = false) {
        if (verbose) std::cout << "[Engine] Abrindo arquivo do modelo GGUF..." << std::endl;
        std::ifstream file(gguf_path, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "[Engine] Falha ao abrir o arquivo GGUF: " << gguf_path << std::endl;
            return false;
        }

        // Obtém o tamanho do arquivo do modelo
        uint64_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (verbose) {
            std::cout << "[Engine] Alocando buffer único de GGUF na RAM (" 
                      << std::fixed << std::setprecision(2) << (double)file_size / (1024.0 * 1024.0) << " MB)..." << std::endl;
        }
        
        // Carrega o arquivo inteiro de uma só vez para evitar latência física de múltiplos reads
        raw_gguf_buffer_ = std::make_unique<std::vector<char>>(file_size);
        file.read(raw_gguf_buffer_->data(), file_size);
        if (!file) {
            std::cerr << "[Engine] Erro ao ler os dados do arquivo GGUF para a RAM." << std::endl;
            return false;
        }

        if (verbose) std::cout << "[Engine] GGUF carregado em RAM. Mapeando tensores da arquitetura..." << std::endl;

        // Mapeia cada um dos 381 tensores para o endereço correto na RAM
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

            // Faz o roteamento/vinculo do tensor para a estrutura correta do modelo
            if (tensor.name == "token_embd.weight") {
                weights_.token_embd = tensor;
                mapped_count++;
            } else if (tensor.name == "output_norm.weight") {
                weights_.output_norm = tensor;
                mapped_count++;
            } else if (tensor.name == "output.weight") {
                weights_.output = tensor;
                mapped_count++;
            } else {
                // É um tensor de camada. Nome do tipo: blk.I.attn_q.weight
                if (tensor.name.rfind("blk.", 0) == 0) {
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
        }

        if (verbose) std::cout << "[Engine] Mapeamento de " << mapped_count << " tensores concluido com sucesso!" << std::endl;
        return true;
    }

    // Pipeline principal de execução (Forward Pass) do MiniCPM5
    std::vector<float> forward(int token_id, int pos, int max_seq_len) {
        // 1. TOKEN EMBEDDING LOOKUP
        std::vector<float> x(MODEL_DIM); 
        
        // Lookup (Extraindo os 2048 valores da matriz token_embd em Q4_K)
        get_embedding(x, weights_.token_embd, token_id);

        // Vetores de buffers temporários reutilizados para evitar alocações dinâmicas na inferência
        std::vector<float> x_norm(MODEL_DIM);
        std::vector<float> q(MODEL_HEADS * (MODEL_DIM / MODEL_HEADS)); // Query (2048)
        std::vector<float> k(MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS)); // Key (256)
        std::vector<float> v(MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS)); // Value (256)
        std::vector<float> attn_out_temp(MODEL_DIM);
        std::vector<float> attn_output(MODEL_DIM);
        std::vector<float> gate(MODEL_FFN_DIM);
        std::vector<float> up(MODEL_FFN_DIM);
        std::vector<float> ffn_intermediate(MODEL_FFN_DIM);
        std::vector<float> ffn_output(MODEL_DIM);

        // 2. ITERAÇÃO DE CAMADAS TRANSFORME (42 CAMADAS)
        for (int l = 0; l < MODEL_LAYERS; l++) {
            const LayerWeights& layer = weights_.layers[l];
            
            // --- BLOCO DE ATENÇÃO (GQA + RoPE) ---
            std::vector<float> residual = x; // Salva resíduo da Atenção

            // RMSNorm antes da atenção
            rmsnorm(x_norm, x, layer.attn_norm);

            // Projeções Q, K, V
            matmul(q, x_norm, layer.attn_q);
            matmul(k, x_norm, layer.attn_k);
            matmul(v, x_norm, layer.attn_v);

            // Aplica os embeddings rotacionais RoPE
            int head_dim = MODEL_DIM / MODEL_HEADS;
            for (int h = 0; h < MODEL_HEADS; h++) {
                apply_rope(q, pos, h, head_dim, MODEL_ROPE_BASE);
            }
            for (int h = 0; h < MODEL_KV_HEADS; h++) {
                apply_rope(k, pos, h, head_dim, MODEL_ROPE_BASE);
            }

            // Atenção GQA com KV Cache
            execute_attention(attn_out_temp, q, k, v, kv_caches_[l], pos, max_seq_len, MODEL_HEADS, MODEL_KV_HEADS, head_dim);

            // Projeção de saída da atenção
            matmul(attn_output, attn_out_temp, layer.attn_output);

            // Conexão Residual da Atenção
            #pragma omp parallel for
            for (int i = 0; i < MODEL_DIM; i++) {
                x[i] = residual[i] + attn_output[i];
            }

            // --- BLOCO FEED-FORWARD (SwiGLU) ---
            residual = x; // Salva resíduo da FFN

            // RMSNorm antes da FFN
            rmsnorm(x_norm, x, layer.ffn_norm);

            // Projeções Gate e Up do SwiGLU
            matmul(gate, x_norm, layer.ffn_gate);
            matmul(up, x_norm, layer.ffn_up);

            // Ativação SwiGLU: silu(gate) * up
            #pragma omp parallel for
            for (int i = 0; i < MODEL_FFN_DIM; i++) {
                ffn_intermediate[i] = silu(gate[i]) * up[i];
            }

            // Down projection
            matmul(ffn_output, ffn_intermediate, layer.ffn_down);

            // Conexão Residual da FFN
            #pragma omp parallel for
            for (int i = 0; i < MODEL_DIM; i++) {
                x[i] = residual[i] + ffn_output[i];
            }
        }

        // 3. FINAL NORMALIZATION
        rmsnorm(x_norm, x, weights_.output_norm);

        // 4. LANGUAGE MODEL HEAD (LM HEAD) PROJECTION
        std::vector<float> logits;
        matmul(logits, x_norm, weights_.output);

        return logits;
    }

private:
    std::unique_ptr<std::vector<char>> raw_gguf_buffer_; // Buffer RAM unificado contendo o GGUF
    ModelWeights weights_;
    std::vector<KVCacheLayer> kv_caches_; // Histórico de chaves/valores de cada camada
};

// ============================================================================
// 4. SAMPLER ESTOCÁSTICO (Top-p & Temperature)
// ============================================================================

int sample_token(std::vector<float>& logits, float temperature = 1.0f, float top_p = 0.95f) {
    int vocab_size = logits.size();

    // 1. Aplica Temperatura (T > 0)
    if (temperature != 1.0f && temperature > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            logits[i] /= temperature;
        }
    }

    // 2. Softmax seguro (usando o max_logit para evitar explosão de Float NaN)
    float max_logit = -1e9f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    std::vector<std::pair<float, int>> probs(vocab_size);
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float p = std::exp(logits[i] - max_logit);
        probs[i] = {p, i};
        sum_exp += p;
    }

    for (int i = 0; i < vocab_size; i++) {
        probs[i].first /= sum_exp;
    }

    // Se top_p for <= 0 ou >= 1, desabilitamos o corte.
    // Caso contrário, ordenamos para o Top-P (Nucleus Sampling)
    if (top_p > 0.0f && top_p < 1.0f) {
        // Ordena as probabilidades de forma decrescente
        std::sort(probs.begin(), probs.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

        // Corte do Top-P: acumula até bater a margem (ex: 0.95)
        float cumsum = 0.0f;
        int last_idx = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i].first;
            last_idx = i;
            if (cumsum >= top_p) {
                break;
            }
        }
        
        // Redimensiona (descarta a cauda "ruim" da distribuição probabilística)
        probs.resize(last_idx + 1);
        
        // Renormaliza para que a soma local volte a ser 1.0
        for (int i = 0; i <= last_idx; i++) {
            probs[i].first /= cumsum;
        }
    }

    // 3. Seleção Aleatória (Roleta Viciada)
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cur_sum = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        cur_sum += probs[i].first;
        if (r <= cur_sum) {
            return probs[i].second;
        }
    }
    return probs.back().second; // Fallback
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // Ensure UTF-8 text emitted by the model is rendered correctly in Windows consoles.
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::string text_prompt = "O Brasil é um país";
    bool enable_think = true;
    std::string model_path = "MiniCPM5-2B-Q4_K_M.gguf";
    int context_size = 8192;
    int max_gen_tokens = 1024;
    bool verbose = false;

    // Parse options
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./minicpm_engine [options]\n"
                      << "Options:\n"
                      << "  --text \"...\"      Input prompt text (default: \"O Brasil é um país\")\n"
                      << "  -m <path>         Path to GGUF model\n"
                      << "  -c <size>         Context size (default: 8192)\n"
                      << "  -n <count>        Max generation tokens (default: 1024)\n"
                      << "  --no-think        Disable <think> reasoning tag generation\n"
                      << "  -v, --verbose     Show detailed generation logs and progress\n"
                      << "  --help, -h        Show this help message\n";
            return 0;
        } else if (arg == "--text" && i + 1 < argc) {
            text_prompt = argv[++i];
        } else if (arg == "--no-think") {
            enable_think = false;
        } else if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            context_size = std::stoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            max_gen_tokens = std::stoi(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        }
    }

    if (verbose) {
        std::cout << "=== MiniCPM5 (2B) C++ High-Level Inference Engine ===" << std::endl;
        std::cout << "[Config] Model: " << model_path << std::endl;
        std::cout << "[Config] Prompt: " << text_prompt << std::endl;
        std::cout << "[Config] Context size: " << context_size << std::endl;
        std::cout << "[Config] Max gen tokens: " << max_gen_tokens << std::endl;
        std::cout << "[Config] Think mode: " << (enable_think ? "enabled" : "disabled") << std::endl;
        std::cout << "[Config] Architecture: " << MODEL_LAYERS << " layers, "
                  << MODEL_HEADS << " heads, " << MODEL_KV_HEADS << " KV heads, "
                  << MODEL_DIM << " dim, " << MODEL_FFN_DIM << " ffn dim" << std::endl;
    }

    // Fallback inicial para encontrar o modelo por padrao
    bool model_found = false;
    if (std::ifstream("MiniCPM5-2B-Q4_K_M.gguf")) {
        model_path = "MiniCPM5-2B-Q4_K_M.gguf";
        model_found = true;
        if (verbose) std::cout << "[Model] Found via fallback: " << model_path << std::endl;
    }
    if (!model_found && argc == 1 && verbose) {
        std::cout << "[Model] No fallback found. Check default paths or use -m <path>." << std::endl;
    }

    // Verify model file exists before proceeding
    {
        std::ifstream verify(model_path, std::ios::binary | std::ios::ate);
        if (!verify) {
            std::cerr << "[Error] Model file not found: " << model_path << std::endl;
            std::cerr << "[Hint] Use -m <path> to specify the GGUF file location." << std::endl;
            return 1;
        }
        uint64_t fsize = verify.tellg();
        if (verbose) {
            std::cout << "[Model] File verified: " << model_path
                      << " (" << std::fixed << std::setprecision(2) << (double)fsize / (1024.0*1024.0) << " MB)" << std::endl;
        }
    }

    if (verbose) std::cout << "\n[Init] Allocating KV cache (" << MODEL_LAYERS << " layers x " << context_size << " slots)..." << std::endl;
    MiniCPM5Engine engine;
    engine.init_kv_cache(context_size);

    if (verbose) std::cout << "[Load] Loading GGUF model..." << std::endl;
    if (!engine.load_model(model_path, verbose)) {
        std::cerr << "[Error] Failed to load model from: " << model_path << std::endl;
        return 1;
    }

    if (verbose) std::cout << "\n[Tokenize] Loading tokenizer from GGUF..." << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.load(model_path, verbose)) {
        std::cerr << "[Error] Failed to read vocabulary from GGUF." << std::endl;
        return 1;
    }

    std::vector<ChatMessage> chat;
    chat.push_back({"user", text_prompt});
    std::string full_prompt = apply_chat_template(chat, true, enable_think);

    if (verbose) {
        std::cout << "\n[Format] Applying chat template (think=" << (enable_think ? "yes" : "no") << ")..." << std::endl;
        std::cout << "[Prompt Raw]\n" << full_prompt << "\n" << std::endl;
        std::cout << "[Tokenize] Converting prompt to token IDs..." << std::endl;
    }

    // Converte a string final formatada em IDs usando nosso tokenizador guloso
    std::vector<int> prompt_tokens = tokenizer.tokenize(full_prompt);

    if (verbose) {
        std::cout << "[Prompt] " << prompt_tokens.size() << " tokens:" << std::endl;
        for (size_t t = 0; t < prompt_tokens.size(); t++) {
            std::cout << "  [" << t << "] id=" << prompt_tokens[t] << " => " << tokenizer.decode(prompt_tokens[t]) << std::endl;
        }
        std::cout << "\n[Inference] Starting forward pass (prompt phase)..." << std::endl;
        std::cout << "\n[Output] ";
    }

    int next_token = -1;
    auto t_start_prompt = std::chrono::high_resolution_clock::now();
    for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
        int token_id = prompt_tokens[pos];
        std::vector<float> logits = engine.forward(token_id, pos, context_size);

        // No último token do prompt, sampleamos o primeiro token gerado
        if (pos == prompt_tokens.size() - 1) {
            if (verbose) std::cout << "\n[Sample] Sampling first generated token from prompt..." << std::endl;
            next_token = sample_token(logits, 1.0f, 0.95f);
            if (verbose) std::cout << "[Sample] Selected token id=" << next_token << " => " << tokenizer.decode(next_token) << std::endl;
        }
    }
    auto t_end_prompt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prompt_time = t_end_prompt - t_start_prompt;

    std::cout << tokenizer.decode(next_token) << std::flush;

    // Geração autoregressiva estocástica
    int current_pos = prompt_tokens.size();
    int num_generated = 0;

    if (verbose) std::cout << "\n[Inference] Starting autoregressive generation (max " << max_gen_tokens << " tokens)..." << std::endl;
    auto t_start_gen = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < max_gen_tokens; step++) {
        std::vector<float> logits = engine.forward(next_token, current_pos, context_size);

        next_token = sample_token(logits, 1.0f, 0.95f);
        current_pos++;
        num_generated++;

        std::string decoded = tokenizer.decode(next_token);
        std::cout << decoded << std::flush;

        if (next_token == 1 || next_token == 2 || next_token == 130073) {
            if (verbose) std::cout << "\n[Stop] EOS token detected (id=" << next_token << "). Stopping generation." << std::endl;
            break;
        }
        if (step == max_gen_tokens - 1) {
            if (verbose) std::cout << "\n[Stop] Max tokens (" << max_gen_tokens << ") reached." << std::endl;
        }
    }
    auto t_end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_time = t_end_gen - t_start_gen;

    if (verbose) {
        std::cout << "\n\n=== Performance Summary ===" << std::endl;
        double prompt_tok_s = (prompt_time.count() > 0) ? (prompt_tokens.size() / prompt_time.count()) : 0.0;
        double gen_tok_s    = (gen_time.count() > 0) ? (num_generated / gen_time.count()) : 0.0;
        std::cout << "Prompt phase: " << prompt_tokens.size() << " tokens in "
                  << std::fixed << std::setprecision(2) << prompt_time.count() << " s"
                  << " (" << std::setprecision(1) << prompt_tok_s << " tok/s)" << std::endl;
        std::cout << "Generation:   " << num_generated << " tokens in "
                  << std::fixed << std::setprecision(2) << gen_time.count() << " s"
                  << " (" << std::setprecision(1) << gen_tok_s << " tok/s)" << std::endl;

        std::cout << "\n=== Engine Ready ===" << std::endl;
    } else {
        std::cout << std::endl;
    }
    return 0;
}
