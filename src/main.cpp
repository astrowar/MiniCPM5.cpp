#include "model.h"
#include "omp_config.h"
#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// SAMPLER ESTOCÁSTICO (Top-p & Temperature)
// ============================================================================

int sample_token(const std::vector<float>& logits, float temperature = 1.0f, float top_p = 0.95f) {
    int vocab_size = logits.size();

    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    bool apply_temp = (temperature != 1.0f && temperature > 0.0f);

    float max_logit = -1e9f;
    if (apply_temp) {
        for (int i = 0; i < vocab_size; i++) {
            float val = logits[i] * inv_temp;
            if (val > max_logit) max_logit = val;
        }
    } else {
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
    }

    std::vector<std::pair<float, int>> probs(vocab_size);
    float sum_exp = 0.0f;
    if (apply_temp) {
        for (int i = 0; i < vocab_size; i++) {
            float p = std::exp((logits[i] * inv_temp) - max_logit);
            probs[i] = {p, i};
            sum_exp += p;
        }
    } else {
        for (int i = 0; i < vocab_size; i++) {
            float p = std::exp(logits[i] - max_logit);
            probs[i] = {p, i};
            sum_exp += p;
        }
    }

    for (int i = 0; i < vocab_size; i++) {
        probs[i].first /= sum_exp;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        std::sort(probs.begin(), probs.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

        float cumsum = 0.0f;
        int last_idx = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i].first;
            last_idx = i;
            if (cumsum >= top_p) break;
        }
        probs.resize(last_idx + 1);
        for (int i = 0; i <= last_idx; i++) {
            probs[i].first /= cumsum;
        }
    }

    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cur_sum = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        cur_sum += probs[i].first;
        if (r <= cur_sum) return probs[i].second;
    }
    return probs.back().second;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::string text_prompt = "O Brasil é um país";
    bool enable_think = false;
    std::string model_path = "MiniCPM5-2B-Q4_K_M.gguf";
    int context_size = 8192;
    int max_gen_tokens = 1024;
    bool verbose = false;
    float temperature = 1.0f; // temperatura de sampling (facil de ajustar aqui)

    omp_config::initialize(8);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./minicpm_engine [options]\n"
                      << "Options:\n"
                      << "  --text \"...\"      Input prompt text (default: \"O Brasil é um país\")\n"
                      << "  -m <path>         Path to GGUF model\n"
                      << "  -c <size>         Context size (default: 8192)\n"
                      << "  -n <count>        Max generation tokens (default: 1024)\n"
                      << "  --think           Enable think tag generation\n"
                      << "  --no-think        Disable think tag generation (default)\n"
                      << "  -v, --verbose     Show detailed generation logs\n"
                      << "  --help, -h        Show this help message\n";
            return 0;
        } else if (arg == "--text" && i + 1 < argc) {
            text_prompt = argv[++i];
        } else if (arg == "--think") {
            enable_think = true;
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
    }

    // Fallback for default model path
    bool model_found = false;
    if (std::ifstream("MiniCPM5-2B-Q4_K_M.gguf")) {
        model_path = "MiniCPM5-2B-Q4_K_M.gguf";
        model_found = true;
    }
    if (!model_found && argc == 1 && verbose) {
        std::cout << "[Model] No fallback found. Use -m <path>." << std::endl;
    }

    // Verify model file exists
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

    if (verbose) std::cout << "\n[Init] Allocating KV cache..." << std::endl;
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

    std::vector<int> prompt_tokens = tokenizer.tokenize(full_prompt);
    if (prompt_tokens.empty()) {
        std::cerr << "[Error] Prompt tokenization produced zero tokens." << std::endl;
        return 1;
    }

    if (verbose) {
        std::cout << "[Prompt] " << prompt_tokens.size() << " tokens." << std::endl;
        std::cout << "\n[Output] ";
    }

    std::unordered_set<int> stop_token_ids;
    std::unordered_set<int> hidden_token_ids;
    std::unordered_set<int> think_begin_token_ids;
    std::unordered_set<int> think_end_token_ids;
    auto register_hidden_token = [&](const std::string& token_text) {
        int token_id = tokenizer.get_id(token_text);
        if (token_id >= 0) hidden_token_ids.insert(token_id);
    };
    auto register_think_begin_token = [&](const std::string& token_text) {
        int token_id = tokenizer.get_id(token_text);
        if (token_id >= 0) think_begin_token_ids.insert(token_id);
    };
    auto register_think_end_token = [&](const std::string& token_text) {
        int token_id = tokenizer.get_id(token_text);
        if (token_id >= 0) think_end_token_ids.insert(token_id);
    };
    auto register_stop_token = [&](const std::string& token_text) {
        int token_id = tokenizer.get_id(token_text);
        if (token_id >= 0) stop_token_ids.insert(token_id);
    };

    register_stop_token("</s>");
    register_stop_token("<|im_end|>");

    register_hidden_token("<s>");
    register_hidden_token("<|im_start|>");
    register_hidden_token("<|im_sep|>");
    register_hidden_token("<|thought_begin|>");
    register_hidden_token("<|thought_end|>");
    register_hidden_token("<think>");
    register_hidden_token("</think>");
    register_hidden_token("/think");
    register_hidden_token("/no_think");

    register_think_begin_token("<|thought_begin|>");
    register_think_begin_token("<think>");
    register_think_begin_token("/think");
    register_think_end_token("<|thought_end|>");
    register_think_end_token("</think>");
    register_think_end_token("/no_think");

    auto is_stop_token = [&](int token_id) {
        return stop_token_ids.find(token_id) != stop_token_ids.end();
    };
    bool in_thinking = false;
    auto emit_token = [&](int token_id) {
        if (think_begin_token_ids.find(token_id) != think_begin_token_ids.end()) {
            if (!in_thinking) {
                in_thinking = true;
                std::cout << "\n[THINKING_BEGIN]\n" << std::flush;
            }
            return false;
        }
        if (think_end_token_ids.find(token_id) != think_end_token_ids.end()) {
            if (in_thinking) {
                in_thinking = false;
                std::cout << "\n[THINKING_END]\n" << std::flush;
            }
            return false;
        }
        if (hidden_token_ids.find(token_id) != hidden_token_ids.end()) {
            return false;
        }
        std::cout << tokenizer.decode(token_id) << std::flush;
        return true;
    };

    if (verbose) {
        std::cout << "[Decode] Stop tokens:";
        if (stop_token_ids.empty()) {
            std::cout << " (none found in vocab)";
        } else {
            for (int id : stop_token_ids) std::cout << " " << id;
        }
        std::cout << std::endl;
    }

    int next_token = -1;

    // Prompt phase
    auto t_start_prompt = std::chrono::high_resolution_clock::now();
    for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
        int token_id = prompt_tokens[pos];
        const std::vector<float>& logits = engine.forward(token_id, pos, context_size);
        if (pos == prompt_tokens.size() - 1) {
            next_token = sample_token(logits, temperature, 0.95f);
        }
    }
    auto t_end_prompt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prompt_time = t_end_prompt - t_start_prompt;

    bool stop_after_prompt = is_stop_token(next_token);
    if (!stop_after_prompt) {
        emit_token(next_token);
    }

    // Autoregressive generation
    int current_pos = prompt_tokens.size();
    int num_generated = 0;

    auto t_start_gen = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < max_gen_tokens && !stop_after_prompt; step++) {
        const std::vector<float>& logits = engine.forward(next_token, current_pos, context_size);
        next_token = sample_token(logits, temperature, 0.95f);
        current_pos++;

        if (is_stop_token(next_token)) {
            if (verbose) std::cout << "\n[Stop] EOS token detected (id=" << next_token << ")." << std::endl;
            break;
        }

        if (emit_token(next_token)) {
            num_generated++;
        }
    }
    auto t_end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_time = t_end_gen - t_start_gen;

    double prompt_tok_s = (prompt_time.count() > 0) ? (prompt_tokens.size() / prompt_time.count()) : 0.0;
    double gen_tok_s = (gen_time.count() > 0) ? (num_generated / gen_time.count()) : 0.0;

    if (verbose) {
        std::cout << "\n\n=== Performance Summary ===" << std::endl;
        std::cout << "Prompt phase: " << prompt_tokens.size() << " tokens in "
                  << std::fixed << std::setprecision(2) << prompt_time.count() << " s"
                  << " (" << std::setprecision(1) << prompt_tok_s << " tok/s)" << std::endl;
        std::cout << "Generation:   " << num_generated << " tokens in "
                  << std::fixed << std::setprecision(2) << gen_time.count() << " s"
                  << " (" << std::setprecision(1) << gen_tok_s << " tok/s)" << std::endl;
    } else {
        std::cout << "\n" << std::fixed << std::setprecision(1)
                  << "  " << num_generated << " tokens em "
                  << std::setprecision(2) << gen_time.count() << "s"
                  << "  →  " << gen_tok_s << " tok/s" << std::endl;
    }

    return 0;
}
