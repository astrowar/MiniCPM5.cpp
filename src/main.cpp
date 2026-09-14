#include "model.h"
#include "omp_config.h"
#include "tokenizer.h"
#include "chat_template.h"
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
#include <ctime>
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
// TOOL CALLING: parser do XML emitido pelo modelo + executor nativo
// ============================================================================

struct ParsedCall {
    std::string name;
    std::vector<std::pair<std::string, std::string>> args;
};

// Remove wrapper CDATA, se presente, do valor de um <param>.
static std::string strip_cdata(const std::string& v) {
    const std::string open = "<![CDATA[";
    const std::string close = "]]>";
    size_t a = v.find(open);
    if (a != std::string::npos) {
        size_t b = v.find(close, a);
        if (b != std::string::npos)
            return v.substr(a + open.size(), b - a - open.size());
    }
    return v;
}

// Localiza o primeiro <function name="..."> e extrai name + pares <param>.
// Retorna name vazio se nenhum <function> for encontrado.
static ParsedCall parse_tool_call(const std::string& text) {
    ParsedCall r;
    size_t f = text.find("<function");
    if (f == std::string::npos) return r;

    size_t nm = text.find("name=\"", f);
    if (nm == std::string::npos) return r;
    nm += 6; // len("name=\"") = 6; o valor comeca no char apos aspas
    size_t nm_end = text.find('"', nm);
    if (nm_end == std::string::npos) return r;
    r.name = text.substr(nm, nm_end - nm);

    size_t pos = nm_end;
    while (true) {
        size_t p = text.find("<param", pos);
        if (p == std::string::npos) break;
        size_t pk = text.find("name=\"", p);
        if (pk == std::string::npos) break;
        pk += 6; // len("name=\"") = 6
        size_t pk_end = text.find('"', pk);
        if (pk_end == std::string::npos) break;
        std::string key = text.substr(pk, pk_end - pk);
        size_t vs = text.find('>', pk_end);
        if (vs == std::string::npos) break;
        vs += 1;
        size_t ve = text.find("</param>", vs);
        if (ve == std::string::npos) break;
        std::string val = text.substr(vs, ve - vs);
        r.args.emplace_back(key, strip_cdata(val));
        pos = ve + 8; // len("</param>")
    }
    return r;
}

// Executor nativo da tool get_datetime: devolve a data/hora local atual.
static std::string exec_get_datetime() {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof buf, "%A, %Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(buf);
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

    // ---- Conversa inicial: system (com tools) + user ----
    std::vector<chat_template::Message> messages;

    chat_template::Message sys_msg;
    sys_msg.role = "system";
    sys_msg.content = "You are a helpful assistant. Use the provided tools when they help answer the user.";
    messages.push_back(std::move(sys_msg));

    chat_template::Message user_msg;
    user_msg.role = "user";
    user_msg.content = text_prompt;
    messages.push_back(std::move(user_msg));

    chat_template::Options opts;
    opts.add_generation_prompt = true;
    opts.enable_thinking = enable_think;
    // Definição da tool get_datetime — o renderer injeta isso no system prompt.
    opts.tools_json = {
        "{\"name\": \"get_datetime\", \"description\": \"Get the current date and time.\", "
        "\"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}"
    };
    chat_template::Renderer renderer("<s>");

    if (verbose) {
        std::cout << "\n[Tools] get_datetime available (think=" << (enable_think ? "yes" : "no") << ")." << std::endl;
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

    // Buffers por turno: conteudo de raciocinio (think) e conteudo principal.
    // A resposta nao e streamizada token a token; e bufferizada e impressa no
    // fim do turno, porque a deteccao de tool call precisa do texto completo.
    std::string think_buf, main_buf;
    bool in_thinking = false;
    auto buffer_token = [&](int token_id) {
        if (think_begin_token_ids.find(token_id) != think_begin_token_ids.end()) {
            in_thinking = true;
            return;
        }
        if (think_end_token_ids.find(token_id) != think_end_token_ids.end()) {
            in_thinking = false;
            return;
        }
        if (hidden_token_ids.find(token_id) != hidden_token_ids.end()) {
            return;
        }
        std::string d = tokenizer.decode(token_id);
        if (in_thinking) think_buf += d;
        else main_buf += d;
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

    // ========================================================================
    // LOOP AGENTICO
    // A cada turno: renderiza a conversa (crescente), decodifica bufferizando,
    // e detecta <function> na saida. Se houver tool call: executa em C++,
    // anexa o resultado como mensagem "tool" e repete. Sem tool call: imprime
    // a resposta final e encerra.
    //
    // Re-encodar o prompt do zero a cada turno e seguro: o KV cache e
    // posicional (offset = pos * kv_len) e a atencao le [0, pos]; cada
    // posicao e reescrita no momento em que vira o pos atual, e a saida do
    // turno anterior volta a fazer parte do prompt seguinte.
    // ========================================================================
    constexpr int MAX_TOOL_TURNS = 4;
    double total_prompt_time = 0.0, total_gen_time = 0.0;
    double total_prompt_tokens = 0.0, total_gen_tokens = 0.0;
    int num_turns = 0;

    for (int turn = 0; turn < MAX_TOOL_TURNS; ++turn) {
        num_turns++;

        std::string full_prompt = renderer.render(messages, opts);
        if (verbose) {
            std::cout << "\n[Turn " << turn << "] Applying chat template..." << std::endl;
            std::cout << "[Prompt Raw]\n" << full_prompt << "\n" << std::endl;
        }

        std::vector<int> prompt_tokens = tokenizer.tokenize(full_prompt);
        if (prompt_tokens.empty()) {
            std::cerr << "[Error] Prompt tokenization produced zero tokens." << std::endl;
            return 1;
        }
        if ((int)prompt_tokens.size() >= context_size) {
            std::cerr << "[Error] Context overflow: " << prompt_tokens.size()
                      << " prompt tokens >= context_size " << context_size << std::endl;
            break;
        }
        if (verbose) {
            std::cout << "[Turn " << turn << "] " << prompt_tokens.size() << " prompt tokens." << std::endl;
        }

        // Estado por turno
        think_buf.clear();
        main_buf.clear();
        in_thinking = false;

        // ---- Prompt phase ----
        auto t0 = std::chrono::high_resolution_clock::now();
        int next_token = -1;
        for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
            int token_id = prompt_tokens[pos];
            const std::vector<float>& logits = engine.forward(token_id, (int)pos, context_size);
            if (pos == prompt_tokens.size() - 1) {
                next_token = sample_token(logits, temperature, 0.95f);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        total_prompt_time += std::chrono::duration<double>(t1 - t0).count();
        total_prompt_tokens += (double)prompt_tokens.size();

        // Primeiro token gerado (previsto pelo ultimo token do prompt).
        if (!is_stop_token(next_token)) {
            // O template ja colocou /think no prompt, entao o modelo comeca a
            // gerar o raciocinio direto; pre-entra no modo think para o buffer
            // rotear para o lugar certo.
            if (enable_think && !in_thinking) in_thinking = true;
            buffer_token(next_token);
        }

        // ---- Decode autoregressivo (bufferizado) ----
        auto t2 = std::chrono::high_resolution_clock::now();
        int current_pos = (int)prompt_tokens.size();
        int turn_gen = 0;
        for (int step = 0; step < max_gen_tokens && !is_stop_token(next_token); step++) {
            const std::vector<float>& logits = engine.forward(next_token, current_pos, context_size);
            next_token = sample_token(logits, temperature, 0.95f);
            current_pos++;

            if (is_stop_token(next_token)) {
                if (verbose) std::cout << "[Turn " << turn << "] EOS token (id=" << next_token << ")." << std::endl;
                break;
            }
            buffer_token(next_token);
            turn_gen++;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        total_gen_time += std::chrono::duration<double>(t3 - t2).count();
        total_gen_tokens += (double)turn_gen;

        if (verbose) {
            std::cout << "[Turn " << turn << "] Generated " << turn_gen << " tokens." << std::endl;
            if (!think_buf.empty()) std::cout << "[Turn " << turn << "] Reasoning: " << think_buf << std::endl;
            std::cout << "[Turn " << turn << "] Raw output: " << main_buf << std::endl;
        }

        // ---- Decisao: tool call ou resposta final? ----
        ParsedCall call = parse_tool_call(main_buf);
        if (call.name.empty()) {
            // Resposta final: imprime e encerra.
            if (verbose && !think_buf.empty()) std::cout << "\n[Final] " << std::endl;
            std::cout << main_buf << std::flush;
            break;
        }

        // Tool call: registra o turno do assistant e executa a tool em C++.
        if (verbose) {
            std::cout << "\n[TOOL CALL] " << call.name;
            for (const auto& kv : call.args) std::cout << "  " << kv.first << "=" << kv.second;
            std::cout << std::endl;
        }

        chat_template::Message asst;
        asst.role = "assistant";
        asst.content = "";  // o XML <function> e reconstruido a partir de tool_calls
        if (!think_buf.empty()) asst.reasoning_content = think_buf;
        chat_template::ToolCall tc;
        tc.name = call.name;
        tc.arguments = call.args;
        asst.tool_calls.push_back(tc);
        messages.push_back(std::move(asst));

        std::string result;
        if (call.name == "get_datetime") result = exec_get_datetime();
        else result = "{\"error\": \"unknown tool: " + call.name + "\"}";
        if (verbose) std::cout << "[TOOL RESULT] " << result << std::endl;

        chat_template::Message tool_msg;
        tool_msg.role = "tool";
        tool_msg.content = result;
        messages.push_back(std::move(tool_msg));
        // -> proxima iteracao re-renderiza com o resultado da tool no contexto
    }

    if (num_turns >= MAX_TOOL_TURNS) {
        std::cerr << "\n[Warn] Max tool turns (" << MAX_TOOL_TURNS << ") reached." << std::endl;
    }

    // ---- Summary de desempenho (soma dos turnos) ----
    double prompt_tok_s = (total_prompt_time > 0) ? (total_prompt_tokens / total_prompt_time) : 0.0;
    double gen_tok_s = (total_gen_time > 0) ? (total_gen_tokens / total_gen_time) : 0.0;

    if (verbose) {
        std::cout << "\n\n=== Performance Summary (" << num_turns << " turnos) ===" << std::endl;
        std::cout << "Prompt phase: " << (size_t)total_prompt_tokens << " tokens in "
                  << std::fixed << std::setprecision(2) << total_prompt_time << " s"
                  << " (" << std::setprecision(1) << prompt_tok_s << " tok/s)" << std::endl;
        std::cout << "Generation:   " << (size_t)total_gen_tokens << " tokens in "
                  << std::fixed << std::setprecision(2) << total_gen_time << " s"
                  << " (" << std::setprecision(1) << gen_tok_s << " tok/s)" << std::endl;
    } else {
        std::cout << "\n" << std::fixed << std::setprecision(1)
                  << "  " << (size_t)total_gen_tokens << " tokens em "
                  << std::setprecision(2) << (total_prompt_time + total_gen_time) << "s"
                  << "  (" << num_turns << " turno(s))" << std::endl;
    }

    return 0;
}
