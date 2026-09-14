#include "model.h"
#include "omp_config.h"
#include "tokenizer.h"
#include "tool.h"
#include "chat_session.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#ifdef _WIN32
#include <windows.h>
#endif

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    // Ensure that stdout is not buffered (essential for real-time token streaming)
    std::setvbuf(stdout, NULL, _IONBF, 0);
    std::cout.setf(std::ios::unitbuf);

    std::string text_prompt;
    bool interactive = false;
    std::string model_path = "MiniCPM5-2B-Q4_K_M.gguf";

    ChatConfig config;

    omp_config::initialize(8);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./minicpm5-cli [options]\n"
                      << "Options:\n"
                      << "  --text \"...\"      Input prompt text (omit for interactive mode)\n"
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
            config.enable_think = true;
        } else if (arg == "--no-think") {
            config.enable_think = false;
        } else if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "-c" && i + 1 < argc) {
            config.context_size = std::stoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            config.max_gen_tokens = std::stoi(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        }
    }

    if (text_prompt.empty()) {
        interactive = true;
    }

    if (config.verbose) {
        std::cout << "=== MiniCPM5 (2B) C++ High-Level Inference Engine ===" << std::endl;
        std::cout << "[Config] Model: " << model_path << std::endl;
        std::cout << "[Config] Mode: " << (interactive ? "interactive (REPL)" : "single-shot") << std::endl;
        if (!interactive) std::cout << "[Config] Prompt: " << text_prompt << std::endl;
        std::cout << "[Config] Context size: " << config.context_size << std::endl;
        std::cout << "[Config] Max gen tokens: " << config.max_gen_tokens << std::endl;
        std::cout << "[Config] Think mode: " << (config.enable_think ? "enabled" : "disabled") << std::endl;
    }

    // Fallback for default model path
    bool model_found = false;
    if (std::ifstream("MiniCPM5-2B-Q4_K_M.gguf")) {
        model_path = "MiniCPM5-2B-Q4_K_M.gguf";
        model_found = true;
    }
    if (!model_found && argc == 1 && config.verbose) {
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
        if (config.verbose) {
            std::cout << "[Model] File verified: " << model_path
                      << " (" << std::fixed << std::setprecision(2) << (double)fsize / (1024.0*1024.0) << " MB)" << std::endl;
        }
    }

    if (config.verbose) std::cout << "\n[Init] Allocating KV cache..." << std::endl;
    MiniCPM5Engine engine;
    engine.init_kv_cache(config.context_size);

    if (config.verbose) std::cout << "[Load] Loading GGUF model..." << std::endl;
    if (!engine.load_model(model_path, config.verbose)) {
        std::cerr << "[Error] Failed to load model from: " << model_path << std::endl;
        return 1;
    }

    if (config.verbose) std::cout << "\n[Tokenize] Loading tokenizer from GGUF..." << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.load(model_path, config.verbose)) {
        std::cerr << "[Error] Failed to read vocabulary from GGUF." << std::endl;
        return 1;
    }

    ToolRegistry registry;
    if (config.verbose) {
        std::cout << "\n[Tools] " << registry.definitions().size() << " tool(s) available (think="
                  << (config.enable_think ? "yes" : "no") << ")." << std::endl;
    }

    ChatSession session(engine, tokenizer, registry, config);
    session.set_system_prompt("You are a helpful assistant. Use the provided tools when they help answer the user.");

    // ========================================================================
    // DISPATCH: single-shot vs interactive REPL
    // ========================================================================
    if (interactive) {
        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;

            session.add_user_message(line);
            if (!session.generate_response()) break;
        }
    } else {
        session.add_user_message(text_prompt);
        session.generate_response();
    }

    return 0;
}
