#include "chat_session.h"
#include <iostream>
#include <chrono>
#include <iomanip>

ChatSession::ChatSession(MiniCPM5Engine& engine, Tokenizer& tokenizer, ToolRegistry& registry, const ChatConfig& config)
    : engine_(engine),
      tokenizer_(tokenizer),
      registry_(registry),
      config_(config),
      sampler_(config.temperature, config.top_p),
      renderer_("<s>"),
      in_thinking_(false) {

    chat_opts_.add_generation_prompt = true;
    chat_opts_.enable_thinking = config_.enable_think;
    chat_opts_.tools_json = registry_.definitions();

    setup_special_tokens();
}

void ChatSession::setup_special_tokens() {
    auto register_stop_token = [&](const std::string& token_text) {
        int token_id = tokenizer_.get_id(token_text);
        if (token_id >= 0) stop_token_ids_.insert(token_id);
    };
    auto register_hidden_token = [&](const std::string& token_text) {
        int token_id = tokenizer_.get_id(token_text);
        if (token_id >= 0) hidden_token_ids_.insert(token_id);
    };
    auto register_think_begin_token = [&](const std::string& token_text) {
        int token_id = tokenizer_.get_id(token_text);
        if (token_id >= 0) think_begin_token_ids_.insert(token_id);
    };
    auto register_think_end_token = [&](const std::string& token_text) {
        int token_id = tokenizer_.get_id(token_text);
        if (token_id >= 0) think_end_token_ids_.insert(token_id);
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

    if (config_.verbose) {
        std::cout << "[Decode] Stop tokens:";
        if (stop_token_ids_.empty()) {
            std::cout << " (none found in vocab)";
        } else {
            for (int id : stop_token_ids_) std::cout << " " << id;
        }
        std::cout << std::endl;
    }
}

void ChatSession::set_system_prompt(const std::string& prompt) {
    chat_template::Message sys_msg;
    sys_msg.role = "system";
    sys_msg.content = prompt;
    messages_.push_back(std::move(sys_msg));
}

void ChatSession::add_user_message(const std::string& msg) {
    chat_template::Message user_msg;
    user_msg.role = "user";
    user_msg.content = msg;
    messages_.push_back(std::move(user_msg));
}

bool ChatSession::is_stop_token(int token_id) const {
    return stop_token_ids_.find(token_id) != stop_token_ids_.end();
}

void ChatSession::set_thinking(bool thinking) {
    if (thinking == in_thinking_) return;
    in_thinking_ = thinking;
    std::cout << (in_thinking_ ? "<think>\n" : "\n</think>\n") << std::flush;
}

void ChatSession::buffer_token(int token_id) {
    if (think_begin_token_ids_.find(token_id) != think_begin_token_ids_.end()) {
        set_thinking(true);
        return;
    }
    if (think_end_token_ids_.find(token_id) != think_end_token_ids_.end()) {
        set_thinking(false);
        return;
    }
    if (hidden_token_ids_.find(token_id) != hidden_token_ids_.end()) {
        return;
    }
    std::string d = tokenizer_.decode(token_id);
    if (in_thinking_) {
        think_buf_ += d;
        std::cout << d << std::flush;
    } else {
        main_buf_ += d;
        std::cout << d << std::flush;
    }
}

bool ChatSession::generate_response() {
    double prompt_time = 0.0, gen_time = 0.0;
    double p_tokens = 0.0, g_tokens = 0.0;
    int num_turns = 0;

    for (int turn = 0; turn < config_.max_tool_turns; ++turn) {
        num_turns++;

        std::string full_prompt = renderer_.render(messages_, chat_opts_);
        if (config_.verbose) {
            std::cout << "\n[Turn " << turn << "] Applying chat template..." << std::endl;
            std::cout << "[Prompt Raw]\n" << full_prompt << "\n" << std::endl;
        }

        std::vector<int> prompt_tokens = tokenizer_.tokenize(full_prompt);
        if (prompt_tokens.empty()) {
            std::cerr << "[Error] Prompt tokenization produced zero tokens." << std::endl;
            return false;
        }
        if (static_cast<int>(prompt_tokens.size()) >= config_.context_size) {
            std::cerr << "[Error] Context overflow: " << prompt_tokens.size()
                      << " prompt tokens >= context_size " << config_.context_size << std::endl;
            return false;
        }
        if (config_.verbose) {
            std::cout << "[Turn " << turn << "] " << prompt_tokens.size() << " prompt tokens." << std::endl;
        }

        // Per-turn State Reset
        think_buf_.clear();
        main_buf_.clear();
        in_thinking_ = false;

        // ---- Prompt Processing Phase ----
        auto t0 = std::chrono::high_resolution_clock::now();
        int next_token = -1;
        for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
            int token_id = prompt_tokens[pos];
            const std::vector<float>& logits = engine_.forward(token_id, static_cast<int>(pos), config_.context_size);
            if (pos == prompt_tokens.size() - 1) {
                next_token = sampler_.sample(logits);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        prompt_time += std::chrono::duration<double>(t1 - t0).count();
        p_tokens += static_cast<double>(prompt_tokens.size());

        // First generated token
        if (!is_stop_token(next_token)) {
            if (config_.enable_think) set_thinking(true);
            buffer_token(next_token);
        }

        // ---- Autoregressive Decoding Phase ----
        auto t2 = std::chrono::high_resolution_clock::now();
        int current_pos = static_cast<int>(prompt_tokens.size());
        int turn_gen = 0;
        for (int step = 0; step < config_.max_gen_tokens && !is_stop_token(next_token); step++) {
            const std::vector<float>& logits = engine_.forward(next_token, current_pos, config_.context_size);
            next_token = sampler_.sample(logits);
            current_pos++;

            if (is_stop_token(next_token)) {
                if (config_.verbose) std::cout << "[Turn " << turn << "] EOS token (id=" << next_token << ")." << std::endl;
                break;
            }
            buffer_token(next_token);
            turn_gen++;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        gen_time += std::chrono::duration<double>(t3 - t2).count();
        g_tokens += static_cast<double>(turn_gen);

        if (config_.verbose) {
            std::cout << "\n[Turn " << turn << "] Generated " << turn_gen << " tokens." << std::endl;
            if (!think_buf_.empty()) std::cout << "[Turn " << turn << "] Reasoning: " << think_buf_ << std::endl;
            std::cout << "[Turn " << turn << "] Raw output: " << main_buf_ << std::endl;
        }

        // ---- Post-Generation Decision: Tool Call or Final Response? ----
        ParsedCall call = parse_tool_call(main_buf_);
        if (call.name.empty()) {
            // Final Response generated. No tools called.
            chat_template::Message asst;
            asst.role = "assistant";
            asst.content = main_buf_;
            if (!think_buf_.empty()) asst.reasoning_content = think_buf_;
            messages_.push_back(std::move(asst));
            break;
        }

        // Tool Call detected. Execute locally and feed result back to the model.
        if (config_.verbose) {
            std::cout << "\n[TOOL CALL] " << call.name;
            for (const auto& kv : call.args) std::cout << "  " << kv.first << "=" << kv.second;
            std::cout << std::endl;
        }

        chat_template::Message asst;
        asst.role = "assistant";
        asst.content = "";
        if (!think_buf_.empty()) asst.reasoning_content = think_buf_;
        chat_template::ToolCall tc;
        tc.name = call.name;
        tc.arguments = call.args;
        asst.tool_calls.push_back(tc);
        messages_.push_back(std::move(asst));

        std::string result = registry_.execute(call.name, call.args);
        if (config_.verbose) std::cout << "[TOOL RESULT] " << result << std::endl;

        chat_template::Message tool_msg;
        tool_msg.role = "tool";
        tool_msg.content = result;
        messages_.push_back(std::move(tool_msg));
    }

    // Performance Metrics
    double prompt_tok_s = (prompt_time > 0) ? (p_tokens / prompt_time) : 0.0;
    double gen_tok_s = (gen_time > 0) ? (g_tokens / gen_time) : 0.0;

    if (config_.verbose) {
        std::cout << "\n=== Performance (" << num_turns << " turns) ===" << std::endl;
        std::cout << "Prompt: " << static_cast<size_t>(p_tokens) << " tok / "
                  << std::fixed << std::setprecision(2) << prompt_time << "s"
                  << " (" << std::setprecision(1) << prompt_tok_s << " tok/s)" << std::endl;
        std::cout << "Gen:    " << static_cast<size_t>(g_tokens) << " tok / "
                  << std::fixed << std::setprecision(2) << gen_time << "s"
                  << " (" << std::setprecision(1) << gen_tok_s << " tok/s)" << std::endl;
    } else {
        std::cout << "\n" << std::fixed << std::setprecision(1)
                  << "  " << static_cast<size_t>(g_tokens) << " tok in "
                  << std::setprecision(2) << (prompt_time + gen_time) << "s"
                  << "  (" << num_turns << " turn(s))" << std::endl;
    }

    return true;
}
