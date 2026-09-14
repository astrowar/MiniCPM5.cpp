#pragma once

#include <vector>
#include <string>
#include <unordered_set>
#include "model.h"
#include "tokenizer.h"
#include "tool.h"
#include "chat_template.h"
#include "sampler.h"

// ============================================================================
// CHAT SESSION & AGENTIC LOOP
//
// Encapsulates the multi-turn conversational state, including history tracking,
// tokenization, streaming generation, thinking blocks, and dynamic tool invocation.
// ============================================================================

struct ChatConfig {
    int context_size = 8192;
    int max_gen_tokens = 1024;
    bool enable_think = false;
    bool verbose = false;
    float temperature = 1.0f;
    float top_p = 0.95f;
    int max_tool_turns = 4;
};

class ChatSession {
public:
    ChatSession(MiniCPM5Engine& engine, Tokenizer& tokenizer, ToolRegistry& registry, const ChatConfig& config);
    
    // Sets the system prompt to instruct the model's persona
    void set_system_prompt(const std::string& prompt);

    // Appends a user message to the active conversation history
    void add_user_message(const std::string& msg);
    
    // Executes the generation loop. Returns false if a fatal error occurs.
    bool generate_response();

private:
    void setup_special_tokens();
    bool is_stop_token(int token_id) const;
    void set_thinking(bool thinking);
    void buffer_token(int token_id);

    MiniCPM5Engine& engine_;
    Tokenizer& tokenizer_;
    ToolRegistry& registry_;
    ChatConfig config_;
    Sampler sampler_;

    chat_template::Renderer renderer_;
    chat_template::Options chat_opts_;
    std::vector<chat_template::Message> messages_;

    std::unordered_set<int> stop_token_ids_;
    std::unordered_set<int> hidden_token_ids_;
    std::unordered_set<int> think_begin_token_ids_;
    std::unordered_set<int> think_end_token_ids_;

    std::string think_buf_;
    std::string main_buf_;
    bool in_thinking_;
};
