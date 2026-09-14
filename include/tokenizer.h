#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>

// ============================================================================
// SEMANTIC TYPE ALIASES (Enhances code readability for complex templates)
// ============================================================================
using TokenList = std::vector<int>;
using VocabList = std::vector<std::string>;
using VocabMap = std::unordered_map<std::string, int>;
using UnicodeToByteMap = std::unordered_map<std::string, uint8_t>;

class Tokenizer {
public:
    Tokenizer();
    bool load(const std::string& gguf_path, bool verbose = false);
    std::string decode(int token_id);
    int get_id(const std::string& str) const;
    TokenList tokenize(std::string text) const;

private:
    VocabList vocab_;
    VocabMap vocab_map_;
    VocabList control_tokens_;
    std::string byte_buffer_;
    VocabList byte_to_unicode_;
    UnicodeToByteMap unicode_to_byte_;

    void init_unicode_mappings();
    void build_control_tokens_index();
    bool is_control_token_candidate(const std::string& token) const;
    bool find_control_token_at(const std::string& text, size_t pos, std::string& out_token) const;
    void tokenize_plain_text(const std::string& plain_text, TokenList& out_tokens) const;
    std::string map_unicode_to_bytes(const std::string& text) const;
    void skip_gguf_value(std::ifstream& f, uint32_t type);
};

struct ChatMessage {
    std::string role;
    std::string content;
};

std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_think = true);
