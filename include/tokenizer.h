#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>

class Tokenizer {
public:
    Tokenizer();
    bool load(const std::string& gguf_path, bool verbose = false);
    std::string decode(int token_id);
    int get_id(const std::string& str) const;
    std::vector<int> tokenize(std::string text) const;

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> vocab_map_;
    std::vector<std::string> control_tokens_;
    std::string byte_buffer_;
    std::vector<std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    void init_unicode_mappings();
    void build_control_tokens_index();
    bool is_control_token_candidate(const std::string& token) const;
    bool find_control_token_at(const std::string& text, size_t pos, std::string& out_token) const;
    void tokenize_plain_text(const std::string& plain_text, std::vector<int>& out_tokens) const;
    std::string map_unicode_to_bytes(const std::string& text) const;
    void skip_gguf_value(std::ifstream& f, uint32_t type);
};

struct ChatMessage {
    std::string role;
    std::string content;
};

std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_think = true);
