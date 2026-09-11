#include "tokenizer.h"
#include "minicpm_special_tokens.h"
#include <iostream>
#include <algorithm>

Tokenizer::Tokenizer() {
    init_unicode_mappings();
}

bool Tokenizer::load(const std::string& gguf_path, bool verbose) {
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
            f.read((char*)&arr_type, 4);
            uint64_t arr_len;
            f.read((char*)&arr_len, 8);

            vocab_.resize(arr_len);
            vocab_map_.reserve(arr_len);
            for (uint64_t j = 0; j < arr_len; j++) {
                uint64_t str_len;
                f.read((char*)&str_len, 8);
                std::string token(str_len, '\0');
                f.read(&token[0], str_len);
                vocab_[j] = token;
                vocab_map_[vocab_[j]] = j;
            }
            build_control_tokens_index();
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

std::string Tokenizer::decode(int token_id) {
    if (token_id < 0 || token_id >= (int)vocab_.size()) return "<UNK>";

    std::string token = vocab_[token_id];

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

        if (i + bytes_needed > byte_buffer_.size()) break;
        valid_output += byte_buffer_.substr(i, bytes_needed);
        i += bytes_needed;
    }
    byte_buffer_.erase(0, i);
    return valid_output;
}

int Tokenizer::get_id(const std::string& str) const {
    auto it = vocab_map_.find(str);
    return (it != vocab_map_.end()) ? it->second : -1;
}

std::vector<int> Tokenizer::tokenize(std::string text) const {
    std::vector<int> tokens;
    size_t start_idx = 0;
    if (!text.empty() && text[0] == '\x01') {
        tokens.push_back(0);
        start_idx = 1;
    }

    size_t i = start_idx;
    std::string plain_segment;
    while (i < text.size()) {
        std::string matched_control_token;
        if (find_control_token_at(text, i, matched_control_token)) {
            auto control_it = vocab_map_.find(matched_control_token);
            if (control_it != vocab_map_.end()) {
                if (!plain_segment.empty()) {
                    tokenize_plain_text(plain_segment, tokens);
                    plain_segment.clear();
                }
                tokens.push_back(control_it->second);
                i += matched_control_token.size();
                continue;
            }
        }
        plain_segment.push_back(text[i]);
        i++;
    }

    if (!plain_segment.empty()) {
        tokenize_plain_text(plain_segment, tokens);
    }
    return tokens;
}

// ============================================================================
// PRIVATE HELPERS
// ============================================================================

void Tokenizer::init_unicode_mappings() {
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

void Tokenizer::build_control_tokens_index() {
    control_tokens_.clear();
    control_tokens_.reserve(64);
    for (const auto& token : vocab_) {
        if (is_control_token_candidate(token)) {
            control_tokens_.push_back(token);
        }
    }
    std::sort(control_tokens_.begin(), control_tokens_.end(), [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a < b;
    });
    control_tokens_.erase(std::unique(control_tokens_.begin(), control_tokens_.end()), control_tokens_.end());
}

bool Tokenizer::is_control_token_candidate(const std::string& token) const {
    if (token.size() >= 4 && token.rfind("<|", 0) == 0 && token.compare(token.size() - 2, 2, "|>") == 0) return true;
    if (token.size() >= 3 && token.front() == '<' && token.back() == '>') return true;
    if (is_minicpm5_special_token(token)) return true;
    return false;
}

bool Tokenizer::find_control_token_at(const std::string& text, size_t pos, std::string& out_token) const {
    for (const auto& token : control_tokens_) {
        if (token.empty()) continue;
        if (pos + token.size() > text.size()) continue;
        if (text.compare(pos, token.size(), token) == 0) {
            out_token = token;
            return true;
        }
    }
    return false;
}

void Tokenizer::tokenize_plain_text(const std::string& plain_text, std::vector<int>& out_tokens) const {
    std::string mapped_text = "";
    for (size_t i = 0; i < plain_text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(plain_text[i]);
        mapped_text += byte_to_unicode_[c];
    }

    size_t i = 0;
    while (i < mapped_text.size()) {
        int best_id = -1;
        size_t best_len = 0;
        for (size_t len = mapped_text.size() - i; len > 0; len--) {
            std::string sub = mapped_text.substr(i, len);
            auto it = vocab_map_.find(sub);
            if (it != vocab_map_.end()) {
                best_id = it->second;
                best_len = len;
                break;
            }
        }
        if (best_id != -1) {
            out_tokens.push_back(best_id);
            i += best_len;
        } else {
            i++;
        }
    }
}

std::string Tokenizer::map_unicode_to_bytes(const std::string& text) const {
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

void Tokenizer::skip_gguf_value(std::ifstream& f, uint32_t type) {
    uint64_t dummy64; uint32_t dummy32;
    switch (type) {
        case 0: case 1: case 7: f.seekg(1, std::ios::cur); break;
        case 2: case 3: f.seekg(2, std::ios::cur); break;
        case 4: case 5: case 6: f.seekg(4, std::ios::cur); break;
        case 10: case 11: case 12: f.seekg(8, std::ios::cur); break;
        case 8:
            f.read((char*)&dummy64, 8);
            f.seekg(dummy64, std::ios::cur);
            break;
        case 9:
            f.read((char*)&dummy32, 4);
            f.read((char*)&dummy64, 8);
            for (uint64_t k = 0; k < dummy64; k++) skip_gguf_value(f, dummy32);
            break;
    }
}

// ============================================================================
// CHAT TEMPLATE
// ============================================================================
// CHAT TEMPLATE
// ============================================================================

std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_think) {
    std::string prompt = "<s>";
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            prompt += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "user") {
            prompt += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "assistant") {
            if (enable_think) {
                prompt += "<|im_start|>assistant\n<|thought_begin|>\n" + msg.content + "<|im_end|>\n";
            } else {
                prompt += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
    }
    if (add_generation_prompt) {
        if (enable_think) {
            prompt += "<|im_start|>assistant\n<|thought_begin|>";
        } else {
            prompt += "<|im_start|>assistant\n";
        }
    }
    return prompt;
}
