#pragma once

#include <array>
#include <string_view>

// GENERATED FILE - run scripts/generate_special_tokens_header.py
inline constexpr std::array<std::string_view, 32> MINI_CPM5_SPECIAL_TOKENS = {
    "<s>",
    "</s>",
    "<tool_call>",
    "</tool_call>",
    "<|im_sep|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<think>",
    "</think>",
    "<tool_response>",
    "</tool_response>",
    "<tools>",
    "</tools>",
    "<arguments>",
    "</arguments>",
    "<parameters>",
    "</parameters>",
    "<function",
    "</function>",
    "<param",
    "</param>",
    "<|im_start|>",
    "<|im_end|>",
    "<unk>",
    "<|thought_begin|>",
    "<|thought_end|>",
    "<|tool_call|>",
    "<|execute_start|>",
    "<|execute_end|>",
    "/think",
    "/no_think",
};

inline bool is_minicpm5_special_token(std::string_view token) {
    for (const auto& known : MINI_CPM5_SPECIAL_TOKENS) {
        if (token == known) return true;
    }
    return false;
}
