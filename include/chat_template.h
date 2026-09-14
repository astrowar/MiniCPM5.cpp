#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace chat_template {

struct ToolCall {
    std::string name;
    std::vector<std::pair<std::string, std::string>> arguments;
};

struct Message {
    std::string role;
    std::string content;
    // When set, used directly as reasoning content.
    // When not set, the renderer extracts reasoning by splitting
    // content on the think open/close tags.
    std::optional<std::string> reasoning_content;
    std::vector<ToolCall> tool_calls;
};

struct Options {
    bool add_generation_prompt = false;
    // nullopt  = "enable_thinking not defined" in Jinja
    // false    = "enable_thinking is false"  -> empty  block
    // true     = "enable_thinking is true"   -> open  block
    std::optional<bool> enable_thinking;
    // Pre-serialized JSON strings (one per tool). When non-empty, a tools
    // system-block is prepended.
    std::vector<std::string> tools_json;
};

class Renderer {
public:
    explicit Renderer(std::string bos_token = "<s>");

    std::string render(const std::vector<Message>& messages, const Options& opts) const;

private:
    std::string bos_token_;

    std::string build_tool_definitions(const std::vector<std::string>& tools_json) const;
    std::string render_prologue(const std::vector<Message>& messages,
                                const Options& opts) const;
    std::string render_user_or_system(const Message& msg) const;
    std::string render_assistant(const Message& msg) const;
    std::string render_tool_block(const std::vector<Message>& messages,
                                  size_t first_idx, size_t last_idx) const;
    std::string render_generation_prompt(const Options& opts) const;

    static std::string lstrip_newlines(const std::string& s);
    static std::string rstrip_newlines(const std::string& s);
    static std::string strip_newlines_both(const std::string& s);
    static bool contains(const std::string& haystack, const std::string& needle);
    static std::string to_cdata_if_needed(const std::string& value);
};

} // namespace chat_template
