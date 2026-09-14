#include "chat_template.h"
#include <algorithm>

namespace chat_template {

static std::string lstrip_n(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && s[i] == '\n') ++i;
    return s.substr(i);
}
static std::string rstrip_n(const std::string& s) {
    size_t i = s.size();
    while (i > 0 && s[i - 1] == '\n') --i;
    return s.substr(0, i);
}
static std::string strip_n(const std::string& s) {
    return lstrip_n(rstrip_n(s));
}
static bool has_substr(const std::string& h, const std::string& n) {
    return h.find(n) != std::string::npos;
}

static const std::string kImStart = "<|im_start|>";
static const std::string kImEnd = "<|im_end|>";
static const std::string kThinkOpen = "<think>";
static const std::string kThinkClose = "</think>";
static const std::string kToolRespOpen = "<tool_response>";
static const std::string kToolRespClose = "</tool_response>";
static const std::string kToolDefSep = "<tool_def_sep>";
static const std::string kToolSep = "<tool_sep>";

std::string Renderer::lstrip_newlines(const std::string& s) { return lstrip_n(s); }
std::string Renderer::rstrip_newlines(const std::string& s) { return rstrip_n(s); }
std::string Renderer::strip_newlines_both(const std::string& s) { return strip_n(s); }
bool Renderer::contains(const std::string& h, const std::string& n) { return has_substr(h, n); }

std::string Renderer::to_cdata_if_needed(const std::string& v) {
    if (v.find('<') != std::string::npos || v.find('&') != std::string::npos || v.find('\n') != std::string::npos)
        return "<![CDATA[" + v + "]]>";
    return v;
}

Renderer::Renderer(std::string bos) : bos_token_(std::move(bos)) {}

std::string Renderer::build_tool_definitions(
        const std::vector<std::string>& tools_json) const {
    std::string d = "# Tools\n\nYou are provided with "
                    "function signatures within <tools></tools> "
                    "XML tags:\n<tools>";
    for (const auto& j : tools_json) d += "\n" + j;
    d += "\n</tools>\n\nTool usage guidelines:\n"
"- You may call zero or more functions. If no function calls are needed, "
         "just answer normally and do not include any <function ... </function>.\n"
"- When calling a function, return an XML object within "
         "<function ... </function> using:\n"
"<function name=\"function-name\"><param name=\"param-name\">param-value</param></function>\n"
"- param-value may be multi-line. If it contains <, & or newline characters, "
         "wrap it in a CDATA block: <param name=\"param-name\"><![CDATA[...multi-line value...]]></param>";
    return d;
}

std::string Renderer::render_prologue(
        const std::vector<Message>& messages,
        const Options& opts) const {
    std::string out = bos_token_;
    if (!opts.tools_json.empty()) {
        const std::string defs = build_tool_definitions(opts.tools_json);
        out += kImStart;
        out += "system\n";
        if (!messages.empty() && messages[0].role == "system") {
            const auto& c = messages[0].content;
            if (has_substr(c, kToolDefSep)) {
                size_t p = c.find(kToolDefSep);
                out += c.substr(0, p) + defs;
                out += c.substr(p + 14);
            } else {
                out += c + "\n\n" + defs;
            }
        } else {
            out += lstrip_n(defs);
        }
        out += kImEnd + "\n";
    } else if (!messages.empty() && messages[0].role == "system") {
        out += kImStart;
        out += "system\n";
        out += messages[0].content;
        out += kImEnd + "\n";
    }
    return out;
}

std::string Renderer::render_user_or_system(const Message& msg) const {
    std::string out = kImStart;
    out += msg.role;
    out += "\n";
    out += msg.content;
    out += kImEnd;
    out += "\n";
    return out;
}

static std::string cdata_val(const std::string& v) {
    if (v.find('<') != std::string::npos || v.find('&') != std::string::npos || v.find('\n') != std::string::npos)
        return "<![CDATA[" + v + "]]>";
    return v;
}

static std::string tool_xml(const ToolCall& tc) {
    std::string x = "<function name=\"";
    x += tc.name;
    x += "\">";
    for (const auto& [k, v] : tc.arguments) {
        x += "<param name=\"";
        x += k;
        x += "\">";
        x += cdata_val(v);
        x += "</param>";
    }
    x += "</function>";
    return x;
}

std::string Renderer::render_assistant(const Message& msg) const {
    std::string content = msg.content;
    std::string reasoning;

    if (msg.reasoning_content.has_value()) {
        reasoning = *msg.reasoning_content;
    } else if (has_substr(content, kThinkClose)) {
        size_t first_close = content.find(kThinkClose);
        std::string before = rstrip_n(content.substr(0, first_close));
        size_t last_open = before.rfind(kThinkOpen);
        if (last_open != std::string::npos)
            reasoning = lstrip_n(before.substr(last_open + 7));
        else
            reasoning = lstrip_n(before);
        size_t last_close = content.rfind(kThinkClose);
        content = lstrip_n(content.substr(last_close + 8));
    }

    if (!msg.tool_calls.empty()) {
        std::vector<std::string> parts;
        size_t pos = 0;
        while (true) {
            size_t p = content.find(kToolSep, pos);
            if (p == std::string::npos) {
                parts.push_back(content.substr(pos));
                break;
            }
            parts.push_back(content.substr(pos, p - pos));
            pos = p + kToolSep.size();
        }
        std::string processed = parts[0];
        size_t tc_count = msg.tool_calls.size();
        size_t sep_count = parts.size() - 1;
        for (size_t i = 1; i < parts.size(); ++i) {
            size_t tool_idx = i - 1;
            if (tool_idx < tc_count)
                processed += tool_xml(msg.tool_calls[tool_idx]) + parts[i];
            else
                processed += parts[i];
        }
        for (size_t i = sep_count; i < tc_count; ++i)
            processed += tool_xml(msg.tool_calls[i]);
        content = processed;
    }

    std::string out = kImStart + "assistant\n";
    if (!reasoning.empty()) {
        out += kThinkOpen + "\n" + strip_n(reasoning) + "\n" + kThinkClose + "\n\n";
        out += lstrip_n(content);
    } else if (!has_substr(content, kThinkOpen) && !has_substr(content, kThinkClose)) {
        out += kThinkOpen + "\n\n" + kThinkClose + "\n\n";
        out += lstrip_n(content);
    } else {
        out += content;
    }
    out += kImEnd + "\n";
    return out;
}

std::string Renderer::render_tool_block(
        const std::vector<Message>& messages,
        size_t first_idx, size_t last_idx) const {
    std::string out;
    if (first_idx == 0 || messages[first_idx - 1].role != "tool") {
        out += kImStart;
        out += "user";
    }
    for (size_t i = first_idx; i <= last_idx; ++i) {
        out += "\n";
        out += kToolRespOpen;
        out += "\n";
        out += messages[i].content;
        out += "\n";
        out += kToolRespClose;
    }
    if (last_idx + 1 >= messages.size() || messages[last_idx + 1].role != "tool") {
        out += kImEnd;
        out += "\n";
    }
    return out;
}

std::string Renderer::render_generation_prompt(const Options& opts) const {
    std::string out = kImStart + "assistant\n";
    if (opts.enable_thinking.has_value()) {
        if (*opts.enable_thinking)
            out += kThinkOpen + "\n";
        else
            out += kThinkOpen + "\n\n" + kThinkClose + "\n\n";
    }
    return out;
}

std::string Renderer::render(
        const std::vector<Message>& messages, const Options& opts) const {
    std::string out = render_prologue(messages, opts);
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        if (msg.role == "user" || (msg.role == "system" && i > 0))
            out += render_user_or_system(msg);
        else if (msg.role == "assistant")
            out += render_assistant(msg);
        else if (msg.role == "tool") {
            size_t j = i;
            while (j + 1 < messages.size() && messages[j + 1].role == "tool") ++j;
            out += render_tool_block(messages, i, j);
            i = j;
        }
    }
    if (opts.add_generation_prompt)
        out += render_generation_prompt(opts);
    return out;
}

} // namespace chat_template
