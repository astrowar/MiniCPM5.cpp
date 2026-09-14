#include "tool.h"

namespace {
// Remove wrapper CDATA, se presente, do valor de um <param>.
std::string strip_cdata(const std::string& v) {
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
} // namespace

ParsedCall parse_tool_call(const std::string& text) {
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
std::string ToolRegistry::execute(const std::string& name, const ParsedArgs& args) const {
    for (const auto& t : tools_) {
        if (t->name() == name) {
            return t->execute(args);
        }
    }
    return "{\"error\": \"Tool not found: \\\"" + name + "\\\"\"}";
}

ToolDefinitionList ToolRegistry::definitions() const {
    ToolDefinitionList out;
    out.reserve(tools_.size());
    for (const auto& t : tools_) out.push_back(t->definition());
    return out;
}
