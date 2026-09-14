#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ============================================================================
// SEMANTIC TYPE ALIASES (Enhances code readability)
// ============================================================================
using ParsedArgs = std::vector<std::pair<std::string, std::string>>;
using ToolDefinitionList = std::vector<std::string>;

// ============================================================================
// TOOL CALLING: infraestrutura generica de tools para o motor
//
// Registre uma funcao/lambda como tool (ver tool_function.h):
//     registry.add_function("add", "Adds two integers", add, {
//         TOOL_ARG(a, "First integer"),
//         TOOL_ARG(b, "Second integer")
//     });
//
// O main so precisa:
//   - ParsedCall parse_tool_call(text)     -> detecta <function> na saida
//   - registry.execute(name, args)         -> despacha para a tool certa
//   - registry.definitions()              -> JSON para Options.tools_json
// ============================================================================

// Um tool call emitido pelo modelo: name + argumentos chave/valor.
struct ParsedCall {
    std::string name;
    ParsedArgs args;
};

// Metadado de um parametro de tool (nome + descricao + opcional).
struct ToolParam {
    std::string name;
    std::string description;
    bool required = true;
};

#define TOOL_ARG(name, desc) ToolParam{#name, desc}

// Classe base para tools. Todo tool concreto deriva dela e implementa as tres
// funcoes puras. execute() e const: a execucao nao mutacao o estado do tool.
class Tool {
public:
    virtual ~Tool() = default;
    // Nome que o modelo usa em <function name="...">.
    virtual std::string name() const = 0;
    // Executa a tool com os argumentos; devolve o resultado (string, ex.: JSON).
    virtual std::string execute(const ParsedArgs& args) const = 0;
    // Definicao JSON da tool (injetada no system prompt via Options.tools_json).
    virtual std::string definition() const = 0;
};

// Forward declaration (definicao em tool_function.h).
template<typename F> class FunctionTool;

// Registro/dispensador de tools: mapeia name -> Tool.
class ToolRegistry {
public:
    // Registro generico de Tool derivado (uso interno).
    template <typename T, typename... Args>
    void add(Args&&... args) {
        static_assert(std::is_base_of<Tool, T>::value, "T deve derivar de Tool");
        tools_.push_back(std::make_shared<T>(std::forward<Args>(args)...));
    }

    // Registra uma funcao C++ como tool (requer include de tool_function.h).
    template <typename F>
    void add_function(const std::string& name, const std::string& description,
                      F function, std::vector<ToolParam> params) {
        tools_.push_back(std::make_shared<FunctionTool<std::decay_t<F>>>(
            name, description, std::move(function), std::move(params)));
    }

    // Executa a tool pelo nome; se nao existir, devolve um JSON de erro.
    std::string execute(const std::string& name, const ParsedArgs& args) const;

    // Definicoes JSON de todas as tools (para Options.tools_json).
    ToolDefinitionList definitions() const;

private:
    std::vector<std::shared_ptr<Tool>> tools_;
};

// Parser do XML emitido pelo modelo: localiza o primeiro <function name="...">
// e extrai name + pares <param name="k">v</param>. Retorna name vazio se nenhum
// <function> for encontrado (=> resposta final, nao tool call).
ParsedCall parse_tool_call(const std::string& text);
