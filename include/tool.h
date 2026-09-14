#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ============================================================================
// TOOL CALLING: infraestrutura generica de tools para o motor
//
// Nova tool = subclasse de Tool + instancia no ToolRegistry:
//     class MinhaTool : public Tool { ... };
//     ToolRegistry registry;
//     registry.add<MinhaTool>();
//
// O main so precisa:
//   - ParsedCall parse_tool_call(text)     -> detecta <function> na saida
//   - registry.execute(name, args)         -> despacha para a tool certa
//   - registry.definitions()              -> JSON para Options.tools_json
// ============================================================================

// Um tool call emitido pelo modelo: name + argumentos chave/valor.
struct ParsedCall {
    std::string name;
    std::vector<std::pair<std::string, std::string>> args;
};

// Classe base para tools. Todo tool concreto deriva dela e implementa as tres
// funcoes puras. execute() e const: a execucao nao mutacao o estado do tool.
class Tool {
public:
    virtual ~Tool() = default;
    // Nome que o modelo usa em <function name="...">.
    virtual std::string name() const = 0;
    // Executa a tool com os argumentos; devolve o resultado (string, ex.: JSON).
    virtual std::string execute(const std::vector<std::pair<std::string, std::string>>& args) const = 0;
    // Definicao JSON da tool (injetada no system prompt via Options.tools_json).
    virtual std::string definition() const = 0;
};

// Registro/dispensador de tools: mapeia name -> Tool.
class ToolRegistry {
public:
    // Inclusao via instancia: registry.add<GetDateTimeTool>();
    // Com argumentos de construtor: registry.add<MinhaTool>("cfg", 42);
    template <typename T, typename... Args>
    void add(Args&&... args) {
        static_assert(std::is_base_of<Tool, T>::value, "T deve derivar de Tool");
        tools_.push_back(std::make_shared<T>(std::forward<Args>(args)...));
    }

    // Executa a tool pelo nome; se nao existir, devolve um JSON de erro.
    std::string execute(const std::string& name,
                        const std::vector<std::pair<std::string, std::string>>& args) const;

    // Definicoes JSON de todas as tools (para Options.tools_json).
    std::vector<std::string> definitions() const;

private:
    std::vector<std::shared_ptr<Tool>> tools_;
};

// Parser do XML emitido pelo modelo: localiza o primeiro <function name="...">
// e extrai name + pares <param name="k">v</param>. Retorna name vazio se nenhum
// <function> for encontrado (=> resposta final, nao tool call).
ParsedCall parse_tool_call(const std::string& text);

// Tool concreta: data/hora local atual (sem parametros obrigatorios).
class GetDateTimeTool : public Tool {
public:
    std::string name() const override;
    std::string execute(const std::vector<std::pair<std::string, std::string>>& args) const override;
    std::string definition() const override;
};
