#include "tool_examples/examples.h"
#include <cmath>
#include <ctime>

// ============================================================================
// Exemplos de tools para MiniCPM5.cpp
//
// Dois estilos:
//   A) Classe herdando de Tool  -> tools com estado ou logica complexa
//   B) add_function (lambda)    -> funcoes simples, stateless
// ============================================================================

// --- Estilo A: classe (GetDateTimeTool) ---

class GetDateTimeTool : public Tool {
public:
    std::string name() const override { return "get_datetime"; }

    std::string execute(const std::vector<std::pair<std::string, std::string>>& args) const override {
        (void)args;
        std::time_t now = std::time(nullptr);
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &now);
#else
        localtime_r(&now, &tm_buf);
#endif
        char buf[64];
        std::strftime(buf, sizeof buf, "%A, %Y-%m-%d %H:%M:%S", &tm_buf);
        return std::string(buf);
    }

    std::string definition() const override {
        return "{\"name\": \"get_datetime\", \"description\": \"Get the current date and time.\", "
               "\"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}";
    }
};

// --- Estilo B: add_function (lambdas) ---

void register_example_tools(ToolRegistry& registry) {
    // Data/hora (classe, sem params)
    registry.add<GetDateTimeTool>();

    // Matematica (lambdas stateless)
    registry.add_function("add", "Adds two numbers",
        [](double a, double b) { return a + b; },
        { TOOL_ARG(a, "First number"), TOOL_ARG(b, "Second number") });

    registry.add_function("multiply", "Multiplies two numbers",
        [](double a, double b) { return a * b; },
        { TOOL_ARG(a, "First number"), TOOL_ARG(b, "Second number") });

    registry.add_function("sqrt", "Computes the square root of a number",
        [](double x) { return std::sqrt(x); },
        { TOOL_ARG(x, "The number") });
}
