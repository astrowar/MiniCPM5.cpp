#include "tool_examples/examples.h"
#include <cmath>
#include <ctime>
#include <cstdio>

// ============================================================================
// Exemplos de tools para MiniCPM5.cpp
//
// Use este arquivo como template para suas proprias tools.
// Cada tool e uma lambda registrada via add_function.
// ============================================================================

void register_example_tools(ToolRegistry& registry) {

    registry.add_function("get_datetime", "Get the current date and time",
        []() -> std::string {
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
        },
        {});

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
