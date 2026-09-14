#include "tool_function.h"
#include <cassert>
#include <iostream>
#include <string>

// --- Funcoes de teste ---

int add(int a, int b) { return a + b; }

double distance(double x, double y) { return std::sqrt(x * x + y * y); }

std::string greet(const std::string& name) { return "Hello, " + name + "!"; }

bool is_even(int n) { return n % 2 == 0; }

void noop() {}

int main() {
    ToolRegistry registry;

    // 1. Funcao simples: int + int -> int
    registry.add_function("add", "Adds two integers", add, {
        TOOL_ARG(a, "First integer"),
        TOOL_ARG(b, "Second integer")
    });

    // 2. Lambda: double -> double
    registry.add_function("celsius_to_f", "Converts Celsius to Fahrenheit",
        [](double c) { return c * 1.8 + 32.0; },
        { TOOL_ARG(c, "Temperature in Celsius") });

    // 3. String argument
    registry.add_function("greet", "Greets someone", greet, {
        TOOL_ARG(name, "Name to greet")
    });

    // 4. Bool return
    registry.add_function("is_even", "Checks if a number is even", is_even, {
        TOOL_ARG(n, "Number to check")
    });

    // 5. Lambda com varios tipos
    registry.add_function("mix", "Mixes types",
        [](int a, double b, const std::string& s) -> std::string {
            return std::to_string(a) + " " + std::to_string(b) + " " + s;
        },
        {
            TOOL_ARG(a, "An integer"),
            TOOL_ARG(b, "A number"),
            TOOL_ARG(s, "A string")
        });

    // --- Testes de execute ---

    auto result = registry.execute("add", {{"a", "10"}, {"b", "20"}});
    assert(result == "30");
    std::cout << "[PASS] add(10,20) = " << result << std::endl;

    result = registry.execute("celsius_to_f", {{"c", "100"}});
    assert(result == "212");
    std::cout << "[PASS] celsius_to_f(100) = " << result << std::endl;

    result = registry.execute("greet", {{"name", "World"}});
    assert(result == "Hello, World!");
    std::cout << "[PASS] greet(World) = " << result << std::endl;

    result = registry.execute("is_even", {{"n", "4"}});
    assert(result == "true");
    std::cout << "[PASS] is_even(4) = " << result << std::endl;

    result = registry.execute("is_even", {{"n", "7"}});
    assert(result == "false");
    std::cout << "[PASS] is_even(7) = " << result << std::endl;

    result = registry.execute("mix", {{"a", "42"}, {"b", "3.14"}, {"s", "hello"}});
    assert(result == "42 3.140000 hello");
    std::cout << "[PASS] mix(42, 3.14, hello) = " << result << std::endl;

    // Tool inexistente
    result = registry.execute("nonexistent", {});
    assert(result.find("error") != std::string::npos);
    std::cout << "[PASS] nonexistent tool returns error" << std::endl;

    // --- Testes de definition ---

    auto defs = registry.definitions();
    assert(defs.size() == 5);

    // Verifica que a definicao do "add" contem os tipos corretos
    assert(defs[0].find("\"integer\"") != std::string::npos);
    assert(defs[0].find("\"name\": \"add\"") != std::string::npos);
    assert(defs[0].find("\"a\"") != std::string::npos);
    assert(defs[0].find("\"b\"") != std::string::npos);
    std::cout << "[PASS] add definition: " << defs[0] << std::endl;

    // Lambda com double
    assert(defs[1].find("\"number\"") != std::string::npos);
    std::cout << "[PASS] celsius_to_f definition: " << defs[1] << std::endl;

    // String
    assert(defs[2].find("\"string\"") != std::string::npos);
    std::cout << "[PASS] greet definition: " << defs[2] << std::endl;

    // is_even: param int
    assert(defs[3].find("\"integer\"") != std::string::npos);
    std::cout << "[PASS] is_even definition: " << defs[3] << std::endl;

    // Mix: integer + number + string
    assert(defs[4].find("\"integer\"") != std::string::npos);
    assert(defs[4].find("\"number\"") != std::string::npos);
    assert(defs[4].find("\"string\"") != std::string::npos);
    std::cout << "[PASS] mix definition: " << defs[4] << std::endl;

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
