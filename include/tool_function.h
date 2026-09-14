#pragma once

#include "tool.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <tuple>

// ============================================================================
// FUNCTION TOOLS: registre funcoes C++ comuns como tools sem boilerplate.
//
// O compilador deduz tipos via function_traits; o usuario fornece apenas
// nomes e descricoes dos parametros.
//
// Uso:
//   #include "tool_function.h"
//
//   int add(int a, int b) { return a + b; }
//
//   ToolRegistry registry;
//   registry.add_function("add", "Adds two integers", add, {
//       TOOL_ARG(a, "First integer"),
//       TOOL_ARG(b, "Second integer")
//   });
//
// Lambdas tambem funcionam:
//   registry.add_function("celsius_to_f", "Converts C to F",
//       [](double c) { return c * 1.8 + 32.0; },
//       { TOOL_ARG(c, "Temperature in Celsius") });
// ============================================================================

// ----------------------------------------------------------------------------
// function_traits: decompoe o tipo de uma funcao/lambda em compile-time
// ----------------------------------------------------------------------------

template<typename T> struct function_traits;

// Ponteiro de funcao: R (*)(Args...)
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> {
    using return_type = R;
    static constexpr std::size_t arity = sizeof...(Args);
    template<std::size_t I>
    using arg = typename std::tuple_element<I, std::tuple<Args...>>::type;
};

// Member function pointer (lambda operator() non-const)
template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...)> {
    using return_type = R;
    static constexpr std::size_t arity = sizeof...(Args);
    template<std::size_t I>
    using arg = typename std::tuple_element<I, std::tuple<Args...>>::type;
};

// Const member function pointer (lambda operator() const)
template<typename C, typename R, typename... Args>
struct function_traits<R(C::*)(Args...) const> {
    using return_type = R;
    static constexpr std::size_t arity = sizeof...(Args);
    template<std::size_t I>
    using arg = typename std::tuple_element<I, std::tuple<Args...>>::type;
};

// Generic: herda de traits do operator()
template<typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};

// ----------------------------------------------------------------------------
// json_type<T>: mapeia tipo C++ para tipo JSON Schema
// ----------------------------------------------------------------------------

template<typename T>
constexpr const char* json_type() {
    using U = std::decay_t<T>;
    if constexpr (std::is_same_v<U, bool>)
        return "boolean";
    else if constexpr (std::is_integral_v<U>)
        return "integer";
    else if constexpr (std::is_floating_point_v<U>)
        return "number";
    else
        return "string";
}

// ----------------------------------------------------------------------------
// parse_tool_value<T>: converte string (do modelo) para tipo T
// ----------------------------------------------------------------------------

template<typename T>
T parse_tool_value(const std::string& value);

template<> inline int    parse_tool_value<int>(const std::string& v)    { return std::stoi(v); }
template<> inline long   parse_tool_value<long>(const std::string& v)   { return std::stol(v); }
template<> inline double parse_tool_value<double>(const std::string& v) { return std::stod(v); }
template<> inline float  parse_tool_value<float>(const std::string& v)  { return std::stof(v); }
template<> inline bool   parse_tool_value<bool>(const std::string& v)   { return v == "true" || v == "1"; }
template<> inline std::string parse_tool_value<std::string>(const std::string& v) { return v; }

// ----------------------------------------------------------------------------
// tool_result_to_string<T>: converte retorno da funcao para string
// ----------------------------------------------------------------------------

template<typename T>
std::string tool_result_to_string(T&& value) {
    using U = std::decay_t<T>;
    if constexpr (std::is_same_v<U, std::string>)
        return std::move(value);
    else if constexpr (std::is_same_v<U, bool>)
        return value ? "true" : "false";
    else if constexpr (std::is_floating_point_v<U>) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%g", (double)value);
        return std::string(buf);
    } else
        return std::to_string(value);
}

// ----------------------------------------------------------------------------
// invoke_function: chama a funcao expandindo o pack de argumentos
// ----------------------------------------------------------------------------

template<typename F, std::size_t... I>
auto invoke_function(const F& fn, const std::vector<std::string>& values,
                     std::index_sequence<I...>) {
    using traits = function_traits<std::decay_t<F>>;
    return fn(parse_tool_value<std::decay_t<typename traits::template arg<I>>>(values[I])...);
}

// ----------------------------------------------------------------------------
// FunctionTool<F>: adapter que implementa Tool encapsulando uma funcao
// ----------------------------------------------------------------------------

template<typename F>
class FunctionTool : public Tool {
private:
    std::string tool_name_;
    std::string tool_description_;
    F function_;
    std::vector<ToolParam> params_;

    using Traits = function_traits<F>;

    template<std::size_t I>
    std::string make_prop(const ToolParam& p) const {
        std::string s = "\"" + p.name + "\"";
        s += ": {\"type\": \"";
        s += json_type<std::decay_t<typename Traits::template arg<I>>>();
        s += "\"";
        if (!p.description.empty()) {
            s += ", \"description\": \"";
            s += p.description;
            s += "\"";
        }
        s += "}";
        return s;
    }

    template<std::size_t... I>
    std::string build_properties(std::index_sequence<I...>) const {
        std::vector<std::string> parts{make_prop<I>(params_[I])...};
        std::string result;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            if (i > 0) result += ", ";
            result += parts[i];
        }
        return result;
    }

public:
    FunctionTool(std::string name, std::string description,
                 F fn, std::vector<ToolParam> params)
        : tool_name_(std::move(name)),
          tool_description_(std::move(description)),
          function_(std::move(fn)),
          params_(std::move(params))
    {
    }

    std::string name() const override { return tool_name_; }

    std::string definition() const override {
        std::string json = "{\"name\": \"" + tool_name_ + "\"";
        json += ", \"description\": \"" + tool_description_ + "\"";
        json += ", \"parameters\": {\"type\": \"object\", \"properties\": {";
        json += build_properties(std::make_index_sequence<Traits::arity>{});
        json += "}, \"required\": [";
        for (std::size_t i = 0; i < params_.size(); ++i) {
            if (i > 0) json += ", ";
            json += "\"" + params_[i].name + "\"";
        }
        json += "]}";
        return json;
    }

    std::string execute(const std::vector<std::pair<std::string, std::string>>& args) const override {
        // Extrai valores na ordem dos parametros declarados
        std::vector<std::string> values;
        values.reserve(params_.size());
        for (const auto& p : params_) {
            std::string val;
            for (const auto& kv : args) {
                if (kv.first == p.name) { val = kv.second; break; }
            }
            values.push_back(std::move(val));
        }

        if constexpr (std::is_void_v<typename Traits::return_type>) {
            invoke_function(function_, values, std::make_index_sequence<Traits::arity>{});
            return "ok";
        } else {
            auto result = invoke_function(function_, values, std::make_index_sequence<Traits::arity>{});
            return tool_result_to_string(result);
        }
    }
};
