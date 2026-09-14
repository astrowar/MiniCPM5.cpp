#pragma once

#include "tool.h"
#include "tool_function.h"

// ============================================================================
// EXEMPLOS DE TOOLS - como adicionar suas proprias ferramentas
//
// Este arquivo nao faz parte da build padrao. Para incluir:
//
//   1. Adicione tool_examples/examples.cpp ao CMakeLists.txt:
//        set(SOURCES ... tool_examples/examples.cpp)
//
//   2. No main.cpp:
//        #include "tool_examples/examples.h"
//        ...
//        ToolRegistry registry;
//        register_example_tools(registry);
//
// Use este arquivo como template para criar suas proprias tools.
// ============================================================================

// Registra todas as tools de exemplo no registry.
void register_example_tools(ToolRegistry& registry);
