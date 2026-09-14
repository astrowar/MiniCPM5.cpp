add_executable(test_function_tool test_function_tool.cpp ${CMAKE_SOURCE_DIR}/src/tool.cpp)
target_include_directories(test_function_tool PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_function_tool PRIVATE -O2)
