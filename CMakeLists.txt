cmake_minimum_required(VERSION 3.14)
project(MiniCPM_Engine VERSION 1.0.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable High Optimization for release builds
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Find OpenMP
find_package(OpenMP)

# Define the include directory
include_directories(include)

# Define the executable and its source files
add_executable(minicpm_engine
    src/main.cpp
    src/ops.cpp
    src/ops_avx2.cpp
)

# Compiler flags for optimization (GCC/Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minicpm_engine PRIVATE -O3 -march=native -ffast-math)
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(minicpm_engine PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found! The engine will run single-threaded and much slower.")
endif()
