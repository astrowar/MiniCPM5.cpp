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

# Set base sources
set(SOURCES src/main.cpp src/ops.cpp)
set(SIMD_FLAGS "")

# Architecture-specific AVX2 detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64|i686|i386|x86)")
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # MSVC: /arch:AVX2 habilita AVX2+FMA simultaneamente
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            list(APPEND SOURCES src/ops_avx2.cpp)
            set(SIMD_FLAGS "/arch:AVX2")
            message(STATUS "MSVC: AVX2/FMA supported. Enabling ops_avx2.cpp.")
            add_compile_definitions(__AVX2__)
        else()
            message(STATUS "MSVC: AVX2 not supported by this compiler version.")
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
            list(APPEND SOURCES src/ops_avx2.cpp)
            set(SIMD_FLAGS "-mavx2" "-mfma")
            message(STATUS "x86 Architecture detected with AVX2/FMA support. Enabling ops_avx2.cpp.")
            add_compile_definitions(__AVX2__) # Força a flag de macro no pre-processador
        else()
            message(STATUS "x86 Architecture detected, but AVX2/FMA not supported by compiler.")
        endif()
    endif()
else()
    message(STATUS "Non-x86 architecture detected (${CMAKE_SYSTEM_PROCESSOR}). Disabling AVX2 support.")
endif()

# Define the executable and its source files
add_executable(minicpm_engine ${SOURCES})

# Compiler flags for optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minicpm_engine PRIVATE -O3 -march=native -ffast-math ${SIMD_FLAGS})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(minicpm_engine PRIVATE /O2 /fp:fast /arch:AVX2 ${SIMD_FLAGS})
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(minicpm_engine PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found! The engine will run single-threaded and much slower.")
endif()
