cmake_minimum_required(VERSION 3.14)
project(MiniCPM_Engine VERSION 1.0.0 LANGUAGES CXX)
include(CheckCXXCompilerFlag)

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
include_directories(include src otimizacao_scalar)

# Set base sources
set(SOURCES src/main.cpp src/model.cpp src/tokenizer.cpp src/ops.cpp src/chat_template.cpp otimizacao_scalar/ops_scalar_optimized.cpp)
set(SIMD_FLAGS "")
option(MINICPM_ENABLE_AVX2 "Enable AVX2 optimized kernels on x86 targets" ON)
option(MINICPM_USE_OPENMP_LLVM "Use /openmp:llvm on MSVC when available" ON)

# Architecture-specific AVX2 detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|amd64|AMD64|i686|i386|x86)")
    if(MINICPM_ENABLE_AVX2)
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
        message(STATUS "x86 architecture detected. AVX2 disabled by MINICPM_ENABLE_AVX2=OFF.")
    endif()
else()
    message(STATUS "Non-x86 architecture detected (${CMAKE_SYSTEM_PROCESSOR}).")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|aarch64.*|ARM.*|AARCH64.*)")
        message(STATUS "ARM architecture detected. Enabling NEON optimization.")
        list(APPEND SOURCES src/ops_neon.cpp)
        add_compile_definitions(__ARM_NEON)
        # For 32-bit ARM, NEON needs to be explicitly enabled in GCC/Clang
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7.*|armv8l|arm)$" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            set(SIMD_FLAGS "-mfpu=neon")
        endif()
    else()
        message(STATUS "Disabling SIMD support (using scalar fallback).")
    endif()
endif()

# Define the executable and its source files
add_executable(minicpm_engine ${SOURCES})

# Compiler flags for optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minicpm_engine PRIVATE -O3 -march=native -ffast-math ${SIMD_FLAGS})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(MINICPM_OMP_FLAG "")
    if(MINICPM_USE_OPENMP_LLVM)
        check_cxx_compiler_flag("/openmp:llvm" COMPILER_SUPPORTS_OPENMP_LLVM)
        if(COMPILER_SUPPORTS_OPENMP_LLVM)
            set(MINICPM_OMP_FLAG "/openmp:llvm")
            message(STATUS "MSVC: enabling LLVM OpenMP runtime via /openmp:llvm.")
        else()
            message(STATUS "MSVC: /openmp:llvm not supported by compiler; using OpenMP target defaults.")
        endif()
    endif()
    target_compile_options(minicpm_engine PRIVATE /O2 /fp:fast ${SIMD_FLAGS} ${MINICPM_OMP_FLAG})
endif()

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    if(MSVC AND MINICPM_OMP_FLAG STREQUAL "/openmp:llvm")
        message(STATUS "MSVC: relying on /openmp:llvm runtime (no OpenMP::OpenMP_CXX link target).")
    else()
        target_link_libraries(minicpm_engine PUBLIC OpenMP::OpenMP_CXX)
    endif()
else()
    message(WARNING "OpenMP not found! The engine will run single-threaded and much slower.")
endif()
