# MiniCPM C++ Inference Engine

A lightweight, high-performance, and **from-scratch C++17 inference engine** designed to run **MiniCPM5 (2B)** models natively.

This engine is designed with zero third-party dependencies (no PyTorch, Hugging Face, or Python requirements) and completely decodes official GGUF weight matrices and vocabulary metadata directly from disk at runtime.

---

## What's New

* **v1.0.0** — Initial release: full GGUF parser, Q4_K/Q6_K/Q8_0 dequantization, GQA with KV cache, RoPE embeddings, SwiGLU activations, temperature + top-p sampling, and an interactive CLI with `</think>` reasoning tag support.

---

## Key Features

* **100% Standalone C++17:** Written completely from scratch. No dependency bloat.
* **Native GGUF Parser:** Decodes architectural constants and the entire 130,560-token vocabulary directly from GGUF metadata blocks.
* **On-the-Fly CPU Dequantization:** Custom optimized standard C++ matrix-vector multiplication (GEMV) kernels for **Q4_K (4-bit)**, **Q6_K (6-bit)**, and **Q8_0 (8-bit)** quantized formats. Weights are kept compressed in RAM to bypass the memory bandwidth bottleneck.
* **Core Transformer Implementation:** Hand-coded Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activations, and Grouped-Query Attention (GQA).
* **Contextual KV Cache:** Dynamic layer-by-layer Key-Value history buffer for sequential autoregressive decoding.
* **OpenMP Parallelization:** Fully multi-threaded math loops that scale efficiently across physical CPU cores.
* **Stochastic Sampler:** Supports Temperature scaling and Top-p (Nucleus) sampling for creative and coherent text output.
* **Interactive CLI:** Complete control over prompts, model path, context size, and reasoning `</think>` tags.

---

## Project Structure

```text
minicpm-cpp-engine/
├── CMakeLists.txt         # Dynamic CMake build system
├── include/
│   ├── minicpm_metadata.h # Mapped GGUF tensor offsets & architectural constants
│   └── ops.h              # Declarations of math kernels & activation functions
└── src/
    ├── main.cpp           # Main generation loop, CLI, and dynamic GGUF Parser
    └── ops.cpp            # Implementations of GEMV, RMSNorm, RoPE, and GQA
```

---

## Compilation & Building

**Dependencies:** CMake 3.14+, a C++17 compiler (GCC/Clang), and **OpenMP** (usually provided with your compiler).

```bash
# Clone and enter the repository directory
cd minicpm-cpp-engine

# Create build directory
mkdir build && cd build

# Configure and compile
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

The compiled binary `minicpm_engine` will be generated in the `build/` directory.

---

## How to Run (CLI Usage)

Put your quantized model (such as `MiniCPM5-2B-Q4_K_M.gguf`) in the directory above the build folder, and run the binary:

```bash
# Show usage and available flags
./minicpm_engine --help

# Run default Portuguese test prompt
./minicpm_engine

# Run a custom prompt, disable the think reasoning tag, and allow 4 threads
OMP_NUM_THREADS=4 ./minicpm_engine --text "Explain the theory of relativity simply." --no-think
```

### Supported Arguments:
* `--text "<prompt>"` : Provide your custom input prompt (default: `"O Brasil é um país"`).
* `-m <path>` : Define path to your GGUF model file.
* `-c <size>` : Configure maximum context size / sequence limit (default: `1024`).
* `-n <count>` : Configure maximum generation tokens (default: `25`).
* `--no-think` : Disables reasoning `</think>` tag injection.
* `-v, --verbose` : Show detailed generation logs and progress.

---

## Performance & Benchmarking

Because LLM generation is **memory bandwidth bound** at batch size 1, keeping weights compressed in RAM and dequantizing them directly into CPU caches ("on-the-fly") yields dramatic speedups.

Linking OpenMP parallelization achieves nearly linear speedup scaling across CPU cores:

### x86_64 (AVX2 / AMD Ryzen 7 4750U Mobile)
| Threads (`OMP_NUM_THREADS`) | Token Generation Speed |
|-----------------------------|------------------------|
| **1 Thread**                | ~9.2 tok/s             |
| **4 Threads**               | **~27.3 tok/s**        |
| **8 Threads**               | ~21.2 tok/s (RAM Saturated) |

### ARM AArch64 (NEON / Raspberry Pi 4 - 4GB)
| Threads (`OMP_NUM_THREADS`) | Token Generation Speed |
|-----------------------------|------------------------|
| **1 Thread**                | ~0.50 tok/s            |
| **2 Threads**               | ~0.97 tok/s            |
| **4 Threads**               | **~1.76 tok/s**        |

For detailed architectural information on ARM compilation, read [ARM_NEON.md](ARM_NEON.md). For broader scalar fallback details, check the internal documentation.

---

## Architectural Overview

### 1. The GGUF Stream Tokenizer
Unlike naive loaders that rely on external `.bin` mappings, the C++ `Tokenizer` class parses the GGUF file stream dynamically:
1. It reads the binary file header and loops through Key-Value metadatas.
2. Locates `"tokenizer.ggml.tokens"` and extracts the raw BPE byte arrays.
3. Applies a native C++ implementation of the **GPT-2 Byte-to-Unicode reverse mapping** to recover raw character bytes.
4. Maintains an internal stateful UTF-8 validation buffer during decoding to hold incomplete multi-byte characters (e.g. `é`) until they are fully resolved by consecutive tokens, preventing visual corruption (`mojibake`).

### 2. On-the-Fly Matrix Multiplication (GEMV)
Weights are fetched from disk directly in block-quantized structures:
* **`block_q8_0` (34 bytes for 32 weights):** Decodes a shared Float16 block scale and multiplies elements inside the loop.
* **`block_q4_K` (144 bytes for 256 weights):** Dynamically unpacks 6-bit sub-block scales/mins and 4-bit weight nibbles inline.
* **`block_q6_K` (210 bytes for 256 weights):** Unpacks lower 4-bit nibbles and merges them with upper 2-bit masks to reconstruct 6-bit weights.

By keeping these structures compressed in RAM, cache misses are minimized, making the engine extremely competitive on lightweight edge hardware.

---

## Limitations

* **Model support:** Currently only tested with **MiniCPM5-2B** in Q4_K_M and Q6_K_M GGUF formats. Other model architectures or quantization types may not work.
* **Single batch size:** The engine runs with `batch_size = 1`, optimized for single-threaded or multi-threaded autoregressive decoding.
* **CPU only:** No GPU acceleration. Performance depends entirely on CPU memory bandwidth and core count.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
