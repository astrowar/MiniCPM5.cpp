# MiniCPM5.cpp - C++ Inference Engine

A lightweight, high-performance, **from-scratch C++17 inference engine** designed to run **MiniCPM5 (2.4B parameters)** models natively on CPU.

Built with **zero third-party dependencies** (no PyTorch, Hugging Face, or Python requirements), this engine completely decodes official GGUF weight matrices and vocabulary metadata directly from disk at runtime, delivering efficient inference through optimized CPU kernels.

---

## 🆕 What's New

* **v1.2.0** (Latest) — Improved AVX2 decode kernels (Q4_K/Q6_K/Q8_K path), optimized Q4 metadata decode, fused Gate+Up hot path cleanup, and portable OpenMP thread-affinity tuning via runtime settings (`OMP_PLACES`/`OMP_PROC_BIND`) for Windows/Linux builds.
* **v1.0.0** — Initial release: full GGUF parser, Q4_K/Q6_K/Q8_0 dequantization, GQA with KV cache, RoPE embeddings, SwiGLU activations, temperature + top-p sampling, and an interactive CLI with `</think>` reasoning tag support.

---

## ✨ Key Features

* **🔧 100% Standalone C++17:** Written completely from scratch. No dependency bloat.
* **📦 Native GGUF Parser:** Decodes architectural constants and the entire 130,560-token vocabulary directly from GGUF metadata blocks.
* **⚡ On-the-Fly CPU Dequantization:** Custom optimized standard C++ matrix-vector multiplication (GEMV) kernels for **Q4_K (4-bit)**, **Q6_K (6-bit)**, and **Q8_0 (8-bit)** quantized formats. Weights are kept compressed in RAM to bypass the memory bandwidth bottleneck.
* **🧠 Core Transformer Implementation:** Hand-coded Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activations, and Grouped-Query Attention (GQA).
* **💾 Contextual KV Cache:** Dynamic layer-by-layer Key-Value history buffer for sequential autoregressive decoding.
* **🚀 OpenMP Parallelization:** Fully multi-threaded math loops that scale efficiently across physical CPU cores.
* **🎲 Stochastic Sampler:** Supports Temperature scaling and Top-p (Nucleus) sampling for creative and coherent text output.
* **💬 Interactive CLI:** Complete control over prompts, model path, context size, and reasoning `</think>` tags.
* **🔧 Tool Calling (Function Calling):** A generic, class-based tool system — the model can invoke native C++ tools (e.g. `get_datetime`) mid-conversation; the engine executes them and feeds the result back, running a multi-turn agentic loop. See [Tool Calling](#-tool-calling).

---

## 📁 Project Structure

```text
MiniCPM5.cpp/
├── CMakeLists.txt              # Dynamic CMake build system
├── README.md                   # This file
├── include/
│   ├── minicpm_metadata.h      # Mapped GGUF tensor offsets & architectural constants
│   ├── minicpm_special_tokens.h # Special / reasoning token definitions
│   ├── model.h                 # Engine: forward pass, weights, KV cache
│   ├── tokenizer.h             # GGUF stream tokenizer
│   ├── chat_template.h         # Chat-template Renderer (prompt formatting)
│   ├── tool.h                  # Tool-calling: base class, registry, parser
│   └── ops.h                   # Declarations of math kernels & activation functions
└── src/
    ├── main.cpp                # CLI + generation / agentic (tool-calling) loop
    ├── model.cpp               # Engine forward pass + GGUF tensor loading
    ├── tokenizer.cpp           # Tokenizer implementation
    ├── chat_template.cpp       # Chat-template Renderer implementation
    ├── tool.cpp                # Tool registry dispatch + built-in get_datetime tool
    ├── omp_config.h            # OpenMP thread-affinity tuning
    ├── ops.cpp                 # Scalar GEMV, RMSNorm, RoPE, and GQA
    ├── ops_avx2.cpp            # AVX2/FMA SIMD kernels (auto-enabled on x86)
    ├── ops_neon.cpp            # NEON SIMD kernels (auto-enabled on ARM)
    └── ops_internal.h          # Internal kernel helpers
```

---

## 🛠️ Compilation & Building

**Prerequisites:** 
- CMake 3.14 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **OpenMP** (usually included with your compiler)

### Linux / macOS

```bash
# Clone the repository
git clone https://github.com/astrowar/MiniCPM5.cpp.git
cd MiniCPM5.cpp

# Create build directory
mkdir build && cd build

# Configure and compile
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

The compiled binary `minicpm_engine` will be generated in the `build/` directory.

### Windows (MSVC)

Recommended configuration for v1.2+ with AVX2 optimizations:

```powershell
# From the repository root
cmake -S . -B build -DMINICPM_ENABLE_AVX2=ON -DMINICPM_USE_OPENMP_LLVM=ON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build --config Release -- /m
```

The binary will be at `build\Release\minicpm_engine.exe`.

### OpenMP Affinity Tuning (Optional)

For optimal decode stability and performance, configure OpenMP environment variables. Example for 8 physical cores:

**Linux/macOS:**
```bash
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=false
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_WAIT_POLICY=ACTIVE
```

**Windows (PowerShell):**
```powershell
$env:OMP_NUM_THREADS=8
$env:OMP_DYNAMIC="false"
$env:OMP_PLACES="cores"
$env:OMP_PROC_BIND="close"
$env:OMP_WAIT_POLICY="ACTIVE"
```

---

## 🚀 How to Run

### Getting the Model

Download a quantized GGUF model file. Recommended options:
- **Q4_K_M** (smaller, faster): `MiniCPM5-2B-Q4_K_M.gguf` (~1.6GB)
- **Q6_K_M** (balanced): `MiniCPM5-2B-Q6_K_M.gguf` (~2.0GB)

Place the model file in your working directory or specify its path with `-m`.

### CLI Usage

```bash
# Show usage and available flags
./minicpm_engine --help

# Run with default Portuguese test prompt
./minicpm_engine

# Run a custom prompt with 4 threads, disable the </think> reasoning tag
OMP_NUM_THREADS=4 ./minicpm_engine --text "Explain the theory of relativity simply." --no-think

# Specify custom model path
./minicpm_engine -m /path/to/MiniCPM5-2B-Q4_K_M.gguf --text "Your prompt here"
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--text "<prompt>"` | Custom input prompt | `"O Brasil é um país"` |
| `-m <path>` | Path to GGUF model file | Auto-detect |
| `-c <size>` | Maximum context size / sequence limit | `8192` |
| `-n <count>` | Maximum number of tokens to generate | `1024` |
| `--no-think` | Disable reasoning `</think>` tag injection | Enabled |
| `-v, --verbose` | Show detailed generation logs and progress | Off |
| `--help` | Display help information | - |

---

## 📊 Performance & Benchmarking

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

## 🏗️ Architectural Overview

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

## 🔧 Tool Calling

The engine supports **function/tool calling**: the model can invoke native C++ tools mid-conversation, and the engine runs a multi-turn **agentic loop** — executing the tool in-process and feeding the result back to the model until it produces a final answer.

Built on a generic, class-based tool system, adding a new tool is as simple as instantiating a subclass (see below).

### How it works

1. Tool definitions are injected into the system prompt via the chat template (`<tools>` block).
2. Each turn, the engine renders the growing conversation, runs the prompt phase, then decodes **buffering** the output (no token streaming) — tool-call detection needs the full turn.
3. The buffered output is parsed for a `<function name="...">` block.
4. **If a tool call is found:** the tool is executed in C++, the assistant turn (with the tool call) and the tool result are appended to the conversation, and the loop repeats.
5. **If no tool call is found:** the final answer is printed and the loop ends.

The loop is capped at `MAX_TOOL_TURNS = 4`. Re-encoding the prompt from position 0 each turn is safe because the KV cache is positional and the previous turn's output folds back into the next prompt.

### Example

With the default `get_datetime` tool registered, a prompt such as *"What day is it today?"* produces a tool call, the engine runs it, and the model answers on the next turn. In verbose mode you will see the tool call and its result logged, then the final natural-language answer.

### Architecture

Tool-calling lives in its own module (`include/tool.h` / `src/tool.cpp`) so it does not clutter `main.cpp`:

* **`Tool`** — abstract base class. A tool implements `name()`, `execute(args)`, and `definition()` (the JSON schema injected into the prompt).
* **`ToolRegistry`** — holds tools as `shared_ptr<Tool>` and exposes a templated `add<T>()` (guarded by a `static_assert` that `T` derives from `Tool`), plus `execute(name, args)` dispatch and `definitions()` for prompt injection.
* **`parse_tool_call()`** — extracts the function name and `<param>` key/value pairs from the model's buffered output.
* **`main.cpp`** — wires it up: registers tools, feeds `registry.definitions()` into the chat template options, runs the agentic loop, and dispatches each detected call through `registry.execute(...)`.

### Adding a new tool

Derive from `Tool` and register the subclass — no changes to `main.cpp` beyond the `add` call:

```cpp
class RandomTool : public Tool {
public:
    std::string name() const override { return "random"; }
    std::string execute(const std::vector<std::pair<std::string, std::string>>& args) const override {
        return std::to_string(/* ... compute ... */);
    }
    std::string definition() const override {
        return "{\"name\": \"random\", \"description\": \"Roll a random number.\", "
               "\"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}";
    }
};
```

```cpp
ToolRegistry registry;
registry.add<GetDateTimeTool>();
registry.add<RandomTool>();   // <- new tool
```

---

## ⚠️ Limitations

* **Model support:** Currently only tested with **MiniCPM5 (2.4B parameters)** in Q4_K_M and Q6_K_M GGUF formats. Other model architectures or quantization types may not work.
* **Single batch size:** The engine runs with `batch_size = 1`, optimized for autoregressive decoding (one token at a time).
* **CPU only:** No GPU acceleration. Performance depends entirely on CPU memory bandwidth and core count.
* **Limited quantization formats:** Only Q4_K, Q6_K, and Q8_0 formats are supported.

## 🐛 Troubleshooting

**Issue:** Slow performance even with multiple threads
- **Solution:** Ensure OpenMP is properly linked. Check with `ldd minicpm_engine` (Linux) or verify OpenMP flags during compilation. Configure OpenMP environment variables as shown above.

**Issue:** Model file not found
- **Solution:** Use absolute path with `-m` flag or place the GGUF file in the same directory as the executable.

**Issue:** Compilation errors about OpenMP
- **Solution:** Install OpenMP libraries: 
  - Ubuntu/Debian: `sudo apt-get install libomp-dev`
  - macOS: `brew install libomp`
  - Windows: Usually included with MSVC or MinGW-w64

**Issue:** Segmentation fault or crash
- **Solution:** Ensure you have enough RAM (at least 4GB free). Try reducing context size with `-c 4096` or `-c 2048`.

---

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- **MiniCPM Team** for the base model architecture
- **GGUF Format** developers for the quantization framework
- Community contributors and testers

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the [GitHub repository](https://github.com/astrowar/MiniCPM5.cpp).
