# ARM NEON & Raspberry Pi 4 Support

This engine is natively optimized for **ARM (AArch64)** architectures, providing a high-performance **NEON** vectorization path and a highly portable pure **Scalar** fallback. This ensures that the engine can run smoothly on low-power Edge devices, including the **Raspberry Pi 4**.

## Architecture & Dispatch

The project uses a dynamic fallback system handled at compile-time by CMake and macro definitions:

1. **AVX2 / FMA (`src/ops_avx2.cpp`):**
   - Automatically enabled on modern x86/x64 processors.
   - Extremely high throughput using 256-bit registers.

2. **ARM NEON (`src/ops_neon.cpp`):**
   - Automatically enabled on ARM architectures (Raspberry Pi, Apple Silicon M1/M2, Android Termux, etc.).
   - Utilizes 128-bit NEON intrinsics (`vmull_s8`, `vaddq_s32`, `vabsq_f32`) to achieve efficient integer math accumulation and on-the-fly quantization without needing to materialize FP32 weights.

3. **Scalar Optimized Fallback (`otimizacao_scalar/ops_scalar_optimized.cpp`):**
   - Enabled on platforms without SIMD support (RISC-V, MIPS, older ARMv6, etc.).
   - Uses the same mathematical concepts (fused operations, cache locality, integer accumulation) but completely unrolled and reliant on compiler auto-vectorization (`-O3 -ffast-math`).

## Fused Operations for Low-Power Devices

On constrained devices like a Raspberry Pi 4, memory bandwidth and thread synchronization overhead are critical bottlenecks. The ARM NEON and Scalar paths utilize **Fused Operators**:

* `rmsnorm_and_quantize_q8k`
* `swiglu_and_quantize_q8k`

Instead of writing FP32 results back to RAM and reading them again to quantize, these functions process blocks of 256 elements entirely within the L1 cache. The result is directly output as a `Q8_K` block ready for matrix multiplication, drastically reducing RAM reads/writes and eliminating extra OpenMP barrier synchronizations.

## Benchmarks on Raspberry Pi 4

Tested on a **Raspberry Pi 4 (4GB)** running natively on AArch64.
Model used: `MiniCPM5-2B-Q4_K_M.gguf` (1.5 GB).

| Threads | Generation Speed (tok/s) | Notes |
|---|---|---|
| **1 Thread** | ~0.50 tok/s | Baseline execution. |
| **2 Threads** | ~0.97 tok/s | Excellent linear scaling. |
| **4 Threads** | **~1.76 tok/s** | Maximum utilization of Cortex-A72 cores. |

The nearly perfect scaling from 2 to 4 threads shows that the CPU cores are fully utilized by the NEON integer instructions without immediately hitting the LPDDR4 memory bandwidth wall. Generating ~1.76 tokens/sec for a modern 2 Billion parameter reasoning model on a low-power SBC is an outstanding result for edge AI deployments.

## How to Build on Raspberry Pi

The build system automatically detects the architecture. Just run:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make clean && make -j4

# Run the benchmark
OMP_NUM_THREADS=4 ./minicpm_engine -m ../MiniCPM5-2B-Q4_K_M.gguf -n 40 --text "Qual é a capital do Brasil?" --no-think
```
