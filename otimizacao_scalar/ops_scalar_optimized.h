#pragma once

// Portable scalar reference for the optimized quantized path.
//
// This file intentionally contains NO ISA intrinsics.  Its purpose is to define
// the exact math/dataflow that an architecture backend (AVX2/AVX-VNNI, NEON,
// SVE/SVE2, RVV, WASM SIMD, etc.) should reproduce.

#include "ops_internal.h"
#include <vector>

enum class KQuantType {
    Q4_K,
    Q6_K,
};

struct QuantMatrixK {
    const char* data = nullptr;
    int rows = 0;
    int cols = 0;
    KQuantType type = KQuantType::Q4_K;
};

// Reusable activation workspace. Keep one of these in the engine (or one per
// worker/thread if the scheduling model requires it) to avoid allocations.
struct Q8KWorkspace {
    std::vector<block_q8_K> blocks;

    block_q8_K* resize_for(int n);
    const block_q8_K* data() const { return blocks.data(); }
};

// -----------------------------------------------------------------------------
// 1) FP32 activation -> Q8_K
// -----------------------------------------------------------------------------
void quantize_row_q8_K_scalar(const float* x, block_q8_K* y, int n);

// -----------------------------------------------------------------------------
// 2) Pure scalar row kernels. No threading, no allocation.
//    These are the core mathematical reference for architecture-specific ports.
// -----------------------------------------------------------------------------
float dot_row_q4_K_q8_K_scalar(
    const block_q4_K* w,
    const block_q8_K* xq,
    int num_blocks);

float dot_row_q6_K_q8_K_scalar(
    const block_q6_K* w,
    const block_q8_K* xq,
    int num_blocks);

// -----------------------------------------------------------------------------
// 3) Pre-quantized GEMV. The activation is already Q8_K.
// -----------------------------------------------------------------------------
void gemv_q4_K_q8_K_scalar(
    const char* matrix_weights,
    const block_q8_K* xq,
    float* out,
    int num_rows,
    int num_cols);

void gemv_q6_K_q8_K_scalar(
    const char* matrix_weights,
    const block_q8_K* xq,
    float* out,
    int num_rows,
    int num_cols);

void gemv_k_q8_K_scalar(
    const QuantMatrixK& matrix,
    const block_q8_K* xq,
    float* out);

// Convenience wrappers: useful as drop-in references, but NOT what Q/K/V or
// gate/up should use, because these quantize x on every call.
void gemv_q4_K_scalar_q8_path(
    const char* matrix_weights,
    const float* x,
    float* out,
    int num_rows,
    int num_cols,
    Q8KWorkspace& workspace);

void gemv_q6_K_scalar_q8_path(
    const char* matrix_weights,
    const float* x,
    float* out,
    int num_rows,
    int num_cols,
    Q8KWorkspace& workspace);

// -----------------------------------------------------------------------------
// 4) Structural fusion / activation reuse.
//
// These functions quantize the FP32 activation ONCE and reuse the same Q8_K
// blocks for all projections.  They also use one outer OpenMP region when
// OpenMP is enabled.  The scalar row kernels remain completely portable.
// -----------------------------------------------------------------------------
void project_qkv_scalar(
    const float* x,
    int x_size,
    const QuantMatrixK& wq,
    const QuantMatrixK& wk,
    const QuantMatrixK& wv,
    float* q,
    float* k,
    float* v,
    Q8KWorkspace& workspace);

void project_gate_up_scalar(
    const float* x,
    int x_size,
    const QuantMatrixK& w_gate,
    const QuantMatrixK& w_up,
    float* gate,
    float* up,
    Q8KWorkspace& workspace);

