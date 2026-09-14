#pragma once

// =============================================================================
// INTERNAL DISPATCH KERNELS (Not part of the public API)
// =============================================================================
// This header contains the explicit hardware backend kernel signatures (Scalar,
// AVX2, NEON) used internally by ops.cpp to perform runtime dispatching.
// =============================================================================

#include "ops.h"

// -----------------------------------------------------------------------------
// SCALAR FALLBACK KERNELS
// -----------------------------------------------------------------------------
void gemv_q4_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);
void gemv_q6_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);

void gemv_q4_K_q8_K_scalar(const char* matrix_weights, const block_q8_K* xq, float* out, int num_rows, int num_cols);
void gemv_q6_K_q8_K_scalar(const char* matrix_weights, const block_q8_K* xq, float* out, int num_rows, int num_cols);

void gemv_qkv_q4_K_q8_K_scalar(const char* wq, const char* wk, const char* wv,
                               const block_q8_K* xq, float* q, float* k, float* v,
                               int q_rows, int kv_rows, int num_cols);
void gemv_qkv_q4_q4_q6_q8_K_scalar(const char* wq, const char* wk, const char* wv,
                                   const block_q8_K* xq, float* q, float* k, float* v,
                                   int q_rows, int kv_rows, int num_cols);
void gemv_gate_up_q4_K_q8_K_scalar(const char* w_gate, const char* w_up,
                                   const block_q8_K* xq, float* gate, float* up,
                                   int num_rows, int num_cols);

// -----------------------------------------------------------------------------
// AVX2 HARDWARE KERNELS
// -----------------------------------------------------------------------------
void gemv_q4_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);
void gemv_q6_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);

// Optimized pre-quantized Q8_K kernels
void quantize_row_q8_K_avx2(const float* x, block_q8_K* y, int n);
void gemv_q4_K_q8_K_avx2(const char* matrix_weights, const block_q8_K* xq, float* out, int num_rows, int num_cols);
void gemv_q6_K_q8_K_avx2(const char* matrix_weights, const block_q8_K* xq, float* out, int num_rows, int num_cols);

// OpenMP Fused Region Kernels
void gemv_qkv_q4_K_q8_K_avx2(const char* wq, const char* wk, const char* wv,
                              const block_q8_K* xq,
                              float* q, float* k, float* v,
                              int q_rows, int kv_rows, int num_cols);
void gemv_qkv_q4_q4_q6_q8_K_avx2(const char* wq, const char* wk, const char* wv,
                                  const block_q8_K* xq,
                                  float* q, float* k, float* v,
                                  int q_rows, int kv_rows, int num_cols);
void gemv_gate_up_q4_K_q8_K_avx2(const char* w_gate, const char* w_up,
                                  const block_q8_K* xq,
                                  float* gate, float* up,
                                  int num_rows, int num_cols);

// -----------------------------------------------------------------------------
// ARM NEON HARDWARE KERNELS
// -----------------------------------------------------------------------------
void quantize_row_q8_K_neon(const float* x, block_q8_K* y, int n);
void gemv_q4_K_neon(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);
void gemv_q6_K_neon(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);
void gemv_q4_K_q8_K_neon(const char* matrix_weights, const block_q8_K* xq, float* out, int num_rows, int num_cols);
void gemv_q6_K_q8_K_neon(const char* matrix_weights, const block_q8_K* xq, float* out, int num_rows, int num_cols);
void gemv_qkv_q4_K_q8_K_neon(const char* wq, const char* wk, const char* wv,
                             const block_q8_K* xq, float* q, float* k, float* v,
                             int q_rows, int kv_rows, int num_cols);
void gemv_qkv_q4_q4_q6_q8_K_neon(const char* wq, const char* wk, const char* wv,
                                 const block_q8_K* xq, float* q, float* k, float* v,
                                 int q_rows, int kv_rows, int num_cols);
void gemv_gate_up_q4_K_q8_K_neon(const char* w_gate, const char* w_up,
                                 const block_q8_K* xq, float* gate, float* up,
                                 int num_rows, int num_cols);
