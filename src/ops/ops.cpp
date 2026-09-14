#include "ops.h"
#include "ops_internal.h"
#include <cmath>
#include <iostream>
#include <cassert>
#include <cstring>

// ============================================================================
// 2. "ON-THE-FLY" GENERAL MATRIX-VECTOR MULTIPLICATION (GEMV) KERNELS
//
// In large language models, matrix multiplications (GEMV / GEMM) are the primary
// computational bottleneck. To execute a single forward pass, the model must read
// billions of weight parameters from memory. Since CPUs are bounded by memory
// bandwidth rather than raw ALU compute, loading standard 32-bit floats (FP32)
// would result in extremely slow execution speeds.
//
// To resolve this, weights are stored in quantized (compressed) block formats
// (e.g., 4-bit Q4_K, 6-bit Q6_K, or 8-bit Q8_0). During inference, these low-bit
// weights are decompressed "on-the-fly" directly inside the CPU's registers or
// caches, where they are multiplied with the input vector. This drastically
// reduces memory traffic (up to 8x) and enables fast local inference on standard CPUs.
// ============================================================================

// ----------------------------------------------------------------------------
// GEMV for Float32 Tensors (Standard Precision)
//
// Computes a standard matrix-vector multiplication without quantization:
//   out = Matrix(W) * Vector(X)
// This is typically used for small, non-quantized layers (like final classification
// heads or small scaling parameters) that demand maximum numerical fidelity.
// ----------------------------------------------------------------------------
void gemv_f32(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const float* w = reinterpret_cast<const float*>(matrix_weights);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        float row_sum = 0.0f;
        int row_offset = r * num_cols;
        for (int c = 0; c < num_cols; ++c) {
            row_sum += w[row_offset + c] * x[c];
        }
        out[r] = row_sum;
    }
}

// ----------------------------------------------------------------------------
// GEMV for Q8_0 Tensors (Block-Wise 8-Bit Quantization)
//
// In the GGML Q8_0 quantization format:
//   - Weights are organized into blocks of 32 (QK8_0 = 32).
//   - Each block stores a shared scaling factor (d) in half-precision Float16 (2 bytes)
//     and 32 signed 8-bit integers (32 bytes).
//   - Total footprint per block: 34 bytes for 32 weights (~8.5 bits per weight).
//   - Decompression formula: weight = scale * quant_value = fp16_to_fp32(d) * qs[i].
//
// This format serves as an excellent balance between decompression speed and
// near-perfect model accuracy.
// ----------------------------------------------------------------------------
void gemv_q8_0(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(matrix_weights);
    int blocks_per_row = num_cols / QK8_0;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        float row_sum = 0.0f;
        int row_block_offset = r * blocks_per_row;

        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q8_0& bloco = blocks[row_block_offset + b];
            float scale = fp16_to_fp32(bloco.d);
            
            float sum_local = 0.0f;
            for (int i = 0; i < QK8_0; ++i) {
                sum_local += bloco.qs[i] * x[b * QK8_0 + i];
            }
            row_sum += sum_local * scale;
        }
        out[r] = row_sum;
    }
}

// ----------------------------------------------------------------------------
// GEMV for Q4_K Tensors (Block-Wise 4-Bit "K-Quantization" - Scalar Implementation)
//
// In the GGML Q4_K format:
//   - Weights are grouped into large superblocks of 256 parameters (QK_K = 256).
//   - Each superblock contains a global scale (d) in FP16 and a global offset (dmin) in FP16.
//   - The superblock is divided into 8 subblocks of 32 weights. Each subblock features
//     its own local 6-bit scale and 6-bit minimum offset (mins) to preserve high dynamic range.
//   - Each 4-bit quantized weight (qs) is reconstructed using the formula:
//       weight = d * scale * qs[i] - dmin * min
//
// This is the default format for consumer-grade CPU setups, achieving massive 
// compression (~4.5 bits per parameter) with minimal degradation in model perplexity.
// ----------------------------------------------------------------------------
void gemv_q4_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    // Distribute calculation of output rows across multiple CPU cores using OpenMP
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        // Use two accumulation registers to reduce pipeline stalling due to data hazards
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        int row_block_offset = r * super_blocks_per_row;

        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q4_K& bloco = blocks[row_block_offset + sb];
            
            // Convert the superblock's global scale and minimum offset from FP16 to FP32
            float d = fp16_to_fp32(bloco.d);
            float dmin = fp16_to_fp32(bloco.dmin);
            int x_offset = sb * QK_K;
            
            // Extract local 6-bit subblock scales and min offsets (contained within 12 bytes total)
            uint8_t scales[8], mins[8];
            decode_q4k_scales_mins(bloco.scales, scales, mins);
            const uint8_t* q = bloco.qs;

            // Process the 256-weight superblock in 4 chunks of 64 weights each
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int j = chunk * 64;
                
                // Precompute the scales and minimum offsets for the two subblocks (32 weights each) within this chunk
                float d1 = d * static_cast<float>(scales[2 * chunk + 0]);
                float m1 = dmin * static_cast<float>(mins[2 * chunk + 0]);
                float d2 = d * static_cast<float>(scales[2 * chunk + 1]);
                float m2 = dmin * static_cast<float>(mins[2 * chunk + 1]);

                // Unpack and multiply 32 pairs of 4-bit weights
                for (int l = 0; l < 32; ++l) {
                    // Extract lower 4 bits (nibble) for the first weight, scaling it and subtracting bias
                    float w1 = d1 * (q[l] & 0xF) - m1;
                    // Extract upper 4 bits (nibble) for the second weight (offset by 32 elements)
                    float w2 = d2 * (q[l] >> 4) - m2;
                    
                    // Multiply with corresponding input vector elements and accumulate
                    sum1 += w1 * x[x_offset + j + l];
                    sum2 += w2 * x[x_offset + j + l + 32];
                }
                // Move weight pointer to the next subblock pair chunk (each holds 64 weights in 32 bytes)
                q += 32;
            }
        }
        out[r] = sum1 + sum2;
    }
}

// Dispatcher for GEMV Q4_K (Automatically select AVX2 vectorization if compiled with AVX2 support)
void gemv_q4_K(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
#if defined(__AVX2__)
    gemv_q4_K_avx2(matrix_weights, x, out, num_rows, num_cols);
#else
    gemv_q4_K_scalar(matrix_weights, x, out, num_rows, num_cols);
#endif
}

// ----------------------------------------------------------------------------
// GEMV for Q6_K Tensors (Block-Wise 6-Bit "K-Quantization" - Scalar Implementation)
//
// In the GGML Q6_K format:
//   - Weights are organized in superblocks of 256 parameters (QK_K = 256).
//   - Each 6-bit quantized weight is split across two arrays: 4 lower bits (ql)
//     and 2 higher bits (qh) packed closely together to optimize byte alignment.
//   - Each superblock includes 16 scales of 8 bits and a global FP16 scale (d).
//   - Reconstructed as: weight = d * scale * (qs[i] - 32).
//
// Ideal for critical model layers (such as the embedding matrix or certain attention
// projection weights) where maximum semantic reasoning and vocabulary precision are required.
// ----------------------------------------------------------------------------
void gemv_q6_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
    const block_q6_K* blocks = reinterpret_cast<const block_q6_K*>(matrix_weights);
    int super_blocks_per_row = num_cols / QK_K;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_rows; ++r) {
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sum4 = 0.0f;
        int row_block_offset = r * super_blocks_per_row;

        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q6_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            int x_offset = sb * QK_K;

            const uint8_t* ql = bloco.ql;
            const uint8_t* qh = bloco.qh;
            const int8_t* sc = bloco.scales;

            for (int n = 0; n < QK_K; n += 128) {
                // First half of the loop (l = 0..15, is = 0)
                float s0 = d * sc[0];
                float s2 = d * sc[2];
                float s4 = d * sc[4];
                float s6 = d * sc[6];

                for (int l = 0; l < 16; ++l) {
                    // Reconstruct 6-bit signed integers (offset -32)
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    sum1 += s0 * q1 * x[x_offset + n + l +  0];
                    sum2 += s2 * q2 * x[x_offset + n + l + 32];
                    sum3 += s4 * q3 * x[x_offset + n + l + 64];
                    sum4 += s6 * q4 * x[x_offset + n + l + 96];
                }

                // Segunda metade do loop (l = 16..31, is = 1)
                float s1 = d * sc[1];
                float s3 = d * sc[3];
                float s5 = d * sc[5];
                float s7 = d * sc[7];

                for (int l = 16; l < 32; ++l) {
                    // Reconstroi os 6-bits signed inteiros (offset -32)
                    int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    sum1 += s1 * q1 * x[x_offset + n + l +  0];
                    sum2 += s3 * q2 * x[x_offset + n + l + 32];
                    sum3 += s5 * q3 * x[x_offset + n + l + 64];
                    sum4 += s7 * q4 * x[x_offset + n + l + 96];
                }

                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        out[r] = sum1 + sum2 + sum3 + sum4;
    }
}

// Dispatcher para GEMV Q6_K (Seleciona automaticamente AVX2 se disponível)
void gemv_q6_K(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols) {
#if defined(__AVX2__)
    gemv_q6_K_avx2(matrix_weights, x, out, num_rows, num_cols);
#else
    gemv_q6_K_scalar(matrix_weights, x, out, num_rows, num_cols);
#endif
}

// ============================================================================
// 3. NEURAL NETWORK OPERATORS & DISPATCHERS
//
// This section contains high-level dispatcher functions for general matrix
// multiplications (matmul) and highly optimized specialized kernels.
// To reduce computational overhead, inputs are quantized to Q8_K before launching
// core weight-matrix operations.
// ============================================================================

// ----------------------------------------------------------------------------
// MATMUL (Universal GEMV Dispatcher)
//
// Automatically routes execution to the corresponding low-level optimized kernel
// based on the Tensor's quantization type (F32, Q8_0, Q4_K, or Q6_K).
// ----------------------------------------------------------------------------
void matmul(std::vector<float>& output, const std::vector<float>& input, const Tensor& tensor) {
    uint64_t num_cols = tensor.dims[0];
    uint64_t num_rows = tensor.dims.size() > 1 ? tensor.dims[1] : 1;
    
    // Adjust output size if necessary
    if (output.size() != num_rows) {
        output.resize(num_rows);
    }

    if (tensor.type_str == "F32") {
        gemv_f32(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q8_0") {
        gemv_q8_0(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q4_K") {
        gemv_q4_K(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q6_K") {
        gemv_q6_K(tensor.data_ptr, input.data(), output.data(), num_rows, num_cols);
    } else {
        // Fallback for unimplemented tensor types
        static bool warned = false;
        if (!warned) {
            std::cout << "[WARNING] matmul kernel for " << tensor.type_str << " not implemented yet. Using fallback values." << std::endl;
            warned = true;
        }
        std::fill(output.begin(), output.end(), 0.001f);
    }
}

// ----------------------------------------------------------------------------
// QUANTIZE_ROW_Q8_K (FP32 to Q8_K Row-Wise Quantization)
//
// Quantizes a row of 32-bit floats (FP32) into a signed 8-bit block-wise format (Q8_K)
// on-the-fly. This is typically applied to the input vector of a layer before
// multiplying it with low-bit weight matrices.
// Quantizing the input reduces memory traffic during dot-product accumulation.
// ----------------------------------------------------------------------------
void quantize_row_q8_K(const float* x, int n, block_q8_K* y) {
#if defined(__AVX2__)
    quantize_row_q8_K_avx2(x, y, n);
#elif defined(__ARM_NEON)
    quantize_row_q8_K_neon(x, y, n);
#else
    quantize_row_q8_K_scalar(x, y, n);
#endif
}

// ----------------------------------------------------------------------------
// MATMUL_Q8K (Quantized Input GEMV)
//
// Performs matrix-vector multiplication directly using a pre-quantized Q8_K input vector.
// This allows the engine to reuse the same quantized input across multiple weight projections,
// eliminating redundant quantization steps and boosting overall performance.
// ----------------------------------------------------------------------------
void matmul_q8k(std::vector<float>& output, const block_q8_K* input_q8k, int num_cols, const Tensor& tensor) {
    uint64_t num_rows = tensor.dims.size() > 1 ? tensor.dims[1] : 1;
    if (output.size() != num_rows) {
        output.resize(num_rows);
    }
#if defined(__AVX2__)
    if (tensor.type_str == "Q4_K") {
        gemv_q4_K_q8_K_avx2(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q6_K") {
        gemv_q6_K_q8_K_avx2(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    }
#elif defined(__ARM_NEON)
    if (tensor.type_str == "Q4_K") {
        gemv_q4_K_q8_K_neon(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q6_K") {
        gemv_q6_K_q8_K_neon(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    }
#else
    if (tensor.type_str == "Q4_K") {
        gemv_q4_K_q8_K_scalar(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    } else if (tensor.type_str == "Q6_K") {
        gemv_q6_K_q8_K_scalar(tensor.data_ptr, input_q8k, output.data(), num_rows, num_cols);
    }
#endif
    else {
        static bool warned = false;
        if (!warned) {
            std::cout << "[WARNING] matmul_q8k: tensor type " << tensor.type_str << " not supported." << std::endl;
            warned = true;
        }
        std::fill(output.begin(), output.end(), 0.0f);
    }
}

// ----------------------------------------------------------------------------
// MATMUL_QKV_Q8K (Fused QKV GEMV Projection)
//
// In Transformer self-attention, the input vector must be projected into Query (Q),
// Key (K), and Value (V) spaces. Instead of spawning separate OpenMP regions for 
// each projection—which incurs substantial thread fork/join overhead—this fused kernel
// combines Q, K, and V projections into a single, unified parallel execution space.
// This greatly increases cache locality and thread efficiency on multi-core CPUs.
// ----------------------------------------------------------------------------
void matmul_qkv_q8k(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v,
                    const block_q8_K* input_q8k, int num_cols,
                    const Tensor& tensor_q, const Tensor& tensor_k, const Tensor& tensor_v) {
    const int q_rows = static_cast<int>(tensor_q.dims.size() > 1 ? tensor_q.dims[1] : 1);
    const int kv_rows = static_cast<int>(tensor_k.dims.size() > 1 ? tensor_k.dims[1] : 1);
    if (q.size() != q_rows) q.resize(q_rows);
    if (k.size() != kv_rows) k.resize(kv_rows);
    if (v.size() != kv_rows) v.resize(kv_rows);

#if defined(__AVX2__)
    if (tensor_v.type_str == "Q4_K") {
        gemv_qkv_q4_K_q8_K_avx2(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                 input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    } else if (tensor_v.type_str == "Q6_K") {
        gemv_qkv_q4_q4_q6_q8_K_avx2(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                     input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    }
#elif defined(__ARM_NEON)
    if (tensor_v.type_str == "Q4_K") {
        gemv_qkv_q4_K_q8_K_neon(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    } else if (tensor_v.type_str == "Q6_K") {
        gemv_qkv_q4_q4_q6_q8_K_neon(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                    input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    }
#else
    if (tensor_v.type_str == "Q4_K") {
        gemv_qkv_q4_K_q8_K_scalar(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                  input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    } else if (tensor_v.type_str == "Q6_K") {
        gemv_qkv_q4_q4_q6_q8_K_scalar(tensor_q.data_ptr, tensor_k.data_ptr, tensor_v.data_ptr,
                                      input_q8k, q.data(), k.data(), v.data(), q_rows, kv_rows, num_cols);
    }
#endif
    else {
        static bool warned = false;
        if (!warned) {
            std::cout << "[WARNING] matmul_qkv_q8k: tensor type v=" << tensor_v.type_str << " not supported." << std::endl;
            warned = true;
        }
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
    }
}

// ----------------------------------------------------------------------------
// MATMUL_GATE_UP_Q8K (Fused Gate and Up GEMV Projection)
//
// Similarly to the QKV projection, this fuses the computation of the Gate and
// Up projections of the SwiGLU block.
// ----------------------------------------------------------------------------
void matmul_gate_up_q8k(std::vector<float>& gate, std::vector<float>& up,
                        const block_q8_K* input_q8k, int num_cols,
                        const Tensor& tensor_gate, const Tensor& tensor_up) {
    const int gate_rows = static_cast<int>(tensor_gate.dims.size() > 1 ? tensor_gate.dims[1] : 1);
    const int up_rows = static_cast<int>(tensor_up.dims.size() > 1 ? tensor_up.dims[1] : 1);
    if (gate.size() != gate_rows) gate.resize(gate_rows);
    if (up.size() != up_rows) up.resize(up_rows);

#if defined(__AVX2__)
    gemv_gate_up_q4_K_q8_K_avx2(tensor_gate.data_ptr, tensor_up.data_ptr,
                                 input_q8k, gate.data(), up.data(), gate_rows, num_cols);
#elif defined(__ARM_NEON)
    gemv_gate_up_q4_K_q8_K_neon(tensor_gate.data_ptr, tensor_up.data_ptr,
                                input_q8k, gate.data(), up.data(), gate_rows, num_cols);
#else
    gemv_gate_up_q4_K_q8_K_scalar(tensor_gate.data_ptr, tensor_up.data_ptr,
                                  input_q8k, gate.data(), up.data(), gate_rows, num_cols);
#endif
}

// ============================================================================
// ACTIVATION AND NORMALIZATION FUNCTIONS (THEORY & IMPLEMENTAION)
// ============================================================================

// ----------------------------------------------------------------------------
// SILU (Sigmoid Linear Unit / Swish)
//
// The SiLU activation function is mathematically defined as:
//   SiLU(x) = x * sigmoid(x) = x / (1 + e^-x)
//
// Unlike the conventional ReLU (which zeros out negative values), SiLU is a
// smooth curve that allows a small gradient flow for negative values (non-monotonicity).
// This smoothness aids gradient flow during the training of deep networks,
// and forms the basis of the SwiGLU activation used in modern LLMs like LLaMA and MiniCPM.
// ----------------------------------------------------------------------------
float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// ----------------------------------------------------------------------------
// RMSNORM (Root Mean Square Layer Normalization)
//
// RMSNorm is a computationally simplified alternative to traditional LayerNorm.
// While LayerNorm centers and scales activations using both mean and variance
// (requiring two passes over the data):
//   LN(x) = (x - mean) / sqrt(variance + eps) * weight
//
// RMSNorm assumes that centering by the mean is unnecessary for training stability
// and focuses solely on regulating the scale through the "Root Mean Square"
// (the square root of the mean of squares), which requires only a single pass over memory:
//   RMS(x) = sqrt( (1 / N) * sum(x_i^2) + eps )
//   RMSNorm(x)_i = (x_i / RMS(x)) * weight_i
//
// This reduces memory bandwidth consumption and CPU clock cycles while preserving
// the same or better convergence stability in large-scale models.
// ----------------------------------------------------------------------------
void rmsnorm(std::vector<float>& out, const std::vector<float>& x, const Tensor& weight_tensor, float eps) {
    const float* w = reinterpret_cast<const float*>(weight_tensor.data_ptr);
    int dim = x.size();
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt(sum / dim + eps);
    if (out.size() != dim) out.resize(dim);
    
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * rms * w[i];
    }
}

// ----------------------------------------------------------------------------
// ROPE (Rotary Position Embeddings)
//
// RoPE is an innovative relative position encoding technique applied directly
// to the Query (Q) and Key (K) vectors of the attention mechanism.
//
// Instead of adding absolute positional embeddings to the input embeddings,
// RoPE splits the dimension of each head (head_dim) into 2D pairs and rotates
// each pair in the complex plane by an angle proportional to the token's position (pos)
// and the dimension frequency:
//   [v_0, v_1] rotated by m * theta:
//   v_0' = v_0 * cos(m*theta) - v_1 * sin(m*theta)
//   v_1' = v_0 * sin(m*theta) + v_1 * cos(m*theta)
//
// Where m = pos, and theta = base^(-2i / head_dim).
// This rotation mathematically ensures that the dot product between Query and Key
// (Q * K_T) depends purely on the relative distance between the two tokens:
//   score(q_m, k_n) = f(q, m)^T * f(k, n) = g(q, k, m - n)
// ----------------------------------------------------------------------------
void apply_rope(std::vector<float>& vec, int pos, int head_idx, int head_dim, float rope_base) {
    for (int i = 0; i < head_dim; i += 2) {
        int idx = head_idx * head_dim + i;
        float theta = 1.0f / std::pow(rope_base, (float)i / head_dim);
        float m_theta = pos * theta;
        float cos_val = std::cos(m_theta);
        float sin_val = std::sin(m_theta);
        
        float v0 = vec[idx];
        float v1 = vec[idx + 1];
        
        vec[idx]     = v0 * cos_val - v1 * sin_val;
        vec[idx + 1] = v0 * sin_val + v1 * cos_val;
    }
}

// ----------------------------------------------------------------------------
// GQA (Grouped-Query Attention) WITH KV CACHE
//
// The traditional Multi-Head Attention (MHA) mechanism pairs a Key (K) and
// Value (V) head for every Query (Q) head. This creates a massive KV Cache
// that consumes significant RAM and limits the maximum context size.
//
// Grouped-Query Attention (GQA) groups multiple Query heads to share a single
// Key/Value head. For example, in an 8:1 ratio (as in MiniCPM), each K and V
// head is shared by 8 Q heads.
// GQA offers almost the same accuracy as traditional MHA, but with a tiny
// fraction of the memory footprint and drastically superior decoding speeds.
//
// This function performs:
//   1. Incremental caching of K and V into the layer's dynamic KV Cache buffer.
//   2. Computation of scaled Attention Scores: (Q * K^T) / sqrt(head_dim).
//   3. Numerically stable Softmax using maximum value subtraction.
//   4. Weighted projection (Scores * V) to produce final attention activations.
// ----------------------------------------------------------------------------
void execute_attention(std::vector<float>& attn_out, 
                       const std::vector<float>& q, 
                       const std::vector<float>& k_curr, 
                       const std::vector<float>& v_curr,
                       KVCacheLayer& kv_cache,
                       int pos,
                       int max_seq_len,
                       int num_heads, int num_kv_heads, int head_dim) {
    
    if (attn_out.size() != num_heads * head_dim) {
        attn_out.resize(num_heads * head_dim);
    }

    if (pos >= max_seq_len) {
        std::cerr << "[WARNING] Maximum context exceeded (pos >= max_seq_len)." << std::endl;
        return;
    }

    // 1. Save current Key and Value to the layer's KV Cache
    // The KV Cache avoids redundant computations by storing previously calculated key and value vectors.
    int kv_size_per_token = num_kv_heads * head_dim;
    int kv_offset = pos * kv_size_per_token;
    
    for (int i = 0; i < kv_size_per_token; ++i) {
        kv_cache.k[kv_offset + i] = k_curr[i];
        kv_cache.v[kv_offset + i] = v_curr[i];
    }

    // 2. Compute Scaled Attention: Softmax(Q * K_T) * V
    // Scaling stabilizes dot-product magnitudes, preventing vanishing gradients during softmax backpropagation.
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int num_queries_per_kv = num_heads / num_kv_heads;

    std::vector<float> all_scores(num_heads * (pos + 1));

    for (int h = 0; h < num_heads; ++h) {
        // GQA Head Mapping: identify which Key/Value head is shared by this Query head
        int kv_h = h / num_queries_per_kv;
        const float* q_h = q.data() + h * head_dim;
        
        float* scores = all_scores.data() + h * (pos + 1);
        float max_score = -1e9f;

        // Q * K_T (Dot Product with all historical and current token keys)
        for (int t = 0; t <= pos; ++t) {
            const float* k_h_t = kv_cache.k.data() + (t * kv_size_per_token) + (kv_h * head_dim);
            
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) {
                dot += q_h[i] * k_h_t[i];
            }
            dot *= scale;
            
            scores[t] = dot;
            // Track the maximum score for numerical stability in Softmax exponentiation
            if (dot > max_score) {
                max_score = dot;
            }
        }

        // Softmax normalization: exp(x - max_x) / sum(exp(x - max_x))
        // Subtracting max_score prevents floating-point overflow for large positive dot products.
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t <= pos; ++t) {
            scores[t] /= sum_exp;
        }

        // Output = Softmax_Scores * V (Weighted sum of token value vectors)
        float* out_h = attn_out.data() + h * head_dim;

        // Initialize directly with t = 0 values to avoid redundant std::fill zeroing calls
        const float* v_h_0 = kv_cache.v.data() + (0 * kv_size_per_token) + (kv_h * head_dim);
        float score_0 = scores[0];
        for (int i = 0; i < head_dim; ++i) {
            out_h[i] = score_0 * v_h_0[i];
        }

        // Accumulate remaining historical token value contributions
        for (int t = 1; t <= pos; ++t) {
            const float* v_h_t = kv_cache.v.data() + (t * kv_size_per_token) + (kv_h * head_dim);
            float score = scores[t];
            
            for (int i = 0; i < head_dim; ++i) {
                out_h[i] += score * v_h_t[i];
            }
        }
    }
}

// Extração de Embeddings baseada no formato do Tensor
void get_embedding(std::vector<float>& out, const Tensor& embd_tensor, int token_id) {
    int dim = embd_tensor.dims[0];
    if (out.size() != dim) out.resize(dim);

    if (embd_tensor.type_str == "Q4_K") {
        const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(embd_tensor.data_ptr);
        int super_blocks_per_row = dim / QK_K;
        
        // Pula para os super-blocos do token_id solicitado
        int row_block_offset = token_id * super_blocks_per_row;
        
        // Esta operação lê uma única linha (2048 elementos / 8 blocos), então OpenMP não é tão necessário aqui
        for (int sb = 0; sb < super_blocks_per_row; ++sb) {
            const block_q4_K& bloco = blocks[row_block_offset + sb];
            float d = fp16_to_fp32(bloco.d);
            float dmin = fp16_to_fp32(bloco.dmin);
            int x_offset = sb * QK_K;
            uint8_t scales[8], mins[8];
            decode_q4k_scales_mins(bloco.scales, scales, mins);
            const uint8_t* q = bloco.qs;

            // Processa em chunks de 64 (mesma matemática da dequantização q4_k)
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int j = chunk * 64;
                float d1 = d * static_cast<float>(scales[2 * chunk + 0]);
                float m1 = dmin * static_cast<float>(mins[2 * chunk + 0]);
                float d2 = d * static_cast<float>(scales[2 * chunk + 1]);
                float m2 = dmin * static_cast<float>(mins[2 * chunk + 1]);

                for (int l = 0; l < 32; ++l) {
                    float w1 = d1 * (q[l] & 0xF) - m1;
                    float w2 = d2 * (q[l] >> 4) - m2;
                    out[x_offset + j + l] = w1;
                    out[x_offset + j + l + 32] = w2;
                }
                q += 32;
            }
        }
    } else {
        std::cerr << "[AVISO] get_embedding so foi implementado para Q4_K." << std::endl;
        std::fill(out.begin(), out.end(), 0.0f);
    }
}

// Operações fundidas de alta performance para otimização de sincronização OpenMP e localidade de cache
void rmsnorm_and_quantize_q8k(block_q8_K* out_q8k, const std::vector<float>& x, const Tensor& weight_tensor, float eps) {
    int dim = x.size();
    const float* w = reinterpret_cast<const float*>(weight_tensor.data_ptr);

    // Compute sum of squares
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    float rms = 1.0f / std::sqrt(sum / dim + eps);

    int num_blocks = dim / QK_K;
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < num_blocks; ib++) {
        const float* xb = x.data() + ib * QK_K;
        const float* wb = w + ib * QK_K;
        block_q8_K& dst = out_q8k[ib];

        // 1. Compute rmsnorm on-the-fly for 256 elements
        float local_x[QK_K];
        float amax = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            local_x[i] = xb[i] * rms * wb[i];
            amax = std::max(amax, std::fabs(local_x[i]));
        }

        // 2. Quantize on-the-fly
        if (amax == 0.0f) {
            dst.d = 0.0f;
            std::memset(dst.qs, 0, QK_K);
            std::memset(dst.bsums, 0, sizeof(dst.bsums));
            continue;
        }

        dst.d = amax / 127.0f;
        float mul = 127.0f / amax;

        int32_t bsums[QK_K / 16] = {0};
        for (int i = 0; i < QK_K; ++i) {
            dst.qs[i] = (int8_t)std::max(-127.0f, std::min(127.0f, (float)std::lround(local_x[i] * mul)));
            bsums[i / 16] += dst.qs[i];
        }
        for (int g = 0; g < QK_K / 16; ++g) {
            dst.bsums[g] = (int16_t)bsums[g];
        }
    }
}

void swiglu_and_quantize_q8k(block_q8_K* out_q8k, const std::vector<float>& gate, const std::vector<float>& up) {
    int dim = gate.size();
    int num_blocks = dim / QK_K;

    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < num_blocks; ib++) {
        const float* gb = gate.data() + ib * QK_K;
        const float* ub = up.data() + ib * QK_K;
        block_q8_K& dst = out_q8k[ib];

        // 1. SwiGLU on-the-fly
        float local_x[QK_K];
        float amax = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            float g = gb[i];
            float silu_g = g / (1.0f + std::exp(-g));
            local_x[i] = silu_g * ub[i];
            amax = std::max(amax, std::fabs(local_x[i]));
        }

        // 2. Quantize on-the-fly
        if (amax == 0.0f) {
            dst.d = 0.0f;
            std::memset(dst.qs, 0, QK_K);
            std::memset(dst.bsums, 0, sizeof(dst.bsums));
            continue;
        }

        dst.d = amax / 127.0f;
        float mul = 127.0f / amax;

        int32_t bsums[QK_K / 16] = {0};
        for (int i = 0; i < QK_K; ++i) {
            dst.qs[i] = (int8_t)std::max(-127.0f, std::min(127.0f, (float)std::lround(local_x[i] * mul)));
            bsums[i / 16] += dst.qs[i];
        }
        for (int g = 0; g < QK_K / 16; ++g) {
            dst.bsums[g] = (int16_t)bsums[g];
        }
    }
}
