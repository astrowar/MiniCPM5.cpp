#pragma once
#include <cstdint>

typedef uint16_t ggml_fp16_t;

// Conversor portátil de Half-Float (16-bits) para Float padrão (32-bits)
static inline float fp16_to_fp32(ggml_fp16_t h) {
    union { uint32_t u; float f; } o;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;
    if (exp == 0x1F) {
        o.u = sign | 0x7F800000 | mant;
    } else if (exp == 0) {
        if (mant == 0) o.u = sign;
        else {
            while (!(mant & 0x00800000)) { mant <<= 1; exp--; }
            o.u = sign | ((exp + 113) << 23) | (mant & 0x007FFFFF);
        }
    } else {
        o.u = sign | ((exp + 112) << 23) | mant;
    }
    return o.f;
}

// Estrutura física Q8_0 (34 bytes)
#define QK8_0 32
struct block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[QK8_0];
};

// Estrutura física Q4_K (144 bytes)
#define QK_K 256
#define K_SCALE_SIZE 12
struct block_q4_K {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

// Estrutura física Q6_K (210 bytes)
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // 128 bytes (lower 4 bits)
    uint8_t qh[QK_K / 4];      // 64 bytes  (upper 2 bits)
    int8_t  scales[QK_K / 16]; // 16 bytes  (sub-block scales)
    ggml_fp16_t d;             // 2 bytes   (super-block scale)
};

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// Assinaturas das implementações de kernels de multiplicação de matrizes
void gemv_q4_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);
void gemv_q4_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);

void gemv_q6_K_scalar(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);
void gemv_q6_K_avx2(const char* matrix_weights, const float* x, float* out, int num_rows, int num_cols);
