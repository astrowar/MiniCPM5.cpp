# Paralelização — Plano de Ataque

## CPU

| Campo | Valor |
|-------|-------|
| Processador | AMD Ryzen 7 PRO 4750U |
| Cores físicas | 8 |
| Threads lógicas (SMT) | 16 |
| Arquitetura SIMD | AVX2 + F16C |
| L2 | 512 KB / core |
| L3 | 8 MB compartilhado |

## Baseline Single-Thread (sem OpenMP)

```
Prompt: "Explain photosynthesis in detail"  |  128 tokens  |  --no-think
Throughput: 3.85 tok/s  (≈ 260 ms/token)
```

## Hot Path — Tempo e % por Seção

Tabela com tempos relativos do profiler (escala relativa válida; absolutos têm bug de divisor).

| Seção | Tempo (us) | % do token | Rows × Cols | Kernel |
|-------|-----------|------------|-------------|--------|
| **gate_up_fused** | **810.5** | **38.0%** | 6144 × 2048 | Q4_K×Q8_K combined dot |
| **down_proj** | **569.2** | **26.7%** | 2048 × 6144 | Q4_K/Q6_K×Q8_K |
| **lm_head** | **357.1** | **16.8%** | 130560 × 2048 | Q6_K×Q8_K |
| qkv_fused | 171.9 | 8.1% | 2560 × 2048 | Q4_K/Q6_K×Q8_K |
| o_proj | 127.2 | 6.0% | 2048 × 2048 | F32 (path legado) |
| attention | 61.3 | 2.9% | 16 heads × seq_len | FP32 scalar |
| rope | 15.7 | 0.7% | 18 × 64 pairs | FP32 trig |
| swiglu | 12.7 | 0.6% | 6144 elements | FP32 silu×mul |
| norms + residuals | ~2.5 | 0.1% | 2048 elements | FP32 |

### Totais

| Categoria | % |
|-----------|---|
| GEMV (paralelizável por row) | **95.6%** |
| Serial (norms, attention, rope, swiglu) | 4.4% |

## Teoria de Escala

### Amdahl (8 cores)

```
S = 1 / (0.044 + 0.956/8) = 1 / 0.1635 ≈ 6.1x
Previsão: 3.85 × 6.1 ≈ 23.5 tok/s
```

### Situação Atual (OpenMP per-operation)

```
Medido: 8.35 tok/s → speedup 2.17x
Perda vs teórico: ~65% do potencial
Causa: ~210 barrier create/destroy por token (42 layers × 5 ops)
```

### Alvo com Thread Pool Persistente

```
Overhead de barrier: ~10-20 μs cada
Custo total OpenMP/token: 210 × 15μs ≈ 3.1 ms
Custo thread pool (1 fence por op): 5 × 8 × 5μs ≈ 0.2 ms
Economia esperada: ~3 ms/token ≈ 1.2% do total
```

Na prática o ganho vem de evitar **spawn/teardown de threads** (cada `#pragma omp parallel for` no MSVC cria e destrói a team). Com pool persistente, threads ficam prontas esperando work.

## Plano de Paralelização (ordem por ROI)

### Prioridade 1: gate_up_fused — 38% do tempo

```
Rows: 6144  →  768 rows/thread (8 threads)
```
- Split estático: thread `t` processa rows `[t*768, (t+1)*768)`
- Cada row é independente (apenas lê xq e escreve gate[r]/up[r])
- **Zero dependências entre iterações**
- Grão ideal: 768 iterações × ~130μs/iteração ≈ 100ms de work por thread

### Prioridade 2: down_proj — 27%

```
Rows: 2048  →  256 rows/thread
```
- Mesmo padrão: split estático, rows independentes
- Grão menor que gate_up mas ainda suficiente (256 × ~275μs ≈ 70ms)

### Prioridade 3: lm_head — 17%

```
Rows: 130560  →  16320 rows/thread
```
- Maior paralelismo absoluto do modelo
- Grão enorme — balanceamento perfeito mesmo com cache misses
- Único ponto que beneficia de **work-stealing** (rows variam em custo se Q6_K tem different scales)

### Prioridade 4: qkv_fused — 8%

```
Rows: 2560  →  320 rows/thread
```
- if/else dispatch por faixa (q: 0-2047, k: 2048-2303, v: 2304-2559)
- Pode usar split estático simples OU dividir por faixa

### Prioridade 5: o_proj — 6%

```
Rows: 2048  →  256 rows/thread
```
- **Ação prévia:** migrar de `matmul()` (FP32 path) para `matmul_q8k`
  - Quantizar `attn_out_temp_` uma vez antes do GEMV
  - Salva ~1-2 us de quantização repetida por row
- Split estático como nos demais

### NÃO paralelizar (custo > benefício)

| Seção | % | Motivo |
|-------|---|--------|
| attention | 2.9% | Só 16 heads — grão pequeno, barrier > work |
| rope | 0.7% | 18 × 64 = 1152 floats — trivial |
| swiglu | 0.6% | 6144 elementos — trivial |
| norms/residuals | 0.1% | 2048 elementos — trivial |

## Arquitetura do Thread Pool

```
┌─────────────────────────────────────────────────────────┐
│  Thread Pool (8 workers, criados 1× no init)            │
│                                                         │
│  Worker 0 ─┐                                            │
│  Worker 1 ─┤  atomic counter / spin-wait                │
│  Worker 2 ─┤  para row assignment                       │
│  Worker 3 ─┤                                            │
│  Worker 4 ─┤                                            │
│  Worker 5 ─┤  fence (barrier) entre ops                 │
│  Worker 6 ─┤                                            │
│  Worker 7 ─┘                                            │
└─────────────────────────────────────────────────────────┘
```

### Interface

```cpp
struct ThreadPool {
    int nthreads;
    std::thread workers[8];
    // Sincronização
    std::atomic<int> phase{0};
    std::atomic<int> count{0};
    int current_op;  // qual função executam

    // Modelo de execução:
    // 1. Main thread seta current_op e phase++
    // 2. Workers acordam, executam op com divisão de rows
    // 3. Workers fazem barrier (spin-wait em phase)
    // 4. Main thread prossegue
    void submit(int op_id, int total_rows);
    void wait();  // fence: main espera todos terminarem
};
```

### Divisão de Work (2 estratégias)

**A) Split estático (simpler, suficiente para rows ≥ 256):**
```
Thread t processa rows [t * chunk, (t+1) * chunk)
```

**B) Dynamic chunk (para lm_head com 130K rows):**
```
Atomic counter shared: next_row
Loop: r = counter.fetch_add(1); while r < total: process(r); r = counter.fetch_add(1)
```

### Mapeamento por Seção

| Seção | Rows | Estratégia | Chunk |
|-------|------|-----------|-------|
| gate_up | 6144 | Estático | 768 |
| down_proj | 2048 | Estático | 256 |
| lm_head | 130560 | Dynamic (atomic) | 1 |
| qkv_fused | 2560 | Estático | 320 |
| o_proj | 2048 | Estático | 256 |

## Implementação — Passos

1. **Criar `thread_pool.h/.cpp`**
   - 8 threads, barrier por fase (sense-reversing ou phase counter)
   - `submit(op, rows)` + `wait()`
   - Inicializar 1× no `ModelWeights::load()`

2. **Adaptar kernels AVX2 para receber row range**
   ```cpp
   // Antes (OpenMP):
   #pragma omp parallel for
   for (int r = 0; r < num_rows; ++r) { ... }

   // Depois (pool):
   void gemv_q4_K_q8_K_range(const char* w, const block_q8_K* xq,
                              float* out, int r_start, int r_end, int num_cols)
   ```

3. **Dispatch no `forward()`**
   ```cpp
   pool.submit(OP_GATE_UP, MODEL_FFN_DIM);  // workers executam
   pool.wait();                              // fence
   ```

4. **Eliminar `#include <omp.h>` e flags CMake de OpenMP**
   - Substituir por `<thread>`, `<atomic>`, `<barrier>` (C++20) ou manual

5. **Benchmark progressivo**
   - Enable por seção (começar com gate_up → medir → down → lm_head → qkv → o_proj)
   - Alvo: >15 tok/s com 8 threads

## Métricas de Sucesso

| Milestone | tok/s alvo | O que habilita |
|-----------|-----------|----------------|
| Pool + gate_up | ~6.5 | 1 seção (38%) |
| + down_proj | ~9.0 | 2 seções (65%) |
| + lm_head | ~11.5 | 3 seções (82%) |
| + qkv + o_proj | ~14-16 | Todas as GEMVs (96%) |
| + attention (16 threads SMT) | ~15-17 | Marginal |

llama.cpp = 14 tok/s → **devemos superar com folga.**
