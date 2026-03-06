/*
 * bench_run.c - Core benchmark loop for i8->i32 matmul.
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_cache.h"
#include "bench_config.h"
#include "bench_kernel.h"

typedef struct {
  uint64_t sum;
  uint64_t best;
  int runs;
} bench_stats_t;

static inline void bench_stats_init(bench_stats_t *stats) {
  stats->sum = 0;
  stats->best = ULLONG_MAX;
  stats->runs = 0;
}

static inline void bench_stats_update(bench_stats_t *stats, uint64_t cycles) {
  stats->sum += cycles;
  if (cycles < stats->best) {
    stats->best = cycles;
  }
  stats->runs += 1;
}

static inline uint64_t bench_stats_avg(const bench_stats_t *stats) {
  if (stats->runs == 0) {
    return 0;
  }
  return stats->sum / (uint64_t)stats->runs;
}

static inline size_t round_up(size_t size, size_t alignment) {
  return ((size + alignment - 1u) / alignment) * alignment;
}

static void *bench_aligned_alloc(size_t alignment, size_t size) {
  if (size == 0) {
    return NULL;
  }
  return aligned_alloc(alignment, round_up(size, alignment));
}

void bench_run(void) {
  const size_t M = MATMUL_M;
  const size_t N = MATMUL_N;
  const size_t K = MATMUL_K;

  const size_t N_ope_tiles = MATMUL_N_OPE_TILES;  // ceil(N/8)

  // Number of 15-row tiles (each tile has 8 OPE rows at offsets 7-14)
  const size_t N_15row_tiles = (M + 14) / 15;

  int8_t  *A     = (int8_t  *)bench_aligned_alloc(8, M * K * sizeof(int8_t));
  int8_t  *A_T   = (int8_t  *)bench_aligned_alloc(8, K * M * sizeof(int8_t));
  int8_t  *B     = (int8_t  *)bench_aligned_alloc(8, K * N * sizeof(int8_t));
  // B_ope: [N_ope_tiles x K x 8], remapped B for OPE 8-column tiles.
  // B_ope[j*K*8 + k*8 + e] = B[k*N + j*8 + e], zero-padded if out of bounds.
  int8_t  *B_ope = (int8_t  *)bench_aligned_alloc(8, N_ope_tiles * K * 8 * sizeof(int8_t));
  // A_ope: [N_15row_tiles x K x 8], OPE-remapped A for rows 7-14 of each 15-row tile.
  // A_ope[t*K*8 + k*8 + r] = A_T[k*M + t*15 + 7 + r], zero-padded if out of bounds.
  int8_t  *A_ope = (int8_t  *)bench_aligned_alloc(8, N_15row_tiles * K * 8 * sizeof(int8_t));
  int32_t *C     = (int32_t *)bench_aligned_alloc(8, M * N * sizeof(int32_t));

  if (!A || !A_T || !B || !B_ope || !A_ope || !C) {
    printf("  ERROR: allocation failed\n");
    free(A); free(A_T); free(B); free(B_ope); free(A_ope); free(C);
    return;
  }

  for (size_t i = 0; i < M * K; ++i) {
    A[i] = (int8_t)(((i * 13u + 7u) % 127u) - 63);
  }
  for (size_t i = 0; i < K * N; ++i) {
    B[i] = (int8_t)(((i * 11u + 3u) % 127u) - 63);
  }

  // Transpose A [M x K] -> A_T [K x M]: column i of A_T = row i of A.
  // A_T[k*M + i] = A[i*K + k].  Within each k-row, bytes 0-6 go to RVV,
  // bytes 7-14 go to OPE (adjacent => share the same cache line).
  for (size_t i = 0; i < M; ++i) {
    for (size_t k = 0; k < K; ++k) {
      A_T[k * M + i] = A[i * K + k];
    }
  }

  // Remap B for OPE: group into [N_ope_tiles][K][8] layout.
  // For tile j, k-step k: 8 consecutive bytes from row k of B starting at col j*8.
  memset(B_ope, 0, N_ope_tiles * K * 8 * sizeof(int8_t));
  for (size_t j = 0; j < N_ope_tiles; ++j) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t e = 0; e < 8; ++e) {
        size_t col = j * 8 + e;
        if (col < N) {
          B_ope[j * K * 8 + k * 8 + e] = B[k * N + col];
        }
      }
    }
  }

  // Remap A for OPE: group into [N_15row_tiles][K][8] layout.
  // For 15-row tile t, OPE handles rows t*15+7 .. t*15+14 (8 rows).
  // A_ope[t*K*8 + k*8 + r] = A_T[k*M + t*15 + 7 + r]
  memset(A_ope, 0, N_15row_tiles * K * 8 * sizeof(int8_t));
  for (size_t t = 0; t < N_15row_tiles; ++t) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t r = 0; r < 8; ++r) {
        size_t row_in_M = t * 15 + 7 + r;
        if (row_in_M < M) {
          A_ope[t * K * 8 + k * 8 + r] = A_T[k * M + row_in_M];
        }
      }
    }
  }

  memset(C, 0, M * N * sizeof(int32_t));

  bench_stats_t cold, hot;
  bench_stats_init(&cold);
  bench_stats_init(&hot);

  // --- Baseline: pure RVV kernel ---
  for (int r = 0; r < MATMUL_BENCH_RUNS_COLD; ++r) {
    memset(C, 0, M * N * sizeof(int32_t));
    uint64_t t0 = rdcycle64();
    i8_i32_matmul(M, N, K, A_T, M, B, C, N, 1);
    uint64_t t1 = rdcycle64();
    bench_stats_update(&cold, t1 - t0);
  }
  i8_i32_matmul(M, N, K, A_T, M, B, C, N, 1);
  for (int r = 0; r < MATMUL_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    i8_i32_matmul(M, N, K, A_T, M, B, C, N, 1);
    uint64_t t1 = rdcycle64();
    bench_stats_update(&hot, t1 - t0);
  }
  printf("  %-28s COLD(runs=%d best=%llu avg=%llu) HOT(runs=%d best=%llu avg=%llu)\n",
         "i8_i32_matmul",
         cold.runs, (unsigned long long)cold.best, (unsigned long long)bench_stats_avg(&cold),
         hot.runs,  (unsigned long long)hot.best,  (unsigned long long)bench_stats_avg(&hot));

  // --- Interleaved RVV+OPE kernel ---
  bench_stats_init(&cold);
  bench_stats_init(&hot);
  for (int r = 0; r < MATMUL_BENCH_RUNS_COLD; ++r) {
    memset(C, 0, M * N * sizeof(int32_t));
    uint64_t t0 = rdcycle64();
    i8_i32_matmul_interleaved(M, N, K, A_T, M, B, B_ope, N_ope_tiles, A_ope, C, N);
    uint64_t t1 = rdcycle64();
    bench_stats_update(&cold, t1 - t0);
  }
  i8_i32_matmul_interleaved(M, N, K, A_T, M, B, B_ope, N_ope_tiles, A_ope, C, N);
  for (int r = 0; r < MATMUL_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    i8_i32_matmul_interleaved(M, N, K, A_T, M, B, B_ope, N_ope_tiles, A_ope, C, N);
    uint64_t t1 = rdcycle64();
    bench_stats_update(&hot, t1 - t0);
  }
  printf("  %-28s COLD(runs=%d best=%llu avg=%llu) HOT(runs=%d best=%llu avg=%llu)\n",
         "i8_i32_matmul_interleaved",
         cold.runs, (unsigned long long)cold.best, (unsigned long long)bench_stats_avg(&cold),
         hot.runs,  (unsigned long long)hot.best,  (unsigned long long)bench_stats_avg(&hot));

  free(A);
  free(A_T);
  free(B);
  free(B_ope);
  free(A_ope);
  free(C);
}
