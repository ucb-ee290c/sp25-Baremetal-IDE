/*
 * bench_config.h - Configuration for matmul-rvv-ope benchmark.
 */
#ifndef MATMUL_BENCH_CONFIG_H
#define MATMUL_BENCH_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef MATMUL_BENCH_TARGET_FREQUENCY_HZ
#define MATMUL_BENCH_TARGET_FREQUENCY_HZ 500000000ULL
#endif

#ifndef MATMUL_BENCH_RUNS_COLD
#define MATMUL_BENCH_RUNS_COLD 10
#endif

#ifndef MATMUL_BENCH_RUNS_HOT
#define MATMUL_BENCH_RUNS_HOT 20
#endif

// Approximate L2 cache size for flush-thrash helper
#ifndef MATMUL_L2_BYTES
#define MATMUL_L2_BYTES (256u * 1024u)
#endif

// Fixed matmul dimensions (60 = 4 x 15-row tiles, clean tiling for RVV+OPE interleave)
#define MATMUL_M 60
#define MATMUL_N 60
#define MATMUL_K 60

// Interleaved tile structure
#define MATMUL_RVV_ROWS    7   // rows computed by RVV per tile (rows 0-6)
#define MATMUL_OPE_ROWS    8   // rows computed by OPE per tile (rows 7-14)
#define MATMUL_TILE_ROWS   15  // MATMUL_RVV_ROWS + MATMUL_OPE_ROWS

// OPE works on 8-column tiles; ceil(N/8)
#define MATMUL_N_OPE_TILES ((MATMUL_N + 7) / 8)

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif // MATMUL_BENCH_CONFIG_H
