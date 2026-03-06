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
#define MATMUL_BENCH_RUNS_COLD 100
#endif

#ifndef MATMUL_BENCH_RUNS_HOT
#define MATMUL_BENCH_RUNS_HOT 200
#endif

// Approximate L2 cache size for flush-thrash helper
#ifndef MATMUL_L2_BYTES
#define MATMUL_L2_BYTES (256u * 1024u)
#endif

// Fixed matmul dimensions
#define MATMUL_M 64
#define MATMUL_N 64
#define MATMUL_K 64

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif // MATMUL_BENCH_CONFIG_H
