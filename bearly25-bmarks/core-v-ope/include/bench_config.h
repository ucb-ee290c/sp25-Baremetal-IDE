/*
 * bench_config.h - Configuration and shared types for core-v-ope benchmarks.
*/

#ifndef CORE_V_OPE_BENCH_CONFIG_H
#define CORE_V_OPE_BENCH_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <limits.h>

#include "hal_ope.h"

// L2 size – 256 KiB, 2 banks
#ifndef OPE_L2_BYTES
#define OPE_L2_BYTES (256 * 1024)
#endif

// How many repetitions per matrix size
#ifndef BENCH_RUNS
#define BENCH_RUNS 5
#endif

#ifndef BENCH_ENABLE_OPE
#define BENCH_ENABLE_OPE 1
#endif

#ifndef BENCH_ENABLE_VEC
#define BENCH_ENABLE_VEC 1
#endif

#ifndef BENCH_HAS_VECNN
#define BENCH_HAS_VECNN 0
#endif

#ifndef BENCH_VERIFY
#define BENCH_VERIFY 1
#endif

typedef struct {
  const char *name;
  int M;
  int N;
  int K;
} OuterSizeCase;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif // CORE_V_OPE_BENCH_CONFIG_H
