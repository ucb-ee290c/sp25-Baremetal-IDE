/*
 * bench_config.h - Shared configuration and types for OPE microbenchmarks.
 *
 * Defines cache sizing, run counts, implementation toggles, and helpers.
 */
#ifndef BENCH_CONFIG_H
#define BENCH_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <limits.h>

#include "hal_ope.h"

// L2 size – 256 KiB, 2 banks
#ifndef OPE_L2_BYTES
#define OPE_L2_BYTES (256 * 1024)
#endif

// How many repetitions per matrix size for each cache regime
#ifndef BENCH_RUNS_COLD
#define BENCH_RUNS_COLD 3
#endif

#ifndef BENCH_RUNS_HOT
#define BENCH_RUNS_HOT 5
#endif

// Toggle which implementation families we actually run
#ifndef BENCH_ENABLE_IMPL_ARB
#define BENCH_ENABLE_IMPL_ARB 1
#endif

#ifndef BENCH_ENABLE_IMPL_SQUARE
#define BENCH_ENABLE_IMPL_SQUARE 1
#endif

typedef struct {
  const char *name;
  int M;
  int N;
  int K;
} OpeSizeCase;

typedef enum {
  OPE_IMPL_ARB = 0,
  OPE_IMPL_SQUARE,
} ope_impl_kind_t;

static inline const char *ope_impl_kind_name(ope_impl_kind_t kind) {
  switch (kind) {
    case OPE_IMPL_ARB: return "arb";
    case OPE_IMPL_SQUARE: return "square";
    default: return "unknown";
  }
}

typedef struct {
  long cycles_ope;
  long cycles_total;
} ope_bench_run_t;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif // BENCH_CONFIG_H
