/*
 * bench_config.h - Shared configuration for RVV matmul benchmarks.
 */
#ifndef RVV_BENCH_CONFIG_H
#define RVV_BENCH_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef RVV_BENCH_TARGET_FREQUENCY_HZ
#define RVV_BENCH_TARGET_FREQUENCY_HZ 500000000ULL
#endif

// Repetitions for each cache state.
#ifndef RVV_BENCH_RUNS_COLD
#define RVV_BENCH_RUNS_COLD 5
#endif

#ifndef RVV_BENCH_RUNS_HOT
#define RVV_BENCH_RUNS_HOT 20
#endif

// Approximate L2 cache size for flush-thrash helper.
#ifndef RVV_L2_BYTES
#define RVV_L2_BYTES (256u * 1024u)
#endif

#ifndef RVV_BENCH_ENABLE_F32
#define RVV_BENCH_ENABLE_F32 1
#endif

#ifndef RVV_BENCH_ENABLE_I8_I16
#define RVV_BENCH_ENABLE_I8_I16 1
#endif

#ifndef RVV_BENCH_ENABLE_I8_I32
#define RVV_BENCH_ENABLE_I8_I32 1
#endif

#ifndef RVV_BENCH_ENABLE_UNPACKED
#define RVV_BENCH_ENABLE_UNPACKED 1
#endif

#ifndef RVV_BENCH_ENABLE_PACKED
#define RVV_BENCH_ENABLE_PACKED 1
#endif

typedef struct {
  const char *name;
  size_t M;
  size_t N;
  size_t K;
} RvvMatmulCase;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif // RVV_BENCH_CONFIG_H
