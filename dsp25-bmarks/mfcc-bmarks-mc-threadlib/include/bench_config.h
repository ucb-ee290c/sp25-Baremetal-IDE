#ifndef MFCC_BENCH_CONFIG_H
#define MFCC_BENCH_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef MFCC_BENCH_TARGET_FREQUENCY_HZ
#define MFCC_BENCH_TARGET_FREQUENCY_HZ 500000000ULL
#endif

// 0: single-frequency mode
// 1: iterate over MFCC_BENCH_PLL_FREQ_LIST
#ifndef MFCC_BENCH_ENABLE_PLL_SWEEP
#define MFCC_BENCH_ENABLE_PLL_SWEEP 0
#endif

#ifndef MFCC_BENCH_PLL_SWEEP_SLEEP_MS
#define MFCC_BENCH_PLL_SWEEP_SLEEP_MS 10000u
#endif

#ifndef MFCC_BENCH_PLL_FREQ_LIST
#define MFCC_BENCH_PLL_FREQ_LIST \
  50000000ULL, \
  150000000ULL, \
  250000000ULL, \
  350000000ULL
#endif

// Number of timed runs per cache mode per driver function.
#ifndef MFCC_BENCH_NUM_ITERATIONS
#define MFCC_BENCH_NUM_ITERATIONS 10
#endif

#ifndef MFCC_BENCH_RUN_COLD
#define MFCC_BENCH_RUN_COLD 1
#endif

#ifndef MFCC_BENCH_RUN_WARM
#define MFCC_BENCH_RUN_WARM 1
#endif

#ifndef MFCC_BENCH_L2_BYTES
#define MFCC_BENCH_L2_BYTES (256u * 1024u)
#endif

#ifndef MFCC_BENCH_ENABLE_F32
#define MFCC_BENCH_ENABLE_F32 1
#endif

#ifndef MFCC_BENCH_ENABLE_Q31
#define MFCC_BENCH_ENABLE_Q31 1
#endif

#ifndef MFCC_BENCH_ENABLE_Q15
#define MFCC_BENCH_ENABLE_Q15 1
#endif

#ifndef MFCC_BENCH_ENABLE_F16
#define MFCC_BENCH_ENABLE_F16 1
#endif

#ifndef MFCC_BENCH_ENABLE_SP1024X23X12_F32
#define MFCC_BENCH_ENABLE_SP1024X23X12_F32 1
#endif

#ifndef MFCC_BENCH_ENABLE_SP1024X23X12_F16
#define MFCC_BENCH_ENABLE_SP1024X23X12_F16 MFCC_BENCH_ENABLE_F16
#endif

// Correctness tolerances against floating reference (generic f32 preferred).
#ifndef MFCC_BENCH_TOL_F32
#define MFCC_BENCH_TOL_F32 0.250f
#endif

#ifndef MFCC_BENCH_TOL_Q31
#define MFCC_BENCH_TOL_Q31 0.050f
#endif

#ifndef MFCC_BENCH_TOL_Q15
#define MFCC_BENCH_TOL_Q15 0.250f
#endif

#ifndef MFCC_BENCH_TOL_F16
#define MFCC_BENCH_TOL_F16 0.270f
#endif

#ifndef MFCC_BENCH_PRINT_HART
#define MFCC_BENCH_PRINT_HART 0u
#endif

static inline uint32_t mfcc_bench_hart_id(void) {
  uint64_t x;
  asm volatile("csrr %0, mhartid" : "=r"(x));
  return (uint32_t)x;
}

static inline bool mfcc_bench_is_print_hart(void) {
  return mfcc_bench_hart_id() == MFCC_BENCH_PRINT_HART;
}

static inline uint64_t mfcc_bench_rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif
