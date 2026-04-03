/*
 * bench_config.h - Shared configuration for RVV matmul benchmarks.
 */
#ifndef RVV_BENCH_CONFIG_H
#define RVV_BENCH_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef RVV_BENCH_TARGET_FREQUENCY_HZ
#define RVV_BENCH_TARGET_FREQUENCY_HZ 1050000000ULL
#endif

// 0: single-frequency mode
// 1: iterate over RVV_BENCH_PLL_FREQ_LIST frequencies
#ifndef RVV_BENCH_ENABLE_PLL_SWEEP
#define RVV_BENCH_ENABLE_PLL_SWEEP 0
#endif

#ifndef RVV_BENCH_PLL_SWEEP_SLEEP_MS
#define RVV_BENCH_PLL_SWEEP_SLEEP_MS 10000u
#endif

// Comma-separated list used when RVV_BENCH_ENABLE_PLL_SWEEP=1
// #define RVV_BENCH_PLL_FREQ_LIST 400000000ULL, 600000000ULL, 800000000ULL
#ifndef RVV_BENCH_PLL_FREQ_LIST
#define RVV_BENCH_PLL_FREQ_LIST \
  50000000ULL, \
  150000000ULL, \
  250000000ULL, \
  350000000ULL 
  // 450000000ULL, \
  // 550000000ULL, \
  // 650000000ULL, \
  // 750000000ULL, \
  // 850000000ULL
#endif

// Repetitions for each cache state.
#ifndef RVV_BENCH_RUNS_COLD
#define RVV_BENCH_RUNS_COLD 10
#endif

#ifndef RVV_BENCH_RUNS_HOT
#define RVV_BENCH_RUNS_HOT 100000
#endif

// Approximate L2 cache size for flush-thrash helper
#ifndef RVV_L2_BYTES
#define RVV_L2_BYTES (256u * 1024u)
#endif

// Per-core Tightly Coupled Memory (TCM / scratchpad) addresses.
#ifndef CORE0_TCM_BASE
#define CORE0_TCM_BASE 0x08010000UL
#endif
#ifndef CORE1_TCM_BASE
#define CORE1_TCM_BASE 0x08012000UL
#endif
#ifndef TCM_SIZE
#define TCM_SIZE 0x2000UL  /* 8 KB */
#endif

#ifndef RVV_BENCH_ENABLE_F32
#define RVV_BENCH_ENABLE_F32 1
#endif

#ifndef RVV_BENCH_ENABLE_I8_I16
#define RVV_BENCH_ENABLE_I8_I16 0
#endif

#ifndef RVV_BENCH_ENABLE_I8_I32
#define RVV_BENCH_ENABLE_I8_I32 1
#endif

#ifndef RVV_BENCH_ENABLE_I32
#define RVV_BENCH_ENABLE_I32 0
#endif

#ifndef RVV_BENCH_ENABLE_I8_I8
#define RVV_BENCH_ENABLE_I8_I8 1
#endif

#ifndef RVV_BENCH_ENABLE_UNPACKED
#define RVV_BENCH_ENABLE_UNPACKED 0
#endif

#ifndef RVV_BENCH_ENABLE_PACKED
#define RVV_BENCH_ENABLE_PACKED 1
#endif

// Hart that is allowed to print benchmark logs.
#ifndef RVV_BENCH_PRINT_HART
#define RVV_BENCH_PRINT_HART 0u
#endif

// Multicore matmul: split M rows across 2 harts.
#ifndef RVV_BENCH_ENABLE_MULTICORE
#define RVV_BENCH_ENABLE_MULTICORE 1
#endif

// When 1, skip single-core runs and only run multicore variants.
#ifndef RVV_BENCH_MULTICORE_ONLY
#define RVV_BENCH_MULTICORE_ONLY 0
#endif

#ifndef RVV_BENCH_MC_ROWS_HART0
#define RVV_BENCH_MC_ROWS_HART0 35u
#endif

#ifndef RVV_BENCH_MC_ROWS_HART1
#define RVV_BENCH_MC_ROWS_HART1 29u
#endif

// Number of harts participating in post-GEMM synchronization.
// With multicore enabled, hart 1 is in WFI (hthread __main) and does
// not participate in the barrier loop, so set to 1 to avoid deadlock.
#ifndef RVV_BENCH_SYNC_HARTS
#if RVV_BENCH_ENABLE_MULTICORE
#define RVV_BENCH_SYNC_HARTS 1u
#else
#define RVV_BENCH_SYNC_HARTS 2u
#endif
#endif

typedef struct {
  const char *name;
  size_t M;
  size_t N;
  size_t K;
} RvvMatmulCase;

static inline uint32_t rvv_bench_hart_id(void) {
  uint64_t x;
  asm volatile("csrr %0, mhartid" : "=r"(x));
  return (uint32_t)x;
}

static inline bool rvv_bench_is_print_hart(void) {
  return rvv_bench_hart_id() == RVV_BENCH_PRINT_HART;
}

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif // RVV_BENCH_CONFIG_H
