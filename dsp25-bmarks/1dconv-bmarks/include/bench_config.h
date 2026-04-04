#ifndef CONV1D_BENCH_CONFIG_H
#define CONV1D_BENCH_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "bench_generated_data.h"

#ifndef CONV_BENCH_TARGET_FREQUENCY_HZ
#define CONV_BENCH_TARGET_FREQUENCY_HZ 500000000ULL
#endif

// 0: single-frequency mode
// 1: iterate over CONV_BENCH_PLL_FREQ_LIST
#ifndef CONV_BENCH_ENABLE_PLL_SWEEP
#define CONV_BENCH_ENABLE_PLL_SWEEP 0
#endif

#ifndef CONV_BENCH_PLL_SWEEP_SLEEP_MS
#define CONV_BENCH_PLL_SWEEP_SLEEP_MS 10000u
#endif

#ifndef CONV_BENCH_PLL_FREQ_LIST
#define CONV_BENCH_PLL_FREQ_LIST \
  50000000ULL, \
  150000000ULL, \
  250000000ULL, \
  350000000ULL
#endif

#ifndef CONV_BENCH_PRINT_HART
#define CONV_BENCH_PRINT_HART 0u
#endif

// Sweep values
#ifndef CONV_BENCH_N_LIST
#define CONV_BENCH_N_LIST \
  64u, \
  128u, \
  256u, \
  512u, \
  1024u
#endif

#ifndef CONV_BENCH_K_LIST
#define CONV_BENCH_K_LIST \
  8u, \
  16u
#endif

#ifndef CONV_BENCH_DILATION_LIST
#define CONV_BENCH_DILATION_LIST \
  1u, \
  2u, \
  4u
#endif

#ifndef CONV_BENCH_MAX_N
#define CONV_BENCH_MAX_N CONV_BENCH_GENERATED_MAX_N
#endif

#ifndef CONV_BENCH_MAX_K
#define CONV_BENCH_MAX_K CONV_BENCH_GENERATED_MAX_K
#endif

#ifndef CONV_BENCH_MAX_DILATION
#define CONV_BENCH_MAX_DILATION 8u
#endif

// Iteration count by cache mode.
#ifndef CONV_BENCH_RUNS_COLD
#define CONV_BENCH_RUNS_COLD 5u
#endif

#ifndef CONV_BENCH_RUNS_WARM
#define CONV_BENCH_RUNS_WARM 25u
#endif

// Cache state toggles
#ifndef CONV_BENCH_ENABLE_CACHE_COLD
#define CONV_BENCH_ENABLE_CACHE_COLD 1
#endif

#ifndef CONV_BENCH_ENABLE_CACHE_WARM
#define CONV_BENCH_ENABLE_CACHE_WARM 1
#endif

// Reuse-mode toggles.
// reuse=0: refill input/kernel every timed run
// reuse=1: keep same input/kernel resident and repeat calls
#ifndef CONV_BENCH_ENABLE_REUSE_OFF
#define CONV_BENCH_ENABLE_REUSE_OFF 1
#endif

#ifndef CONV_BENCH_ENABLE_REUSE_ON
#define CONV_BENCH_ENABLE_REUSE_ON 1
#endif

// Memory placement mode toggles.
#ifndef CONV_BENCH_ENABLE_MEMMODE_DRAM_L2
#define CONV_BENCH_ENABLE_MEMMODE_DRAM_L2 1
#endif

#ifndef CONV_BENCH_ENABLE_MEMMODE_SCRATCHPAD
#define CONV_BENCH_ENABLE_MEMMODE_SCRATCHPAD 1
#endif

// Number of generated datasets to use (<= CONV_BENCH_GENERATED_NUM_DATASETS).
#ifndef CONV_BENCH_DATASET_COUNT
#define CONV_BENCH_DATASET_COUNT CONV_BENCH_GENERATED_NUM_DATASETS
#endif

// Approximate L2 size for cache thrash helper.
#ifndef CONV_BENCH_L2_BYTES
#define CONV_BENCH_L2_BYTES (256u * 1024u)
#endif

// Memory-map defaults for buffer placement.
#ifndef CONV_BENCH_DRAM_INPUT_BASE
#define CONV_BENCH_DRAM_INPUT_BASE 0x8FFA0000UL
#endif

#ifndef CONV_BENCH_DRAM_KERNEL_BASE
#define CONV_BENCH_DRAM_KERNEL_BASE 0x8FFB0000UL
#endif

#ifndef CONV_BENCH_DRAM_OUTPUT_BASE
#define CONV_BENCH_DRAM_OUTPUT_BASE 0x8FFC0000UL
#endif

#ifndef CONV_BENCH_DRAM_REGION_BYTES
#define CONV_BENCH_DRAM_REGION_BYTES (64u * 1024u)
#endif

#ifndef CONV_BENCH_SCRATCH_INPUT_BASE
#define CONV_BENCH_SCRATCH_INPUT_BASE 0x08001000UL
#endif

#ifndef CONV_BENCH_SCRATCH_KERNEL_BASE
#define CONV_BENCH_SCRATCH_KERNEL_BASE 0x08003000UL
#endif

#ifndef CONV_BENCH_SCRATCH_OUTPUT_BASE
#define CONV_BENCH_SCRATCH_OUTPUT_BASE 0x08004000UL
#endif

#ifndef CONV_BENCH_SCRATCH_REGION_BYTES
#define CONV_BENCH_SCRATCH_REGION_BYTES (16u * 1024u)
#endif

// Accuracy thresholds against scalar baseline.
#ifndef CONV_BENCH_ACCURACY_ABS_TOL
#define CONV_BENCH_ACCURACY_ABS_TOL 0.001f
#endif

#ifndef CONV_BENCH_ACCURACY_REL_TOL
#define CONV_BENCH_ACCURACY_REL_TOL 0.001f
#endif

#ifndef CONV_BENCH_ACC_EPSILON
#define CONV_BENCH_ACC_EPSILON 1.0e-9f
#endif

#ifndef CONV_BENCH_NO_REUSE_SHIFT_STRIDE
#define CONV_BENCH_NO_REUSE_SHIFT_STRIDE 29u
#endif

#ifndef CONV_BENCH_PRINT_CASE_DETAILS
#define CONV_BENCH_PRINT_CASE_DETAILS 1
#endif

static inline uint32_t conv_bench_hart_id(void) {
  uint64_t x;
  asm volatile("csrr %0, mhartid" : "=r"(x));
  return (uint32_t)x;
}

static inline bool conv_bench_is_print_hart(void) {
  return conv_bench_hart_id() == CONV_BENCH_PRINT_HART;
}

static inline uint64_t conv_bench_rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

#endif
