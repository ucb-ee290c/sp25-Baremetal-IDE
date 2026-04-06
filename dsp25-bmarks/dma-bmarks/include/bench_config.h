#ifndef DSP25_DMA_BENCH_CONFIG_H
#define DSP25_DMA_BENCH_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

#ifndef DMA_BENCH_TARGET_FREQUENCY_HZ
#define DMA_BENCH_TARGET_FREQUENCY_HZ 50000000ULL
#endif

/* 0: single-frequency mode, 1: sweep DMA_BENCH_PLL_FREQ_LIST */
#ifndef DMA_BENCH_ENABLE_PLL_SWEEP
#define DMA_BENCH_ENABLE_PLL_SWEEP 0
#endif

#ifndef DMA_BENCH_PLL_SWEEP_SLEEP_MS
#define DMA_BENCH_PLL_SWEEP_SLEEP_MS 10000u
#endif

#ifndef DMA_BENCH_PLL_FREQ_LIST
#define DMA_BENCH_PLL_FREQ_LIST \
  50000000ULL, \
  150000000ULL, \
  250000000ULL
#endif

#ifndef DMA_BENCH_PRINT_HART
#define DMA_BENCH_PRINT_HART 0u
#endif

#ifndef DMA_BENCH_NUM_RUNS
#define DMA_BENCH_NUM_RUNS 8u
#endif

#ifndef DMA_BENCH_HOT_REPEAT_RUNS
#define DMA_BENCH_HOT_REPEAT_RUNS DMA_BENCH_NUM_RUNS
#endif

#ifndef DMA_BENCH_BASE_SEED
#define DMA_BENCH_BASE_SEED 0x12345678u
#endif

/* DMA payload width: bytes per packet = (1 << DMA_BENCH_LOGW).
 * Default to logw=2 (4B) because this is the most broadly validated mode
 * in current dsp25 DMA tests. Set to 6 for 64B packets if your config supports it. */
#ifndef DMA_BENCH_LOGW
#define DMA_BENCH_LOGW 2u
#endif

#ifndef DMA_BENCH_PRIORITY
#define DMA_BENCH_PRIORITY 1u
#endif

#ifndef DMA_BENCH_CORE_ID
#define DMA_BENCH_CORE_ID 0u
#endif

#ifndef DMA_BENCH_IDLE_SPIN_CYCLES
#define DMA_BENCH_IDLE_SPIN_CYCLES 30
#endif

/* DMA mode toggles */
#ifndef DMA_BENCH_ENABLE_SINGLE_CHANNEL
#define DMA_BENCH_ENABLE_SINGLE_CHANNEL 1
#endif

#ifndef DMA_BENCH_ENABLE_MULTI_CHANNEL
#define DMA_BENCH_ENABLE_MULTI_CHANNEL 1
#endif

#ifndef DMA_BENCH_MULTI_CHANNELS
#define DMA_BENCH_MULTI_CHANNELS 2u
#endif

/* Cache-state toggles */
#ifndef DMA_BENCH_ENABLE_CACHE_COLD
#define DMA_BENCH_ENABLE_CACHE_COLD 1
#endif

#ifndef DMA_BENCH_ENABLE_CACHE_WARM_SRC
#define DMA_BENCH_ENABLE_CACHE_WARM_SRC 1
#endif

#ifndef DMA_BENCH_ENABLE_CACHE_WARM_DST
#define DMA_BENCH_ENABLE_CACHE_WARM_DST 1
#endif

#ifndef DMA_BENCH_ENABLE_CACHE_WARM_BOTH
#define DMA_BENCH_ENABLE_CACHE_WARM_BOTH 1
#endif

#ifndef DMA_BENCH_ENABLE_CACHE_HOT_REPEAT
#define DMA_BENCH_ENABLE_CACHE_HOT_REPEAT 1
#endif

#ifndef DMA_BENCH_CACHE_LINE_BYTES
#define DMA_BENCH_CACHE_LINE_BYTES 64u
#endif

#ifndef DMA_BENCH_CACHE_EVICT_BYTES
#define DMA_BENCH_CACHE_EVICT_BYTES (256u * 1024u)
#endif

/* Region base addresses */
/* Use the same DRAM windows as validated dma-tests (0x8500_0000 range). */
#ifndef DMA_BENCH_DRAM_SRC_BASE
#define DMA_BENCH_DRAM_SRC_BASE 0x85000000UL
#endif

#ifndef DMA_BENCH_DRAM_DST_BASE
#define DMA_BENCH_DRAM_DST_BASE 0x85100000UL
#endif

#ifndef DMA_BENCH_SCRATCHPAD_BASE
#define DMA_BENCH_SCRATCHPAD_BASE 0x08000000UL
#endif

/* Case toggles and sizes (no TCM cases by design). */
#ifndef DMA_BENCH_ENABLE_CASE_DRAM_TO_DRAM
#define DMA_BENCH_ENABLE_CASE_DRAM_TO_DRAM 1
#endif

#ifndef DMA_BENCH_ENABLE_CASE_DRAM_TO_SCRATCH
#define DMA_BENCH_ENABLE_CASE_DRAM_TO_SCRATCH 1
#endif

#ifndef DMA_BENCH_ENABLE_CASE_SCRATCH_TO_DRAM
#define DMA_BENCH_ENABLE_CASE_SCRATCH_TO_DRAM 1
#endif

#ifndef DMA_BENCH_DRAM_TO_DRAM_BYTES
#define DMA_BENCH_DRAM_TO_DRAM_BYTES (64u * 1024u)
#endif

#ifndef DMA_BENCH_DRAM_TO_SCRATCH_BYTES
#define DMA_BENCH_DRAM_TO_SCRATCH_BYTES (16u * 1024u)
#endif

#ifndef DMA_BENCH_SCRATCH_TO_DRAM_BYTES
#define DMA_BENCH_SCRATCH_TO_DRAM_BYTES (16u * 1024u)
#endif

static inline uint32_t dma_bench_hart_id(void) {
  uint64_t x;
  asm volatile("csrr %0, mhartid" : "=r"(x));
  return (uint32_t)x;
}

static inline bool dma_bench_is_print_hart(void) {
  return dma_bench_hart_id() == DMA_BENCH_PRINT_HART;
}

#endif
