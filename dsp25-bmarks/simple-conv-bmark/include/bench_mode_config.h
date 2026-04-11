#ifndef SIMPLE_CONV_BENCH_MODE_CONFIG_H
#define SIMPLE_CONV_BENCH_MODE_CONFIG_H

/*
 * Set each mode to 1 to run it, or 0 to disable it.
 */
#define SIMPLE_CONV_INPUT_LENGTH 256

/* Keep K=8 to match current RVV kernel specialization in main.c */
#define SIMPLE_CONV_RUN_ACCELERATOR_MMIO 1
#define SIMPLE_CONV_RUN_ACCELERATOR_MMIO_STEADY_STATE 1
#define SIMPLE_CONV_RUN_ACCELERATOR_MMIO_RTL_STYLE 1
#define SIMPLE_CONV_RUN_SCALAR_REFERENCE 0
#define SIMPLE_CONV_RUN_RVV_SINGLE_CORE_REFERENCE 1
#define SIMPLE_CONV_RUN_RVV_SINGLE_CORE_RTL_STYLE 1
#define SIMPLE_CONV_RUN_RVV_MULTI_CORE_REFERENCE  0
#define SIMPLE_CONV_RUN_ACCELERATOR_DMA  0

/*
 * Steady-state MMIO mode:
 * - each timed run executes this many back-to-back accelerator chunks
 * - benchmark reports both per-run and per-chunk cycles
 */
#define SIMPLE_CONV_STEADY_STATE_CHUNKS_PER_TIMED_RUN 4
#define SIMPLE_CONV_STEADY_STATE_WARMUP_RUNS          4
#define SIMPLE_CONV_STEADY_STATE_TIMED_RUNS           16

/*
 * RTL-style MMIO mode:
 * - emulates 1d-conv-rtl benchmark_*_wide timing scope (START->N output drain)
 * - setup + preload are intentionally outside the timed region
 * - keep this conservative for taped-out revisions to avoid pre-start FIFO stalls
 */
#define SIMPLE_CONV_RTL_STYLE_PRELOAD_PACKETS 8
#define SIMPLE_CONV_RTL_STYLE_ALLOW_UNSAFE_PRELOAD 0

/* Backward compatibility for older code paths */
#define SIMPLE_CONV_RUN_RVV_REFERENCE SIMPLE_CONV_RUN_RVV_MULTI_CORE_REFERENCE

#endif /* SIMPLE_CONV_BENCH_MODE_CONFIG_H */
