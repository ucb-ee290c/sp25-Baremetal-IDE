/*
 * bench_config.h - Configuration and shared types for acc-conv benchmarks.
 */
#ifndef ACC_CONV_BENCH_CONFIG_H
#define ACC_CONV_BENCH_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef CONV_BENCH_TARGET_FREQUENCY_HZ
#define CONV_BENCH_TARGET_FREQUENCY_HZ 50000000ULL
#endif

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

#ifndef CONV_BENCH_RUNS_COLD
#define CONV_BENCH_RUNS_COLD 8
#endif

#ifndef CONV_BENCH_RUNS_HOT
#define CONV_BENCH_RUNS_HOT 32
#endif

#ifndef CONV_BENCH_CACHE_THRASH_BYTES
#define CONV_BENCH_CACHE_THRASH_BYTES (512u * 1024u)
#endif

#ifndef CONV_BENCH_CACHE_LINE_BYTES
#define CONV_BENCH_CACHE_LINE_BYTES 64u
#endif

/* Number of cycles to busy-wait between consecutive perform_convolution calls.
 * Set to 0 to disable (default). Override via -DCONV_BENCH_INTER_CALL_CYCLES=N. */
#ifndef CONV_BENCH_INTER_CALL_CYCLES
#define CONV_BENCH_INTER_CALL_CYCLES 0ULL
#endif

typedef struct {
  const char *name;
  int batch;
  int channels;
  int height;
  int width;
} ConvBenchCase;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline int conv_valid_out_dim(int input, int kernel) {
  return input - kernel + 1;
}

#endif // ACC_CONV_BENCH_CONFIG_H
