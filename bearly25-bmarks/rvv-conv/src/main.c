/*
 * main.c - Entry point for rvv-conv assembly benchmarks.
 */
#include <stdio.h>

#include "bench_config.h"
#include "bench_impl.h"
#include "bench_sizes.h"
#include "chip_config.h"
#include "simple_setup.h"

uint64_t target_frequency = CONV_BENCH_TARGET_FREQUENCY_HZ;

static void print_config(uint64_t frequency_hz) {
  printf("  frequency=%llu Hz\n", (unsigned long long)frequency_hz);
  printf("  runs(cold)=%d runs(hot)=%d\n",
         CONV_BENCH_RUNS_COLD, CONV_BENCH_RUNS_HOT);
  printf("  cache_thrash_bytes=%u\n", (unsigned)CONV_BENCH_CACHE_THRASH_BYTES);
  printf("  kernels: f32_3x3=%d f32_5x5=%d i8_3x3=%d i8_5x5=%d\n",
         CONV_BENCH_ENABLE_F32_3X3,
         CONV_BENCH_ENABLE_F32_5X5,
         CONV_BENCH_ENABLE_I8_3X3,
         CONV_BENCH_ENABLE_I8_5X5);
  printf("  cases=%d (batch,channels,height,width)\n", CORE_V_CONV_NUM_CASES);
}

static void run_suite_for_frequency(uint64_t frequency_hz) {
  printf("=== RVV-CONV ASM BENCH @ %llu Hz ===\n",
         (unsigned long long)frequency_hz);
  print_config(frequency_hz);

  for (int i = 0; i < CORE_V_CONV_NUM_CASES; ++i) {
    bench_run_case(&RVV_CONV_CASES[i]);
  }

  printf("=== RVV-CONV ASM BENCH DONE @ %llu Hz ===\n",
         (unsigned long long)frequency_hz);
}

void app_init(void) {
  init_test(target_frequency);
}

void app_main(void) {
  run_suite_for_frequency(target_frequency);
}

#if CONV_BENCH_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = {
  CONV_BENCH_PLL_FREQ_LIST
};
#endif

int main(void) {
#if CONV_BENCH_ENABLE_PLL_SWEEP
  const size_t num_freqs =
      sizeof(k_pll_sweep_freqs_hz) / sizeof(k_pll_sweep_freqs_hz[0]);
  if (num_freqs == 0u) {
    return 0;
  }

  target_frequency = k_pll_sweep_freqs_hz[0];
  init_test(target_frequency);
  run_suite_for_frequency(target_frequency);

  for (size_t i = 1; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, CONV_BENCH_PLL_SWEEP_SLEEP_MS);
    run_suite_for_frequency(target_frequency);
  }
  return 0;
#else
  app_init();
  app_main();
  return 0;
#endif
}
