/*
 * main.c - Entry point for acc-conv accelerator benchmarks.
 */
#include <stdio.h>

#include "bench_config.h"
#include "bench_impl.h"
#include "bench_sizes.h"
#include "chip_config.h"
#include "simple_setup.h"

uint64_t target_frequency = CONV_BENCH_TARGET_FREQUENCY_HZ;

static void run_suite_for_frequency(uint64_t frequency_hz) {
  printf("=== ACC-CONV BENCH @ %llu Hz ===\n", (unsigned long long)frequency_hz);
  printf("  frequency=%llu Hz\n", (unsigned long long)frequency_hz);
  printf("  runs(cold)=%d runs(hot)=%d\n", CONV_BENCH_RUNS_COLD, CONV_BENCH_RUNS_HOT);
  printf("  cache_thrash_bytes=%u\n", (unsigned)CONV_BENCH_CACHE_THRASH_BYTES);
  printf("  cases=%d (batch,channels,height,width)\n", ACC_CONV_NUM_CASES);

  for (int i = 0; i < ACC_CONV_NUM_CASES; ++i)
    bench_run_case(&ACC_CONV_CASES[i]);

  printf("=== ACC-CONV BENCH DONE @ %llu Hz ===\n", (unsigned long long)frequency_hz);
}

#if CONV_BENCH_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = { CONV_BENCH_PLL_FREQ_LIST };
#endif

int main(void) {
#if CONV_BENCH_ENABLE_PLL_SWEEP
  const size_t num_freqs =
      sizeof(k_pll_sweep_freqs_hz) / sizeof(k_pll_sweep_freqs_hz[0]);
  if (num_freqs == 0u) return 0;

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
  init_test(target_frequency);
  run_suite_for_frequency(target_frequency);
  return 0;
#endif
}
