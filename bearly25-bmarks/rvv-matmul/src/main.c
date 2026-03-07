#include <stdio.h>

#include "bench_cache.h"
#include "bench_config.h"
#include "bench_impl.h"
#include "bench_sizes.h"
#include "simple_setup.h"

uint64_t target_frequency = RVV_BENCH_TARGET_FREQUENCY_HZ;

static void run_suite_for_frequency(uint64_t frequency_hz) {
  if (rvv_bench_is_print_hart()) {
    printf("\n=== RVV MATMUL BENCHMARKS @ %llu Hz ===\n",
           (unsigned long long)frequency_hz);
    printf("  runs(cold)=%d, runs(hot)=%d\n", RVV_BENCH_RUNS_COLD, RVV_BENCH_RUNS_HOT);
    printf("  cache_thrash_bytes=%u\n", (unsigned)(RVV_L2_BYTES * 2u));
    printf("  kernels: f32=%d i8_i16=%d i8_i32=%d packed=%d unpacked=%d\n",
           RVV_BENCH_ENABLE_F32,
           RVV_BENCH_ENABLE_I8_I16,
           RVV_BENCH_ENABLE_I8_I32,
           RVV_BENCH_ENABLE_PACKED,
           RVV_BENCH_ENABLE_UNPACKED);

    printf("\n--- SQUARE CASES ---\n");
  }
  for (int i = 0; i < RVV_BENCH_NUM_SQUARE_CASES; ++i) {
    bench_run_case(&RVV_BENCH_SQUARE_CASES[i]);
  }

  if (rvv_bench_is_print_hart()) {
    printf("\n--- RECTANGULAR CASES ---\n");
  }
  for (int i = 0; i < RVV_BENCH_NUM_RECT_CASES; ++i) {
    bench_run_case(&RVV_BENCH_RECT_CASES[i]);
  }

  if (rvv_bench_is_print_hart()) {
    printf("\n=== RVV MATMUL BENCHMARKS DONE @ %llu Hz ===\n",
           (unsigned long long)frequency_hz);
  }
}

void app_init(void) {
  init_test(target_frequency);
  bench_cache_init();
}

void app_main(void) {
  run_suite_for_frequency(target_frequency);
}

#if RVV_BENCH_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = {
  RVV_BENCH_PLL_FREQ_LIST
};
#endif

int main(void) {
#if RVV_BENCH_ENABLE_PLL_SWEEP
  const size_t num_freqs =
      sizeof(k_pll_sweep_freqs_hz) / sizeof(k_pll_sweep_freqs_hz[0]);
  if (num_freqs == 0u) {
    return 0;
  }

  target_frequency = k_pll_sweep_freqs_hz[0];
  init_test(target_frequency);
  bench_cache_init();
  run_suite_for_frequency(target_frequency);

  for (size_t i = 1; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, RVV_BENCH_PLL_SWEEP_SLEEP_MS);
    run_suite_for_frequency(target_frequency);
  }
  return 0;
#else
  app_init();
  app_main();
  return 0;
#endif
}

int __main(void) {
#if RVV_BENCH_ENABLE_PLL_SWEEP
  const size_t num_freqs =
      sizeof(k_pll_sweep_freqs_hz) / sizeof(k_pll_sweep_freqs_hz[0]);
  if (num_freqs == 0u) {
    return 0;
  }

  target_frequency = k_pll_sweep_freqs_hz[0];
  init_test(target_frequency);
  bench_cache_init();
  run_suite_for_frequency(target_frequency);

  for (size_t i = 1; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, RVV_BENCH_PLL_SWEEP_SLEEP_MS);
    run_suite_for_frequency(target_frequency);
  }
  return 0;
#else
  app_init();
  app_main();
  return 0;
#endif
}

