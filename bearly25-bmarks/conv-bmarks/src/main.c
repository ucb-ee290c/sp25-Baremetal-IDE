#include <stdio.h>

#include "conv_bench.h"
#include "main.h"
#include "simple_setup.h"

uint64_t target_frequency = CONV_BENCH_TARGET_FREQUENCY_HZ;

static void run_suite_for_frequency(uint64_t frequency_hz) {
  printf("\n=== CONV-BMARKS @ %llu Hz ===\n",
         (unsigned long long)frequency_hz);
  conv_bench_run_all();
  printf("=== CONV-BMARKS DONE @ %llu Hz ===\n",
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

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile("wfi");
  }
}
