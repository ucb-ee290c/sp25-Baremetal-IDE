/*
 * main.c - Entry point for OPE matmul microbenchmarks.
 *
 * Runs square and rectangular cases across selected implementations and
 * reports timing statistics.
 */
#include <stdio.h>

#include "bench_config.h"
#include "bench_cache.h"
#include "bench_sizes.h"
#include "bench_impl.h"

void app_init(void) {
  printf("DEBUG: app_init() started\n");
  printf("DEBUG: calling bench_cache_init()...\n");
  // bench_cache_init();
  printf("DEBUG: bench_cache_init() completed\n");
}

void app_main(void) {
  printf("=== OPE BENCHMARKS ===\n");
  printf("  L2 size approx: %d KiB\n", (int)(OPE_L2_BYTES / 1024));
  printf("  BENCH_RUNS_COLD=%d, BENCH_RUNS_HOT=%d\n", BENCH_RUNS_COLD, BENCH_RUNS_HOT);

  printf("\n--- SQUARE SIZES (general) ---\n");
  for (int i = 0; i < OPE_BENCH_NUM_SQUARE_CASES; ++i) {
    const OpeSizeCase *cs = &OPE_BENCH_SQUARE_CASES[i];

#if BENCH_ENABLE_IMPL_ARB
    bench_run_case(cs, OPE_IMPL_ARB);
#endif

#if BENCH_ENABLE_IMPL_SQUARE
    bench_run_case(cs, OPE_IMPL_SQUARE);
#endif
  }

  printf("\n--- RECTANGULAR / UNALIGNED SIZES ---\n");
  for (int i = 0; i < OPE_BENCH_NUM_RECT_CASES; ++i) {
    const OpeSizeCase *cs = &OPE_BENCH_RECT_CASES[i];

#if BENCH_ENABLE_IMPL_ARB
    bench_run_case(cs, OPE_IMPL_ARB);
#endif
  }
  printf("\n=== OPE BENCHMARKS DONE ===\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
