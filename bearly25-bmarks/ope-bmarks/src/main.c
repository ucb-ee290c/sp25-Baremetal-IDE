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
#include "chip_config.h"
#include "hal_ope.h"
#include "simple_setup.h"

// Heap debugging
extern char __heap_start[];
extern char __heap_end[];
extern char __end[];

uint64_t target_frequency = 400000000l;

static void print_heap_usage(void) {
  extern char *_sbrk(ptrdiff_t);
  char *current = _sbrk(0);  // Get current break without changing it
  size_t used = (size_t)(current - __end);
  size_t total = (size_t)(__heap_end - __end);
  printf("  [HEAP] used=%zu bytes (%zu KB), total=%zu bytes (%zu KB)\n", 
         used, used/1024, total, total/1024);
}

void app_init() {
  init_test(target_frequency);

  printf("DEBUG: app_init() started\n");
  printf("DEBUG: calling bench_cache_init()...\n");
  bench_cache_init();
  printf("DEBUG: bench_cache_init() completed\n");
}

void app_main(void) {
  printf("=== OPE BENCHMARKS ===\n");
  printf("  L2 size approx: %d KiB\n", (int)(OPE_L2_BYTES / 1024));
  printf("  BENCH_RUNS_COLD=%d, BENCH_RUNS_HOT=%d\n", BENCH_RUNS_COLD, BENCH_RUNS_HOT);
  print_heap_usage();

  // Pre-allocate workspace for largest expected matrix size to avoid fragmentation
  // Adjust max size based on your test cases (e.g., 256 for sq_256)
  printf("  Initializing OPE workspace for max 256x256 matrices...\n");
  ope_init_workspace(256, 256, 256);
  print_heap_usage();

  printf("\n--- SQUARE SIZES (general) ---\n");
  for (int i = 0; i < OPE_BENCH_NUM_SQUARE_CASES; ++i) {
    const OpeSizeCase *cs = &OPE_BENCH_SQUARE_CASES[i];

#if BENCH_ENABLE_IMPL_ARB
    bench_run_case(cs, OPE_IMPL_ARB);
    print_heap_usage();
#endif

#if BENCH_ENABLE_IMPL_SQUARE
    bench_run_case(cs, OPE_IMPL_SQUARE);
    print_heap_usage();
#endif
  }

  printf("\n--- RECTANGULAR / UNALIGNED SIZES ---\n");
  for (int i = 0; i < OPE_BENCH_NUM_RECT_CASES; ++i) {
    const OpeSizeCase *cs = &OPE_BENCH_RECT_CASES[i];

#if BENCH_ENABLE_IMPL_ARB
    bench_run_case(cs, OPE_IMPL_ARB);
    print_heap_usage();
#endif
  }

  ope_free_workspace();
  printf("\n=== OPE BENCHMARKS DONE ===\n");
  print_heap_usage();
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
