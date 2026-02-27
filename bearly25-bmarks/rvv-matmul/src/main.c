#include <stdio.h>

#include "bench_cache.h"
#include "bench_config.h"
#include "bench_impl.h"
#include "bench_sizes.h"
#include "simple_setup.h"

uint64_t target_frequency = RVV_BENCH_TARGET_FREQUENCY_HZ;

void app_init(void) {
  init_test(target_frequency);
  bench_cache_init();
}

void app_main(void) {
  printf("=== RVV MATMUL BENCHMARKS ===\n");
  printf("  runs(cold)=%d, runs(hot)=%d\n", RVV_BENCH_RUNS_COLD, RVV_BENCH_RUNS_HOT);
  printf("  cache_thrash_bytes=%u\n", (unsigned)(RVV_L2_BYTES * 2u));
  printf("  kernels: f32=%d i8_i16=%d i8_i32=%d packed=%d unpacked=%d\n",
         RVV_BENCH_ENABLE_F32,
         RVV_BENCH_ENABLE_I8_I16,
         RVV_BENCH_ENABLE_I8_I32,
         RVV_BENCH_ENABLE_PACKED,
         RVV_BENCH_ENABLE_UNPACKED);

  printf("\n--- SQUARE CASES ---\n");
  for (int i = 0; i < RVV_BENCH_NUM_SQUARE_CASES; ++i) {
    bench_run_case(&RVV_BENCH_SQUARE_CASES[i]);
  }

  printf("\n--- RECTANGULAR CASES ---\n");
  for (int i = 0; i < RVV_BENCH_NUM_RECT_CASES; ++i) {
    bench_run_case(&RVV_BENCH_RECT_CASES[i]);
  }

  printf("\n=== RVV MATMUL BENCHMARKS DONE ===\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
