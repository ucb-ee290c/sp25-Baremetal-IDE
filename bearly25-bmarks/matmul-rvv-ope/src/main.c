#include <stdio.h>

#include "bench_cache.h"
#include "bench_config.h"
#include "simple_setup.h"

uint64_t target_frequency = MATMUL_BENCH_TARGET_FREQUENCY_HZ;

void bench_run(void);

void app_init(void) {
  bench_cache_init();
  init_test(target_frequency);
}

void app_main(void) {
  printf("\n=== I8->I32 MATMUL BENCHMARK (64x64) @ %llu Hz ===\n",
         (unsigned long long)target_frequency);
  printf("  runs(cold)=%d, runs(hot)=%d\n",
         MATMUL_BENCH_RUNS_COLD, MATMUL_BENCH_RUNS_HOT);
  printf("  cache_thrash_bytes=%u\n", (unsigned)(MATMUL_L2_BYTES * 2u));

  bench_run();

  printf("\n=== BENCHMARK DONE ===\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
