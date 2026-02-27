#include <stdio.h>

#include "conv_bench.h"
#include "main.h"
#include "simple_setup.h"

uint64_t target_frequency = CONV_BENCH_TARGET_FREQUENCY_HZ;

void app_init(void) {
  init_test(target_frequency);
}

void app_main(void) {
  conv_bench_run_all();
}

int main(void) {
  app_init();
  app_main();
  return 0;
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile("wfi");
  }
}
