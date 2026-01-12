/*
 * main.c - Entry point for core-v-conv benchmarks.
 */
#include <stdio.h>

#include "bench_config.h"
#include "bench_sizes.h"
#include "bench_impl.h"
#include "chip_config.h"

void app_init(void) {
  UART_InitType UART0_init_config;
  UART0_init_config.baudrate = 115200;
  UART0_init_config.mode = UART_MODE_TX_RX;
  UART0_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART0_init_config);
}

static void print_config(void) {
  printf("  BENCH_RUNS=%d, BENCH_VERIFY=%d\n", BENCH_RUNS, BENCH_VERIFY);
  printf("  CONV_CHANNELS=%d, STRIDE=%d, PADDING=%d, RELU=%d\n",
         CONV_CHANNELS, CONV_STRIDE, CONV_PADDING, CONV_RELU);
#if BENCH_ENABLE_VEC
#if BENCH_HAS_VECNN
  printf("  VEC: enabled (vecnn)\n");
#else
  printf("  VEC: disabled (vecnn not built)\n");
#endif
#else
  printf("  VEC: disabled\n");
#endif
}

void app_main(void) {
  printf("=== CORE-V-CONV BENCH ===\n");
  print_config();

  for (int i = 0; i < CORE_V_CONV_NUM_CASES; ++i) {
    bench_run_case(&CORE_V_CONV_CASES[i]);
  }

  printf("=== CORE-V-CONV BENCH DONE ===\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
