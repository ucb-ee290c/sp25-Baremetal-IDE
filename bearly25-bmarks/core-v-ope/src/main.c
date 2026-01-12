/*
 * main.c - Entry point for core-v-ope outer product benchmarks.
*/

#include <stdio.h>

#include "bench_config.h"
#include "bench_sizes.h"
#include "bench_impl.h"
#include "chip_config.h"

uint64_t target_frequency = 500000000l;

void app_init(void) {
  UART_InitType UART0_init_config;
  UART0_init_config.baudrate = 115200;
  UART0_init_config.mode = UART_MODE_TX_RX;
  UART0_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART0_init_config);

  // set_all_clocks(RCC_CLOCK_SELECTOR, 0);
  // configure_pll(PLL, 10, 0);
  // set_all_clocks(RCC_CLOCK_SELECTOR, 1);

  // UART0->DIV = (target_frequency / 115200) - 1;
}

static void print_config(void) {
  printf("  BENCH_RUNS=%d, BENCH_VERIFY=%d\n", BENCH_RUNS, BENCH_VERIFY);
#if BENCH_ENABLE_OPE
  printf("  OPE: enabled\n");
#else
  printf("  OPE: disabled\n");
#endif
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
  printf("=== CORE-V-OPE OUTER PRODUCT BENCH ===\n");
  print_config();

#if BENCH_ENABLE_OPE
  int max_M = 0;
  int max_N = 0;
  int max_K = 0;
  for (int i = 0; i < CORE_V_OPE_NUM_CASES; ++i) {
    const OuterSizeCase *cs = &CORE_V_OPE_CASES[i];
    if (cs->M > max_M) max_M = cs->M;
    if (cs->N > max_N) max_N = cs->N;
    if (cs->K > max_K) max_K = cs->K;
  }
  printf("  Initializing OPE workspace for max %dx%dx%d...\n", max_M, max_N, max_K);
  ope_init_workspace(max_M, max_N, max_K);
#endif

  for (int i = 0; i < CORE_V_OPE_NUM_CASES; ++i) {
    bench_run_case(&CORE_V_OPE_CASES[i]);
  }

#if BENCH_ENABLE_OPE
  ope_free_workspace();
#endif

  printf("=== CORE-V-OPE BENCH DONE ===\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
