/* =========================================================================
 * main.c - Bearly25 memory-latency microbenchmarks
 *
 * Orchestrates a set of L1/L2/DRAM/scratchpad/TCM latency tests and prints
 * CSV-style stats per core.
 * ========================================================================= */

#include <stdio.h>

#include "chip_config.h"
#include "memlat_core.h"
#include "memlat_tests.h"

void app_init(void) {
  UART_InitType UART0_init_config;
  UART0_init_config.baudrate = 115200;
  UART0_init_config.mode = UART_MODE_TX_RX;
  UART0_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART0_init_config);
}

void app_main(void) {
  int core_id = (int)memlat_read_hartid();

  if (core_id == 0) {
    printf("=== Bearly25 Memory-Latency Microbench ===\n");
    printf("# fields: core, region, mode, mean_cycles,"
           " min_cycles, p95_cycles, p99_cycles\n");
  }

  // 1) L1 hit latency
  memlat_run_l1_hit_test(core_id);
  // 2) L2 local / remote hit latency
  memlat_run_l2_local_hit_test(core_id);
  memlat_run_l2_remote_hit_test(core_id);
  // 3) DRAM cold-miss latency
  memlat_run_dram_cold_miss_test(core_id);
  // 4) Scratchpad hit latency
  memlat_run_scratchpad_hit_test(core_id);
  // 5) TCM hit latency
  memlat_run_tcm_hit_test(core_id);
  // 6) Scratchpad under NoC load
  memlat_run_scratchpad_under_noc_load(core_id);

  printf("mem-lat: all tests complete on core %d\n", core_id);
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
