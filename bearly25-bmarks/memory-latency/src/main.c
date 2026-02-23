/* =========================================================================
 * main.c - Bearly25 memory-latency microbenchmarks
 *
 * Runs 7 latency tests from Core 0 using random pointer-chase rings:
 *   1. L1 Hit            - small working set fits in L1D
 *   2. L2 Local Hit      - medium working set, targets local L2 bank
 *   3. L2 Remote Hit     - medium working set, targets remote L2 bank
 *   4. DRAM              - large working set, overflows L2 entirely
 *   5. Scratchpad        - MBUS SRAM at 0x08000000
 *   6. Local TCM         - Core 0 TCM at 0x08010000
 *   7. Remote TCM        - Core 1 TCM at 0x08012000 (from Core 0)
 *
 * Output: one line per test with min / mean / median / max cycles/access.
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
  uint32_t hart = memlat_hartid();
  printf("\n=== Bearly25 Memory-Latency Benchmark (Core %lu) ===\n", (unsigned long)hart);
  printf("  Methodology: random pointer-chase ring (data-dependent loads)\n");
  printf("  L1D = 8 KB, L2 = 2x128 KB, bank_bit = addr[6]\n\n");

  memlat_test_l1_hit();
  memlat_test_l2_local_hit();
  memlat_test_l2_remote_hit();
  memlat_test_dram();

  memlat_test_scratchpad();
  memlat_test_local_tcm();
  memlat_test_remote_tcm();

  printf("\n=== All tests complete ===\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
