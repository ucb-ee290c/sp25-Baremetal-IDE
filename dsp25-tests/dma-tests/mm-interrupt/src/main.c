#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-interrupt"
uint64_t target_frequency = 150000000l;

int main(int argc, char **argv) {
  const size_t words = 96;
  const uintptr_t src = DMA_TEST_REGION0;
  const uintptr_t dst = DMA_TEST_REGION1;

  dma_transaction_t tx;
  bool finished = false;
  size_t hart;
  int fail;

  (void)argc;
  (void)argv;

  // Initialize UART0 for Serial Monitor
  // UART_InitType UART0_init_config;
  // UART0_init_config.baudrate = 115200;
  // UART0_init_config.mode = UART_MODE_TX_RX;
  // UART0_init_config.stopbits = UART_STOPBITS_2;
  // uart_init(UART0, &UART0_init_config);
  init_test(target_frequency);
  // UART_InitType UART0_init_config;
  // UART0_init_config.baudrate = 115200;
  // UART0_init_config.mode = UART_MODE_TX_RX;
  // UART0_init_config.stopbits = UART_STOPBITS_2;
  // uart_init(UART0, &UART0_init_config);

  printf("[%s] start\n", TEST_NAME);

  setup_interrupts();

  dma_test_fill_words(src, words, 23U);
  dma_test_zero_words(dst, words);

  tx.core = 0;
  tx.transaction_id = 0x40;
  tx.transaction_priority = 1;
  tx.peripheral_id = 0;
  tx.addr_r = src;
  tx.addr_w = dst;
  tx.inc_r = 4;
  tx.inc_w = 4;
  tx.len = (uint16_t)words;
  tx.logw = 2;
  tx.do_interrupt = true;
  tx.do_address_gate = false;

  if (!set_DMA_C(0, tx, true)) {
    printf("[%s] failed to configure channel\n", TEST_NAME);
    return 1;
  }

  start_DMA(0, tx.transaction_id, &finished);

  hart = read_csr(mhartid);
  dma_wait_till_done(hart, &finished);

  if (!finished) {
    printf("[%s] transaction did not complete\n", TEST_NAME);
    return 1;
  }

  fail = dma_test_expect_equal_words(src, dst, words, TEST_NAME);

  dma_reset();

  if (fail) {
    printf("[%s] FAIL\n", TEST_NAME);
    return 1;
  }

  printf("[%s] PASS\n", TEST_NAME);
  return 0;
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile("wfi");
  }
}
