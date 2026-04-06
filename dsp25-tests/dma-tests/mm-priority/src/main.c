#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-priority"
uint64_t target_frequency = 150000000l;

static int dma_words_equal(uintptr_t a, uintptr_t b, size_t words) {
  size_t i;
  for (i = 0; i < words; ++i) {
    if (reg_read32(a + (i * 4UL)) != reg_read32(b + (i * 4UL))) {
      return 0;
    }
  }
  return 1;
}

int main(int argc, char **argv) {
  const size_t words = 128;
  const uintptr_t low_src = DMA_TEST_REGION0;
  const uintptr_t high_src = DMA_TEST_REGION1;
  const uintptr_t dst = DMA_TEST_REGION2;

  dma_transaction_t low_tx;
  dma_transaction_t high_tx;
  int low_match;
  int high_match;

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

  dma_test_fill_words(low_src, words, 3U);
  dma_test_fill_words(high_src, words, 7U);
  dma_test_zero_words(dst, words);

  low_tx.core = 0;
  low_tx.transaction_id = 0x20;
  low_tx.transaction_priority = 1;
  low_tx.peripheral_id = 0;
  low_tx.addr_r = low_src;
  low_tx.addr_w = dst;
  low_tx.inc_r = 4;
  low_tx.inc_w = 4;
  low_tx.len = (uint16_t)words;
  low_tx.logw = 2;
  low_tx.do_interrupt = false;
  low_tx.do_address_gate = false;

  high_tx = low_tx;
  high_tx.transaction_id = 0x21;
  high_tx.transaction_priority = 3;
  high_tx.addr_r = high_src;

  if (!set_DMA_C(0, low_tx, true) || !set_DMA_C(1, high_tx, true)) {
    printf("[%s] failed to configure channels\n", TEST_NAME);
    return 1;
  }

  start_DMA(0, low_tx.transaction_id, NULL);
  start_DMA(1, high_tx.transaction_id, NULL);
  dma_wait_till_inactive(30);

  low_match = dma_words_equal(low_src, dst, words);
  high_match = dma_words_equal(high_src, dst, words);

  dma_reset();

  if (!low_match && !high_match) {
    (void)dma_test_expect_equal_words(low_src, dst, words, TEST_NAME "-low-src");
    (void)dma_test_expect_equal_words(high_src, dst, words, TEST_NAME "-high-src");
    printf("[%s] FAIL\n", TEST_NAME);
    return 1;
  }

  if (low_match) {
    printf("[%s] winner=low-src\n", TEST_NAME);
  } else {
    printf("[%s] winner=high-src\n", TEST_NAME);
  }

  printf("[%s] PASS\n", TEST_NAME);
  return 0;
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile("wfi");
  }
}
