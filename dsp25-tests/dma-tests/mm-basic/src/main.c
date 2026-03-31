#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"

#define TEST_NAME "dma-mm-basic"

int main(int argc, char **argv) {
  const size_t words = 256;
  const uintptr_t src = DMA_TEST_REGION0;
  const uintptr_t dst = DMA_TEST_REGION1;
  int fail;
  dma_transaction_t tx;

  (void)argc;
  (void)argv;

  printf("[%s] start\n", TEST_NAME);

  dma_test_fill_words(src, words, 1U);
  dma_test_zero_words(dst, words);

  tx.core = 0;
  tx.transaction_id = 0x10;
  tx.transaction_priority = 1;
  tx.peripheral_id = 0;
  tx.addr_r = src;
  tx.addr_w = dst;
  tx.inc_r = 4;
  tx.inc_w = 4;
  tx.len = (uint16_t)words;
  tx.logw = 2;
  tx.do_interrupt = false;
  tx.do_address_gate = false;

  if (!set_DMA_C(0, tx, true)) {
    printf("[%s] failed to configure channel\n", TEST_NAME);
    return 1;
  }

  start_DMA(0, tx.transaction_id, NULL);
  dma_wait_till_inactive(20);

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
