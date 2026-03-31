#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"

#define TEST_NAME "dma-mm-many"

static dma_transaction_t make_tx(uint16_t tid, uintptr_t src, uintptr_t dst, size_t words) {
  dma_transaction_t tx;

  tx.core = 0;
  tx.transaction_id = tid;
  tx.transaction_priority = 1;
  tx.peripheral_id = 0;
  tx.addr_r = src;
  tx.addr_w = dst;
  tx.inc_r = 4;
  tx.inc_w = 4;
  tx.len = (uint16_t)words;
  tx.logw = 2;
  tx.do_interrupt = false;
  tx.do_address_gate = true;

  return tx;
}

int main(int argc, char **argv) {
  const size_t words = 64;

  const uintptr_t src0 = DMA_TEST_REGION0;
  const uintptr_t src1 = DMA_TEST_REGION1;
  const uintptr_t src2 = DMA_TEST_REGION2;

  const uintptr_t dst0 = DMA_TEST_REGION3 + 0x000UL;
  const uintptr_t dst1 = DMA_TEST_REGION3 + 0x400UL;
  const uintptr_t dst2 = DMA_TEST_REGION3 + 0x800UL;
  const uintptr_t dst3 = DMA_TEST_REGION3 + 0xC00UL;

  int fail = 0;

  (void)argc;
  (void)argv;

  printf("[%s] start\n", TEST_NAME);

  dma_test_fill_words(src0, words, 11U);
  dma_test_fill_words(src1, words, 13U);
  dma_test_fill_words(src2, words, 17U);

  dma_test_zero_words(dst0, words);
  dma_test_zero_words(dst1, words);
  dma_test_zero_words(dst2, words);
  dma_test_zero_words(dst3, words);

  if (!set_DMA_C(0, make_tx(0x30, src0, dst0, words), true)) {
    printf("[%s] failed to configure channel 0\n", TEST_NAME);
    return 1;
  }
  start_DMA(0, 0x30, NULL);

  if (!set_DMA_C(1, make_tx(0x31, src1, dst1, words), true)) {
    printf("[%s] failed to configure channel 1 (tx1)\n", TEST_NAME);
    return 1;
  }
  start_DMA(1, 0x31, NULL);

  /* Queue a second transaction on the same channel. */
  if (!set_DMA_C(1, make_tx(0x32, dst1, dst3, words), true)) {
    printf("[%s] failed to configure channel 1 (tx2)\n", TEST_NAME);
    return 1;
  }
  start_DMA(1, 0x32, NULL);

  if (!set_DMA_C(2, make_tx(0x33, src2, dst2, words), true)) {
    printf("[%s] failed to configure channel 2\n", TEST_NAME);
    return 1;
  }
  start_DMA(2, 0x33, NULL);

  dma_wait_till_inactive(40);

  fail |= dma_test_expect_equal_words(src0, dst0, words, TEST_NAME "-tx0");
  fail |= dma_test_expect_equal_words(src1, dst1, words, TEST_NAME "-tx1");
  fail |= dma_test_expect_equal_words(src2, dst2, words, TEST_NAME "-tx3");
  fail |= dma_test_expect_equal_words(src1, dst3, words, TEST_NAME "-queued");

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
