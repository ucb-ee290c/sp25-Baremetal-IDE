/*
 * dma-mm-memcpy-basic
 *
 * Simple DMA memory-to-memory copy at each supported packet width
 * (logw = 0..3 → 1, 2, 4, 8 bytes per packet).  Verifies correctness
 * at every width so you can see how logw / inc_r / inc_w relate.
 *
 * No interrupts — polls for completion.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-memcpy-basic"
uint64_t target_frequency = 150000000UL;

/* Total bytes to copy in each sub-test. */
#define TOTAL_BYTES 256

static int test_memcpy(uint8_t logw, uint16_t tid_base) {
  const size_t packet_bytes = 1U << logw;          /* 1, 2, 4, or 8 */
  const size_t packets = TOTAL_BYTES / packet_bytes;
  const uintptr_t src = DMA_TEST_REGION0;
  const uintptr_t dst = DMA_TEST_REGION1;
  dma_transaction_t tx;
  size_t start, elapsed;
  int fail;

  printf("  logw=%u  packet=%u B  packets=%u ... ",
         (unsigned)logw, (unsigned)packet_bytes, (unsigned)packets);

  /* Fill source, zero destination. */
  dma_test_fill_words(src, TOTAL_BYTES / 4, tid_base);
  dma_test_zero_words(dst, TOTAL_BYTES / 4);

  tx.core = 0;
  tx.transaction_id = tid_base;
  tx.transaction_priority = 1;
  tx.peripheral_id = 0;
  tx.addr_r = src;
  tx.addr_w = dst;
  tx.inc_r = (uint16_t)packet_bytes;   /* contiguous read  */
  tx.inc_w = (uint16_t)packet_bytes;   /* contiguous write */
  tx.len = (uint16_t)packets;
  tx.logw = logw;
  tx.do_interrupt = false;
  tx.do_address_gate = false;

  if (!set_DMA_C(0, tx, true)) {
    printf("FAIL (config)\n");
    return 1;
  }

  start = ticks();
  start_DMA(0, tx.transaction_id, NULL);
  dma_wait_till_inactive(20);
  elapsed = ticks() - start;

  fail = dma_test_expect_equal_words(src, dst, TOTAL_BYTES / 4, TEST_NAME);
  dma_reset();

  if (fail) {
    printf("FAIL\n");
  } else {
    printf("PASS  (%lu ticks)\n", (unsigned long)elapsed);
  }

  return fail;
}

int main(int argc, char **argv) {
  int fail = 0;

  (void)argc;
  (void)argv;

  init_test(target_frequency);

  printf("[%s] start\n", TEST_NAME);
  printf("  Copying %u bytes at each logw setting\n\n", TOTAL_BYTES);

  fail |= test_memcpy(0, 0x01);   /* 1-byte packets  */
  fail |= test_memcpy(1, 0x02);   /* 2-byte packets  */
  fail |= test_memcpy(2, 0x03);   /* 4-byte packets  */
  fail |= test_memcpy(3, 0x04);   /* 8-byte packets  */

  printf("\n");
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
