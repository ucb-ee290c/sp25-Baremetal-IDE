/*
 * dma-mm-stride
 *
 * Tests non-contiguous (strided) DMA transfers.
 *
 * Sub-tests:
 *   1. Scatter: contiguous read, strided write  (pack → sparse)
 *   2. Gather:  strided read, contiguous write  (sparse → pack)
 *   3. Stride-to-stride: both sides non-contiguous
 *
 * This helps you understand how inc_r / inc_w work independently
 * of each other and of the packet width.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-stride"
uint64_t target_frequency = 150000000UL;

#define NUM_ELEMENTS 16
#define LOGW         2          /* 4-byte packets */
#define PACKET_BYTES 4

static int run_strided_test(const char *label,
                            uintptr_t src, uint16_t inc_r,
                            uintptr_t dst, uint16_t inc_w,
                            uint16_t tid) {
  dma_transaction_t tx;
  size_t i, start, elapsed;
  int fail = 0;

  printf("  %-20s  inc_r=%3u  inc_w=%3u ... ", label,
         (unsigned)inc_r, (unsigned)inc_w);

  /* Write known pattern into source at the read stride. */
  for (i = 0; i < NUM_ELEMENTS; ++i) {
    reg_write32(src + (i * (size_t)inc_r), (uint32_t)(i + 1));
  }

  /* Zero the destination region (generous size). */
  dma_test_zero_words(dst, NUM_ELEMENTS * 16);

  tx.core = 0;
  tx.transaction_id = tid;
  tx.transaction_priority = 1;
  tx.peripheral_id = 0;
  tx.addr_r = src;
  tx.addr_w = dst;
  tx.inc_r = inc_r;
  tx.inc_w = inc_w;
  tx.len = NUM_ELEMENTS;
  tx.logw = LOGW;
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

  /* Verify: element i should appear at dst + i * inc_w. */
  for (i = 0; i < NUM_ELEMENTS; ++i) {
    uint32_t expected = (uint32_t)(i + 1);
    uint32_t observed = reg_read32(dst + (i * (size_t)inc_w));
    if (observed != expected) {
      if (fail < 4) {
        printf("\n    mismatch[%u]: exp=0x%08x obs=0x%08x",
               (unsigned)i, expected, observed);
      }
      fail = 1;
    }
  }

  if (fail) {
    printf("  FAIL\n");
  } else {
    printf("PASS  (%lu ticks)\n", (unsigned long)elapsed);
  }

  return fail;
}

int main(int argc, char **argv) {
  int fail = 0;
  const uintptr_t region_a = DMA_TEST_REGION0;
  const uintptr_t region_b = DMA_TEST_REGION1;

  (void)argc;
  (void)argv;

  init_test(target_frequency);

  printf("[%s] start\n\n", TEST_NAME);

  /* 1. Contiguous read → contiguous write (baseline). */
  fail |= run_strided_test("contiguous",
                           region_a, PACKET_BYTES,
                           region_b, PACKET_BYTES, 0x50);

  /* 2. Scatter: contiguous read → strided write (skip every other word). */
  fail |= run_strided_test("scatter (gap=4B)",
                           region_a, PACKET_BYTES,
                           region_b, PACKET_BYTES * 2, 0x51);

  /* 3. Scatter: contiguous read → large stride write. */
  fail |= run_strided_test("scatter (gap=12B)",
                           region_a, PACKET_BYTES,
                           region_b, PACKET_BYTES * 4, 0x52);

  /* 4. Gather: strided read → contiguous write. */
  fail |= run_strided_test("gather (gap=4B)",
                           region_a, PACKET_BYTES * 2,
                           region_b, PACKET_BYTES, 0x53);

  /* 5. Gather: large stride read → contiguous write. */
  fail |= run_strided_test("gather (gap=12B)",
                           region_a, PACKET_BYTES * 4,
                           region_b, PACKET_BYTES, 0x54);

  /* 6. Both sides strided. */
  fail |= run_strided_test("stride-to-stride",
                           region_a, PACKET_BYTES * 3,
                           region_b, PACKET_BYTES * 2, 0x55);

  dma_reset();

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
