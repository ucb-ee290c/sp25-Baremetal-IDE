/*
 * dma-mm-memcpy-bench
 *
 * Bandwidth benchmark: CPU word-by-word copy  vs  DMA copy.
 * Sweeps several transfer sizes and prints tick counts + bytes/tick
 * so you can characterize the DMA crossover point and peak throughput.
 *
 * Uses logw=3 (8-byte / 64-bit packets) for maximum DMA bandwidth.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-memcpy-bench"
uint64_t target_frequency = 150000000UL;

#define SRC_BASE DMA_TEST_REGION0
#define DST_BASE DMA_TEST_REGION1

/* CPU baseline: copy `words` 32-bit words from src to dst using the core. */
static size_t cpu_memcpy_words(uintptr_t src, uintptr_t dst, size_t words) {
  size_t i;
  size_t start = ticks();

  for (i = 0; i < words; ++i) {
    reg_write32(dst + (i * 4UL), reg_read32(src + (i * 4UL)));
  }

  return ticks() - start;
}

/* DMA copy: uses 64-bit packets (logw=3) with contiguous stride. */
static size_t dma_memcpy_words(uintptr_t src, uintptr_t dst, size_t words,
                               uint16_t tid) {
  dma_transaction_t tx;
  size_t start, elapsed;

  tx.core = 0;
  tx.transaction_id = tid;
  tx.transaction_priority = 1;
  tx.peripheral_id = 0;
  tx.addr_r = src;
  tx.addr_w = dst;
  tx.inc_r = 8;
  tx.inc_w = 8;
  tx.len = (uint16_t)((words * 4) / 8);  /* bytes -> 8-byte packets */
  tx.logw = 3;
  tx.do_interrupt = false;
  tx.do_address_gate = false;

  if (!set_DMA_C(0, tx, true)) {
    return 0;
  }

  start = ticks();
  start_DMA(0, tx.transaction_id, NULL);
  dma_wait_till_inactive(20);
  elapsed = ticks() - start;

  dma_reset();
  return elapsed;
}

static int verify(uintptr_t src, uintptr_t dst, size_t words,
                  const char *label) {
  return dma_test_expect_equal_words(src, dst, words, label);
}

int main(int argc, char **argv) {
  /* Transfer sizes in 32-bit words (must be multiple of 2 for 64-bit DMA). */
  static const size_t sizes[] = {8, 16, 32, 64, 128, 256, 512};
  const size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);
  size_t si;
  int fail = 0;

  (void)argc;
  (void)argv;

  init_test(target_frequency);

  printf("[%s] start\n", TEST_NAME);
  printf("\n");
  printf("  %6s  %10s  %10s  %10s  %10s\n",
         "words", "cpu_ticks", "dma_ticks", "cpu_B/tick", "dma_B/tick");
  printf("  %6s  %10s  %10s  %10s  %10s\n",
         "------", "----------", "----------", "----------", "----------");

  for (si = 0; si < nsizes; ++si) {
    size_t words = sizes[si];
    size_t bytes = words * 4;
    size_t cpu_t, dma_t;
    uint16_t tid = (uint16_t)(0x10 + si);

    /* Fill source, zero destination for CPU test. */
    dma_test_fill_words(SRC_BASE, words, (uint32_t)(si + 1));
    dma_test_zero_words(DST_BASE, words);

    cpu_t = cpu_memcpy_words(SRC_BASE, DST_BASE, words);
    fail |= verify(SRC_BASE, DST_BASE, words, "cpu");

    /* Zero destination for DMA test. */
    dma_test_zero_words(DST_BASE, words);

    dma_t = dma_memcpy_words(SRC_BASE, DST_BASE, words, tid);
    fail |= verify(SRC_BASE, DST_BASE, words, "dma");

    /* Print results.  Multiply by 100 to get fixed-point x100 B/tick. */
    printf("  %6u  %10lu  %10lu",
           (unsigned)words, (unsigned long)cpu_t, (unsigned long)dma_t);

    if (cpu_t > 0) {
      printf("  %7lu.%02lu", (unsigned long)(bytes / cpu_t),
             (unsigned long)((bytes * 100 / cpu_t) % 100));
    } else {
      printf("  %10s", "inf");
    }

    if (dma_t > 0) {
      printf("  %7lu.%02lu", (unsigned long)(bytes / dma_t),
             (unsigned long)((bytes * 100 / dma_t) % 100));
    } else {
      printf("  %10s", "inf");
    }

    printf("\n");
  }

  printf("\n");
  if (fail) {
    printf("[%s] FAIL (data mismatch)\n", TEST_NAME);
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
