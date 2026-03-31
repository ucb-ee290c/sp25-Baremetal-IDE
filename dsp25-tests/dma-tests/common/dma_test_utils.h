#ifndef DSP25_DMA_TEST_UTILS_H
#define DSP25_DMA_TEST_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_mmio.h"

#define DMA_TEST_REGION0 (0x85000000UL)
#define DMA_TEST_REGION1 (0x85001000UL)
#define DMA_TEST_REGION2 (0x85002000UL)
#define DMA_TEST_REGION3 (0x85003000UL)

static inline uint32_t dma_test_pattern(uint32_t seed, size_t idx) {
  return (seed * 0x9e3779b1U) ^ (uint32_t)(idx * 0x45d9f3bU + seed);
}

static inline void dma_test_fill_words(uintptr_t base, size_t words, uint32_t seed) {
  size_t i;
  for (i = 0; i < words; ++i) {
    reg_write32(base + (i * 4UL), dma_test_pattern(seed, i));
  }
}

static inline void dma_test_zero_words(uintptr_t base, size_t words) {
  size_t i;
  for (i = 0; i < words; ++i) {
    reg_write32(base + (i * 4UL), 0U);
  }
}

static inline int dma_test_expect_equal_words(uintptr_t expected_base,
                                              uintptr_t observed_base,
                                              size_t words,
                                              const char *label) {
  size_t i;
  int fail = 0;

  for (i = 0; i < words; ++i) {
    uint32_t expected = reg_read32(expected_base + (i * 4UL));
    uint32_t observed = reg_read32(observed_base + (i * 4UL));
    if (expected != observed) {
      if (fail < 8) {
        printf("[%s] mismatch[%u]: exp=0x%08x obs=0x%08x\n",
               label,
               (unsigned)i,
               expected,
               observed);
      }
      fail = 1;
    }
  }

  return fail;
}

#endif /* DSP25_DMA_TEST_UTILS_H */
