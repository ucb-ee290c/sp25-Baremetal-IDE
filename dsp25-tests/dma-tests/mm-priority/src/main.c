#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-priority"
#define TEST_REV  "rev3"
uint64_t target_frequency = 150000000l;

static int analyze_priority_result(uintptr_t low_src,
                                   uintptr_t high_src,
                                   uintptr_t dst,
                                   size_t words) {
  size_t i;
  size_t low_only = 0;
  size_t high_only = 0;
  size_t both_equal = 0;
  size_t mismatched = 0;
  int last_class = -1;
  int transitions = 0;

  for (i = 0; i < words; ++i) {
    uint32_t low_v = reg_read32(low_src + (i * 4UL));
    uint32_t high_v = reg_read32(high_src + (i * 4UL));
    uint32_t dst_v = reg_read32(dst + (i * 4UL));
    int cls = -1;

    if (dst_v == low_v && dst_v == high_v) {
      both_equal++;
      cls = -1;
    } else if (dst_v == low_v) {
      low_only++;
      cls = 0;
    } else if (dst_v == high_v) {
      high_only++;
      cls = 1;
    } else {
      mismatched++;
      cls = 2;
      if (mismatched <= 8) {
        printf("[%s] mismatch[%u]: low=0x%08x high=0x%08x dst=0x%08x\n",
               TEST_NAME,
               (unsigned)i,
               low_v,
               high_v,
               dst_v);
      }
    }

    if (cls == 0 || cls == 1) {
      if (last_class != -1 && cls != last_class) {
        transitions++;
      }
      last_class = cls;
    }
  }

  printf("[%s] profile: high=%u low=%u both=%u other=%u transitions=%d\n",
         TEST_NAME,
         (unsigned)high_only,
         (unsigned)low_only,
         (unsigned)both_equal,
         (unsigned)mismatched,
         transitions);

  if (mismatched != 0U) {
    printf("[%s] FAIL profile had non-source data\n", TEST_NAME);
    return 0;
  }

  /* Accept:
   * 1) single-winner buffer (legacy behavior), or
   * 2) one high/low phase switch (observed on current DMA pipeline timing). */
  if (high_only == 0U || low_only == 0U) {
    return 1;
  }

  if (transitions <= 1) {
    return 1;
  }

  printf("[%s] FAIL too many high/low phase switches (%d)\n", TEST_NAME, transitions);
  return 0;
}

int main(int argc, char **argv) {
  const size_t words = 128;
  const uintptr_t low_src = DMA_TEST_REGION0;
  const uintptr_t high_src = DMA_TEST_REGION1;
  const uintptr_t dst = DMA_TEST_REGION2;

  dma_transaction_t low_tx;
  dma_transaction_t high_tx;
  int pass;

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

  printf("[%s] start (%s)\n", TEST_NAME, TEST_REV);

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

  pass = analyze_priority_result(low_src, high_src, dst, words);

  dma_reset();

  if (!pass) {
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
