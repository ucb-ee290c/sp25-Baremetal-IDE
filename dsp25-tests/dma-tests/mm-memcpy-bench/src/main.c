#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-memcpy-bench"
uint64_t target_frequency = 150000000l;

static size_t cpu_copy_words(uintptr_t src, uintptr_t dst, size_t words) {
  size_t i;
  size_t start = ticks();

  for (i = 0; i < words; ++i) {
    reg_write32(dst + (i * 4UL), reg_read32(src + (i * 4UL)));
  }

  return ticks() - start;
}

static size_t dma_copy_words(uintptr_t src, uintptr_t dst, size_t words, uint16_t tid) {
  dma_transaction_t tx;
  size_t start;

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
  tx.do_address_gate = false;

  if (!set_DMA_C(0, tx, true)) {
    return 0;
  }

  start = ticks();
  start_DMA(0, tid, NULL);
  dma_wait_till_inactive(20);

  return ticks() - start;
}

int main(int argc, char **argv) {
  const size_t sizes[] = {8, 16, 32, 64, 128, 256};
  const uintptr_t src = DMA_TEST_REGION0;
  const uintptr_t cpu_dst = DMA_TEST_REGION1;
  const uintptr_t dma_dst = DMA_TEST_REGION2;
  const size_t max_words = 256;
  size_t i;
  int fail = 0;

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

  printf("[%s] start\n\n", TEST_NAME);
  printf("   words   cpu_ticks   dma_ticks  cpu_B/tick  dma_B/tick\n");
  printf("  ------  ----------  ----------  ----------  ----------\n");

  for (i = 0; i < (sizeof(sizes) / sizeof(sizes[0])); ++i) {
    size_t words = sizes[i];
    size_t bytes = words * 4U;
    size_t cpu_ticks;
    size_t dma_ticks;
    double cpu_rate;
    double dma_rate;

    dma_test_fill_words(src, max_words, (uint32_t)(0x97U + words));
    dma_test_zero_words(cpu_dst, max_words);
    dma_test_zero_words(dma_dst, max_words);

    cpu_ticks = cpu_copy_words(src, cpu_dst, words);
    dma_ticks = dma_copy_words(src, dma_dst, words, (uint16_t)(0x80U + i));

    if (dma_ticks == 0U) {
      printf("%8u  %10lu  %10s  %10s  %10s\n",
             (unsigned)words,
             (unsigned long)cpu_ticks,
             "FAIL",
             "-",
             "-");
      fail = 1;
      continue;
    }

    cpu_rate = (cpu_ticks == 0U) ? 0.0 : ((double)bytes / (double)cpu_ticks);
    dma_rate = (dma_ticks == 0U) ? 0.0 : ((double)bytes / (double)dma_ticks);

    printf("%8u  %10lu  %10lu  %10.2f  %10.2f\n",
           (unsigned)words,
           (unsigned long)cpu_ticks,
           (unsigned long)dma_ticks,
           cpu_rate,
           dma_rate);

    fail |= dma_test_expect_equal_words(src, dma_dst, words, "dma");
  }

  dma_reset();

  if (fail) {
    printf("\n[%s] FAIL (data mismatch)\n", TEST_NAME);
    return 1;
  }

  printf("\n[%s] PASS\n", TEST_NAME);
  return 0;
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile("wfi");
  }
}
