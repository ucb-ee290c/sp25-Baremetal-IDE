#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "hal_dma.h"
#include "dma_test_utils.h"
#include "simple_setup.h"

#define TEST_NAME "dma-mm-memcpy-basic"
uint64_t target_frequency = 150000000l;

static int run_case(uint8_t logw) {
  const size_t total_bytes = 256;
  const size_t words = total_bytes / 4U;
  const uintptr_t src = DMA_TEST_REGION0;
  const uintptr_t dst = DMA_TEST_REGION1;
  const uint16_t tid = (uint16_t)(0x60U + logw);
  const uint16_t packet_bytes = (uint16_t)(1U << logw);
  const uint16_t packets = (uint16_t)(total_bytes / packet_bytes);

  dma_transaction_t tx;
  int fail;
  size_t start;
  size_t elapsed;

  dma_test_fill_words(src, words, (uint32_t)(0x31U + logw));
  dma_test_zero_words(dst, words);

  tx.core = 0;
  tx.transaction_id = tid;
  tx.transaction_priority = 1;
  tx.peripheral_id = 0;
  tx.addr_r = src;
  tx.addr_w = dst;
  tx.inc_r = packet_bytes;
  tx.inc_w = packet_bytes;
  tx.len = packets;
  tx.logw = logw;
  tx.do_interrupt = false;
  tx.do_address_gate = false;

  printf("  logw=%u  packet=%u B  packets=%u ... ",
         (unsigned)logw,
         (unsigned)packet_bytes,
         (unsigned)packets);

  if (!set_DMA_C(0, tx, true)) {
    printf("FAIL (config)\n");
    return 1;
  }

  start = ticks();
  start_DMA(0, tid, NULL);
  dma_wait_till_inactive(20);
  elapsed = ticks() - start;

  fail = dma_test_expect_equal_words(src, dst, words, TEST_NAME);
  dma_reset();

  if (fail) {
    printf("FAIL\n");
    return 1;
  }

  printf("PASS  (%lu ticks)\n", (unsigned long)elapsed);
  return 0;
}

int main(int argc, char **argv) {
  uint8_t logw;
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

  printf("[%s] start\n", TEST_NAME);
  printf("  Copying 256 bytes at each logw setting\n\n");

  for (logw = 0; logw <= 3U; ++logw) {
    fail |= run_case(logw);
  }

  if (fail) {
    printf("\n[%s] FAIL\n", TEST_NAME);
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
