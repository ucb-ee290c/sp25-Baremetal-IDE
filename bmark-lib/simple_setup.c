#include "simple_setup.h"

#define UART_BAUDRATE 115200u

static void sleep_ms_blocking(uint32_t sleep_ms) {
  if (sleep_ms == 0u) {
    return;
  }

  uint64_t ticks = ((uint64_t)sleep_ms * (uint64_t)MTIME_FREQ + 999u) / 1000u;
  if (ticks == 0u) {
    ticks = 1u;
  }

  uint64_t start = clint_get_time(CLINT);
  while ((clint_get_time(CLINT) - start) < ticks) {
    asm volatile("nop");
  }
}

void init_test(uint64_t target_frequency) {
  UART_InitType uart_init_config;
  uart_init_config.baudrate = UART_BAUDRATE;
  uart_init_config.mode = UART_MODE_TX_RX;
  uart_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &uart_init_config);

  uint32_t pll_ratio = (uint32_t)(target_frequency / SYS_CLK_FREQ);
  if (pll_ratio == 0) {
    pll_ratio = 1;
  }

  set_all_clocks(RCC_CLOCK_SELECTOR, CLKSEL_SLOW);
  configure_pll(PLL, pll_ratio, 0);
  set_all_clocks(RCC_CLOCK_SELECTOR, CLKSEL_PLL0);

  UART0->DIV = (uint32_t)((target_frequency / UART_BAUDRATE) - 1u);
}

void reconfigure_pll(uint64_t target_frequency, uint32_t sleep_ms) {
  sleep_ms_blocking(sleep_ms);

  uint32_t pll_ratio = (uint32_t)(target_frequency / SYS_CLK_FREQ);
  if (pll_ratio == 0u) {
    pll_ratio = 1u;
  }

  set_all_clocks(RCC_CLOCK_SELECTOR, CLKSEL_SLOW);
  configure_pll(PLL, pll_ratio, 0);
  set_all_clocks(RCC_CLOCK_SELECTOR, CLKSEL_PLL0);

  UART0->DIV = (uint32_t)((target_frequency / UART_BAUDRATE) - 1u);

  sleep_ms_blocking(sleep_ms);
}

uint64_t rdcycle(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}
