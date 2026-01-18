#include "simple_setup.h"

#define UART_BAUDRATE 115200u

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
