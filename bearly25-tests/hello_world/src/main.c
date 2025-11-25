#include "main.h"
#include "chip_config.h"

void app_init() {
  printf("In the init function.\n");
}

void app_main() {

  // UART_InitType UART_init_config;
  // UART_init_config.baudrate = 115200;
  // UART_init_config.mode = UART_MODE_TX_RX;
  // UART_init_config.stopbits = UART_STOPBITS_2;
  // uart_init(UART0, &UART_init_config);

  long chip_freq;
  long chip_mtime_freq;
  const char header_ack = 0x06;
  UART_Type *debug_uart;
  debug_uart = UART0;

  int packet_size;
  char header_start;

  UART_InitType UART_init_config;
  UART_init_config.baudrate = 115200;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  uart_init(debug_uart, &UART_init_config);

  // while (1) {
  //   // Waits for SOH (Start of Header) 0x01 or UART ENQ (Enquiry) 0x05
  //   uart_receive(debug_uart, &header_start, 1, 0);
    
  //   if (header_start == 0x05) {
  //     // If ENQ (Enquiry), send back Acknowledge (0x06) to say we are valid.
  //     uart_transmit(debug_uart, &header_ack, 1, 0);
  //   } else if (header_start == 0x01) {
  //     // If SOH (Start of Header), prepare to read a header.
  //     break;
  //   }
  // }

  // uart_receive(debug_uart, &packet_size, 4, 0);
  // uart_receive(debug_uart, &chip_freq, 8, 0);
  // uart_receive(debug_uart, &(t.testid), 1, 0);

  int uart_divisor = (chip_freq / 115200) - 1;
  debug_uart->DIV = uart_divisor;

  uint64_t mhartid = READ_CSR("mhartid");
  printf("Hello world from hart %ld!\n", mhartid);
  printf("In the main function.\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}