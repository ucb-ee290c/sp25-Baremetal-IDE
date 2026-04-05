/*
 * main.c - Hello-world bring-up test for Bearly25.
 *
 * Initializes UART (optional), prints hart ID, and serves as a basic
 * binary load/smoke test.
 */
#include "main.h"
#include "chip_config.h"
// #include "simple_setup.h"

int count = 0;
// uint64_t target_frequency = 500000000l;

void app_init() {
  printf("In the init function.\n");
}

void handle_sigint(int sig) {
  //printf("\nCaught signal %d, exiting...\n", sig);
  //exit(0);
}

void c2c_write_dram_read() {
  unsigned long addr = 0x480010000;
  uint32_t *dram = (uint32_t*) 0x80010000;
  *dram = 0xDEADBEEF;
  printf("set dram to %x\n", *dram);
  msleep(500);
  uint32_t *c2c = (uint32_t *) addr;
  printf("set c2c = address %lx\n", addr);
  printf("reading: %x\n", *c2c);
  msleep(1000);
}

void dram_write_c2c_read() {
  unsigned long addr = 0x480000000;
  uint32_t *dram = (uint32_t*) 0x80000000;
  // printf("set value in dram to %x\n", *dram);
  uint32_t *c2c = (uint32_t *) addr;
  printf("set c2c = address %lx\n", addr);
  *c2c = 0xDEADBEEF;
  msleep(500);
  printf("reading from dram: %x\n", *dram);
  msleep(1000);  
}

void app_main() {
  printf("In app_main.\n");
  while (1) {
    c2c_write_dram_read();
    // dram_write_c2c_read();
  }
}

// ila_0 ila (
//   clk(_dutWrangler_auto_out_3_clock),

//   probe0(serial_tl_1_in_bits_phit_IBUF),
//   probe1(serial_tl_1_out_bits_phit_OBUF),
//   probe2(serial_tl_1_clock_in_IBUF_BUFG),
//   probe3(serial_tl_1_clock_out_OBUF),
//   probe4(serial_tl_1_in_valid_IBUF),
//   probe5(serial_tl_1_out_valid_OBUF),
//   probe6(serial_tl_1_reset_in_IBUF),
//   probe7(serial_tl_1_reset_in_OBUF)
//   );

int main(void) {
  UART_InitType UART0_init_config;
  UART0_init_config.baudrate = 115200;
  UART0_init_config.mode = UART_MODE_TX_RX;
  UART0_init_config.stopbits = UART_STOPBITS_2;

  uart_init(UART0, &UART0_init_config);
  app_init();
  signal(SIGINT, handle_sigint);
  app_main();
  return 0;
}

/*
 * Main function for secondary harts
 *
 * Multi-threaded programs should provide their own implementation.
 */
 void __attribute__((weak, noreturn)) __main(void) {
  uint64_t mhartid = READ_CSR("mhartid");
  while (1) {
    asm volatile("wfi");
  }
}
