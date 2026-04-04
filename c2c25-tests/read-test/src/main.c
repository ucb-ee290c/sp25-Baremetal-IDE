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

void app_main() {
  printf("In app_main.\n");
  while (1) {
    // uint32_t *c2c = (uint32_t *)0x180000000UL; 
    uint32_t *c2c = (uint32_t *) 0x40000000UL;
    printf("set c2c = address\n");
    uint32_t read;
    *c2c = 1;
    printf("set value in c2c to 1\n");
    read = *(uint32_t *)0x40000000UL;
    printf("set read to value %d\n", read);
    char *msg = "Writing 1 to C2C\n";
    puts(msg);
    printf("read: %d\n", read);
    // printf("%d\n", count);
    // count++;
    msleep(1000);
  }
}

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
