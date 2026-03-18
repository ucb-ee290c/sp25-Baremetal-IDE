/*
 * main.c - Hello-world bring-up test for Bearly25.
 *
 * Initializes UART (optional), prints hart ID, and serves as a basic
 * binary load/smoke test.
 */
#include "main.h"
#include "chip_config.h"

void app_init() {
  printf("In the init function.\n");
}

void app_main() {
  while (1) {
    // uint32_t *c2c = (uint32_t *)0x180000000UL; 
    // 0x180000000 = to second chip + 0x80000000 for second chip's scratchpad
    uint32_t *c2c = (uint32_t *)0x200000000UL;
    *c2c = 1;
    printf("Writing 1 to C2C\n");
    sleep(2);
  }
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
