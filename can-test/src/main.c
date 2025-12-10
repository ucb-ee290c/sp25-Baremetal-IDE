/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "chip_config.h"
#include "rocketcore.h"
#include <inttypes.h>
#include <stdbool.h>

// PLL target frequency = 500 MHz
uint64_t target_frequency = 500000000l;


void app_init() {
  configure_pll(PLL, target_frequency/50000000, 0);
  set_all_clocks(RCC_CLOCK_SELECTOR, 1);

  UART_InitType UART_init_config;
  UART_init_config.baudrate = 115200;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART_init_config);
  UART0->DIV = (target_frequency / 115200) - 1;

  printf("Init done\r\n");
}


// NOTE: This is from DSP24 Audio
// https://github.com/ucb-bar/sp24-Baremetal-IDE/blob/audio/app/src/main.c
void can_test(void) {
  printf("CAN test start\r\n");
}


/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  /* MCU Configuration--------------------------------------------------------*/

  /* Configure the system clock */
  /* Configure the system clock */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */  
  /* USER CODE BEGIN Init */
  app_init();
  /* USER CODE END Init */
 
  // i2s_square_wave_test();
  can_test();

  return 0;
}

/*
 * Main function for secondary harts
 * 
 * Multi-threaded programs should provide their own implementation.
 */
void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
   asm volatile ("wfi");
  }
}