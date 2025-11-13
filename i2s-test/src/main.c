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

// I2S at 44kHz -> ~440hz square wave
#define PULSE_WIDTH_SAMPLES (50)
#define PULSE_PERIOD_SAMPLES (100)
#define AMPLITUDE (0x7FFFFF) // Sample amplitude for 32 bit depth

#define CHANNEL 0

I2S_PARAMS I2S_DEFAULT = {
    .channel     = 0,
    .tx_en       = 1,
    .rx_en       = 1,
    .bitdepth_tx = 3, // 32 bits
    .bitdepth_rx = 3,
    .clkgen      = 1,
    .dacen       = 0,
    .ws_len      = 3,
    .clkdiv      = 176,
    .tx_fp       = 0,
    .rx_fp       = 0,
    .tx_force_left = 0,
    .rx_force_left = 0
};

uint64_t target_frequency = 500000000l;


void app_init() {
  configure_pll(PLL_BASE, target_frequency/50000000, 0);
  set_all_clocks(RCC_CLOCK_SELECTOR, 1);

  uint64_t mhartid = READ_CSR("mhartid");

  printf("(BEGIN) On hart: %d", mhartid);

  printf("I2S params initializing");

  set_I2S_params(&I2S_DEFAULT);

  printf("Init done");
}

void i2s_square_wave_test(void) {

  uint32_t counter = 0;
  uint32_t sample[2] = {AMPLITUDE, AMPLITUDE};
  while (1) {
    // Divide by 4 because 4 samples fit in one 64 bit transaction
    uint64_t data;
    if (counter < PULSE_WIDTH_SAMPLES / 4) {
      // TODO: Need to verify sample order here
      data = ((uint64_t)sample[0] << 32) | (uint64_t)sample[1];
    } else {
      data = 0;
    }

    write_I2S_tx(CHANNEL, true, data);
    write_I2S_tx(CHANNEL, false, data);

    counter = (counter + 1) % (PULSE_PERIOD_SAMPLES / 4);
	}

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
  UART_InitType UART_init_config;
  UART_init_config.baudrate = 115200;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0_BASE, &UART_init_config);
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */  
  /* USER CODE BEGIN Init */
  app_init();
  /* USER CODE END Init */

  i2s_square_wave_test();

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