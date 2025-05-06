/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body for 2D Convolution Test
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
#include "hal_dma.h"
#include "chip_config.h"
#include <stdlib.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

dma_transaction_t r;

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */


/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN PUC */

void app_init() {
    // Initialize the 2D convolution engine
    setup_interrupts();
    printf("TEST (%s): Starting test\n", TEST);

    // load sample data
    for (int i = 0; i < LEN; i++) reg_write32(MEM_ADDR_1 + 4 * i, i + 1);
    r.core = read_csr(mhartid);
    r.transaction_id = 0;
    r.transaction_priority = 1;
    r.addr_r = MEM_ADDR_1;
    r.addr_w = MEM_ADDR_2;
    r.inc_r = WIDTH;
    r.inc_w = WIDTH;
    r.len = LEN * 4 / WIDTH;
    r.logw = LOGW;
    r.do_interrupt = INTERRUPT;
    r.do_address_gate = GATED;

    printf("TEST (%s): Done writing initial data\n", TEST);
}

void app_main() {
    size_t start = ticks();

    bool finished = false;
    set_DMA_C(0, r, true);
    start_DMA(0, 0, &finished);

    if (INTERRUPT) dma_wait_till_done(0, &finished);
    else dma_wait_till_inactive(5);

    // check
    dma_reset();
    printf("TEST (%s): %ld ticks, checking result\n", TEST, ticks() - start);
    int fail = 0;
    for (int i = 0; i < LEN; i++) fail |= check_val32(i, i + 1, MEM_ADDR_2 + 4 * i, PRINT_ON_ERROR);

    
    if (!fail) printf("TEST (%s): All tests passed!\n", TEST);
}

// Simple main function that just runs once
int main() {
    app_init();
    app_main();
    
    // Add a small delay to ensure output is printed
    // for (volatile int i = 0; i < 1000; i++);
    
    return 0;
}
/* USER CODE END PUC */

/**
  * @brief  The application entry point.
  * @retval int
  */
// int main(int argc, char **argv) {
//   /* MCU Configuration--------------------------------------------------------*/

//   /* Configure the system clock */
//   /* USER CODE BEGIN SysInit */

//   /* USER CODE END SysInit */

//   /* Initialize all configured peripherals */  
//   /* USER CODE BEGIN Init */
//   app_init();
//   /* USER CODE END Init */

//   /* Infinite loop */
//   /* USER CODE BEGIN WHILE */
//   while (1) {
//     app_main();
//     return 0;
//   }
//   /* USER CODE END WHILE */
// }

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