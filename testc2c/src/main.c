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
#include "simple_setup.h"

#define SPAD_TEST 0xc0000000

// L1: 8 KiB, 2-way, 64B lines → stride 0x1000 between same-set addresses
// L2: 256 KiB, 8-way, 64B lines → stride 0x8000 between same-set addresses
#define L1_WAYS       2
#define L1_SET_STRIDE 0x1000
#define L2_WAYS       8
#define L2_SET_STRIDE 0x8000
#define EVICT_COUNT   (L2_WAYS + L1_WAYS)

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
int count = 0;
uint64_t target_frequency = 150000000l;

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

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
}

void handle_sigint(int sig) {
  //printf("\nCaught signal %d, exiting...\n", sig);
  //exit(0);
}

void app_main() {
  char *msg = "It's April 9th, 2026. I am born :D\n";
  
  printf("%s", msg);

  *(volatile uint32_t *)SPAD_TEST = 0xDEADBEEF;
  *(volatile uint32_t *)(SPAD_TEST + 4) = 0xDEEABEEF;
  *(volatile uint32_t *)(SPAD_TEST + 8) = 0x1111BEEF;
  *(volatile uint32_t *)(SPAD_TEST + 12) = 0x2222BEEF;
  *(volatile uint32_t *)(SPAD_TEST + 16) = 0x3333BEEF;
  *(volatile uint32_t *)(SPAD_TEST + 20) = 0x4444BEEF;
  *(volatile uint32_t *)(SPAD_TEST + 24) = 0x4444BEEF;



  // Evict SPAD_TEST cache line from both L1 and L2 by filling their sets
  // First 8 accesses at L2 stride evict from L2, next 2 at L1 stride evict from L1
  static volatile uint8_t evict_buf[EVICT_COUNT * L2_SET_STRIDE]
    __attribute__((aligned(L2_SET_STRIDE)));
  for (int i = 0; i < EVICT_COUNT; i++) {
    evict_buf[i * L2_SET_STRIDE] = 0;
  }
  asm volatile("fence w, w" ::: "memory");

  while (1) {
      __asm__ volatile("wfi");
  }



}
/* USER CODE END PUC */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(int argc, char **argv) {
  /* MCU Configuration--------------------------------------------------------*/

  /* Configure the system clock */
  /* Configure the system clock */

  /* USER CODE BEGIN SysInit */
  /* USER CODE BEGIN SysInit */
  // Initialize UART0 for Serial Monitor
  // UART_InitType UART0_init_config;
  // UART0_init_config.baudrate = 115200;
  // UART0_init_config.mode = UART_MODE_TX_RX;
  // UART0_init_config.stopbits = UART_STOPBITS_2;
  // uart_init(UART0, &UART0_init_config);

  /* USER CODE END SysInit */

  init_test(target_frequency);
  // UART_InitType UART0_init_config;
  // UART0_init_config.baudrate = 115200;
  // UART0_init_config.mode = UART_MODE_TX_RX;
  // UART0_init_config.stopbits = UART_STOPBITS_2;
  // uart_init(UART0, &UART0_init_config);

  /* Initialize all configured peripherals */
  /* USER CODE BEGIN Init */
  app_init();
  signal(SIGINT, handle_sigint);
  /* USER CODE END Init */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  app_main();
  // while (1) {
  //   app_main();
  // }
  // return 0;
  /* USER CODE END WHILE */
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
