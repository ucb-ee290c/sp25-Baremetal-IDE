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

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

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

uint8_t counter = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */


/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN PUC */

void app_init() {
  // torch::executor::runtime_init();
}

union Converter {
    float f;
    uint32_t u;
};

void app_main() {
  uint64_t mhartid = READ_CSR("mhartid");

  printf("Hello world from hart %d: %d\n", mhartid, counter);
}

// -----------------------------------------------------------------------------
// Cycle counter
// -----------------------------------------------------------------------------
static uint64_t read_cycles(void) {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}


#define IN_LEN 48
#define KERNEL_LEN 8
#define IN_DILATION 1

#include <stdbool.h>

void first_convolution() {

  // IO:

  // const uint32_t in_len = 48;
  // const uint8_t KERNEL_LEN = 8;

  // Config:

  // Input Vector (FP32): 
  // 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 9.0, 7.0, 3.0, 6.0, 4.0, 5.0, 2.0, 5.0,
  // 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 9.0, 7.0, 3.0, 6.0, 4.0, 5.0, 2.0, 5.0,
  // 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 9.0, 7.0, 3.0, 6.0, 4.0, 5.0, 2.0, 5.0
  uint32_t in_arr[IN_LEN] = {
      0x3F800000, 0xBF800000, 0x40000000, 0x40400000,     
      0xC0800000, 0x40A00000, 0xBF800000, 0x40000000,     
      0x41100000, 0x40E00000, 0x40400000, 0x40C00000,     
      0x40800000, 0x40A00000, 0x40000000, 0x40A00000,    

      0x3F800000, 0xBF800000, 0x40000000, 0x40400000,     
      0xC0800000, 0x40A00000, 0xBF800000, 0x40000000,     
      0x41100000, 0x40E00000, 0x40400000, 0x40C00000,     
      0x40800000, 0x40A00000, 0x40000000, 0x40A00000, 

      0x3F800000, 0xBF800000, 0x40000000, 0x40400000,     
      0xC0800000, 0x40A00000, 0xBF800000, 0x40000000,     
      0x41100000, 0x40E00000, 0x40400000, 0x40C00000,     
      0x40800000, 0x40A00000, 0x40000000, 0x40A00000 
    };
                  
  // Kernel Vector (FP32): 1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0
  uint32_t in_kernel[KERNEL_LEN] = {
    0x3F800000, 0x40400000, 0x40400000, 0x3F800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000  
  };
   
  // Create array for output data
  uint32_t output_len = IN_LEN+KERNEL_LEN+-1;
  uint32_t test_out[output_len];
  
  size_t n = sizeof(test_out);  // Size of test output in bytes

  printf("Starting Convolution\n");

  uint64_t start_cycle = read_cycles();
  
  uint8_t status = perform_convolution_1D_FAIL(in_arr, (uint32_t) IN_LEN, in_kernel, (uint8_t) KERNEL_LEN, test_out, (uint16_t) IN_DILATION);

  uint64_t end_cycle = read_cycles();

  if (status != 0) {
    // get_register_status_human_readable() does not work
    // printf("Convolution Failed: %s (%d)\n", get_register_status_human_readable(), status);

    printf("Convolution Failed: %d\n", status);
    switch (status) {
      case -1:   printf("Invalid Kernel Length. Must be 8 or 16"); break;
      case 0x01: printf("BUSY"); break;
      case 0x02: printf("COMPL"); break;
      case 0x04: printf("ERROR"); break;
      case 0x08: printf("INVALID"); break;
      case 0x10: printf("INFINITE"); break;
      case 0x20: printf("OVERFLOW"); break;
      case 0x40: printf("UNDERFLOW"); break;
      case 0x80: printf("INEXACT"); break;
      default: printf("UNKNOWN STATUS"); break;
    }

    return status;
  }

  printf("Convolution completed!\n");
  printf("Convolution Call took Cycles: %" PRIu64 "\n", end_cycle - start_cycle);

  // Print the input
  printf("\nInput (FP32):\n");
  for (int i = 0; i < IN_LEN; i++) {
    printf("0x%08X ", in_arr[i]);
  }

  // Print the kernel
  printf("\nKernel (FP32):\n");
  for (int i = 0; i < KERNEL_LEN; i++) {
    printf("0x%08X ", in_kernel[i]);
  }

  // Print the test results
  printf("\nTest Output (FP32):\n");
  for (int i = 0; i < output_len; i++) {
    printf("0x%08X ", test_out[i]);
  }

  // Calculate the correct convolution (Golden Model)
  float ref_out[output_len];
  perform_naive_convolution_1D(in_arr, IN_LEN, in_kernel, KERNEL_LEN, IN_DILATION, ref_out);
  printf("\nReference Output (FP32):\n");
  union Converter converter;
  for (int i = 0; i < output_len; i++) {
      converter.f = ref_out[i];
      printf("0x%08X ", converter.u);
  }

  printf("\nComparing test to reference.\n");

  // Check if the test passed
  if (memcmp(test_out, ref_out, n) == 0) {
      printf("[TEST PASSED]: Test Output matches Reference Output.");
  } else {
      printf("[TEST FAILED]: Test Output does not match Reference Output.");
  }
  printf("\n\n");
}

/* USER CODE END PUC */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  app_main();

  // printf('\n');  // NOTE: If you uncomment this line your program will hang because
  printf("\n\nFirst convolution:\n\n");
  first_convolution();

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