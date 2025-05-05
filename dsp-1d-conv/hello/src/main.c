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

void test_simple(){
  union Converter {
    float f;
    uint32_t u;
  };

  uint32_t in_arr[16] = {
    0x3F99999A, 0x40266666, 0xC0580000, 0x3F33CCCC, 
    0x40A00000, 0xC01D3333, 0x40B33333, 0xBFC66666, 
    0x40A19999, 0x40D99999, 0xC0A00000, 0x3F8CCCCD, 
    0x40C00000, 0xBFC00000, 0x40019999, 0x40466666
  };
  uint32_t in_len = 16;
  uint16_t in_dilation = 1;
  uint32_t in_kernel[8] = {
    0xC0800000, 0x40800000, 0x3F800000, 0xC0800000, 
    0x40800000, 0xC0800000, 0x40800000, 0x40000000
  };
  uint8_t kernel_len = 8;
  
  // Define the convolution accelerator structure
  ConvAccel_Type conv;
  
  printf("Setting values of MMIO registers\n");
  
  // Initialize the accelerator
  conv_init(&conv);
  
  // Set parameters for the convolution operation
  int result = conv_set_params(&conv, in_arr, in_len, in_dilation, in_kernel, kernel_len);
  if (result != 0) {
    printf("Error setting parameters\n");
    return;
  }
  
  puts("Starting Convolution");
  // Start the convolution operation
  start_conv(&conv);
  
  puts("Waiting for convolution to complete");
  
  printf("Input (FP32): ");
  for (int i = 0; i < 16; i++) {
    printf("%#x ", in_arr[i]);
  }
  
  // Create array for output
  uint32_t test_out[23];
  int status = 0;
  
  printf("\nTest Output (FP32 binary): ");
  
  // Read the output using our function
  conv_read_output(&conv, test_out, 23, status, in_arr);
  
  // Print the results
  for (int i = 0; i < 23; i++) {
    printf("0x%08x ", test_out[i]);
  }
  
  // Print final status
  printf("\nFinal status: %d\n", status);
  printf("Output count: %d\n", get_register_out_count(&conv));
}

void app_main() {
  uint64_t mhartid = READ_CSR("mhartid");

  printf("Hello world from hart %d: %d\n", mhartid, counter);
}
/* USER CODE END PUC */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  test_simple();
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