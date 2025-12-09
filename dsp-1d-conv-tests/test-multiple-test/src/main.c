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

void convolution_1D(uint32_t *arr, size_t arr_len, uint32_t *kernel, size_t kernel_len, size_t dilation, float *output) {
  
  /* 
  Computes the convolution of arr with the given kernel and dilation factor and stores the result in output, specifically 
  based on the implementation of the convolution block. The first value in the output array is computed with the kernel's 
  left element aligned with the array's left element.

  arr:        pointer to input array      FP16 array
  arr_len:    length of input array       
  kernel:     pointer to kernel array     FP16 array (represented as uint16_t)
  kernel_len: length of kernel array 
  dilation:   dilation factor
  output:     pointer to output array     FP16 array (represented as uint16_t) 

  Example input and output: 

  arr:        {1, 2, 3, 4}
  arr_len:    4
  kernel:     {-1, 1, -1}
  kernel_len: 3
  dilation:   1

  output: {-2, -3, -1, -4} ({-1*1 + 1*2 + -1*3, -1*2 + 1*3 + -1*4, -1*3 + 1*4 + -1*0, -1*4 + 1*0 + -1*0})

  For border values (at the end), we assume the array is zero-extended to fit the length of the kernel (including dilation).
  */

  size_t output_len = arr_len + (kernel_len - 1) * dilation;

    for (int i = 0; i < output_len; i++) {
        output[i] = 0.0f;

        for (int j = 0; j < kernel_len; j++) {
            int arr_index = i + j * dilation - (kernel_len - 1) * dilation;

            uint32_t item = 0;
            if (arr_index >= 0 && arr_index < arr_len) {
                item = arr[arr_index];
            }

            float float_input = *(float*)&item;
            float float_kernel = *(float*)&kernel[j]; 
            output[i] += float_input * float_kernel;
        }
    }
}

void app_main() {
  uint64_t mhartid = READ_CSR("mhartid");

  printf("Hello world from hart %d: %d\n", mhartid, counter);
}


void first_convolution() {

  // IO:
  uint32_t in_arr[16] = {0x3F800000,0xBF800000,0x40000000,0x40400000,0xC0800000,
                           0x40A00000,0xBF800000,0x40000000,0x41100000,0x40E00000,
                           0x40400000,0x40C00000,0x40800000,0x40A00000,0x40000000,0x40A00000};
  uint32_t in_kernel[8] = {0x3F800000,0x40400000,0x40400000,0x3F800000,0,0,0,0};
 
  uint32_t in_len = 16;
  uint8_t kernel_len = 8;

  // Config:
  uint16_t in_dilation = 1;
  
  // Create array for output data
  uint32_t output_len = in_len+kernel_len+-1;
  uint32_t test_out[output_len];
  
  size_t n = sizeof(test_out);  // Size of test output in bytes

  printf("Starting Convolution\n");
  
  uint8_t status = perform_convolution_1D(in_arr, in_len, in_kernel, kernel_len, test_out, in_dilation);

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

  // Print the input
  printf("Input (FP32):\n");
  for (int i = 0; i < in_len; i++) {
    printf("0x%08x ", in_arr[i]);
  }

  // Print the kernel
  printf("Kernel (FP32):\n");
  for (int i = 0; i < kernel_len; i++) {
    printf("0x%08x ", in_kernel[i]);
  }

  // Print the test results
  printf("\nTest Output (FP32):\n");
  for (int i = 0; i < output_len; i++) {
    printf("0x%08x ", test_out[i]);
  }

  // Calculate the correct convolution (Golden Model)
  float ref_out[output_len];
  perform_naive_convolution_1D(in_arr, in_len, in_kernel, kernel_len, in_dilation, ref_out);
  printf("\nReference Output (FP32):\n");
  union Converter converter;
  for (int i = 0; i < output_len; i++) {
      converter.f = ref_out[i];
      printf("0x%08x ", converter.u);
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


void second_convolution() {

  // IO:
  uint32_t in_arr[16] = {0x41200000,0x41300000,0x41400000,0x41500000,0x3F800000,0x40000000,
                           0x40400000,0x40800000,0x41000000,0x40E00000,0x40C00000,0x40A00000,
                           0x40800000,0xBF800000,0xC0000000,0xC0400000};
  uint32_t in_kernel[8] = {0x3F800000,0xBF800000,0x3F800000,0,0,0,0,0};
 
  uint32_t in_len = 16;
  uint8_t kernel_len = 8;
  
  // Config:
  uint16_t in_dilation = 1;
  
  // Create array for output data
  uint32_t output_len = in_len+kernel_len+-1;
  uint32_t test_out[output_len];
  
  size_t n = sizeof(test_out);  // Size of test output in bytes

  printf("Starting Convolution\n");
  
  uint8_t status = perform_convolution_1D(in_arr, in_len, in_kernel, kernel_len, test_out, in_dilation);

  if (status != 0) {
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

  // Print the input
  printf("Input (FP32):\n");
  for (int i = 0; i < in_len; i++) {
    printf("0x%08x ", in_arr[i]);
  }

  // Print the kernel
  printf("Kernel (FP32):\n");
  for (int i = 0; i < kernel_len; i++) {
    printf("0x%08x ", in_kernel[i]);
  }

  // Print the test results
  printf("\nTest Output (FP32):\n");
  for (int i = 0; i < output_len; i++) {
    printf("0x%08x ", test_out[i]);
  }

  // Calculate the correct convolution (Golden Model)
  float ref_out[output_len];
  perform_naive_convolution_1D(in_arr, in_len, in_kernel, kernel_len, in_dilation, ref_out);
  printf("\nReference Output (FP32):\n");
  union Converter converter;
  for (int i = 0; i < output_len; i++) {
      converter.f = ref_out[i];
      printf("0x%08x ", converter.u);
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
  printf("\n\n");
  printf("First convolution:\n");
  first_convolution();

  printf("Second convolution:\n");
  second_convolution();
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


