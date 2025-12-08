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

void test_simple(){
  union Converter {
    float f;
    uint32_t u;
  };


  uint32_t in_arr[8] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000}; // {1, 2, 3, 4, 5, 6, 7, 8} in FP16
  uint32_t in_kernel[8] = {0x00000000, 0x3F800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x0000000}; // {0, 1, 0, 0, 0, 0, 0, 0} in FP16

  uint32_t in_len = 8;
  uint16_t in_dilation = 1;
  uint8_t kernel_len = 8;

  // TODO: make this variable a function based on the input parameters eventually
  uint32_t output_len = 15;

  
  printf("Setting values of MMIO registers\n");
  
  // Initialize the accelerator
  conv_init();
  
  // Set parameters for the convolution operation
  int result = conv_set_params(in_arr, in_len, in_dilation, in_kernel, kernel_len);
  if (result != 0) {
    printf("Error setting parameters\n");
    return;
  }
  
  puts("Starting Convolution");
  // Start the convolution operation
  start_conv();
  
  puts("Waiting for convolution to complete");
  
  printf("Input (FP32): ");
  for (int i = 0; i < in_len; i++) {
    printf("%#x ", in_arr[i]);
  }
  
  // Create array for output
  uint32_t test_out[output_len];
  int status = 0;
  
  printf("\nTest Output (FP32 binary): ");
  
  // Read the output using our function
  conv_read_output(test_out, output_len);
  
  // Print the results
  for (int i = 0; i < output_len; i++) {
    printf("0x%08x ", test_out[i]);
  }
  
  // Print final status
  printf("\nFinal status: ");
  switch (status) {
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
  printf("\nOutput count: %d\n", get_register_out_count());

  float ref_out[output_len];
  convolution_1D(in_arr, in_len, in_kernel, kernel_len, in_dilation, ref_out);
  printf("\nReference Output (FP32 binary): ");
  union Converter converter;
  for (int i = 0; i < output_len; i++) {
      converter.f = ref_out[i];
      printf("0x%08X ", converter.u);
  }
  printf("\n");
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


