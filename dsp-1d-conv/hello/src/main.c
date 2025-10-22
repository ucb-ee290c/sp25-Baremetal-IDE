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
  
  printf("Setting values of MMIO registers\n");
  printf("CONV1D address: 0x%x\n", CONV1D);
  
  // Initialize the accelerator
  conv_init(CONV1D);
  
  // Set parameters for the convolution operation
  int result = conv_set_params(CONV1D, in_arr, in_len, in_dilation, in_kernel, kernel_len);
  if (result != 0) {
    printf("Error setting parameters\n");
    return;
  }
  
  puts("Starting Convolution");
  // Start the convolution operation
  start_conv(CONV1D);
  
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
  conv_read_output(CONV1D, test_out, 23, &status, in_arr);
  
  // Print the results
  for (int i = 0; i < 23; i++) {
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
  printf("\nOutput count: %d\n", get_register_out_count(CONV1D));

  float ref_out[32];
  convolution_1D(in_arr, in_len, in_kernel, kernel_len, in_dilation, ref_out);
  printf("\nReference Output (FP32 binary): ");
  union Converter converter;
  for (int i = 0; i < 23; i++) {
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