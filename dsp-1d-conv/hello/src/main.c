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

// Constants
#define CACHELINE        64


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
}


// This is our golden model that we will compare the test output to
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

// Perform 1D convolution using the wrapper function
void app_main() {
  printf("In app_main\n");

  union Converter {
    float f;
    uint32_t u;
  };
  
  // Inputs and outputs

  // __attribute__((aligned(CACHELINE))) uint32_t in_arr[8] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000}; // {1, 2, 3, 4, 5, 6, 7, 8} in FP16
  // __attribute__((aligned(CACHELINE))) uint32_t in_kernel[8] = {0x00000000, 0x3F800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x0000000}; // {0, 1, 0, 0, 0, 0, 0, 0} in FP16

  uint32_t in_arr[8] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000}; // {1, 2, 3, 4, 5, 6, 7, 8} in FP16
  uint32_t in_kernel[8] = {0x00000000, 0x3F800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x0000000}; // {0, 1, 0, 0, 0, 0, 0, 0} in FP16

  
  uint32_t in_len[1] = {8};
  uint16_t in_dilation[1] = {1};

  uint32_t test_out[23];

  // Start the convolution operation
  printf("Starting Convolution\n");

  uint8_t status = perform_convolution((uint64_t) &in_arr,     
                                       (uint64_t) &test_out, 
                                       (uint16_t) &in_kernel, 
                                       (uint32_t) &in_len, 
                                       (uint16_t) &in_dilation);
  
  if (status != 0x0) {
      printf("Convolution Failed\n");
      printf("Error Status: %p\n", status);
      switch (status) {
        case 0x01: printf("BUSY\n"); break;
        case 0x02: printf("COMPL\n"); break;
        case 0x04: printf("ERROR\n"); break;
        case 0x08: printf("INVALID\n"); break;
        case 0x10: printf("INFINITE\n"); break;
        case 0x20: printf("OVERFLOW\n"); break;
        case 0x40: printf("UNDERFLOW\n"); break;
        case 0x80: printf("INEXACT\n"); break;
        default: printf("UNKNOWN STATUS\n"); break;
      }
    }

  printf("Convolution Finished!\n");
  printf("Check status: %p\n", status);

  // Compare hardware output to software output
  
  printf("Input (FP32): \n");
  for (int i = 0; i < 16; i++) {
    printf("%#x ", in_arr[i]);
  }
  
  printf("\nTest Output (FP32 binary): \n");
  
  // Print the results
  for (int i = 0; i < 23; i++) {
    printf("0x%08x ", test_out[i]);
  }

  // // Print out the reference golden model
  
  // float ref_out[32];


  // convolution_1D(in_arr, in_len, in_kernel, kernel_len, in_dilation, ref_out);
  // printf("\nReference Output (FP32 binary): ");
  // union Converter converter;
  // for (int i = 0; i < 23; i++) {
  //     converter.f = ref_out[i];
  //     printf("0x%08X ", converter.u);
  // }
  // printf("\n");

    // Print out the correct reference output


  // uint32_t ref_out[8] = {0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000, 0x00000000}; // {2, 3, 4, 5, 6, 7, 8, 0} in FP16
  float ref_out[15] = {0, 0, 0 ,0 , 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0};
  union Converter converter;
  printf("\nReference Output (FP16 binary): \n");
  for (int i = 0; i < 15; i++) {
      converter.f = ref_out[i];
      // printf("%f", (ref_out[i]));
      printf("0x%08X ", converter.u);
      // printf("%#x ", (ref_out[i]));
  }
  printf("\n");


  
  if (memcmp(test_out, ref_out, 60) == 0) {
      printf("[TEST PASSED]: Test Output matches Reference Output.\n");
  } else {
      printf("[TEST FAILED]: Test Output does not match Reference Output.\n");
  }
  printf("\n\n");
}

// Simple main function that just runs once
int main() {
  // // Make sure we're on hart 0
  // uint64_t mhartid;
  // asm volatile ("csrr %0, mhartid" : "=r" (mhartid));
  // if (mhartid != 0) {
  //     // If not on hart 0, just return
  //     return 0;
  // }

  
  uint64_t mhartid = READ_CSR("mhartid");

  printf("Hello world from hart %d: %d\n", mhartid, counter);
  
  // app_init();

  while (1) {
    printf("Before app_main in while loop\n");

    app_main();
    return 0;
  }
  // app_main();
  
  // // Add a small delay to ensure output is printed
  // for (volatile int i = 0; i < 1000; i++);
  
  // return 0;
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