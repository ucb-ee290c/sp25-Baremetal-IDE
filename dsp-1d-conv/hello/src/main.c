// /* USER CODE BEGIN Header */
// /**
//   ******************************************************************************
//   * @file           : main.c
//   * @brief          : Main program body
//   ******************************************************************************
//   * @attention
//   *
//   * This software is licensed under terms that can be found in the LICENSE file
//   * in the root directory of this software component.
//   * If no LICENSE file comes with this software, it is provided AS-IS.
//   *
//   ******************************************************************************
//   */
// /* USER CODE END Header */
// /* Includes ------------------------------------------------------------------*/
// #include "main.h"

// /* Private includes ----------------------------------------------------------*/
// /* USER CODE BEGIN Includes */

// /* USER CODE END Includes */

// /* Private typedef -----------------------------------------------------------*/
// /* USER CODE BEGIN PTD */

// /* USER CODE END PTD */

// /* Private define ------------------------------------------------------------*/
// /* USER CODE BEGIN PD */

// /* USER CODE END PD */

// /* Private macro -------------------------------------------------------------*/
// /* USER CODE BEGIN PM */

// /* USER CODE END PM */

// /* Private variables ---------------------------------------------------------*/
// /* USER CODE BEGIN PV */

// uint8_t counter = 0;

// /* USER CODE END PV */

// /* Private function prototypes -----------------------------------------------*/
// /* USER CODE BEGIN PFP */


// /* USER CODE END PFP */

// /* Private user code ---------------------------------------------------------*/
// /* USER CODE BEGIN PUC */

// #define BASE_ADDR 0x08800000

// #define INPUT_ADDR      0x08800000
// #define OUTPUT_ADDR     0x08800020
// #define KERNEL_ADDR     0x08800040
// #define START_ADDR      0x0880006C
// #define LENGTH_ADDR     0x08800078
// #define DILATION_ADDR   0x0880007C
// #define READ_CHECK_ADDR   0x0880008D

// void app_init() {
//   // torch::executor::runtime_init();
// }

// void test_simple(){
//   union Converter {
//     float f;
//     uint32_t u;
//   };

//   // uint32_t in_arr[16] = {
//   //   0x3F99999A, 0x40266666, 0xC0580000, 0x3F33CCCC, 
//   //   0x40A00000, 0xC01D3333, 0x40B33333, 0xBFC66666, 
//   //   0x40A19999, 0x40D99999, 0xC0A00000, 0x3F8CCCCD, 
//   //   0x40C00000, 0xBFC00000, 0x40019999, 0x40466666
//   // };
//   // uint32_t in_len = 16;
//   // uint16_t in_dilation = 1;
//   // uint32_t in_kernel[8] = {
//   //   0xC0800000, 0x40800000, 0x3F800000, 0xC0800000, 
//   //   0x40800000, 0xC0800000, 0x40800000, 0x40000000
//   // };
//   // uint8_t kernel_len = 8;
  
//   uint32_t in_arr[8] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000}; // {1, 2, 3, 4, 5, 6, 7, 8} in FP16
//   uint32_t in_len[1] = {8};
//   uint16_t in_dilation[1] = {1};
//   uint32_t in_kernel[8] = {0x00000000, 0x3F800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x0000000}; // {0, 1, 0, 0, 0, 0, 0, 0} in FP16
//   uint8_t kernel_len = 8;

//   // #############################################################  
//   // #############################################################  


//   // // Define the convolution accelerator structure
//   // printf("Setting values of MMIO registers\n");
  
//   // // Initialize the accelerator
//   // conv_init(CONV1D);


  
//   // // Set parameters for the convolution operation
//   // int result = conv_set_params(CONV1D, in_arr, in_len, in_dilation, in_kernel, kernel_len);
//   // if (result != 0) {
//   //   printf("Error setting parameters\n");
//   //   return;
//   // }
  
//   // #############################################################  
    
//   puts("Setting values of MMIO registers");
//   reg_write64(INPUT_ADDR, *((uint64_t*) in_arr));             // 64 bits: 2 FP32s
//   reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr + 2)));       // 64 bits: 2 FP32s 
//   reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr + 4)));       // 64 bits: 2 FP32s 
//   reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr + 6)));      // 64 bits: 2 FP32s (Total 8)

//   reg_write32(LENGTH_ADDR, in_len[0]);
//   reg_write16(DILATION_ADDR, in_dilation[0]);
//   reg_write64(KERNEL_ADDR, *((uint64_t*) in_kernel));         // 64 bits: 2 FP32s
//   reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel + 2)));   // 64 bits: 2 FP32s 
//   reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel + 4)));   // 64 bits: 2 FP32s 
//   reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel + 6)));   // 64 bits: 2 FP32s (Total 8)

//   // #############################################################  
//   // #############################################################  

  
//   puts("Starting Convolution");
//   // Start the convolution operation
//   start_conv(CONV1D);
//   // reg_write8(START_ADDR, 1);


//   puts("Waiting for convolution to complete");
//   // status = get_register_status(CONV1D);
//   // printf(status);
//   printf("Input (FP32): ");
//   for (int i = 0; i < 8; i++) {
//     printf("%#x ", in_arr[i]);
//   }
  
//   // Create array for output
//   // uint32_t *test_out[23];
//     uint32_t test_out[15];
  
//   // printf("\nTest Output (FP32 binary): ");

//   // #############################################################  
//   // #############################################################  
//   // printf("\nReading Convolution Output (FP32 binary):");
  
//   // uint8_t status = 0; 
//   // // Read the output using our function
//   // conv_read_output(CONV1D, test_out, 15, status, &in_arr);
//   // // Print the results
//   // printf("\nOutput: ");
//   // for (int i = 0; i < 15; i++) {
//   //   printf("0x%08x ", test_out[i]);
//   // }

//   // #############################################################  

  
//     printf("\nTest Output (FP32 binary): ");
//     for (int i = 0; i < 7; i++) {
//       uintptr_t reg_read = reg_read8(READ_CHECK_ADDR);
//       if (reg_read) {
//         uint64_t current_out = reg_read64(OUTPUT_ADDR);
//         uint32_t* unpacked_out = (uint32_t*) &current_out;
//         for (int j = 0; j < 2; j++) {
//             test_out[i*2 + j] = unpacked_out[j];
//             printf("0x%08x ", test_out[i*2 + j]);
//         }
//       } else {
//         printf("Waiting for 1d conv to be ready");
//       }
//     }

//     // Final 1 read: 1 output
//     uint64_t last_out = reg_read64(OUTPUT_ADDR);
//     uint32_t* unpacked_out = (uint32_t*) &last_out;
//     test_out[14] = unpacked_out[0];
//     printf("0x%08x ", unpacked_out[0]);
    
//   // #############################################################  
//   // #############################################################  

  
//   // Print final status
//   // printf("\nFinal status: %d\n", &status);
//   printf("Output count: %d\n", get_register_out_count(CONV1D));
// }

// void app_main() {
//   uint64_t mhartid = READ_CSR("mhartid");

//   printf("Hello world from hart %d: %d\n", mhartid, counter);
// }
// /* USER CODE END PUC */

// /**
//   * @brief  The application entry point.
//   * @retval int
//   */
// int main(int argc, char **argv) {
//   test_simple();
//   return 0;
// }

// /*
//  * Main function for secondary harts
//  * 
//  * Multi-threaded programs should provide their own implementation.
//  */
// void __attribute__((weak, noreturn)) __main(void) {
//   while (1) {
//    asm volatile ("wfi");
//   }
// }



/* 
################################################################
################################################################
################################################################
*/

/* 
    MUST BE RUN WITH logMaxKernelSize = 3 OR IT WILL NOT WORK.
    
    The expectation for software is to zero-extend the kernel array to fit the maximum kernel size.
    In this case, where logMaxKernelSize = 3, the maximum kernel size is 8, so the kernel array must 
    be zero extended by 0 to be of size 8.
*/

// #include "../../../tests/mmio.h"
// #include <stdio.h>
// #include <string.h>
// #include <stdint.h>
// #include <inttypes.h>

// # include "main.h"

// #define BASE_ADDR 0x08800000

// #define INPUT_ADDR      0x08800000
// #define OUTPUT_ADDR     0x08800020
// #define KERNEL_ADDR     0x08800040
// #define START_ADDR      0x0880006C
// #define LENGTH_ADDR     0x08800078
// #define DILATION_ADDR   0x0880007C

// union Converter {
//     float f;
//     uint32_t u;
// };

// int main(void) {
//     puts("Starting test");
//     uint32_t in_arr[8] = {0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000}; // {1, 2, 3, 4, 5, 6, 7, 8} in FP16
//     uint32_t in_len[1] = {8};
//     uint16_t in_dilation[1] = {1};
//     uint32_t in_kernel[8] = {0x00000000, 0x3F800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x0000000}; // {0, 1, 0, 0, 0, 0, 0, 0} in FP16
                                                                
//     puts("Setting values of MMIO registers");
    
//     reg_write64(INPUT_ADDR, *((uint64_t*) in_arr));             // 64 bits: 2 FP32s
//     reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr + 2)));       // 64 bits: 2 FP32s 
//     reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr + 4)));       // 64 bits: 2 FP32s 
//     reg_write64(INPUT_ADDR, *((uint64_t*) (in_arr + 6)));      // 64 bits: 2 FP32s (Total 8)

//     reg_write32(LENGTH_ADDR, in_len[0]);
//     reg_write16(DILATION_ADDR, in_dilation[0]);
//     reg_write64(KERNEL_ADDR, *((uint64_t*) in_kernel));         // 64 bits: 2 FP32s
//     reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel + 2)));   // 64 bits: 2 FP32s 
//     reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel + 4)));   // 64 bits: 2 FP32s 
//     reg_write64(KERNEL_ADDR, *((uint64_t*) (in_kernel + 6)));   // 64 bits: 2 FP32s (Total 8)


//     puts("Starting Convolution");
//     reg_write8(START_ADDR, 1);

//     puts("Waiting for convolution to complete");
    
//     printf("Input (FP32): ");
//     for (int i = 0; i < 8; i++) {
//         printf("%#x ", in_arr[i]);
//     }

//     uint32_t test_out[15];
//     printf("\nTest Output (FP32 binary): ");
//     for (int i = 0; i < 7; i++) {
//         uint64_t current_out = reg_read64(OUTPUT_ADDR);
//         uint32_t* unpacked_out = (uint32_t*) &current_out;
//         for (int j = 0; j < 2; j++) {
//             test_out[i*2 + j] = unpacked_out[j];
//             printf("0x%08x ", test_out[i*2 + j]);
//         }
//     }

//     // Final 1 read: 1 output
//     uint64_t last_out = reg_read64(OUTPUT_ADDR);
//     uint32_t* unpacked_out = (uint32_t*) &last_out;
//     test_out[14] = unpacked_out[0];
//     printf("0x%08x ", unpacked_out[0]);
    
//     // uint32_t ref_out[8] = {0x40000000, 0x40400000, 0x40800000, 0x40A00000, 0x40C00000, 0x40E00000, 0x41000000, 0x00000000}; // {2, 3, 4, 5, 6, 7, 8, 0} in FP16
//     float ref_out[15] = {0, 0, 0 ,0 , 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0};
//     union Converter converter;
//     printf("\nReference Output (FP16 binary): ");
//     for (int i = 0; i < 15; i++) {
//         converter.f = ref_out[i];
//         // printf("%f", (ref_out[i]));
//         printf("0x%08X ", converter.u);
//         // printf("%#x ", (ref_out[i]));
//     }
//     printf("\n");

//     if (memcmp(test_out, ref_out, 60) == 0) {
//         printf("[TEST PASSED]: Test Output matches Reference Output.");
//     } else {
//         printf("[TEST FAILED]: Test Output does not match Reference Output.");
//     }
//     printf("\n\n");


//     return 0;
// }

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
  // ConvAccel_Type conv;
  
  printf("Setting values of MMIO registers\n");
  
  // Initialize the accelerator
  printf("Writing addr: 0x%08x\n", (uintptr_t)&CONV1D->INPUT);
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
  // conv_read_output(CONV1D, test_out, 23, status, in_arr);
  conv_read_output_brokenout(CONV1D, test_out, 23, status, in_arr);
  // Print the results
  for (int i = 0; i < 23; i++) {
    printf("0x%08x ", test_out[i]);
  }
  
  // Print final status
  printf("\nFinal status: %d\n", status);
  printf("Output count: %d\n", get_register_out_count(CONV1D));
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
  // printf("INPUT:       0x%02lx\n", offsetof(ConvAccel_Type, INPUT));
  //   printf("OUTPUT:      0x%02lx\n", offsetof(ConvAccel_Type, OUTPUT));
  //   printf("KERNEL:      0x%02lx\n", offsetof(ConvAccel_Type, KERNEL));
  //   printf("STATUS:      0x%02lx\n", offsetof(ConvAccel_Type, STATUS));
  //   printf("START:       0x%02lx\n", offsetof(ConvAccel_Type, START));
  //   printf("CLEAR:       0x%02lx\n", offsetof(ConvAccel_Type, CLEAR));
  //   printf("OUT_COUNT:   0x%02lx\n", offsetof(ConvAccel_Type, OUT_COUNT));
  //   printf("LENGTH:      0x%02lx\n", offsetof(ConvAccel_Type, LENGTH));
  //   printf("DILATION:    0x%02lx\n", offsetof(ConvAccel_Type, DILATION));
  //   printf("REQ_ENQUE:   0x%02lx\n", offsetof(ConvAccel_Type, REQ_ENQ));
  //   printf("READ_CHECK:  0x%02lx\n", offsetof(ConvAccel_Type, READ_CHECK));
  //   printf("KERNEL_LEN:  0x%02lx\n", offsetof(ConvAccel_Type, KERNEL_LEN));
  //   printf("MMIO_RESET:  0x%02lx\n", offsetof(ConvAccel_Type, MMIO_RESET));

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

void conv_read_output_brokenout(ConvAccel_Type *conv, uint32_t *output, int output_len, int status, uint32_t* input) {
// void conv_read_output(ConvAccel_Type *conv, uint32_t *output, int output_len, uint8_t* status, uint32_t* input) {
    int i = 0;
    // puts("Here");
    // Read pairs of FP32s (2 per 64-bit read)

    printf("Read check: %d\n", get_register_read_check(CONV1D));
    printf("Register Status: %d\n", get_register_status(CONV1D));
    printf("Output count: %d\n", get_register_out_count(CONV1D));

    for (; i < (output_len-1)/2 ; i += 1) {
        // printf("Output count: %d\n", get_register_out_count(CONV1D));
        // printf("Register Status count: %d\n", get_register_status(CONV1D));
        // printf("Read check addr: 0x%08x\n", (uintptr_t)&CONV1D->READ_CHECK);

        // printf("Read check: %d\n", get_register_read_check(CONV1D));

        int timeout = 1000000;
        while (get_register_read_check(CONV1D) == 0 && timeout-- > 0) {
            printf("Read check: %d\n", get_register_read_check(CONV1D));
            printf("Register Status: %d\n", get_register_status(CONV1D));
            printf("Output count: %d\n", get_register_out_count(CONV1D));

            asm volatile("nop");  // prevent optimization
        }
        if (timeout <= 0) {
            printf("ERROR: Accelerator timed out!\n");
            
            return;
        }

        uint64_t current_out = reg_read64((uintptr_t)&conv->OUTPUT);
        // printf("Current Out: 0x%08x ", current_out);
        uint32_t *unpacked = (uint32_t *) &current_out;

        // if (i < 4) {
        //     reg_write64((uintptr_t)&conv->INPUT, *((uint64_t*) (input + (6 + 2*(i+1)))));
        // }
        output[i]     = unpacked[0];
        output[i + 1] = unpacked[1];
    }

    // Final 1 read: 1 output
    uint64_t last_out = reg_read64((uintptr_t)&conv->OUTPUT);
    uint32_t* unpacked_out = (uint32_t*) &last_out;
    output[output_len - 1] = unpacked_out[0];
    
    // *status = reg_read8((uintptr_t)&conv->STATUS);
    status = reg_read8((uintptr_t)&conv->STATUS);
}