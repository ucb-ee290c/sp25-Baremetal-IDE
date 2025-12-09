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


// --- 1D Convolution Driver Functions ---

void start_conv_manual() {
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 1);
}

uint8_t get_register_status_manual() {
  return reg_read8((uintptr_t)(MMIO_BASE + CONV_STATUS_ADDR));
}

uint32_t get_register_out_count_manual() {
  return reg_read32((uintptr_t)(MMIO_BASE + CONV_COUNT_ADDR));
}

uint8_t get_register_read_check_manual() {
  return reg_read8((uintptr_t)(MMIO_BASE + READ_CHECK_ADDR));
}

void conv_init_manual() {
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  
  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 1);   
  reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 1);
}

// This function now ONLY writes parameters (Length, Dilation, Kernel)
int conv_set_params_kernel_only_manual(uint32_t input_length, uint16_t dilation, uint32_t* kernel, uint8_t kernel_length) {
  // Clear MMIO Reset, Clear Datapath, Stop
  reg_write8((uintptr_t)(MMIO_BASE + CONV_MMIO_RESET), 0);
  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 0); 
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);  

  // Write parameters
  reg_write32((uintptr_t)(MMIO_BASE + CONV_LENGTH_ADDR), input_length);
  reg_write16((uintptr_t)(MMIO_BASE + CONV_DILATION_ADDR), dilation);

  // Write kernel data and kernel length encoding (One-time setup)
  if (kernel_length == 8) {
    for (int i = 0; i < 8; i += 2) {
      reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
    }
    reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 0);
  } else if (kernel_length == 16) {
    for (int i = 0; i < 16; i += 2) {
      reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR), *((uint64_t*) (kernel + i)));
    }
    reg_write8((uintptr_t)(MMIO_BASE + CONV_KERNEL_LEN_ADDR), 1);  
  } else {
    return -1;  
  }

  return 0;
}

// Separate function to stream input data (used repeatedly in the main loop)
void conv_stream_input_batch_manual(uint32_t *input, size_t start_element, size_t num_packets) {
    for (size_t i = 0; i < num_packets; ++i) {
        size_t element_idx = start_element + (i * FP32_PER_PACKET);
        reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR), *((uint64_t*) (input + element_idx)));
    }
}

// Separate function to read output data (used repeatedly in the main loop)
void conv_read_output_batch_manual(uint32_t *output, size_t start_element, size_t num_packets) {
    for (size_t i = 0; i < num_packets; ++i) {
        size_t element_idx = start_element + (i * FP32_PER_PACKET);
        
        // Wait until at least one 64-bit word (2 FP32) is ready in the output FIFO
        while (get_register_out_count_manual() < 1) {
            // Spin-wait for output data
        }
        
        uint64_t current_out = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
        
        // Unpack the 64-bit word into two 32-bit FP32 elements
        uint32_t *unpacked = (uint32_t *) &current_out;
        output[element_idx]   = unpacked[0];
        output[element_idx + 1] = unpacked[1];
    }
}

// The main streaming driver function
uint8_t perform_convolution_1D_manual(
    uint32_t* input, 
    uint32_t input_length, 
    uint32_t* kernel, 
    uint8_t kernel_length, 
    uint32_t* output, 
    uint16_t dilation
) {
    // 1. Initial Setup
    conv_init_manual();
    conv_set_params_kernel_only_manual(input_length, dilation, kernel, kernel_length);

    // Calculate total packets
    size_t input_packets = input_length / FP32_PER_PACKET;
    // Assuming a valid convolution length, output_packets calculation is based on the DMA driver
    size_t kernel_packets = kernel_length / FP32_PER_PACKET; 
    size_t output_packets = input_packets + kernel_packets;

    size_t in_packet_idx = 0;
    size_t out_packet_idx = 0;

    // 2. Start the convolution engine
    start_conv_manual();

    // 3. Streaming in Batches
    while (out_packet_idx < output_packets) {
        // Calculate batch size, capped by the FIFO capacity (8 packets)
        size_t remaining_out_packets = output_packets - out_packet_idx;
        size_t current_batch_packets = remaining_out_packets < FIFO_CAPACITY_PACKETS
                                        ? remaining_out_packets
                                        : FIFO_CAPACITY_PACKETS;
        
        // The number of input packets to stream in this batch is limited by the remaining input
        size_t input_stream_packets = 0;
        if (in_packet_idx < input_packets) {
            size_t remaining_in_packets = input_packets - in_packet_idx;
            input_stream_packets = remaining_in_packets < current_batch_packets
                                    ? remaining_in_packets
                                    : current_batch_packets;
        }

        // A. Stream Input Batch (MMIO write)
        if (input_stream_packets > 0) {
            // Convert packet index to FP32 element index
            size_t start_element = in_packet_idx * FP32_PER_PACKET; 
            conv_stream_input_batch_manual(input, start_element, input_stream_packets);
            in_packet_idx += input_stream_packets;
        }

        // B. Read Output Batch (MMIO read)
        // Read the full batch, waiting for data if necessary (handled inside conv_read_output_batch)
        size_t start_element = out_packet_idx * FP32_PER_PACKET;
        conv_read_output_batch_manual(output, start_element, current_batch_packets);
        out_packet_idx += current_batch_packets;

        // Note: Unlike the DMA driver which waits for bus silence, 
        // the MMIO driver uses the `get_register_out_count()` check 
        // inside `conv_read_output_batch` for synchronization (spin-wait).
    }

    // 4. Final Status Read (Output fully drained)
    return get_register_status_manual();
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
  
  uint8_t status = perform_convolution_1D_manual(in_arr, (uint32_t) IN_LEN, in_kernel, (uint8_t) KERNEL_LEN, test_out, (uint16_t) IN_DILATION);

  uint64_t end_cycle = read_cycles();

  printf("Convolution took Cycles: %" PRIu64 "\n", end_cycle - start_cycle);





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
  printf("\n\nFirst convolution:\n");
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