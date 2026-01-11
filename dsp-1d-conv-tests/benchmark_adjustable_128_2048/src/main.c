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

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
#define MAX_INPUT_ELEMENTS   2048
#define KERNEL_ELEMENTS      8
#define MAX_OUTPUT_ELEMENTS  (MAX_INPUT_ELEMENTS + KERNEL_ELEMENTS)


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

    puts("Starting 1D Conv Test (Aligned Math)\n");

    // Kernel: 1.0, 0, 0...
    static uint32_t kernel1[KERNEL_ELEMENTS] = {
        0x3f800000, 0, 0, 0, 0, 0, 0, 0
    };

    // Test lengths
    const size_t test_lengths[] = {2, 4, 8, 16, 32, 64, 256, 512, 1024, 2048, 4096 };
    const int num_tests = sizeof(test_lengths) / sizeof(test_lengths[0]);

    for (int t = 0; t < num_tests; ++t) {
        run_one_test(t, test_lengths[t], kernel1);
    }

    puts("\nAll tests finished.\n");
    return 0;
}


// -----------------------------------------------------------------------------
// Cycle counter
// -----------------------------------------------------------------------------
static uint64_t read_cycles(void) {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}


#define IN_LEN 256
#define KERNEL_LEN 8
#define IN_DILATION 1

#include <stdbool.h>

// -----------------------------------------------------------------------------
// Test Runner
// -----------------------------------------------------------------------------
void run_one_test(int test_id, size_t N, uint32_t *kernel_buf) {

  // Shared buffers
  static uint32_t input_buf[MAX_INPUT_ELEMENTS];
  static uint32_t hw_out_buf[MAX_OUTPUT_ELEMENTS];
  static uint32_t golden_buf[MAX_OUTPUT_ELEMENTS];

  // 1. Generate Input Data: 0.0, 1.0, 2.0...
  for (size_t i = 0; i < N; ++i) {
      float f_val = (float)(i % 256);
      memcpy(&input_buf[i], &f_val, sizeof(float));
  }

  // 2. Clear outputs
  size_t out_len = N + KERNEL_ELEMENTS - 1;
  memset(hw_out_buf, 0, out_len * sizeof(uint32_t));
  memset(golden_buf, 0, out_len * sizeof(uint32_t));
  // Asm vola
  asm volatile("fence");

  // 3. Run Hardware Driver
  uint32_t base_id = 100 * (test_id + 1);

  uint64_t start = read_cycles();
  uint8_t status = perform_convolution_1D(input_buf, (uint32_t) N, kernel_buf, (uint8_t) KERNEL_ELEMENTS, hw_out_buf, (uint16_t) 1);

  // dma_1dConvDriver(
  //     input_buf, hw_out_buf, kernel_buf,
  //     N, KERNEL_ELEMENTS, 1, base_id
  // );
  uint64_t cycles = read_cycles() - start;
  
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

  // (Using %u for N to avoid printf bug)
  printf("Test %d (N=%u): Cycles = %d\n", test_id + 1, (unsigned)N, cycles);
  uint32_t naive_cycles = read_cycles();
  // 4. Run Software Golden
  // compute_golden_full_conv(input_buf, N, kernel_buf, KERNEL_ELEMENTS, 1, golden_buf);
  
  perform_naive_convolution_1D(input_buf, N, kernel_buf, KERNEL_ELEMENTS, 1, golden_buf);
  uint32_t naive_end = read_cycles();
  printf("  Software Golden Cycles = %d\n", naive_end - naive_cycles);
  // 5. Compare
  // Now that SW is aligned to the "ramp up", we expect a perfect 1-to-1 match.
  int errors = 0;
  for (size_t i = 0; i < out_len; ++i) {
      if (hw_out_buf[i] != golden_buf[i]) {
          float hw_f, sw_f;
          memcpy(&hw_f, &hw_out_buf[i], sizeof(float));
          memcpy(&sw_f, &golden_buf[i], sizeof(float));
          asm volatile("fence");
          printf("  [FAIL] Mismatch at idx %u: HW=0x%08x (%.1f) != SW=0x%08x (%.1f)\n",
                 (unsigned)i, hw_out_buf[i], hw_f, golden_buf[i], sw_f);
          errors++;
          if (errors > 5) {
              printf("  ... (stopping after 5 errors)\n");
              break;
          }
      }
  }

  if (errors == 0) {
      printf("  [PASS] Hardware matches Software perfectly.\n");
  }
}



/* USER CODE END PUC */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  // Initialize UART0 for Serial Monitor
  UART_InitType UART0_init_config;
  UART0_init_config.baudrate = 115200;
  UART0_init_config.mode = UART_MODE_TX_RX;
  UART0_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART0_init_config);

  app_main();

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