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

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
#define MAX_INPUT_ELEMENTS   2048
#define KERNEL_ELEMENTS      8
#define MAX_OUTPUT_ELEMENTS  (MAX_INPUT_ELEMENTS + KERNEL_ELEMENTS)

// Driver Prototype
void dma_1dConvDriver(
    uint32_t *input_buffer_ptr,
    uint32_t *output_buffer_ptr,
    uint32_t *kernel_buffer_ptr,
    size_t    total_elements,
    size_t    kernel_elements,
    uint16_t  dilation,
    uint32_t  base_id
);

static uint64_t read_cycles(void) {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

// -----------------------------------------------------------------------------
// Utility: Compute Golden Output (Aligned to Full Convolution)
// -----------------------------------------------------------------------------
static void compute_golden_full_conv(
    const uint32_t *in_u32, size_t input_len,
    const uint32_t *ker_u32, size_t kernel_len,
    uint16_t dilation, uint32_t *out_u32) 
{
    float in_f[MAX_INPUT_ELEMENTS];
    float ker_f[KERNEL_ELEMENTS];

    // Convert raw bits to float
    for (size_t i = 0; i < input_len; ++i) memcpy(&in_f[i], &in_u32[i], sizeof(float));
    for (size_t k = 0; k < kernel_len; ++k) memcpy(&ker_f[k], &ker_u32[k], sizeof(float));

    // Full Convolution Output Length
    size_t out_len = input_len + kernel_len - 1;

    // Offset: We start calculating when the LAST kernel element hits the FIRST input.
    // This effectively shifts the "time" index by (KernelLen - 1).
    int offset = (int)kernel_len - 1;

    for (size_t i = 0; i < out_len; ++i) {
        double acc = 0.0;
        for (size_t k = 0; k < kernel_len; ++k) {
            // Standard Conv Index: i - k
            // Adjusted for "Full" ramp-up: i - k - offset
            long idx = (long)i - (long)k * (long)dilation - (long)offset;
            
            // Boundary check (Zero padding for "sliding in")
            if (idx >= 0 && (size_t)idx < input_len) {
                acc += (double)in_f[idx] * (double)ker_f[k];
            }
        }
        float acc_f = (float)acc;
        memcpy(&out_u32[i], &acc_f, sizeof(float));
    }
}

// -----------------------------------------------------------------------------
// Test Runner
// -----------------------------------------------------------------------------
static void run_one_test(int test_id, size_t N, uint32_t *kernel_buf) {
    
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
    dma_1dConvDriver(
        input_buf, hw_out_buf, kernel_buf,
        N, KERNEL_ELEMENTS, 1, base_id
    );
    uint64_t cycles = read_cycles() - start;

    // (Using %u for N to avoid printf bug)
    printf("Test %d (N=%u): Cycles = %" PRIu64 "\n", test_id + 1, (unsigned)N, cycles);

    // 4. Run Software Golden
    compute_golden_full_conv(input_buf, N, kernel_buf, KERNEL_ELEMENTS, 1, golden_buf);

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

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
void app_main(void) {
    puts("Starting 1D Conv Test (Aligned Math)\n");

    // Kernel: 1.0, 0, 0...
    static uint32_t kernel1[KERNEL_ELEMENTS] = {
        0x3f800000, 0, 0, 0, 0, 0, 0, 0
    };

    // Test lengths
    const size_t test_lengths[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int num_tests = sizeof(test_lengths) / sizeof(test_lengths[0]);

    for (int t = 0; t < num_tests; ++t) {
        run_one_test(t, test_lengths[t], kernel1);
    }

    puts("\nAll tests finished.");
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