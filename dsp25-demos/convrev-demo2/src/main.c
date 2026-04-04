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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "hal_i2s.h"
#include "hal_conv.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */


/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// --- AUDIO Configuration ---
// Set this to 1 to use "ir_data.h". Set to 0 to generate synthetic reverb.
#define USE_IMPORTED_IR   0 

#if USE_IMPORTED_IR
#include "ir_data.h" // Must contain: const float IR_DATA[]; and int IR_DATA_LEN;
#endif

// --- Hardware & Buffer Config ---
#define SAMPLE_RATE       44100
#define I2S_CHANNEL_NUM   0
#define CHUNK_SIZE        128  // Frames processed per loop
#define KERNEL_LEN        16   // Hardware Limit
#define IN_DILATION       1

// --- Reverb Settings ---
// How many 16-sample blocks to use. 
// 64 blocks * 16 samples = 1024 samples total tail.
#define IR_PARTITIONS     64 

// Accumulation Buffer Size
#define ACCUM_BUFFER_LEN  ((IR_PARTITIONS * KERNEL_LEN) + CHUNK_SIZE + 64)
/* USER CODE END PD */


/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
// --- Buffers ---
// The Impulse Response sliced into small blocks for the hardware
uint32_t ir_partitions[IR_PARTITIONS][KERNEL_LEN];

// Ring-like buffer for summing up future echoes
float    accum_buffer[ACCUM_BUFFER_LEN];

// Audio I/O Buffers
uint64_t i2s_rx_buffer[CHUNK_SIZE];
float    input_float[CHUNK_SIZE];
uint32_t hw_result_buffer[CHUNK_SIZE + KERNEL_LEN]; 
uint64_t i2s_tx_buffer[CHUNK_SIZE];
/* USER CODE END PV */


/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
void init_ir_partitions(void);
void process_audio_chunk(void);
void float_to_i2s(float* in_data, uint64_t* out_data, int length);
void i2s_to_float(uint64_t* in_data, float* out_data, int length);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN PUC */
void convrev_init() {
    printf("Initializing Reverb System...\n");

    // 1. I2S Configuration
    i2s_params_t i2s_params;
    i2s_params.clkgen = 1;
    i2s_params.tx_en = 1;
    i2s_params.rx_en = 1;
    i2s_params.dacen = 1; 
    i2s_params.ws_len = 32; 
    i2s_params.bitdepth_tx = 32;
    i2s_params.bitdepth_rx = 32;
    config_I2S(I2S_CHANNEL_NUM, &i2s_params);

    // 2. Clear Accumulation Buffer
    memset(accum_buffer, 0, ACCUM_BUFFER_LEN * sizeof(float));

    // 3. Prepare the Impulse Response (Partitioning)
    init_ir_partitions();

    printf("System Ready. Mode: %s\n", USE_IMPORTED_IR ? "Imported IR" : "Synthetic IR");
}

void convrev_main() {
    process_audio_chunk();
}

/**
 * @brief Logic to load OR generate IR and slice it into 16-sample partitions.
 */
void init_ir_partitions(void) {
    printf("Partitioning Impulse Response...\n");
    
    unsigned long seed = 12345;
    int total_ir_len = IR_PARTITIONS * KERNEL_LEN;

    for (int p = 0; p < IR_PARTITIONS; p++) {
        for (int k = 0; k < KERNEL_LEN; k++) {
            
            // Current global index in the imaginary long IR array
            int global_idx = (p * KERNEL_LEN) + k;
            float sample_val = 0.0f;

            // --- OPTION A: Load from Header File ---
            #if USE_IMPORTED_IR
                if (global_idx < IR_DATA_LEN) {
                    sample_val = IR_DATA[global_idx];
                } else {
                    sample_val = 0.0f; // Zero-pad if IR_DATA is shorter than partitions
                }
            
            // --- OPTION B: Generate Synthetic IR ---
            #else
                // White noise generation
                seed = (1103515245 * seed + 12345) % 2147483648;
                float noise = ((float)seed / 1073741824.0f) - 1.0f;

                // Exponential decay
                float decay = expf( -4.0f * (float)global_idx / (float)total_ir_len );
                sample_val = noise * decay * 0.2f; 
            #endif

            // Store as uint32 representation for the hardware driver
            ir_partitions[p][k] = *((uint32_t*)&sample_val);
        }
    }
}

/**
 * @brief Main Audio Processing Loop (Partitioned Convolution)
 */
void process_audio_chunk() {
    // 1. Read Audio
    for (int i = 0; i < CHUNK_SIZE; i++) {
        i2s_rx_buffer[i] = read_I2S_rx(I2S_CHANNEL_NUM, I2S_LEFT);
    }
    i2s_to_float(i2s_rx_buffer, input_float, CHUNK_SIZE);

    // 2. Perform Partitioned Convolution
    for (int p = 0; p < IR_PARTITIONS; p++) {
        int time_offset = p * KERNEL_LEN;

        // Call Hardware Driver
        perform_convolution_1D(
            (uint32_t*)input_float, 
            CHUNK_SIZE, 
            ir_partitions[p], 
            KERNEL_LEN, 
            hw_result_buffer, 
            IN_DILATION
        );

        // Accumulate (Overlap-Add)
        int result_len = CHUNK_SIZE + KERNEL_LEN - 1;
        for (int i = 0; i < result_len; i++) {
            if ((time_offset + i) < ACCUM_BUFFER_LEN) {
                float val = *((float*)&hw_result_buffer[i]);
                accum_buffer[time_offset + i] += val;
            }
        }
    }

    // 3. Output Audio (Input + Reverb Tail)
    float_to_i2s(accum_buffer, i2s_tx_buffer, CHUNK_SIZE);

    for (int i = 0; i < CHUNK_SIZE; i++) {
        write_I2S_tx(I2S_CHANNEL_NUM, I2S_LEFT, i2s_tx_buffer[i]);
        write_I2S_tx(I2S_CHANNEL_NUM, I2S_RIGHT, i2s_tx_buffer[i]);
    }

    // 4. Shift Accumulator for next frame
    int remaining = ACCUM_BUFFER_LEN - CHUNK_SIZE;
    memmove(&accum_buffer[0], &accum_buffer[CHUNK_SIZE], remaining * sizeof(float));
    memset(&accum_buffer[remaining], 0, CHUNK_SIZE * sizeof(float));
}

// --- Helpers ---

void i2s_to_float(uint64_t* in_data, float* out_data, int length) {
    const float scaling = 1.0f / 2147483648.0f; 
    for (int i = 0; i < length; i++) {
        // Cast to signed 32-bit before float conversion to handle negative numbers
        out_data[i] = (float)((int32_t)in_data[i]) * scaling;
    }
}

void float_to_i2s(float* in_data, uint64_t* out_data, int length) {
    const float scaling = 2147483647.0f;
    for (int i = 0; i < length; i++) {
        float s = in_data[i];
        if (s > 0.99f) s = 0.99f;
        if (s < -0.99f) s = -0.99f;
        out_data[i] = (uint64_t)((int32_t)(s * scaling));
    }
}
/* USER CODE END PUC */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  /* MCU Configuration--------------------------------------------------------*/

  /* Configure the system clock */
  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */  
  /* USER CODE BEGIN Init */
  convrev_init();
  /* USER CODE END Init */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1) {
    convrev_main();
    return 0;
  }
  /* USER CODE END WHILE */
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