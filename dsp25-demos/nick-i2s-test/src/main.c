// cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=$(TYPE) -D CMAKE_C_FLAGS="-mcmodel=medany" -D CMAKE_ASM_FLAGS="-mcmodel=medany" -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -DCHIP=$(CHIP)
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
#include "chip_config.h"
#include "rocketcore.h"
#include <inttypes.h>
#include <stdbool.h>
#include <meep.h>
#include <roll.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// I2S at 44kHz -> ~440hz square wave
#define PULSE_PERIOD_SAMPLES (100) // 44kHz -> 100 samples per 440Hz period
#define PULSE_WIDTH_SAMPLES (50) // 50% duty cycle
#define AMPLITUDE_32 (0xEEEEEEEE) // Sample amplitude for 32 bit depth
#define AMPLITUDE_16 (0xEEEE) // Sample amplitude for 16 bit depth


#define SPEAKER_CHANNEL 1
#define MIC_CHANNEL 0

#define BLOCK_SIZE      32      // Number of samples to process per batch
#define KERNEL_SIZE     8      // Hardware limit: 8 or 16
#define DILATION        1       // Standard convolution
#define CONV_OUT_SIZE   (BLOCK_SIZE + KERNEL_SIZE) 
#define NUM_SEGMENTS    4
#define KERNEL_CHUNK    16

// Helper for float conversion
#define S32_TO_FLOAT(x) ((float)(x) / 2147483648.0f)
#define FLOAT_TO_S32(x) ((int32_t)((x) * 2147483647.0f))
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

// TODO: Verify clocks and clockdiv. Stolen from dsp-24 bmarks
// https://github.com/ucb-bar/sp24-Baremetal-IDE/blob/dsp24-bmarks/i2s-test/src/main.c
i2s_params_t i2s_params_mic = {
    .tx_en       = 1,
    .rx_en       = 1,
    .bitdepth_tx = I2S_BITDEPTH_32,
    .bitdepth_rx = I2S_BITDEPTH_32,
    .clkgen      = 1,
    .dacen       = 0,
    .ws_len      = 3,
    .clkdiv      = 8,
    .tx_fp       = 0,
    .rx_fp       = 0,
    .tx_force_left = 0,
    .rx_force_left = 0
};

i2s_params_t i2s_params_speaker = {
  .tx_en       = 1,
  .rx_en       = 1,
  .bitdepth_tx = I2S_BITDEPTH_32,
  .bitdepth_rx = I2S_BITDEPTH_32,
  .clkgen      = 1,
  .dacen       = 1,
  .ws_len      = 3,
  .clkdiv      = 8,
  // .clkdiv      = 4, # terrible greasy nasty sound
  .tx_fp       = 0,
  .rx_fp       = 0,
  .tx_force_left = 0,
  .rx_force_left = 0
};

// PLL target frequency = 500 MHz; different than system clock of 50 MHz
uint64_t target_frequency = 500000000;


void app_init() {
  UART_InitType UART_init_config;
  UART_init_config.baudrate = 115200;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART_init_config);

  printf("Configuring PLL\r\n");

  //set_all_clocks(RCC_CLOCK_SELECTOR, 0);
  //configure_pll(PLL, target_frequency/SYS_CLK_FREQ, 0);
  //set_all_clocks(RCC_CLOCK_SELECTOR, 1);

  //UART0->DIV = (target_frequency / 115200) - 1;

  printf("I2S params initializing\r\n");

  config_I2S(SPEAKER_CHANNEL, &i2s_params_speaker);
  config_I2S(MIC_CHANNEL, &i2s_params_mic);

  printf("Init done\r\n");
}

// TLDR: Plays a constant tone -- doesn't use mic
void i2s_square_wave_test(void) {

  uint32_t counter = 0;
  uint32_t sample[2] = {AMPLITUDE_32, AMPLITUDE_32};
  while (1) {
    // Divide by 2 because 2 samples fit in one 64 bit transaction
    uint64_t data;
    if (counter < PULSE_WIDTH_SAMPLES / 4) {
      data = ((uint64_t)sample[0] << 32) | (uint64_t)sample[1];
    } else {
      data = 0;
    }
    write_I2S_tx(SPEAKER_CHANNEL, true, data);
    write_I2S_tx(SPEAKER_CHANNEL, false, data);

    counter = (counter + 1) % (PULSE_PERIOD_SAMPLES / 4);
	}

}


// TLDR: Records live audio (certain length) and plays it back with a time delay
void i2s_hollaback_test(void) {
  printf("Entered i2s_mic_test\r\n");
  uint64_t counter = 0;
  uint64_t playback = 0;
  // const static int playback_seconds = 5;
  // const static int num_samples_per_i2s_frame = 2;
  
  // const uint64_t recording_cycle_length = 5 * 44100 / 4;
  // static uint64_t recorded_audio[5 * 44100 / 4];

  // const uint64_t recording_cycle_length = (uint64_t) 0.1 * 44100 / 2;
  // static uint64_t recorded_audio[(uint64_t) (0.1 * 44100 / 2)];

  const uint64_t recording_cycle_length = 44000;
  static uint64_t recorded_audio[44000];
  
  // For reference: uint64_t target_frequency = 500000000l;
  while (1) {
    printf("Recording!\r\n");
    while (counter < recording_cycle_length) {
      recorded_audio[counter] = read_I2S_rx(MIC_CHANNEL, I2S_LEFT);
      counter++;
    }
    printf("Playing!\r\n");
    while (playback < counter) {
      // int32_t* audio_ptr = &recorded_audio;
      // int16_t sample1 = (int16_t)(audio_ptr[playback*4    ] >> 16);
      // int16_t sample2 = (int16_t)(audio_ptr[playback*4 + 1] >> 16);
      // int16_t sample3 = (int16_t)(audio_ptr[playback*4 + 2] >> 16);
      // int16_t sample4 = (int16_t)(audio_ptr[playback*4 + 3] >> 16);
      // uint64_t sample = (((uint64_t)sample1) << 48) || 
      //                   (((uint64_t)sample2) << 32) || 
      //                   (((uint64_t)sample3) << 16) || 
      //                   (((uint64_t)sample4) << 0);

      // uint16_t* audio_ptr = &recorded_audio;
      // uint64_t left = ((uint64_t)(audio_ptr[((playback * 8) + 7) % counter]) << 48) | 
      //                 ((uint64_t)(audio_ptr[((playback * 8) + 5) % counter]) << 32) | 
      //                 ((uint64_t)(audio_ptr[((playback * 8) + 3) % counter]) << 16) | 
      //                 ((uint64_t)(audio_ptr[((playback * 8) + 1) % counter]));

      write_I2S_tx(SPEAKER_CHANNEL, I2S_LEFT, recorded_audio[playback]);
      write_I2S_tx(SPEAKER_CHANNEL, I2S_RIGHT, recorded_audio[playback]);
      playback++; 
    }
    counter = 0;
    playback = 0;
	}

  printf("Error: skipped while loop\r\n");

}

// TLDR: Plays a wav file
void i2s_wav_playback_test(void) {

  uint32_t i = 0;
  // uint32_t j = 0;
  // uint16_t* audio = meep_wav;
  uint16_t* audio = roll_wav;
  // uint16_t* audio = NULL;
  while (1) {
    uint32_t len = 994704 / 2;
    
    // Current samples to transmit
    uint64_t left = ((uint64_t)(audio[(i + 0) % len]) << 48) | 
                    ((uint64_t)(audio[(i + 0) % len]) << 32) | 
                    ((uint64_t)(audio[(i + 0) % len]) << 16) | 
                    (uint64_t)(audio[i % len]);

    uint64_t right = ((uint64_t)(audio[(i + 1) % len]) << 48) | 
                     ((uint64_t)(audio[(i + 1) % len]) << 32) | 
                     ((uint64_t)(audio[(i + 1) % len]) << 16) | 
                     (uint64_t)(audio[(i + 1) % len]);

    write_I2S_tx(SPEAKER_CHANNEL, true, left);
    write_I2S_tx(SPEAKER_CHANNEL, false, right);

    i += 2;
    // j += 8;
    // i = j/4;
	}

}

// TLDR: Plays live audio from mic through output (no time delay)
void i2s_live_feedback_test(void) {
  uint64_t mic_output;
  int32_t* samples = &mic_output;
  while (1) {
    // printf("reading mic_output");
    mic_output = read_I2S_rx(MIC_CHANNEL, I2S_LEFT);

    // uint64_t speaker_output = ((uint64_t)(samples[1] << 1) << 32) | ((uint64_t)(samples[0] << 2));

    // printf("sending output to speaker");
    write_I2S_tx(SPEAKER_CHANNEL, I2S_LEFT, mic_output);
    write_I2S_tx(SPEAKER_CHANNEL, I2S_RIGHT, mic_output);
  }
}

//================================convrev start=========================================
// // Conversion Helpers
// static inline float s32_to_float(int32_t sample) {
//     return (float)sample / 2147483648.0f; // Normalize to -1.0 to 1.0
// }

// static inline int32_t float_to_s32(float sample) {
//     if (sample > 1.0f) sample = 1.0f;
//     if (sample < -1.0f) sample = -1.0f;
//     return (int32_t)(sample * 2147483647.0f);
// }

// void i2s_convolution_reverb_demo(void) {
//     uint32_t input_buffer[BLOCK_SIZE];  
//     uint32_t output_buffer[CONV_OUT_SIZE]; 

//     float tail_buffer[KERNEL_SIZE]; 
//     memset(tail_buffer, 0, sizeof(tail_buffer)); // Clear tail initially

//     float kernel_float[KERNEL_SIZE] = {
//         .10f, 0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.04f, 0.03f,
//         0.02f, 0.01f, 0.009f, 0.008f, 0.007f, 0.006f, 0.005f, 0.04f
//     };

//     uint32_t* kernel_ptr = (uint32_t*)kernel_float;

//     while (1) {
//         for (int i = 0; i < BLOCK_SIZE; i++) {
//             uint64_t rx_data = read_I2S_rx(MIC_CHANNEL, I2S_LEFT);
//             int32_t sample_l = (int32_t)(rx_data & 0xFFFFFFFF);
//             ((float*)input_buffer)[i] = s32_to_float(sample_l);
//         }

//         perform_convolution_1D(
//             input_buffer, 
//             BLOCK_SIZE, 
//             kernel_ptr, 
//             KERNEL_SIZE, 
//             output_buffer, 
//             DILATION
//         );
        
//         float* out_f = (float*)output_buffer;

//         for (int k = 0; k < KERNEL_SIZE; k++) {
//             out_f[k] += tail_buffer[k];
//         }

//         for (int k = 0; k < KERNEL_SIZE; k++) {
//             tail_buffer[k] = out_f[BLOCK_SIZE + k];
//         }

//         for (int i = 0; i < BLOCK_SIZE; i++) {
//             int32_t final_sample = float_to_s32(out_f[i]);
//             uint64_t tx_data = ((uint64_t)final_sample << 32) | (uint32_t)final_sample;
//             write_I2S_tx(SPEAKER_CHANNEL, I2S_LEFT, tx_data);
//         }
//     }
// }
//================================convrev end=========================================

// --- GLOBALS (Static to avoid stack overflow) ---
// 1. The Long Kernel split into chunks [4][16]
//    (This example is a simple fading echo)
float kernel_segments[NUM_SEGMENTS][KERNEL_CHUNK];

// 2. History of Input Blocks (Ring Buffer)
//    We need to remember past audio to convolve with the later kernel segments
float input_history[NUM_SEGMENTS][BLOCK_SIZE];
int history_head = 0; // Index pointing to the "Current" block in history

// 3. Hardware Buffers (Reused for every segment calculation)
static uint32_t hw_input_buf[BLOCK_SIZE];
static uint32_t hw_output_buf[BLOCK_SIZE + KERNEL_CHUNK];

// 4. Overlap-Add Tail Buffer (Stores the decay between main blocks)
static float tail_buffer[KERNEL_CHUNK];


void init_long_kernel() {
    // Generate a long decay curve across all segments
    // Total Length = 64 samples
    for (int s = 0; s < NUM_SEGMENTS; s++) {
        for (int k = 0; k < KERNEL_CHUNK; k++) {
            // Create a simple exponential decay
            // The further back the segment, the quieter it should be
            int total_idx = (s * KERNEL_CHUNK) + k;
           
            // Example: Gain starts at 0.5 and decays
            float decay = 1.0f;
            for (int i=0; i<total_idx; i++) {
              decay *= 0.9f;
            }
            float val = 0.5f * decay;
           
            kernel_segments[s][k] = val;
        }
    }
   
    // Clear buffers
    memset(input_history, 0, sizeof(input_history));
    memset(tail_buffer, 0, sizeof(tail_buffer));
}

void i2s_long_reverb_test(void) {
    init_long_kernel();
    printf("Starting Partitioned Convolution (Segments: %d)\r\n", NUM_SEGMENTS);

    while (1) {
        // --- 1. Read New Audio & Update History ---
        // Advance the head of our circular history buffer
        history_head = (history_head - 1 + NUM_SEGMENTS) % NUM_SEGMENTS;
       
        // Fill the "Current" slot with new mic data
        for (int i = 0; i < BLOCK_SIZE; i++) {
            uint64_t rx_raw = read_I2S_rx(MIC_CHANNEL, I2S_LEFT);
            int32_t sample_i = (int32_t)(rx_raw & 0xFFFFFFFF);
           
            // Store as float in history
            input_history[history_head][i] = S32_TO_FLOAT(sample_i);
        }

        // --- 2. The Accumulator ---
        // This will hold the sum of all Partial Convolutions
        // Length = BLOCK_SIZE + KERNEL_CHUNK (48 samples)
        float accumulator[BLOCK_SIZE + KERNEL_CHUNK];
        memset(accumulator, 0, sizeof(accumulator));

        // --- 3. Run Partitioned Convolution Loop ---
        for (int s = 0; s < NUM_SEGMENTS; s++) {
           
            // A. Figure out which past input block corresponds to this kernel segment
            // If Segment 0 (Start of Kernel), we use Current Audio (Head)
            // If Segment 1, we use Previous Audio (Head + 1)
            int hist_idx = (history_head + s) % NUM_SEGMENTS;

            // B. Prepare Hardware Input Buffer
            // Convert the float history back to uint32 for the driver
            // (Optimize this by storing uint32 in history if possible, but float is safer for math)
            for(int i=0; i<BLOCK_SIZE; i++) {
                ((float*)hw_input_buf)[i] = input_history[hist_idx][i];
            }

            // C. Run Hardware Convolution
            // Convolve: History[s] * Kernel[s]
            perform_convolution_1D(
                hw_input_buf, BLOCK_SIZE,
                (uint32_t*)kernel_segments[s], KERNEL_CHUNK,
                hw_output_buf, DILATION
            );

            // D. Accumulate Result
            float* hw_out_f = (float*)hw_output_buf;
            for(int i=0; i < BLOCK_SIZE + KERNEL_CHUNK; i++) {
                accumulator[i] += hw_out_f[i];
            }
        }

        // --- 4. Overlap-Add & Output ---
        for (int i = 0; i < BLOCK_SIZE; i++) {
            // Add the tail from the PREVIOUS loop iteration
            if (i < KERNEL_CHUNK) {
                accumulator[i] += tail_buffer[i];
            }

            // Clamp and Output
            float safe_sample = CLAMP(accumulator[i], -0.99f, 0.99f);
            int32_t out_int = FLOAT_TO_S32(safe_sample);
           
            uint64_t tx_data = ((uint64_t)out_int << 32) | (uint32_t)out_int;
            write_I2S_tx(SPEAKER_CHANNEL, I2S_LEFT, tx_data);
        }

        // --- 5. Save New Tail ---
        // Save the end of the accumulator for the NEXT block
        for (int k = 0; k < KERNEL_CHUNK; k++) {
            tail_buffer[k] = accumulator[BLOCK_SIZE + k];
        }
    }
}

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  /* MCU Configuration--------------------------------------------------------*/

  /* Configure the system clock */
  /* Configure the system clock */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */  
  /* USER CODE BEGIN Init */
  //printf("About to start app init\r\n");
  app_init();
  /* USER CODE END Init */
 
  
  printf("About to start test\r\n");
  // i2s_live_feedback_test();
  // i2s_square_wave_test();
  // i2s_hollaback_test();
  // i2s_wav_playback_test();
  // i2s_convolution_reverb_demo();
  i2s_long_reverb_test();

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