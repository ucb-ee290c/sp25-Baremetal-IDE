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

// I2S at 44kHz -> ~440hz square wave
#define PULSE_PERIOD_SAMPLES (100) // 44kHz -> 100 samples per 440Hz period
#define PULSE_WIDTH_SAMPLES (50) // 50% duty cycle
#define AMPLITUDE (0xEEEEEEEE) // Sample amplitude for 32 bit depth

#define SPEAKER_CHANNEL 2
#define MIC_CHANNEL 3

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
    .clkdiv      = 88,
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
  .clkdiv      = 88,
  .tx_fp       = 0,
  .rx_fp       = 0,
  .tx_force_left = 0,
  .rx_force_left = 0
};

// PLL target frequency = 500 MHz; different than system clock of 50 MHz
uint64_t target_frequency = 500000000l;


void app_init() {
  configure_pll(PLL, target_frequency/50000000, 0);
  set_all_clocks(CLOCK_SELECTOR, 1);

  UART_InitType UART_init_config;
  UART_init_config.baudrate = 115200;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  uart_init(UART0, &UART_init_config);
  UART0->DIV = (target_frequency / 115200) - 1;

  printf("I2S params initializing\r\n");

  config_I2S(SPEAKER_CHANNEL, &i2s_params_speaker);
  config_I2S(MIC_CHANNEL, &i2s_params_mic);

  printf("Init done\r\n");
}

void i2s_square_wave_test(void) {

  uint32_t counter = 0;
  uint32_t sample[2] = {AMPLITUDE, AMPLITUDE};
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


// NOTE: This is from DSP24 Audio
// https://github.com/ucb-bar/sp24-Baremetal-IDE/blob/audio/app/src/main.c
void i2s_playback_test(void) {
  uint64_t counter = 0;
  uint64_t playback = 0;
  uint64_t recording_length = 5; //In Seconds
  uint64_t recording_cycle_length = (recording_length * 44100 / 4);
  uint64_t recorded_audio[recording_cycle_length];

  // For reference: uint64_t target_frequency = 500000000l;
  while (1) {
    printf("Recording!\r\n");
    while (counter < recording_cycle_length) {
      recorded_audio[counter] = read_I2S_rx(MIC_CHANNEL, I2S_LEFT);
      counter++;
    }
    printf("Playing!\r\n");
    while (playback < counter) {
      write_I2S_tx(SPEAKER_CHANNEL, I2S_LEFT, recorded_audio[playback]);
      playback += 2; // Not sure why we increment by 2 here
    }
	}

}

void i2s_mic_test(void) {
  printf("Entered i2s_mic_test\r\n");
  uint64_t counter = 0;
  uint64_t playback = 0;
  // const static int playback_seconds = 5;
  // const static int num_samples_per_i2s_frame = 2;
  
  // const uint64_t recording_cycle_length = 5 * 44100 / 4;
  // static uint64_t recorded_audio[5 * 44100 / 4];

  // const uint64_t recording_cycle_length = (uint64_t) 0.1 * 44100 / 2;
  // static uint64_t recorded_audio[(uint64_t) (0.1 * 44100 / 2)];

  const uint64_t recording_cycle_length = 2048;
  static uint64_t recorded_audio[2048];
  
  // For reference: uint64_t target_frequency = 500000000l;
  while (1) {
    // printf("Recording!\r\n");
    while (counter < recording_cycle_length) {
      recorded_audio[counter] = read_I2S_rx(MIC_CHANNEL, I2S_LEFT);
      counter++;
    }
    // printf("Playing!\r\n");
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

void i2s_wav_playback_test(void) {

  uint32_t i = 0;
  uint16_t* audio = NULL;

  while (1) {
    uint32_t len = 994704 / 2;
    
    // Current samples to transmit
    uint64_t left = ((uint64_t)(audio[(i + 6) % len]) << 48) | 
                    ((uint64_t)(audio[(i + 4) % len]) << 32) | 
                    ((uint64_t)(audio[(i + 2) % len]) << 16) | 
                    (uint64_t)(audio[i % len]);

    uint64_t right = ((uint64_t)(audio[(i + 7) % len]) << 48) | 
                     ((uint64_t)(audio[(i + 5) % len]) << 32) | 
                     ((uint64_t)(audio[(i + 3) % len]) << 16) | 
                     (uint64_t)(audio[(i + 1) % len]);

    write_I2S_tx(SPEAKER_CHANNEL, true, left);
    write_I2S_tx(SPEAKER_CHANNEL, false, right);

    i += 8;
	}

}

void i2s_feedback_test(void) {
  uint64_t mic_output;
  int32_t* samples = &mic_output;
  while (1) {
    mic_output = read_I2S_rx(MIC_CHANNEL, I2S_LEFT);

    // uint64_t speaker_output = ((uint64_t)(samples[1] << 1) << 32) | ((uint64_t)(samples[0] << 2));


    write_I2S_tx(SPEAKER_CHANNEL, I2S_LEFT, mic_output);
    write_I2S_tx(SPEAKER_CHANNEL, I2S_RIGHT, mic_output);
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
  app_init();
  /* USER CODE END Init */
 
  // i2s_square_wave_test();
  i2s_feedback_test();
  // i2s_mic_test();

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