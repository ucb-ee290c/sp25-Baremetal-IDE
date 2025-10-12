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
#include <layers.h>
#include <data/inputs.h>
#include <data/model_params.h>

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



void app_main() {
  uint64_t mhartid = READ_CSR("mhartid");

  printf("Hello world from hart %d: %d\n", mhartid, counter);

  /* Intermediate activation buffers */
  static float dense0_out[BATCHES * 64]; /* after Dense-0 + ReLU */
  static float logits    [BATCHES * 10];  /* output logits from Dense-1 */
  static float probs     [BATCHES * 10];  /* softmax probabilities */

  unsigned long cycles_start, cycles_end, instr_start, instr_end;
  asm volatile ("rdcycle %0" : "=r" (cycles_start));
  asm volatile ("rdinstret %0" : "=r" (instr_start));

  /* ---------------- Layer 0: 784 → 64 + ReLU ------------------------- */
  // fully_connected_f32 takes: input_dim, output_dim, batches, input, weight bias array, output, activation flag
  fully_connected_f32(784, 64, BATCHES, input, layer0, dense0_out, 1);

  /* ---------------- Layer 1: 64 → 10 (logits) ------------------------ */
  fully_connected_f32(64, 10, BATCHES, dense0_out, layer1, logits, 0);

  /* ---------------- Softmax per batch --------------------------------- */
  for (size_t b = 0; b < BATCHES; ++b) {
      softmax_vec(&logits[b * 10], &probs[b * 10], 10, 1);
  }

  asm volatile ("fence");
  asm volatile ("rdcycle %0" : "=r" (cycles_end));
  asm volatile ("rdinstret %0" : "=r" (instr_end));

  printf("  Execution cycles:      %lu\n", cycles_end - cycles_start);
  printf("  Instructions executed: %lu\n\n", instr_end - instr_start);

  /* ---------------- Print probabilities and predicted classes ------------------------------- */
  for (size_t b = 0; b < BATCHES; ++b) {
      int predicted = 0;
      float max_prob = probs[b * 10];
      for (int c = 1; c < 10; ++c) {
          if (probs[b * 10 + c] > max_prob) {
              max_prob = probs[b * 10 + c];
              predicted = c;
          }
      }
      printf("Input %d: Predicted digit %d, probabilities: ", b, predicted);
      for (int c = 0; c < 10; ++c) {
          printf("%d ", (int) (100 * probs[b * 10 + c]));
      }
      printf("\n");
  }

  // sleep(1);
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
  app_init();
  /* USER CODE END Init */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1) {
    app_main();
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