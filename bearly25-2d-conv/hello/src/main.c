/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body for 2D Convolution Test
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
#include "hal_2d_conv.h"
#include "chip_config.h"
#include <stdio.h>
#include <stdlib.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

// Example 3x3 kernel (Sobel edge detection kernel)
static int8_t example_kernel[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

// Example 5x5 input image
#define IMG_HEIGHT 5
#define IMG_WIDTH 5
static uint8_t input_image[IMG_HEIGHT][IMG_WIDTH] = {
    {10, 20, 30, 40, 50},
    {20, 30, 40, 50, 60},
    {30, 40, 50, 60, 70},
    {40, 50, 60, 70, 80},
    {50, 60, 70, 80, 90}
};

// Output buffer for the convolution result
// With a 3x3 kernel and no padding, the output dimensions will be (IMG_HEIGHT-2) x (IMG_WIDTH-2)
#define OUT_HEIGHT (IMG_HEIGHT - 2)
#define OUT_WIDTH (IMG_WIDTH - 2)
static int16_t output_image[OUT_HEIGHT][OUT_WIDTH];

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */


/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN PUC */

void app_init() {
    // Initialize the 2D convolution engine
    conv2d_init(CONV2D);
    printf("2D Convolution Engine initialized\n");
}

void app_main() {
    // Configure the 2D convolution engine
    conv2d_configure(
        CONV2D,
        (uint64_t)input_image,   // Source address
        (uint64_t)output_image,  // Destination address
        IMG_HEIGHT,              // Input height
        IMG_WIDTH,               // Input width
        3,                       // Kernel size (3x3)
        1,                       // Use ReLU activation
        1                        // Stride
    );
    
    // Set the convolution kernel
    conv2d_set_kernel(CONV2D, example_kernel, 3);
    
    // Start the convolution operation
    printf("Starting 2D convolution...\n");
    conv2d_start(CONV2D);
    
    // Wait for the operation to complete
    conv2d_wait_complete(CONV2D);
    
    // Check if operation completed successfully
    if (conv2d_is_ready(CONV2D) == 0xFF) {
        printf("ERROR: 2D convolution operation timed out!\n");
        return;
    }
    
    printf("2D convolution complete!\n");
    
    // Print the result
    printf("Convolution result:\n");
    for (int i = 0; i < OUT_HEIGHT; i++) {
        for (int j = 0; j < OUT_WIDTH; j++) {
            printf("%3d ", output_image[i][j]);
        }
        printf("\n");
    }
}

// Simple main function that just runs once
int main() {
    // Make sure we're on hart 0
    uint64_t mhartid;
    asm volatile ("csrr %0, mhartid" : "=r" (mhartid));
    if (mhartid != 0) {
        // If not on hart 0, just return
        return 0;
    }
    
    app_init();
    app_main();
    
    // Add a small delay to ensure output is printed
    for (volatile int i = 0; i < 1000; i++);
    
    return 0;
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
// void __attribute__((weak, noreturn)) __main(void) {
//   while (1) {
//    asm volatile ("wfi");
//   }
// }