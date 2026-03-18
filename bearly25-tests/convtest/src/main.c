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
/*
 * main.c - 2D convolution accelerator test.
 *
 * Compares a software reference convolution against the hardware block and
 * reports mismatches and timing.
 */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "hal_2d_conv.h"
#include "chip_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include "layers.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */


// Constants
#define MMIO_BASE        0x08808000
#define CACHELINE        64
#define WIDTH            5 //<TOKEN_WIDTH>
#define HEIGHT           5 //<TOKEN_HEIGHT>
#define KERNEL_SIZE      3 //<TOKEN_KERNEL_SIZE>
#define USE_RELU         0 //<TOKEN_USE_RELU>
#define STRIDE           1 //<TOKEN_STRIDE>

typedef enum {
    INT8,
    INT16
} DataType;

__attribute__((aligned(CACHELINE))) int16_t outputImage[HEIGHT-2][WIDTH-2];

__attribute__((aligned(CACHELINE))) int16_t expectedOutputImage[HEIGHT-2][WIDTH-2];

  // SOBEL KERNEL
__attribute__((aligned(CACHELINE))) int8_t kernel[3][3] = {
    { -1,  0,  1 },
    { -2,  0,  2 },
    { -1,  0, -1 } //SOBEL KERNEL
};

__attribute__((aligned(CACHELINE))) int8_t testImage[HEIGHT][WIDTH] = {
    {10, 20, 30, 40, 50},
    {20, 30, 40, 50, 60},
    {30, 40, 50, 60, 70},
    {40, 50, 60, 70, 80},
    {50, 60, 70, 80, 90}
};


unsigned long read_cycles(void)
{
    unsigned long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

void printImage(void *image, int height, int width, DataType type) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (type == INT8) {
                // For Printing Input Image 
                printf("%3d ", ((int8_t*)image)[y * width + x]);
            } else if (type == INT16) {
                // For Printing Convolution Output / Expected Output 
                printf("%5hd ", ((int16_t*)image)[y * width + x]);
            }
        }
        printf("\n");
    }
}

// void populateTestImage(int8_t* testImage, int height, int width) {
//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x < width; ++x) {
//             int index = y * width + x;
//             testImage[index] = matrix[index];  
//         }
//     }
// }

void convolve(int8_t* input, int inputWidth, int inputHeight, int8_t useReLU,
              int8_t kernel[3][3], int16_t* output) {

    // Iterate through image
    for (int i = 1; i < inputHeight - 1; i++) {
        for (int j = 1; j < inputWidth - 1; j++) {
            int32_t sum = 0;

            // Iterate through kernel (3x3)
            for (int m = -1; m <= 1; m++) { 
                for (int n = -1; n <= 1; n++) { 
                    int ii = i + m;
                    int jj = j + n;

                    // Perform the multiplication and addition for the convolution
                    sum += input[ii * inputWidth + jj] * kernel[m + 1][n + 1];
                }
            }

            // Saturate to int16_t range
            if (sum > 32767) {
                sum = 32767;
            }
            if (sum < -32768) {
                sum = -32768;
            }

            int outputIndex = (i - 1) * (inputWidth - 2 * 1) + (j - 1);
            output[outputIndex] = useReLU && sum < 0 ? 0 : sum;
        }
    }
}




// // Example 3x3 kernel (Sobel edge detection kernel)
// static int8_t example_kernel[9] = {
//     -1, 0, 1,
//     -2, 0, 2,
//     -1, 0, 1
// };

// // Example 5x5 input image
// #define IMG_HEIGHT 5
// #define IMG_WIDTH 5
// static uint8_t input_image[IMG_HEIGHT][IMG_WIDTH] = {
//     {10, 20, 30, 40, 50},
//     {20, 30, 40, 50, 60},
//     {30, 40, 50, 60, 70},
//     {40, 50, 60, 70, 80},
//     {50, 60, 70, 80, 90}
// };

// // Output buffer for the convolution result
// // With a 3x3 kernel and no padding, the output dimensions will be (IMG_HEIGHT-2) x (IMG_WIDTH-2)
// #define OUT_HEIGHT (IMG_HEIGHT - 2)
// #define OUT_WIDTH (IMG_WIDTH - 2)
// static int16_t output_image[OUT_HEIGHT][OUT_WIDTH];

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
    UART_InitType UART0_init_config;
    UART0_init_config.baudrate = 115200;
    UART0_init_config.mode = UART_MODE_TX_RX;
    UART0_init_config.stopbits = UART_STOPBITS_2;
    uart_init(UART0, &UART0_init_config);
    // Initialize the 2D convolution engine
    // conv2d_init(CONV2D);
    printf("2D Convolution Engine initialized\n");
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmul_test(float* xout, float* x, float* w, int n, int d) {
    fully_connected_f32_nobias(
        (size_t)n,          // input_size
        (size_t)1,          // output_size
        (size_t)d,          // batches
        x,                  // input
        (const float*)w,    // weights_with_bias (see NOTE below)
        xout,               // output
        0                   // relu off
    );
}

int n = 4;
int d = 3;

void app_main() {

    float W[] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };

    float x[] = { 1, 1, 1, 1 };

    float y_ref[d];
    float y_fc[d];

    // Clear outputs
    for (int i = 0; i < d; i++) {
        y_ref[i] = 0.0f;
        y_fc[i]  = 0.0f;
    }

    matmul(y_ref, x, W, n, d);
    matmul_test(y_fc, x, W, n, d);

    printf("Reference vs FC output:\n");
    for (int i = 0; i < d; i++) {
        printf("i=%d  ref=%df  fc=%df\n",
               i, y_ref[i], y_fc[i]);
    }

    return 0;
    // Perform 2D convolution using the wrapper function
    // printf("\nInput Image is size %d by %d\n", HEIGHT, WIDTH);
    // printf("Starting 2D convolution...\n");

    // convolve((int8_t*) &testImage, WIDTH, HEIGHT, 0, kernel, (int16_t*) &expectedOutputImage);
    // perform_convolution(
    //     (uint64_t)input_image,   // Source address
    //     (uint64_t)output_image,  // Destination address
    //     IMG_HEIGHT,              // Input height
    //     IMG_WIDTH,               // Input width
    //     example_kernel,          // Kernel values
    //     3,                       // Kernel size (3x3)
    //     1,                       // Use ReLU activation
    //     1                        // Stride
    // );
    
    // Save for later, in case we want to manually call functions

    // // Set the convolution kernel
    // conv2d_set_kernel(CONV2D, example_kernel, 3);
    
    // // Start the convolution operation
    // printf("Starting 2D convolution...\n");
    // conv2d_start(CONV2D);
    
    // // Wait for the operation to complete
    // conv2d_wait_complete(CONV2D);
    
    // // Check if operation completed successfully
    // if (conv2d_is_ready(CONV2D) == 0xFF) {
    //     printf("ERROR: 2D convolution operation timed out!\n");
    //     return;
    // }
    
    // printf("2D convolution complete!\n");
    
    // Print the result
    // printf("Convolution result:\n");
    // for (int i = 0; i < OUT_HEIGHT; i++) {
    //     for (int j = 0; j < OUT_WIDTH; j++) {
    //         printf("%3d ", output_image[i][j]);
    //     }
    //     printf("\n");
    // }

    // puts("testing puts");

    // uint8_t status = perform_convolution((uint64_t)&testImage, (uint64_t)&outputImage, HEIGHT, WIDTH, (uint8_t*)kernel, KERNEL_SIZE, (uint8_t)USE_RELU, STRIDE);

    // if (/f("Check status: %p\n", status);
    // printf("outputImage: %p\n", outputImage);
    // printImage(&outputImage, HEIGHT-2, WIDTH-2, INT16);



    // Compare hardware output to software output
    // puts("Comparing hardware output to expected output...");
    // if (memcmp(outputImage, expectedOutputImage, sizeof(outputImage)) == 0) {
    //     printf("Convolution output matches expected result.\n");
    // } else {
    //     printf("Test failed: Mismatch in convolution output.\n");
    // }

    // printf("expectedOutputImage: %p\n", expectedOutputImage);
    // printImage(&expectedOutputImage, HEIGHT-2, WIDTH-2, INT16);
    //Print outputImage
    // printf("outputImage: %p\n", outputImage);
    // printImage(&outputImage, HEIGHT-2, WIDTH-2, INT16);

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
void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
   asm volatile ("wfi");
  }
}
