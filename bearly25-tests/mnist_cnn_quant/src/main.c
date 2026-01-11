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
/*
 * main.c - Quantized MNIST CNN inference test.
 *
 * Runs depthwise/pointwise convs, maxpool, and int8 FC layers with
 * dequantized softmax outputs for inspection.
 */
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

static int argmax10(const float *v)
{
    int idx = 0;
    float mx = v[0];
    for (int i = 1; i < 10; ++i) {
        if (v[i] > mx) {
            mx = v[i];
            idx = i;
        }
    }
    return idx;
}

void app_main() {
  uint64_t mhartid = READ_CSR("mhartid");

  printf("Hello world from hart %d: %d\n", mhartid, counter);

  /* Intermediate activation buffers */
    static int8_t  input_q   [BATCHES * 28 * 28];
    static int8_t  conv0_out [1 * 26 * 26];
    static int8_t  pw0_out   [16 * 26 * 26];
    static int8_t  pool0_out [16 *  8 *  8];

    static int8_t  conv1_out [16 *  6 *  6];
    static int8_t  pw1_out   [6 * 6 * 32];
    static int8_t  pool1_out [32 *  2 *  2];   /* = 128 int8 */

    static int8_t  dense0_q  [BATCHES * 32];
    static int8_t  logits_q  [BATCHES * 10];

    static float   logits_f32[BATCHES * 10];
    static float   probs      [BATCHES * 10];


  int i = 0;
    while (i < 20) {
        /* cycle counter ------------------------------------------------ */
        unsigned long cyc0, cyc1, ins0, ins1;
        asm volatile("rdcycle %0"   : "=r"(cyc0));
        asm volatile("rdinstret %0" : "=r"(ins0));

        /* --------------------------------------------------------------
           0) Quant input
           -------------------------------------------------------------- */
        quant_f32(
            BATCHES * 28*28,
            input + i*28*28,
            input_q,
            qp_input
        );

        /* --------------------------------------------------------------
           1) Conv-0 : 1×28×28 → 16×26×26   (DW)
           -------------------------------------------------------------- */
        conv2D_3x3_int8(
            /* H,W          */ 28, 28,
            /* Cin          */ 1,
            /* stride       */ 1,
            /* padding      */ 0,
            /* weights      */ (const void*) dw0_wb_q, 
            /* io           */ input_q, conv0_out,
            /* relu         */ 0,
            /* rq params    */ rq_conv0_dw
        );

        conv_1x1_int8(
            26, 26, 
            1, 16,
            1, 0, 
            conv0_out, 
            (const void*) pw0_wb_q,
            pw0_out, 
            1, 
            rq_conv0_pw
        );

        /* --------------------------------------------------------------
           2) MaxPool-0 : 3×3,str=3  –> 16×8×8
           -------------------------------------------------------------- */
        maxpool_int8(
            /* out rows,cols */ 8, 8,
            /* in  rows,cols */ 26, 26,
            /* channels      */ 16,
            /* stride        */ 3,
            pw0_out, pool0_out
        );

        /* --------------------------------------------------------------
           3) Conv-1 : 16×8×8 → 32×6×6
           -------------------------------------------------------------- */
        conv2D_3x3_int8(
            /* H,W          */ 8, 8,
            /* Cin,Cout     */ 16,
            /* stride       */ 1,
            /* padding      */ 0,
            (const void*) dw1_wb_q,
            pool0_out, conv1_out,
            0,
            rq_conv1_dw
        );

        conv_1x1_int8(
            6, 6, 
            16, 32,
            1, 0, 
            conv1_out, 
            (const void*) pw1_wb_q,
            pw1_out,
            1, 
            rq_conv1_pw
        );

        /* --------------------------------------------------------------
           4) MaxPool-1 : 3×3,str=3  –> 32×2×2
           -------------------------------------------------------------- */
        maxpool_int8(
            /* out rows,cols */ 2, 2,
            /* in  rows,cols */ 6, 6,
            /* channels      */ 32,
            /* stride        */ 3,
            pw1_out, pool1_out
        );

        /* --------------------------------------------------------------
           5) FC-0 : 128 → 32   (+ReLU)
           -------------------------------------------------------------- */
        quant_fully_connected_int8(
            /* in, out       */ 128, 32,
            /* batches       */ 1,
            /* input         */ pool1_out,
            /* weights+bias  */ (const void*) fc0_wb_q,
            /* output        */ dense0_q,
            /* relu flag     */ 1, 1,
            /* rq params     */ rq_fc0
        );

        /* --------------------------------------------------------------
           6) FC-1 : 32 → 10    (logits only)
           -------------------------------------------------------------- */
        quant_fully_connected_int8(
            32, 10,
            1,
            dense0_q,
            (const void*) fc1_wb_q,
            logits_q,
            0, 1,
            rq_fc1
        );

        /* --------------------------------------------------------------
           7) Dequant logits → float, Softmax, print
           -------------------------------------------------------------- */
        dequant_f32(
            BATCHES * 10,
            logits_q,
            logits_f32,
            qp_logits
        );
        for (size_t b = 0; b < BATCHES; b++) {
            softmax_vec(&logits_f32[b*10], &probs[b*10], 10, 1);
        }

        asm volatile("fence");
        asm volatile("rdcycle %0"   : "=r"(cyc1));
        asm volatile("rdinstret %0" : "=r"(ins1));

        printf("Cycles      : %lu\n", cyc1 - cyc0);
        printf("Instructions: %lu\n", ins1 - ins0);

        for (size_t b = 0; b < BATCHES; b++) {
            int pred = argmax10(&probs[b*10]);
            printf("Sample %d → %d  probs:", i, pred);
            for (int c = 0; c < 10; c++) {
                printf(" %d", (int)(probs[b*10 + c]*100));
            }
            printf("\n");
        }

        i++;
    }

    return 0;

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
