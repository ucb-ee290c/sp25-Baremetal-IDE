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
#include "hal_2d_conv.h"
#include "chip_config.h"
#include <data/inputs.h>
#include <data/model_params.h>
#include <riscv_vector.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define CACHELINE 64

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

void requantize_2D1(
    size_t size, 
    int32_t bias,
    int16_t* input, 
    int8_t* output, 
    float scale, 
    int32_t zero_point
) 
{
    register vfloat32m8_t vfacc0;
    register vint32m8_t vacc0;
    register vint16m4_t vout0;
    vint8m2_t vout80;

    const int32_t output_min_less_zero_point = -128 - zero_point;
    const int32_t output_max_less_zero_point = 127 - zero_point;


    do {
        register size_t vl = __riscv_vsetvl_e16m4(size);
        vacc0 = __riscv_vwcvt_x_x_v_i32m8(__riscv_vle16_v_i16m4(input, vl), vl);
        vacc0 = __riscv_vadd_vx_i32m8(vacc0, bias, vl);
        vfacc0 = __riscv_vfcvt_f_x_v_f32m8(vacc0, vl);
        vfacc0 = __riscv_vfmul_vf_f32m8(vfacc0, scale, vl);
        vfacc0 = __riscv_vfmax_vf_f32m8(vfacc0, output_min_less_zero_point, vl);
        vfacc0 = __riscv_vfmin_vf_f32m8(vfacc0, output_max_less_zero_point, vl);
        vout0 = __riscv_vfncvt_x_f_w_i16m4(vfacc0, vl);
        vout0 = __riscv_vadd_vx_i16m4(vout0, (int32_t) zero_point, vl);
        vout80 = __riscv_vncvt_x_x_w_i8m2(vout0, vl);
        __riscv_vse8_v_i8m2(output, vout80, vl);

        input += vl;
        output += vl;
        size -= vl;
    } while (size != 0);
}


void dwconv_3x3_int8_VCO_acc(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    int relu,
    requantization_params_t requant_params
) {
    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (rows + 2) * a_stride;
    // Each channel's output is rows x b_stride (b_stride equals cols)
    size_t b_channel_size = rows * b_stride;
    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    __attribute__((aligned(CACHELINE))) int16_t out_conv[rows][cols];
    __attribute__((aligned(CACHELINE))) int8_t in_conv[rows+2][cols+2];


    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const int8_t *k_ch = w + ch * 9;
        
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        memcpy(in_conv, a_ch, (rows + 2) * (cols + 2));

        perform_convolution(
          (uint64_t)in_conv,   // Source address
          (uint64_t)out_conv,  // Destination address
          rows + 2,              // Input height
          cols + 2 ,               // Input width
          k_ch,          // Kernel values
          3,                       // Kernel size (3x3)
          relu,                       // Use ReLU activation
          1                        // Stride
        );

        requantize_2D1((rows)*(cols), ((const int32_t*) weights)[ch], (int16_t*) out_conv, b_ch, requant_params.scale[ch], requant_params.zero_point);
    }
}

void conv2D_3x3_int8_accelerator (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const void *dw_weights,  // length = Cin*(1 + 9)
    int8_t *input,       // CHW: [Cin][H][W]
    int8_t *output,            // CHW: [Cout][H_out][W_out]
    int relu,
    requantization_params_t requant_params_dwconv
) {
    size_t H_out = (H - 3)/stride + 1;
    size_t W_out = (W - 3)/stride + 1;

    dwconv_3x3_int8_VCO_acc(
        H_out, W_out, 
        Cin, 
        W, W_out, 
        dw_weights, 
        input, 
        output,
        relu, 
        requant_params_dwconv
    );
}


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
        conv2D_3x3_int8_accelerator(
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
        conv2D_3x3_int8_accelerator(
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