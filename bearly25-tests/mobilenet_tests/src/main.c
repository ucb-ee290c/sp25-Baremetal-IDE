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
#include <riscv_vector.h> 
#include <stdlib.h>


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

// void dwconv2D_3x3_int8 (
//     size_t H, size_t W,
//     size_t Cin,
//     size_t stride,
//     size_t padding,
//     const void *dw_weights,  // Cin*(1+9) int8_t
//     int8_t *input,           // [Cin][H][W]
//     int8_t *output,          // [Cin][H_out][W_out]
//     int relu,
//     requantization_params_t requant_params_dwconv
// );

#include "dwconv_p1s1_5x5_c3.h"   

void print_int8_matrix(
                    int8_t *arr,
                       size_t rows,
                       size_t cols)
{
    printf("matrix: \n");

    for (size_t r = 0; r < rows; r++) {
        printf("  [");
        for (size_t c = 0; c < cols; c++) {
            int idx = r * cols + c;
            printf("%4d", arr[idx]);
            if (c + 1 < cols) printf(", ");
        }
        printf("]\n");
    }
}

void print_vint32_m4(vint32m4_t vec, size_t n) {
    // Configure VL (vector length) for 32-bit elements, LMUL=4
    size_t vl = __riscv_vsetvl_e32m4(n);

    // Temporary buffer to hold vector contents (C99 VLA)
    int32_t buffer[vl];

    // Store vector elements into the buffer
    __riscv_vse32_v_i32m4(buffer, vec, vl);

    // Print each element
    for (size_t i = 0; i < vl; ++i) {
        printf("%d ", buffer[i]);
    }
    printf("\n");
}

void print_vint8_m1(vint8m1_t vec, size_t n) {
    // Configure VL (vector length) for 32-bit elements, LMUL=4
    size_t vl = __riscv_vsetvl_e8m1(n);

    // Temporary buffer to hold vector contents (C99 VLA)
    int8_t buffer[vl];

    // Store vector elements into the buffer
    __riscv_vse8_v_i8m1(buffer, vec, vl);

    // Print each element
    for (size_t i = 0; i < vl; ++i) {
        printf("%d ", buffer[i]);
    }
    printf("\n");
}

void print_vint16_m2(vint16m2_t vec, size_t n) {
    // Configure VL (vector length) for 32-bit elements, LMUL=4
    size_t vl = __riscv_vsetvl_e16m2(n);

    // Temporary buffer to hold vector contents (C99 VLA)
    int16_t buffer[vl];

    // Store vector elements into the buffer
    __riscv_vse16_v_i16m2(buffer, vec, vl);

    // Print each element
    for (size_t i = 0; i < vl; ++i) {
        printf("%d ", buffer[i]);
    }
    printf("\n");
}

static void relu6_postprocess_single_channel(
    int8_t *output,
    size_t rows,
    size_t stride,
    float scale,
    int32_t zero_point
) {
    size_t len = rows * stride;

    /*
     * Compute q6 = floor(6 / scale) + zp.
     * scale is positive, so truncation via (int32_t) is equivalent to floor().
     * This avoids pulling in libm for floorf.
     */
    int32_t q6 = (int32_t)((6.0f / scale) + (float)zero_point);
    if (q6 > 127) q6 = 127;
    if (q6 < -128) q6 = -128;

    for (size_t i = 0; i < len; i++) {
        int32_t v = output[i];
        if (v < zero_point) v = zero_point;
        if (v > q6) v = q6;
        output[i] = (int8_t)v;
    }
}

void pad_input_channel_(
    size_t input_cols, 
    size_t input_rows, 
    size_t x_padding, 
    size_t y_padding, 
    const int8_t* input, 
    int8_t* output
) 
{
    // printf("output: %p \n", output);
    register vint8m8_t temp;
    register size_t output_cols = input_cols + 2*x_padding;
    register size_t output_rows = input_rows + 2*y_padding;
    register size_t o_cols = output_cols - x_padding;
    register int32_t vl;
    do {
        register const int8_t* i = input;
        register int8_t* o = output; 
        vl = __riscv_vsetvl_e8m8(o_cols);
        register size_t rows = input_rows;

        // add padding in the top of vl-columns
        for (size_t i = 0; i < y_padding; i++) {
            __riscv_vse8_v_i8m8(o, __riscv_vmv_v_x_i8m8(0, vl), vl); o += output_cols;
        }

        do { 
            temp = __riscv_vle8_v_i8m8(i, vl);
            if (o_cols == output_cols - x_padding) {
                vint8m8_t zeros = __riscv_vmv_v_x_i8m8(0, vl);
                temp = __riscv_vslideup_vx_i8m8(zeros, temp, x_padding, vl);
            }
            __riscv_vse8_v_i8m8(o, temp, vl); o += output_cols;
            i += input_cols;
            rows -= 1;
        } while (rows != 0);

        // add padding in the bottom of vl-columns
        for (size_t i = 0; i < y_padding; i++) {
            __riscv_vse8_v_i8m8(o, __riscv_vmv_v_x_i8m8(0, vl), vl); o += output_cols;
        }

        input = (o_cols == output_cols - x_padding) ? input + vl - x_padding : input + vl;
        o_cols -= vl; 
        output = output + vl;

    } while (o_cols != 0);

    // printf("output: %p \n", output);
    for (size_t r = 0; r < output_rows; r++){
        vl = __riscv_vsetvl_e8m8(x_padding);
        __riscv_vse8_v_i8m8(output, __riscv_vmv_v_x_i8m8(0, vl), vl);
        output += output_cols;
    }
}


void vec_conv_c_code_(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 2;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = -128 - zero_point; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 

    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        print_vint16_m2(vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        print_vint16_m2(vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        print_vint16_m2(vload0, vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

        printf("cols: %d \n", cols);

        do {
            printf("k: %d \n", k6);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;

            print_vint32_m4(vrow0, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl); ap_2 += a_stride;             
            
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

            row_count -= 2;
        } while (row_count != 0);

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        print_vint32_m4(vrow0, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        if (!rows_odd) {
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl); ap += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl); ap_1 += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl); ap_2 += a_stride;

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl);

        }
        a += vl;
        b += vl;
        cols -= vl;

    } while (cols != 0);
    
}

void vec_conv_c_code_relu_(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 2;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = 0; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 


    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

        // printf("cols: %d \n", cols);

        do {
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;

            // print_vint32_m4(vrow0, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl); ap_2 += a_stride;             
            
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);

            row_count -= 2;
        } while (row_count != 0);

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        // print_vint32_m4(vrow0, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        if (!rows_odd) {
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl); ap += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl); ap_1 += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl); ap_2 += a_stride;

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_1, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ap_2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl);

        }
        a += vl;
        b += vl;
        cols -= vl;

    } while (cols != 0);
}

void vec_conv_c_code_stride2_(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 1;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = -128 - zero_point; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 

    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        print_vint32_m4(vrow0, vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        print_vint32_m4(vrow0, vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;
        print_vint32_m4(vrow0, vl);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

        // printf("cols: %d \n", cols);

        do {
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl); ap_2 += a_stride;             
            
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

            row_count -= 1;
        } while (row_count != 0);

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        // print_vint32_m4(vrow0, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        a += vl;
        b += vl;
        cols -= vl;

    } while (cols != 0);
    
}

void vec_conv_c_code_stride2_relu_(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    register size_t row_check = rows;
    row_check -= 2;
    register size_t row_count;
    register int rows_odd = rows & 1;

    register vint16m2_t vload0; 
    register vint16m2_t vload1; 
    register vint16m2_t vload2;
    register vint32m4_t vrow0; 
    register vint32m4_t vrow1; 
    vfloat32m4_t vfacc;
    register vint32m4_t vbias;
    vint16m2_t vout16;
    vint8m1_t vout8;

    register int16_t k0 = (int16_t) k[0]; register int16_t k1 = (int16_t) k[1]; register int16_t k2 = (int16_t) k[2];
    register int16_t k3 = (int16_t) k[3]; register int16_t k4 = (int16_t) k[4]; register int16_t k5 = (int16_t) k[5];
    register int16_t k6 = (int16_t) k[6]; register int16_t k7 = (int16_t) k[7]; register int16_t k8 = (int16_t) k[8];

    float vout_min_minus_zp = 0; 
    float vout_max_minus_zp = 127 - zero_point;  

    const int8_t* ap; const int8_t* ap_1; const int8_t* ap_2;
    int8_t* bp; 


    do {
        register size_t vl = __riscv_vsetvl_e32m4(cols);
        ap = a; ap_1 = ap + 1; ap_2 = ap + 2;
        bp = b; 
        row_count = row_check; 
        vbias = __riscv_vmv_v_x_i32m4(bias, vl);

        vl = __riscv_vsetvl_e8m1(cols);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl); ap += a_stride;
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl); ap_1 += a_stride;
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl); ap_2 += a_stride;


        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl); ap += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl); ap_1 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl); ap_2 += a_stride;
        vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);

        vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
        vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
        vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

        // printf("cols: %d \n", cols);

        do {
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl); ap_2 += a_stride;

            // print_vint32_m4(vrow0, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl); ap += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl); ap_1 += a_stride;
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl); ap_2 += a_stride;             
            
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k3, vload0, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k4, vload1, vl);
            vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k5, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

            vrow1 = __riscv_vwmacc_vx_i32m4(vbias, k0, vload0, vl);
            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k1, vload1, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k2, vload2, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);

            row_count -= 2;
        } while (row_count != 0);

        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k6, vload0, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k7, vload1, vl);
        vrow0 = __riscv_vwmacc_vx_i32m4(vrow0, k8, vload2, vl);

        // print_vint32_m4(vrow0, vl);
        
        vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow0, vl);
        vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
        vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
        vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
        vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
        vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
        vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
        __riscv_vse8_v_i8m1(bp, vout8, vl); bp += b_stride;

        if (!rows_odd) {
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k3, vload0, vl); ap += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k4, vload1, vl); ap_1 += a_stride;
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k5, vload2, vl); ap_2 += a_stride;

            vload0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap, 2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k6, vload0, vl);
            vload1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_1, 2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k7, vload1, vl);
            vload2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vlse8_v_i8m1(ap_2, 2, vl), vl);
            vrow1 = __riscv_vwmacc_vx_i32m4(vrow1, k8, vload2, vl);

            // print_vint32_m4(vrow1, vl);
            
            vfacc = __riscv_vfcvt_f_x_v_f32m4(vrow1, vl);
            vfacc = __riscv_vfmul_vf_f32m4(vfacc, scale, vl);
            vfacc = __riscv_vfmax_vf_f32m4(vfacc, vout_min_minus_zp, vl);
            vfacc = __riscv_vfmin_vf_f32m4(vfacc, vout_max_minus_zp, vl);
            vout16 = __riscv_vfncvt_x_f_w_i16m2(vfacc, vl);
            vout16 = __riscv_vadd_vx_i16m2(vout16, zero_point, vl);
            vout8 = __riscv_vncvt_x_x_w_i8m1(vout16, vl);
            __riscv_vse8_v_i8m1(bp, vout8, vl);

        }
        a += vl;
        b += vl;
        cols -= 2*vl;

    } while (cols != 0);
}

void dwconv_3x3_int8_VCO_(
    size_t input_rows, size_t input_cols,
    size_t stride, size_t padding,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    const int8_t *in_conv; 
    int8_t in_conv_buf [input_rows + 2*padding][input_cols + 2*padding];
    size_t rows = (input_rows + 2*padding - 3)/stride + 1;
    size_t cols = (input_cols + 2*padding - 3)/stride + 1;

    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (input_rows) * input_cols;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;
    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const int8_t *k_ch = w + ch * 9;
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        print_int8_matrix(k_ch, 3, 3);

        if (padding) {
            printf("Padding \n");
            print_int8_matrix(a_ch, input_cols, input_rows);
            pad_input_channel_(input_cols, input_rows, padding, padding, (const int8_t*) a_ch, (int8_t*) in_conv_buf);
            printf("Done Padding \n");
            in_conv = (int8_t*) in_conv_buf; 
            print_int8_matrix(in_conv, input_cols + 2, input_rows + 2);
        } else {
            in_conv = a_ch;
        }

        if (stride == 1) {
            printf("Conv \n");
            vec_conv_c_code_(rows, cols, a_stride + 2*padding, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
            printf("Done Conv \n");
        } else if (stride == 2) {
            vec_conv_c_code_stride2_(rows, cols, a_stride + 2*padding, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        }
    }
}

void dwconv_3x3_int8_VCO_relu_(
    size_t input_rows, size_t input_cols,
    size_t stride, size_t padding,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,      // weights: first 'channels' bias values, then 9 weights per channel
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    const int8_t *in_conv; 
    int8_t in_conv_buf [input_rows + 2*padding][input_cols + 2*padding];
    size_t rows = (input_rows + 2*padding - 3)/stride + 1;
    size_t cols = (input_cols + 2*padding - 3)/stride + 1;

    // Each channel's input is assumed to be a padded matrix with (rows+2) rows.
    size_t a_channel_size = (input_rows) * a_stride;
    // Each channel's output is rows x b_stride (typically b_stride equals cols)
    size_t b_channel_size = rows * b_stride;
    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    for (size_t ch = 0; ch < channels; ch++) {
        // The bias for this channel is stored at weights[ch].
        // float bias = weights[ch];
        // The 3x3 kernel for this channel is stored starting at weights[channels] with 9 floats per channel.
        const int8_t *k_ch = w + ch * 9;
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        print_int8_matrix(k_ch, 3, 3);

        if (padding) {
            pad_input_channel_(input_cols, input_rows, padding, padding, (const int8_t*) a_ch, (int8_t*) in_conv_buf);
            in_conv = (int8_t*) in_conv_buf; 
        } else {
            in_conv = a_ch;
        }

        if (stride == 1) {
            vec_conv_c_code_relu_(rows, cols, (a_stride + 2*padding)*stride, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        } else if (stride == 2) {
            vec_conv_c_code_stride2_relu_(rows, cols, (a_stride + 2*padding)*stride, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        }
    }
}

void vec_conv_c_code_relu6_(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    vec_conv_c_code_(rows, cols, a_stride, b_stride, k, a, b, bias, zero_point, scale);
    relu6_postprocess_single_channel(b, rows, b_stride, scale, zero_point);
}

void vec_conv_c_code_stride2_relu6_(
    size_t rows, size_t cols, 
    size_t a_stride, size_t b_stride, 
    const int8_t*k, 
    const int8_t*a, 
    int8_t* b, 
    int32_t bias, 
    int32_t zero_point, 
    float scale
) {
    vec_conv_c_code_stride2_(rows, cols, a_stride, b_stride, k, a, b, bias, zero_point, scale);
    relu6_postprocess_single_channel(b, rows, b_stride, scale, zero_point);
}

void dwconv_3x3_int8_VCO_relu6_(
    size_t input_rows, size_t input_cols,
    size_t stride, size_t padding,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
) {
    const int8_t *in_conv; 
    int8_t in_conv_buf [input_rows + 2*padding][input_cols + 2*padding];
    size_t rows = (input_rows + 2*padding - 3)/stride + 1;
    size_t cols = (input_cols + 2*padding - 3)/stride + 1;

    size_t a_channel_size = (input_rows) * a_stride;
    size_t b_channel_size = rows * b_stride;
    const int8_t* w = (const int8_t*) ((const int32_t*) weights + channels);

    for (size_t ch = 0; ch < channels; ch++) {
        const int8_t *k_ch = w + ch * 9;
        int8_t *a_ch = input + ch * a_channel_size;
        int8_t *b_ch = output + ch * b_channel_size;

        if (padding) {
            pad_input_channel_(input_cols, input_rows, padding, padding, (const int8_t*) a_ch, (int8_t*) in_conv_buf);
            in_conv = (int8_t*) in_conv_buf; 
        } else {
            in_conv = a_ch;
        }

        if (stride == 1) {
            vec_conv_c_code_relu6_(rows, cols, (a_stride + 2*padding)*stride, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        } else if (stride == 2) {
            vec_conv_c_code_stride2_relu6_(rows, cols, (a_stride + 2*padding)*stride, b_stride, k_ch, in_conv, b_ch, ((const int32_t*) weights)[ch], requant_params.zero_point, requant_params.scale[ch]);
        }
    }
}

void dwconv2D_3x3_int8_ (
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
    size_t H_out = (H + 2*padding - 3)/stride + 1;
    size_t W_out = (W + 2*padding - 3)/stride + 1;


    if (!relu) {
        dwconv_3x3_int8_VCO_(
            H, W,
            stride, padding, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output, 
            requant_params_dwconv
        );
    } else {
        dwconv_3x3_int8_VCO_relu_(
            H, W,
            stride, padding, 
            Cin, 
            W, W_out, 
            dw_weights, 
            input, 
            output, 
            requant_params_dwconv
        );
    }
}

void dwconv2D_3x3_int8_relu6 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding,
    const void *dw_weights,
    int8_t *input,
    int8_t *output,
    requantization_params_t requant_params_dwconv
) {
    size_t H_out = (H + 2*padding - 3)/stride + 1;
    size_t W_out = (W + 2*padding - 3)/stride + 1;

    dwconv_3x3_int8_VCO_relu6_(
        H, W,
        stride, padding, 
        Cin, 
        W, W_out, 
        dw_weights, 
        input, 
        output, 
        requant_params_dwconv
    );
}

static int compare_int8_buffers(const int8_t *a, const int8_t *b, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            if (errors < 16) { // don’t spam too much
                printf("  Mismatch at %zu: got %d, expected %d\n",
                       i, (int)a[i], (int)b[i]);
            }
            errors++;
        }
    }
    return errors;
}



void app_init() {
  // torch::executor::runtime_init();

}

void app_main() {
    const size_t H       = DWCONV_P1S1_5X5_C3_H;
    const size_t W       = DWCONV_P1S1_5X5_C3_W;
    const size_t Cin     = DWCONV_P1S1_5X5_C3_CIN;
    const size_t stride  = DWCONV_P1S1_5X5_C3_STRIDE;
    const size_t padding = DWCONV_P1S1_5X5_C3_PADDING;
    const size_t H_out   = DWCONV_P1S1_5X5_C3_H_OUT;
    const size_t W_out   = DWCONV_P1S1_5X5_C3_W_OUT;

    const size_t in_elems   = Cin * H * W;
    const size_t out_elems  = Cin * H_out * W_out;
    const size_t weight_len = Cin * (4 + 9); // bias (int32 as 4 bytes) + 3x3

    (void)in_elems;
    (void)weight_len;

    // Allocate output buffers
    int8_t *outA_norelu = (int8_t *)calloc(out_elems, sizeof(int8_t));
    int8_t *outA_relu   = (int8_t *)calloc(out_elems, sizeof(int8_t));
    int8_t *outA_relu6  = (int8_t *)calloc(out_elems, sizeof(int8_t));

    int8_t *outB_norelu = (int8_t *)calloc(out_elems, sizeof(int8_t));
    int8_t *outB_relu   = (int8_t *)calloc(out_elems, sizeof(int8_t));
    int8_t *outB_relu6  = (int8_t *)calloc(out_elems, sizeof(int8_t));

    int8_t *out_residual = (int8_t *)calloc(out_elems, sizeof(int8_t));

    if (!outA_norelu || !outA_relu || !outA_relu6 ||
        !outB_norelu || !outB_relu || !outB_relu6 ||
        !out_residual) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Set up requant params for Conv A
    requantization_params_t rqA;
    rqA.scale      = (float *)dwconv_p1s1_5x5_c3_scales;
    rqA.zero_point = 0;

    // Set up requant params for Conv B
    requantization_params_t rqB;
    rqB.scale      = (float *)dwconv_p1s1_5x5_c3_scales_b;
    rqB.zero_point = 0;

    // Requant params for residual add
    requantization_params_t rqRes;
    rqRes.scale      = (float *)dwconv_p1s1_5x5_c3_res_scales;
    rqRes.zero_point = 0;

    printf("Starting Conv A...\n");

    // --- Conv A: no ReLU ---
    dwconv2D_3x3_int8(
        H, W, Cin,
        stride, padding,
        (const void *)dwconv_p1s1_5x5_c3_weights,
        (int8_t *)dwconv_p1s1_5x5_c3_input,   // cast away const for API
        outA_norelu,
        /*relu=*/0,
        rqA
    );

    int errA0 = compare_int8_buffers(
        outA_norelu,
        dwconv_p1s1_5x5_c3_ref_norelu,
        out_elems
    );
    printf("Conv A No-ReLU test: %s (%d mismatches)\n",
           errA0 == 0 ? "PASS" : "FAIL", errA0);

    // --- Conv A: ReLU ---
    dwconv2D_3x3_int8(
        H, W, Cin,
        stride, padding,
        (const void *)dwconv_p1s1_5x5_c3_weights,
        (int8_t *)dwconv_p1s1_5x5_c3_input,
        outA_relu,
        /*relu=*/1,
        rqA
    );

    int errA1 = compare_int8_buffers(
        outA_relu,
        dwconv_p1s1_5x5_c3_ref_relu,
        out_elems
    );
    printf("Conv A ReLU test:    %s (%d mismatches)\n",
           errA1 == 0 ? "PASS" : "FAIL", errA1);

#ifdef DWCONV_P1S1_5X5_C3_HAS_RELU6
    // --- Conv A: ReLU6 ---
    dwconv2D_3x3_int8_relu6(
        H, W, Cin,
        stride, padding,
        (const void *)dwconv_p1s1_5x5_c3_weights,
        (int8_t *)dwconv_p1s1_5x5_c3_input,
        outA_relu6,
        rqA
    );

    int errA2 = compare_int8_buffers(
        outA_relu6,
        dwconv_p1s1_5x5_c3_ref_relu6,
        out_elems
    );
    printf("Conv A ReLU6 test:   %s (%d mismatches)\n",
           errA2 == 0 ? "PASS" : "FAIL", errA2);
#else
    int errA2 = 0;
    printf("Conv A ReLU6 test:   SKIPPED\n");
#endif

    printf("\nStarting Conv B...\n");

    // --- Conv B: no ReLU ---
    dwconv2D_3x3_int8(
        H, W, Cin,
        stride, padding,
        (const void *)dwconv_p1s1_5x5_c3_weights_b,
        (int8_t *)dwconv_p1s1_5x5_c3_input,
        outB_norelu,
        /*relu=*/0,
        rqB
    );

    int errB0 = compare_int8_buffers(
        outB_norelu,
        dwconv_p1s1_5x5_c3_ref_b_norelu,
        out_elems
    );
    printf("Conv B No-ReLU test: %s (%d mismatches)\n",
           errB0 == 0 ? "PASS" : "FAIL", errB0);

    // --- Conv B: ReLU ---
    dwconv2D_3x3_int8(
        H, W, Cin,
        stride, padding,
        (const void *)dwconv_p1s1_5x5_c3_weights_b,
        (int8_t *)dwconv_p1s1_5x5_c3_input,
        outB_relu,
        /*relu=*/1,
        rqB
    );

    int errB1 = compare_int8_buffers(
        outB_relu,
        dwconv_p1s1_5x5_c3_ref_b_relu,
        out_elems
    );
    printf("Conv B ReLU test:    %s (%d mismatches)\n",
           errB1 == 0 ? "PASS" : "FAIL", errB1);

#ifdef DWCONV_P1S1_5X5_C3_HAS_RELU6
    // --- Conv B: ReLU6 ---
    dwconv2D_3x3_int8_relu6(
        H, W, Cin,
        stride, padding,
        (const void *)dwconv_p1s1_5x5_c3_weights_b,
        (int8_t *)dwconv_p1s1_5x5_c3_input,
        outB_relu6,
        rqB
    );

    int errB2 = compare_int8_buffers(
        outB_relu6,
        dwconv_p1s1_5x5_c3_ref_b_relu6,
        out_elems
    );
    printf("Conv B ReLU6 test:   %s (%d mismatches)\n",
           errB2 == 0 ? "PASS" : "FAIL", errB2);
#else
    int errB2 = 0;
    printf("Conv B ReLU6 test:   SKIPPED\n");
#endif

    printf("\nStarting residual_add (A_norelu + B_norelu)...\n");

    // Residual add uses the no-ReLU outputs of A and B
    residual_add(
        H_out, W_out, Cin,
        outA_norelu, outB_norelu,
        out_residual,
        rqRes
    );

    int errR = compare_int8_buffers(
        out_residual,
        dwconv_p1s1_5x5_c3_res_ref,
        out_elems
    );
    printf("Residual add test:   %s (%d mismatches)\n",
           errR == 0 ? "PASS" : "FAIL", errR);

    free(outA_norelu);
    free(outA_relu);
    free(outA_relu6);
    free(outB_norelu);
    free(outB_relu);
    free(outB_relu6);
    free(out_residual);

    int ok =
        (errA0 == 0 && errA1 == 0 && errA2 == 0 &&
         errB0 == 0 && errB1 == 0 && errB2 == 0 &&
         errR  == 0);

    return;
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
  app_main();
  return 0;
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
