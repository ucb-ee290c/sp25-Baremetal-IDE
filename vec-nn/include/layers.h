#include <stdint.h>
#include <stddef.h>
#ifndef VECNN_LAYERS_H
#define VECNN_LAYERS_H

/*---------------------------------------------*/
/*                                             */
/* Quantization helpers                        */
/*                                             */
/*---------------------------------------------*/
typedef struct {
    float   scale;
    int32_t zero_point;
} quantization_params_t;

typedef struct {
    float*  scale;
    int32_t zero_point;
} requantization_params_t;

void quant_f32(
    size_t size, 
    float* input, 
    int8_t* output, 
    quantization_params_t qp
);

void dequant_f32(
    size_t size, 
    int8_t* input, 
    float* output, 
    quantization_params_t qp
);

/*---------------------------------------------*/
/*                                             */
/* Transpose                                   */
/*                                             */
/*---------------------------------------------*/
void transpose_int8 (int8_t* input, int8_t* output, size_t rows, size_t cols);

/*---------------------------------------------*/
/*                                             */
/* Fully Connected Layers                      */
/*                                             */
/*---------------------------------------------*/
void fully_connected_f32 (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    float* input, 
    const float* weights_with_bias,
    float* output, 
    int relu
);

void quant_fully_connected_int8 (
    size_t input_size,
    size_t output_size,
    size_t batches,
    int8_t* input,
    const void* weights_with_bias,
    int8_t* output,
    int relu, int bias32,
    requantization_params_t requant_params
);

/*
 * quant_fully_connected_int8_t — transposed int8 fully-connected layer.
 *
 * Equivalent semantics to quant_fully_connected_int8 but expects the weight
 * matrix already transposed and packed, so that the vectorized dimension is
 * output_size (N) rather than the batch dimension.
 *
 * Changes vs. quant_fully_connected_int8:
 *   - Requantization: single scalar `scale` applied to the int32 accumulator
 *     to produce float32 output (no per-channel scale, no int8 narrowing).
 *   - Input bias term: zero — weights must be pre-converted from uint8 to
 *     int8 by subtracting 128 before calling (done once at model load time),
 *     so no zero-point correction is required at inference.
 *
 * weights_t_pack layout: [(input_size+1) × output_size] int8 bytes
 *   Row 0             : output_size zero bytes  (zero bias)
 *   Rows 1..input_size: rows of W_T as signed int8 (= original_uint8 − 128)
 *
 * Typical call for single-token transformer inference (batches=1):
 *   quant_fully_connected_int8_t(n, d, 1, x_q, w_t_pack, xout,
 *                                1.0f / (127.0f * 127.0f));
 */
void quant_fully_connected_int8_t(
    size_t input_size,
    size_t output_size,
    size_t batches,
    const int8_t* input,
    const void* weights_t_pack,
    float* output,
    float scale
);

/*---------------------------------------------*/
/*                                             */
/* 2D Convolution Layers                       */
/*                                             */
/*---------------------------------------------*/
void dwconv2D_3x3_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    const void *dw_weights,  // length = Cin*(1 + 9)
    int8_t *input,       // CHW: [Cin][H][W]
    int8_t *output,            // CHW: [Cout][H_out][W_out]
    int relu,
    requantization_params_t requant_params_dwconv
);

/*---------------------------------------------*/
/*                                             */
/* Pointwise Convolution Layers                */
/*                                             */
/*---------------------------------------------*/
void conv_1x1_int8(
    size_t rows, size_t cols, 
    size_t channels_in,
    size_t channels_out,
    size_t stride,
    size_t padding, // 0 for valid, 1 for same, 2 for full (NOT SUPPORTED YET)
    int8_t* input,
    const void* weights,
    int8_t* output, 
    int relu,
    requantization_params_t rqp
);

/*---------------------------------------------*/
/*                                             */
/* Pooling Layers.                             */
/*                                             */
/*---------------------------------------------*/
void maxpool_int8(
    size_t output_rows, size_t output_cols, 
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    int8_t *input, 
    int8_t *output
);

void maxpool_f32(
    size_t output_rows, size_t output_cols, 
    size_t input_rows, size_t input_cols,
    size_t channels,
    size_t stride,
    float *input, 
    float *output
);


/*---------------------------------------------*/
/*                                             */
/* Softmax                                     */
/*                                             */
/*---------------------------------------------*/
void softmax_vec(
    const float *i, 
    float *o, 
    size_t channels,
    size_t innerSize
);


/*---------------------------------------------*/
/*                                             */
/* Residual Add                                */
/*                                             */
/*---------------------------------------------*/

void residual_add(
    size_t rows, size_t cols, 
    size_t channels, 
    int8_t* a, int8_t* b, 
    int8_t* output, 
    requantization_params_t rqp
);

/*---------------------------------------------*/
/*                                             */
/* Activation Layers                           */
/*                                             */
/*---------------------------------------------*/
void relu6_int8(
    size_t channels,
    size_t inner_size,
    const float *input,
    int8_t *output,
    requantization_params_t requant_params
);

#endif
