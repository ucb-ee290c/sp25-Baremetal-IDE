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

#endif