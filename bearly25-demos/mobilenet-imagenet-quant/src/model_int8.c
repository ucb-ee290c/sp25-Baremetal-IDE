/*
 * model_code_int8.c - INT8 Quantized MobileNetV2 Inference Code
 * Auto-generated from model_int8.c
 *
 * Contains:
 * - Tensor unions for memory optimization
 * - Node functions (quantize/dequantize, conv, etc.)
 * - entry() function for inference
 */

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define MAX(X,Y) ( X > Y ? X : Y)
#define MIN(X,Y) ( X < Y ? X : Y)
#define CLIP(X,L) ( MAX(MIN(X,L), -L) )

#if __STDC_VERSION__ < 199901L
#define FUNC_PREFIX
#else
#define FUNC_PREFIX static inline
#endif


// Include weights from header files
#include "model_weights_int8.h"
#include "classifier_weights_int8.h"

union tensor_union_0 {
float tensor_classifier_1_bias[1000];
uint8_t tensor_logits_QuantizeLinear_Output[1][1000];
};
static union tensor_union_0 tu0;

union tensor_union_1 {
float tensor_classifier_1_weight_DequantizeLinear_Output[1000][1280];
};
static union tensor_union_1 tu1;

union tensor_union_2 {
uint8_t tensor_input_QuantizeLinear_Output[1][3][224][224];
float tensor__features_features_0_features_0_2_Clip_output_0[1][32][112][112];
float tensor__features_features_0_features_0_2_Clip_output_0_DequantizeLinear_Output[1][32][112][112];
uint8_t tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][32][112][112];
float tensor__features_features_1_conv_conv_1_Conv_output_0[1][16][112][112];
float tensor__features_features_1_conv_conv_1_Conv_output_0_DequantizeLinear_Output[1][16][112][112];
uint8_t tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][96][112][112];
float tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0[1][96][56][56];
float tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][96][56][56];
uint8_t tensor__features_features_2_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][24][56][56];
float tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0[1][144][56][56];
float tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][144][56][56];
uint8_t tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][144][56][56];
float tensor__features_features_3_conv_conv_2_Conv_output_0[1][24][56][56];
float tensor__features_features_3_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][24][56][56];
uint8_t tensor__features_features_3_Add_output_0_QuantizeLinear_Output[1][24][56][56];
float tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0[1][144][56][56];
float tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][144][56][56];
uint8_t tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][144][28][28];
float tensor__features_features_4_conv_conv_2_Conv_output_0[1][32][28][28];
float tensor__features_features_4_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][32][28][28];
uint8_t tensor__features_features_5_Add_output_0_QuantizeLinear_Output[1][32][28][28];
float tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0[1][192][28][28];
float tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][192][28][28];
uint8_t tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][192][28][28];
float tensor__features_features_6_conv_conv_2_Conv_output_0[1][32][28][28];
float tensor__features_features_6_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][32][28][28];
uint8_t tensor__features_features_6_Add_output_0_QuantizeLinear_Output[1][32][28][28];
float tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0[1][192][28][28];
float tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][192][28][28];
uint8_t tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][192][14][14];
float tensor__features_features_7_conv_conv_2_Conv_output_0[1][64][14][14];
float tensor__features_features_7_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][64][14][14];
uint8_t tensor__features_features_8_Add_output_0_QuantizeLinear_Output[1][64][14][14];
float tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0[1][384][14][14];
float tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_9_conv_conv_2_Conv_output_0[1][64][14][14];
float tensor__features_features_9_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][64][14][14];
uint8_t tensor__features_features_9_Add_output_0_QuantizeLinear_Output[1][64][14][14];
float tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0[1][384][14][14];
float tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_10_conv_conv_2_Conv_output_0[1][64][14][14];
float tensor__features_features_10_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][64][14][14];
uint8_t tensor__features_features_10_Add_output_0_QuantizeLinear_Output[1][64][14][14];
float tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0[1][384][14][14];
float tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_11_conv_conv_2_Conv_output_0[1][96][14][14];
float tensor__features_features_11_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][96][14][14];
uint8_t tensor__features_features_12_Add_output_0_QuantizeLinear_Output[1][96][14][14];
float tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0[1][576][14][14];
float tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][576][14][14];
uint8_t tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][576][14][14];
float tensor__features_features_13_conv_conv_2_Conv_output_0[1][96][14][14];
float tensor__features_features_13_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][96][14][14];
uint8_t tensor__features_features_13_Add_output_0_QuantizeLinear_Output[1][96][14][14];
float tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0[1][576][14][14];
float tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][576][14][14];
uint8_t tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][576][7][7];
float tensor__features_features_14_conv_conv_2_Conv_output_0[1][160][7][7];
float tensor__features_features_14_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][160][7][7];
uint8_t tensor__features_features_15_Add_output_0_QuantizeLinear_Output[1][160][7][7];
float tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0[1][960][7][7];
float tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][960][7][7];
uint8_t tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][960][7][7];
float tensor__features_features_16_conv_conv_2_Conv_output_0[1][160][7][7];
float tensor__features_features_16_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][160][7][7];
uint8_t tensor__features_features_16_Add_output_0_QuantizeLinear_Output[1][160][7][7];
float tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0[1][960][7][7];
float tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][960][7][7];
uint8_t tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][960][7][7];
float tensor__features_features_17_conv_conv_2_Conv_output_0[1][320][7][7];
float tensor__features_features_17_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][320][7][7];
uint8_t tensor__features_features_18_features_18_2_Clip_output_0_QuantizeLinear_Output[1][1280][7][7];
float tensor__GlobalAveragePool_output_0[1][1280][1][1];
float tensor__GlobalAveragePool_output_0_DequantizeLinear_Output[1][1280][1][1];
uint8_t tensor__Flatten_output_0_QuantizeLinear_Output[1][1280];
float tensor_logits_QuantizeLinear_Input[1][1000];
};
static union tensor_union_2 tu2;

union tensor_union_3 {
float tensor_onnx__Conv_538_DequantizeLinear_Output[32][3][3][3];
uint8_t tensor__features_features_0_features_0_2_Clip_output_0_QuantizeLinear_Output[1][32][112][112];
float tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0[1][32][112][112];
float tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][32][112][112];
uint8_t tensor__features_features_1_conv_conv_1_Conv_output_0_QuantizeLinear_Output[1][16][112][112];
float tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0[1][96][112][112];
float tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][96][112][112];
uint8_t tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][96][56][56];
float tensor__features_features_2_conv_conv_2_Conv_output_0[1][24][56][56];
float tensor__features_features_2_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][24][56][56];
float tensor__features_features_3_Add_output_0_DequantizeLinear_Output[1][24][56][56];
uint8_t tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][144][56][56];
float tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0[1][144][28][28];
float tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][144][28][28];
uint8_t tensor__features_features_4_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][32][28][28];
float tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0[1][192][28][28];
float tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][192][28][28];
uint8_t tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][192][28][28];
float tensor__features_features_5_conv_conv_2_Conv_output_0[1][32][28][28];
float tensor__features_features_5_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][32][28][28];
float tensor__features_features_5_Add_output_0_DequantizeLinear_Output[1][32][28][28];
float tensor__features_features_6_Add_output_0_DequantizeLinear_Output[1][32][28][28];
uint8_t tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][192][28][28];
float tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0[1][192][14][14];
float tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][192][14][14];
uint8_t tensor__features_features_7_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][64][14][14];
float tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0[1][384][14][14];
float tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_8_conv_conv_2_Conv_output_0[1][64][14][14];
float tensor__features_features_8_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][64][14][14];
float tensor__features_features_8_Add_output_0_DequantizeLinear_Output[1][64][14][14];
float tensor__features_features_9_Add_output_0_DequantizeLinear_Output[1][64][14][14];
float tensor__features_features_10_Add_output_0_DequantizeLinear_Output[1][64][14][14];
uint8_t tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0[1][384][14][14];
float tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_11_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][96][14][14];
float tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0[1][576][14][14];
float tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][576][14][14];
uint8_t tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][576][14][14];
float tensor__features_features_12_conv_conv_2_Conv_output_0[1][96][14][14];
float tensor__features_features_12_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][96][14][14];
float tensor__features_features_12_Add_output_0_DequantizeLinear_Output[1][96][14][14];
float tensor__features_features_13_Add_output_0_DequantizeLinear_Output[1][96][14][14];
uint8_t tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][576][14][14];
float tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0[1][576][7][7];
float tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][576][7][7];
uint8_t tensor__features_features_14_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][160][7][7];
float tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0[1][960][7][7];
float tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output[1][960][7][7];
uint8_t tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output[1][960][7][7];
float tensor__features_features_15_conv_conv_2_Conv_output_0[1][160][7][7];
float tensor__features_features_15_conv_conv_2_Conv_output_0_DequantizeLinear_Output[1][160][7][7];
float tensor__features_features_15_Add_output_0_DequantizeLinear_Output[1][160][7][7];
float tensor__features_features_16_Add_output_0_DequantizeLinear_Output[1][160][7][7];
uint8_t tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][960][7][7];
float tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0[1][960][7][7];
float tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][960][7][7];
uint8_t tensor__features_features_17_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][320][7][7];
float tensor__features_features_18_features_18_2_Clip_output_0[1][1280][7][7];
float tensor__features_features_18_features_18_2_Clip_output_0_DequantizeLinear_Output[1][1280][7][7];
uint8_t tensor__GlobalAveragePool_output_0_QuantizeLinear_Output[1][1280][1][1];
float tensor__Flatten_output_0[1][1280];
float tensor__Flatten_output_0_DequantizeLinear_Output[1][1280];
};
static union tensor_union_3 tu3;

union tensor_union_4 {
float tensor_onnx__Conv_539[32];
uint8_t tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][144][56][56];
float tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0[1][144][56][56];
float tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][144][56][56];
uint8_t tensor__features_features_3_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][24][56][56];
float tensor__features_features_3_Add_output_0[1][24][56][56];
uint8_t tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][192][28][28];
float tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0[1][192][28][28];
float tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][192][28][28];
uint8_t tensor__features_features_5_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][32][28][28];
float tensor__features_features_5_Add_output_0[1][32][28][28];
uint8_t tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][192][28][28];
float tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0[1][192][28][28];
float tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][192][28][28];
uint8_t tensor__features_features_6_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][32][28][28];
float tensor__features_features_6_Add_output_0[1][32][28][28];
uint8_t tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0[1][384][14][14];
float tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_8_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][64][14][14];
float tensor__features_features_8_Add_output_0[1][64][14][14];
uint8_t tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0[1][384][14][14];
float tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_9_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][64][14][14];
float tensor__features_features_9_Add_output_0[1][64][14][14];
uint8_t tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][384][14][14];
float tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0[1][384][14][14];
float tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][384][14][14];
uint8_t tensor__features_features_10_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][64][14][14];
float tensor__features_features_10_Add_output_0[1][64][14][14];
uint8_t tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][576][14][14];
float tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0[1][576][14][14];
float tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][576][14][14];
uint8_t tensor__features_features_12_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][96][14][14];
float tensor__features_features_12_Add_output_0[1][96][14][14];
uint8_t tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][576][14][14];
float tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0[1][576][14][14];
float tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][576][14][14];
uint8_t tensor__features_features_13_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][96][14][14];
float tensor__features_features_13_Add_output_0[1][96][14][14];
uint8_t tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][960][7][7];
float tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0[1][960][7][7];
float tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][960][7][7];
uint8_t tensor__features_features_15_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][160][7][7];
float tensor__features_features_15_Add_output_0[1][160][7][7];
uint8_t tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output[1][960][7][7];
float tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0[1][960][7][7];
float tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output[1][960][7][7];
uint8_t tensor__features_features_16_conv_conv_2_Conv_output_0_QuantizeLinear_Output[1][160][7][7];
float tensor__features_features_16_Add_output_0[1][160][7][7];
};
static union tensor_union_4 tu4;

union tensor_union_5 {
float tensor_onnx__Conv_541_DequantizeLinear_Output[32][1][3][3];
};
static union tensor_union_5 tu5;

union tensor_union_6 {
float tensor_onnx__Conv_542[32];
};
static union tensor_union_6 tu6;

union tensor_union_7 {
float tensor_onnx__Conv_544_DequantizeLinear_Output[16][32][1][1];
};
static union tensor_union_7 tu7;

union tensor_union_8 {
float tensor_onnx__Conv_545[16];
};
static union tensor_union_8 tu8;

union tensor_union_9 {
float tensor_onnx__Conv_547_DequantizeLinear_Output[96][16][1][1];
};
static union tensor_union_9 tu9;

union tensor_union_10 {
float tensor_onnx__Conv_548[96];
};
static union tensor_union_10 tu10;

union tensor_union_11 {
float tensor_onnx__Conv_550_DequantizeLinear_Output[96][1][3][3];
};
static union tensor_union_11 tu11;

union tensor_union_12 {
float tensor_onnx__Conv_551[96];
};
static union tensor_union_12 tu12;

union tensor_union_13 {
float tensor_onnx__Conv_553_DequantizeLinear_Output[24][96][1][1];
};
static union tensor_union_13 tu13;

union tensor_union_14 {
float tensor_onnx__Conv_554[24];
};
static union tensor_union_14 tu14;

union tensor_union_15 {
float tensor_onnx__Conv_556_DequantizeLinear_Output[144][24][1][1];
};
static union tensor_union_15 tu15;

union tensor_union_16 {
float tensor_onnx__Conv_557[144];
};
static union tensor_union_16 tu16;

union tensor_union_17 {
float tensor_onnx__Conv_559_DequantizeLinear_Output[144][1][3][3];
};
static union tensor_union_17 tu17;

union tensor_union_18 {
float tensor_onnx__Conv_560[144];
};
static union tensor_union_18 tu18;

union tensor_union_19 {
float tensor_onnx__Conv_562_DequantizeLinear_Output[24][144][1][1];
};
static union tensor_union_19 tu19;

union tensor_union_20 {
float tensor_onnx__Conv_563[24];
};
static union tensor_union_20 tu20;

union tensor_union_21 {
float tensor_onnx__Conv_565_DequantizeLinear_Output[144][24][1][1];
};
static union tensor_union_21 tu21;

union tensor_union_22 {
float tensor_onnx__Conv_566[144];
};
static union tensor_union_22 tu22;

union tensor_union_23 {
float tensor_onnx__Conv_568_DequantizeLinear_Output[144][1][3][3];
};
static union tensor_union_23 tu23;

union tensor_union_24 {
float tensor_onnx__Conv_569[144];
};
static union tensor_union_24 tu24;

union tensor_union_25 {
float tensor_onnx__Conv_571_DequantizeLinear_Output[32][144][1][1];
};
static union tensor_union_25 tu25;

union tensor_union_26 {
float tensor_onnx__Conv_572[32];
};
static union tensor_union_26 tu26;

union tensor_union_27 {
float tensor_onnx__Conv_574_DequantizeLinear_Output[192][32][1][1];
};
static union tensor_union_27 tu27;

union tensor_union_28 {
float tensor_onnx__Conv_575[192];
};
static union tensor_union_28 tu28;

union tensor_union_29 {
float tensor_onnx__Conv_577_DequantizeLinear_Output[192][1][3][3];
};
static union tensor_union_29 tu29;

union tensor_union_30 {
float tensor_onnx__Conv_578[192];
};
static union tensor_union_30 tu30;

union tensor_union_31 {
float tensor_onnx__Conv_580_DequantizeLinear_Output[32][192][1][1];
};
static union tensor_union_31 tu31;

union tensor_union_32 {
float tensor_onnx__Conv_581[32];
};
static union tensor_union_32 tu32;

union tensor_union_33 {
float tensor_onnx__Conv_583_DequantizeLinear_Output[192][32][1][1];
};
static union tensor_union_33 tu33;

union tensor_union_34 {
float tensor_onnx__Conv_584[192];
};
static union tensor_union_34 tu34;

union tensor_union_35 {
float tensor_onnx__Conv_586_DequantizeLinear_Output[192][1][3][3];
};
static union tensor_union_35 tu35;

union tensor_union_36 {
float tensor_onnx__Conv_587[192];
};
static union tensor_union_36 tu36;

union tensor_union_37 {
float tensor_onnx__Conv_589_DequantizeLinear_Output[32][192][1][1];
};
static union tensor_union_37 tu37;

union tensor_union_38 {
float tensor_onnx__Conv_590[32];
};
static union tensor_union_38 tu38;

union tensor_union_39 {
float tensor_onnx__Conv_592_DequantizeLinear_Output[192][32][1][1];
};
static union tensor_union_39 tu39;

union tensor_union_40 {
float tensor_onnx__Conv_593[192];
};
static union tensor_union_40 tu40;

union tensor_union_41 {
float tensor_onnx__Conv_595_DequantizeLinear_Output[192][1][3][3];
};
static union tensor_union_41 tu41;

union tensor_union_42 {
float tensor_onnx__Conv_596[192];
};
static union tensor_union_42 tu42;

union tensor_union_43 {
float tensor_onnx__Conv_598_DequantizeLinear_Output[64][192][1][1];
};
static union tensor_union_43 tu43;

union tensor_union_44 {
float tensor_onnx__Conv_599[64];
};
static union tensor_union_44 tu44;

union tensor_union_45 {
float tensor_onnx__Conv_601_DequantizeLinear_Output[384][64][1][1];
};
static union tensor_union_45 tu45;

union tensor_union_46 {
float tensor_onnx__Conv_602[384];
};
static union tensor_union_46 tu46;

union tensor_union_47 {
float tensor_onnx__Conv_604_DequantizeLinear_Output[384][1][3][3];
};
static union tensor_union_47 tu47;

union tensor_union_48 {
float tensor_onnx__Conv_605[384];
};
static union tensor_union_48 tu48;

union tensor_union_49 {
float tensor_onnx__Conv_607_DequantizeLinear_Output[64][384][1][1];
};
static union tensor_union_49 tu49;

union tensor_union_50 {
float tensor_onnx__Conv_608[64];
};
static union tensor_union_50 tu50;

union tensor_union_51 {
float tensor_onnx__Conv_610_DequantizeLinear_Output[384][64][1][1];
};
static union tensor_union_51 tu51;

union tensor_union_52 {
float tensor_onnx__Conv_611[384];
};
static union tensor_union_52 tu52;

union tensor_union_53 {
float tensor_onnx__Conv_613_DequantizeLinear_Output[384][1][3][3];
};
static union tensor_union_53 tu53;

union tensor_union_54 {
float tensor_onnx__Conv_614[384];
};
static union tensor_union_54 tu54;

union tensor_union_55 {
float tensor_onnx__Conv_616_DequantizeLinear_Output[64][384][1][1];
};
static union tensor_union_55 tu55;

union tensor_union_56 {
float tensor_onnx__Conv_617[64];
};
static union tensor_union_56 tu56;

union tensor_union_57 {
float tensor_onnx__Conv_619_DequantizeLinear_Output[384][64][1][1];
};
static union tensor_union_57 tu57;

union tensor_union_58 {
float tensor_onnx__Conv_620[384];
};
static union tensor_union_58 tu58;

union tensor_union_59 {
float tensor_onnx__Conv_622_DequantizeLinear_Output[384][1][3][3];
};
static union tensor_union_59 tu59;

union tensor_union_60 {
float tensor_onnx__Conv_623[384];
};
static union tensor_union_60 tu60;

union tensor_union_61 {
float tensor_onnx__Conv_625_DequantizeLinear_Output[64][384][1][1];
};
static union tensor_union_61 tu61;

union tensor_union_62 {
float tensor_onnx__Conv_626[64];
};
static union tensor_union_62 tu62;

union tensor_union_63 {
float tensor_onnx__Conv_628_DequantizeLinear_Output[384][64][1][1];
};
static union tensor_union_63 tu63;

union tensor_union_64 {
float tensor_onnx__Conv_629[384];
};
static union tensor_union_64 tu64;

union tensor_union_65 {
float tensor_onnx__Conv_631_DequantizeLinear_Output[384][1][3][3];
};
static union tensor_union_65 tu65;

union tensor_union_66 {
float tensor_onnx__Conv_632[384];
};
static union tensor_union_66 tu66;

union tensor_union_67 {
float tensor_onnx__Conv_634_DequantizeLinear_Output[96][384][1][1];
};
static union tensor_union_67 tu67;

union tensor_union_68 {
float tensor_onnx__Conv_635[96];
};
static union tensor_union_68 tu68;

union tensor_union_69 {
float tensor_onnx__Conv_637_DequantizeLinear_Output[576][96][1][1];
};
static union tensor_union_69 tu69;

union tensor_union_70 {
float tensor_onnx__Conv_638[576];
};
static union tensor_union_70 tu70;

union tensor_union_71 {
float tensor_onnx__Conv_640_DequantizeLinear_Output[576][1][3][3];
};
static union tensor_union_71 tu71;

union tensor_union_72 {
float tensor_onnx__Conv_641[576];
};
static union tensor_union_72 tu72;

union tensor_union_73 {
float tensor_onnx__Conv_643_DequantizeLinear_Output[96][576][1][1];
};
static union tensor_union_73 tu73;

union tensor_union_74 {
float tensor_onnx__Conv_644[96];
};
static union tensor_union_74 tu74;

union tensor_union_75 {
float tensor_onnx__Conv_646_DequantizeLinear_Output[576][96][1][1];
};
static union tensor_union_75 tu75;

union tensor_union_76 {
float tensor_onnx__Conv_647[576];
};
static union tensor_union_76 tu76;

union tensor_union_77 {
float tensor_onnx__Conv_649_DequantizeLinear_Output[576][1][3][3];
};
static union tensor_union_77 tu77;

union tensor_union_78 {
float tensor_onnx__Conv_650[576];
};
static union tensor_union_78 tu78;

union tensor_union_79 {
float tensor_onnx__Conv_652_DequantizeLinear_Output[96][576][1][1];
};
static union tensor_union_79 tu79;

union tensor_union_80 {
float tensor_onnx__Conv_653[96];
};
static union tensor_union_80 tu80;

union tensor_union_81 {
float tensor_onnx__Conv_655_DequantizeLinear_Output[576][96][1][1];
};
static union tensor_union_81 tu81;

union tensor_union_82 {
float tensor_onnx__Conv_656[576];
};
static union tensor_union_82 tu82;

union tensor_union_83 {
float tensor_onnx__Conv_658_DequantizeLinear_Output[576][1][3][3];
};
static union tensor_union_83 tu83;

union tensor_union_84 {
float tensor_onnx__Conv_659[576];
};
static union tensor_union_84 tu84;

union tensor_union_85 {
float tensor_onnx__Conv_661_DequantizeLinear_Output[160][576][1][1];
};
static union tensor_union_85 tu85;

union tensor_union_86 {
float tensor_onnx__Conv_662[160];
};
static union tensor_union_86 tu86;

union tensor_union_87 {
float tensor_onnx__Conv_664_DequantizeLinear_Output[960][160][1][1];
};
static union tensor_union_87 tu87;

union tensor_union_88 {
float tensor_onnx__Conv_665[960];
};
static union tensor_union_88 tu88;

union tensor_union_89 {
float tensor_onnx__Conv_667_DequantizeLinear_Output[960][1][3][3];
};
static union tensor_union_89 tu89;

union tensor_union_90 {
float tensor_onnx__Conv_668[960];
};
static union tensor_union_90 tu90;

union tensor_union_91 {
float tensor_onnx__Conv_670_DequantizeLinear_Output[160][960][1][1];
};
static union tensor_union_91 tu91;

union tensor_union_92 {
float tensor_onnx__Conv_671[160];
};
static union tensor_union_92 tu92;

union tensor_union_93 {
float tensor_onnx__Conv_673_DequantizeLinear_Output[960][160][1][1];
};
static union tensor_union_93 tu93;

union tensor_union_94 {
float tensor_onnx__Conv_674[960];
};
static union tensor_union_94 tu94;

union tensor_union_95 {
float tensor_onnx__Conv_676_DequantizeLinear_Output[960][1][3][3];
};
static union tensor_union_95 tu95;

union tensor_union_96 {
float tensor_onnx__Conv_677[960];
};
static union tensor_union_96 tu96;

union tensor_union_97 {
float tensor_onnx__Conv_679_DequantizeLinear_Output[160][960][1][1];
};
static union tensor_union_97 tu97;

union tensor_union_98 {
float tensor_onnx__Conv_680[160];
};
static union tensor_union_98 tu98;

union tensor_union_99 {
float tensor_onnx__Conv_682_DequantizeLinear_Output[960][160][1][1];
};
static union tensor_union_99 tu99;

union tensor_union_100 {
float tensor_onnx__Conv_683[960];
};
static union tensor_union_100 tu100;

union tensor_union_101 {
float tensor_onnx__Conv_685_DequantizeLinear_Output[960][1][3][3];
};
static union tensor_union_101 tu101;

union tensor_union_102 {
float tensor_onnx__Conv_686[960];
};
static union tensor_union_102 tu102;

union tensor_union_103 {
float tensor_onnx__Conv_688_DequantizeLinear_Output[320][960][1][1];
};
static union tensor_union_103 tu103;

union tensor_union_104 {
float tensor_onnx__Conv_689[320];
};
static union tensor_union_104 tu104;

union tensor_union_105 {
float tensor_onnx__Conv_691_DequantizeLinear_Output[1280][320][1][1];
};
static union tensor_union_105 tu105;

union tensor_union_106 {
float tensor_onnx__Conv_692[1280];
};
static union tensor_union_106 tu106;

union tensor_union_107 {
float tensor_input_DequantizeLinear_Output[1][3][224][224];
};
static union tensor_union_107 tu107;


/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: classifier.1.bias_DequantizeLinear
 */
FUNC_PREFIX void node_classifier_1_bias_DequantizeLinear( const int32_t x[1000], const float x_scale[1000], const int32_t x_zero_point[1000], float y[1000] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1000; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: classifier.1.weight_DequantizeLinear
 */
FUNC_PREFIX void node_classifier_1_weight_DequantizeLinear( const int8_t x[1000][1280], const float x_scale[1000], const int8_t x_zero_point[1000], float y[1000][1280] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1000; i0++)
	for (unsigned i1 = 0; i1 < 1280; i1++)
	{
		y[i0][i1] = (x[i0][i1] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: input_QuantizeLinear
 */
FUNC_PREFIX void node_input_QuantizeLinear( const float x[1][3][224][224], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][3][224][224] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 3; i1++)
	for (unsigned i2 = 0; i2 < 224; i2++)
	for (unsigned i3 = 0; i3 < 224; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_538_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_538_DequantizeLinear( const int8_t x[32][3][3][3], const float x_scale[32], const int8_t x_zero_point[32], float y[32][3][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	for (unsigned i1 = 0; i1 < 3; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_539_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_539_DequantizeLinear( const int32_t x[32], const float x_scale[32], const int32_t x_zero_point[32], float y[32] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_541_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_541_DequantizeLinear( const int8_t x[32][1][3][3], const float x_scale[32], const int8_t x_zero_point[32], float y[32][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_542_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_542_DequantizeLinear( const int32_t x[32], const float x_scale[32], const int32_t x_zero_point[32], float y[32] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_544_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_544_DequantizeLinear( const int8_t x[16][32][1][1], const float x_scale[16], const int8_t x_zero_point[16], float y[16][32][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 16; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_545_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_545_DequantizeLinear( const int32_t x[16], const float x_scale[16], const int32_t x_zero_point[16], float y[16] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 16; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_547_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_547_DequantizeLinear( const int8_t x[96][16][1][1], const float x_scale[96], const int8_t x_zero_point[96], float y[96][16][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	for (unsigned i1 = 0; i1 < 16; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_548_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_548_DequantizeLinear( const int32_t x[96], const float x_scale[96], const int32_t x_zero_point[96], float y[96] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_550_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_550_DequantizeLinear( const int8_t x[96][1][3][3], const float x_scale[96], const int8_t x_zero_point[96], float y[96][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_551_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_551_DequantizeLinear( const int32_t x[96], const float x_scale[96], const int32_t x_zero_point[96], float y[96] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_553_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_553_DequantizeLinear( const int8_t x[24][96][1][1], const float x_scale[24], const int8_t x_zero_point[24], float y[24][96][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 24; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_554_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_554_DequantizeLinear( const int32_t x[24], const float x_scale[24], const int32_t x_zero_point[24], float y[24] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 24; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_556_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_556_DequantizeLinear( const int8_t x[144][24][1][1], const float x_scale[144], const int8_t x_zero_point[144], float y[144][24][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_557_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_557_DequantizeLinear( const int32_t x[144], const float x_scale[144], const int32_t x_zero_point[144], float y[144] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_559_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_559_DequantizeLinear( const int8_t x[144][1][3][3], const float x_scale[144], const int8_t x_zero_point[144], float y[144][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_560_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_560_DequantizeLinear( const int32_t x[144], const float x_scale[144], const int32_t x_zero_point[144], float y[144] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_562_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_562_DequantizeLinear( const int8_t x[24][144][1][1], const float x_scale[24], const int8_t x_zero_point[24], float y[24][144][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 24; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_563_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_563_DequantizeLinear( const int32_t x[24], const float x_scale[24], const int32_t x_zero_point[24], float y[24] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 24; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_565_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_565_DequantizeLinear( const int8_t x[144][24][1][1], const float x_scale[144], const int8_t x_zero_point[144], float y[144][24][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_566_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_566_DequantizeLinear( const int32_t x[144], const float x_scale[144], const int32_t x_zero_point[144], float y[144] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_568_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_568_DequantizeLinear( const int8_t x[144][1][3][3], const float x_scale[144], const int8_t x_zero_point[144], float y[144][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_569_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_569_DequantizeLinear( const int32_t x[144], const float x_scale[144], const int32_t x_zero_point[144], float y[144] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 144; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_571_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_571_DequantizeLinear( const int8_t x[32][144][1][1], const float x_scale[32], const int8_t x_zero_point[32], float y[32][144][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_572_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_572_DequantizeLinear( const int32_t x[32], const float x_scale[32], const int32_t x_zero_point[32], float y[32] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_574_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_574_DequantizeLinear( const int8_t x[192][32][1][1], const float x_scale[192], const int8_t x_zero_point[192], float y[192][32][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_575_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_575_DequantizeLinear( const int32_t x[192], const float x_scale[192], const int32_t x_zero_point[192], float y[192] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_577_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_577_DequantizeLinear( const int8_t x[192][1][3][3], const float x_scale[192], const int8_t x_zero_point[192], float y[192][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_578_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_578_DequantizeLinear( const int32_t x[192], const float x_scale[192], const int32_t x_zero_point[192], float y[192] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_580_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_580_DequantizeLinear( const int8_t x[32][192][1][1], const float x_scale[32], const int8_t x_zero_point[32], float y[32][192][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_581_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_581_DequantizeLinear( const int32_t x[32], const float x_scale[32], const int32_t x_zero_point[32], float y[32] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_583_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_583_DequantizeLinear( const int8_t x[192][32][1][1], const float x_scale[192], const int8_t x_zero_point[192], float y[192][32][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_584_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_584_DequantizeLinear( const int32_t x[192], const float x_scale[192], const int32_t x_zero_point[192], float y[192] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_586_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_586_DequantizeLinear( const int8_t x[192][1][3][3], const float x_scale[192], const int8_t x_zero_point[192], float y[192][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_587_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_587_DequantizeLinear( const int32_t x[192], const float x_scale[192], const int32_t x_zero_point[192], float y[192] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_589_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_589_DequantizeLinear( const int8_t x[32][192][1][1], const float x_scale[32], const int8_t x_zero_point[32], float y[32][192][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_590_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_590_DequantizeLinear( const int32_t x[32], const float x_scale[32], const int32_t x_zero_point[32], float y[32] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 32; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_592_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_592_DequantizeLinear( const int8_t x[192][32][1][1], const float x_scale[192], const int8_t x_zero_point[192], float y[192][32][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_593_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_593_DequantizeLinear( const int32_t x[192], const float x_scale[192], const int32_t x_zero_point[192], float y[192] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_595_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_595_DequantizeLinear( const int8_t x[192][1][3][3], const float x_scale[192], const int8_t x_zero_point[192], float y[192][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_596_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_596_DequantizeLinear( const int32_t x[192], const float x_scale[192], const int32_t x_zero_point[192], float y[192] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 192; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_598_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_598_DequantizeLinear( const int8_t x[64][192][1][1], const float x_scale[64], const int8_t x_zero_point[64], float y[64][192][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_599_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_599_DequantizeLinear( const int32_t x[64], const float x_scale[64], const int32_t x_zero_point[64], float y[64] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_601_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_601_DequantizeLinear( const int8_t x[384][64][1][1], const float x_scale[384], const int8_t x_zero_point[384], float y[384][64][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_602_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_602_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_604_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_604_DequantizeLinear( const int8_t x[384][1][3][3], const float x_scale[384], const int8_t x_zero_point[384], float y[384][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_605_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_605_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_607_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_607_DequantizeLinear( const int8_t x[64][384][1][1], const float x_scale[64], const int8_t x_zero_point[64], float y[64][384][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_608_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_608_DequantizeLinear( const int32_t x[64], const float x_scale[64], const int32_t x_zero_point[64], float y[64] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_610_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_610_DequantizeLinear( const int8_t x[384][64][1][1], const float x_scale[384], const int8_t x_zero_point[384], float y[384][64][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_611_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_611_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_613_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_613_DequantizeLinear( const int8_t x[384][1][3][3], const float x_scale[384], const int8_t x_zero_point[384], float y[384][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_614_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_614_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_616_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_616_DequantizeLinear( const int8_t x[64][384][1][1], const float x_scale[64], const int8_t x_zero_point[64], float y[64][384][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_617_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_617_DequantizeLinear( const int32_t x[64], const float x_scale[64], const int32_t x_zero_point[64], float y[64] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_619_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_619_DequantizeLinear( const int8_t x[384][64][1][1], const float x_scale[384], const int8_t x_zero_point[384], float y[384][64][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_620_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_620_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_622_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_622_DequantizeLinear( const int8_t x[384][1][3][3], const float x_scale[384], const int8_t x_zero_point[384], float y[384][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_623_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_623_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_625_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_625_DequantizeLinear( const int8_t x[64][384][1][1], const float x_scale[64], const int8_t x_zero_point[64], float y[64][384][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_626_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_626_DequantizeLinear( const int32_t x[64], const float x_scale[64], const int32_t x_zero_point[64], float y[64] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 64; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_628_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_628_DequantizeLinear( const int8_t x[384][64][1][1], const float x_scale[384], const int8_t x_zero_point[384], float y[384][64][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_629_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_629_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_631_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_631_DequantizeLinear( const int8_t x[384][1][3][3], const float x_scale[384], const int8_t x_zero_point[384], float y[384][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_632_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_632_DequantizeLinear( const int32_t x[384], const float x_scale[384], const int32_t x_zero_point[384], float y[384] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 384; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_634_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_634_DequantizeLinear( const int8_t x[96][384][1][1], const float x_scale[96], const int8_t x_zero_point[96], float y[96][384][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_635_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_635_DequantizeLinear( const int32_t x[96], const float x_scale[96], const int32_t x_zero_point[96], float y[96] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_637_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_637_DequantizeLinear( const int8_t x[576][96][1][1], const float x_scale[576], const int8_t x_zero_point[576], float y[576][96][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_638_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_638_DequantizeLinear( const int32_t x[576], const float x_scale[576], const int32_t x_zero_point[576], float y[576] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_640_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_640_DequantizeLinear( const int8_t x[576][1][3][3], const float x_scale[576], const int8_t x_zero_point[576], float y[576][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_641_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_641_DequantizeLinear( const int32_t x[576], const float x_scale[576], const int32_t x_zero_point[576], float y[576] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_643_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_643_DequantizeLinear( const int8_t x[96][576][1][1], const float x_scale[96], const int8_t x_zero_point[96], float y[96][576][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_644_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_644_DequantizeLinear( const int32_t x[96], const float x_scale[96], const int32_t x_zero_point[96], float y[96] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_646_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_646_DequantizeLinear( const int8_t x[576][96][1][1], const float x_scale[576], const int8_t x_zero_point[576], float y[576][96][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_647_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_647_DequantizeLinear( const int32_t x[576], const float x_scale[576], const int32_t x_zero_point[576], float y[576] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_649_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_649_DequantizeLinear( const int8_t x[576][1][3][3], const float x_scale[576], const int8_t x_zero_point[576], float y[576][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_650_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_650_DequantizeLinear( const int32_t x[576], const float x_scale[576], const int32_t x_zero_point[576], float y[576] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_652_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_652_DequantizeLinear( const int8_t x[96][576][1][1], const float x_scale[96], const int8_t x_zero_point[96], float y[96][576][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_653_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_653_DequantizeLinear( const int32_t x[96], const float x_scale[96], const int32_t x_zero_point[96], float y[96] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 96; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_655_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_655_DequantizeLinear( const int8_t x[576][96][1][1], const float x_scale[576], const int8_t x_zero_point[576], float y[576][96][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_656_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_656_DequantizeLinear( const int32_t x[576], const float x_scale[576], const int32_t x_zero_point[576], float y[576] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_658_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_658_DequantizeLinear( const int8_t x[576][1][3][3], const float x_scale[576], const int8_t x_zero_point[576], float y[576][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_659_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_659_DequantizeLinear( const int32_t x[576], const float x_scale[576], const int32_t x_zero_point[576], float y[576] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 576; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_661_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_661_DequantizeLinear( const int8_t x[160][576][1][1], const float x_scale[160], const int8_t x_zero_point[160], float y[160][576][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 160; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_662_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_662_DequantizeLinear( const int32_t x[160], const float x_scale[160], const int32_t x_zero_point[160], float y[160] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 160; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_664_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_664_DequantizeLinear( const int8_t x[960][160][1][1], const float x_scale[960], const int8_t x_zero_point[960], float y[960][160][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_665_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_665_DequantizeLinear( const int32_t x[960], const float x_scale[960], const int32_t x_zero_point[960], float y[960] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_667_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_667_DequantizeLinear( const int8_t x[960][1][3][3], const float x_scale[960], const int8_t x_zero_point[960], float y[960][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_668_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_668_DequantizeLinear( const int32_t x[960], const float x_scale[960], const int32_t x_zero_point[960], float y[960] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_670_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_670_DequantizeLinear( const int8_t x[160][960][1][1], const float x_scale[160], const int8_t x_zero_point[160], float y[160][960][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 160; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_671_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_671_DequantizeLinear( const int32_t x[160], const float x_scale[160], const int32_t x_zero_point[160], float y[160] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 160; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_673_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_673_DequantizeLinear( const int8_t x[960][160][1][1], const float x_scale[960], const int8_t x_zero_point[960], float y[960][160][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_674_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_674_DequantizeLinear( const int32_t x[960], const float x_scale[960], const int32_t x_zero_point[960], float y[960] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_676_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_676_DequantizeLinear( const int8_t x[960][1][3][3], const float x_scale[960], const int8_t x_zero_point[960], float y[960][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_677_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_677_DequantizeLinear( const int32_t x[960], const float x_scale[960], const int32_t x_zero_point[960], float y[960] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_679_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_679_DequantizeLinear( const int8_t x[160][960][1][1], const float x_scale[160], const int8_t x_zero_point[160], float y[160][960][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 160; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_680_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_680_DequantizeLinear( const int32_t x[160], const float x_scale[160], const int32_t x_zero_point[160], float y[160] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 160; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_682_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_682_DequantizeLinear( const int8_t x[960][160][1][1], const float x_scale[960], const int8_t x_zero_point[960], float y[960][160][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_683_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_683_DequantizeLinear( const int32_t x[960], const float x_scale[960], const int32_t x_zero_point[960], float y[960] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_685_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_685_DequantizeLinear( const int8_t x[960][1][3][3], const float x_scale[960], const int8_t x_zero_point[960], float y[960][1][3][3] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	for (unsigned i1 = 0; i1 < 1; i1++)
	for (unsigned i2 = 0; i2 < 3; i2++)
	for (unsigned i3 = 0; i3 < 3; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_686_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_686_DequantizeLinear( const int32_t x[960], const float x_scale[960], const int32_t x_zero_point[960], float y[960] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 960; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_688_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_688_DequantizeLinear( const int8_t x[320][960][1][1], const float x_scale[320], const int8_t x_zero_point[320], float y[320][960][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 320; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_689_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_689_DequantizeLinear( const int32_t x[320], const float x_scale[320], const int32_t x_zero_point[320], float y[320] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 320; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_691_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_691_DequantizeLinear( const int8_t x[1280][320][1][1], const float x_scale[1280], const int8_t x_zero_point[1280], float y[1280][320][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1280; i0++)
	for (unsigned i1 = 0; i1 < 320; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: onnx::Conv_692_DequantizeLinear
 */
FUNC_PREFIX void node_onnx__Conv_692_DequantizeLinear( const int32_t x[1280], const float x_scale[1280], const int32_t x_zero_point[1280], float y[1280] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1280; i0++)
	{
		y[i0] = (x[i0] - x_zero_point[i0]) * x_scale[i0];
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: input_DequantizeLinear
 */
FUNC_PREFIX void node_input_DequantizeLinear( const uint8_t x[1][3][224][224], const float *x_scale, const uint8_t *x_zero_point, float y[1][3][224][224] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 3; i1++)
	for (unsigned i2 = 0; i2 < 224; i2++)
	for (unsigned i3 = 0; i3 < 224; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.0/features.0.0/Conv
 */
FUNC_PREFIX void node__features_features_0_features_0_0_Conv( const float x[1][3][224][224], const float w[32][3][3][3], const float bias[32], float y[1][32][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=-1; o0<112; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<112; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<3; c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.0/features.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_0_features_0_2_Clip_output_0_QuantizeLinear( const float x[1][32][112][112], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][32][112][112] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.0/features.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_0_features_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][32][112][112], const float *x_scale, const uint8_t *x_zero_point, float y[1][32][112][112] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.1/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_1_conv_conv_0_conv_0_0_Conv( const float x[1][32][112][112], const float w[32][1][3][3], const float bias[32], float y[1][32][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 32
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<32; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<112; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<112; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 112 && ii1 >= 0 && ii1 < 112 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.1/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][32][112][112], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][32][112][112] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.1/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][32][112][112], const float *x_scale, const uint8_t *x_zero_point, float y[1][32][112][112] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.1/conv/conv.1/Conv
 */
FUNC_PREFIX void node__features_features_1_conv_conv_1_Conv( const float x[1][32][112][112], const float w[16][32][1][1], const float bias[16], float y[1][16][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<16; m++) {
		for( int32_t o0=0, i0=0; o0<112; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<112; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.1/conv/conv.1/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_1_conv_conv_1_Conv_output_0_QuantizeLinear( const float x[1][16][112][112], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][16][112][112] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 16; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.1/conv/conv.1/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_1_conv_conv_1_Conv_output_0_DequantizeLinear( const uint8_t x[1][16][112][112], const float *x_scale, const uint8_t *x_zero_point, float y[1][16][112][112] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 16; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.2/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_2_conv_conv_0_conv_0_0_Conv( const float x[1][16][112][112], const float w[96][16][1][1], const float bias[96], float y[1][96][112][112] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<112; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<112; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<16; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.2/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][96][112][112], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][96][112][112] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.2/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][96][112][112], const float *x_scale, const uint8_t *x_zero_point, float y[1][96][112][112] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 112; i2++)
	for (unsigned i3 = 0; i3 < 112; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.2/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_2_conv_conv_1_conv_1_0_Conv( const float x[1][96][112][112], const float w[96][1][3][3], const float bias[96], float y[1][96][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 96
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<96; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<56; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<56; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.2/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][96][56][56], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][96][56][56] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.2/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][96][56][56], const float *x_scale, const uint8_t *x_zero_point, float y[1][96][56][56] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.2/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_2_conv_conv_2_Conv( const float x[1][96][56][56], const float w[24][96][1][1], const float bias[24], float y[1][24][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<24; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.2/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_2_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][24][56][56], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][24][56][56] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.2/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_2_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][24][56][56], const float *x_scale, const uint8_t *x_zero_point, float y[1][24][56][56] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.3/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_3_conv_conv_0_conv_0_0_Conv( const float x[1][24][56][56], const float w[144][24][1][1], const float bias[144], float y[1][144][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<144; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<24; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.3/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][144][56][56], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][144][56][56] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.3/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][144][56][56], const float *x_scale, const uint8_t *x_zero_point, float y[1][144][56][56] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.3/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_3_conv_conv_1_conv_1_0_Conv( const float x[1][144][56][56], const float w[144][1][3][3], const float bias[144], float y[1][144][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 144
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<144; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 56 && ii1 >= 0 && ii1 < 56 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.3/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][144][56][56], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][144][56][56] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.3/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][144][56][56], const float *x_scale, const uint8_t *x_zero_point, float y[1][144][56][56] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.3/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_3_conv_conv_2_Conv( const float x[1][144][56][56], const float w[24][144][1][1], const float bias[24], float y[1][24][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<24; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<144; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.3/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_3_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][24][56][56], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][24][56][56] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.3/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_3_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][24][56][56], const float *x_scale, const uint8_t *x_zero_point, float y[1][24][56][56] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.3/Add
 */
FUNC_PREFIX void node__features_features_3_Add( const float A[1][24][56][56], const float B[1][24][56][56], float C[1][24][56][56] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<24; i1++)
	for (unsigned i2=0; i2<56; i2++)
	for (unsigned i3=0; i3<56; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.3/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_3_Add_output_0_QuantizeLinear( const float x[1][24][56][56], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][24][56][56] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.3/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_3_Add_output_0_DequantizeLinear( const uint8_t x[1][24][56][56], const float *x_scale, const uint8_t *x_zero_point, float y[1][24][56][56] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 24; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.4/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_4_conv_conv_0_conv_0_0_Conv( const float x[1][24][56][56], const float w[144][24][1][1], const float bias[144], float y[1][144][56][56] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<144; m++) {
		for( int32_t o0=0, i0=0; o0<56; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<56; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<24; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.4/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][144][56][56], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][144][56][56] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.4/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][144][56][56], const float *x_scale, const uint8_t *x_zero_point, float y[1][144][56][56] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 56; i2++)
	for (unsigned i3 = 0; i3 < 56; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.4/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_4_conv_conv_1_conv_1_0_Conv( const float x[1][144][56][56], const float w[144][1][3][3], const float bias[144], float y[1][144][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 144
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<144; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<28; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<28; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.4/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][144][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][144][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.4/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][144][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][144][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 144; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.4/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_4_conv_conv_2_Conv( const float x[1][144][28][28], const float w[32][144][1][1], const float bias[32], float y[1][32][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<144; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.4/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_4_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][32][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][32][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.4/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_4_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][32][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][32][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.5/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_5_conv_conv_0_conv_0_0_Conv( const float x[1][32][28][28], const float w[192][32][1][1], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.5/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][192][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][192][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.5/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][192][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][192][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.5/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_5_conv_conv_1_conv_1_0_Conv( const float x[1][192][28][28], const float w[192][1][3][3], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 192
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<192; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 28 && ii1 >= 0 && ii1 < 28 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.5/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][192][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][192][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.5/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][192][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][192][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.5/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_5_conv_conv_2_Conv( const float x[1][192][28][28], const float w[32][192][1][1], const float bias[32], float y[1][32][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.5/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_5_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][32][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][32][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.5/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_5_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][32][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][32][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.5/Add
 */
FUNC_PREFIX void node__features_features_5_Add( const float A[1][32][28][28], const float B[1][32][28][28], float C[1][32][28][28] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<32; i1++)
	for (unsigned i2=0; i2<28; i2++)
	for (unsigned i3=0; i3<28; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.5/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_5_Add_output_0_QuantizeLinear( const float x[1][32][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][32][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.5/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_5_Add_output_0_DequantizeLinear( const uint8_t x[1][32][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][32][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.6/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_6_conv_conv_0_conv_0_0_Conv( const float x[1][32][28][28], const float w[192][32][1][1], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.6/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][192][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][192][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.6/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][192][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][192][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.6/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_6_conv_conv_1_conv_1_0_Conv( const float x[1][192][28][28], const float w[192][1][3][3], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 192
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<192; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 28 && ii1 >= 0 && ii1 < 28 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.6/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][192][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][192][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.6/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][192][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][192][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.6/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_6_conv_conv_2_Conv( const float x[1][192][28][28], const float w[32][192][1][1], const float bias[32], float y[1][32][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<32; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.6/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_6_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][32][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][32][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.6/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_6_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][32][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][32][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.6/Add
 */
FUNC_PREFIX void node__features_features_6_Add( const float A[1][32][28][28], const float B[1][32][28][28], float C[1][32][28][28] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<32; i1++)
	for (unsigned i2=0; i2<28; i2++)
	for (unsigned i3=0; i3<28; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.6/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_6_Add_output_0_QuantizeLinear( const float x[1][32][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][32][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.6/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_6_Add_output_0_DequantizeLinear( const uint8_t x[1][32][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][32][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 32; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.7/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_7_conv_conv_0_conv_0_0_Conv( const float x[1][32][28][28], const float w[192][32][1][1], const float bias[192], float y[1][192][28][28] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<192; m++) {
		for( int32_t o0=0, i0=0; o0<28; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<28; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<32; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.7/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][192][28][28], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][192][28][28] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.7/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][192][28][28], const float *x_scale, const uint8_t *x_zero_point, float y[1][192][28][28] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 28; i2++)
	for (unsigned i3 = 0; i3 < 28; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.7/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_7_conv_conv_1_conv_1_0_Conv( const float x[1][192][28][28], const float w[192][1][3][3], const float bias[192], float y[1][192][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 192
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<192; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.7/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][192][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][192][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.7/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][192][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][192][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 192; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.7/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_7_conv_conv_2_Conv( const float x[1][192][14][14], const float w[64][192][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<192; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.7/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_7_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][64][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][64][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.7/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_7_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][64][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][64][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.8/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_8_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.8/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.8/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.8/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_8_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.8/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.8/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.8/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_8_conv_conv_2_Conv( const float x[1][384][14][14], const float w[64][384][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.8/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_8_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][64][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][64][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.8/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_8_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][64][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][64][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.8/Add
 */
FUNC_PREFIX void node__features_features_8_Add( const float A[1][64][14][14], const float B[1][64][14][14], float C[1][64][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<64; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.8/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_8_Add_output_0_QuantizeLinear( const float x[1][64][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][64][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.8/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_8_Add_output_0_DequantizeLinear( const uint8_t x[1][64][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][64][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.9/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_9_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.9/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.9/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.9/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_9_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.9/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.9/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.9/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_9_conv_conv_2_Conv( const float x[1][384][14][14], const float w[64][384][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.9/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_9_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][64][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][64][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.9/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_9_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][64][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][64][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.9/Add
 */
FUNC_PREFIX void node__features_features_9_Add( const float A[1][64][14][14], const float B[1][64][14][14], float C[1][64][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<64; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.9/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_9_Add_output_0_QuantizeLinear( const float x[1][64][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][64][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.9/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_9_Add_output_0_DequantizeLinear( const uint8_t x[1][64][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][64][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.10/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_10_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.10/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.10/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.10/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_10_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.10/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.10/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.10/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_10_conv_conv_2_Conv( const float x[1][384][14][14], const float w[64][384][1][1], const float bias[64], float y[1][64][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<64; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.10/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_10_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][64][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][64][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.10/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_10_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][64][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][64][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.10/Add
 */
FUNC_PREFIX void node__features_features_10_Add( const float A[1][64][14][14], const float B[1][64][14][14], float C[1][64][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<64; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.10/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_10_Add_output_0_QuantizeLinear( const float x[1][64][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][64][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.10/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_10_Add_output_0_DequantizeLinear( const uint8_t x[1][64][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][64][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 64; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.11/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_11_conv_conv_0_conv_0_0_Conv( const float x[1][64][14][14], const float w[384][64][1][1], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<384; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<64; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.11/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.11/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.11/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_11_conv_conv_1_conv_1_0_Conv( const float x[1][384][14][14], const float w[384][1][3][3], const float bias[384], float y[1][384][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 384
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<384; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.11/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][384][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][384][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.11/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][384][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][384][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 384; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.11/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_11_conv_conv_2_Conv( const float x[1][384][14][14], const float w[96][384][1][1], const float bias[96], float y[1][96][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<384; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.11/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_11_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][96][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][96][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.11/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_11_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][96][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][96][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.12/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_12_conv_conv_0_conv_0_0_Conv( const float x[1][96][14][14], const float w[576][96][1][1], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<576; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.12/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][576][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][576][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.12/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][576][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][576][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.12/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_12_conv_conv_1_conv_1_0_Conv( const float x[1][576][14][14], const float w[576][1][3][3], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 576
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<576; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.12/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][576][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][576][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.12/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][576][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][576][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.12/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_12_conv_conv_2_Conv( const float x[1][576][14][14], const float w[96][576][1][1], const float bias[96], float y[1][96][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<576; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.12/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_12_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][96][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][96][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.12/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_12_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][96][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][96][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.12/Add
 */
FUNC_PREFIX void node__features_features_12_Add( const float A[1][96][14][14], const float B[1][96][14][14], float C[1][96][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<96; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.12/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_12_Add_output_0_QuantizeLinear( const float x[1][96][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][96][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.12/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_12_Add_output_0_DequantizeLinear( const uint8_t x[1][96][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][96][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.13/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_13_conv_conv_0_conv_0_0_Conv( const float x[1][96][14][14], const float w[576][96][1][1], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<576; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.13/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][576][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][576][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.13/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][576][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][576][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.13/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_13_conv_conv_1_conv_1_0_Conv( const float x[1][576][14][14], const float w[576][1][3][3], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 576
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<576; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 14 && ii1 >= 0 && ii1 < 14 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.13/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][576][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][576][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.13/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][576][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][576][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.13/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_13_conv_conv_2_Conv( const float x[1][576][14][14], const float w[96][576][1][1], const float bias[96], float y[1][96][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<96; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<576; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.13/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_13_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][96][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][96][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.13/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_13_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][96][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][96][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.13/Add
 */
FUNC_PREFIX void node__features_features_13_Add( const float A[1][96][14][14], const float B[1][96][14][14], float C[1][96][14][14] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<96; i1++)
	for (unsigned i2=0; i2<14; i2++)
	for (unsigned i3=0; i3<14; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.13/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_13_Add_output_0_QuantizeLinear( const float x[1][96][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][96][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.13/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_13_Add_output_0_DequantizeLinear( const uint8_t x[1][96][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][96][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 96; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.14/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_14_conv_conv_0_conv_0_0_Conv( const float x[1][96][14][14], const float w[576][96][1][1], const float bias[576], float y[1][576][14][14] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<576; m++) {
		for( int32_t o0=0, i0=0; o0<14; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<14; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<96; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.14/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][576][14][14], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][576][14][14] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.14/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][576][14][14], const float *x_scale, const uint8_t *x_zero_point, float y[1][576][14][14] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 14; i2++)
	for (unsigned i3 = 0; i3 < 14; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.14/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_14_conv_conv_1_conv_1_0_Conv( const float x[1][576][14][14], const float w[576][1][3][3], const float bias[576], float y[1][576][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 576
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 2 2 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<576; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=2) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=2) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii1 >= 0 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.14/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][576][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][576][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.14/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][576][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][576][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 576; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.14/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_14_conv_conv_2_Conv( const float x[1][576][7][7], const float w[160][576][1][1], const float bias[160], float y[1][160][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<160; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<576; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.14/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_14_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][160][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][160][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.14/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_14_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][160][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][160][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.15/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_15_conv_conv_0_conv_0_0_Conv( const float x[1][160][7][7], const float w[960][160][1][1], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<960; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<160; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.15/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][960][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][960][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.15/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][960][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][960][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.15/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_15_conv_conv_1_conv_1_0_Conv( const float x[1][960][7][7], const float w[960][1][3][3], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 960
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<960; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 7 && ii1 >= 0 && ii1 < 7 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.15/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][960][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][960][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.15/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][960][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][960][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.15/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_15_conv_conv_2_Conv( const float x[1][960][7][7], const float w[160][960][1][1], const float bias[160], float y[1][160][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<160; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<960; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.15/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_15_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][160][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][160][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.15/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_15_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][160][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][160][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.15/Add
 */
FUNC_PREFIX void node__features_features_15_Add( const float A[1][160][7][7], const float B[1][160][7][7], float C[1][160][7][7] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<160; i1++)
	for (unsigned i2=0; i2<7; i2++)
	for (unsigned i3=0; i3<7; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.15/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_15_Add_output_0_QuantizeLinear( const float x[1][160][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][160][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.15/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_15_Add_output_0_DequantizeLinear( const uint8_t x[1][160][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][160][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.16/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_16_conv_conv_0_conv_0_0_Conv( const float x[1][160][7][7], const float w[960][160][1][1], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<960; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<160; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.16/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][960][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][960][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.16/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][960][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][960][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.16/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_16_conv_conv_1_conv_1_0_Conv( const float x[1][960][7][7], const float w[960][1][3][3], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 960
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<960; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 7 && ii1 >= 0 && ii1 < 7 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.16/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][960][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][960][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.16/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][960][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][960][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.16/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_16_conv_conv_2_Conv( const float x[1][960][7][7], const float w[160][960][1][1], const float bias[160], float y[1][160][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<160; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<960; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.16/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_16_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][160][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][160][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.16/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_16_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][160][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][160][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Add
 * Name in ONNX file: /features/features.16/Add
 */
FUNC_PREFIX void node__features_features_16_Add( const float A[1][160][7][7], const float B[1][160][7][7], float C[1][160][7][7] )
{
	/* Add
	   Implemented with Elementwise_2 template.
	   Attributes (these are the union of attributes for all 2-element-wise
	               operands. So most likely these values are ignored by onnx2c).
	   shift_dir: NOT_GIVEN
	   fmod: 0
	 */
	for (unsigned i0=0; i0<1; i0++)
	for (unsigned i1=0; i1<160; i1++)
	for (unsigned i2=0; i2<7; i2++)
	for (unsigned i3=0; i3<7; i3++)
	{
		C[0][i1][i2][i3] = A[0][i1][i2][i3]+B[0][i1][i2][i3];;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.16/Add_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_16_Add_output_0_QuantizeLinear( const float x[1][160][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][160][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.16/Add_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_16_Add_output_0_DequantizeLinear( const uint8_t x[1][160][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][160][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 160; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.17/conv/conv.0/conv.0.0/Conv
 */
FUNC_PREFIX void node__features_features_17_conv_conv_0_conv_0_0_Conv( const float x[1][160][7][7], const float w[960][160][1][1], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<960; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<160; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.17/conv/conv.0/conv.0.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( const float x[1][960][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][960][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.17/conv/conv.0/conv.0.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][960][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][960][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.17/conv/conv.1/conv.1.0/Conv
 */
FUNC_PREFIX void node__features_features_17_conv_conv_1_conv_1_0_Conv( const float x[1][960][7][7], const float w[960][1][3][3], const float bias[960], float y[1][960][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 960
	 * kernel_shape: 3 3 
	 * pads: 1 1 1 1 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	uint32_t go = 1; // output group size, i.e. maps/group
	uint32_t gi = 1; // inptput group size, i.e. channels/group
	for( uint32_t g=0; g<960; g++) {
	for( uint32_t m=go*g; m<go*(g+1); m++) {
		for( int32_t o0=0, i0=-1; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=-1; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=gi*g; c<gi*(g+1); c++ ) {
			for( uint32_t k0=0; k0<3; k0++ ) {
			for( uint32_t k1=0; k1<3; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				if( ii0 >= 0 && ii0 < 7 && ii1 >= 0 && ii1 < 7 ) {
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c-(gi*g)][k0][k1];
				} /* if valid */
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
		} /* g */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.17/conv/conv.1/conv.1.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( const float x[1][960][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][960][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.17/conv/conv.1/conv.1.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][960][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][960][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 960; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.17/conv/conv.2/Conv
 */
FUNC_PREFIX void node__features_features_17_conv_conv_2_Conv( const float x[1][960][7][7], const float w[320][960][1][1], const float bias[320], float y[1][320][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<320; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<960; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.17/conv/conv.2/Conv_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_17_conv_conv_2_Conv_output_0_QuantizeLinear( const float x[1][320][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][320][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 320; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.17/conv/conv.2/Conv_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_17_conv_conv_2_Conv_output_0_DequantizeLinear( const uint8_t x[1][320][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][320][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 320; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Conv
 * Name in ONNX file: /features/features.18/features.18.0/Conv
 */
FUNC_PREFIX void node__features_features_18_features_18_0_Conv( const float x[1][320][7][7], const float w[1280][320][1][1], const float bias[1280], float y[1][1280][7][7] )
{
	/* Conv
	 *
	 * auto_pad: NOTSET
	 * dilations: 1 1 
	 * group: 1
	 * kernel_shape: 1 1 
	 * pads: 0 0 0 0 
	 * strides: 1 1 
	 */
	for( uint32_t b=0; b<1; b++ ) {
	for( uint32_t m=0; m<1280; m++) {
		for( int32_t o0=0, i0=0; o0<7; o0++, i0+=1) {
		for( int32_t o1=0, i1=0; o1<7; o1++, i1+=1) {
			y[b][m][o0][o1] = bias[m];
			for( int32_t c=0; c<320; c++ ) {
			for( uint32_t k0=0; k0<1; k0++ ) {
			for( uint32_t k1=0; k1<1; k1++ ) {
				int ii0 = i0+k0 * 1;
				int ii1 = i1+k1 * 1;
				y[b][m][o0][o1] += x[b][c][ii0][ii1] * w[m][c][k0][k1];
			} /* k */
			} /* k */
			} /* c */
		} /* o */
		} /* o */
	} /* m */
	} /* b */
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /features/features.18/features.18.2/Clip_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__features_features_18_features_18_2_Clip_output_0_QuantizeLinear( const float x[1][1280][7][7], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][1280][7][7] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1280; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /features/features.18/features.18.2/Clip_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__features_features_18_features_18_2_Clip_output_0_DequantizeLinear( const uint8_t x[1][1280][7][7], const float *x_scale, const uint8_t *x_zero_point, float y[1][1280][7][7] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1280; i1++)
	for (unsigned i2 = 0; i2 < 7; i2++)
	for (unsigned i3 = 0; i3 < 7; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           GlobalAveragePool
 * Name in ONNX file: /GlobalAveragePool
 */
FUNC_PREFIX void node__GlobalAveragePool( const float input[1][1280][7][7], float output[1][1280][1][1] )
{
	/* GlobalAveragePool */
	for( int32_t b=0; b<1; b++ ) {
	for( int32_t c=0; c<1280; c++ ) {
		float dimsum=0.0f;
		for( int32_t d0 = 0; d0<7; d0++ ) {
		for( int32_t d1 = 0; d1<7; d1++ ) {
			dimsum +=  input[b][c][d0][d1];
		}
		}
		output[b][c][0][0] = dimsum / 49;
	}
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /GlobalAveragePool_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__GlobalAveragePool_output_0_QuantizeLinear( const float x[1][1280][1][1], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][1280][1][1] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1280; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		int t = (int)roundf(x[i0][i1][i2][i3] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1][i2][i3] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /GlobalAveragePool_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__GlobalAveragePool_output_0_DequantizeLinear( const uint8_t x[1][1280][1][1], const float *x_scale, const uint8_t *x_zero_point, float y[1][1280][1][1] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1280; i1++)
	for (unsigned i2 = 0; i2 < 1; i2++)
	for (unsigned i3 = 0; i3 < 1; i3++)
	{
		y[i0][i1][i2][i3] = (x[i0][i1][i2][i3] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Flatten
 * Name in ONNX file: /Flatten
 */
FUNC_PREFIX void node__Flatten( const float input[1][1280][1][1], float output[1][1280] )
{
	/* Flatten*/
	float *input_ = (float*)input;
	float *output_ = (float*)output;
	for( uint32_t i=0; i<1280; i++ )
		output_[i] = input_[i];

}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: /Flatten_output_0_QuantizeLinear
 */
FUNC_PREFIX void node__Flatten_output_0_QuantizeLinear( const float x[1][1280], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][1280] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1280; i1++)
	{
		int t = (int)roundf(x[i0][i1] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: /Flatten_output_0_DequantizeLinear
 */
FUNC_PREFIX void node__Flatten_output_0_DequantizeLinear( const uint8_t x[1][1280], const float *x_scale, const uint8_t *x_zero_point, float y[1][1280] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1280; i1++)
	{
		y[i0][i1] = (x[i0][i1] - x_zero_point[0]) * x_scale[0];
	}
}

/*
 * Operand:           Gemm
 * Name in ONNX file: /classifier/classifier.1/Gemm
 */
FUNC_PREFIX void node__classifier_classifier_1_Gemm( const float A[1][1280], const float B[1000][1280], const float C[1000], float Y[1][1000] )
{
	/* Gemm */
	/* alpha   = 1.00000000000000000000
	   beta    = 1.00000000000000000000
	   transA  = 0
	   transB  = 1
	 */
	const int M = 1;
	const int K = 1280;
	const int N = 1000;
	float alpha = 1.00000000000000000000;
	float beta = 1.00000000000000000000;
	float (*C_)[1000]  = (float(*)[1000])C;
	for( uint32_t r=0; r<M; r++ )
		for( uint32_t c=0; c<N; c++ ) {
			float ABrc = 0;
			for( uint32_t i=0; i<K; i++ ) {
				float B_el = B[c][i];
				ABrc += A[r][i] * B_el;
			}
			float tmp = ABrc * alpha;
			tmp += C_[0][c] * beta;
			Y[r][c] = tmp;
	}
}

/*
 * Operand:           QuantizeLinear
 * Name in ONNX file: logits_QuantizeLinear
 */
FUNC_PREFIX void node_logits_QuantizeLinear( const float x[1][1000], const float *y_scale, const uint8_t *y_zero_point, uint8_t y[1][1000] )
{
	/* QuantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1000; i1++)
	{
		int t = (int)roundf(x[i0][i1] / y_scale[0]) + (int)y_zero_point[0];
		y[i0][i1] = (uint8_t)MIN(MAX(t, 0), UINT8_MAX);
	}
}

/*
 * Operand:           DequantizeLinear
 * Name in ONNX file: logits_DequantizeLinear
 */
FUNC_PREFIX void node_logits_DequantizeLinear( const uint8_t x[1][1000], const float *x_scale, const uint8_t *x_zero_point, float y[1][1000] )
{
	/* DequantizeLinear */
	for (unsigned i0 = 0; i0 < 1; i0++)
	for (unsigned i1 = 0; i1 < 1000; i1++)
	{
		y[i0][i1] = (x[i0][i1] - x_zero_point[0]) * x_scale[0];
	}
}


void entry(const float tensor_input[1][3][224][224], float tensor_logits[1][1000]){
	node_classifier_1_bias_DequantizeLinear( tensor_classifier_1_bias_quantized, tensor_classifier_1_bias_quantized_scale, tensor_classifier_1_bias_quantized_zero_point, tu0.tensor_classifier_1_bias);
	node_classifier_1_weight_DequantizeLinear( tensor_classifier_1_weight_quantized, tensor_classifier_1_weight_scale, tensor_classifier_1_weight_zero_point, tu1.tensor_classifier_1_weight_DequantizeLinear_Output);
	node_input_QuantizeLinear( tensor_input, &tensor_input_scale, &tensor_input_zero_point, tu2.tensor_input_QuantizeLinear_Output);
	node_onnx__Conv_538_DequantizeLinear( tensor_onnx__Conv_538_quantized, tensor_onnx__Conv_538_scale, tensor_onnx__Conv_538_zero_point, tu3.tensor_onnx__Conv_538_DequantizeLinear_Output);
	node_onnx__Conv_539_DequantizeLinear( tensor_onnx__Conv_539_quantized, tensor_onnx__Conv_539_quantized_scale, tensor_onnx__Conv_539_quantized_zero_point, tu4.tensor_onnx__Conv_539);
	node_onnx__Conv_541_DequantizeLinear( tensor_onnx__Conv_541_quantized, tensor_onnx__Conv_541_scale, tensor_onnx__Conv_541_zero_point, tu5.tensor_onnx__Conv_541_DequantizeLinear_Output);
	node_onnx__Conv_542_DequantizeLinear( tensor_onnx__Conv_542_quantized, tensor_onnx__Conv_542_quantized_scale, tensor_onnx__Conv_542_quantized_zero_point, tu6.tensor_onnx__Conv_542);
	node_onnx__Conv_544_DequantizeLinear( tensor_onnx__Conv_544_quantized, tensor_onnx__Conv_544_scale, tensor_onnx__Conv_544_zero_point, tu7.tensor_onnx__Conv_544_DequantizeLinear_Output);
	node_onnx__Conv_545_DequantizeLinear( tensor_onnx__Conv_545_quantized, tensor_onnx__Conv_545_quantized_scale, tensor_onnx__Conv_545_quantized_zero_point, tu8.tensor_onnx__Conv_545);
	node_onnx__Conv_547_DequantizeLinear( tensor_onnx__Conv_547_quantized, tensor_onnx__Conv_547_scale, tensor_onnx__Conv_547_zero_point, tu9.tensor_onnx__Conv_547_DequantizeLinear_Output);
	node_onnx__Conv_548_DequantizeLinear( tensor_onnx__Conv_548_quantized, tensor_onnx__Conv_548_quantized_scale, tensor_onnx__Conv_548_quantized_zero_point, tu10.tensor_onnx__Conv_548);
	node_onnx__Conv_550_DequantizeLinear( tensor_onnx__Conv_550_quantized, tensor_onnx__Conv_550_scale, tensor_onnx__Conv_550_zero_point, tu11.tensor_onnx__Conv_550_DequantizeLinear_Output);
	node_onnx__Conv_551_DequantizeLinear( tensor_onnx__Conv_551_quantized, tensor_onnx__Conv_551_quantized_scale, tensor_onnx__Conv_551_quantized_zero_point, tu12.tensor_onnx__Conv_551);
	node_onnx__Conv_553_DequantizeLinear( tensor_onnx__Conv_553_quantized, tensor_onnx__Conv_553_scale, tensor_onnx__Conv_553_zero_point, tu13.tensor_onnx__Conv_553_DequantizeLinear_Output);
	node_onnx__Conv_554_DequantizeLinear( tensor_onnx__Conv_554_quantized, tensor_onnx__Conv_554_quantized_scale, tensor_onnx__Conv_554_quantized_zero_point, tu14.tensor_onnx__Conv_554);
	node_onnx__Conv_556_DequantizeLinear( tensor_onnx__Conv_556_quantized, tensor_onnx__Conv_556_scale, tensor_onnx__Conv_556_zero_point, tu15.tensor_onnx__Conv_556_DequantizeLinear_Output);
	node_onnx__Conv_557_DequantizeLinear( tensor_onnx__Conv_557_quantized, tensor_onnx__Conv_557_quantized_scale, tensor_onnx__Conv_557_quantized_zero_point, tu16.tensor_onnx__Conv_557);
	node_onnx__Conv_559_DequantizeLinear( tensor_onnx__Conv_559_quantized, tensor_onnx__Conv_559_scale, tensor_onnx__Conv_559_zero_point, tu17.tensor_onnx__Conv_559_DequantizeLinear_Output);
	node_onnx__Conv_560_DequantizeLinear( tensor_onnx__Conv_560_quantized, tensor_onnx__Conv_560_quantized_scale, tensor_onnx__Conv_560_quantized_zero_point, tu18.tensor_onnx__Conv_560);
	node_onnx__Conv_562_DequantizeLinear( tensor_onnx__Conv_562_quantized, tensor_onnx__Conv_562_scale, tensor_onnx__Conv_562_zero_point, tu19.tensor_onnx__Conv_562_DequantizeLinear_Output);
	node_onnx__Conv_563_DequantizeLinear( tensor_onnx__Conv_563_quantized, tensor_onnx__Conv_563_quantized_scale, tensor_onnx__Conv_563_quantized_zero_point, tu20.tensor_onnx__Conv_563);
	node_onnx__Conv_565_DequantizeLinear( tensor_onnx__Conv_565_quantized, tensor_onnx__Conv_565_scale, tensor_onnx__Conv_565_zero_point, tu21.tensor_onnx__Conv_565_DequantizeLinear_Output);
	node_onnx__Conv_566_DequantizeLinear( tensor_onnx__Conv_566_quantized, tensor_onnx__Conv_566_quantized_scale, tensor_onnx__Conv_566_quantized_zero_point, tu22.tensor_onnx__Conv_566);
	node_onnx__Conv_568_DequantizeLinear( tensor_onnx__Conv_568_quantized, tensor_onnx__Conv_568_scale, tensor_onnx__Conv_568_zero_point, tu23.tensor_onnx__Conv_568_DequantizeLinear_Output);
	node_onnx__Conv_569_DequantizeLinear( tensor_onnx__Conv_569_quantized, tensor_onnx__Conv_569_quantized_scale, tensor_onnx__Conv_569_quantized_zero_point, tu24.tensor_onnx__Conv_569);
	node_onnx__Conv_571_DequantizeLinear( tensor_onnx__Conv_571_quantized, tensor_onnx__Conv_571_scale, tensor_onnx__Conv_571_zero_point, tu25.tensor_onnx__Conv_571_DequantizeLinear_Output);
	node_onnx__Conv_572_DequantizeLinear( tensor_onnx__Conv_572_quantized, tensor_onnx__Conv_572_quantized_scale, tensor_onnx__Conv_572_quantized_zero_point, tu26.tensor_onnx__Conv_572);
	node_onnx__Conv_574_DequantizeLinear( tensor_onnx__Conv_574_quantized, tensor_onnx__Conv_574_scale, tensor_onnx__Conv_574_zero_point, tu27.tensor_onnx__Conv_574_DequantizeLinear_Output);
	node_onnx__Conv_575_DequantizeLinear( tensor_onnx__Conv_575_quantized, tensor_onnx__Conv_575_quantized_scale, tensor_onnx__Conv_575_quantized_zero_point, tu28.tensor_onnx__Conv_575);
	node_onnx__Conv_577_DequantizeLinear( tensor_onnx__Conv_577_quantized, tensor_onnx__Conv_577_scale, tensor_onnx__Conv_577_zero_point, tu29.tensor_onnx__Conv_577_DequantizeLinear_Output);
	node_onnx__Conv_578_DequantizeLinear( tensor_onnx__Conv_578_quantized, tensor_onnx__Conv_578_quantized_scale, tensor_onnx__Conv_578_quantized_zero_point, tu30.tensor_onnx__Conv_578);
	node_onnx__Conv_580_DequantizeLinear( tensor_onnx__Conv_580_quantized, tensor_onnx__Conv_580_scale, tensor_onnx__Conv_580_zero_point, tu31.tensor_onnx__Conv_580_DequantizeLinear_Output);
	node_onnx__Conv_581_DequantizeLinear( tensor_onnx__Conv_581_quantized, tensor_onnx__Conv_581_quantized_scale, tensor_onnx__Conv_581_quantized_zero_point, tu32.tensor_onnx__Conv_581);
	node_onnx__Conv_583_DequantizeLinear( tensor_onnx__Conv_583_quantized, tensor_onnx__Conv_583_scale, tensor_onnx__Conv_583_zero_point, tu33.tensor_onnx__Conv_583_DequantizeLinear_Output);
	node_onnx__Conv_584_DequantizeLinear( tensor_onnx__Conv_584_quantized, tensor_onnx__Conv_584_quantized_scale, tensor_onnx__Conv_584_quantized_zero_point, tu34.tensor_onnx__Conv_584);
	node_onnx__Conv_586_DequantizeLinear( tensor_onnx__Conv_586_quantized, tensor_onnx__Conv_586_scale, tensor_onnx__Conv_586_zero_point, tu35.tensor_onnx__Conv_586_DequantizeLinear_Output);
	node_onnx__Conv_587_DequantizeLinear( tensor_onnx__Conv_587_quantized, tensor_onnx__Conv_587_quantized_scale, tensor_onnx__Conv_587_quantized_zero_point, tu36.tensor_onnx__Conv_587);
	node_onnx__Conv_589_DequantizeLinear( tensor_onnx__Conv_589_quantized, tensor_onnx__Conv_589_scale, tensor_onnx__Conv_589_zero_point, tu37.tensor_onnx__Conv_589_DequantizeLinear_Output);
	node_onnx__Conv_590_DequantizeLinear( tensor_onnx__Conv_590_quantized, tensor_onnx__Conv_590_quantized_scale, tensor_onnx__Conv_590_quantized_zero_point, tu38.tensor_onnx__Conv_590);
	node_onnx__Conv_592_DequantizeLinear( tensor_onnx__Conv_592_quantized, tensor_onnx__Conv_592_scale, tensor_onnx__Conv_592_zero_point, tu39.tensor_onnx__Conv_592_DequantizeLinear_Output);
	node_onnx__Conv_593_DequantizeLinear( tensor_onnx__Conv_593_quantized, tensor_onnx__Conv_593_quantized_scale, tensor_onnx__Conv_593_quantized_zero_point, tu40.tensor_onnx__Conv_593);
	node_onnx__Conv_595_DequantizeLinear( tensor_onnx__Conv_595_quantized, tensor_onnx__Conv_595_scale, tensor_onnx__Conv_595_zero_point, tu41.tensor_onnx__Conv_595_DequantizeLinear_Output);
	node_onnx__Conv_596_DequantizeLinear( tensor_onnx__Conv_596_quantized, tensor_onnx__Conv_596_quantized_scale, tensor_onnx__Conv_596_quantized_zero_point, tu42.tensor_onnx__Conv_596);
	node_onnx__Conv_598_DequantizeLinear( tensor_onnx__Conv_598_quantized, tensor_onnx__Conv_598_scale, tensor_onnx__Conv_598_zero_point, tu43.tensor_onnx__Conv_598_DequantizeLinear_Output);
	node_onnx__Conv_599_DequantizeLinear( tensor_onnx__Conv_599_quantized, tensor_onnx__Conv_599_quantized_scale, tensor_onnx__Conv_599_quantized_zero_point, tu44.tensor_onnx__Conv_599);
	node_onnx__Conv_601_DequantizeLinear( tensor_onnx__Conv_601_quantized, tensor_onnx__Conv_601_scale, tensor_onnx__Conv_601_zero_point, tu45.tensor_onnx__Conv_601_DequantizeLinear_Output);
	node_onnx__Conv_602_DequantizeLinear( tensor_onnx__Conv_602_quantized, tensor_onnx__Conv_602_quantized_scale, tensor_onnx__Conv_602_quantized_zero_point, tu46.tensor_onnx__Conv_602);
	node_onnx__Conv_604_DequantizeLinear( tensor_onnx__Conv_604_quantized, tensor_onnx__Conv_604_scale, tensor_onnx__Conv_604_zero_point, tu47.tensor_onnx__Conv_604_DequantizeLinear_Output);
	node_onnx__Conv_605_DequantizeLinear( tensor_onnx__Conv_605_quantized, tensor_onnx__Conv_605_quantized_scale, tensor_onnx__Conv_605_quantized_zero_point, tu48.tensor_onnx__Conv_605);
	node_onnx__Conv_607_DequantizeLinear( tensor_onnx__Conv_607_quantized, tensor_onnx__Conv_607_scale, tensor_onnx__Conv_607_zero_point, tu49.tensor_onnx__Conv_607_DequantizeLinear_Output);
	node_onnx__Conv_608_DequantizeLinear( tensor_onnx__Conv_608_quantized, tensor_onnx__Conv_608_quantized_scale, tensor_onnx__Conv_608_quantized_zero_point, tu50.tensor_onnx__Conv_608);
	node_onnx__Conv_610_DequantizeLinear( tensor_onnx__Conv_610_quantized, tensor_onnx__Conv_610_scale, tensor_onnx__Conv_610_zero_point, tu51.tensor_onnx__Conv_610_DequantizeLinear_Output);
	node_onnx__Conv_611_DequantizeLinear( tensor_onnx__Conv_611_quantized, tensor_onnx__Conv_611_quantized_scale, tensor_onnx__Conv_611_quantized_zero_point, tu52.tensor_onnx__Conv_611);
	node_onnx__Conv_613_DequantizeLinear( tensor_onnx__Conv_613_quantized, tensor_onnx__Conv_613_scale, tensor_onnx__Conv_613_zero_point, tu53.tensor_onnx__Conv_613_DequantizeLinear_Output);
	node_onnx__Conv_614_DequantizeLinear( tensor_onnx__Conv_614_quantized, tensor_onnx__Conv_614_quantized_scale, tensor_onnx__Conv_614_quantized_zero_point, tu54.tensor_onnx__Conv_614);
	node_onnx__Conv_616_DequantizeLinear( tensor_onnx__Conv_616_quantized, tensor_onnx__Conv_616_scale, tensor_onnx__Conv_616_zero_point, tu55.tensor_onnx__Conv_616_DequantizeLinear_Output);
	node_onnx__Conv_617_DequantizeLinear( tensor_onnx__Conv_617_quantized, tensor_onnx__Conv_617_quantized_scale, tensor_onnx__Conv_617_quantized_zero_point, tu56.tensor_onnx__Conv_617);
	node_onnx__Conv_619_DequantizeLinear( tensor_onnx__Conv_619_quantized, tensor_onnx__Conv_619_scale, tensor_onnx__Conv_619_zero_point, tu57.tensor_onnx__Conv_619_DequantizeLinear_Output);
	node_onnx__Conv_620_DequantizeLinear( tensor_onnx__Conv_620_quantized, tensor_onnx__Conv_620_quantized_scale, tensor_onnx__Conv_620_quantized_zero_point, tu58.tensor_onnx__Conv_620);
	node_onnx__Conv_622_DequantizeLinear( tensor_onnx__Conv_622_quantized, tensor_onnx__Conv_622_scale, tensor_onnx__Conv_622_zero_point, tu59.tensor_onnx__Conv_622_DequantizeLinear_Output);
	node_onnx__Conv_623_DequantizeLinear( tensor_onnx__Conv_623_quantized, tensor_onnx__Conv_623_quantized_scale, tensor_onnx__Conv_623_quantized_zero_point, tu60.tensor_onnx__Conv_623);
	node_onnx__Conv_625_DequantizeLinear( tensor_onnx__Conv_625_quantized, tensor_onnx__Conv_625_scale, tensor_onnx__Conv_625_zero_point, tu61.tensor_onnx__Conv_625_DequantizeLinear_Output);
	node_onnx__Conv_626_DequantizeLinear( tensor_onnx__Conv_626_quantized, tensor_onnx__Conv_626_quantized_scale, tensor_onnx__Conv_626_quantized_zero_point, tu62.tensor_onnx__Conv_626);
	node_onnx__Conv_628_DequantizeLinear( tensor_onnx__Conv_628_quantized, tensor_onnx__Conv_628_scale, tensor_onnx__Conv_628_zero_point, tu63.tensor_onnx__Conv_628_DequantizeLinear_Output);
	node_onnx__Conv_629_DequantizeLinear( tensor_onnx__Conv_629_quantized, tensor_onnx__Conv_629_quantized_scale, tensor_onnx__Conv_629_quantized_zero_point, tu64.tensor_onnx__Conv_629);
	node_onnx__Conv_631_DequantizeLinear( tensor_onnx__Conv_631_quantized, tensor_onnx__Conv_631_scale, tensor_onnx__Conv_631_zero_point, tu65.tensor_onnx__Conv_631_DequantizeLinear_Output);
	node_onnx__Conv_632_DequantizeLinear( tensor_onnx__Conv_632_quantized, tensor_onnx__Conv_632_quantized_scale, tensor_onnx__Conv_632_quantized_zero_point, tu66.tensor_onnx__Conv_632);
	node_onnx__Conv_634_DequantizeLinear( tensor_onnx__Conv_634_quantized, tensor_onnx__Conv_634_scale, tensor_onnx__Conv_634_zero_point, tu67.tensor_onnx__Conv_634_DequantizeLinear_Output);
	node_onnx__Conv_635_DequantizeLinear( tensor_onnx__Conv_635_quantized, tensor_onnx__Conv_635_quantized_scale, tensor_onnx__Conv_635_quantized_zero_point, tu68.tensor_onnx__Conv_635);
	node_onnx__Conv_637_DequantizeLinear( tensor_onnx__Conv_637_quantized, tensor_onnx__Conv_637_scale, tensor_onnx__Conv_637_zero_point, tu69.tensor_onnx__Conv_637_DequantizeLinear_Output);
	node_onnx__Conv_638_DequantizeLinear( tensor_onnx__Conv_638_quantized, tensor_onnx__Conv_638_quantized_scale, tensor_onnx__Conv_638_quantized_zero_point, tu70.tensor_onnx__Conv_638);
	node_onnx__Conv_640_DequantizeLinear( tensor_onnx__Conv_640_quantized, tensor_onnx__Conv_640_scale, tensor_onnx__Conv_640_zero_point, tu71.tensor_onnx__Conv_640_DequantizeLinear_Output);
	node_onnx__Conv_641_DequantizeLinear( tensor_onnx__Conv_641_quantized, tensor_onnx__Conv_641_quantized_scale, tensor_onnx__Conv_641_quantized_zero_point, tu72.tensor_onnx__Conv_641);
	node_onnx__Conv_643_DequantizeLinear( tensor_onnx__Conv_643_quantized, tensor_onnx__Conv_643_scale, tensor_onnx__Conv_643_zero_point, tu73.tensor_onnx__Conv_643_DequantizeLinear_Output);
	node_onnx__Conv_644_DequantizeLinear( tensor_onnx__Conv_644_quantized, tensor_onnx__Conv_644_quantized_scale, tensor_onnx__Conv_644_quantized_zero_point, tu74.tensor_onnx__Conv_644);
	node_onnx__Conv_646_DequantizeLinear( tensor_onnx__Conv_646_quantized, tensor_onnx__Conv_646_scale, tensor_onnx__Conv_646_zero_point, tu75.tensor_onnx__Conv_646_DequantizeLinear_Output);
	node_onnx__Conv_647_DequantizeLinear( tensor_onnx__Conv_647_quantized, tensor_onnx__Conv_647_quantized_scale, tensor_onnx__Conv_647_quantized_zero_point, tu76.tensor_onnx__Conv_647);
	node_onnx__Conv_649_DequantizeLinear( tensor_onnx__Conv_649_quantized, tensor_onnx__Conv_649_scale, tensor_onnx__Conv_649_zero_point, tu77.tensor_onnx__Conv_649_DequantizeLinear_Output);
	node_onnx__Conv_650_DequantizeLinear( tensor_onnx__Conv_650_quantized, tensor_onnx__Conv_650_quantized_scale, tensor_onnx__Conv_650_quantized_zero_point, tu78.tensor_onnx__Conv_650);
	node_onnx__Conv_652_DequantizeLinear( tensor_onnx__Conv_652_quantized, tensor_onnx__Conv_652_scale, tensor_onnx__Conv_652_zero_point, tu79.tensor_onnx__Conv_652_DequantizeLinear_Output);
	node_onnx__Conv_653_DequantizeLinear( tensor_onnx__Conv_653_quantized, tensor_onnx__Conv_653_quantized_scale, tensor_onnx__Conv_653_quantized_zero_point, tu80.tensor_onnx__Conv_653);
	node_onnx__Conv_655_DequantizeLinear( tensor_onnx__Conv_655_quantized, tensor_onnx__Conv_655_scale, tensor_onnx__Conv_655_zero_point, tu81.tensor_onnx__Conv_655_DequantizeLinear_Output);
	node_onnx__Conv_656_DequantizeLinear( tensor_onnx__Conv_656_quantized, tensor_onnx__Conv_656_quantized_scale, tensor_onnx__Conv_656_quantized_zero_point, tu82.tensor_onnx__Conv_656);
	node_onnx__Conv_658_DequantizeLinear( tensor_onnx__Conv_658_quantized, tensor_onnx__Conv_658_scale, tensor_onnx__Conv_658_zero_point, tu83.tensor_onnx__Conv_658_DequantizeLinear_Output);
	node_onnx__Conv_659_DequantizeLinear( tensor_onnx__Conv_659_quantized, tensor_onnx__Conv_659_quantized_scale, tensor_onnx__Conv_659_quantized_zero_point, tu84.tensor_onnx__Conv_659);
	node_onnx__Conv_661_DequantizeLinear( tensor_onnx__Conv_661_quantized, tensor_onnx__Conv_661_scale, tensor_onnx__Conv_661_zero_point, tu85.tensor_onnx__Conv_661_DequantizeLinear_Output);
	node_onnx__Conv_662_DequantizeLinear( tensor_onnx__Conv_662_quantized, tensor_onnx__Conv_662_quantized_scale, tensor_onnx__Conv_662_quantized_zero_point, tu86.tensor_onnx__Conv_662);
	node_onnx__Conv_664_DequantizeLinear( tensor_onnx__Conv_664_quantized, tensor_onnx__Conv_664_scale, tensor_onnx__Conv_664_zero_point, tu87.tensor_onnx__Conv_664_DequantizeLinear_Output);
	node_onnx__Conv_665_DequantizeLinear( tensor_onnx__Conv_665_quantized, tensor_onnx__Conv_665_quantized_scale, tensor_onnx__Conv_665_quantized_zero_point, tu88.tensor_onnx__Conv_665);
	node_onnx__Conv_667_DequantizeLinear( tensor_onnx__Conv_667_quantized, tensor_onnx__Conv_667_scale, tensor_onnx__Conv_667_zero_point, tu89.tensor_onnx__Conv_667_DequantizeLinear_Output);
	node_onnx__Conv_668_DequantizeLinear( tensor_onnx__Conv_668_quantized, tensor_onnx__Conv_668_quantized_scale, tensor_onnx__Conv_668_quantized_zero_point, tu90.tensor_onnx__Conv_668);
	node_onnx__Conv_670_DequantizeLinear( tensor_onnx__Conv_670_quantized, tensor_onnx__Conv_670_scale, tensor_onnx__Conv_670_zero_point, tu91.tensor_onnx__Conv_670_DequantizeLinear_Output);
	node_onnx__Conv_671_DequantizeLinear( tensor_onnx__Conv_671_quantized, tensor_onnx__Conv_671_quantized_scale, tensor_onnx__Conv_671_quantized_zero_point, tu92.tensor_onnx__Conv_671);
	node_onnx__Conv_673_DequantizeLinear( tensor_onnx__Conv_673_quantized, tensor_onnx__Conv_673_scale, tensor_onnx__Conv_673_zero_point, tu93.tensor_onnx__Conv_673_DequantizeLinear_Output);
	node_onnx__Conv_674_DequantizeLinear( tensor_onnx__Conv_674_quantized, tensor_onnx__Conv_674_quantized_scale, tensor_onnx__Conv_674_quantized_zero_point, tu94.tensor_onnx__Conv_674);
	node_onnx__Conv_676_DequantizeLinear( tensor_onnx__Conv_676_quantized, tensor_onnx__Conv_676_scale, tensor_onnx__Conv_676_zero_point, tu95.tensor_onnx__Conv_676_DequantizeLinear_Output);
	node_onnx__Conv_677_DequantizeLinear( tensor_onnx__Conv_677_quantized, tensor_onnx__Conv_677_quantized_scale, tensor_onnx__Conv_677_quantized_zero_point, tu96.tensor_onnx__Conv_677);
	node_onnx__Conv_679_DequantizeLinear( tensor_onnx__Conv_679_quantized, tensor_onnx__Conv_679_scale, tensor_onnx__Conv_679_zero_point, tu97.tensor_onnx__Conv_679_DequantizeLinear_Output);
	node_onnx__Conv_680_DequantizeLinear( tensor_onnx__Conv_680_quantized, tensor_onnx__Conv_680_quantized_scale, tensor_onnx__Conv_680_quantized_zero_point, tu98.tensor_onnx__Conv_680);
	node_onnx__Conv_682_DequantizeLinear( tensor_onnx__Conv_682_quantized, tensor_onnx__Conv_682_scale, tensor_onnx__Conv_682_zero_point, tu99.tensor_onnx__Conv_682_DequantizeLinear_Output);
	node_onnx__Conv_683_DequantizeLinear( tensor_onnx__Conv_683_quantized, tensor_onnx__Conv_683_quantized_scale, tensor_onnx__Conv_683_quantized_zero_point, tu100.tensor_onnx__Conv_683);
	node_onnx__Conv_685_DequantizeLinear( tensor_onnx__Conv_685_quantized, tensor_onnx__Conv_685_scale, tensor_onnx__Conv_685_zero_point, tu101.tensor_onnx__Conv_685_DequantizeLinear_Output);
	node_onnx__Conv_686_DequantizeLinear( tensor_onnx__Conv_686_quantized, tensor_onnx__Conv_686_quantized_scale, tensor_onnx__Conv_686_quantized_zero_point, tu102.tensor_onnx__Conv_686);
	node_onnx__Conv_688_DequantizeLinear( tensor_onnx__Conv_688_quantized, tensor_onnx__Conv_688_scale, tensor_onnx__Conv_688_zero_point, tu103.tensor_onnx__Conv_688_DequantizeLinear_Output);
	node_onnx__Conv_689_DequantizeLinear( tensor_onnx__Conv_689_quantized, tensor_onnx__Conv_689_quantized_scale, tensor_onnx__Conv_689_quantized_zero_point, tu104.tensor_onnx__Conv_689);
	node_onnx__Conv_691_DequantizeLinear( tensor_onnx__Conv_691_quantized, tensor_onnx__Conv_691_scale, tensor_onnx__Conv_691_zero_point, tu105.tensor_onnx__Conv_691_DequantizeLinear_Output);
	node_onnx__Conv_692_DequantizeLinear( tensor_onnx__Conv_692_quantized, tensor_onnx__Conv_692_quantized_scale, tensor_onnx__Conv_692_quantized_zero_point, tu106.tensor_onnx__Conv_692);
	node_input_DequantizeLinear( tu2.tensor_input_QuantizeLinear_Output, &tensor_input_scale, &tensor_input_zero_point, tu107.tensor_input_DequantizeLinear_Output);
	node__features_features_0_features_0_0_Conv( tu107.tensor_input_DequantizeLinear_Output, tu3.tensor_onnx__Conv_538_DequantizeLinear_Output, tu4.tensor_onnx__Conv_539, tu2.tensor__features_features_0_features_0_2_Clip_output_0);
	node__features_features_0_features_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_0_features_0_2_Clip_output_0, &tensor__features_features_0_features_0_2_Clip_output_0_scale, &tensor__features_features_0_features_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_0_features_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_0_features_0_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_0_features_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_0_features_0_2_Clip_output_0_scale, &tensor__features_features_0_features_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_0_features_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_1_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_0_features_0_2_Clip_output_0_DequantizeLinear_Output, tu5.tensor_onnx__Conv_541_DequantizeLinear_Output, tu6.tensor_onnx__Conv_542, tu3.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_1_conv_conv_1_Conv( tu3.tensor__features_features_1_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu7.tensor_onnx__Conv_544_DequantizeLinear_Output, tu8.tensor_onnx__Conv_545, tu2.tensor__features_features_1_conv_conv_1_Conv_output_0);
	node__features_features_1_conv_conv_1_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_1_conv_conv_1_Conv_output_0, &tensor__features_features_1_conv_conv_1_Conv_output_0_scale, &tensor__features_features_1_conv_conv_1_Conv_output_0_zero_point, tu3.tensor__features_features_1_conv_conv_1_Conv_output_0_QuantizeLinear_Output);
	node__features_features_1_conv_conv_1_Conv_output_0_DequantizeLinear( tu3.tensor__features_features_1_conv_conv_1_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_1_conv_conv_1_Conv_output_0_scale, &tensor__features_features_1_conv_conv_1_Conv_output_0_zero_point, tu2.tensor__features_features_1_conv_conv_1_Conv_output_0_DequantizeLinear_Output);
	node__features_features_2_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_1_conv_conv_1_Conv_output_0_DequantizeLinear_Output, tu9.tensor_onnx__Conv_547_DequantizeLinear_Output, tu10.tensor_onnx__Conv_548, tu3.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_2_conv_conv_1_conv_1_0_Conv( tu3.tensor__features_features_2_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu11.tensor_onnx__Conv_550_DequantizeLinear_Output, tu12.tensor_onnx__Conv_551, tu2.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_2_conv_conv_2_Conv( tu2.tensor__features_features_2_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu13.tensor_onnx__Conv_553_DequantizeLinear_Output, tu14.tensor_onnx__Conv_554, tu3.tensor__features_features_2_conv_conv_2_Conv_output_0);
	node__features_features_2_conv_conv_2_Conv_output_0_QuantizeLinear( tu3.tensor__features_features_2_conv_conv_2_Conv_output_0, &tensor__features_features_2_conv_conv_2_Conv_output_0_scale, &tensor__features_features_2_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_2_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_2_conv_conv_2_Conv_output_0_DequantizeLinear( tu2.tensor__features_features_2_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_2_conv_conv_2_Conv_output_0_scale, &tensor__features_features_2_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_2_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_3_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_2_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu15.tensor_onnx__Conv_556_DequantizeLinear_Output, tu16.tensor_onnx__Conv_557, tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_3_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_3_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu17.tensor_onnx__Conv_559_DequantizeLinear_Output, tu18.tensor_onnx__Conv_560, tu4.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_3_conv_conv_2_Conv( tu4.tensor__features_features_3_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu19.tensor_onnx__Conv_562_DequantizeLinear_Output, tu20.tensor_onnx__Conv_563, tu2.tensor__features_features_3_conv_conv_2_Conv_output_0);
	node__features_features_3_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_3_conv_conv_2_Conv_output_0, &tensor__features_features_3_conv_conv_2_Conv_output_0_scale, &tensor__features_features_3_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_3_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_3_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_3_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_3_conv_conv_2_Conv_output_0_scale, &tensor__features_features_3_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_3_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_3_Add( tu3.tensor__features_features_2_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu2.tensor__features_features_3_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_3_Add_output_0);
	node__features_features_3_Add_output_0_QuantizeLinear( tu4.tensor__features_features_3_Add_output_0, &tensor__features_features_3_Add_output_0_scale, &tensor__features_features_3_Add_output_0_zero_point, tu2.tensor__features_features_3_Add_output_0_QuantizeLinear_Output);
	node__features_features_3_Add_output_0_DequantizeLinear( tu2.tensor__features_features_3_Add_output_0_QuantizeLinear_Output, &tensor__features_features_3_Add_output_0_scale, &tensor__features_features_3_Add_output_0_zero_point, tu3.tensor__features_features_3_Add_output_0_DequantizeLinear_Output);
	node__features_features_4_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_3_Add_output_0_DequantizeLinear_Output, tu21.tensor_onnx__Conv_565_DequantizeLinear_Output, tu22.tensor_onnx__Conv_566, tu2.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_4_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_4_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu23.tensor_onnx__Conv_568_DequantizeLinear_Output, tu24.tensor_onnx__Conv_569, tu3.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_4_conv_conv_2_Conv( tu3.tensor__features_features_4_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu25.tensor_onnx__Conv_571_DequantizeLinear_Output, tu26.tensor_onnx__Conv_572, tu2.tensor__features_features_4_conv_conv_2_Conv_output_0);
	node__features_features_4_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_4_conv_conv_2_Conv_output_0, &tensor__features_features_4_conv_conv_2_Conv_output_0_scale, &tensor__features_features_4_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_4_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_4_conv_conv_2_Conv_output_0_DequantizeLinear( tu3.tensor__features_features_4_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_4_conv_conv_2_Conv_output_0_scale, &tensor__features_features_4_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_4_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_5_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_4_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu27.tensor_onnx__Conv_574_DequantizeLinear_Output, tu28.tensor_onnx__Conv_575, tu3.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_5_conv_conv_1_conv_1_0_Conv( tu3.tensor__features_features_5_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu29.tensor_onnx__Conv_577_DequantizeLinear_Output, tu30.tensor_onnx__Conv_578, tu4.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_5_conv_conv_2_Conv( tu4.tensor__features_features_5_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu31.tensor_onnx__Conv_580_DequantizeLinear_Output, tu32.tensor_onnx__Conv_581, tu3.tensor__features_features_5_conv_conv_2_Conv_output_0);
	node__features_features_5_conv_conv_2_Conv_output_0_QuantizeLinear( tu3.tensor__features_features_5_conv_conv_2_Conv_output_0, &tensor__features_features_5_conv_conv_2_Conv_output_0_scale, &tensor__features_features_5_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_5_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_5_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_5_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_5_conv_conv_2_Conv_output_0_scale, &tensor__features_features_5_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_5_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_5_Add( tu2.tensor__features_features_4_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu3.tensor__features_features_5_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_5_Add_output_0);
	node__features_features_5_Add_output_0_QuantizeLinear( tu4.tensor__features_features_5_Add_output_0, &tensor__features_features_5_Add_output_0_scale, &tensor__features_features_5_Add_output_0_zero_point, tu2.tensor__features_features_5_Add_output_0_QuantizeLinear_Output);
	node__features_features_5_Add_output_0_DequantizeLinear( tu2.tensor__features_features_5_Add_output_0_QuantizeLinear_Output, &tensor__features_features_5_Add_output_0_scale, &tensor__features_features_5_Add_output_0_zero_point, tu3.tensor__features_features_5_Add_output_0_DequantizeLinear_Output);
	node__features_features_6_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_5_Add_output_0_DequantizeLinear_Output, tu33.tensor_onnx__Conv_583_DequantizeLinear_Output, tu34.tensor_onnx__Conv_584, tu2.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_6_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_6_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu35.tensor_onnx__Conv_586_DequantizeLinear_Output, tu36.tensor_onnx__Conv_587, tu4.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_6_conv_conv_2_Conv( tu4.tensor__features_features_6_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu37.tensor_onnx__Conv_589_DequantizeLinear_Output, tu38.tensor_onnx__Conv_590, tu2.tensor__features_features_6_conv_conv_2_Conv_output_0);
	node__features_features_6_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_6_conv_conv_2_Conv_output_0, &tensor__features_features_6_conv_conv_2_Conv_output_0_scale, &tensor__features_features_6_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_6_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_6_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_6_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_6_conv_conv_2_Conv_output_0_scale, &tensor__features_features_6_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_6_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_6_Add( tu3.tensor__features_features_5_Add_output_0_DequantizeLinear_Output, tu2.tensor__features_features_6_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_6_Add_output_0);
	node__features_features_6_Add_output_0_QuantizeLinear( tu4.tensor__features_features_6_Add_output_0, &tensor__features_features_6_Add_output_0_scale, &tensor__features_features_6_Add_output_0_zero_point, tu2.tensor__features_features_6_Add_output_0_QuantizeLinear_Output);
	node__features_features_6_Add_output_0_DequantizeLinear( tu2.tensor__features_features_6_Add_output_0_QuantizeLinear_Output, &tensor__features_features_6_Add_output_0_scale, &tensor__features_features_6_Add_output_0_zero_point, tu3.tensor__features_features_6_Add_output_0_DequantizeLinear_Output);
	node__features_features_7_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_6_Add_output_0_DequantizeLinear_Output, tu39.tensor_onnx__Conv_592_DequantizeLinear_Output, tu40.tensor_onnx__Conv_593, tu2.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_7_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_7_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu41.tensor_onnx__Conv_595_DequantizeLinear_Output, tu42.tensor_onnx__Conv_596, tu3.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_7_conv_conv_2_Conv( tu3.tensor__features_features_7_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu43.tensor_onnx__Conv_598_DequantizeLinear_Output, tu44.tensor_onnx__Conv_599, tu2.tensor__features_features_7_conv_conv_2_Conv_output_0);
	node__features_features_7_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_7_conv_conv_2_Conv_output_0, &tensor__features_features_7_conv_conv_2_Conv_output_0_scale, &tensor__features_features_7_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_7_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_7_conv_conv_2_Conv_output_0_DequantizeLinear( tu3.tensor__features_features_7_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_7_conv_conv_2_Conv_output_0_scale, &tensor__features_features_7_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_7_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_8_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_7_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu45.tensor_onnx__Conv_601_DequantizeLinear_Output, tu46.tensor_onnx__Conv_602, tu3.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_8_conv_conv_1_conv_1_0_Conv( tu3.tensor__features_features_8_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu47.tensor_onnx__Conv_604_DequantizeLinear_Output, tu48.tensor_onnx__Conv_605, tu4.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_8_conv_conv_2_Conv( tu4.tensor__features_features_8_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu49.tensor_onnx__Conv_607_DequantizeLinear_Output, tu50.tensor_onnx__Conv_608, tu3.tensor__features_features_8_conv_conv_2_Conv_output_0);
	node__features_features_8_conv_conv_2_Conv_output_0_QuantizeLinear( tu3.tensor__features_features_8_conv_conv_2_Conv_output_0, &tensor__features_features_8_conv_conv_2_Conv_output_0_scale, &tensor__features_features_8_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_8_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_8_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_8_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_8_conv_conv_2_Conv_output_0_scale, &tensor__features_features_8_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_8_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_8_Add( tu2.tensor__features_features_7_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu3.tensor__features_features_8_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_8_Add_output_0);
	node__features_features_8_Add_output_0_QuantizeLinear( tu4.tensor__features_features_8_Add_output_0, &tensor__features_features_8_Add_output_0_scale, &tensor__features_features_8_Add_output_0_zero_point, tu2.tensor__features_features_8_Add_output_0_QuantizeLinear_Output);
	node__features_features_8_Add_output_0_DequantizeLinear( tu2.tensor__features_features_8_Add_output_0_QuantizeLinear_Output, &tensor__features_features_8_Add_output_0_scale, &tensor__features_features_8_Add_output_0_zero_point, tu3.tensor__features_features_8_Add_output_0_DequantizeLinear_Output);
	node__features_features_9_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_8_Add_output_0_DequantizeLinear_Output, tu51.tensor_onnx__Conv_610_DequantizeLinear_Output, tu52.tensor_onnx__Conv_611, tu2.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_9_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_9_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu53.tensor_onnx__Conv_613_DequantizeLinear_Output, tu54.tensor_onnx__Conv_614, tu4.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_9_conv_conv_2_Conv( tu4.tensor__features_features_9_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu55.tensor_onnx__Conv_616_DequantizeLinear_Output, tu56.tensor_onnx__Conv_617, tu2.tensor__features_features_9_conv_conv_2_Conv_output_0);
	node__features_features_9_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_9_conv_conv_2_Conv_output_0, &tensor__features_features_9_conv_conv_2_Conv_output_0_scale, &tensor__features_features_9_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_9_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_9_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_9_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_9_conv_conv_2_Conv_output_0_scale, &tensor__features_features_9_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_9_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_9_Add( tu3.tensor__features_features_8_Add_output_0_DequantizeLinear_Output, tu2.tensor__features_features_9_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_9_Add_output_0);
	node__features_features_9_Add_output_0_QuantizeLinear( tu4.tensor__features_features_9_Add_output_0, &tensor__features_features_9_Add_output_0_scale, &tensor__features_features_9_Add_output_0_zero_point, tu2.tensor__features_features_9_Add_output_0_QuantizeLinear_Output);
	node__features_features_9_Add_output_0_DequantizeLinear( tu2.tensor__features_features_9_Add_output_0_QuantizeLinear_Output, &tensor__features_features_9_Add_output_0_scale, &tensor__features_features_9_Add_output_0_zero_point, tu3.tensor__features_features_9_Add_output_0_DequantizeLinear_Output);
	node__features_features_10_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_9_Add_output_0_DequantizeLinear_Output, tu57.tensor_onnx__Conv_619_DequantizeLinear_Output, tu58.tensor_onnx__Conv_620, tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_10_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_10_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu59.tensor_onnx__Conv_622_DequantizeLinear_Output, tu60.tensor_onnx__Conv_623, tu4.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_10_conv_conv_2_Conv( tu4.tensor__features_features_10_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu61.tensor_onnx__Conv_625_DequantizeLinear_Output, tu62.tensor_onnx__Conv_626, tu2.tensor__features_features_10_conv_conv_2_Conv_output_0);
	node__features_features_10_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_10_conv_conv_2_Conv_output_0, &tensor__features_features_10_conv_conv_2_Conv_output_0_scale, &tensor__features_features_10_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_10_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_10_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_10_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_10_conv_conv_2_Conv_output_0_scale, &tensor__features_features_10_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_10_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_10_Add( tu3.tensor__features_features_9_Add_output_0_DequantizeLinear_Output, tu2.tensor__features_features_10_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_10_Add_output_0);
	node__features_features_10_Add_output_0_QuantizeLinear( tu4.tensor__features_features_10_Add_output_0, &tensor__features_features_10_Add_output_0_scale, &tensor__features_features_10_Add_output_0_zero_point, tu2.tensor__features_features_10_Add_output_0_QuantizeLinear_Output);
	node__features_features_10_Add_output_0_DequantizeLinear( tu2.tensor__features_features_10_Add_output_0_QuantizeLinear_Output, &tensor__features_features_10_Add_output_0_scale, &tensor__features_features_10_Add_output_0_zero_point, tu3.tensor__features_features_10_Add_output_0_DequantizeLinear_Output);
	node__features_features_11_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_10_Add_output_0_DequantizeLinear_Output, tu63.tensor_onnx__Conv_628_DequantizeLinear_Output, tu64.tensor_onnx__Conv_629, tu2.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_11_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_11_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu65.tensor_onnx__Conv_631_DequantizeLinear_Output, tu66.tensor_onnx__Conv_632, tu3.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_11_conv_conv_2_Conv( tu3.tensor__features_features_11_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu67.tensor_onnx__Conv_634_DequantizeLinear_Output, tu68.tensor_onnx__Conv_635, tu2.tensor__features_features_11_conv_conv_2_Conv_output_0);
	node__features_features_11_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_11_conv_conv_2_Conv_output_0, &tensor__features_features_11_conv_conv_2_Conv_output_0_scale, &tensor__features_features_11_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_11_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_11_conv_conv_2_Conv_output_0_DequantizeLinear( tu3.tensor__features_features_11_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_11_conv_conv_2_Conv_output_0_scale, &tensor__features_features_11_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_11_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_12_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_11_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu69.tensor_onnx__Conv_637_DequantizeLinear_Output, tu70.tensor_onnx__Conv_638, tu3.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_12_conv_conv_1_conv_1_0_Conv( tu3.tensor__features_features_12_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu71.tensor_onnx__Conv_640_DequantizeLinear_Output, tu72.tensor_onnx__Conv_641, tu4.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_12_conv_conv_2_Conv( tu4.tensor__features_features_12_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu73.tensor_onnx__Conv_643_DequantizeLinear_Output, tu74.tensor_onnx__Conv_644, tu3.tensor__features_features_12_conv_conv_2_Conv_output_0);
	node__features_features_12_conv_conv_2_Conv_output_0_QuantizeLinear( tu3.tensor__features_features_12_conv_conv_2_Conv_output_0, &tensor__features_features_12_conv_conv_2_Conv_output_0_scale, &tensor__features_features_12_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_12_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_12_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_12_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_12_conv_conv_2_Conv_output_0_scale, &tensor__features_features_12_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_12_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_12_Add( tu2.tensor__features_features_11_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu3.tensor__features_features_12_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_12_Add_output_0);
	node__features_features_12_Add_output_0_QuantizeLinear( tu4.tensor__features_features_12_Add_output_0, &tensor__features_features_12_Add_output_0_scale, &tensor__features_features_12_Add_output_0_zero_point, tu2.tensor__features_features_12_Add_output_0_QuantizeLinear_Output);
	node__features_features_12_Add_output_0_DequantizeLinear( tu2.tensor__features_features_12_Add_output_0_QuantizeLinear_Output, &tensor__features_features_12_Add_output_0_scale, &tensor__features_features_12_Add_output_0_zero_point, tu3.tensor__features_features_12_Add_output_0_DequantizeLinear_Output);
	node__features_features_13_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_12_Add_output_0_DequantizeLinear_Output, tu75.tensor_onnx__Conv_646_DequantizeLinear_Output, tu76.tensor_onnx__Conv_647, tu2.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_13_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_13_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu77.tensor_onnx__Conv_649_DequantizeLinear_Output, tu78.tensor_onnx__Conv_650, tu4.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_13_conv_conv_2_Conv( tu4.tensor__features_features_13_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu79.tensor_onnx__Conv_652_DequantizeLinear_Output, tu80.tensor_onnx__Conv_653, tu2.tensor__features_features_13_conv_conv_2_Conv_output_0);
	node__features_features_13_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_13_conv_conv_2_Conv_output_0, &tensor__features_features_13_conv_conv_2_Conv_output_0_scale, &tensor__features_features_13_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_13_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_13_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_13_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_13_conv_conv_2_Conv_output_0_scale, &tensor__features_features_13_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_13_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_13_Add( tu3.tensor__features_features_12_Add_output_0_DequantizeLinear_Output, tu2.tensor__features_features_13_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_13_Add_output_0);
	node__features_features_13_Add_output_0_QuantizeLinear( tu4.tensor__features_features_13_Add_output_0, &tensor__features_features_13_Add_output_0_scale, &tensor__features_features_13_Add_output_0_zero_point, tu2.tensor__features_features_13_Add_output_0_QuantizeLinear_Output);
	node__features_features_13_Add_output_0_DequantizeLinear( tu2.tensor__features_features_13_Add_output_0_QuantizeLinear_Output, &tensor__features_features_13_Add_output_0_scale, &tensor__features_features_13_Add_output_0_zero_point, tu3.tensor__features_features_13_Add_output_0_DequantizeLinear_Output);
	node__features_features_14_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_13_Add_output_0_DequantizeLinear_Output, tu81.tensor_onnx__Conv_655_DequantizeLinear_Output, tu82.tensor_onnx__Conv_656, tu2.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_14_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_14_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu83.tensor_onnx__Conv_658_DequantizeLinear_Output, tu84.tensor_onnx__Conv_659, tu3.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_14_conv_conv_2_Conv( tu3.tensor__features_features_14_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu85.tensor_onnx__Conv_661_DequantizeLinear_Output, tu86.tensor_onnx__Conv_662, tu2.tensor__features_features_14_conv_conv_2_Conv_output_0);
	node__features_features_14_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_14_conv_conv_2_Conv_output_0, &tensor__features_features_14_conv_conv_2_Conv_output_0_scale, &tensor__features_features_14_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_14_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_14_conv_conv_2_Conv_output_0_DequantizeLinear( tu3.tensor__features_features_14_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_14_conv_conv_2_Conv_output_0_scale, &tensor__features_features_14_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_14_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_15_conv_conv_0_conv_0_0_Conv( tu2.tensor__features_features_14_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu87.tensor_onnx__Conv_664_DequantizeLinear_Output, tu88.tensor_onnx__Conv_665, tu3.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_15_conv_conv_1_conv_1_0_Conv( tu3.tensor__features_features_15_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu89.tensor_onnx__Conv_667_DequantizeLinear_Output, tu90.tensor_onnx__Conv_668, tu4.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_15_conv_conv_2_Conv( tu4.tensor__features_features_15_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu91.tensor_onnx__Conv_670_DequantizeLinear_Output, tu92.tensor_onnx__Conv_671, tu3.tensor__features_features_15_conv_conv_2_Conv_output_0);
	node__features_features_15_conv_conv_2_Conv_output_0_QuantizeLinear( tu3.tensor__features_features_15_conv_conv_2_Conv_output_0, &tensor__features_features_15_conv_conv_2_Conv_output_0_scale, &tensor__features_features_15_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_15_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_15_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_15_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_15_conv_conv_2_Conv_output_0_scale, &tensor__features_features_15_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_15_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_15_Add( tu2.tensor__features_features_14_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu3.tensor__features_features_15_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_15_Add_output_0);
	node__features_features_15_Add_output_0_QuantizeLinear( tu4.tensor__features_features_15_Add_output_0, &tensor__features_features_15_Add_output_0_scale, &tensor__features_features_15_Add_output_0_zero_point, tu2.tensor__features_features_15_Add_output_0_QuantizeLinear_Output);
	node__features_features_15_Add_output_0_DequantizeLinear( tu2.tensor__features_features_15_Add_output_0_QuantizeLinear_Output, &tensor__features_features_15_Add_output_0_scale, &tensor__features_features_15_Add_output_0_zero_point, tu3.tensor__features_features_15_Add_output_0_DequantizeLinear_Output);
	node__features_features_16_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_15_Add_output_0_DequantizeLinear_Output, tu93.tensor_onnx__Conv_673_DequantizeLinear_Output, tu94.tensor_onnx__Conv_674, tu2.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu4.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu4.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_16_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_16_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu95.tensor_onnx__Conv_676_DequantizeLinear_Output, tu96.tensor_onnx__Conv_677, tu4.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu4.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu4.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_16_conv_conv_2_Conv( tu4.tensor__features_features_16_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu97.tensor_onnx__Conv_679_DequantizeLinear_Output, tu98.tensor_onnx__Conv_680, tu2.tensor__features_features_16_conv_conv_2_Conv_output_0);
	node__features_features_16_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_16_conv_conv_2_Conv_output_0, &tensor__features_features_16_conv_conv_2_Conv_output_0_scale, &tensor__features_features_16_conv_conv_2_Conv_output_0_zero_point, tu4.tensor__features_features_16_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_16_conv_conv_2_Conv_output_0_DequantizeLinear( tu4.tensor__features_features_16_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_16_conv_conv_2_Conv_output_0_scale, &tensor__features_features_16_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_16_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_16_Add( tu3.tensor__features_features_15_Add_output_0_DequantizeLinear_Output, tu2.tensor__features_features_16_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu4.tensor__features_features_16_Add_output_0);
	node__features_features_16_Add_output_0_QuantizeLinear( tu4.tensor__features_features_16_Add_output_0, &tensor__features_features_16_Add_output_0_scale, &tensor__features_features_16_Add_output_0_zero_point, tu2.tensor__features_features_16_Add_output_0_QuantizeLinear_Output);
	node__features_features_16_Add_output_0_DequantizeLinear( tu2.tensor__features_features_16_Add_output_0_QuantizeLinear_Output, &tensor__features_features_16_Add_output_0_scale, &tensor__features_features_16_Add_output_0_zero_point, tu3.tensor__features_features_16_Add_output_0_DequantizeLinear_Output);
	node__features_features_17_conv_conv_0_conv_0_0_Conv( tu3.tensor__features_features_16_Add_output_0_DequantizeLinear_Output, tu99.tensor_onnx__Conv_682_DequantizeLinear_Output, tu100.tensor_onnx__Conv_683, tu2.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0);
	node__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear( tu2.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0, &tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu3.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear( tu3.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_scale, &tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_zero_point, tu2.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_17_conv_conv_1_conv_1_0_Conv( tu2.tensor__features_features_17_conv_conv_0_conv_0_2_Clip_output_0_DequantizeLinear_Output, tu101.tensor_onnx__Conv_685_DequantizeLinear_Output, tu102.tensor_onnx__Conv_686, tu3.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0);
	node__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0, &tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu2.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_scale, &tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_zero_point, tu3.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output);
	node__features_features_17_conv_conv_2_Conv( tu3.tensor__features_features_17_conv_conv_1_conv_1_2_Clip_output_0_DequantizeLinear_Output, tu103.tensor_onnx__Conv_688_DequantizeLinear_Output, tu104.tensor_onnx__Conv_689, tu2.tensor__features_features_17_conv_conv_2_Conv_output_0);
	node__features_features_17_conv_conv_2_Conv_output_0_QuantizeLinear( tu2.tensor__features_features_17_conv_conv_2_Conv_output_0, &tensor__features_features_17_conv_conv_2_Conv_output_0_scale, &tensor__features_features_17_conv_conv_2_Conv_output_0_zero_point, tu3.tensor__features_features_17_conv_conv_2_Conv_output_0_QuantizeLinear_Output);
	node__features_features_17_conv_conv_2_Conv_output_0_DequantizeLinear( tu3.tensor__features_features_17_conv_conv_2_Conv_output_0_QuantizeLinear_Output, &tensor__features_features_17_conv_conv_2_Conv_output_0_scale, &tensor__features_features_17_conv_conv_2_Conv_output_0_zero_point, tu2.tensor__features_features_17_conv_conv_2_Conv_output_0_DequantizeLinear_Output);
	node__features_features_18_features_18_0_Conv( tu2.tensor__features_features_17_conv_conv_2_Conv_output_0_DequantizeLinear_Output, tu105.tensor_onnx__Conv_691_DequantizeLinear_Output, tu106.tensor_onnx__Conv_692, tu3.tensor__features_features_18_features_18_2_Clip_output_0);
	node__features_features_18_features_18_2_Clip_output_0_QuantizeLinear( tu3.tensor__features_features_18_features_18_2_Clip_output_0, &tensor__features_features_18_features_18_2_Clip_output_0_scale, &tensor__features_features_18_features_18_2_Clip_output_0_zero_point, tu2.tensor__features_features_18_features_18_2_Clip_output_0_QuantizeLinear_Output);
	node__features_features_18_features_18_2_Clip_output_0_DequantizeLinear( tu2.tensor__features_features_18_features_18_2_Clip_output_0_QuantizeLinear_Output, &tensor__features_features_18_features_18_2_Clip_output_0_scale, &tensor__features_features_18_features_18_2_Clip_output_0_zero_point, tu3.tensor__features_features_18_features_18_2_Clip_output_0_DequantizeLinear_Output);
	node__GlobalAveragePool( tu3.tensor__features_features_18_features_18_2_Clip_output_0_DequantizeLinear_Output, tu2.tensor__GlobalAveragePool_output_0);
	node__GlobalAveragePool_output_0_QuantizeLinear( tu2.tensor__GlobalAveragePool_output_0, &tensor__GlobalAveragePool_output_0_scale, &tensor__GlobalAveragePool_output_0_zero_point, tu3.tensor__GlobalAveragePool_output_0_QuantizeLinear_Output);
	node__GlobalAveragePool_output_0_DequantizeLinear( tu3.tensor__GlobalAveragePool_output_0_QuantizeLinear_Output, &tensor__GlobalAveragePool_output_0_scale, &tensor__GlobalAveragePool_output_0_zero_point, tu2.tensor__GlobalAveragePool_output_0_DequantizeLinear_Output);
	node__Flatten( tu2.tensor__GlobalAveragePool_output_0_DequantizeLinear_Output, tu3.tensor__Flatten_output_0);
	node__Flatten_output_0_QuantizeLinear( tu3.tensor__Flatten_output_0, &tensor__Flatten_output_0_scale, &tensor__Flatten_output_0_zero_point, tu2.tensor__Flatten_output_0_QuantizeLinear_Output);
	node__Flatten_output_0_DequantizeLinear( tu2.tensor__Flatten_output_0_QuantizeLinear_Output, &tensor__Flatten_output_0_scale, &tensor__Flatten_output_0_zero_point, tu3.tensor__Flatten_output_0_DequantizeLinear_Output);
	node__classifier_classifier_1_Gemm( tu3.tensor__Flatten_output_0_DequantizeLinear_Output, tu1.tensor_classifier_1_weight_DequantizeLinear_Output, tu0.tensor_classifier_1_bias, tu2.tensor_logits_QuantizeLinear_Input);
	node_logits_QuantizeLinear( tu2.tensor_logits_QuantizeLinear_Input, &tensor_logits_scale, &tensor_logits_zero_point, tu0.tensor_logits_QuantizeLinear_Output);
	node_logits_DequantizeLinear( tu0.tensor_logits_QuantizeLinear_Output, &tensor_logits_scale, &tensor_logits_zero_point, tensor_logits);
}
