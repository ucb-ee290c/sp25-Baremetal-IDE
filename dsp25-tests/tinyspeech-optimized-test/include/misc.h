#ifndef MISC_H
#define MISC_H

#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <math.h>
#include <float.h>
#include "tensor.h"

#ifndef CONVERT_FLOAT
#define CONVERT_FLOAT 1
#endif

#ifndef CONVERT_INT8
#define CONVERT_INT8 0
#endif

void quantize_weights(Tensor *w, Tensor *u, float* scale, u_int8_t retain_float);

void dequantize_weights(Tensor* quantized_weights, Tensor* dequantized_weights, float scale);
Tensor sigmoid(Tensor *tensor);
void attention(Tensor *residual, Tensor *S, Tensor *scale); 
float mean(int8_t *data, int size);
void relu(Tensor* input);
int8_t clamp(int8_t val, int8_t min_val, int8_t max_val);
float compute_mean_abs(int32_t *w, int32_t len);

#endif
