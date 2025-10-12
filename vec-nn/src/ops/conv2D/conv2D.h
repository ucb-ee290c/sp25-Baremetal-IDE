#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#include "stdio.h"
#include <stdint.h>
#include "layers.h"

void dwconv_3x3_int8_VCO(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
);

void dwconv_3x3_int8_VCO_relu(
    size_t rows, size_t cols,
    size_t channels,
    size_t a_stride, size_t b_stride,
    const void *weights,
    int8_t *input, 
    int8_t *output,
    requantization_params_t requant_params
);