#pragma once

#include <stddef.h>
#include <stdint.h>

#include "layers.h"

/*
 * Fallback/standalone kernels that are not present in vec-nn.
 *
 * weights layout: per output channel, contiguous Cin*3*3 int8 values.
 * bias layout: per output channel int32.
 */
void conv3x3_stride2_int8(const int8_t *input, size_t H, size_t W, size_t Cin,
                          const int8_t *weights, const int32_t *bias, size_t Cout,
                          int padding, int8_t *output, requantization_params_t rqp);
