#pragma once

#include <stddef.h>
#include <stdint.h>

#include "layers.h"

/* Packing helpers turn ONNX-style weights/bias into vec-nn layouts. */
size_t pack_conv1x1_int8(const int8_t *weights, const int32_t *bias,
                         size_t out_ch, size_t in_ch, int8_t *dst);

size_t pack_fc_int8(const int8_t *weights, const int32_t *bias,
                    size_t out_ch, size_t in_ch, int8_t *dst);

size_t pack_dw3x3_int8(const int8_t *weights, const int32_t *bias,
                       size_t channels, int8_t *dst);

/* Quantization glue. */
void compute_requant_scales(const float *w_scale, float act_scale,
                            float out_scale, size_t count, float *dst);

requantization_params_t make_requant_params(float *scales, int32_t zero_point);

void recenter_uint8_to_int8(const uint8_t *src, int8_t *dst,
                            size_t size, int32_t offset);
