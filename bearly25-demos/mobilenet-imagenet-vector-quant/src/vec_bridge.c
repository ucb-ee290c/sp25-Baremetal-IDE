#include "vec_bridge.h"

#include <string.h>

size_t pack_conv1x1_int8(const int8_t *weights, const int32_t *bias,
                         size_t out_ch, size_t in_ch, int8_t *dst) {
    /* Layout: for each output channel -> int32 bias, then in_ch int8 weights */
    uint8_t *write = (uint8_t *)dst;
    for (size_t oc = 0; oc < out_ch; ++oc) {
        memcpy(write, &bias[oc], sizeof(int32_t));
        write += sizeof(int32_t);
        const int8_t *row = weights + oc * in_ch;
        memcpy(write, row, in_ch * sizeof(int8_t));
        write += in_ch * sizeof(int8_t);
    }
    return (size_t)(write - (uint8_t *)dst);
}

size_t pack_fc_int8(const int8_t *weights, const int32_t *bias,
                    size_t out_ch, size_t in_ch, int8_t *dst) {
    return pack_conv1x1_int8(weights, bias, out_ch, in_ch, dst);
}

size_t pack_dw3x3_int8(const int8_t *weights, const int32_t *bias,
                       size_t channels, int8_t *dst) {
    /* Layout per channel: int32 bias, then 9 int8 weights */
    uint8_t *write = (uint8_t *)dst;
    for (size_t c = 0; c < channels; ++c) {
        memcpy(write, &bias[c], sizeof(int32_t));
        write += sizeof(int32_t);
        const int8_t *k = weights + c * 9;
        memcpy(write, k, 9 * sizeof(int8_t));
        write += 9 * sizeof(int8_t);
    }
    return (size_t)(write - (uint8_t *)dst);
}

void compute_requant_scales(const float *w_scale, float act_scale,
                            float out_scale, size_t count, float *dst) {
    const float base = act_scale / out_scale;
    for (size_t i = 0; i < count; ++i) {
        dst[i] = base * w_scale[i];
    }
}

requantization_params_t make_requant_params(float *scales, int32_t zero_point) {
    requantization_params_t rqp;
    rqp.scale = scales;
    rqp.zero_point = zero_point;
    return rqp;
}

void recenter_uint8_to_int8(const uint8_t *src, int8_t *dst,
                            size_t size, int32_t offset) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = (int8_t)((int32_t)src[i] + offset);
    }
}
