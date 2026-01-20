#include "vec_extra_kernels.h"

#include <math.h>

static inline int32_t clamp_int8(int32_t v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return v;
}

void conv3x3_stride2_int8(const int8_t *input, size_t H, size_t W, size_t Cin,
                          const int8_t *weights, const int32_t *bias, size_t Cout,
                          int padding, int8_t *output, requantization_params_t rqp) {
    const size_t H_out = (H + 2 * (size_t)padding - 3) / 2 + 1;
    const size_t W_out = (W + 2 * (size_t)padding - 3) / 2 + 1;
    const size_t kernel_size = 3 * 3;

    for (size_t oc = 0; oc < Cout; ++oc) {
        const int8_t *w_ch = weights + oc * (Cin * kernel_size);
        const float s = rqp.scale[oc];
        const int32_t zp = rqp.zero_point;
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                int32_t acc = bias ? bias[oc] : 0;
                const int ih_base = (int)(oh * 2) - padding;
                const int iw_base = (int)(ow * 2) - padding;

                for (size_t c = 0; c < Cin; ++c) {
                    const int8_t *w_base = w_ch + c * kernel_size;
                    for (size_t kh = 0; kh < 3; ++kh) {
                        const int ih = ih_base + (int)kh;
                        if (ih < 0 || ih >= (int)H) continue;
                        for (size_t kw = 0; kw < 3; ++kw) {
                            const int iw = iw_base + (int)kw;
                            if (iw < 0 || iw >= (int)W) continue;
                            const int idx = (c * H + (size_t)ih) * W + (size_t)iw;
                            acc += (int32_t)input[idx] * (int32_t)w_base[kh * 3 + kw];
                        }
                    }
                }

                const float f = (float)acc * s + (float)zp;
                const int32_t q = (int32_t)lrintf(f);
                output[(oc * H_out + oh) * W_out + ow] = (int8_t)clamp_int8(q);
            }
        }
    }
}
