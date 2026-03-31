#include "modules.h"
#include "misc.h"

#include <float.h>
#include <stdint.h>
#include <stddef.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

static inline float tensor_get_value(const Tensor *t, int32_t idx) {
    if (t->f_data != NULL) {
        return t->f_data[idx];
    }
    return (float)t->data[idx];
}

static inline float tensor_get_channel_value(const Tensor *t, int32_t ch) {
    if (t->size <= 1) {
        return tensor_get_value(t, 0);
    }
    return tensor_get_value(t, ch);
}

static inline float decode_packed_float(float raw) {
    if (!isfinite(raw)) {
        return 1.0f;
    }

    if ((raw > 1000000.0f) && (raw < 4294967295.0f)) {
        uint32_t bits = (uint32_t)raw;
        uint32_t candidates[2] = {bits, bits};
        int32_t n = 1;

        if (bits == 0x80000000u) {
            candidates[1] = 0x7fffffffu;
            n = 2;
        }

        for (int32_t i = 0; i < n; i++) {
            union {
                uint32_t u;
                float f;
            } cvt;
            cvt.u = candidates[i];
            if (isfinite(cvt.f) && (fabsf(cvt.f) > 1e-12f)) {
                return cvt.f;
            }
        }

        return 1.0f;
    }

    return raw;
}

static inline float bn_param_value(const Tensor *t, int32_t ch) {
    if (t->f_data != NULL) {
        return tensor_get_channel_value(t, ch);
    }
    if (t->size <= 0) {
        return 0.0f;
    }
    int32_t id = (t->size <= 1) ? 0 : ch;
    if (id >= t->size) {
        id = t->size - 1;
    }
    return (float)t->data[id] / 127.0f;
}

static void conv2d_scalar_impl(const Tensor *input,
                               const Tensor *weights,
                               const Tensor *bias,
                               float out_scale,
                               int32_t stride,
                               int32_t padding,
                               Tensor *output) {
    int32_t batch_size = input->shape[0];
    int32_t in_channels = input->shape[1];
    int32_t in_height = input->shape[2];
    int32_t in_width = input->shape[3];
    int32_t out_channels = weights->shape[0];
    int32_t kernel_height = weights->shape[2];
    int32_t kernel_width = weights->shape[3];
    int32_t out_height = output->shape[2];
    int32_t out_width = output->shape[3];

    for (int32_t n = 0; n < batch_size; n++) {
        for (int32_t oc = 0; oc < out_channels; oc++) {
            for (int32_t h = 0; h < out_height; h++) {
                for (int32_t w = 0; w < out_width; w++) {
                    float sum = tensor_get_channel_value(bias, oc);

                    for (int32_t ic = 0; ic < in_channels; ic++) {
                        for (int32_t kh = 0; kh < kernel_height; kh++) {
                            for (int32_t kw = 0; kw < kernel_width; kw++) {
                                int32_t ih = h * stride + kh - padding;
                                int32_t iw = w * stride + kw - padding;

                                if ((ih >= 0) && (ih < in_height) && (iw >= 0) && (iw < in_width)) {
                                    int32_t in_index = n * (in_channels * in_height * in_width) +
                                                       ic * (in_height * in_width) +
                                                       ih * in_width + iw;
                                    int32_t weight_index = oc * (in_channels * kernel_height * kernel_width) +
                                                           ic * (kernel_height * kernel_width) +
                                                           kh * kernel_width + kw;
                                    sum += tensor_get_value(input, in_index) * tensor_get_value(weights, weight_index);
                                }
                            }
                        }
                    }

                    int32_t out_index = n * (out_channels * out_height * out_width) +
                                        oc * (out_height * out_width) +
                                        h * out_width + w;
                    output->f_data[out_index] = sum / out_scale;
                }
            }
        }
    }
}

#if defined(__riscv_vector)
static int conv2d_rvv_implicit_gemm_impl(const Tensor *input,
                                         const Tensor *weights,
                                         const Tensor *bias,
                                         float out_scale,
                                         int32_t padding,
                                         Tensor *output) {
    const int32_t batch_size = input->shape[0];
    const int32_t in_channels = input->shape[1];
    const int32_t in_height = input->shape[2];
    const int32_t in_width = input->shape[3];
    const int32_t out_channels = weights->shape[0];
    const int32_t out_height = output->shape[2];
    const int32_t out_width = output->shape[3];
    const int32_t in_hw = in_height * in_width;
    const int32_t out_hw = out_height * out_width;
    const int32_t K = in_channels * 9;
    const float inv_scale = 1.0f / out_scale;

    float *patch = (float *)malloc((size_t)K * sizeof(float));
    if (patch == NULL) {
        return 0;
    }

    for (int32_t n = 0; n < batch_size; n++) {
        const float *in_n = input->f_data + n * (in_channels * in_hw);
        float *out_n = output->f_data + n * (out_channels * out_hw);

        for (int32_t oh = 0; oh < out_height; oh++) {
            for (int32_t ow = 0; ow < out_width; ow++) {
                int32_t kidx = 0;
                for (int32_t ic = 0; ic < in_channels; ic++) {
                    const float *in_c = in_n + ic * in_hw;
                    for (int32_t kh = 0; kh < 3; kh++) {
                        int32_t ih = oh + kh - padding;
                        for (int32_t kw = 0; kw < 3; kw++) {
                            int32_t iw = ow + kw - padding;
                            float v = 0.0f;
                            if ((ih >= 0) && (ih < in_height) && (iw >= 0) && (iw < in_width)) {
                                v = in_c[ih * in_width + iw];
                            }
                            patch[kidx++] = v;
                        }
                    }
                }

                const int32_t m = oh * out_width + ow;
                int32_t oc0 = 0;
                while (oc0 < out_channels) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));
                    vfloat32m4_t vacc = __riscv_vle32_v_f32m4(bias->f_data + oc0, vl);

                    for (int32_t k = 0; k < K; k++) {
                        const float *wcol = weights->f_data + oc0 * K + k;
                        vfloat32m4_t vw = __riscv_vlse32_v_f32m4(wcol, (ptrdiff_t)(K * (int32_t)sizeof(float)), vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, patch[k], vw, vl);
                    }

                    if (inv_scale != 1.0f) {
                        vacc = __riscv_vfmul_vf_f32m4(vacc, inv_scale, vl);
                    }

                    float *dst = out_n + oc0 * out_hw + m;
                    __riscv_vsse32_v_f32m4(dst, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc, vl);
                    oc0 += (int32_t)vl;
                }
            }
        }
    }

    free(patch);
    return 1;
}
#endif

Tensor batchnorm2d(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *scale, Tensor *mean, Tensor *variance) {
    int32_t C = input->shape[1];
    int32_t H = input->shape[2];
    int32_t W = input->shape[3];

    Tensor output = f_create_tensor(input->shape, 4);
    float out_scale = decode_packed_float(tensor_get_value(scale, 0));
    if (fabsf(out_scale) < 1e-12f) {
        out_scale = 1.0f;
    }

    for (int32_t n = 0; n < input->shape[0]; n++) {
        for (int32_t c = 0; c < C; c++) {
            float g = bn_param_value(gamma, c);
            float b = bn_param_value(beta, c);
            float m = bn_param_value(mean, c);
            float v = bn_param_value(variance, c);
            if (v < 0.0f) {
                v = -v;
            }
            if (v < 1e-6f) {
                v = 1.0f;
            }
            float var_sqrt = sqrtf(v + 0.0001f);
            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    int32_t idx = n * (C * H * W) + c * (H * W) + h * W + w;
                    float x = tensor_get_value(input, idx);
                    output.f_data[idx] = (g * (x - m) / var_sqrt + b) / out_scale;
                }
            }
        }
    }

    free_tensor(input);
    return output;
}

Tensor adaptive_avg_pool2d(Tensor *input) {
    int32_t batch_size = input->shape[0];
    int32_t channels = input->shape[1];
    int32_t height = input->shape[2];
    int32_t width = input->shape[3];

    u_int8_t shape[4] = {(u_int8_t)batch_size, (u_int8_t)channels, 1, 1};
    Tensor output = f_create_tensor(shape, 4);

    for (int32_t n = 0; n < batch_size; n++) {
        for (int32_t c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int32_t h = 0; h < height; h++) {
                for (int32_t w = 0; w < width; w++) {
                    int32_t index = n * (channels * height * width) + c * (height * width) + h * width + w;
                    sum += tensor_get_value(input, index);
                }
            }
            int32_t out_index = n * channels + c;
            output.f_data[out_index] = sum / (float)(height * width);
        }
    }

    return output;
}

Tensor conv2d(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale, u_int8_t stride, u_int8_t padding) {
    int32_t batch_size = input->shape[0];
    int32_t in_height = input->shape[2];
    int32_t in_width = input->shape[3];
    int32_t out_channels = weights->shape[0];
    int32_t kernel_height = weights->shape[2];
    int32_t kernel_width = weights->shape[3];

    int32_t out_height = (in_height + (2 * padding) - kernel_height) / stride + 1;
    int32_t out_width = (in_width + (2 * padding) - kernel_width) / stride + 1;

    u_int8_t output_shape[4] = {
        (u_int8_t)batch_size,
        (u_int8_t)out_channels,
        (u_int8_t)out_height,
        (u_int8_t)out_width,
    };

    Tensor output = f_create_tensor(output_shape, 4);
    float out_scale = decode_packed_float(tensor_get_value(scale, 0));
    if (fabsf(out_scale) < 1e-12f) {
        out_scale = 1.0f;
    }

#if defined(__riscv_vector)
    int handled = 0;
    if ((input->f_data != NULL) && (weights->f_data != NULL) && (bias->f_data != NULL) &&
        (stride == 1) && (padding == 1) && (kernel_height == 3) && (kernel_width == 3)) {
        handled = conv2d_rvv_implicit_gemm_impl(input, weights, bias, out_scale, (int32_t)padding, &output);
    }
    if (!handled) {
        conv2d_scalar_impl(input, weights, bias, out_scale, (int32_t)stride, (int32_t)padding, &output);
    }
#else
    conv2d_scalar_impl(input, weights, bias, out_scale, (int32_t)stride, (int32_t)padding, &output);
#endif

    free_tensor(input);
    return output;
}

Tensor fc_layer(Tensor *input, Tensor *weights) {
    int32_t batch_size = input->shape[0];
    int32_t input_features = input->shape[1];
    int32_t output_features = weights->shape[0];

    u_int8_t shape[2] = {(u_int8_t)batch_size, (u_int8_t)output_features};
    Tensor output = f_create_tensor(shape, 2);

#if defined(__riscv_vector)
    if ((input->f_data != NULL) && (weights->f_data != NULL)) {
        for (int32_t n = 0; n < batch_size; n++) {
            const float *in_ptr = input->f_data + n * input_features;
            for (int32_t o = 0; o < output_features; o++) {
                const float *w_ptr = weights->f_data + o * input_features;
                int32_t i = 0;
                vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                while (i < input_features) {
                    size_t vl = __riscv_vsetvl_e32m8((size_t)(input_features - i));
                    vfloat32m8_t va = __riscv_vle32_v_f32m8(in_ptr + i, vl);
                    vfloat32m8_t vb = __riscv_vle32_v_f32m8(w_ptr + i, vl);
                    vfloat32m8_t vm = __riscv_vfmul_vv_f32m8(va, vb, vl);
                    vsum = __riscv_vfredusum_vs_f32m8_f32m1(vm, vsum, vl);
                    i += (int32_t)vl;
                }
                output.f_data[n * output_features + o] = __riscv_vfmv_f_s_f32m1_f32(vsum);
            }
        }
        return output;
    }
#endif

    for (int32_t n = 0; n < batch_size; n++) {
        for (int32_t o = 0; o < output_features; o++) {
            float sum = 0.0f;
            for (int32_t i = 0; i < input_features; i++) {
                int32_t in_idx = n * input_features + i;
                int32_t w_idx = o * input_features + i;
                sum += tensor_get_value(input, in_idx) * tensor_get_value(weights, w_idx);
            }
            output.f_data[n * output_features + o] = sum;
        }
    }

    return output;
}

Tensor maxpool2d(Tensor *input, int kernel_size, int stride) {
    u_int8_t shape[4] = {
        input->shape[0],
        input->shape[1],
        (u_int8_t)(((int32_t)input->shape[2] - kernel_size) / stride + 1),
        (u_int8_t)(((int32_t)input->shape[3] - kernel_size) / stride + 1),
    };

    Tensor output = (input->data != NULL) ? create_tensor(shape, 4) : f_create_tensor(shape, 4);

    for (int32_t b = 0; b < output.shape[0]; b++) {
        for (int32_t c = 0; c < output.shape[1]; c++) {
            for (int32_t oh = 0; oh < output.shape[2]; oh++) {
                for (int32_t ow = 0; ow < output.shape[3]; ow++) {
                    int32_t output_index = b * (output.shape[1] * output.shape[2] * output.shape[3]) +
                                           c * (output.shape[2] * output.shape[3]) +
                                           oh * output.shape[3] + ow;

                    if (input->data != NULL) {
                        int8_t max_value = INT8_MIN;
                        for (int32_t kh = 0; kh < kernel_size; kh++) {
                            for (int32_t kw = 0; kw < kernel_size; kw++) {
                                int32_t ih = oh * stride + kh;
                                int32_t iw = ow * stride + kw;
                                int32_t input_index = b * (input->shape[1] * input->shape[2] * input->shape[3]) +
                                                      c * (input->shape[2] * input->shape[3]) +
                                                      ih * input->shape[3] + iw;
                                if (input->data[input_index] > max_value) {
                                    max_value = input->data[input_index];
                                }
                            }
                        }
                        output.data[output_index] = max_value;
                    } else {
                        float max_value = -FLT_MAX;
                        for (int32_t kh = 0; kh < kernel_size; kh++) {
                            for (int32_t kw = 0; kw < kernel_size; kw++) {
                                int32_t ih = oh * stride + kh;
                                int32_t iw = ow * stride + kw;
                                int32_t input_index = b * (input->shape[1] * input->shape[2] * input->shape[3]) +
                                                      c * (input->shape[2] * input->shape[3]) +
                                                      ih * input->shape[3] + iw;
                                if (input->f_data[input_index] > max_value) {
                                    max_value = input->f_data[input_index];
                                }
                            }
                        }
                        output.f_data[output_index] = max_value;
                    }
                }
            }
        }
    }

    return output;
}

void softmax(Tensor *input) {
    int32_t batch_size = input->shape[0];
    int32_t num_classes = input->shape[1];

    if (input->f_data == NULL) {
        input->f_data = (float *)malloc((size_t)input->size * sizeof(float));
        for (int32_t i = 0; i < input->size; i++) {
            input->f_data[i] = (float)input->data[i];
        }
    }

    for (int32_t n = 0; n < batch_size; n++) {
        float max_val = -FLT_MAX;
        for (int32_t c = 0; c < num_classes; c++) {
            int32_t index = n * num_classes + c;
            if (input->f_data[index] > max_val) {
                max_val = input->f_data[index];
            }
        }

        float sum_exp = 0.0f;
        for (int32_t c = 0; c < num_classes; c++) {
            int32_t index = n * num_classes + c;
            input->f_data[index] = expf(input->f_data[index] - max_val);
            sum_exp += input->f_data[index];
        }

        for (int32_t c = 0; c < num_classes; c++) {
            int32_t index = n * num_classes + c;
            input->f_data[index] /= sum_exp;
        }
    }
}

Tensor upsample_nearest(Tensor *input, int8_t scale_factor) {
    u_int8_t shape[4] = {
        input->shape[0],
        input->shape[1],
        (u_int8_t)(input->shape[2] * scale_factor),
        (u_int8_t)(input->shape[3] * scale_factor),
    };

    Tensor output = (input->data != NULL) ? create_tensor(shape, 4) : f_create_tensor(shape, 4);

    for (int32_t b = 0; b < output.shape[0]; b++) {
        for (int32_t c = 0; c < output.shape[1]; c++) {
            for (int32_t h = 0; h < output.shape[2]; h++) {
                int32_t nearest_h = h / scale_factor;
                for (int32_t w = 0; w < output.shape[3]; w++) {
                    int32_t nearest_w = w / scale_factor;

                    int32_t input_index = b * (input->shape[1] * input->shape[2] * input->shape[3]) +
                                          c * (input->shape[2] * input->shape[3]) +
                                          nearest_h * input->shape[3] + nearest_w;

                    int32_t output_index = b * (output.shape[1] * output.shape[2] * output.shape[3]) +
                                           c * (output.shape[2] * output.shape[3]) +
                                           h * output.shape[3] + w;

                    if (input->data != NULL) {
                        output.data[output_index] = input->data[input_index];
                    } else {
                        output.f_data[output_index] = input->f_data[input_index];
                    }
                }
            }
        }
    }

    return output;
}
