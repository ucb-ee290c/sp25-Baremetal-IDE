#include "modules.h"
#include "misc.h"

#include <float.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#ifndef TINYSPEECH_CONV_FUSE_RELU
#define TINYSPEECH_CONV_FUSE_RELU 0
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
                    float y = sum / out_scale;
#if TINYSPEECH_CONV_FUSE_RELU
                    if (y < 0.0f) {
                        y = 0.0f;
                    }
#endif
                    output->f_data[out_index] = y;
                }
            }
        }
    }
}

#if defined(__riscv_vector)
#define TINYSPEECH_PACK_CONV1_SIZE (24 * 1 * 9)
#define TINYSPEECH_PACK_CONV2_SIZE (48 * 24 * 9)
#define TINYSPEECH_PACK_CONV3_SIZE (96 * 48 * 9)
#define TINYSPEECH_PADDED_IN_MAX_FLOATS (48 * 14 * 96)

static float g_conv1_pack[TINYSPEECH_PACK_CONV1_SIZE];
static float g_conv2_pack[TINYSPEECH_PACK_CONV2_SIZE];
static float g_conv3_pack[TINYSPEECH_PACK_CONV3_SIZE];
static float g_padded_in_buf[TINYSPEECH_PADDED_IN_MAX_FLOATS];

static const float *g_conv1_src = NULL;
static const float *g_conv2_src = NULL;
static const float *g_conv3_src = NULL;

static int build_padded_input_f32(const float *src_n,
                                  int32_t in_channels,
                                  int32_t in_height,
                                  int32_t in_width,
                                  int32_t padding,
                                  float **dst_out,
                                  int32_t *pad_h_out,
                                  int32_t *pad_w_out) {
    const int32_t pad_h = in_height + (2 * padding);
    const int32_t pad_w = in_width + (2 * padding);
    const int32_t pad_hw = pad_h * pad_w;
    const int32_t total = in_channels * pad_hw;
    if (total > TINYSPEECH_PADDED_IN_MAX_FLOATS) {
        return 0;
    }

    float *dst = g_padded_in_buf;
    memset(dst, 0, (size_t)total * sizeof(float));

    for (int32_t ic = 0; ic < in_channels; ic++) {
        const float *src_c = src_n + ic * (in_height * in_width);
        float *dst_c = dst + ic * pad_hw;
        for (int32_t ih = 0; ih < in_height; ih++) {
            memcpy(dst_c + (ih + padding) * pad_w + padding,
                   src_c + ih * in_width,
                   (size_t)in_width * sizeof(float));
        }
    }

    *dst_out = dst;
    *pad_h_out = pad_h;
    *pad_w_out = pad_w;
    return 1;
}

static inline void pack_oc_major_to_k_major(const float *src,
                                            float *dst,
                                            int32_t out_channels,
                                            int32_t K) {
    for (int32_t k = 0; k < K; k++) {
        for (int32_t oc = 0; oc < out_channels; oc++) {
            dst[k * out_channels + oc] = src[oc * K + k];
        }
    }
}

static const float *get_packed_conv_weights(const Tensor *weights,
                                            int32_t out_channels,
                                            int32_t K) {
    const float *src = weights->f_data;
    if (src == NULL) {
        return NULL;
    }

    if ((out_channels == 24) && (K == 9)) {
        if (g_conv1_src != src) {
            pack_oc_major_to_k_major(src, g_conv1_pack, out_channels, K);
            g_conv1_src = src;
        }
        return g_conv1_pack;
    }

    if ((out_channels == 48) && (K == 216)) {
        if (g_conv2_src != src) {
            pack_oc_major_to_k_major(src, g_conv2_pack, out_channels, K);
            g_conv2_src = src;
        }
        return g_conv2_pack;
    }

    if ((out_channels == 96) && (K == 432)) {
        if (g_conv3_src != src) {
            pack_oc_major_to_k_major(src, g_conv3_pack, out_channels, K);
            g_conv3_src = src;
        }
        return g_conv3_pack;
    }

    return NULL;
}

static inline void conv_point_rvv_border(const float *in_n,
                                         const float *wpack,
                                         const float *bias_data,
                                         float *out_n,
                                         int32_t out_channels,
                                         int32_t in_channels,
                                         int32_t in_height,
                                         int32_t in_width,
                                         int32_t out_hw,
                                         int32_t oh,
                                         int32_t ow,
                                         int32_t padding,
                                         int32_t m,
                                         float inv_scale) {
    const int32_t in_hw = in_height * in_width;

    int32_t oc0 = 0;
    while (oc0 < out_channels) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));
        vfloat32m4_t vacc = __riscv_vle32_v_f32m4(bias_data + oc0, vl);

        int32_t k = 0;
        for (int32_t ic = 0; ic < in_channels; ic++) {
            const float *in_c = in_n + ic * in_hw;
            for (int32_t kh = 0; kh < 3; kh++) {
                int32_t ih = oh + kh - padding;
                for (int32_t kw = 0; kw < 3; kw++, k++) {
                    int32_t iw = ow + kw - padding;
                    float x = 0.0f;
                    if ((ih >= 0) && (ih < in_height) && (iw >= 0) && (iw < in_width)) {
                        x = in_c[ih * in_width + iw];
                    }
                    const float *wcol = wpack + k * out_channels + oc0;
                    vfloat32m4_t vw = __riscv_vle32_v_f32m4(wcol, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, x, vw, vl);
                }
            }
        }

        if (inv_scale != 1.0f) {
            vacc = __riscv_vfmul_vf_f32m4(vacc, inv_scale, vl);
        }
#if TINYSPEECH_CONV_FUSE_RELU
        vacc = __riscv_vfmax_vf_f32m4(vacc, 0.0f, vl);
#endif

        float *dst = out_n + oc0 * out_hw + m;
        __riscv_vsse32_v_f32m4(dst, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc, vl);
        oc0 += (int32_t)vl;
    }
}

static inline void conv_point_rvv_interior(const float *in_n,
                                           const float *wpack,
                                           const float *bias_data,
                                           float *out_n,
                                           int32_t out_channels,
                                           int32_t in_channels,
                                           int32_t in_height,
                                           int32_t in_width,
                                           int32_t out_hw,
                                           int32_t oh,
                                           int32_t ow,
                                           int32_t padding,
                                           int32_t m,
                                           float inv_scale) {
    const int32_t in_hw = in_height * in_width;
    const int32_t ih0 = oh - padding;
    const int32_t iw0 = ow - padding;

    int32_t oc0 = 0;
    while (oc0 < out_channels) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));
        vfloat32m4_t vacc = __riscv_vle32_v_f32m4(bias_data + oc0, vl);

        int32_t kbase = 0;
        for (int32_t ic = 0; ic < in_channels; ic++, kbase += 9) {
            const float *p = in_n + ic * in_hw + ih0 * in_width + iw0;
            const float x00 = p[0];
            const float x01 = p[1];
            const float x02 = p[2];
            const float x10 = p[in_width + 0];
            const float x11 = p[in_width + 1];
            const float x12 = p[in_width + 2];
            const float x20 = p[(2 * in_width) + 0];
            const float x21 = p[(2 * in_width) + 1];
            const float x22 = p[(2 * in_width) + 2];

            vfloat32m4_t vw;
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 0) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x00, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 1) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x01, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 2) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x02, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 3) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x10, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 4) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x11, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 5) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x12, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 6) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x20, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 7) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x21, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 8) * out_channels + oc0, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, x22, vw, vl);
        }

        if (inv_scale != 1.0f) {
            vacc = __riscv_vfmul_vf_f32m4(vacc, inv_scale, vl);
        }
#if TINYSPEECH_CONV_FUSE_RELU
        vacc = __riscv_vfmax_vf_f32m4(vacc, 0.0f, vl);
#endif

        float *dst = out_n + oc0 * out_hw + m;
        __riscv_vsse32_v_f32m4(dst, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc, vl);
        oc0 += (int32_t)vl;
    }
}

static inline void conv_point2_rvv_interior(const float *in_n,
                                            const float *wpack,
                                            const float *bias_data,
                                            float *out_n,
                                            int32_t out_channels,
                                            int32_t in_channels,
                                            int32_t in_height,
                                            int32_t in_width,
                                            int32_t out_hw,
                                            int32_t oh,
                                            int32_t ow,
                                            int32_t padding,
                                            int32_t m0,
                                            int32_t m1,
                                            float inv_scale) {
    const int32_t in_hw = in_width * in_height;
    const int32_t ih0 = oh - padding;
    const int32_t iw0 = ow - padding;

    int32_t oc0 = 0;
    while (oc0 < out_channels) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));
        vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
        vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);

        int32_t kbase = 0;
        for (int32_t ic = 0; ic < in_channels; ic++, kbase += 9) {
            const float *p = in_n + ic * in_hw + ih0 * in_width + iw0;
            const float x00 = p[0];
            const float x01 = p[1];
            const float x02 = p[2];
            const float x03 = p[3];
            const float x10 = p[in_width + 0];
            const float x11 = p[in_width + 1];
            const float x12 = p[in_width + 2];
            const float x13 = p[in_width + 3];
            const float x20 = p[(2 * in_width) + 0];
            const float x21 = p[(2 * in_width) + 1];
            const float x22 = p[(2 * in_width) + 2];
            const float x23 = p[(2 * in_width) + 3];

            vfloat32m4_t vw;
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 0) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x00, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x01, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 1) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x01, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x02, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 2) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x02, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x03, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 3) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x10, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x11, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 4) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x11, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x12, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 5) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x12, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x13, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 6) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x20, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x21, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 7) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x21, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x22, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 8) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x22, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x23, vw, vl);
        }

        if (inv_scale != 1.0f) {
            vacc0 = __riscv_vfmul_vf_f32m4(vacc0, inv_scale, vl);
            vacc1 = __riscv_vfmul_vf_f32m4(vacc1, inv_scale, vl);
        }
#if TINYSPEECH_CONV_FUSE_RELU
        vacc0 = __riscv_vfmax_vf_f32m4(vacc0, 0.0f, vl);
        vacc1 = __riscv_vfmax_vf_f32m4(vacc1, 0.0f, vl);
#endif

        float *dst0 = out_n + oc0 * out_hw + m0;
        float *dst1 = out_n + oc0 * out_hw + m1;
        __riscv_vsse32_v_f32m4(dst0, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc0, vl);
        __riscv_vsse32_v_f32m4(dst1, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc1, vl);
        oc0 += (int32_t)vl;
    }
}

static inline void conv_point4_rvv_interior(const float *in_n,
                                            const float *wpack,
                                            const float *bias_data,
                                            float *out_n,
                                            int32_t out_channels,
                                            int32_t in_channels,
                                            int32_t in_height,
                                            int32_t in_width,
                                            int32_t out_hw,
                                            int32_t oh,
                                            int32_t ow,
                                            int32_t padding,
                                            int32_t m0,
                                            float inv_scale) {
    const int32_t in_hw = in_width * in_height;
    const int32_t ih0 = oh - padding;
    const int32_t iw0 = ow - padding;

    int32_t oc0 = 0;
    while (oc0 < out_channels) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));
        vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
        vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
        vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
        vfloat32m4_t vacc3 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);

        int32_t kbase = 0;
        for (int32_t ic = 0; ic < in_channels; ic++, kbase += 9) {
            const float *p = in_n + ic * in_hw + ih0 * in_width + iw0;
            const float x00 = p[0];
            const float x01 = p[1];
            const float x02 = p[2];
            const float x03 = p[3];
            const float x04 = p[4];
            const float x05 = p[5];
            const float x10 = p[in_width + 0];
            const float x11 = p[in_width + 1];
            const float x12 = p[in_width + 2];
            const float x13 = p[in_width + 3];
            const float x14 = p[in_width + 4];
            const float x15 = p[in_width + 5];
            const float x20 = p[(2 * in_width) + 0];
            const float x21 = p[(2 * in_width) + 1];
            const float x22 = p[(2 * in_width) + 2];
            const float x23 = p[(2 * in_width) + 3];
            const float x24 = p[(2 * in_width) + 4];
            const float x25 = p[(2 * in_width) + 5];

            vfloat32m4_t vw;
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 0) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x00, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x01, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x02, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x03, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 1) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x01, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x02, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x03, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x04, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 2) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x02, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x03, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x04, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x05, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 3) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x10, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x11, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x12, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x13, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 4) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x11, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x12, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x13, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x14, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 5) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x12, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x13, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x14, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x15, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 6) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x20, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x21, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x22, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x23, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 7) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x21, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x22, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x23, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x24, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 8) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x22, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x23, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x24, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x25, vw, vl);
        }

        if (inv_scale != 1.0f) {
            vacc0 = __riscv_vfmul_vf_f32m4(vacc0, inv_scale, vl);
            vacc1 = __riscv_vfmul_vf_f32m4(vacc1, inv_scale, vl);
            vacc2 = __riscv_vfmul_vf_f32m4(vacc2, inv_scale, vl);
            vacc3 = __riscv_vfmul_vf_f32m4(vacc3, inv_scale, vl);
        }
#if TINYSPEECH_CONV_FUSE_RELU
        vacc0 = __riscv_vfmax_vf_f32m4(vacc0, 0.0f, vl);
        vacc1 = __riscv_vfmax_vf_f32m4(vacc1, 0.0f, vl);
        vacc2 = __riscv_vfmax_vf_f32m4(vacc2, 0.0f, vl);
        vacc3 = __riscv_vfmax_vf_f32m4(vacc3, 0.0f, vl);
#endif

        float *dst0 = out_n + oc0 * out_hw + m0;
        float *dst1 = dst0 + 1;
        float *dst2 = dst0 + 2;
        float *dst3 = dst0 + 3;
        __riscv_vsse32_v_f32m4(dst0, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc0, vl);
        __riscv_vsse32_v_f32m4(dst1, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc1, vl);
        __riscv_vsse32_v_f32m4(dst2, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc2, vl);
        __riscv_vsse32_v_f32m4(dst3, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc3, vl);
        oc0 += (int32_t)vl;
    }
}

static inline void conv_point4_rvv_interior_spec_ic(const float *in_n,
                                                    const float *wpack,
                                                    const float *bias_data,
                                                    float *out_n,
                                                    int32_t out_channels,
                                                    int32_t in_height,
                                                    int32_t in_width,
                                                    int32_t out_hw,
                                                    int32_t oh,
                                                    int32_t ow,
                                                    int32_t padding,
                                                    int32_t m0,
                                                    float inv_scale,
                                                    int32_t fixed_in_channels) {
    const int32_t in_hw = in_width * in_height;
    const int32_t ih0 = oh - padding;
    const int32_t iw0 = ow - padding;

    int32_t oc0 = 0;
    while (oc0 < out_channels) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));
        vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
        vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
        vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
        vfloat32m4_t vacc3 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);

        int32_t kbase = 0;
        for (int32_t ic = 0; ic < fixed_in_channels; ic++, kbase += 9) {
            const float *p = in_n + ic * in_hw + ih0 * in_width + iw0;
            const float x00 = p[0];
            const float x01 = p[1];
            const float x02 = p[2];
            const float x03 = p[3];
            const float x04 = p[4];
            const float x05 = p[5];
            const float x10 = p[in_width + 0];
            const float x11 = p[in_width + 1];
            const float x12 = p[in_width + 2];
            const float x13 = p[in_width + 3];
            const float x14 = p[in_width + 4];
            const float x15 = p[in_width + 5];
            const float x20 = p[(2 * in_width) + 0];
            const float x21 = p[(2 * in_width) + 1];
            const float x22 = p[(2 * in_width) + 2];
            const float x23 = p[(2 * in_width) + 3];
            const float x24 = p[(2 * in_width) + 4];
            const float x25 = p[(2 * in_width) + 5];

            vfloat32m4_t vw;
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 0) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x00, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x01, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x02, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x03, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 1) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x01, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x02, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x03, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x04, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 2) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x02, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x03, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x04, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x05, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 3) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x10, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x11, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x12, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x13, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 4) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x11, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x12, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x13, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x14, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 5) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x12, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x13, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x14, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x15, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 6) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x20, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x21, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x22, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x23, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 7) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x21, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x22, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x23, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x24, vw, vl);
            vw = __riscv_vle32_v_f32m4(wpack + (kbase + 8) * out_channels + oc0, vl);
            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x22, vw, vl);
            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x23, vw, vl);
            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x24, vw, vl);
            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x25, vw, vl);
        }

        if (inv_scale != 1.0f) {
            vacc0 = __riscv_vfmul_vf_f32m4(vacc0, inv_scale, vl);
            vacc1 = __riscv_vfmul_vf_f32m4(vacc1, inv_scale, vl);
            vacc2 = __riscv_vfmul_vf_f32m4(vacc2, inv_scale, vl);
            vacc3 = __riscv_vfmul_vf_f32m4(vacc3, inv_scale, vl);
        }
#if TINYSPEECH_CONV_FUSE_RELU
        vacc0 = __riscv_vfmax_vf_f32m4(vacc0, 0.0f, vl);
        vacc1 = __riscv_vfmax_vf_f32m4(vacc1, 0.0f, vl);
        vacc2 = __riscv_vfmax_vf_f32m4(vacc2, 0.0f, vl);
        vacc3 = __riscv_vfmax_vf_f32m4(vacc3, 0.0f, vl);
#endif

        float *dst0 = out_n + oc0 * out_hw + m0;
        float *dst1 = dst0 + 1;
        float *dst2 = dst0 + 2;
        float *dst3 = dst0 + 3;
        __riscv_vsse32_v_f32m4(dst0, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc0, vl);
        __riscv_vsse32_v_f32m4(dst1, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc1, vl);
        __riscv_vsse32_v_f32m4(dst2, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc2, vl);
        __riscv_vsse32_v_f32m4(dst3, (ptrdiff_t)(out_hw * (int32_t)sizeof(float)), vacc3, vl);
        oc0 += (int32_t)vl;
    }
}

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
    const int32_t out_hw = out_height * out_width;
    const int32_t K = in_channels * 9;
    const float inv_scale = 1.0f / out_scale;
    const float *wpack = get_packed_conv_weights(weights, out_channels, K);

    if (wpack == NULL) {
        return 0;
    }
    int32_t layer_kind = 0;
    if ((in_channels == 1) && (out_channels == 24) && (K == 9)) {
        layer_kind = 1;
    } else if ((in_channels == 24) && (out_channels == 48) && (K == 216)) {
        layer_kind = 2;
    } else if ((in_channels == 48) && (out_channels == 96) && (K == 432)) {
        layer_kind = 3;
    }

    for (int32_t n = 0; n < batch_size; n++) {
        const float *in_src_n = input->f_data + n * (in_channels * in_height * in_width);
        float *in_n = NULL;
        int32_t pad_h = 0;
        int32_t pad_w = 0;
        if (!build_padded_input_f32(in_src_n, in_channels, in_height, in_width, (int32_t)padding,
                                    &in_n, &pad_h, &pad_w)) {
            return 0;
        }
        float *out_n = output->f_data + n * (out_channels * out_hw);

        for (int32_t oh = 0; oh < out_height; oh++) {
            int32_t ow = 0;
            for (; (ow + 3) < out_width; ow += 4) {
                const int32_t m0 = oh * out_width + ow;
                if (layer_kind == 1) {
                    conv_point4_rvv_interior_spec_ic(in_n, wpack, bias->f_data, out_n,
                                                     out_channels, pad_h, pad_w,
                                                     out_hw, oh, ow, 0, m0, inv_scale, 1);
                } else if (layer_kind == 2) {
                    conv_point4_rvv_interior_spec_ic(in_n, wpack, bias->f_data, out_n,
                                                     out_channels, pad_h, pad_w,
                                                     out_hw, oh, ow, 0, m0, inv_scale, 24);
                } else if (layer_kind == 3) {
                    conv_point4_rvv_interior_spec_ic(in_n, wpack, bias->f_data, out_n,
                                                     out_channels, pad_h, pad_w,
                                                     out_hw, oh, ow, 0, m0, inv_scale, 48);
                } else {
                    conv_point4_rvv_interior(in_n, wpack, bias->f_data, out_n,
                                             out_channels, in_channels, pad_h, pad_w,
                                             out_hw, oh, ow, 0, m0, inv_scale);
                }
            }

            for (; (ow + 1) < out_width; ow += 2) {
                const int32_t m0 = oh * out_width + ow;
                const int32_t m1 = m0 + 1;
                conv_point2_rvv_interior(in_n, wpack, bias->f_data, out_n,
                                         out_channels, in_channels, pad_h, pad_w,
                                         out_hw, oh, ow, 0, m0, m1, inv_scale);
            }

            for (; ow < out_width; ow++) {
                const int32_t m = oh * out_width + ow;
                conv_point_rvv_interior(in_n, wpack, bias->f_data, out_n,
                                        out_channels, in_channels, pad_h, pad_w,
                                        out_hw, oh, ow, 0, m, inv_scale);
            }
        }
    }
    return 1;
}

static inline int conv_point_is_interior(int32_t oh, int32_t ow,
                                         int32_t in_height, int32_t in_width,
                                         int32_t padding) {
    return (oh >= padding) && (oh < (in_height + padding - 2)) &&
           (ow >= padding) && (ow < (in_width + padding - 2));
}

static inline vfloat32m4_t conv_block_rvv_border(const float *in_n,
                                                 const float *wpack,
                                                 const float *bias_data,
                                                 int32_t out_channels,
                                                 int32_t in_channels,
                                                 int32_t in_height,
                                                 int32_t in_width,
                                                 int32_t oh,
                                                 int32_t ow,
                                                 int32_t padding,
                                                 int32_t oc0,
                                                 size_t vl,
                                                 float inv_scale) {
    const int32_t in_hw = in_height * in_width;
    vfloat32m4_t vacc = __riscv_vle32_v_f32m4(bias_data + oc0, vl);

    int32_t k = 0;
    for (int32_t ic = 0; ic < in_channels; ic++) {
        const float *in_c = in_n + ic * in_hw;
        for (int32_t kh = 0; kh < 3; kh++) {
            int32_t ih = oh + kh - padding;
            for (int32_t kw = 0; kw < 3; kw++, k++) {
                int32_t iw = ow + kw - padding;
                float x = 0.0f;
                if ((ih >= 0) && (ih < in_height) && (iw >= 0) && (iw < in_width)) {
                    x = in_c[ih * in_width + iw];
                }
                vfloat32m4_t vw = __riscv_vle32_v_f32m4(wpack + k * out_channels + oc0, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, x, vw, vl);
            }
        }
    }

    if (inv_scale != 1.0f) {
        vacc = __riscv_vfmul_vf_f32m4(vacc, inv_scale, vl);
    }
#if TINYSPEECH_CONV_FUSE_RELU
    vacc = __riscv_vfmax_vf_f32m4(vacc, 0.0f, vl);
#endif
    return vacc;
}

static inline vfloat32m4_t conv_block_rvv_interior(const float *in_n,
                                                   const float *wpack,
                                                   const float *bias_data,
                                                   int32_t out_channels,
                                                   int32_t in_channels,
                                                   int32_t in_height,
                                                   int32_t in_width,
                                                   int32_t oh,
                                                   int32_t ow,
                                                   int32_t padding,
                                                   int32_t oc0,
                                                   size_t vl,
                                                   float inv_scale) {
    const int32_t in_hw = in_height * in_width;
    const int32_t ih0 = oh - padding;
    const int32_t iw0 = ow - padding;
    vfloat32m4_t vacc = __riscv_vle32_v_f32m4(bias_data + oc0, vl);

    int32_t kbase = 0;
    for (int32_t ic = 0; ic < in_channels; ic++, kbase += 9) {
        const float *p = in_n + ic * in_hw + ih0 * in_width + iw0;
        const float x00 = p[0];
        const float x01 = p[1];
        const float x02 = p[2];
        const float x10 = p[in_width + 0];
        const float x11 = p[in_width + 1];
        const float x12 = p[in_width + 2];
        const float x20 = p[(2 * in_width) + 0];
        const float x21 = p[(2 * in_width) + 1];
        const float x22 = p[(2 * in_width) + 2];

        vfloat32m4_t vw;
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 0) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x00, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 1) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x01, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 2) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x02, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 3) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x10, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 4) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x11, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 5) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x12, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 6) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x20, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 7) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x21, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 8) * out_channels + oc0, vl);
        vacc = __riscv_vfmacc_vf_f32m4(vacc, x22, vw, vl);
    }

    if (inv_scale != 1.0f) {
        vacc = __riscv_vfmul_vf_f32m4(vacc, inv_scale, vl);
    }
#if TINYSPEECH_CONV_FUSE_RELU
    vacc = __riscv_vfmax_vf_f32m4(vacc, 0.0f, vl);
#endif
    return vacc;
}

static inline vfloat32m4_t conv_gap_block4_rvv_interior_48ic(const float *in_n,
                                                              const float *wpack,
                                                              const float *bias_data,
                                                              int32_t out_channels,
                                                              int32_t in_height,
                                                              int32_t in_width,
                                                              int32_t oh,
                                                              int32_t ow,
                                                              int32_t oc0,
                                                              size_t vl,
                                                              float inv_scale) {
    const int32_t in_hw = in_height * in_width;
    const int32_t ih0 = oh;
    const int32_t iw0 = ow;
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(bias_data + oc0, vl);
    vfloat32m4_t vacc1 = vacc0;
    vfloat32m4_t vacc2 = vacc0;
    vfloat32m4_t vacc3 = vacc0;

    int32_t kbase = 0;
    for (int32_t ic = 0; ic < 48; ic++, kbase += 9) {
        const float *p = in_n + ic * in_hw + ih0 * in_width + iw0;
        const float x00 = p[0];
        const float x01 = p[1];
        const float x02 = p[2];
        const float x03 = p[3];
        const float x04 = p[4];
        const float x05 = p[5];
        const float x10 = p[in_width + 0];
        const float x11 = p[in_width + 1];
        const float x12 = p[in_width + 2];
        const float x13 = p[in_width + 3];
        const float x14 = p[in_width + 4];
        const float x15 = p[in_width + 5];
        const float x20 = p[(2 * in_width) + 0];
        const float x21 = p[(2 * in_width) + 1];
        const float x22 = p[(2 * in_width) + 2];
        const float x23 = p[(2 * in_width) + 3];
        const float x24 = p[(2 * in_width) + 4];
        const float x25 = p[(2 * in_width) + 5];

        vfloat32m4_t vw;
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 0) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x00, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x01, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x02, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x03, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 1) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x01, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x02, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x03, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x04, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 2) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x02, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x03, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x04, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x05, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 3) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x10, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x11, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x12, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x13, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 4) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x11, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x12, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x13, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x14, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 5) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x12, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x13, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x14, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x15, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 6) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x20, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x21, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x22, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x23, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 7) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x21, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x22, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x23, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x24, vw, vl);
        vw = __riscv_vle32_v_f32m4(wpack + (kbase + 8) * out_channels + oc0, vl);
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, x22, vw, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, x23, vw, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, x24, vw, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, x25, vw, vl);
    }

    if (inv_scale != 1.0f) {
        vacc0 = __riscv_vfmul_vf_f32m4(vacc0, inv_scale, vl);
        vacc1 = __riscv_vfmul_vf_f32m4(vacc1, inv_scale, vl);
        vacc2 = __riscv_vfmul_vf_f32m4(vacc2, inv_scale, vl);
        vacc3 = __riscv_vfmul_vf_f32m4(vacc3, inv_scale, vl);
    }
#if TINYSPEECH_CONV_FUSE_RELU
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, 0.0f, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, 0.0f, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, 0.0f, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, 0.0f, vl);
#endif
    vfloat32m4_t vsum = __riscv_vfadd_vv_f32m4(vacc0, vacc1, vl);
    vsum = __riscv_vfadd_vv_f32m4(vsum, vacc2, vl);
    vsum = __riscv_vfadd_vv_f32m4(vsum, vacc3, vl);
    return vsum;
}

static int conv2d_relu_maxpool2d_rvv_impl(const Tensor *input,
                                          const Tensor *weights,
                                          const Tensor *bias,
                                          float out_scale,
                                          int32_t padding,
                                          int32_t pool_kernel_size,
                                          int32_t pool_stride,
                                          Tensor *output) {
    const int32_t batch_size = input->shape[0];
    const int32_t in_channels = input->shape[1];
    const int32_t in_height = input->shape[2];
    const int32_t in_width = input->shape[3];
    const int32_t out_channels = weights->shape[0];
    const int32_t pool_out_h = output->shape[2];
    const int32_t pool_out_w = output->shape[3];
    const int32_t pool_out_hw = pool_out_h * pool_out_w;
    const int32_t K = in_channels * 9;
    const float inv_scale = 1.0f / out_scale;
    const float *wpack = get_packed_conv_weights(weights, out_channels, K);
    if (wpack == NULL) {
        return 0;
    }
    if ((pool_kernel_size != 2) || (pool_stride != 2)) {
        return 0;
    }

    for (int32_t n = 0; n < batch_size; n++) {
        const float *in_src_n = input->f_data + n * (in_channels * in_height * in_width);
        float *in_n = NULL;
        int32_t pad_h = 0;
        int32_t pad_w = 0;
        if (!build_padded_input_f32(in_src_n, in_channels, in_height, in_width, padding,
                                    &in_n, &pad_h, &pad_w)) {
            return 0;
        }
        float *out_n = output->f_data + n * (out_channels * pool_out_hw);

        for (int32_t oph = 0; oph < pool_out_h; oph++) {
            const int32_t oh0 = oph * pool_stride;
            const int32_t oh1 = oh0 + 1;
            for (int32_t opw = 0; opw < pool_out_w; opw++) {
                const int32_t ow0 = opw * pool_stride;
                const int32_t ow1 = ow0 + 1;
                const int32_t m = oph * pool_out_w + opw;

                int32_t oc0 = 0;
                while (oc0 < out_channels) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));

                    vfloat32m4_t v00 = conv_block_rvv_interior(in_n, wpack, bias->f_data, out_channels, in_channels,
                                                               pad_h, pad_w, oh0, ow0, 0, oc0, vl, inv_scale);
                    vfloat32m4_t v01 = conv_block_rvv_interior(in_n, wpack, bias->f_data, out_channels, in_channels,
                                                               pad_h, pad_w, oh0, ow1, 0, oc0, vl, inv_scale);
                    vfloat32m4_t v10 = conv_block_rvv_interior(in_n, wpack, bias->f_data, out_channels, in_channels,
                                                               pad_h, pad_w, oh1, ow0, 0, oc0, vl, inv_scale);
                    vfloat32m4_t v11 = conv_block_rvv_interior(in_n, wpack, bias->f_data, out_channels, in_channels,
                                                               pad_h, pad_w, oh1, ow1, 0, oc0, vl, inv_scale);

                    vfloat32m4_t vmax = __riscv_vfmax_vv_f32m4(v00, v01, vl);
                    vmax = __riscv_vfmax_vv_f32m4(vmax, v10, vl);
                    vmax = __riscv_vfmax_vv_f32m4(vmax, v11, vl);

                    float *dst = out_n + oc0 * pool_out_hw + m;
                    __riscv_vsse32_v_f32m4(dst, (ptrdiff_t)(pool_out_hw * (int32_t)sizeof(float)), vmax, vl);
                    oc0 += (int32_t)vl;
                }
            }
        }
    }
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

#if defined(__riscv_vector)
    if (input->f_data != NULL) {
        const int32_t hw = height * width;
        for (int32_t n = 0; n < batch_size; n++) {
            const float *in_n = input->f_data + n * (channels * hw);
            for (int32_t c = 0; c < channels; c++) {
                const float *src = in_n + c * hw;
                int32_t i = 0;
                vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                while (i < hw) {
                    size_t vl = __riscv_vsetvl_e32m8((size_t)(hw - i));
                    vfloat32m8_t vx = __riscv_vle32_v_f32m8(src + i, vl);
                    acc = __riscv_vfredusum_vs_f32m8_f32m1(vx, acc, vl);
                    i += (int32_t)vl;
                }
                float sum = __riscv_vfmv_f_s_f32m1_f32(acc);
                int32_t out_index = n * channels + c;
                output.f_data[out_index] = sum / (float)hw;
            }
        }
        return output;
    }
#endif

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

Tensor conv2d_relu_maxpool2d(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale,
                             u_int8_t stride, u_int8_t padding,
                             int pool_kernel_size, int pool_stride) {
    const int32_t batch_size = input->shape[0];
    const int32_t in_height = input->shape[2];
    const int32_t in_width = input->shape[3];
    const int32_t out_channels = weights->shape[0];
    const int32_t kernel_height = weights->shape[2];
    const int32_t kernel_width = weights->shape[3];

    const int32_t conv_out_h = (in_height + (2 * padding) - kernel_height) / stride + 1;
    const int32_t conv_out_w = (in_width + (2 * padding) - kernel_width) / stride + 1;
    const int32_t pool_out_h = (conv_out_h - pool_kernel_size) / pool_stride + 1;
    const int32_t pool_out_w = (conv_out_w - pool_kernel_size) / pool_stride + 1;

    u_int8_t output_shape[4] = {
        (u_int8_t)batch_size,
        (u_int8_t)out_channels,
        (u_int8_t)pool_out_h,
        (u_int8_t)pool_out_w,
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
        handled = conv2d_relu_maxpool2d_rvv_impl(input, weights, bias, out_scale,
                                                 (int32_t)padding, pool_kernel_size, pool_stride, &output);
    }
    if (!handled) {
        free_tensor(&output);
        Tensor conv = conv2d(input, weights, bias, scale, stride, padding);
        Tensor pooled = maxpool2d(&conv, pool_kernel_size, pool_stride);
        free_tensor(&conv);
        return pooled;
    }
#else
    free_tensor(&output);
    Tensor conv = conv2d(input, weights, bias, scale, stride, padding);
    Tensor pooled = maxpool2d(&conv, pool_kernel_size, pool_stride);
    free_tensor(&conv);
    return pooled;
#endif

    free_tensor(input);
    return output;
}

Tensor conv2d_relu_gap(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale,
                       u_int8_t stride, u_int8_t padding) {
    const int32_t batch_size = input->shape[0];
    const int32_t in_height = input->shape[2];
    const int32_t in_width = input->shape[3];
    const int32_t out_channels = weights->shape[0];
    const int32_t kernel_height = weights->shape[2];
    const int32_t kernel_width = weights->shape[3];

    const int32_t out_height = (in_height + (2 * padding) - kernel_height) / stride + 1;
    const int32_t out_width = (in_width + (2 * padding) - kernel_width) / stride + 1;
    const int32_t out_hw = out_height * out_width;

    u_int8_t output_shape[4] = {
        (u_int8_t)batch_size,
        (u_int8_t)out_channels,
        1,
        1,
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
        const int32_t in_channels = input->shape[1];
        const int32_t K = in_channels * 9;
        const float *wpack = get_packed_conv_weights(weights, out_channels, K);
        if (wpack != NULL) {
            const float inv_scale = 1.0f / out_scale;
            const float inv_hw = 1.0f / (float)out_hw;
            int ok = 1;
            for (int32_t n = 0; n < batch_size; n++) {
                const float *in_src_n = input->f_data + n * (in_channels * in_height * in_width);
                float *in_n = NULL;
                int32_t pad_h = 0;
                int32_t pad_w = 0;
                if (!build_padded_input_f32(in_src_n, in_channels, in_height, in_width, (int32_t)padding,
                                            &in_n, &pad_h, &pad_w)) {
                    ok = 0;
                    break;
                }
                float *out_n = output.f_data + n * out_channels;
                for (int32_t oc0 = 0; oc0 < out_channels; ) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(out_channels - oc0));
                    vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                    for (int32_t oh = 0; oh < out_height; oh++) {
                        int32_t ow = 0;
                        if (in_channels == 48) {
                            for (; (ow + 3) < out_width; ow += 4) {
                                vfloat32m4_t v4 =
                                    conv_gap_block4_rvv_interior_48ic(in_n, wpack, bias->f_data,
                                                                       out_channels, pad_h, pad_w,
                                                                       oh, ow, oc0, vl, inv_scale);
                                vsum = __riscv_vfadd_vv_f32m4(vsum, v4, vl);
                            }
                        }
                        for (; ow < out_width; ow++) {
                            vfloat32m4_t v =
                                conv_block_rvv_interior(in_n, wpack, bias->f_data, out_channels, in_channels,
                                                        pad_h, pad_w, oh, ow, 0, oc0, vl, inv_scale);
#if !TINYSPEECH_CONV_FUSE_RELU
                            v = __riscv_vfmax_vf_f32m4(v, 0.0f, vl);
#endif
                            vsum = __riscv_vfadd_vv_f32m4(vsum, v, vl);
                        }
                    }
                    vsum = __riscv_vfmul_vf_f32m4(vsum, inv_hw, vl);
                    __riscv_vse32_v_f32m4(out_n + oc0, vsum, vl);
                    oc0 += (int32_t)vl;
                }
            }
            handled = ok;
        }
    }
    if (!handled) {
        free_tensor(&output);
        Tensor conv = conv2d(input, weights, bias, scale, stride, padding);
#if !TINYSPEECH_CONV_FUSE_RELU
        relu(&conv);
#endif
        Tensor pooled = adaptive_avg_pool2d(&conv);
        free_tensor(&conv);
        return pooled;
    }
#else
    free_tensor(&output);
    Tensor conv = conv2d(input, weights, bias, scale, stride, padding);
#if !TINYSPEECH_CONV_FUSE_RELU
    relu(&conv);
#endif
    Tensor pooled = adaptive_avg_pool2d(&conv);
    free_tensor(&conv);
    return pooled;
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

    if ((input->f_data != NULL) && (weights->f_data != NULL) &&
        (input_features == 96) && (output_features == 6)) {
        for (int32_t n = 0; n < batch_size; n++) {
            const float *x = input->f_data + n * 96;
            const float *w0 = weights->f_data + 0 * 96;
            const float *w1 = weights->f_data + 1 * 96;
            const float *w2 = weights->f_data + 2 * 96;
            const float *w3 = weights->f_data + 3 * 96;
            const float *w4 = weights->f_data + 4 * 96;
            const float *w5 = weights->f_data + 5 * 96;

            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            float s4 = 0.0f;
            float s5 = 0.0f;

            for (int32_t i = 0; i < 96; i += 4) {
                const float x0 = x[i + 0];
                const float x1 = x[i + 1];
                const float x2 = x[i + 2];
                const float x3 = x[i + 3];

                s0 = fmaf(x0, w0[i + 0], s0);
                s0 = fmaf(x1, w0[i + 1], s0);
                s0 = fmaf(x2, w0[i + 2], s0);
                s0 = fmaf(x3, w0[i + 3], s0);

                s1 = fmaf(x0, w1[i + 0], s1);
                s1 = fmaf(x1, w1[i + 1], s1);
                s1 = fmaf(x2, w1[i + 2], s1);
                s1 = fmaf(x3, w1[i + 3], s1);

                s2 = fmaf(x0, w2[i + 0], s2);
                s2 = fmaf(x1, w2[i + 1], s2);
                s2 = fmaf(x2, w2[i + 2], s2);
                s2 = fmaf(x3, w2[i + 3], s2);

                s3 = fmaf(x0, w3[i + 0], s3);
                s3 = fmaf(x1, w3[i + 1], s3);
                s3 = fmaf(x2, w3[i + 2], s3);
                s3 = fmaf(x3, w3[i + 3], s3);

                s4 = fmaf(x0, w4[i + 0], s4);
                s4 = fmaf(x1, w4[i + 1], s4);
                s4 = fmaf(x2, w4[i + 2], s4);
                s4 = fmaf(x3, w4[i + 3], s4);

                s5 = fmaf(x0, w5[i + 0], s5);
                s5 = fmaf(x1, w5[i + 1], s5);
                s5 = fmaf(x2, w5[i + 2], s5);
                s5 = fmaf(x3, w5[i + 3], s5);
            }

            float *out = output.f_data + n * 6;
            out[0] = s0;
            out[1] = s1;
            out[2] = s2;
            out[3] = s3;
            out[4] = s4;
            out[5] = s5;
        }
        return output;
    }

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

#if defined(__riscv_vector)
    if ((input->f_data != NULL) && (kernel_size == 2) && (stride == 2)) {
        const ptrdiff_t in_stride_bytes = (ptrdiff_t)(2 * (int32_t)sizeof(float));
        const int32_t in_h = input->shape[2];
        const int32_t in_w = input->shape[3];
        const int32_t out_h = output.shape[2];
        const int32_t out_w = output.shape[3];

        for (int32_t b = 0; b < output.shape[0]; b++) {
            for (int32_t c = 0; c < output.shape[1]; c++) {
                const float *in_base =
                    input->f_data +
                    b * (input->shape[1] * in_h * in_w) +
                    c * (in_h * in_w);
                float *out_base =
                    output.f_data +
                    b * (output.shape[1] * out_h * out_w) +
                    c * (out_h * out_w);

                for (int32_t oh = 0; oh < out_h; oh++) {
                    const float *r0 = in_base + (2 * oh) * in_w;
                    const float *r1 = r0 + in_w;
                    float *dst = out_base + oh * out_w;

                    int32_t ow = 0;
                    while (ow < out_w) {
                        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_w - ow));
                        const float *p00 = r0 + (2 * ow);
                        const float *p01 = p00 + 1;
                        const float *p10 = r1 + (2 * ow);
                        const float *p11 = p10 + 1;

                        vfloat32m4_t v00 = __riscv_vlse32_v_f32m4(p00, in_stride_bytes, vl);
                        vfloat32m4_t v01 = __riscv_vlse32_v_f32m4(p01, in_stride_bytes, vl);
                        vfloat32m4_t v10 = __riscv_vlse32_v_f32m4(p10, in_stride_bytes, vl);
                        vfloat32m4_t v11 = __riscv_vlse32_v_f32m4(p11, in_stride_bytes, vl);

                        vfloat32m4_t vmax = __riscv_vfmax_vv_f32m4(v00, v01, vl);
                        vmax = __riscv_vfmax_vv_f32m4(vmax, v10, vl);
                        vmax = __riscv_vfmax_vv_f32m4(vmax, v11, vl);
                        __riscv_vse32_v_f32m4(dst + ow, vmax, vl);
                        ow += (int32_t)vl;
                    }
                }
            }
        }
        return output;
    }
#endif

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
