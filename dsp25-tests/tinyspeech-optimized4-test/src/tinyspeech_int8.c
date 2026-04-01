#include "tinyspeech_int8.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define TS_IN_H 12
#define TS_IN_W 94

#define TS_L1_OC 24
#define TS_L1_IC 1
#define TS_L1_OH 12
#define TS_L1_OW 94
#define TS_L1_PH 6
#define TS_L1_PW 47

#define TS_L2_OC 48
#define TS_L2_IC 24
#define TS_L2_OH 6
#define TS_L2_OW 47
#define TS_L2_PH 3
#define TS_L2_PW 23

#define TS_L3_OC 96
#define TS_L3_IC 48
#define TS_L3_OH 3
#define TS_L3_OW 23

#define TS_FC_OUT 6
#define TS_FC_IN 96

#define TS_K 9
#define TS_POOL_AREA 4
#define TS_GAP_AREA (TS_L3_OH * TS_L3_OW)

static int8_t g_w1_q[TS_L1_OC * TS_L1_IC * TS_K] __attribute__((aligned(64)));
static int8_t g_w2_q[TS_L2_OC * TS_L2_IC * TS_K] __attribute__((aligned(64)));
static int8_t g_w3_q[TS_L3_OC * TS_L3_IC * TS_K] __attribute__((aligned(64)));
static int8_t g_wfc_q[TS_FC_OUT * TS_FC_IN] __attribute__((aligned(64)));

static float g_w1_scale = 1.0f;
static float g_w2_scale = 1.0f;
static float g_w3_scale = 1.0f;
static float g_wfc_scale = 1.0f;
static int g_prepared = 0;

static int8_t g_in0[TS_IN_H * TS_IN_W] __attribute__((aligned(64)));
static int8_t g_pad1[(TS_IN_H + 2) * (TS_IN_W + 2)] __attribute__((aligned(64)));
static int32_t g_conv1_acc[TS_L1_OC * TS_L1_OH * TS_L1_OW] __attribute__((aligned(64)));
static int8_t g_act1[TS_L1_OC * TS_L1_PH * TS_L1_PW] __attribute__((aligned(64)));

static int8_t g_pad2[TS_L2_IC * (TS_L2_OH + 2) * (TS_L2_OW + 2)] __attribute__((aligned(64)));
static int32_t g_conv2_acc[TS_L2_OC * TS_L2_OH * TS_L2_OW] __attribute__((aligned(64)));
static int8_t g_act2[TS_L2_OC * TS_L2_PH * TS_L2_PW] __attribute__((aligned(64)));

static int8_t g_pad3[TS_L3_IC * (TS_L3_OH + 2) * (TS_L3_OW + 2)] __attribute__((aligned(64)));
static int32_t g_gap3_acc[TS_L3_OC] __attribute__((aligned(64)));
static int8_t g_act3[TS_FC_IN] __attribute__((aligned(64)));

static int32_t g_bias1_q[TS_L1_OC] __attribute__((aligned(64)));
static int32_t g_bias2_q[TS_L2_OC] __attribute__((aligned(64)));
static int32_t g_bias3_q[TS_L3_OC] __attribute__((aligned(64)));

static inline uint64_t rdcycle64_int8(void) {
    uint64_t x;
    __asm__ volatile("rdcycle %0" : "=r"(x));
    return x;
}

static inline int8_t clamp_i8(int32_t x) {
    if (x > 127) {
        return 127;
    }
    if (x < -128) {
        return -128;
    }
    return (int8_t)x;
}

static inline int8_t clamp_u7(int32_t x) {
    if (x > 127) {
        return 127;
    }
    if (x < 0) {
        return 0;
    }
    return (int8_t)x;
}

static inline int32_t round_to_i32(float x) {
    if (x >= (float)INT32_MAX) {
        return INT32_MAX;
    }
    if (x <= (float)INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)lrintf(x);
}

static float quantize_weights_symmetric(const float *src, int32_t n, int8_t *dst) {
    float max_abs = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-12f) {
        scale = 1.0f / 127.0f;
    }
    float inv_scale = 1.0f / scale;

    for (int32_t i = 0; i < n; i++) {
        dst[i] = clamp_i8(round_to_i32(src[i] * inv_scale));
    }

    return scale;
}

static void make_bias_q(const Tensor *bias, float in_scale, float w_scale, int32_t *dst, int32_t n) {
    float denom = in_scale * w_scale;
    if (fabsf(denom) < 1e-20f) {
        denom = 1e-20f;
    }
    float inv = 1.0f / denom;
    for (int32_t i = 0; i < n; i++) {
        float b = (bias != NULL && bias->f_data != NULL) ? bias->f_data[i] : 0.0f;
        dst[i] = round_to_i32(b * inv);
    }
}

static void pad_input_1ch(const int8_t *src, int32_t h, int32_t w, int8_t *dst, int32_t pad_h, int32_t pad_w) {
    memset(dst, 0, (size_t)(pad_h * pad_w) * sizeof(int8_t));
    for (int32_t i = 0; i < h; i++) {
        memcpy(dst + (i + 1) * pad_w + 1, src + i * w, (size_t)w * sizeof(int8_t));
    }
}

static void pad_input_c(const int8_t *src, int32_t c, int32_t h, int32_t w, int8_t *dst, int32_t pad_h, int32_t pad_w) {
    memset(dst, 0, (size_t)(c * pad_h * pad_w) * sizeof(int8_t));
    const int32_t src_hw = h * w;
    const int32_t dst_hw = pad_h * pad_w;
    for (int32_t ch = 0; ch < c; ch++) {
        const int8_t *s = src + ch * src_hw;
        int8_t *d = dst + ch * dst_hw;
        for (int32_t i = 0; i < h; i++) {
            memcpy(d + (i + 1) * pad_w + 1, s + i * w, (size_t)w * sizeof(int8_t));
        }
    }
}

static void conv3x3_acc_1c(const int8_t *pad,
                           int32_t pad_w,
                           const int8_t *w,
                           const int32_t *bq,
                           int32_t out_c,
                           int32_t out_h,
                           int32_t out_w,
                           int32_t *out) {
    for (int32_t oc = 0; oc < out_c; oc++) {
        const int8_t *wk = w + oc * TS_K;
        for (int32_t oh = 0; oh < out_h; oh++) {
            const int32_t base_r = oh * pad_w;
            for (int32_t ow = 0; ow < out_w; ow++) {
                const int8_t *p = pad + base_r + ow;
                int32_t acc = bq[oc];
                acc += (int32_t)p[0] * (int32_t)wk[0];
                acc += (int32_t)p[1] * (int32_t)wk[1];
                acc += (int32_t)p[2] * (int32_t)wk[2];
                acc += (int32_t)p[pad_w + 0] * (int32_t)wk[3];
                acc += (int32_t)p[pad_w + 1] * (int32_t)wk[4];
                acc += (int32_t)p[pad_w + 2] * (int32_t)wk[5];
                acc += (int32_t)p[(2 * pad_w) + 0] * (int32_t)wk[6];
                acc += (int32_t)p[(2 * pad_w) + 1] * (int32_t)wk[7];
                acc += (int32_t)p[(2 * pad_w) + 2] * (int32_t)wk[8];
                out[(oc * out_h + oh) * out_w + ow] = acc;
            }
        }
    }
}

static void conv3x3_acc_c(const int8_t *pad,
                          int32_t in_c,
                          int32_t pad_h,
                          int32_t pad_w,
                          const int8_t *w,
                          const int32_t *bq,
                          int32_t out_c,
                          int32_t out_h,
                          int32_t out_w,
                          int32_t *out) {
    const int32_t pad_hw = pad_h * pad_w;
    const int32_t w_ic_stride = TS_K;
    const int32_t w_oc_stride = in_c * w_ic_stride;

    for (int32_t oc = 0; oc < out_c; oc++) {
        const int8_t *w_oc = w + oc * w_oc_stride;
        for (int32_t oh = 0; oh < out_h; oh++) {
            for (int32_t ow = 0; ow < out_w; ow++) {
                int32_t acc = bq[oc];
                for (int32_t ic = 0; ic < in_c; ic++) {
                    const int8_t *p = pad + ic * pad_hw + oh * pad_w + ow;
                    const int8_t *wk = w_oc + ic * w_ic_stride;
                    acc += (int32_t)p[0] * (int32_t)wk[0];
                    acc += (int32_t)p[1] * (int32_t)wk[1];
                    acc += (int32_t)p[2] * (int32_t)wk[2];
                    acc += (int32_t)p[pad_w + 0] * (int32_t)wk[3];
                    acc += (int32_t)p[pad_w + 1] * (int32_t)wk[4];
                    acc += (int32_t)p[pad_w + 2] * (int32_t)wk[5];
                    acc += (int32_t)p[(2 * pad_w) + 0] * (int32_t)wk[6];
                    acc += (int32_t)p[(2 * pad_w) + 1] * (int32_t)wk[7];
                    acc += (int32_t)p[(2 * pad_w) + 2] * (int32_t)wk[8];
                }
                out[(oc * out_h + oh) * out_w + ow] = acc;
            }
        }
    }
}

static float requant_pool_relu(const int32_t *conv_acc,
                               int32_t out_c,
                               int32_t conv_h,
                               int32_t conv_w,
                               int32_t pool_h,
                               int32_t pool_w,
                               float in_scale,
                               float w_scale,
                               int8_t *dst) {
    int32_t max_acc = 0;
    for (int32_t i = 0; i < out_c * conv_h * conv_w; i++) {
        int32_t v = conv_acc[i];
        if (v > max_acc) {
            max_acc = v;
        }
    }

    float out_scale = (in_scale * w_scale * (float)max_acc) / 127.0f;
    if (out_scale < 1e-20f) {
        out_scale = in_scale * w_scale;
        if (out_scale < 1e-20f) {
            out_scale = 1.0f;
        }
    }
    float rq = (in_scale * w_scale) / out_scale;

    for (int32_t oc = 0; oc < out_c; oc++) {
        const int32_t *src_c = conv_acc + oc * conv_h * conv_w;
        int8_t *dst_c = dst + oc * pool_h * pool_w;
        for (int32_t ph = 0; ph < pool_h; ph++) {
            int32_t h0 = ph << 1;
            int32_t h1 = h0 + 1;
            for (int32_t pw = 0; pw < pool_w; pw++) {
                int32_t w0 = pw << 1;
                int32_t w1 = w0 + 1;
                int32_t m = src_c[h0 * conv_w + w0];
                int32_t v = src_c[h0 * conv_w + w1];
                if (v > m) {
                    m = v;
                }
                v = src_c[h1 * conv_w + w0];
                if (v > m) {
                    m = v;
                }
                v = src_c[h1 * conv_w + w1];
                if (v > m) {
                    m = v;
                }
                if (m < 0) {
                    m = 0;
                }
                dst_c[ph * pool_w + pw] = clamp_u7(round_to_i32((float)m * rq));
            }
        }
    }

    return out_scale;
}

static float conv_relu_gap_to_i8(const int8_t *input,
                                 float in_scale,
                                 const int8_t *w,
                                 float w_scale,
                                 const int32_t *bq,
                                 int8_t *out_i8) {
    pad_input_c(input, TS_L3_IC, TS_L3_OH, TS_L3_OW, g_pad3, TS_L3_OH + 2, TS_L3_OW + 2);

    int32_t max_avg = 0;
    for (int32_t oc = 0; oc < TS_L3_OC; oc++) {
        const int8_t *w_oc = w + oc * TS_L3_IC * TS_K;
        int64_t sum_pos = 0;
        for (int32_t oh = 0; oh < TS_L3_OH; oh++) {
            for (int32_t ow = 0; ow < TS_L3_OW; ow++) {
                int32_t acc = bq[oc];
                for (int32_t ic = 0; ic < TS_L3_IC; ic++) {
                    const int8_t *p = g_pad3 + ic * ((TS_L3_OH + 2) * (TS_L3_OW + 2)) +
                                      oh * (TS_L3_OW + 2) + ow;
                    const int8_t *wk = w_oc + ic * TS_K;
                    acc += (int32_t)p[0] * (int32_t)wk[0];
                    acc += (int32_t)p[1] * (int32_t)wk[1];
                    acc += (int32_t)p[2] * (int32_t)wk[2];
                    acc += (int32_t)p[(TS_L3_OW + 2) + 0] * (int32_t)wk[3];
                    acc += (int32_t)p[(TS_L3_OW + 2) + 1] * (int32_t)wk[4];
                    acc += (int32_t)p[(TS_L3_OW + 2) + 2] * (int32_t)wk[5];
                    acc += (int32_t)p[(2 * (TS_L3_OW + 2)) + 0] * (int32_t)wk[6];
                    acc += (int32_t)p[(2 * (TS_L3_OW + 2)) + 1] * (int32_t)wk[7];
                    acc += (int32_t)p[(2 * (TS_L3_OW + 2)) + 2] * (int32_t)wk[8];
                }
                if (acc > 0) {
                    sum_pos += acc;
                }
            }
        }
        int32_t avg_acc = (int32_t)((sum_pos + (TS_GAP_AREA / 2)) / TS_GAP_AREA);
        if (avg_acc > max_avg) {
            max_avg = avg_acc;
        }
        g_gap3_acc[oc] = avg_acc;
    }

    float out_scale = (in_scale * w_scale * (float)max_avg) / 127.0f;
    if (out_scale < 1e-20f) {
        out_scale = in_scale * w_scale;
        if (out_scale < 1e-20f) {
            out_scale = 1.0f;
        }
    }
    float rq = (in_scale * w_scale) / out_scale;

    for (int32_t i = 0; i < TS_L3_OC; i++) {
        int32_t q = round_to_i32((float)g_gap3_acc[i] * rq);
        out_i8[i] = clamp_u7(q);
    }

    return out_scale;
}

int tinyspeech_int8_prepare(const Tensor *conv1_w,
                            const Tensor *conv2_w,
                            const Tensor *conv3_w,
                            const Tensor *fc_w) {
    if (conv1_w == NULL || conv2_w == NULL || conv3_w == NULL || fc_w == NULL) {
        g_prepared = 0;
        return 0;
    }
    if (conv1_w->f_data == NULL || conv2_w->f_data == NULL || conv3_w->f_data == NULL || fc_w->f_data == NULL) {
        g_prepared = 0;
        return 0;
    }
    if (conv1_w->size != (TS_L1_OC * TS_L1_IC * TS_K) ||
        conv2_w->size != (TS_L2_OC * TS_L2_IC * TS_K) ||
        conv3_w->size != (TS_L3_OC * TS_L3_IC * TS_K) ||
        fc_w->size != (TS_FC_OUT * TS_FC_IN)) {
        g_prepared = 0;
        return 0;
    }

    g_w1_scale = quantize_weights_symmetric(conv1_w->f_data, conv1_w->size, g_w1_q);
    g_w2_scale = quantize_weights_symmetric(conv2_w->f_data, conv2_w->size, g_w2_q);
    g_w3_scale = quantize_weights_symmetric(conv3_w->f_data, conv3_w->size, g_w3_q);
    g_wfc_scale = quantize_weights_symmetric(fc_w->f_data, fc_w->size, g_wfc_q);

    g_prepared = 1;
    return 1;
}

int tinyspeech_int8_is_ready(void) {
    return g_prepared;
}

Tensor tinyspeech_run_inference_int8(const Tensor *input,
                                     const Tensor *conv1_bias,
                                     const Tensor *conv2_bias,
                                     const Tensor *conv3_bias,
                                     tinyspeech_cycle_profile_t *profile) {
    u_int8_t out_shape[2] = {1, TS_FC_OUT};
    Tensor logits = f_create_tensor(out_shape, 2);

    if (!g_prepared || input == NULL || profile == NULL || input->dims < 4) {
        memset(logits.f_data, 0, (size_t)TS_FC_OUT * sizeof(float));
        return logits;
    }
    if ((int32_t)input->shape[0] != 1 ||
        (int32_t)input->shape[1] != 1 ||
        (int32_t)input->shape[2] != TS_IN_H ||
        (int32_t)input->shape[3] != TS_IN_W ||
        input->size < (TS_IN_H * TS_IN_W)) {
        memset(logits.f_data, 0, (size_t)TS_FC_OUT * sizeof(float));
        return logits;
    }

    uint64_t t0 = rdcycle64_int8();
    if (input->data != NULL) {
        memcpy(g_in0, input->data, (size_t)TS_IN_H * TS_IN_W * sizeof(int8_t));
    } else if (input->f_data != NULL) {
        for (int32_t i = 0; i < TS_IN_H * TS_IN_W; i++) {
            g_in0[i] = clamp_i8(round_to_i32(input->f_data[i]));
        }
    } else {
        memset(g_in0, 0, (size_t)TS_IN_H * TS_IN_W * sizeof(int8_t));
    }
    uint64_t t1 = rdcycle64_int8();
    profile->input_cast = t1 - t0;

    float s0 = 1.0f;

    t0 = rdcycle64_int8();
    make_bias_q(conv1_bias, s0, g_w1_scale, g_bias1_q, TS_L1_OC);
    pad_input_1ch(g_in0, TS_IN_H, TS_IN_W, g_pad1, TS_IN_H + 2, TS_IN_W + 2);
    conv3x3_acc_1c(g_pad1, TS_IN_W + 2, g_w1_q, g_bias1_q, TS_L1_OC, TS_L1_OH, TS_L1_OW, g_conv1_acc);
    float s1 = requant_pool_relu(g_conv1_acc, TS_L1_OC, TS_L1_OH, TS_L1_OW, TS_L1_PH, TS_L1_PW, s0, g_w1_scale, g_act1);
    t1 = rdcycle64_int8();
    profile->conv1_pool1 = t1 - t0;

    t0 = rdcycle64_int8();
    make_bias_q(conv2_bias, s1, g_w2_scale, g_bias2_q, TS_L2_OC);
    pad_input_c(g_act1, TS_L2_IC, TS_L2_OH, TS_L2_OW, g_pad2, TS_L2_OH + 2, TS_L2_OW + 2);
    conv3x3_acc_c(g_pad2, TS_L2_IC, TS_L2_OH + 2, TS_L2_OW + 2, g_w2_q, g_bias2_q, TS_L2_OC, TS_L2_OH, TS_L2_OW, g_conv2_acc);
    float s2 = requant_pool_relu(g_conv2_acc, TS_L2_OC, TS_L2_OH, TS_L2_OW, TS_L2_PH, TS_L2_PW, s1, g_w2_scale, g_act2);
    t1 = rdcycle64_int8();
    profile->conv2_pool2 = t1 - t0;

    t0 = rdcycle64_int8();
    make_bias_q(conv3_bias, s2, g_w3_scale, g_bias3_q, TS_L3_OC);
    float s3 = conv_relu_gap_to_i8(g_act2, s2, g_w3_q, g_w3_scale, g_bias3_q, g_act3);
    t1 = rdcycle64_int8();
    profile->conv3_gap = t1 - t0;

    t0 = rdcycle64_int8();
    float out_scale = s3 * g_wfc_scale;
    for (int32_t oc = 0; oc < TS_FC_OUT; oc++) {
        const int8_t *wrow = g_wfc_q + oc * TS_FC_IN;
        int32_t acc = 0;
        for (int32_t i = 0; i < TS_FC_IN; i += 8) {
            acc += (int32_t)g_act3[i + 0] * (int32_t)wrow[i + 0];
            acc += (int32_t)g_act3[i + 1] * (int32_t)wrow[i + 1];
            acc += (int32_t)g_act3[i + 2] * (int32_t)wrow[i + 2];
            acc += (int32_t)g_act3[i + 3] * (int32_t)wrow[i + 3];
            acc += (int32_t)g_act3[i + 4] * (int32_t)wrow[i + 4];
            acc += (int32_t)g_act3[i + 5] * (int32_t)wrow[i + 5];
            acc += (int32_t)g_act3[i + 6] * (int32_t)wrow[i + 6];
            acc += (int32_t)g_act3[i + 7] * (int32_t)wrow[i + 7];
        }
        logits.f_data[oc] = (float)acc * out_scale;
    }
    t1 = rdcycle64_int8();
    profile->fc_logits = t1 - t0;
    profile->softmax = 0;

    return logits;
}
