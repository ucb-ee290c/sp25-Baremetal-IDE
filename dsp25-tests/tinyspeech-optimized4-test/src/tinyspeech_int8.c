#include "tinyspeech_int8.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

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
static int8_t g_w1_pack[TS_L1_OC * TS_L1_IC * TS_K] __attribute__((aligned(64)));
static int8_t g_w2_pack[TS_L2_OC * TS_L2_IC * TS_K] __attribute__((aligned(64)));
static int8_t g_w3_pack[TS_L3_OC * TS_L3_IC * TS_K] __attribute__((aligned(64)));
static int16_t g_w1_pack16[TS_L1_OC * TS_L1_IC * TS_K] __attribute__((aligned(64)));
static int16_t g_w2_pack16[TS_L2_OC * TS_L2_IC * TS_K] __attribute__((aligned(64)));
static int16_t g_w3_pack16[TS_L3_OC * TS_L3_IC * TS_K] __attribute__((aligned(64)));

static float g_w1_scale = 1.0f;
static float g_w2_scale = 1.0f;
static float g_w3_scale = 1.0f;
static float g_wfc_scale = 1.0f;
static int g_prepared = 0;
static int g_collect_calib = 0;
static int g_fixed_qparams_valid = 0;
static int32_t g_calib_max1 = 0;
static int32_t g_calib_max2 = 0;
static int32_t g_calib_max3 = 0;
static uint32_t g_mul1_q31 = 0;
static uint32_t g_mul2_q31 = 0;
static uint32_t g_mul3_q31 = 0;
static float g_s1_fixed = 1.0f;
static float g_s2_fixed = 1.0f;
static float g_s3_fixed = 1.0f;

static int8_t g_in0[TS_IN_H * TS_IN_W] __attribute__((aligned(64)));
static int8_t g_pad1[(TS_IN_H + 2) * (TS_IN_W + 2)] __attribute__((aligned(64)));
static int32_t g_conv1_acc[TS_L1_OC * TS_L1_OH * TS_L1_OW] __attribute__((aligned(64)));
static int32_t g_pool1_acc[TS_L1_OC * TS_L1_PH * TS_L1_PW] __attribute__((aligned(64)));
static int8_t g_act1[TS_L1_OC * TS_L1_PH * TS_L1_PW] __attribute__((aligned(64)));

static int8_t g_pad2[TS_L2_IC * (TS_L2_OH + 2) * (TS_L2_OW + 2)] __attribute__((aligned(64)));
static int32_t g_conv2_acc[TS_L2_OC * TS_L2_OH * TS_L2_OW] __attribute__((aligned(64)));
static int32_t g_pool2_acc[TS_L2_OC * TS_L2_PH * TS_L2_PW] __attribute__((aligned(64)));
static int8_t g_act2[TS_L2_OC * TS_L2_PH * TS_L2_PW] __attribute__((aligned(64)));

static int8_t g_pad3[TS_L3_IC * (TS_L3_OH + 2) * (TS_L3_OW + 2)] __attribute__((aligned(64)));
static int32_t g_gap3_acc[TS_L3_OC] __attribute__((aligned(64)));
static int8_t g_act3[TS_FC_IN] __attribute__((aligned(64)));
static int32_t g_conv3_acc[TS_L3_OC * TS_L3_OH * TS_L3_OW] __attribute__((aligned(64)));

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

static inline uint32_t requant_mul_q31_from_max(int32_t max_acc) {
    if (max_acc <= 0) {
        return 0;
    }
    return (uint32_t)(((uint64_t)127u << 31) / (uint32_t)max_acc);
}

static inline int8_t requant_u7_from_acc(int32_t v, uint32_t mul_q31) {
    if (v <= 0 || mul_q31 == 0) {
        return 0;
    }
    int32_t q = (int32_t)(((int64_t)v * (int64_t)mul_q31 + (1ll << 30)) >> 31);
    if (q > 127) {
        q = 127;
    }
    return (int8_t)q;
}

static inline void calib_track_max(int layer, int32_t max_acc) {
    if (!g_collect_calib) {
        return;
    }
    if (layer == 1) {
        if (max_acc > g_calib_max1) {
            g_calib_max1 = max_acc;
        }
    } else if (layer == 2) {
        if (max_acc > g_calib_max2) {
            g_calib_max2 = max_acc;
        }
    } else if (layer == 3) {
        if (max_acc > g_calib_max3) {
            g_calib_max3 = max_acc;
        }
    }
}

static inline int32_t choose_calib_max(int32_t observed_max) {
    if (observed_max <= 0) {
        return 1;
    }
    int64_t bumped = ((int64_t)observed_max * 105 + 99) / 100;
    if (bumped > INT32_MAX) {
        bumped = INT32_MAX;
    }
    return (int32_t)bumped;
}

static inline float out_scale_from_max(float in_scale, float w_scale, int32_t max_acc) {
    float out_scale = (in_scale * w_scale * (float)max_acc) / 127.0f;
    if (out_scale < 1e-20f) {
        out_scale = in_scale * w_scale;
        if (out_scale < 1e-20f) {
            out_scale = 1.0f;
        }
    }
    return out_scale;
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

static void pack_w_oc_to_k_major(const int8_t *src, int8_t *dst, int32_t out_c, int32_t in_c) {
    const int32_t K = in_c * TS_K;
    for (int32_t k = 0; k < K; k++) {
        for (int32_t oc = 0; oc < out_c; oc++) {
            dst[k * out_c + oc] = src[oc * K + k];
        }
    }
}

static void pack_w_oc_to_k_major_i16(const int8_t *src, int16_t *dst, int32_t out_c, int32_t in_c) {
    const int32_t K = in_c * TS_K;
    for (int32_t k = 0; k < K; k++) {
        for (int32_t oc = 0; oc < out_c; oc++) {
            dst[k * out_c + oc] = (int16_t)src[oc * K + k];
        }
    }
}

static void conv3x3_acc_1c(const int8_t *pad,
                           int32_t pad_w,
                           const int8_t *w8,
                           const int16_t *w16,
                           const int32_t *bq,
                           int32_t out_c,
                           int32_t out_h,
                           int32_t out_w,
                           int32_t *out) {
#if defined(__riscv_vector)
    for (int32_t oh = 0; oh < out_h; oh++) {
        const int32_t base_r = oh * pad_w;
        for (int32_t ow = 0; ow < out_w; ow++) {
            const int8_t *p = pad + base_r + ow;
            const int8_t x0 = p[0];
            const int8_t x1 = p[1];
            const int8_t x2 = p[2];
            const int8_t x3 = p[pad_w + 0];
            const int8_t x4 = p[pad_w + 1];
            const int8_t x5 = p[pad_w + 2];
            const int8_t x6 = p[(2 * pad_w) + 0];
            const int8_t x7 = p[(2 * pad_w) + 1];
            const int8_t x8 = p[(2 * pad_w) + 2];

            for (int32_t oc0 = 0; oc0 < out_c; ) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(out_c - oc0));
                vint32m4_t vacc = __riscv_vle32_v_i32m4(bq + oc0, vl);

                vint16m2_t vw;
                vw = __riscv_vle16_v_i16m2(w16 + 0 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x0, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 1 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x1, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 2 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x2, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 3 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x3, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 4 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x4, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 5 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x5, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 6 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x6, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 7 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x7, vw, vl);
                vw = __riscv_vle16_v_i16m2(w16 + 8 * out_c + oc0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, x8, vw, vl);

                int32_t *dst = out + (oh * out_w + ow) + oc0 * out_h * out_w;
                __riscv_vsse32_v_i32m4(dst, (ptrdiff_t)(out_h * out_w * (int32_t)sizeof(int32_t)), vacc, vl);
                oc0 += (int32_t)vl;
            }
        }
    }
#else
    for (int32_t oc = 0; oc < out_c; oc++) {
        const int8_t *wk = w8 + oc * TS_K;
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
#endif
}

static void conv3x3_acc_c(const int8_t *pad,
                          int32_t in_c,
                          int32_t pad_h,
                          int32_t pad_w,
                          const int8_t *w8,
                          const int16_t *w16,
                          const int32_t *bq,
                          int32_t out_c,
                          int32_t out_h,
                          int32_t out_w,
                          int32_t *out) {
#if defined(__riscv_vector)
    (void)pad_h;
    for (int32_t oh = 0; oh < out_h; oh++) {
        for (int32_t ow = 0; ow < out_w; ow++) {
            for (int32_t oc0 = 0; oc0 < out_c; ) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(out_c - oc0));
                vint32m4_t vacc = __riscv_vle32_v_i32m4(bq + oc0, vl);

                int32_t k = 0;
                for (int32_t ic = 0; ic < in_c; ic++) {
                    const int8_t *p = pad + ic * (pad_h * pad_w) + oh * pad_w + ow;
                    const int8_t x0 = p[0];
                    const int8_t x1 = p[1];
                    const int8_t x2 = p[2];
                    const int8_t x3 = p[pad_w + 0];
                    const int8_t x4 = p[pad_w + 1];
                    const int8_t x5 = p[pad_w + 2];
                    const int8_t x6 = p[(2 * pad_w) + 0];
                    const int8_t x7 = p[(2 * pad_w) + 1];
                    const int8_t x8 = p[(2 * pad_w) + 2];

                    vint16m2_t vw;
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 0) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x0, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 1) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x1, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 2) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x2, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 3) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x3, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 4) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x4, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 5) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x5, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 6) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x6, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 7) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x7, vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 8) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, x8, vw, vl);
                    k += TS_K;
                }

                int32_t *dst = out + (oh * out_w + ow) + oc0 * out_h * out_w;
                __riscv_vsse32_v_i32m4(dst, (ptrdiff_t)(out_h * out_w * (int32_t)sizeof(int32_t)), vacc, vl);
                oc0 += (int32_t)vl;
            }
        }
    }
#else
    const int32_t pad_hw = pad_h * pad_w;
    const int32_t w_ic_stride = TS_K;
    const int32_t w_oc_stride = in_c * w_ic_stride;

    for (int32_t oc = 0; oc < out_c; oc++) {
        const int8_t *w_oc = w8 + oc * w_oc_stride;
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
#endif
}

#if defined(__riscv_vector)
static void conv3x3_pool2x2_acc_1c_rvv(const int8_t *pad,
                                       int32_t pad_w,
                                       const int16_t *w16,
                                       const int32_t *bq,
                                       int32_t out_c,
                                       int32_t pool_h,
                                       int32_t pool_w,
                                       int32_t *pool_acc) {
    const int32_t pool_hw = pool_h * pool_w;
    for (int32_t oc0 = 0; oc0 < out_c; ) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_c - oc0));
        vint32m4_t vbias = __riscv_vle32_v_i32m4(bq + oc0, vl);
        vint16m2_t vw0 = __riscv_vle16_v_i16m2(w16 + 0 * out_c + oc0, vl);
        vint16m2_t vw1 = __riscv_vle16_v_i16m2(w16 + 1 * out_c + oc0, vl);
        vint16m2_t vw2 = __riscv_vle16_v_i16m2(w16 + 2 * out_c + oc0, vl);
        vint16m2_t vw3 = __riscv_vle16_v_i16m2(w16 + 3 * out_c + oc0, vl);
        vint16m2_t vw4 = __riscv_vle16_v_i16m2(w16 + 4 * out_c + oc0, vl);
        vint16m2_t vw5 = __riscv_vle16_v_i16m2(w16 + 5 * out_c + oc0, vl);
        vint16m2_t vw6 = __riscv_vle16_v_i16m2(w16 + 6 * out_c + oc0, vl);
        vint16m2_t vw7 = __riscv_vle16_v_i16m2(w16 + 7 * out_c + oc0, vl);
        vint16m2_t vw8 = __riscv_vle16_v_i16m2(w16 + 8 * out_c + oc0, vl);

        for (int32_t ph = 0; ph < pool_h; ph++) {
            const int32_t h0 = ph << 1;
            for (int32_t pw = 0; pw < pool_w; pw++) {
                const int32_t w0 = pw << 1;
                const int8_t *p00 = pad + h0 * pad_w + w0;
                const int8_t *p01 = p00 + 1;
                const int8_t *p10 = p00 + pad_w;
                const int8_t *p11 = p10 + 1;

                vint32m4_t vacc00 = vbias;
                vint32m4_t vacc01 = vbias;
                vint32m4_t vacc10 = vbias;
                vint32m4_t vacc11 = vbias;

                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[0], vw0, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[0], vw0, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[0], vw0, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[0], vw0, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[1], vw1, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[1], vw1, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[1], vw1, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[1], vw1, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[2], vw2, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[2], vw2, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[2], vw2, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[2], vw2, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 0], vw3, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 0], vw3, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 0], vw3, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 0], vw3, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 1], vw4, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 1], vw4, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 1], vw4, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 1], vw4, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 2], vw5, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 2], vw5, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 2], vw5, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 2], vw5, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 0], vw6, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 0], vw6, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 0], vw6, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 0], vw6, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 1], vw7, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 1], vw7, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 1], vw7, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 1], vw7, vl);
                vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 2], vw8, vl);
                vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 2], vw8, vl);
                vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 2], vw8, vl);
                vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 2], vw8, vl);

                vint32m4_t vmax = __riscv_vmax_vv_i32m4(vacc00, vacc01, vl);
                vint32m4_t vmax2 = __riscv_vmax_vv_i32m4(vacc10, vacc11, vl);
                vmax = __riscv_vmax_vv_i32m4(vmax, vmax2, vl);
                int32_t *dst = pool_acc + (ph * pool_w + pw) + oc0 * pool_hw;
                __riscv_vsse32_v_i32m4(dst, (ptrdiff_t)(pool_hw * (int32_t)sizeof(int32_t)), vmax, vl);
            }
        }

        oc0 += (int32_t)vl;
    }
}

static void conv3x3_pool2x2_acc_c_rvv(const int8_t *pad,
                                      int32_t in_c,
                                      int32_t pad_h,
                                      int32_t pad_w,
                                      const int16_t *w16,
                                      const int32_t *bq,
                                      int32_t out_c,
                                      int32_t pool_h,
                                      int32_t pool_w,
                                      int32_t *pool_acc) {
    const int32_t pad_hw = pad_h * pad_w;
    const int32_t pool_hw = pool_h * pool_w;
    for (int32_t ph = 0; ph < pool_h; ph++) {
        const int32_t h0 = ph << 1;
        for (int32_t pw = 0; pw < pool_w; pw++) {
            const int32_t w0 = pw << 1;
            for (int32_t oc0 = 0; oc0 < out_c; ) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(out_c - oc0));
                vint32m4_t vbias = __riscv_vle32_v_i32m4(bq + oc0, vl);
                vint32m4_t vacc00 = vbias;
                vint32m4_t vacc01 = vbias;
                vint32m4_t vacc10 = vbias;
                vint32m4_t vacc11 = vbias;

                int32_t k = 0;
                for (int32_t ic = 0; ic < in_c; ic++) {
                    const int8_t *pin = pad + ic * pad_hw + h0 * pad_w + w0;
                    const int8_t *p00 = pin;
                    const int8_t *p01 = pin + 1;
                    const int8_t *p10 = pin + pad_w;
                    const int8_t *p11 = p10 + 1;
                    vint16m2_t vw;

                    vw = __riscv_vle16_v_i16m2(w16 + (k + 0) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[0], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[0], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[0], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 1) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[1], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[1], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[1], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 2) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[2], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[2], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[2], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 3) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 0], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 0], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 0], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 4) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 1], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 1], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 1], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 5) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 2], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 2], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 2], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 6) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 0], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 0], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 0], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 7) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 1], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 1], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 1], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 8) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 2], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 2], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 2], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 2], vw, vl);
                    k += TS_K;
                }

                vint32m4_t vmax = __riscv_vmax_vv_i32m4(vacc00, vacc01, vl);
                vint32m4_t vmax2 = __riscv_vmax_vv_i32m4(vacc10, vacc11, vl);
                vmax = __riscv_vmax_vv_i32m4(vmax, vmax2, vl);
                int32_t *dst = pool_acc + (ph * pool_w + pw) + oc0 * pool_hw;
                __riscv_vsse32_v_i32m4(dst, (ptrdiff_t)(pool_hw * (int32_t)sizeof(int32_t)), vmax, vl);
                oc0 += (int32_t)vl;
            }
        }
    }
}

static void conv3x3_pool2x2_requant_relu_to_padded_c_rvv(const int8_t *pad,
                                                          int32_t in_c,
                                                          int32_t pad_h,
                                                          int32_t pad_w,
                                                          const int16_t *w16,
                                                          const int32_t *bq,
                                                          int32_t out_c,
                                                          int32_t pool_h,
                                                          int32_t pool_w,
                                                          uint32_t mul_q31,
                                                          int8_t *dst_pad,
                                                          int32_t dst_pad_h,
                                                          int32_t dst_pad_w) {
    memset(dst_pad, 0, (size_t)(out_c * dst_pad_h * dst_pad_w) * sizeof(int8_t));
    const int32_t pad_hw = pad_h * pad_w;
    const int32_t dst_hw = dst_pad_h * dst_pad_w;
    int32_t lane_buf[TS_L2_OC] __attribute__((aligned(64)));

    for (int32_t ph = 0; ph < pool_h; ph++) {
        const int32_t h0 = ph << 1;
        for (int32_t pw = 0; pw < pool_w; pw++) {
            const int32_t w0 = pw << 1;
            for (int32_t oc0 = 0; oc0 < out_c; ) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(out_c - oc0));
                vint32m4_t vbias = __riscv_vle32_v_i32m4(bq + oc0, vl);
                vint32m4_t vacc00 = vbias;
                vint32m4_t vacc01 = vbias;
                vint32m4_t vacc10 = vbias;
                vint32m4_t vacc11 = vbias;

                int32_t k = 0;
                for (int32_t ic = 0; ic < in_c; ic++) {
                    const int8_t *pin = pad + ic * pad_hw + h0 * pad_w + w0;
                    const int8_t *p00 = pin;
                    const int8_t *p01 = pin + 1;
                    const int8_t *p10 = pin + pad_w;
                    const int8_t *p11 = p10 + 1;
                    vint16m2_t vw;

                    vw = __riscv_vle16_v_i16m2(w16 + (k + 0) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[0], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[0], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[0], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 1) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[1], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[1], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[1], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 2) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[2], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[2], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[2], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 3) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 0], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 0], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 0], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 4) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 1], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 1], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 1], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 5) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[pad_w + 2], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[pad_w + 2], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[pad_w + 2], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[pad_w + 2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 6) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 0], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 0], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 0], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 7) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 1], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 1], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 1], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 8) * out_c + oc0, vl);
                    vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, p00[(2 * pad_w) + 2], vw, vl);
                    vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, p01[(2 * pad_w) + 2], vw, vl);
                    vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, p10[(2 * pad_w) + 2], vw, vl);
                    vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, p11[(2 * pad_w) + 2], vw, vl);
                    k += TS_K;
                }

                vint32m4_t vmax = __riscv_vmax_vv_i32m4(vacc00, vacc01, vl);
                vint32m4_t vmax2 = __riscv_vmax_vv_i32m4(vacc10, vacc11, vl);
                vmax = __riscv_vmax_vv_i32m4(vmax, vmax2, vl);
                __riscv_vse32_v_i32m4(lane_buf, vmax, vl);
                for (size_t lane = 0; lane < vl; lane++) {
                    int32_t oc = oc0 + (int32_t)lane;
                    dst_pad[oc * dst_hw + (ph + 1) * dst_pad_w + (pw + 1)] =
                        requant_u7_from_acc(lane_buf[lane], mul_q31);
                }
                oc0 += (int32_t)vl;
            }
        }
    }
}

static void conv3x3_relu_gap_acc_c_rvv(const int8_t *pad,
                                       int32_t in_c,
                                       int32_t pad_h,
                                       int32_t pad_w,
                                       const int16_t *w16,
                                       const int32_t *bq,
                                       int32_t out_c,
                                       int32_t out_h,
                                       int32_t out_w,
                                       int32_t *gap_sum) {
    const int32_t pad_hw = pad_h * pad_w;
    for (int32_t oc0 = 0; oc0 < out_c; ) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(out_c - oc0));
        vint32m4_t vsum = __riscv_vmv_v_x_i32m4(0, vl);
        const vint32m4_t vzero = __riscv_vmv_v_x_i32m4(0, vl);
        const vint32m4_t vbias = __riscv_vle32_v_i32m4(bq + oc0, vl);

        for (int32_t oh = 0; oh < out_h; oh++) {
            int32_t ow = 0;
            for (; ow + 3 < out_w; ow += 4) {
                vint32m4_t vacc0 = vbias;
                vint32m4_t vacc1 = vbias;
                vint32m4_t vacc2 = vbias;
                vint32m4_t vacc3 = vbias;

                int32_t k = 0;
                for (int32_t ic = 0; ic < in_c; ic++) {
                    const int8_t *p0 = pad + ic * pad_hw + oh * pad_w + ow;
                    const int8_t *p1 = p0 + 1;
                    const int8_t *p2 = p0 + 2;
                    const int8_t *p3 = p0 + 3;
                    vint16m2_t vw;

                    vw = __riscv_vle16_v_i16m2(w16 + (k + 0) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[0], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[0], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[0], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 1) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[1], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[1], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[1], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 2) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[2], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[2], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[2], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 3) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[pad_w + 0], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[pad_w + 0], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[pad_w + 0], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[pad_w + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 4) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[pad_w + 1], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[pad_w + 1], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[pad_w + 1], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[pad_w + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 5) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[pad_w + 2], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[pad_w + 2], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[pad_w + 2], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[pad_w + 2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 6) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[(2 * pad_w) + 0], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[(2 * pad_w) + 0], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[(2 * pad_w) + 0], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[(2 * pad_w) + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 7) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[(2 * pad_w) + 1], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[(2 * pad_w) + 1], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[(2 * pad_w) + 1], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[(2 * pad_w) + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 8) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[(2 * pad_w) + 2], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[(2 * pad_w) + 2], vw, vl);
                    vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, p2[(2 * pad_w) + 2], vw, vl);
                    vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, p3[(2 * pad_w) + 2], vw, vl);
                    k += TS_K;
                }

                vacc0 = __riscv_vmax_vv_i32m4(vacc0, vzero, vl);
                vacc1 = __riscv_vmax_vv_i32m4(vacc1, vzero, vl);
                vacc2 = __riscv_vmax_vv_i32m4(vacc2, vzero, vl);
                vacc3 = __riscv_vmax_vv_i32m4(vacc3, vzero, vl);
                vsum = __riscv_vadd_vv_i32m4(vsum, vacc0, vl);
                vsum = __riscv_vadd_vv_i32m4(vsum, vacc1, vl);
                vsum = __riscv_vadd_vv_i32m4(vsum, vacc2, vl);
                vsum = __riscv_vadd_vv_i32m4(vsum, vacc3, vl);
            }

            for (; ow + 1 < out_w; ow += 2) {
                vint32m4_t vacc0 = vbias;
                vint32m4_t vacc1 = vbias;

                int32_t k = 0;
                for (int32_t ic = 0; ic < in_c; ic++) {
                    const int8_t *p0 = pad + ic * pad_hw + oh * pad_w + ow;
                    const int8_t *p1 = p0 + 1;
                    vint16m2_t vw;

                    vw = __riscv_vle16_v_i16m2(w16 + (k + 0) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[0], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 1) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[1], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 2) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[2], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 3) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[pad_w + 0], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[pad_w + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 4) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[pad_w + 1], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[pad_w + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 5) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[pad_w + 2], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[pad_w + 2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 6) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[(2 * pad_w) + 0], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[(2 * pad_w) + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 7) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[(2 * pad_w) + 1], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[(2 * pad_w) + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 8) * out_c + oc0, vl);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, p0[(2 * pad_w) + 2], vw, vl);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, p1[(2 * pad_w) + 2], vw, vl);
                    k += TS_K;
                }

                vacc0 = __riscv_vmax_vv_i32m4(vacc0, vzero, vl);
                vacc1 = __riscv_vmax_vv_i32m4(vacc1, vzero, vl);
                vsum = __riscv_vadd_vv_i32m4(vsum, vacc0, vl);
                vsum = __riscv_vadd_vv_i32m4(vsum, vacc1, vl);
            }

            if (ow < out_w) {
                vint32m4_t vacc = vbias;
                int32_t k = 0;
                for (int32_t ic = 0; ic < in_c; ic++) {
                    const int8_t *p = pad + ic * pad_hw + oh * pad_w + ow;
                    vint16m2_t vw;
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 0) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 1) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 2) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 3) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[pad_w + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 4) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[pad_w + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 5) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[pad_w + 2], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 6) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[(2 * pad_w) + 0], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 7) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[(2 * pad_w) + 1], vw, vl);
                    vw = __riscv_vle16_v_i16m2(w16 + (k + 8) * out_c + oc0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, p[(2 * pad_w) + 2], vw, vl);
                    k += TS_K;
                }
                vacc = __riscv_vmax_vv_i32m4(vacc, vzero, vl);
                vsum = __riscv_vadd_vv_i32m4(vsum, vacc, vl);
            }
        }

        __riscv_vse32_v_i32m4(gap_sum + oc0, vsum, vl);
        oc0 += (int32_t)vl;
    }
}
#endif

static float requant_pool_relu(const int32_t *conv_acc,
                               int32_t out_c,
                               int32_t conv_h,
                               int32_t conv_w,
                               int32_t pool_h,
                               int32_t pool_w,
                               float in_scale,
                               float w_scale,
                               int8_t *dst,
                               int32_t *max_acc_out) {
    int32_t max_acc = 0;
    for (int32_t i = 0; i < out_c * conv_h * conv_w; i++) {
        int32_t v = conv_acc[i];
        if (v > max_acc) {
            max_acc = v;
        }
    }

    float out_scale = out_scale_from_max(in_scale, w_scale, max_acc);
    uint32_t mul_q31 = requant_mul_q31_from_max(max_acc);

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
                dst_c[ph * pool_w + pw] = requant_u7_from_acc(m, mul_q31);
            }
        }
    }

    if (max_acc_out != NULL) {
        *max_acc_out = max_acc;
    }

    return out_scale;
}

static void requant_pool_relu_fixed(const int32_t *conv_acc,
                                    int32_t out_c,
                                    int32_t conv_h,
                                    int32_t conv_w,
                                    int32_t pool_h,
                                    int32_t pool_w,
                                    uint32_t mul_q31,
                                    int8_t *dst) {
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
                dst_c[ph * pool_w + pw] = requant_u7_from_acc(m, mul_q31);
            }
        }
    }
}

static float requant_relu_from_acc_to_padded(const int32_t *acc,
                                             int32_t c,
                                             int32_t h,
                                             int32_t w,
                                             float in_scale,
                                             float w_scale,
                                             int8_t *dst_pad,
                                             int32_t pad_h,
                                             int32_t pad_w,
                                             int32_t *max_acc_out) {
    int32_t max_acc = 0;
    const int32_t count = c * h * w;
    for (int32_t i = 0; i < count; i++) {
        int32_t v = acc[i];
        if (v > max_acc) {
            max_acc = v;
        }
    }

    float out_scale = out_scale_from_max(in_scale, w_scale, max_acc);
    uint32_t mul_q31 = requant_mul_q31_from_max(max_acc);

    memset(dst_pad, 0, (size_t)(c * pad_h * pad_w) * sizeof(int8_t));
    const int32_t src_hw = h * w;
    const int32_t dst_hw = pad_h * pad_w;
    for (int32_t ch = 0; ch < c; ch++) {
        const int32_t *src_c = acc + ch * src_hw;
        int8_t *dst_c = dst_pad + ch * dst_hw;
        for (int32_t i = 0; i < h; i++) {
            int8_t *drow = dst_c + (i + 1) * pad_w + 1;
            const int32_t *srow = src_c + i * w;
            for (int32_t j = 0; j < w; j++) {
                drow[j] = requant_u7_from_acc(srow[j], mul_q31);
            }
        }
    }

    if (max_acc_out != NULL) {
        *max_acc_out = max_acc;
    }

    return out_scale;
}

static void requant_relu_from_acc_to_padded_fixed(const int32_t *acc,
                                                  int32_t c,
                                                  int32_t h,
                                                  int32_t w,
                                                  uint32_t mul_q31,
                                                  int8_t *dst_pad,
                                                  int32_t pad_h,
                                                  int32_t pad_w) {
    memset(dst_pad, 0, (size_t)(c * pad_h * pad_w) * sizeof(int8_t));
    const int32_t src_hw = h * w;
    const int32_t dst_hw = pad_h * pad_w;
    for (int32_t ch = 0; ch < c; ch++) {
        const int32_t *src_c = acc + ch * src_hw;
        int8_t *dst_c = dst_pad + ch * dst_hw;
        for (int32_t i = 0; i < h; i++) {
            int8_t *drow = dst_c + (i + 1) * pad_w + 1;
            const int32_t *srow = src_c + i * w;
            for (int32_t j = 0; j < w; j++) {
                drow[j] = requant_u7_from_acc(srow[j], mul_q31);
            }
        }
    }
}

static float conv_relu_gap_to_i8(const int8_t *input,
                                 float in_scale,
                                 const int8_t *w8,
                                 const int16_t *w16,
                                 float w_scale,
                                 const int32_t *bq,
                                 int8_t *out_i8,
                                 int32_t *max_avg_out) {
    pad_input_c(input, TS_L3_IC, TS_L3_OH, TS_L3_OW, g_pad3, TS_L3_OH + 2, TS_L3_OW + 2);

    conv3x3_acc_c(g_pad3, TS_L3_IC, TS_L3_OH + 2, TS_L3_OW + 2, w8, w16, bq, TS_L3_OC, TS_L3_OH, TS_L3_OW, g_conv3_acc);

    int32_t max_avg = 0;
    for (int32_t oc = 0; oc < TS_L3_OC; oc++) {
        int64_t sum_pos = 0;
        const int32_t *src = g_conv3_acc + oc * (TS_L3_OH * TS_L3_OW);
        for (int32_t i = 0; i < TS_L3_OH * TS_L3_OW; i++) {
            int32_t acc = src[i];
            if (acc > 0) {
                sum_pos += acc;
            }
        }
        int32_t avg_acc = (int32_t)((sum_pos + (TS_GAP_AREA / 2)) / TS_GAP_AREA);
        if (avg_acc > max_avg) {
            max_avg = avg_acc;
        }
        g_gap3_acc[oc] = avg_acc;
    }

    float out_scale = out_scale_from_max(in_scale, w_scale, max_avg);
    uint32_t mul_q31 = requant_mul_q31_from_max(max_avg);

    for (int32_t i = 0; i < TS_L3_OC; i++) {
        out_i8[i] = requant_u7_from_acc(g_gap3_acc[i], mul_q31);
    }

    if (max_avg_out != NULL) {
        *max_avg_out = max_avg;
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
    pack_w_oc_to_k_major(g_w1_q, g_w1_pack, TS_L1_OC, TS_L1_IC);
    pack_w_oc_to_k_major(g_w2_q, g_w2_pack, TS_L2_OC, TS_L2_IC);
    pack_w_oc_to_k_major(g_w3_q, g_w3_pack, TS_L3_OC, TS_L3_IC);
    pack_w_oc_to_k_major_i16(g_w1_q, g_w1_pack16, TS_L1_OC, TS_L1_IC);
    pack_w_oc_to_k_major_i16(g_w2_q, g_w2_pack16, TS_L2_OC, TS_L2_IC);
    pack_w_oc_to_k_major_i16(g_w3_q, g_w3_pack16, TS_L3_OC, TS_L3_IC);

    g_collect_calib = 0;
    g_fixed_qparams_valid = 0;
    g_calib_max1 = 0;
    g_calib_max2 = 0;
    g_calib_max3 = 0;
    g_mul1_q31 = 0;
    g_mul2_q31 = 0;
    g_mul3_q31 = 0;
    g_s1_fixed = 1.0f;
    g_s2_fixed = 1.0f;
    g_s3_fixed = 1.0f;

    g_prepared = 1;
    return 1;
}

int tinyspeech_int8_is_ready(void) {
    return g_prepared;
}

void tinyspeech_int8_calib_reset(void) {
    if (!g_prepared) {
        return;
    }
    g_collect_calib = 1;
    g_fixed_qparams_valid = 0;
    g_calib_max1 = 0;
    g_calib_max2 = 0;
    g_calib_max3 = 0;
    g_mul1_q31 = 0;
    g_mul2_q31 = 0;
    g_mul3_q31 = 0;
    g_s1_fixed = 1.0f;
    g_s2_fixed = 1.0f;
    g_s3_fixed = 1.0f;
}

int tinyspeech_int8_calib_finalize(const Tensor *conv1_bias,
                                   const Tensor *conv2_bias,
                                   const Tensor *conv3_bias) {
    if (!g_prepared) {
        g_collect_calib = 0;
        g_fixed_qparams_valid = 0;
        return 0;
    }

    int32_t m1 = choose_calib_max(g_calib_max1);
    int32_t m2 = choose_calib_max(g_calib_max2);
    int32_t m3 = choose_calib_max(g_calib_max3);

    g_mul1_q31 = requant_mul_q31_from_max(m1);
    g_mul2_q31 = requant_mul_q31_from_max(m2);
    g_mul3_q31 = requant_mul_q31_from_max(m3);
    g_s1_fixed = out_scale_from_max(1.0f, g_w1_scale, m1);
    g_s2_fixed = out_scale_from_max(g_s1_fixed, g_w2_scale, m2);
    g_s3_fixed = out_scale_from_max(g_s2_fixed, g_w3_scale, m3);

    make_bias_q(conv1_bias, 1.0f, g_w1_scale, g_bias1_q, TS_L1_OC);
    make_bias_q(conv2_bias, g_s1_fixed, g_w2_scale, g_bias2_q, TS_L2_OC);
    make_bias_q(conv3_bias, g_s2_fixed, g_w3_scale, g_bias3_q, TS_L3_OC);

    g_collect_calib = 0;
    g_fixed_qparams_valid = 1;
    return 1;
}

int tinyspeech_int8_fixed_qparams_ready(void) {
    return g_fixed_qparams_valid;
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

    const float s0 = 1.0f;
    const int use_fixed = g_fixed_qparams_valid;
#if defined(__riscv_vector)
    const int16_t *w1_conv16 = g_w1_pack16;
    const int16_t *w2_conv16 = g_w2_pack16;
    const int16_t *w3_conv16 = g_w3_pack16;
#else
    const int8_t *w1_conv8 = g_w1_q;
    const int8_t *w2_conv8 = g_w2_q;
    const int8_t *w3_conv8 = g_w3_q;
    const int16_t *w1_conv16 = NULL;
    const int16_t *w2_conv16 = NULL;
    const int16_t *w3_conv16 = NULL;
#endif

    t0 = rdcycle64_int8();
    if (!use_fixed) {
        make_bias_q(conv1_bias, s0, g_w1_scale, g_bias1_q, TS_L1_OC);
    }
    pad_input_1ch(g_in0, TS_IN_H, TS_IN_W, g_pad1, TS_IN_H + 2, TS_IN_W + 2);
    float s1 = 1.0f;
#if defined(__riscv_vector)
    conv3x3_pool2x2_acc_1c_rvv(g_pad1, TS_IN_W + 2, w1_conv16, g_bias1_q,
                               TS_L1_OC, TS_L1_PH, TS_L1_PW, g_pool1_acc);
    if (use_fixed) {
        requant_relu_from_acc_to_padded_fixed(g_pool1_acc, TS_L1_OC, TS_L1_PH, TS_L1_PW,
                                              g_mul1_q31, g_pad2, TS_L2_OH + 2, TS_L2_OW + 2);
        s1 = g_s1_fixed;
    } else {
        int32_t max1 = 0;
        s1 = requant_relu_from_acc_to_padded(g_pool1_acc, TS_L1_OC, TS_L1_PH, TS_L1_PW,
                                             s0, g_w1_scale, g_pad2, TS_L2_OH + 2, TS_L2_OW + 2, &max1);
        calib_track_max(1, max1);
    }
#else
    conv3x3_acc_1c(g_pad1, TS_IN_W + 2, w1_conv8, w1_conv16, g_bias1_q, TS_L1_OC, TS_L1_OH, TS_L1_OW, g_conv1_acc);
    if (use_fixed) {
        requant_pool_relu_fixed(g_conv1_acc, TS_L1_OC, TS_L1_OH, TS_L1_OW,
                                TS_L1_PH, TS_L1_PW, g_mul1_q31, g_act1);
        s1 = g_s1_fixed;
    } else {
        int32_t max1 = 0;
        s1 = requant_pool_relu(g_conv1_acc, TS_L1_OC, TS_L1_OH, TS_L1_OW,
                               TS_L1_PH, TS_L1_PW, s0, g_w1_scale, g_act1, &max1);
        calib_track_max(1, max1);
    }
#endif
    t1 = rdcycle64_int8();
    profile->conv1_pool1 = t1 - t0;

    t0 = rdcycle64_int8();
    if (!use_fixed) {
        make_bias_q(conv2_bias, s1, g_w2_scale, g_bias2_q, TS_L2_OC);
    }
#if !defined(__riscv_vector)
    pad_input_c(g_act1, TS_L2_IC, TS_L2_OH, TS_L2_OW, g_pad2, TS_L2_OH + 2, TS_L2_OW + 2);
#endif
    float s2 = 1.0f;
#if defined(__riscv_vector)
    if (use_fixed) {
        conv3x3_pool2x2_requant_relu_to_padded_c_rvv(g_pad2, TS_L2_IC, TS_L2_OH + 2, TS_L2_OW + 2,
                                                     w2_conv16, g_bias2_q, TS_L2_OC, TS_L2_PH, TS_L2_PW,
                                                     g_mul2_q31, g_pad3, TS_L3_OH + 2, TS_L3_OW + 2);
        s2 = g_s2_fixed;
    } else {
        int32_t max2 = 0;
        conv3x3_pool2x2_acc_c_rvv(g_pad2, TS_L2_IC, TS_L2_OH + 2, TS_L2_OW + 2,
                                  w2_conv16, g_bias2_q, TS_L2_OC, TS_L2_PH, TS_L2_PW, g_pool2_acc);
        s2 = requant_relu_from_acc_to_padded(g_pool2_acc, TS_L2_OC, TS_L2_PH, TS_L2_PW,
                                             s1, g_w2_scale, g_pad3, TS_L3_OH + 2, TS_L3_OW + 2, &max2);
        calib_track_max(2, max2);
    }
#else
    conv3x3_acc_c(g_pad2, TS_L2_IC, TS_L2_OH + 2, TS_L2_OW + 2, w2_conv8, w2_conv16, g_bias2_q, TS_L2_OC, TS_L2_OH, TS_L2_OW, g_conv2_acc);
    if (use_fixed) {
        requant_pool_relu_fixed(g_conv2_acc, TS_L2_OC, TS_L2_OH, TS_L2_OW,
                                TS_L2_PH, TS_L2_PW, g_mul2_q31, g_act2);
        s2 = g_s2_fixed;
    } else {
        int32_t max2 = 0;
        s2 = requant_pool_relu(g_conv2_acc, TS_L2_OC, TS_L2_OH, TS_L2_OW,
                               TS_L2_PH, TS_L2_PW, s1, g_w2_scale, g_act2, &max2);
        calib_track_max(2, max2);
    }
#endif
    t1 = rdcycle64_int8();
    profile->conv2_pool2 = t1 - t0;

    t0 = rdcycle64_int8();
    if (!use_fixed) {
        make_bias_q(conv3_bias, s2, g_w3_scale, g_bias3_q, TS_L3_OC);
    }
    float s3 = 1.0f;
#if defined(__riscv_vector)
    conv3x3_relu_gap_acc_c_rvv(g_pad3, TS_L3_IC, TS_L3_OH + 2, TS_L3_OW + 2,
                               w3_conv16, g_bias3_q, TS_L3_OC, TS_L3_OH, TS_L3_OW, g_gap3_acc);
    if (use_fixed) {
        for (int32_t oc = 0; oc < TS_L3_OC; oc++) {
            int32_t avg_acc = (g_gap3_acc[oc] + (TS_GAP_AREA / 2)) / TS_GAP_AREA;
            g_act3[oc] = requant_u7_from_acc(avg_acc, g_mul3_q31);
        }
        s3 = g_s3_fixed;
    } else {
        int32_t max_avg = 0;
        for (int32_t oc = 0; oc < TS_L3_OC; oc++) {
            int32_t avg_acc = (g_gap3_acc[oc] + (TS_GAP_AREA / 2)) / TS_GAP_AREA;
            g_gap3_acc[oc] = avg_acc;
            if (avg_acc > max_avg) {
                max_avg = avg_acc;
            }
        }
        s3 = out_scale_from_max(s2, g_w3_scale, max_avg);
        uint32_t mul3_q31 = requant_mul_q31_from_max(max_avg);
        for (int32_t i = 0; i < TS_L3_OC; i++) {
            g_act3[i] = requant_u7_from_acc(g_gap3_acc[i], mul3_q31);
        }
        calib_track_max(3, max_avg);
    }
#else
    if (use_fixed) {
        pad_input_c(g_act2, TS_L3_IC, TS_L3_OH, TS_L3_OW, g_pad3, TS_L3_OH + 2, TS_L3_OW + 2);
        conv3x3_acc_c(g_pad3, TS_L3_IC, TS_L3_OH + 2, TS_L3_OW + 2, w3_conv8, w3_conv16, g_bias3_q,
                      TS_L3_OC, TS_L3_OH, TS_L3_OW, g_conv3_acc);
        for (int32_t oc = 0; oc < TS_L3_OC; oc++) {
            int64_t sum_pos = 0;
            const int32_t *src = g_conv3_acc + oc * (TS_L3_OH * TS_L3_OW);
            for (int32_t i = 0; i < TS_L3_OH * TS_L3_OW; i++) {
                int32_t acc = src[i];
                if (acc > 0) {
                    sum_pos += acc;
                }
            }
            int32_t avg_acc = (int32_t)((sum_pos + (TS_GAP_AREA / 2)) / TS_GAP_AREA);
            g_act3[oc] = requant_u7_from_acc(avg_acc, g_mul3_q31);
        }
        s3 = g_s3_fixed;
    } else {
        int32_t max3 = 0;
        s3 = conv_relu_gap_to_i8(g_act2, s2, w3_conv8, w3_conv16, g_w3_scale, g_bias3_q, g_act3, &max3);
        calib_track_max(3, max3);
    }
#endif
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
