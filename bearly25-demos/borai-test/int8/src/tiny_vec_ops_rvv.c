#include "tiny_vec_ops_rvv.h"

#include <math.h>
#include <string.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

static inline int borai_is_tiny_vec_n(int n) {
    return (n == 64) || (n == 172);
}

int borai_tiny_rmsnorm_f32(float* o, const float* x, const float* weight, int n) {
    if (!borai_is_tiny_vec_n(n)) {
        return 0;
    }

#if defined(__riscv_vector)
    int i = 0;
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(n - i));
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vm = __riscv_vfmul_vv_f32m4(vx, vx, vl);
        acc = __riscv_vfredusum_vs_f32m4_f32m1(vm, acc, vl);
        i += (int)vl;
    }
    float ss = __riscv_vfmv_f_s_f32m1_f32(acc);
    ss = ss / (float)n;
    ss = 1.0f / sqrtf(ss + 1e-5f);

    i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(n - i));
        vfloat32m4_t vx = __riscv_vle32_v_f32m4(x + i, vl);
        vfloat32m4_t vw = __riscv_vle32_v_f32m4(weight + i, vl);
        vfloat32m4_t vn = __riscv_vfmul_vf_f32m4(vx, ss, vl);
        vfloat32m4_t vo = __riscv_vfmul_vv_f32m4(vn, vw, vl);
        __riscv_vse32_v_f32m4(o + i, vo, vl);
        i += (int)vl;
    }
    return 1;
#else
    float ss = 0.0f;
    for (int j = 0; j < n; j++) {
        ss += x[j] * x[j];
    }
    ss = 1.0f / sqrtf((ss / (float)n) + 1e-5f);
    for (int j = 0; j < n; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    return 1;
#endif
}

int borai_tiny_quantize_i8(float* out_scale, int8_t* out_q, const float* x, int n) {
    if (!borai_is_tiny_vec_n(n)) {
        return 0;
    }

    float wmax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > wmax) {
            wmax = a;
        }
    }

    if (wmax == 0.0f) {
        *out_scale = 1.0f;
        memset(out_q, 0, (size_t)n * sizeof(int8_t));
        return 1;
    }

    const float qmax = 127.0f;
    const float scale = wmax / qmax;
    const float inv_scale = 1.0f / scale;
    *out_scale = scale;

#if defined(__riscv_vector)
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8((size_t)(n - i));
        vfloat32m8_t v = __riscv_vle32_v_f32m8(x + i, vl);
        v = __riscv_vfmul_vf_f32m8(v, inv_scale, vl);
        v = __riscv_vfmax_vf_f32m8(v, -127.0f, vl);
        v = __riscv_vfmin_vf_f32m8(v, 127.0f, vl);
        vint16m4_t q16 = __riscv_vfncvt_x_f_w_i16m4(v, vl);
        vint8m2_t q8 = __riscv_vncvt_x_x_w_i8m2(q16, vl);
        __riscv_vse8_v_i8m2(out_q + i, q8, vl);
        i += (int)vl;
    }
    return 1;
#else
    for (int i = 0; i < n; i++) {
        int q = (int)roundf(x[i] * inv_scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out_q[i] = (int8_t)q;
    }
    return 1;
#endif
}

int borai_tiny_dot_qk_head_f32(float* out, const float* q, const float* k, int n) {
    if (n != 8) {
        return 0;
    }

    *out =
        q[0] * k[0] + q[1] * k[1] +
        q[2] * k[2] + q[3] * k[3] +
        q[4] * k[4] + q[5] * k[5] +
        q[6] * k[6] + q[7] * k[7];
    return 1;
}

int borai_tiny_axpy_head_f32(float* dst, const float* v, float a, int n) {
    if (n != 8) {
        return 0;
    }

    dst[0] += a * v[0];
    dst[1] += a * v[1];
    dst[2] += a * v[2];
    dst[3] += a * v[3];
    dst[4] += a * v[4];
    dst[5] += a * v[5];
    dst[6] += a * v[6];
    dst[7] += a * v[7];
    return 1;
}

static inline float borai_dot8_f32(const float* a, const float* b) {
    return
        a[0] * b[0] + a[1] * b[1] +
        a[2] * b[2] + a[3] * b[3] +
        a[4] * b[4] + a[5] * b[5] +
        a[6] * b[6] + a[7] * b[7];
}

int borai_tiny_attn_h8_pos012_f32(
    float* xb,
    float* att,
    const float* q,
    const float* key_cache_layer,
    const float* value_cache_layer,
    int pos,
    int kv_dim,
    int kv_head_off,
    float inv_sqrt_head_size)
{
    if (pos < 0 || pos > 2) {
        return 0;
    }

    const float* k0 = key_cache_layer + kv_head_off;
    const float* v0 = value_cache_layer + kv_head_off;

    if (pos == 0) {
        att[0] = 1.0f;
        xb[0] = v0[0]; xb[1] = v0[1]; xb[2] = v0[2]; xb[3] = v0[3];
        xb[4] = v0[4]; xb[5] = v0[5]; xb[6] = v0[6]; xb[7] = v0[7];
        return 1;
    }

    const float* k1 = key_cache_layer + kv_dim + kv_head_off;
    const float* v1 = value_cache_layer + kv_dim + kv_head_off;
    float s0 = borai_dot8_f32(q, k0) * inv_sqrt_head_size;
    float s1 = borai_dot8_f32(q, k1) * inv_sqrt_head_size;

    if (pos == 1) {
        float max_val = s0 > s1 ? s0 : s1;
        float e0 = expf(s0 - max_val);
        float e1 = expf(s1 - max_val);
        float inv_sum = 1.0f / (e0 + e1);
        float a0 = e0 * inv_sum;
        float a1 = e1 * inv_sum;
        att[0] = a0;
        att[1] = a1;
        xb[0] = a0 * v0[0] + a1 * v1[0];
        xb[1] = a0 * v0[1] + a1 * v1[1];
        xb[2] = a0 * v0[2] + a1 * v1[2];
        xb[3] = a0 * v0[3] + a1 * v1[3];
        xb[4] = a0 * v0[4] + a1 * v1[4];
        xb[5] = a0 * v0[5] + a1 * v1[5];
        xb[6] = a0 * v0[6] + a1 * v1[6];
        xb[7] = a0 * v0[7] + a1 * v1[7];
        return 1;
    }

    {
        const float* k2 = key_cache_layer + (kv_dim * 2) + kv_head_off;
        const float* v2 = value_cache_layer + (kv_dim * 2) + kv_head_off;
        float s2 = borai_dot8_f32(q, k2) * inv_sqrt_head_size;

        float max_val = s0;
        if (s1 > max_val) max_val = s1;
        if (s2 > max_val) max_val = s2;

        float e0 = expf(s0 - max_val);
        float e1 = expf(s1 - max_val);
        float e2 = expf(s2 - max_val);
        float inv_sum = 1.0f / (e0 + e1 + e2);
        float a0 = e0 * inv_sum;
        float a1 = e1 * inv_sum;
        float a2 = e2 * inv_sum;

        att[0] = a0;
        att[1] = a1;
        att[2] = a2;

        xb[0] = a0 * v0[0] + a1 * v1[0] + a2 * v2[0];
        xb[1] = a0 * v0[1] + a1 * v1[1] + a2 * v2[1];
        xb[2] = a0 * v0[2] + a1 * v1[2] + a2 * v2[2];
        xb[3] = a0 * v0[3] + a1 * v1[3] + a2 * v2[3];
        xb[4] = a0 * v0[4] + a1 * v1[4] + a2 * v2[4];
        xb[5] = a0 * v0[5] + a1 * v1[5] + a2 * v2[5];
        xb[6] = a0 * v0[6] + a1 * v1[6] + a2 * v2[6];
        xb[7] = a0 * v0[7] + a1 * v1[7] + a2 * v2[7];
    }
    return 1;
}

void borai_swiglu_apply_range(float* hb, const float* hb2, int start, int end) {
    for (int i = start; i < end; i++) {
        float x = hb[i];
#ifdef BORAIQ_FAST_SWIGLU_EXP
        /* Fast sigmoid approximation:
         * sigma(x) ~= 0.5 * (x / (1 + |x|)) + 0.5 */
        float sig = 0.5f * (x / (1.0f + fabsf(x))) + 0.5f;
#else
        float sig = 1.0f / (1.0f + expf(-x));
#endif
        hb[i] = (x * sig) * hb2[i];
    }
}

void borai_swiglu_apply(float* hb, const float* hb2, int n) {
    borai_swiglu_apply_range(hb, hb2, 0, n);
}
