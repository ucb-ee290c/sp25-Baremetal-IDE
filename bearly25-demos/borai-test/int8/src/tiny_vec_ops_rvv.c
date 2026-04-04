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
