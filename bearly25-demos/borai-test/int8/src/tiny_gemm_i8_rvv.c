#include "tiny_gemm_i8_rvv.h"

#include <stddef.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>

static inline void qgemm_k64_m8(
    const int8_t* a,
    const int8_t* w_t_pack,
    float* c,
    int n_out,
    float scale)
{
    const size_t stride = (size_t)n_out;
    const int8_t* w_data = w_t_pack + stride; /* skip bias row */
    size_t off = 0;

    while (off < (size_t)n_out) {
        size_t vl = __riscv_vsetvl_e32m8((size_t)n_out - off);
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);
        const int8_t* bp = w_data + off;
        int k = 0;
        for (; k + 3 < 64; k += 4) {
            vint16m4_t vb0 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp, vl), vl);
            vint16m4_t vb1 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp + stride, vl), vl);
            vint16m4_t vb2 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp + (stride * 2), vl), vl);
            vint16m4_t vb3 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp + (stride * 3), vl), vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 0], vb0, vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 1], vb1, vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 2], vb2, vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 3], vb3, vl);
            bp += stride * 4;
        }
        for (; k < 64; k++) {
            vint16m4_t vb = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp, vl), vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k], vb, vl);
            bp += stride;
        }
        vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(acc, vl);
        vf = __riscv_vfmul_vf_f32m8(vf, scale, vl);
        __riscv_vse32_v_f32m8(c + off, vf, vl);
        off += vl;
    }
}

static inline void qgemm_k64_m4(
    const int8_t* a,
    const int8_t* w_t_pack,
    float* c,
    int n_out,
    float scale)
{
    const size_t stride = (size_t)n_out;
    const int8_t* w_data = w_t_pack + stride; /* skip bias row */
    size_t off = 0;

    while (off < (size_t)n_out) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)n_out - off);
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
        const int8_t* bp = w_data + off;
        int k = 0;
        for (; k + 3 < 64; k += 4) {
            vint16m2_t vb0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp, vl), vl);
            vint16m2_t vb1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp + stride, vl), vl);
            vint16m2_t vb2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp + (stride * 2), vl), vl);
            vint16m2_t vb3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp + (stride * 3), vl), vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 0], vb0, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 1], vb1, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 2], vb2, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 3], vb3, vl);
            bp += stride * 4;
        }
        for (; k < 64; k++) {
            vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp, vl), vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k], vb, vl);
            bp += stride;
        }
        vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(acc, vl);
        vf = __riscv_vfmul_vf_f32m4(vf, scale, vl);
        __riscv_vse32_v_f32m4(c + off, vf, vl);
        off += vl;
    }
}

static inline void qgemm_k172_m8(
    const int8_t* a,
    const int8_t* w_t_pack,
    float* c,
    int n_out,
    float scale)
{
    const size_t stride = (size_t)n_out;
    const int8_t* w_data = w_t_pack + stride; /* skip bias row */
    size_t off = 0;

    while (off < (size_t)n_out) {
        size_t vl = __riscv_vsetvl_e32m8((size_t)n_out - off);
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vl);
        const int8_t* bp = w_data + off;
        int k = 0;
        for (; k + 3 < 172; k += 4) {
            vint16m4_t vb0 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp, vl), vl);
            vint16m4_t vb1 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp + stride, vl), vl);
            vint16m4_t vb2 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp + (stride * 2), vl), vl);
            vint16m4_t vb3 = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp + (stride * 3), vl), vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 0], vb0, vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 1], vb1, vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 2], vb2, vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k + 3], vb3, vl);
            bp += stride * 4;
        }
        for (; k < 172; k++) {
            vint16m4_t vb = __riscv_vwcvt_x_x_v_i16m4(__riscv_vle8_v_i8m2(bp, vl), vl);
            acc = __riscv_vwmacc_vx_i32m8(acc, a[k], vb, vl);
            bp += stride;
        }
        vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(acc, vl);
        vf = __riscv_vfmul_vf_f32m8(vf, scale, vl);
        __riscv_vse32_v_f32m8(c + off, vf, vl);
        off += vl;
    }
}

static inline void qgemm_k172_m4(
    const int8_t* a,
    const int8_t* w_t_pack,
    float* c,
    int n_out,
    float scale)
{
    const size_t stride = (size_t)n_out;
    const int8_t* w_data = w_t_pack + stride; /* skip bias row */
    size_t off = 0;

    while (off < (size_t)n_out) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)n_out - off);
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
        const int8_t* bp = w_data + off;
        int k = 0;
        for (; k + 3 < 172; k += 4) {
            vint16m2_t vb0 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp, vl), vl);
            vint16m2_t vb1 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp + stride, vl), vl);
            vint16m2_t vb2 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp + (stride * 2), vl), vl);
            vint16m2_t vb3 = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp + (stride * 3), vl), vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 0], vb0, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 1], vb1, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 2], vb2, vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k + 3], vb3, vl);
            bp += stride * 4;
        }
        for (; k < 172; k++) {
            vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(bp, vl), vl);
            acc = __riscv_vwmacc_vx_i32m4(acc, a[k], vb, vl);
            bp += stride;
        }
        vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(acc, vl);
        vf = __riscv_vfmul_vf_f32m4(vf, scale, vl);
        __riscv_vse32_v_f32m4(c + off, vf, vl);
        off += vl;
    }
}

/* Exact-shape entrypoints used by borai-test (dim=64, hidden=172, vocab=512). */
static inline void qgemm_k64_n32(const int8_t* a, const int8_t* w_t_pack, float* c, float scale) {
    qgemm_k64_m4(a, w_t_pack, c, 32, scale);
}

static inline void qgemm_k64_n64(const int8_t* a, const int8_t* w_t_pack, float* c, float scale) {
    qgemm_k64_m8(a, w_t_pack, c, 64, scale);
}

static inline void qgemm_k64_n172(const int8_t* a, const int8_t* w_t_pack, float* c, float scale) {
    qgemm_k64_m8(a, w_t_pack, c, 172, scale);
}

static inline void qgemm_k64_n256(const int8_t* a, const int8_t* w_t_pack, float* c, float scale) {
    qgemm_k64_m8(a, w_t_pack, c, 256, scale);
}

static inline void qgemm_k64_n512(const int8_t* a, const int8_t* w_t_pack, float* c, float scale) {
    qgemm_k64_m8(a, w_t_pack, c, 512, scale);
}

static inline void qgemm_k172_n64(const int8_t* a, const int8_t* w_t_pack, float* c, float scale) {
    qgemm_k172_m8(a, w_t_pack, c, 64, scale);
}

#endif /* __riscv_vector */

int borai_tiny_matmul_t_i8_fout(
    const int8_t* xq,
    const int8_t* w_t_pack,
    float* xout,
    int n_in,
    int n_out,
    float scale)
{
#if defined(__riscv_vector)
    if (n_in == 64) {
        switch (n_out) {
            case 32:
                qgemm_k64_n32(xq, w_t_pack, xout, scale);
                return 1;
            case 64:
                qgemm_k64_n64(xq, w_t_pack, xout, scale);
                return 1;
            case 172:
                qgemm_k64_n172(xq, w_t_pack, xout, scale);
                return 1;
            case 256:
                qgemm_k64_n256(xq, w_t_pack, xout, scale);
                return 1;
            case 512:
                qgemm_k64_n512(xq, w_t_pack, xout, scale);
                return 1;
            default:
                break;
        }
        if (n_out >= 64) {
            qgemm_k64_m8(xq, w_t_pack, xout, n_out, scale);
        } else {
            qgemm_k64_m4(xq, w_t_pack, xout, n_out, scale);
        }
        return 1;
    }
    if (n_in == 172) {
        if (n_out == 64) {
            qgemm_k172_n64(xq, w_t_pack, xout, scale);
            return 1;
        }
        if (n_out >= 64) {
            qgemm_k172_m8(xq, w_t_pack, xout, n_out, scale);
        } else {
            qgemm_k172_m4(xq, w_t_pack, xout, n_out, scale);
        }
        return 1;
    }
#else
    (void)xq;
    (void)w_t_pack;
    (void)xout;
    (void)n_in;
    (void)n_out;
    (void)scale;
#endif
    return 0;
}
