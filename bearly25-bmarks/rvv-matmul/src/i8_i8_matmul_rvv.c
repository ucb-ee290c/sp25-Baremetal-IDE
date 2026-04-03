#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <riscv_vector.h>

// 15x(m2) int8->int8 microkernel, no widening, overflow wraps.
// 15 accumulators × 2 regs + 1 weight × 2 regs = 32 regs.
void gemm_i8_i8_15xm2_packed(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int8_t* a0  = a;
  int8_t* c0  = c;

  const int8_t* a1  = (const int8_t*) ((uintptr_t) a0  + a_stride);
  int8_t* c1  = (int8_t*) ((uintptr_t) c0  + cm_stride);

  const int8_t* a2  = (const int8_t*) ((uintptr_t) a1  + a_stride);
  int8_t* c2  = (int8_t*) ((uintptr_t) c1  + cm_stride);

  const int8_t* a3  = (const int8_t*) ((uintptr_t) a2  + a_stride);
  int8_t* c3  = (int8_t*) ((uintptr_t) c2  + cm_stride);

  const int8_t* a4  = (const int8_t*) ((uintptr_t) a3  + a_stride);
  int8_t* c4  = (int8_t*) ((uintptr_t) c3  + cm_stride);

  const int8_t* a5  = (const int8_t*) ((uintptr_t) a4  + a_stride);
  int8_t* c5  = (int8_t*) ((uintptr_t) c4  + cm_stride);

  const int8_t* a6  = (const int8_t*) ((uintptr_t) a5  + a_stride);
  int8_t* c6  = (int8_t*) ((uintptr_t) c5  + cm_stride);

  const int8_t* a7  = (const int8_t*) ((uintptr_t) a6  + a_stride);
  int8_t* c7  = (int8_t*) ((uintptr_t) c6  + cm_stride);

  const int8_t* a8  = (const int8_t*) ((uintptr_t) a7  + a_stride);
  int8_t* c8  = (int8_t*) ((uintptr_t) c7  + cm_stride);

  const int8_t* a9  = (const int8_t*) ((uintptr_t) a8  + a_stride);
  int8_t* c9  = (int8_t*) ((uintptr_t) c8  + cm_stride);

  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9  + a_stride);
  int8_t* c10 = (int8_t*) ((uintptr_t) c9  + cm_stride);

  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  int8_t* c11 = (int8_t*) ((uintptr_t) c10 + cm_stride);

  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  int8_t* c12 = (int8_t*) ((uintptr_t) c11 + cm_stride);

  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  int8_t* c13 = (int8_t*) ((uintptr_t) c12 + cm_stride);

  const int8_t* a14 = (const int8_t*) ((uintptr_t) a13 + a_stride);
  int8_t* c14 = (int8_t*) ((uintptr_t) c13 + cm_stride);

  size_t nr = __riscv_vsetvlmax_e8m2();
  nr = (nc > nr) ? nr : nc;
  size_t vl = nr;

  do {
    if (nc < nr) {
      vl = __riscv_vsetvl_e8m2(nc);
      nr = nc;
    }
    nc -= vl;

    vint8m2_t vacc0  = __riscv_vle8_v_i8m2(w, vl);
    w = w + nr;

    vint8m2_t vacc1  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc2  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc3  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc4  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc5  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc6  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc7  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc8  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc9  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc10 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc11 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc12 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc13 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc14 = __riscv_vmv_v_v_i8m2(vacc0, vl);

    size_t k = kc;
    do {
      const int8_t va0  = *a0++;
      const int8_t va1  = *a1++;
      const int8_t va2  = *a2++;
      const int8_t va3  = *a3++;
      const int8_t va4  = *a4++;
      const int8_t va5  = *a5++;
      const int8_t va6  = *a6++;
      const int8_t va7  = *a7++;
      const int8_t va8  = *a8++;
      const int8_t va9  = *a9++;
      const int8_t va10 = *a10++;
      const int8_t va11 = *a11++;
      const int8_t va12 = *a12++;
      const int8_t va13 = *a13++;
      const int8_t va14 = *a14++;

      vint8m2_t vb = __riscv_vle8_v_i8m2(w, vl);
      w = w + nr;

      vacc0  = __riscv_vmacc_vx_i8m2(vacc0,  va0,  vb, vl);
      vacc1  = __riscv_vmacc_vx_i8m2(vacc1,  va1,  vb, vl);
      vacc2  = __riscv_vmacc_vx_i8m2(vacc2,  va2,  vb, vl);
      vacc3  = __riscv_vmacc_vx_i8m2(vacc3,  va3,  vb, vl);
      vacc4  = __riscv_vmacc_vx_i8m2(vacc4,  va4,  vb, vl);
      vacc5  = __riscv_vmacc_vx_i8m2(vacc5,  va5,  vb, vl);
      vacc6  = __riscv_vmacc_vx_i8m2(vacc6,  va6,  vb, vl);
      vacc7  = __riscv_vmacc_vx_i8m2(vacc7,  va7,  vb, vl);
      vacc8  = __riscv_vmacc_vx_i8m2(vacc8,  va8,  vb, vl);
      vacc9  = __riscv_vmacc_vx_i8m2(vacc9,  va9,  vb, vl);
      vacc10 = __riscv_vmacc_vx_i8m2(vacc10, va10, vb, vl);
      vacc11 = __riscv_vmacc_vx_i8m2(vacc11, va11, vb, vl);
      vacc12 = __riscv_vmacc_vx_i8m2(vacc12, va12, vb, vl);
      vacc13 = __riscv_vmacc_vx_i8m2(vacc13, va13, vb, vl);
      vacc14 = __riscv_vmacc_vx_i8m2(vacc14, va14, vb, vl);

      k -= 1;
    } while (k != 0);

    a0  -= kc;  a1  -= kc;  a2  -= kc;  a3  -= kc;
    a4  -= kc;  a5  -= kc;  a6  -= kc;  a7  -= kc;
    a8  -= kc;  a9  -= kc;  a10 -= kc;  a11 -= kc;
    a12 -= kc;  a13 -= kc;  a14 -= kc;

    __riscv_vse8_v_i8m2(c0,  vacc0,  vl); c0  += vl;
    __riscv_vse8_v_i8m2(c1,  vacc1,  vl); c1  += vl;
    __riscv_vse8_v_i8m2(c2,  vacc2,  vl); c2  += vl;
    __riscv_vse8_v_i8m2(c3,  vacc3,  vl); c3  += vl;
    __riscv_vse8_v_i8m2(c4,  vacc4,  vl); c4  += vl;
    __riscv_vse8_v_i8m2(c5,  vacc5,  vl); c5  += vl;
    __riscv_vse8_v_i8m2(c6,  vacc6,  vl); c6  += vl;
    __riscv_vse8_v_i8m2(c7,  vacc7,  vl); c7  += vl;
    __riscv_vse8_v_i8m2(c8,  vacc8,  vl); c8  += vl;
    __riscv_vse8_v_i8m2(c9,  vacc9,  vl); c9  += vl;
    __riscv_vse8_v_i8m2(c10, vacc10, vl); c10 += vl;
    __riscv_vse8_v_i8m2(c11, vacc11, vl); c11 += vl;
    __riscv_vse8_v_i8m2(c12, vacc12, vl); c12 += vl;
    __riscv_vse8_v_i8m2(c13, vacc13, vl); c13 += vl;
    __riscv_vse8_v_i8m2(c14, vacc14, vl); c14 += vl;
  } while (nc != 0);
}

// 4x(m2) tail microkernel
void gemm_i8_i8_4xm2_packed(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);

  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);

  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);

  size_t nr = __riscv_vsetvlmax_e8m2();
  nr = (nc > nr) ? nr : nc;
  size_t vl = nr;

  do {
    if (nc < nr) {
      vl = __riscv_vsetvl_e8m2(nc);
      nr = nc;
    }
    nc -= vl;

    vint8m2_t vacc0 = __riscv_vle8_v_i8m2(w, vl);
    w = w + nr;

    vint8m2_t vacc1 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc2 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc3 = __riscv_vmv_v_v_i8m2(vacc0, vl);

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      const int8_t va1 = *a1++;
      const int8_t va2 = *a2++;
      const int8_t va3 = *a3++;

      vint8m2_t vb = __riscv_vle8_v_i8m2(w, vl);
      w = w + nr;

      vacc0 = __riscv_vmacc_vx_i8m2(vacc0, va0, vb, vl);
      vacc1 = __riscv_vmacc_vx_i8m2(vacc1, va1, vb, vl);
      vacc2 = __riscv_vmacc_vx_i8m2(vacc2, va2, vb, vl);
      vacc3 = __riscv_vmacc_vx_i8m2(vacc3, va3, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;  a1 -= kc;  a2 -= kc;  a3 -= kc;

    __riscv_vse8_v_i8m2(c0, vacc0, vl); c0 += vl;
    __riscv_vse8_v_i8m2(c1, vacc1, vl); c1 += vl;
    __riscv_vse8_v_i8m2(c2, vacc2, vl); c2 += vl;
    __riscv_vse8_v_i8m2(c3, vacc3, vl); c3 += vl;
  } while (nc != 0);
}

// 1x(m2) fallback for remainders < 4
void gemm_i8_i8_1xm2_packed(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int8_t* a0 = a;
  int8_t* c0 = c;

  size_t nr = __riscv_vsetvlmax_e8m2();
  nr = (nc > nr) ? nr : nc;
  size_t vl = nr;

  do {
    if (nc < nr) {
      vl = __riscv_vsetvl_e8m2(nc);
      nr = nc;
    }
    nc -= vl;

    vint8m2_t vacc0 = __riscv_vle8_v_i8m2(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      vint8m2_t vb = __riscv_vle8_v_i8m2(w, vl);
      w = w + nr;
      vacc0 = __riscv_vmacc_vx_i8m2(vacc0, va0, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;

    __riscv_vse8_v_i8m2(c0, vacc0, vl);
    c0 += vl;

  } while (nc != 0);
}

size_t packed_nr_i8_i8(void) {
    return __riscv_vsetvlmax_e8m2();
}

void int8_int8_gemm_packed(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride;
    const size_t cm_stride_bytes = c_row_stride * sizeof(int8_t);
    const size_t cn_stride_bytes = c_col_stride;

    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 15) {
            gemm_i8_i8_15xm2_packed(
                15, N, kc_bytes,
                A + row * a_row_stride, a_stride_bytes,
                B,
                C + row * c_row_stride, cm_stride_bytes, cn_stride_bytes
            );
            row += 15;
        } else if (rows_left >= 4) {
            gemm_i8_i8_4xm2_packed(
                4, N, kc_bytes,
                A + row * a_row_stride, a_stride_bytes,
                B,
                C + row * c_row_stride, cm_stride_bytes, cn_stride_bytes
            );
            row += 4;
        } else {
            gemm_i8_i8_1xm2_packed(
                1, N, kc_bytes,
                A + row * a_row_stride, a_stride_bytes,
                B,
                C + row * c_row_stride, cm_stride_bytes, cn_stride_bytes
            );
            row += 1;
        }
    }
}

void pack_weight_matrix(
    size_t K,
    size_t N,
    const int8_t* B,
    int8_t* B_packed)
  {
    size_t nr = __riscv_vsetvlmax_e8m2();
    nr = (N < nr) ? N : nr;
    int i = 0;

    for (size_t nc = 0; nc < N; nc += nr){
      if (N - nc < nr) {
        nr = N - nc;
      }

      for (size_t row = 0; row < K + 1; row++) {
        for (size_t c = 0; c < nr; c++) {
          B_packed[i] = B[row * N + nc + c];
          i ++;
        }
      }
    }
  }

// Unpacked variants

void gemm_i8_i8_15xm2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int8_t* a0  = a;
  int8_t* c0  = c;

  const int8_t* a1  = (const int8_t*) ((uintptr_t) a0  + a_stride);
  int8_t* c1  = (int8_t*) ((uintptr_t) c0  + cm_stride);

  const int8_t* a2  = (const int8_t*) ((uintptr_t) a1  + a_stride);
  int8_t* c2  = (int8_t*) ((uintptr_t) c1  + cm_stride);

  const int8_t* a3  = (const int8_t*) ((uintptr_t) a2  + a_stride);
  int8_t* c3  = (int8_t*) ((uintptr_t) c2  + cm_stride);

  const int8_t* a4  = (const int8_t*) ((uintptr_t) a3  + a_stride);
  int8_t* c4  = (int8_t*) ((uintptr_t) c3  + cm_stride);

  const int8_t* a5  = (const int8_t*) ((uintptr_t) a4  + a_stride);
  int8_t* c5  = (int8_t*) ((uintptr_t) c4  + cm_stride);

  const int8_t* a6  = (const int8_t*) ((uintptr_t) a5  + a_stride);
  int8_t* c6  = (int8_t*) ((uintptr_t) c5  + cm_stride);

  const int8_t* a7  = (const int8_t*) ((uintptr_t) a6  + a_stride);
  int8_t* c7  = (int8_t*) ((uintptr_t) c6  + cm_stride);

  const int8_t* a8  = (const int8_t*) ((uintptr_t) a7  + a_stride);
  int8_t* c8  = (int8_t*) ((uintptr_t) c7  + cm_stride);

  const int8_t* a9  = (const int8_t*) ((uintptr_t) a8  + a_stride);
  int8_t* c9  = (int8_t*) ((uintptr_t) c8  + cm_stride);

  const int8_t* a10 = (const int8_t*) ((uintptr_t) a9  + a_stride);
  int8_t* c10 = (int8_t*) ((uintptr_t) c9  + cm_stride);

  const int8_t* a11 = (const int8_t*) ((uintptr_t) a10 + a_stride);
  int8_t* c11 = (int8_t*) ((uintptr_t) c10 + cm_stride);

  const int8_t* a12 = (const int8_t*) ((uintptr_t) a11 + a_stride);
  int8_t* c12 = (int8_t*) ((uintptr_t) c11 + cm_stride);

  const int8_t* a13 = (const int8_t*) ((uintptr_t) a12 + a_stride);
  int8_t* c13 = (int8_t*) ((uintptr_t) c12 + cm_stride);

  const int8_t* a14 = (const int8_t*) ((uintptr_t) a13 + a_stride);
  int8_t* c14 = (int8_t*) ((uintptr_t) c13 + cm_stride);

  size_t nr = nc;
  size_t vl = nr;
  const int8_t* w_new = w;
  do {
    vl = __riscv_vsetvl_e8m2(nc);
    nc -= vl;
    w_new = w + vl;

    vint8m2_t vacc0  = __riscv_vle8_v_i8m2(w, vl);
    w = w + nr;

    vint8m2_t vacc1  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc2  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc3  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc4  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc5  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc6  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc7  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc8  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc9  = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc10 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc11 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc12 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc13 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc14 = __riscv_vmv_v_v_i8m2(vacc0, vl);

    size_t k = kc;
    do {
      const int8_t va0  = *a0++;
      const int8_t va1  = *a1++;
      const int8_t va2  = *a2++;
      const int8_t va3  = *a3++;
      const int8_t va4  = *a4++;
      const int8_t va5  = *a5++;
      const int8_t va6  = *a6++;
      const int8_t va7  = *a7++;
      const int8_t va8  = *a8++;
      const int8_t va9  = *a9++;
      const int8_t va10 = *a10++;
      const int8_t va11 = *a11++;
      const int8_t va12 = *a12++;
      const int8_t va13 = *a13++;
      const int8_t va14 = *a14++;

      vint8m2_t vb = __riscv_vle8_v_i8m2(w, vl);
      w = w + nr;

      vacc0  = __riscv_vmacc_vx_i8m2(vacc0,  va0,  vb, vl);
      vacc1  = __riscv_vmacc_vx_i8m2(vacc1,  va1,  vb, vl);
      vacc2  = __riscv_vmacc_vx_i8m2(vacc2,  va2,  vb, vl);
      vacc3  = __riscv_vmacc_vx_i8m2(vacc3,  va3,  vb, vl);
      vacc4  = __riscv_vmacc_vx_i8m2(vacc4,  va4,  vb, vl);
      vacc5  = __riscv_vmacc_vx_i8m2(vacc5,  va5,  vb, vl);
      vacc6  = __riscv_vmacc_vx_i8m2(vacc6,  va6,  vb, vl);
      vacc7  = __riscv_vmacc_vx_i8m2(vacc7,  va7,  vb, vl);
      vacc8  = __riscv_vmacc_vx_i8m2(vacc8,  va8,  vb, vl);
      vacc9  = __riscv_vmacc_vx_i8m2(vacc9,  va9,  vb, vl);
      vacc10 = __riscv_vmacc_vx_i8m2(vacc10, va10, vb, vl);
      vacc11 = __riscv_vmacc_vx_i8m2(vacc11, va11, vb, vl);
      vacc12 = __riscv_vmacc_vx_i8m2(vacc12, va12, vb, vl);
      vacc13 = __riscv_vmacc_vx_i8m2(vacc13, va13, vb, vl);
      vacc14 = __riscv_vmacc_vx_i8m2(vacc14, va14, vb, vl);

      k -= 1;
    } while (k != 0);

    a0  -= kc;  a1  -= kc;  a2  -= kc;  a3  -= kc;
    a4  -= kc;  a5  -= kc;  a6  -= kc;  a7  -= kc;
    a8  -= kc;  a9  -= kc;  a10 -= kc;  a11 -= kc;
    a12 -= kc;  a13 -= kc;  a14 -= kc;

    __riscv_vse8_v_i8m2(c0,  vacc0,  vl); c0  += vl;
    __riscv_vse8_v_i8m2(c1,  vacc1,  vl); c1  += vl;
    __riscv_vse8_v_i8m2(c2,  vacc2,  vl); c2  += vl;
    __riscv_vse8_v_i8m2(c3,  vacc3,  vl); c3  += vl;
    __riscv_vse8_v_i8m2(c4,  vacc4,  vl); c4  += vl;
    __riscv_vse8_v_i8m2(c5,  vacc5,  vl); c5  += vl;
    __riscv_vse8_v_i8m2(c6,  vacc6,  vl); c6  += vl;
    __riscv_vse8_v_i8m2(c7,  vacc7,  vl); c7  += vl;
    __riscv_vse8_v_i8m2(c8,  vacc8,  vl); c8  += vl;
    __riscv_vse8_v_i8m2(c9,  vacc9,  vl); c9  += vl;
    __riscv_vse8_v_i8m2(c10, vacc10, vl); c10 += vl;
    __riscv_vse8_v_i8m2(c11, vacc11, vl); c11 += vl;
    __riscv_vse8_v_i8m2(c12, vacc12, vl); c12 += vl;
    __riscv_vse8_v_i8m2(c13, vacc13, vl); c13 += vl;
    __riscv_vse8_v_i8m2(c14, vacc14, vl); c14 += vl;
    w = w_new;
  } while (nc != 0);
}

void gemm_i8_i8_4xm2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);

  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);

  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);

  size_t nr = nc;
  size_t vl = nr;
  const int8_t* w_new = w;
  do {
    vl = __riscv_vsetvl_e8m2(nc);
    nc -= vl;
    w_new = w + vl;

    vint8m2_t vacc0 = __riscv_vle8_v_i8m2(w, vl);
    w = w + nr;

    vint8m2_t vacc1 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc2 = __riscv_vmv_v_v_i8m2(vacc0, vl);
    vint8m2_t vacc3 = __riscv_vmv_v_v_i8m2(vacc0, vl);

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      const int8_t va1 = *a1++;
      const int8_t va2 = *a2++;
      const int8_t va3 = *a3++;

      vint8m2_t vb = __riscv_vle8_v_i8m2(w, vl);
      w = w + nr;

      vacc0 = __riscv_vmacc_vx_i8m2(vacc0, va0, vb, vl);
      vacc1 = __riscv_vmacc_vx_i8m2(vacc1, va1, vb, vl);
      vacc2 = __riscv_vmacc_vx_i8m2(vacc2, va2, vb, vl);
      vacc3 = __riscv_vmacc_vx_i8m2(vacc3, va3, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;  a1 -= kc;  a2 -= kc;  a3 -= kc;

    __riscv_vse8_v_i8m2(c0, vacc0, vl); c0 += vl;
    __riscv_vse8_v_i8m2(c1, vacc1, vl); c1 += vl;
    __riscv_vse8_v_i8m2(c2, vacc2, vl); c2 += vl;
    __riscv_vse8_v_i8m2(c3, vacc3, vl); c3 += vl;
    w = w_new;
  } while (nc != 0);
}

void gemm_i8_i8_1xm2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const int8_t* w,
    int8_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int8_t* a0 = a;
  int8_t* c0 = c;

  size_t nr = nc;
  size_t vl = nr;
  const int8_t* w_new = w;
  do {
    vl = __riscv_vsetvl_e8m2(nc);
    nc -= vl;
    w_new = w + vl;

    vint8m2_t vacc0 = __riscv_vle8_v_i8m2(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      vint8m2_t vb = __riscv_vle8_v_i8m2(w, vl);
      w = w + nr;
      vacc0 = __riscv_vmacc_vx_i8m2(vacc0, va0, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;

    __riscv_vse8_v_i8m2(c0, vacc0, vl);
    c0 += vl;
    w = w_new;
  } while (nc != 0);
}

void int8_int8_gemm(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride;
    const size_t cm_stride_bytes = c_row_stride * sizeof(int8_t);
    const size_t cn_stride_bytes = c_col_stride;

    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 15) {
            gemm_i8_i8_15xm2(
                15, N, kc_bytes,
                A + row * a_row_stride, a_stride_bytes,
                B,
                C + row * c_row_stride, cm_stride_bytes, cn_stride_bytes
            );
            row += 15;
        } else if (rows_left >= 4) {
            gemm_i8_i8_4xm2(
                4, N, kc_bytes,
                A + row * a_row_stride, a_stride_bytes,
                B,
                C + row * c_row_stride, cm_stride_bytes, cn_stride_bytes
            );
            row += 4;
        } else {
            gemm_i8_i8_1xm2(
                1, N, kc_bytes,
                A + row * a_row_stride, a_stride_bytes,
                B,
                C + row * c_row_stride, cm_stride_bytes, cn_stride_bytes
            );
            row += 1;
        }
    }
}
