#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <riscv_vector.h>

// 7x(m4) int32->int32 microkernel, non-widening, overflow wraps.
void gemm_i32_7xm4_packed(
    size_t mr,
    size_t nc,
    size_t kc,
    const int32_t* a,
    size_t a_stride,
    const int32_t* w,
    int32_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int32_t* a0 = a;
  int32_t* c0 = c;

  const int32_t* a1 = (const int32_t*) ((uintptr_t) a0 + a_stride);
  int32_t* c1 = (int32_t*) ((uintptr_t) c0 + cm_stride);

  const int32_t* a2 = (const int32_t*) ((uintptr_t) a1 + a_stride);
  int32_t* c2 = (int32_t*) ((uintptr_t) c1 + cm_stride);

  const int32_t* a3 = (const int32_t*) ((uintptr_t) a2 + a_stride);
  int32_t* c3 = (int32_t*) ((uintptr_t) c2 + cm_stride);

  const int32_t* a4 = (const int32_t*) ((uintptr_t) a3 + a_stride);
  int32_t* c4 = (int32_t*) ((uintptr_t) c3 + cm_stride);

  const int32_t* a5 = (const int32_t*) ((uintptr_t) a4 + a_stride);
  int32_t* c5 = (int32_t*) ((uintptr_t) c4 + cm_stride);

  const int32_t* a6 = (const int32_t*) ((uintptr_t) a5 + a_stride);
  int32_t* c6 = (int32_t*) ((uintptr_t) c5 + cm_stride);

  size_t nr = __riscv_vsetvlmax_e32m4();
  nr = (nc > nr) ? nr : nc;
  size_t vl = nr;

  do {
    if (nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
      nr = nc;
    }
    nc -= vl;

    // Load bias row directly as i32
    vint32m4_t vacc0 = __riscv_vle32_v_i32m4(w, vl);
    w = w + nr;

    vint32m4_t vacc1 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc2 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc3 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc4 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc5 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc6 = __riscv_vmv_v_v_i32m4(vacc0, vl);

    size_t k = kc;
    do {
      const int32_t va0 = *a0++;
      const int32_t va1 = *a1++;
      const int32_t va2 = *a2++;
      const int32_t va3 = *a3++;
      const int32_t va4 = *a4++;
      const int32_t va5 = *a5++;
      const int32_t va6 = *a6++;

      vint32m4_t vb = __riscv_vle32_v_i32m4(w, vl);
      w = w + nr;

      // Non-widening multiply-accumulate: i32 * i32 -> i32
      vacc0 = __riscv_vmacc_vx_i32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vmacc_vx_i32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vmacc_vx_i32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vmacc_vx_i32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vmacc_vx_i32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vmacc_vx_i32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vmacc_vx_i32m4(vacc6, va6, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;
    a1 -= kc;
    a2 -= kc;
    a3 -= kc;
    a4 -= kc;
    a5 -= kc;
    a6 -= kc;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;

    __riscv_vse32_v_i32m4(c1, vacc1, vl);
    c1 += vl;

    __riscv_vse32_v_i32m4(c2, vacc2, vl);
    c2 += vl;

    __riscv_vse32_v_i32m4(c3, vacc3, vl);
    c3 += vl;

    __riscv_vse32_v_i32m4(c4, vacc4, vl);
    c4 += vl;

    __riscv_vse32_v_i32m4(c5, vacc5, vl);
    c5 += vl;

    __riscv_vse32_v_i32m4(c6, vacc6, vl);
    c6 += vl;
  } while (nc != 0);
}

void gemm_i32_1xm4_packed(
    size_t mr,
    size_t nc,
    size_t kc,
    const int32_t* a,
    size_t a_stride,
    const int32_t* w,
    int32_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int32_t* a0 = a;
  int32_t* c0 = c;

  size_t nr = __riscv_vsetvlmax_e32m4();
  nr = (nc > nr) ? nr : nc;
  size_t vl = nr;

  do {
    if (nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
      nr = nc;
    }
    nc -= vl;

    vint32m4_t vacc0 = __riscv_vle32_v_i32m4(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const int32_t va0 = *a0++;
      vint32m4_t vb = __riscv_vle32_v_i32m4(w, vl);
      w = w + nr;
      vacc0 = __riscv_vmacc_vx_i32m4(vacc0, va0, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;

  } while (nc != 0);
}

void copy_int32_to_tcm(const int32_t *src, size_t n) {
  int32_t *tcm_input = (int32_t *)0x78000000;

  size_t vl;
  while (n > 0) {
      vl = __riscv_vsetvl_e32m8(n);

      vint32m8_t v = __riscv_vle32_v_i32m8(src, vl);
      __riscv_vse32_v_i32m8(tcm_input, v, vl);

      src       += vl;
      tcm_input += vl;
      n         -= vl;
  }
}

void int32_gemm_packed(
    size_t M, size_t N, size_t K,
    const int32_t* A, size_t a_row_stride,
    const int32_t* B,
    int32_t* C, size_t c_row_stride,
    size_t c_col_stride)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride * sizeof(int32_t);
    const size_t cm_stride_bytes = c_row_stride * sizeof(int32_t);
    const size_t cn_stride_bytes = c_col_stride;

    copy_int32_to_tcm(B, K*N+N);
    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 7) {
            gemm_i32_7xm4_packed(
                7,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                (int32_t *)0x78000000,
                C + row * c_row_stride,
                cm_stride_bytes,
                cn_stride_bytes
            );
            row += 7;
        } else {
            gemm_i32_1xm4_packed(
                1,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                (int32_t *)0x78000000,
                C + row * c_row_stride,
                cm_stride_bytes,
                cn_stride_bytes
            );
            row += 1;
        }
    }
}

void pack_weight_matrix(
    size_t K,
    size_t N,
    const int32_t* B,
    int32_t* B_packed)
  {
    size_t nr = __riscv_vsetvlmax_e32m4();
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

void gemm_i32_7xm4(
    size_t mr,
    size_t nc,
    size_t kc,
    const int32_t* a,
    size_t a_stride,
    const int32_t* w,
    int32_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int32_t* a0 = a;
  int32_t* c0 = c;

  const int32_t* a1 = (const int32_t*) ((uintptr_t) a0 + a_stride);
  int32_t* c1 = (int32_t*) ((uintptr_t) c0 + cm_stride);

  const int32_t* a2 = (const int32_t*) ((uintptr_t) a1 + a_stride);
  int32_t* c2 = (int32_t*) ((uintptr_t) c1 + cm_stride);

  const int32_t* a3 = (const int32_t*) ((uintptr_t) a2 + a_stride);
  int32_t* c3 = (int32_t*) ((uintptr_t) c2 + cm_stride);

  const int32_t* a4 = (const int32_t*) ((uintptr_t) a3 + a_stride);
  int32_t* c4 = (int32_t*) ((uintptr_t) c3 + cm_stride);

  const int32_t* a5 = (const int32_t*) ((uintptr_t) a4 + a_stride);
  int32_t* c5 = (int32_t*) ((uintptr_t) c4 + cm_stride);

  const int32_t* a6 = (const int32_t*) ((uintptr_t) a5 + a_stride);
  int32_t* c6 = (int32_t*) ((uintptr_t) c5 + cm_stride);

  size_t nr = nc;
  size_t vl = nr;
  const int32_t* w_new = w;
  do {
    vl = __riscv_vsetvl_e32m4(nc);
    nc -= vl;
    w_new = w + vl;

    vint32m4_t vacc0 = __riscv_vle32_v_i32m4(w, vl);
    w = w + nr;

    vint32m4_t vacc1 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc2 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc3 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc4 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc5 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc6 = __riscv_vmv_v_v_i32m4(vacc0, vl);

    size_t k = kc;
    do {
      const int32_t va0 = *a0++;
      const int32_t va1 = *a1++;
      const int32_t va2 = *a2++;
      const int32_t va3 = *a3++;
      const int32_t va4 = *a4++;
      const int32_t va5 = *a5++;
      const int32_t va6 = *a6++;

      vint32m4_t vb = __riscv_vle32_v_i32m4(w, vl);
      w = w + nr;

      vacc0 = __riscv_vmacc_vx_i32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vmacc_vx_i32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vmacc_vx_i32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vmacc_vx_i32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vmacc_vx_i32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vmacc_vx_i32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vmacc_vx_i32m4(vacc6, va6, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;
    a1 -= kc;
    a2 -= kc;
    a3 -= kc;
    a4 -= kc;
    a5 -= kc;
    a6 -= kc;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;

    __riscv_vse32_v_i32m4(c1, vacc1, vl);
    c1 += vl;

    __riscv_vse32_v_i32m4(c2, vacc2, vl);
    c2 += vl;

    __riscv_vse32_v_i32m4(c3, vacc3, vl);
    c3 += vl;

    __riscv_vse32_v_i32m4(c4, vacc4, vl);
    c4 += vl;

    __riscv_vse32_v_i32m4(c5, vacc5, vl);
    c5 += vl;

    __riscv_vse32_v_i32m4(c6, vacc6, vl);
    c6 += vl;
    w = w_new;
  } while (nc != 0);
}

void gemm_i32_1xm4(
    size_t mr,
    size_t nc,
    size_t kc,
    const int32_t* a,
    size_t a_stride,
    const int32_t* w,
    int32_t* c,
    size_t cm_stride,
    size_t cn_stride)
{
  const int32_t* a0 = a;
  int32_t* c0 = c;

  size_t nr = nc;
  size_t vl = nr;
  const int32_t* w_new = w;
  do {
    vl = __riscv_vsetvl_e32m4(nc);
    nc -= vl;
    w_new = w + vl;

    vint32m4_t vacc0 = __riscv_vle32_v_i32m4(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const int32_t va0 = *a0++;
      vint32m4_t vb = __riscv_vle32_v_i32m4(w, vl);
      w = w + nr;
      vacc0 = __riscv_vmacc_vx_i32m4(vacc0, va0, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;
    w = w_new;
  } while (nc != 0);
}

void int32_gemm(
    size_t M, size_t N, size_t K,
    const int32_t* A, size_t a_row_stride,
    const int32_t* B,
    int32_t* C, size_t c_row_stride,
    size_t c_col_stride)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride * sizeof(int32_t);
    const size_t cm_stride_bytes = c_row_stride * sizeof(int32_t);
    const size_t cn_stride_bytes = c_col_stride;

    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 7) {
            gemm_i32_7xm4(
                7,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                B,
                C + row * c_row_stride,
                cm_stride_bytes,
                cn_stride_bytes
            );
            row += 7;
        } else {
            gemm_i32_1xm4(
                1,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                B,
                C + row * c_row_stride,
                cm_stride_bytes,
                cn_stride_bytes
            );
            row += 1;
        }
    }
}
