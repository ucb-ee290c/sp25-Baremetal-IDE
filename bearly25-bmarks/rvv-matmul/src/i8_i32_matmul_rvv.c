#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <riscv_vector.h>
// 7x(m4) int8->int32 microkernel with widening multiply.
// A is [mr × kc], B is fused into w (int8 weights) laid out per column-block, C is [mr × nc] in int32.
void gemm_i8_i32_7xm4_packed(
    size_t mr,        // number of rows to process (1..7)
    size_t nc,        // number of columns to process
    size_t kc,        // number of "channels" or "inner dimension"
    const int8_t* a,  // input matrix A
    size_t a_stride,           // byte stride between consecutive rows of A
    const int8_t* w,  // weights (B)
    int32_t* c,       // output matrix C (int32)
    size_t cm_stride,          // byte stride between consecutive rows of C
    size_t cn_stride           // byte stride between consecutive columns-blocks of C
)
{

  const int8_t* a0 = a;
  int32_t* c0 = c;

  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int32_t* c1 = (int32_t*) ((uintptr_t) c0 + cm_stride);

  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int32_t* c2 = (int32_t*) ((uintptr_t) c1 + cm_stride);

  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int32_t* c3 = (int32_t*) ((uintptr_t) c2 + cm_stride);

  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int32_t* c4 = (int32_t*) ((uintptr_t) c3 + cm_stride);

  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int32_t* c5 = (int32_t*) ((uintptr_t) c4 + cm_stride);

  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int32_t* c6 = (int32_t*) ((uintptr_t) c5 + cm_stride);

  // For NR="m4": we use a vsetvlmax for 32-bit int with LMUL=m4
  size_t nr = __riscv_vsetvlmax_e32m4();
  nr = (nc > nr) ? nr : nc;
  size_t vl = nr;

  // Loop over columns in chunks of VL
  do {
    // If fewer than nr columns remain, reduce VL
    if (nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
      nr = nc;
    }
    nc -= vl;

    // Create int32 accumulators for each row
    // Initialize them to zero. You could also load a "bias" from w if you wanted.
    vint32m4_t vacc0 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
    w = w + nr;

    vint32m4_t vacc1 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc2 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc3 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc4 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc5 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc6 = __riscv_vmv_v_v_i32m4(vacc0, vl);

    // Multiply-accumulate across kc
    size_t k = kc;
    do {
      // Load 1 int8 from each row
      const int8_t va0 = *a0++;
      const int8_t va1 = *a1++;
      const int8_t va2 = *a2++;
      const int8_t va3 = *a3++;
      const int8_t va4 = *a4++;
      const int8_t va5 = *a5++;
      const int8_t va6 = *a6++;

      // Load one vector of int8 from the weights
      vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w = w + nr;

      // Perform widening multiply to int16, then add to the int32 accumulators
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;
    a1 -= kc;
    a2 -= kc;
    a3 -= kc;
    a4 -= kc;
    a5 -= kc;
    a6 -= kc;

    // Now we store the results (int32) in the output C
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

void gemm_i8_i32_1xm4_packed(
    size_t mr,        // number of rows to process (1..7)
    size_t nc,        // number of columns to process
    size_t kc,        // number of "channels" or "inner dimension"
    const int8_t* a,  // input matrix A
    size_t a_stride,           // byte stride between consecutive rows of A
    const int8_t* w,  // weights (B)
    int32_t* c,       // output matrix C (int32)
    size_t cm_stride,          // byte stride between consecutive rows of C
    size_t cn_stride           // byte stride between consecutive columns-blocks of C
)
{

  const int8_t* a0 = a;
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

    vint32m4_t vacc0 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
    w = w + nr;

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w = w + nr;
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;

  } while (nc != 0);
}

void copy_int8_to_tcm(const int8_t *src, size_t n) {
  // Destination pointer in TCM
  int8_t *tcm_input = (int8_t *)0x78000000;

  size_t vl;
  while (n > 0) {
      // Set VL to process up to 'n' bytes with 8-bit elements, SEW=8, LMUL=1
      vl = __riscv_vsetvl_e8m8(n);

      // Load vl int8 elements from src
      vint8m8_t v = __riscv_vle8_v_i8m8(src, vl);
      // Store them to TCM
      __riscv_vse8_v_i8m8(tcm_input, v, vl);

      // Advance pointers and decrease remaining count
      src       += vl;
      tcm_input += vl;
      n         -= vl;
  }
}


void int8_int32_gemm_packed(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int32_t* C, size_t c_row_stride,
    size_t c_col_stride)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride;
    const size_t cm_stride_bytes = c_row_stride * sizeof(int32_t);
    const size_t cn_stride_bytes = c_col_stride;

    copy_int8_to_tcm(B, K*N+N);
    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 7) {
            gemm_i8_i32_7xm4_packed(
                7,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                (int8_t *)0x78000000,
                C + row * c_row_stride,
                cm_stride_bytes,
                cn_stride_bytes
            );
            row += 7;
        } else {
            gemm_i8_i32_1xm4_packed(
                1,
                N,
                kc_bytes,
                A + row * a_row_stride,
                a_stride_bytes,
                (int8_t *)0x78000000,
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
    const int8_t* B,
    int8_t* B_packed)
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

void gemm_i8_i32_7xm4(
    size_t mr,        // number of rows to process (1..7)
    size_t nc,        // number of columns to process
    size_t kc,        // number of "channels" or "inner dimension"
    const int8_t* a,  // input matrix A
    size_t a_stride,           // byte stride between consecutive rows of A
    const int8_t* w,  // weights (B)
    int32_t* c,       // output matrix C (int32)
    size_t cm_stride,          // byte stride between consecutive rows of C
    size_t cn_stride           // byte stride between consecutive columns-blocks of C
)
{

  const int8_t* a0 = a;
  int32_t* c0 = c;

  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int32_t* c1 = (int32_t*) ((uintptr_t) c0 + cm_stride);

  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int32_t* c2 = (int32_t*) ((uintptr_t) c1 + cm_stride);

  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int32_t* c3 = (int32_t*) ((uintptr_t) c2 + cm_stride);

  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  int32_t* c4 = (int32_t*) ((uintptr_t) c3 + cm_stride);

  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  int32_t* c5 = (int32_t*) ((uintptr_t) c4 + cm_stride);

  const int8_t* a6 = (const int8_t*) ((uintptr_t) a5 + a_stride);
  int32_t* c6 = (int32_t*) ((uintptr_t) c5 + cm_stride);

  // For NR="m4": we use a vsetvlmax for 32-bit int with LMUL=m4
  size_t nr = nc;
  size_t vl = nr;
  const int8_t* w_new = w;
  // Loop over columns in chunks of VL
  do {
    // If fewer than nr columns remain, reduce VL
    vl = __riscv_vsetvl_e32m4(nc);
    nc -= vl;
    w_new = w + vl;

    // Create int32 accumulators for each row
    vint32m4_t vacc0 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
    w = w + nr;

    vint32m4_t vacc1 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc2 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc3 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc4 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc5 = __riscv_vmv_v_v_i32m4(vacc0, vl);
    vint32m4_t vacc6 = __riscv_vmv_v_v_i32m4(vacc0, vl);

    // Multiply-accumulate across kc
    size_t k = kc;
    do {
      // Load 1 int8 from each row
      const int8_t va0 = *a0++;
      const int8_t va1 = *a1++;
      const int8_t va2 = *a2++;
      const int8_t va3 = *a3++;
      const int8_t va4 = *a4++;
      const int8_t va5 = *a5++;
      const int8_t va6 = *a6++;

      // Load one vector of int8 from the weights
      vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w = w + nr;

      // Perform widening multiply to int16, then add to the int32 accumulators
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;
    a1 -= kc;
    a2 -= kc;
    a3 -= kc;
    a4 -= kc;
    a5 -= kc;
    a6 -= kc;

    // Now we store the results (int32) in the output C
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

void gemm_i8_i32_1xm4(
    size_t mr,        // number of rows to process (1..7)
    size_t nc,        // number of columns to process
    size_t kc,        // number of "channels" or "inner dimension"
    const int8_t* a,  // input matrix A
    size_t a_stride,           // byte stride between consecutive rows of A
    const int8_t* w,  // weights (B)
    int32_t* c,       // output matrix C (int32)
    size_t cm_stride,          // byte stride between consecutive rows of C
    size_t cn_stride           // byte stride between consecutive columns-blocks of C
)
{

  const int8_t* a0 = a;
  int32_t* c0 = c;

  size_t nr = nc;
  size_t vl = nr;
  const int8_t* w_new = w;
  do {
    vl = __riscv_vsetvl_e32m4(nc);
    nc -= vl;
    w_new = w + vl;

    vint32m4_t vacc0 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
    w = w + nr;

    size_t k = kc;
    do {
      const int8_t va0 = *a0++;
      vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w = w + nr;
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;
    w = w_new;
  } while (nc != 0);
}

void int8_int32_gemm(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int32_t* C, size_t c_row_stride,
    size_t c_col_stride)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride;
    const size_t cm_stride_bytes = c_row_stride * sizeof(int32_t);
    const size_t cn_stride_bytes = c_col_stride;

    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 7) {
            gemm_i8_i32_7xm4(
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
            gemm_i8_i32_1xm4(
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
