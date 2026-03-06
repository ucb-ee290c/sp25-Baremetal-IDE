#include <stddef.h>
#include <stdint.h>

#include "bench_kernel.h"
#include "riscv_vector.h"

/*
 * i8_i32_matmul - 64x64 int8->int32 matmul kernel stub.
 *
 * Inputs:
 *   M, N, K        - matrix dimensions (fixed 64x64x64 for this benchmark)
 *   A              - pointer to [M x K] row-major int8 input matrix
 *   a_row_stride   - number of elements between consecutive rows of A
 *   B              - pointer to [K x N] row-major int8 weight matrix
 *   C              - pointer to [M x N] row-major int32 output matrix
 *   c_row_stride   - number of elements between consecutive rows of C
 *   c_col_stride   - stride between consecutive column elements of C (typically 1)
 *
 * Output:
 *   C[i][j] = sum_k A[i][k] * B[k][j]   (int32 accumulation, no bias)
 */



void gemm_i8_i32_8xm1(
    size_t mr,        // number of rows to process (1..8)
    size_t nc,        // number of columns to process
    size_t kc,        // number of "channels" or "inner dimension"
    const int8_t* a,  // input matrix A (transposed: [K x M] row-major)
    size_t a_stride,  // leading dim of A^T (= M); stride between k steps
    const int8_t* w,  // weights (B)
    int32_t* c,       // output matrix C (int32)
    size_t cm_stride, // byte stride between consecutive rows of C
    size_t cn_stride  // byte stride between consecutive column-blocks of C
)
{

  // A is transposed: stored as A^T [K x M] row-major.
  // Row i of A = column i of A^T => rows are adjacent (offset +1).
  // Advancing along k = advancing by a_stride (leading dim of A^T = M).
  const int8_t* a0 = a;
  int32_t* c0 = c;

  const int8_t* a1 = a0 + 1;
  int32_t* c1 = (int32_t*) ((uintptr_t) c0 + cm_stride);

  const int8_t* a2 = a0 + 2;
  int32_t* c2 = (int32_t*) ((uintptr_t) c1 + cm_stride);

  const int8_t* a3 = a0 + 3;
  int32_t* c3 = (int32_t*) ((uintptr_t) c2 + cm_stride);

  const int8_t* a4 = a0 + 4;
  int32_t* c4 = (int32_t*) ((uintptr_t) c3 + cm_stride);

  const int8_t* a5 = a0 + 5;
  int32_t* c5 = (int32_t*) ((uintptr_t) c4 + cm_stride);

  const int8_t* a6 = a0 + 6;
  int32_t* c6 = (int32_t*) ((uintptr_t) c5 + cm_stride);

  // const int8_t* a7 = a0 + 7;
  // int32_t* c7 = (int32_t*) ((uintptr_t) c6 + cm_stride);

  size_t nr = nc;
  size_t vl = nr;
  const int8_t* w_new = w;
  // Loop over columns in chunks of VL
  do {
    vl = __riscv_vsetvl_e32m4(nc);
    nc -= vl;
    w_new = w + vl;

    // Accumulators pinned to v0,v4,...,v28 (8 x LMUL=4 fills all 32 vregs).
    // Initialize from w (bias/first row of packed B), then broadcast to others.
    register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
    w = w + nr;
    register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
    register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
    register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
    register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
    register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
    register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);
    // register vint32m4_t vacc7 asm("v28") = __riscv_vmv_v_v_i32m4(vacc0, vl);

    // Multiply-accumulate across kc
    size_t k = kc;
    do {
      // Load 1 int8 from each row; advance by a_stride to next k step
      const int8_t va0 = *a0; a0 += a_stride;
      const int8_t va1 = *a1; a1 += a_stride;
      const int8_t va2 = *a2; a2 += a_stride;
      const int8_t va3 = *a3; a3 += a_stride;
      const int8_t va4 = *a4; a4 += a_stride;
      const int8_t va5 = *a5; a5 += a_stride;
      const int8_t va6 = *a6; a6 += a_stride;
      // const int8_t va7 = *a7; a7 += a_stride;

      // Load one vector of int8 from the weights
      register vint16m2_t vb asm("v28") = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w = w + nr;

      // Widening multiply-accumulate into int32
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);
      // vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, va7, vb, vl);

      k -= 1;
    } while (k != 0);

    a0 -= kc * a_stride;
    a1 -= kc * a_stride;
    a2 -= kc * a_stride;
    a3 -= kc * a_stride;
    a4 -= kc * a_stride;
    a5 -= kc * a_stride;
    a6 -= kc * a_stride;
    // a7 -= kc * a_stride;

    // Store results
    __riscv_vse32_v_i32m4(c0, vacc0, vl); c0 += vl;
    __riscv_vse32_v_i32m4(c1, vacc1, vl); c1 += vl;
    __riscv_vse32_v_i32m4(c2, vacc2, vl); c2 += vl;
    __riscv_vse32_v_i32m4(c3, vacc3, vl); c3 += vl;
    __riscv_vse32_v_i32m4(c4, vacc4, vl); c4 += vl;
    __riscv_vse32_v_i32m4(c5, vacc5, vl); c5 += vl;
    __riscv_vse32_v_i32m4(c6, vacc6, vl); c6 += vl;
    // __riscv_vse32_v_i32m4(c7, vacc7, vl); c7 += vl;
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
      const int8_t va0 = *a0; a0 += a_stride;
      vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w = w + nr;
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl); 

      k -= 1;
    } while (k != 0);

    a0 -= kc*a_stride;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;
    w = w_new;
  } while (nc != 0);
}

void i8_i32_matmul(size_t M, size_t N, size_t K,
                   const int8_t *A, size_t a_row_stride,
                   const int8_t *B,
                   int32_t *C, size_t c_row_stride, size_t c_col_stride)
{
    const size_t kc_bytes = K;
    const size_t a_stride_bytes = a_row_stride;
    const size_t cm_stride_bytes = c_row_stride * sizeof(int32_t);
    const size_t cn_stride_bytes = c_col_stride;

    size_t row = 0;
    while (row < M) {
        size_t rows_left = M - row;

        if (rows_left >= 7) {
            gemm_i8_i32_8xm1(
                7,
                N,
                kc_bytes,
                A + row,
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
                A + row,
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
