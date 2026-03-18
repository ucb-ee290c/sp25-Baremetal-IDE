#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "riscv_vector.h"
#include "rocc.h"

// ---- OPE low-level interface (mirrors hal_ope.c definitions) ----

#ifndef OPE_CUSTOM
#define OPE_CUSTOM 0
#endif

#ifndef OPE_EXT_FLIP
#define OPE_EXT_FLIP 1
#endif

#define _FCTN7_ACC     0b00
#define _FCTN7_EXTRACT 0b01
#define _FCTN7_ZERO    0b10

#define OP_ZERO() ROCC_INSTRUCTION(OPE_CUSTOM, _FCTN7_ZERO)

static inline void _op_acc_l(int8_t *U, int8_t *V, int L) {
  register uint64_t rs1 asm("x11") = (uint64_t)U;
  register uint64_t rs2 asm("x12") = (uint64_t)V;
  switch (L) {
    case  1: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 0<<2)); break;
    case  2: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 1<<2)); break;
    case  3: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 2<<2)); break;
    case  4: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 3<<2)); break;
    case  5: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 4<<2)); break;
    case  6: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 5<<2)); break;
    case  7: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 6<<2)); break;
    case  8: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 7<<2)); break;
    case  9: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 8<<2)); break;
    case 10: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|( 9<<2)); break;
    case 11: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(10<<2)); break;
    case 12: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(11<<2)); break;
    case 13: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(12<<2)); break;
    case 14: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(13<<2)); break;
    case 15: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(14<<2)); break;
    case 16: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(15<<2)); break;
    case 17: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(16<<2)); break;
    case 18: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(17<<2)); break;
    case 19: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(18<<2)); break;
    case 20: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(19<<2)); break;
    case 21: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(20<<2)); break;
    case 22: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(21<<2)); break;
    case 23: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(22<<2)); break;
    case 24: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(23<<2)); break;
    case 25: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(24<<2)); break;
    case 26: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(25<<2)); break;
    case 27: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(26<<2)); break;
    case 28: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(27<<2)); break;
    case 29: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(28<<2)); break;
    case 30: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(29<<2)); break;
    case 31: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(30<<2)); break;
    case 32: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_ACC|(31<<2)); break;
  }
}

#define OP_ACC_L(U, V, L) _op_acc_l((int8_t *)(U), (int8_t *)(V), (L))

#define _OP_EXT_S_T(rs1, rs2)  \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_EXTRACT|(1<<2)|(1<<3))
#define _OP_EXT_NS_T(rs2)      \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, 0,   rs2, _FCTN7_EXTRACT|(1<<2)|(0<<3))
#define _OP_EXT_S_NT(rs1, rs2) \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, _FCTN7_EXTRACT|(0<<2)|(1<<3))
#define _OP_EXT_NS_NT(rs2)     \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, 0,   rs2, _FCTN7_EXTRACT|(0<<2)|(0<<3))

static inline void _op_ext_stride(int32_t *arr, int stride_elements, int transposed) {
  register uint64_t rs2 asm("x12") = (uint64_t)arr;
  if (stride_elements == 0 || stride_elements == 8) {
    if (transposed) { _OP_EXT_NS_T(rs2);  }
    else            { _OP_EXT_NS_NT(rs2); }
  } else {
    register uint64_t rs1 asm("x11") = (uint64_t)stride_elements;
    if (transposed) { _OP_EXT_S_T(rs1, rs2);  }
    else            { _OP_EXT_S_NT(rs1, rs2); }
  }
  asm volatile("fence w, r" ::: "memory");
}

#define OP_EXT_STRIDE(arr, stride, transposed) \
  _op_ext_stride((arr), (stride), (transposed))

// ---- end OPE interface ----


/*
 * accumulate_outer_products_8x8 - for every row i in [0,N) read 8 int8
 * elements from input1[i][n..n+7] (stride-1) and input2[i][l..l+7]
 * (stride-1), compute their outer product with the OPE, and write the
 * accumulated 8×8 int32 result to output[n..n+7][l..l+7].
 *
 * Mathematically:
 *   output[n+r][l+c] = Σ_i  input1[i][n+r] * input2[i][l+c]
 *                      for r,c ∈ [0,8)
 *
 * Since both inputs are row-major, the 8 elements at each row starting
 * at column n (or l) are already contiguous — no packing needed.
 * OP_EXT_STRIDE writes the result directly to output with row stride N.
 *
 * NOTE: firing N consecutive OP_ACC_L(L=1) calls without interleaved
 * work can flood the OPE FIFO for large N (see residual-loop comment in
 * gemm_i8_i32_15row_interleaved). Keep N small or interleave other work.
 *
 * Preconditions: n+8 ≤ N,  l+8 ≤ N.
 */
void accumulate_outer_products_8x8(
    size_t        N,
    const int8_t *input1,  /* NxN row-major int8  */
    const int8_t *input2,  /* NxN row-major int8  */
    int32_t      *output,  /* NxN row-major int32 (overwritten, not added) */
    size_t        n,       /* col offset in input1; also output row base */
    size_t        l        /* col offset in input2; also output col base */
)
{
    OP_ZERO();

    for (size_t i = 0; i < N; i++)
        OP_ACC_L(input2 + i * N + l, input1 + i * N + n, 1);

    /* Write 8×8 result directly to output[n..n+7][l..l+7] with row stride N. */
    OP_EXT_STRIDE(output + n * N + l, (int)N, OPE_EXT_FLIP);
}

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