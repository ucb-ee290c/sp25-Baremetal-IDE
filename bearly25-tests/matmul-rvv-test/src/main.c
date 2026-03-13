/*
 * matmul-rvv-test/src/main.c
 *
 * RVV A^T * B kernel test — single-core or dual-core depending on compile flag.
 *
 * Build with -DDUAL_CORE to enable the two-hart split:
 *   Hart 0: output rows   0 .. 63  (A_T columns  0..63)
 *   Hart 1: output rows  64 .. 127 (A_T columns 64..127)
 *
 * Without -DDUAL_CORE the original 64x64 single-core test is built.
 *
 * Dual-core synchronization (two volatile flags):
 *   hart0_ready  — set by hart 0 after data init; hart 1 spins on it
 *   hart1_done   — set by hart 1 after its computation; hart 0 spins on it
 *
 * Cycle measurement: t0 just before hart 0 starts, t1 just after hart1_done.
 */

#ifdef DUAL_CORE
# define M       128
# define N       128
# define K       128
# define M_HALF   64
#else
# define M  64
# define N  64
# define K  64
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>


#include "chip_config.h"

#include "riscv_vector.h"

/* -------------------------------------------------------------------------
 * Dual-core synchronization flags (file-scope so both harts share them)
 * ------------------------------------------------------------------------- */
#ifdef DUAL_CORE
static volatile int hart0_ready = 0;  /* hart 0 sets this after data init  */
static volatile int hart1_done  = 0;  /* hart 1 sets this after computation */
#endif

/* -------------------------------------------------------------------------
 * Cycle counter
 * ------------------------------------------------------------------------- */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
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
  // ak is declared inside the nc loop so it resets automatically each iteration.
  int32_t* c0 = c;
  int32_t* c1 = (int32_t*) ((uintptr_t) c0 + cm_stride);
  int32_t* c2 = (int32_t*) ((uintptr_t) c1 + cm_stride);
  int32_t* c3 = (int32_t*) ((uintptr_t) c2 + cm_stride);
  int32_t* c4 = (int32_t*) ((uintptr_t) c3 + cm_stride);
  int32_t* c5 = (int32_t*) ((uintptr_t) c4 + cm_stride);
  int32_t* c6 = (int32_t*) ((uintptr_t) c5 + cm_stride);

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

    // Single pointer into A^T; offset addressing (ak[0..6]) replaces 7
    // separate advancing pointers, saving 6 scalar add instructions per k-step.
    const int8_t *ak = a;

    // Software-pipelined multiply-accumulate across kc.
    //
    // Register budget after 7×m4 accumulators (v0-v27) and vb in v28-v29:
    //   v30-v31 are free → a second vint16m2_t fits with zero spilling.
    //
    // Double-buffer: vb_e (v28) holds the "even" k-step's weights;
    //                vb_o (v30) holds the "odd"  k-step's weights.
    // Each load is issued before the 7 scalar A-loads that fill its latency,
    // so the vle8+vwcvt is fully hidden behind integer-pipeline work.
    //
    // Prologue: load weights for k=0 into v28 before the loop starts.
    register vint16m2_t vb_e asm("v28") =
        __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
    w += nr;
    register vint16m2_t vb_o asm("v30");

    size_t k = kc;
    while (k >= 2) {
      // Issue odd-half prefetch (k+1) into v30 BEFORE computing even half (k).
      // The 7 scalar loads below fill the vle8+vwcvt latency.
      vb_o = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w += nr;

      const int8_t va0 = ak[0];
      const int8_t va1 = ak[1];
      const int8_t va2 = ak[2];
      const int8_t va3 = ak[3];
      const int8_t va4 = ak[4];
      const int8_t va5 = ak[5];
      const int8_t va6 = ak[6];
      ak += a_stride;

      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);

      // Issue next-pair prefetch (k+2) into v28 BEFORE computing odd half (k+1).
      // Only when a future iteration exists (k >= 3); skipped on the final pair
      // so we never read past the end of w.
      if (k >= 3) {
        vb_e = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
        w += nr;
      }

      const int8_t wb0 = ak[0];
      const int8_t wb1 = ak[1];
      const int8_t wb2 = ak[2];
      const int8_t wb3 = ak[3];
      const int8_t wb4 = ak[4];
      const int8_t wb5 = ak[5];
      const int8_t wb6 = ak[6];
      ak += a_stride;

      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, wb0, vb_o, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, wb1, vb_o, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, wb2, vb_o, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, wb3, vb_o, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, wb4, vb_o, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, wb5, vb_o, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, wb6, vb_o, vl);

      k -= 2;
    }

    // Epilogue for odd kc: vb_e was preloaded in the last pair's middle step.
    if (k == 1) {
      const int8_t va0 = ak[0];
      const int8_t va1 = ak[1];
      const int8_t va2 = ak[2];
      const int8_t va3 = ak[3];
      const int8_t va4 = ak[4];
      const int8_t va5 = ak[5];
      const int8_t va6 = ak[6];
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
      vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
      vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
      vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
      vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
      vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
      vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
    }
    // No pointer resets needed: ak is local to this nc iteration.

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

    // Software-pipelined multiply-accumulate across kc.
    // Same double-buffer scheme as gemm_i8_i32_8xm1: vb holds the "even"
    // k-step's weights; vb_nxt holds the "odd" k-step's weights.
    // Only 1 accumulator (4 regs) + 2 vb buffers (2 regs each) = 8/32 regs,
    // so the compiler assigns freely with no spill risk.
    //
    // Prologue: load weights for k=0 before the loop.
    vint16m2_t vb     = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
    w += nr;
    vint16m2_t vb_nxt;

    size_t k = kc;
    while (k >= 2) {
      // Prefetch odd-half weights (k+1) BEFORE computing even half (k).
      vb_nxt = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
      w += nr;

      const int8_t va0 = *a0; a0 += a_stride;
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);

      // Prefetch next-pair weights (k+2) into vb BEFORE computing odd half (k+1).
      // Guard prevents reading past the end of w on the final pair.
      if (k >= 3) {
        vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
        w += nr;
      }

      const int8_t wb0 = *a0; a0 += a_stride;
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, wb0, vb_nxt, vl);

      k -= 2;
    }

    // Epilogue for odd kc: vb was preloaded in the last pair's middle step.
    if (k == 1) {
      const int8_t va0 = *a0; a0 += a_stride;
      vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
    }

    a0 -= kc*a_stride;

    __riscv_vse32_v_i32m4(c0, vacc0, vl);
    c0 += vl;
    w = w_new;
  } while (nc != 0);
}

// void gemm_i8_i32_8xm1(
//     size_t mr,        // number of rows to process (1..8)
//     size_t nc,        // number of columns to process
//     size_t kc,        // number of "channels" or "inner dimension"
//     const int8_t* a,  // input matrix A (transposed: [K x M] row-major)
//     size_t a_stride,  // leading dim of A^T (= M); stride between k steps
//     const int8_t* w,  // weights (B)
//     int32_t* c,       // output matrix C (int32)
//     size_t cm_stride, // byte stride between consecutive rows of C
//     size_t cn_stride  // byte stride between consecutive column-blocks of C
// )
// {

//   // A is transposed: stored as A^T [K x M] row-major.
//   // Row i of A = column i of A^T => rows are adjacent (offset +1).
//   // Advancing along k = advancing by a_stride (leading dim of A^T = M).
//   const int8_t* a0 = a;
//   int32_t* c0 = c;

//   const int8_t* a1 = a0 + 1;
//   int32_t* c1 = (int32_t*) ((uintptr_t) c0 + cm_stride);

//   const int8_t* a2 = a0 + 2;
//   int32_t* c2 = (int32_t*) ((uintptr_t) c1 + cm_stride);

//   const int8_t* a3 = a0 + 3;
//   int32_t* c3 = (int32_t*) ((uintptr_t) c2 + cm_stride);

//   const int8_t* a4 = a0 + 4;
//   int32_t* c4 = (int32_t*) ((uintptr_t) c3 + cm_stride);

//   const int8_t* a5 = a0 + 5;
//   int32_t* c5 = (int32_t*) ((uintptr_t) c4 + cm_stride);

//   const int8_t* a6 = a0 + 6;
//   int32_t* c6 = (int32_t*) ((uintptr_t) c5 + cm_stride);

//   // const int8_t* a7 = a0 + 7;
//   // int32_t* c7 = (int32_t*) ((uintptr_t) c6 + cm_stride);

//   size_t nr = nc;
//   size_t vl = nr;
//   const int8_t* w_new = w;
//   // Loop over columns in chunks of VL
//   do {
//     vl = __riscv_vsetvl_e32m4(nc);
//     nc -= vl;
//     w_new = w + vl;

//     // Accumulators pinned to v0,v4,...,v28 (8 x LMUL=4 fills all 32 vregs).
//     // Initialize from w (bias/first row of packed B), then broadcast to others.
//     register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
//     w = w + nr;
//     register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
//     register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
//     register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
//     register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
//     register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
//     register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);
//     // register vint32m4_t vacc7 asm("v28") = __riscv_vmv_v_v_i32m4(vacc0, vl);

//     // Multiply-accumulate across kc
//     size_t k = kc;
//     do {
//       // Load 1 int8 from each row; advance by a_stride to next k step
//       const int8_t va0 = *a0; a0 += a_stride;
//       const int8_t va1 = *a1; a1 += a_stride;
//       const int8_t va2 = *a2; a2 += a_stride;
//       const int8_t va3 = *a3; a3 += a_stride;
//       const int8_t va4 = *a4; a4 += a_stride;
//       const int8_t va5 = *a5; a5 += a_stride;
//       const int8_t va6 = *a6; a6 += a_stride;
//       // const int8_t va7 = *a7; a7 += a_stride;

//       // Load one vector of int8 from the weights
//       register vint16m2_t vb asm("v28") = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
//       w = w + nr;

//       // Widening multiply-accumulate into int32
//       vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
//       vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
//       vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
//       vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
//       vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
//       vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
//       vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);
//       // vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, va7, vb, vl);

//       k -= 1;
//     } while (k != 0);

//     a0 -= kc * a_stride;
//     a1 -= kc * a_stride;
//     a2 -= kc * a_stride;
//     a3 -= kc * a_stride;
//     a4 -= kc * a_stride;
//     a5 -= kc * a_stride;
//     a6 -= kc * a_stride;
//     // a7 -= kc * a_stride;

//     // Store results
//     __riscv_vse32_v_i32m4(c0, vacc0, vl); c0 += vl;
//     __riscv_vse32_v_i32m4(c1, vacc1, vl); c1 += vl;
//     __riscv_vse32_v_i32m4(c2, vacc2, vl); c2 += vl;
//     __riscv_vse32_v_i32m4(c3, vacc3, vl); c3 += vl;
//     __riscv_vse32_v_i32m4(c4, vacc4, vl); c4 += vl;
//     __riscv_vse32_v_i32m4(c5, vacc5, vl); c5 += vl;
//     __riscv_vse32_v_i32m4(c6, vacc6, vl); c6 += vl;
//     // __riscv_vse32_v_i32m4(c7, vacc7, vl); c7 += vl;
//     w = w_new;
//   } while (nc != 0);
// }

// void gemm_i8_i32_1xm4(
//     size_t mr,        // number of rows to process (1..7)
//     size_t nc,        // number of columns to process
//     size_t kc,        // number of "channels" or "inner dimension"
//     const int8_t* a,  // input matrix A
//     size_t a_stride,           // byte stride between consecutive rows of A
//     const int8_t* w,  // weights (B)
//     int32_t* c,       // output matrix C (int32)
//     size_t cm_stride,          // byte stride between consecutive rows of C
//     size_t cn_stride           // byte stride between consecutive columns-blocks of C
// )
// {

//   const int8_t* a0 = a;
//   int32_t* c0 = c;

//   size_t nr = nc;
//   size_t vl = nr;
//   const int8_t* w_new = w;
//   do {
//     vl = __riscv_vsetvl_e32m4(nc);
//     nc -= vl;
//     w_new = w + vl;

//     vint32m4_t vacc0 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl), vl);
//     w = w + nr;

//     size_t k = kc;
//     do {
//       const int8_t va0 = *a0; a0 += a_stride;
//       vint16m2_t vb = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(w, vl), vl);
//       w = w + nr;
//       vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);

//       k -= 1;
//     } while (k != 0);

//     a0 -= kc*a_stride;

//     __riscv_vse32_v_i32m4(c0, vacc0, vl);
//     c0 += vl;
//     w = w_new;
//   } while (nc != 0);
// }

void i8_i32_matmul(size_t m, size_t n, size_t k,
                   const int8_t *A, size_t a_row_stride,
                   const int8_t *B,
                   int32_t *C, size_t c_row_stride, size_t c_col_stride)
{
    const size_t kc_bytes = k;
    const size_t a_stride_bytes = a_row_stride;
    const size_t cm_stride_bytes = c_row_stride * sizeof(int32_t);
    const size_t cn_stride_bytes = c_col_stride;

    size_t row = 0;
    while (row < m) {
        size_t rows_left = m - row;

        if (rows_left >= 7) {
            gemm_i8_i32_8xm1(
                7,
                n,
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
                n,
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

/* -------------------------------------------------------------------------
 * Scalar reference: C[i][j] = Σ_k A_T[k*M+i] * B[k*N+j]
 * ------------------------------------------------------------------------- */

static void ref_matmul(const int8_t *A_T, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A_T[k * M + i] * (int32_t)B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

/* -------------------------------------------------------------------------
 * Print a 64x64 int32 matrix (abbreviated: first 8 rows × 8 cols)
 * ------------------------------------------------------------------------- */

static void print_matrix_i32(const char *label, const int32_t *mat,
                              int rows, int cols, int stride)
{
    const int SHOW = 8;
    printf("\n%s (%dx%d, showing top-left %dx%d):\n",
           label, rows, cols, SHOW, SHOW);
    for (int i = 0; i < SHOW; i++) {
        printf("  [");
        for (int j = 0; j < SHOW; j++) {
            printf("%7d", mat[i * stride + j]);
            if (j < SHOW - 1) printf(",");
        }
        printf(" ]\n");
    }
}

/* -------------------------------------------------------------------------
 * Compare C_ope and C_ref; print first mismatch and error count.
 * Returns 0 on PASS, -1 on FAIL.
 * ------------------------------------------------------------------------- */

static int compare(const int32_t *got, const int32_t *exp)
{
    int errors = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (got[i * N + j] != exp[i * N + j]) {
                if (errors == 0)
                    printf("  First mismatch [%d][%d]: got %d, expected %d\n",
                           i, j, got[i * N + j], exp[i * N + j]);
                errors++;
            }
        }
    }
    if (errors)
        printf("  %d mismatches out of %d elements\n", errors, M * N);
    return errors ? -1 : 0;
}

/* -------------------------------------------------------------------------
 * Hardware UART init
 * ------------------------------------------------------------------------- */

void app_init(void) {
}

/* -------------------------------------------------------------------------
 * Main test
 * ------------------------------------------------------------------------- */

void app_main(void) {
    /* Static buffers — file-scope lifetime, shared between harts */
    static int8_t  A_T   [K * M];
    static int8_t  B     [K * N];
    static int8_t  B_pack[(K+1) * N];
    static int32_t C_rvv [M * N];
    static int32_t C_ref [M * N];

#ifdef DUAL_CORE
    /* -----------------------------------------------------------------------
     * Dual-core path: read hart ID and branch
     * --------------------------------------------------------------------- */
    unsigned long hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));

    if (hartid == 0) {
        printf("=== dual-core i8_i32_matmul  A_T[%dx%d] * B[%dx%d] ===\n",
               K, M, K, N);
        printf("    Hart 0: rows 0..%d   Hart 1: rows %d..%d\n",
               M_HALF - 1, M_HALF, M - 1);

        /* Hart 0 fills all shared buffers */
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < M; i++)
                A_T[k * M + i] = (int8_t)((k * 3 + i * 7 + 1) % 17 - 8);
            for (int j = 0; j < N; j++)
                B[k * N + j]   = (int8_t)((k * 5 + j * 2 + 3) % 13 - 6);
        }
        memset(B_pack, 0, N);
        memcpy(B_pack + N, B, K * N);
        memset(C_rvv, 0, sizeof(C_rvv));

        /* Signal hart 1 that data is ready */
        hart0_ready = 1;

        /* Hart 0 computes rows 0 .. M_HALF-1 */
        uint64_t t0 = rdcycle64();
        i8_i32_matmul(M_HALF, N, K,
                      A_T,          /* A_T column 0  */
                      M,
                      B_pack,
                      C_rvv, N, 1);

        /* Wait for hart 1 to finish rows M_HALF .. M-1 */
        while (!hart1_done);
        uint64_t t1 = rdcycle64();

        printf("\n  Cycles (both harts, wall-clock): %lu\n",
               (unsigned long)(t1 - t0));

    } else {
        /* Hart 1: wait for data, compute rows M_HALF .. M-1 */
        while (!hart0_ready);

        i8_i32_matmul(M_HALF, N, K,
                      A_T + M_HALF, /* A_T column M_HALF */
                      M,
                      B_pack,
                      C_rvv + M_HALF * N, N, 1);

        hart1_done = 1;
    }

#else
    /* -----------------------------------------------------------------------
     * Single-core path (original 64x64 test)
     * --------------------------------------------------------------------- */
    printf("=== i8_i32_matmul RVV TEST  A_T[%dx%d] * B[%dx%d] ===\n",
           K, M, K, N);

    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++)
            A_T[k * M + i] = (int8_t)((k * 3 + i * 7 + 1) % 17 - 8);
        for (int j = 0; j < N; j++)
            B[k * N + j]   = (int8_t)((k * 5 + j * 2 + 3) % 13 - 6);
    }
    memset(B_pack, 0, N);
    memcpy(B_pack + N, B, K * N);

    // ref_matmul(A_T, B, C_ref);

    memset(C_rvv, 0, sizeof(C_rvv));
    uint64_t t0 = rdcycle64();
    i8_i32_matmul(M, N, K,
                  A_T, M,
                  B_pack,
                  C_rvv, N, 1);
    uint64_t t1 = rdcycle64();

    // print_matrix_i32("RVV output", C_rvv, M, N, N);
    // print_matrix_i32("Reference ", C_ref, M, N, N);
    // int rc = compare(C_rvv, C_ref);

    printf("\n  Cycles: %lu\n", (unsigned long)(t1 - t0));
    // printf("  Result: %s\n", rc == 0 ? "PASS" : "FAIL");
#endif
}

int main(void) {
    app_init();
    app_main();
    return 0;
}
