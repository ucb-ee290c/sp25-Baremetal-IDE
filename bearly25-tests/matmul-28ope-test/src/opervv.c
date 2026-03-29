/*
 * opervv.c — Separate RVV (28 rows) + OPE (8 rows) matmul kernels
 *
 * Two completely independent functions:
 *   gemm_rvv_28rows():  Saturn RVV, 4 groups of 7 rows
 *   gemm_ope_8rows():   OPE, 8×8 tile accumulation (pre-packed inputs)
 *
 * Called separately from main.c to isolate RVV and OPE.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "riscv_vector.h"
#include "rocc.h"

/* ──────────────────────── OPE low-level interface ──────────────────────── */

#ifndef OPE_CUSTOM
#define OPE_CUSTOM 0
#endif

#ifndef OPE_EXT_FLIP
#define OPE_EXT_FLIP 1
#endif

#define FCTN7_ACC     0b00
#define FCTN7_EXTRACT 0b01
#define FCTN7_ZERO    0b10

#define OP_ZERO() ROCC_INSTRUCTION(OPE_CUSTOM, FCTN7_ZERO)

static inline void OP_ACC_L(int8_t *U, int8_t *V, int L) {
  register uint64_t rs1 asm("x11") = (uint64_t)U;
  register uint64_t rs2 asm("x12") = (uint64_t)V;
  switch (L) {
    case  1: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 0<<2)); break;
    case  2: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 1<<2)); break;
    case  3: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 2<<2)); break;
    case  4: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 3<<2)); break;
    case  5: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 4<<2)); break;
    case  6: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 5<<2)); break;
    case  7: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 6<<2)); break;
    case  8: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 7<<2)); break;
    case  9: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 8<<2)); break;
    case 10: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 9<<2)); break;
    case 11: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(10<<2)); break;
    case 12: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(11<<2)); break;
    case 13: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(12<<2)); break;
    case 14: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(13<<2)); break;
    case 15: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(14<<2)); break;
    case 16: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(15<<2)); break;
    case 17: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(16<<2)); break;
    case 18: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(17<<2)); break;
    case 19: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(18<<2)); break;
    case 20: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(19<<2)); break;
    case 21: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(20<<2)); break;
    case 22: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(21<<2)); break;
    case 23: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(22<<2)); break;
    case 24: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(23<<2)); break;
    case 25: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(24<<2)); break;
    case 26: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(25<<2)); break;
    case 27: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(26<<2)); break;
    case 28: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(27<<2)); break;
    case 29: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(28<<2)); break;
    case 30: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(29<<2)); break;
    case 31: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(30<<2)); break;
    case 32: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(31<<2)); break;
  }
}

#define _OP_EXT_S_T(rs1, rs2)  \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_EXTRACT|(1<<2)|(1<<3))
#define _OP_EXT_NS_T(rs2)      \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, 0,   rs2, FCTN7_EXTRACT|(1<<2)|(0<<3))
#define _OP_EXT_S_NT(rs1, rs2) \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_EXTRACT|(0<<2)|(1<<3))
#define _OP_EXT_NS_NT(rs2)     \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, 0,   rs2, FCTN7_EXTRACT|(0<<2)|(0<<3))

static inline void OP_EXT_STRIDE(int32_t *arr, int stride_elements, int transposed) {
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

/* ──────────────────────── end OPE interface ──────────────────────── */

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

/* ====================================================================
 * gemm_rvv_28rows — Pure RVV kernel for 28 output rows (4 groups of 7)
 * ==================================================================== */
void gemm_rvv_28rows(
    size_t nc,
    size_t kc,
    const int8_t *a,
    size_t a_stride,
    const int8_t *w,
    size_t b_row_stride,
    int32_t *c,
    size_t cm_stride
)
{
    int32_t *cbase0 = c;
    int32_t *cbase1 = (int32_t *)((uintptr_t)c +  7 * cm_stride);
    int32_t *cbase2 = (int32_t *)((uintptr_t)c + 14 * cm_stride);
    int32_t *cbase3 = (int32_t *)((uintptr_t)c + 21 * cm_stride);

    const size_t nr = b_row_stride;
    size_t nc_rem = nc;
    const int8_t *ww = w;

    do {
        size_t vl = __riscv_vsetvl_e32m4(nc_rem);
        nc_rem -= vl;
        const int8_t *w_next = ww + vl;
        const int8_t *w_nc   = ww;

        /* RVV group 0: rows 0–6 */
        {
            const int8_t *ak = a;
            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint16m2_t vb_e asm("v28") =
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
            ww += nr;
            register vint16m2_t vb_o asm("v30");
            size_t k = kc;
            while (k >= 2) {
                vb_o = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                ak += a_stride;
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
                if (k >= 3) {
                    vb_e = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                    ww += nr;
                }
                const int8_t wb0 = ak[0], wb1 = ak[1], wb2 = ak[2], wb3 = ak[3];
                const int8_t wb4 = ak[4], wb5 = ak[5], wb6 = ak[6];
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
            if (k == 1) {
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
            }
            __riscv_vse32_v_i32m4(cbase0,                                       vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase0 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase0 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase0 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase0 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase0 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase0 + 6*cm_stride), vacc6, vl);
        }
        cbase0 += vl;

        /* RVV group 1: rows 7–13 */
        {
            const int8_t *ak = a + 7;
            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint16m2_t vb_e asm("v28") =
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
            ww += nr;
            register vint16m2_t vb_o asm("v30");
            size_t k = kc;
            while (k >= 2) {
                vb_o = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                ak += a_stride;
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
                if (k >= 3) {
                    vb_e = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                    ww += nr;
                }
                const int8_t wb0 = ak[0], wb1 = ak[1], wb2 = ak[2], wb3 = ak[3];
                const int8_t wb4 = ak[4], wb5 = ak[5], wb6 = ak[6];
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
            if (k == 1) {
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
            }
            __riscv_vse32_v_i32m4(cbase1,                                       vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase1 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase1 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase1 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase1 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase1 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase1 + 6*cm_stride), vacc6, vl);
        }
        cbase1 += vl;

        /* RVV group 2: rows 14–20 */
        {
            const int8_t *ak = a + 14;
            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint16m2_t vb_e asm("v28") =
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
            ww += nr;
            register vint16m2_t vb_o asm("v30");
            size_t k = kc;
            while (k >= 2) {
                vb_o = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                ak += a_stride;
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
                if (k >= 3) {
                    vb_e = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                    ww += nr;
                }
                const int8_t wb0 = ak[0], wb1 = ak[1], wb2 = ak[2], wb3 = ak[3];
                const int8_t wb4 = ak[4], wb5 = ak[5], wb6 = ak[6];
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
            if (k == 1) {
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
            }
            __riscv_vse32_v_i32m4(cbase2,                                       vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase2 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase2 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase2 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase2 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase2 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase2 + 6*cm_stride), vacc6, vl);
        }
        cbase2 += vl;

        /* RVV group 3: rows 21–27 */
        {
            const int8_t *ak = a + 21;
            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint16m2_t vb_e asm("v28") =
                __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
            ww += nr;
            register vint16m2_t vb_o asm("v30");
            size_t k = kc;
            while (k >= 2) {
                vb_o = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                ak += a_stride;
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
                if (k >= 3) {
                    vb_e = __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                    ww += nr;
                }
                const int8_t wb0 = ak[0], wb1 = ak[1], wb2 = ak[2], wb3 = ak[3];
                const int8_t wb4 = ak[4], wb5 = ak[5], wb6 = ak[6];
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
            if (k == 1) {
                const int8_t va0 = ak[0], va1 = ak[1], va2 = ak[2], va3 = ak[3];
                const int8_t va4 = ak[4], va5 = ak[5], va6 = ak[6];
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb_e, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb_e, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb_e, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb_e, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb_e, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb_e, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb_e, vl);
            }
            __riscv_vse32_v_i32m4(cbase3,                                       vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase3 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase3 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase3 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase3 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase3 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t *)((uintptr_t)cbase3 + 6*cm_stride), vacc6, vl);
        }
        cbase3 += vl;

        ww = w_next;
    } while (nc_rem != 0);
}

/* ====================================================================
 * gemm_ope_8rows — Pure OPE kernel for 8 output rows
 *
 * Processes N columns in 8-column tiles.
 * a_ope: pre-packed A (K×8, column-major groups)
 * b_ope: pre-packed B (N/8 tiles, each K×8)
 * c:     output pointer to row 28 of C
 * n:     number of columns
 * k:     inner dimension
 * c_stride: bytes between C rows
 * ==================================================================== */
void gemm_ope_8rows(
    const int8_t *a_ope,
    const int8_t *b_ope,
    int32_t *c,
    size_t n,
    size_t k,
    size_t c_stride
)
{
    const int ope_stride = (int)(c_stride / sizeof(int32_t));
    size_t n_tiles = n / 8;
    int32_t *c_ptr = c;

    printf("[ope] start: %lu tiles, k=%lu, stride=%d\n",
           (unsigned long)n_tiles, (unsigned long)k, ope_stride);

    for (size_t t = 0; t < n_tiles; t++) {
        OP_ZERO();

        int8_t *ap = (int8_t *)a_ope;
        int8_t *bp = (int8_t *)(b_ope + t * k * 8);
        size_t k_rem = k;
        while (k_rem > 0) {
            int L = (int)MIN(32, k_rem);
            OP_ACC_L(ap, bp, L);
            ap += L * 8;
            bp += L * 8;
            k_rem -= (size_t)L;
        }

        OP_EXT_STRIDE(c_ptr, ope_stride, OPE_EXT_FLIP);
        printf("[ope] tile %lu OK\n", (unsigned long)t);
        c_ptr += 8;
    }
    asm volatile("fence w, rw" ::: "memory");
    printf("[ope] done\n");
}
