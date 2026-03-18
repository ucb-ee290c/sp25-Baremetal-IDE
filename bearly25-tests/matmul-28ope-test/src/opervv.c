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


void gemm_i8_i32_28xm1(
    size_t mr,           // unused
    size_t nc,           // number of output columns
    size_t kc,           // inner dimension
    const int8_t* a,     // A^T base for this M-tile
    size_t a_stride,     // = M
    const int8_t* w,     // B_pack: [(K+1)×N], row 0 = zero bias
    size_t b_row_stride, // = N (B_pack row stride in int8 elements)
    int32_t* c,          // C base for this M-tile
    size_t cm_stride,    // byte stride between C rows
    size_t cn_stride     // unused
)
{
    // Four base C-row pointers (one per i4 group); each advances by vl per nc-chunk.
    int32_t *cbase0 = c;
    int32_t *cbase1 = (int32_t*)((uintptr_t)c +  7 * cm_stride);
    int32_t *cbase2 = (int32_t*)((uintptr_t)c + 14 * cm_stride);
    int32_t *cbase3 = (int32_t*)((uintptr_t)c + 21 * cm_stride);
    // OPE output: rows 28-35 of this M-tile.
    int32_t *cbase_ope = (int32_t*)((uintptr_t)c + 28 * cm_stride);

    const size_t nr = b_row_stride;  // B_pack row stride (= N full columns)
    size_t nc_rem = nc;
    const int8_t *ww = w;

    do {
        size_t vl = __riscv_vsetvl_e32m4(nc_rem);
        nc_rem -= vl;
        const int8_t *w_next = ww + vl;   // start of next nc-chunk in bias row
        const int8_t *w_nc   = ww;        // B_pack[0][col] for this nc-chunk

        // Reset OPE for this column chunk.
        OP_ZERO();

        // ── i4 = 0: rows 0-6 — RVV + OPE interleaved in k-loop ──────────
        {
            const int8_t *a0 = a+0, *a1 = a+1, *a2 = a+2, *a3 = a+3;
            const int8_t *a4 = a+4, *a5 = a+5, *a6 = a+6;

            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);

            // OPE pointers: B[k=0][col], A^T[k=0][28..35]
            const int8_t *w_ope = ww;         // = B_pack[1][col] after bias advance
            const int8_t *a_ope = a + 28;

            size_t k = kc;
            do {
                // Fire OPE (RoCC fire-and-forget). CPU proceeds immediately.
                // The 7 scalar loads + vle8 + 7 vwmacc below hide the OPE latency.
                OP_ACC_L(w_ope, a_ope, 1);
                w_ope += b_row_stride;
                a_ope += a_stride;

                const int8_t va0 = *a0; a0 += a_stride;
                const int8_t va1 = *a1; a1 += a_stride;
                const int8_t va2 = *a2; a2 += a_stride;
                const int8_t va3 = *a3; a3 += a_stride;
                const int8_t va4 = *a4; a4 += a_stride;
                const int8_t va5 = *a5; a5 += a_stride;
                const int8_t va6 = *a6; a6 += a_stride;

                register vint16m2_t vb asm("v28") =
                    __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;

                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);
                k--;
            } while (k != 0);

            a0-=kc*a_stride; a1-=kc*a_stride; a2-=kc*a_stride; a3-=kc*a_stride;
            a4-=kc*a_stride; a5-=kc*a_stride; a6-=kc*a_stride;

            __riscv_vse32_v_i32m4(cbase0,                                      vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase0 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase0 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase0 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase0 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase0 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase0 + 6*cm_stride), vacc6, vl);
        }
        cbase0 += vl;

        // ── i4 = 1: rows 7-13 — pure RVV (OPE drains FIFO backlog) ──────
        {
            const int8_t *a0 = a+7, *a1 = a+8, *a2 = a+9,  *a3 = a+10;
            const int8_t *a4 = a+11, *a5 = a+12, *a6 = a+13;

            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);

            size_t k = kc;
            do {
                const int8_t va0 = *a0; a0 += a_stride;
                const int8_t va1 = *a1; a1 += a_stride;
                const int8_t va2 = *a2; a2 += a_stride;
                const int8_t va3 = *a3; a3 += a_stride;
                const int8_t va4 = *a4; a4 += a_stride;
                const int8_t va5 = *a5; a5 += a_stride;
                const int8_t va6 = *a6; a6 += a_stride;
                register vint16m2_t vb asm("v28") =
                    __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);
                k--;
            } while (k != 0);

            a0-=kc*a_stride; a1-=kc*a_stride; a2-=kc*a_stride; a3-=kc*a_stride;
            a4-=kc*a_stride; a5-=kc*a_stride; a6-=kc*a_stride;

            __riscv_vse32_v_i32m4(cbase1,                                      vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase1 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase1 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase1 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase1 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase1 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase1 + 6*cm_stride), vacc6, vl);
        }
        cbase1 += vl;

        // ── i4 = 2: rows 14-20 ───────────────────────────────────────────
        {
            const int8_t *a0 = a+14, *a1 = a+15, *a2 = a+16, *a3 = a+17;
            const int8_t *a4 = a+18, *a5 = a+19, *a6 = a+20;

            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);

            size_t k = kc;
            do {
                const int8_t va0 = *a0; a0 += a_stride;
                const int8_t va1 = *a1; a1 += a_stride;
                const int8_t va2 = *a2; a2 += a_stride;
                const int8_t va3 = *a3; a3 += a_stride;
                const int8_t va4 = *a4; a4 += a_stride;
                const int8_t va5 = *a5; a5 += a_stride;
                const int8_t va6 = *a6; a6 += a_stride;
                register vint16m2_t vb asm("v28") =
                    __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);
                k--;
            } while (k != 0);

            a0-=kc*a_stride; a1-=kc*a_stride; a2-=kc*a_stride; a3-=kc*a_stride;
            a4-=kc*a_stride; a5-=kc*a_stride; a6-=kc*a_stride;

            __riscv_vse32_v_i32m4(cbase2,                                      vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase2 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase2 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase2 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase2 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase2 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase2 + 6*cm_stride), vacc6, vl);
        }
        cbase2 += vl;

        // ── i4 = 3: rows 21-27 ───────────────────────────────────────────
        {
            const int8_t *a0 = a+21, *a1 = a+22, *a2 = a+23, *a3 = a+24;
            const int8_t *a4 = a+25, *a5 = a+26, *a6 = a+27;

            ww = w_nc;
            register vint32m4_t vacc0 asm("v0")  = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl), vl);
            ww += nr;
            register vint32m4_t vacc1 asm("v4")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc2 asm("v8")  = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc3 asm("v12") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc4 asm("v16") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc5 asm("v20") = __riscv_vmv_v_v_i32m4(vacc0, vl);
            register vint32m4_t vacc6 asm("v24") = __riscv_vmv_v_v_i32m4(vacc0, vl);

            size_t k = kc;
            do {
                const int8_t va0 = *a0; a0 += a_stride;
                const int8_t va1 = *a1; a1 += a_stride;
                const int8_t va2 = *a2; a2 += a_stride;
                const int8_t va3 = *a3; a3 += a_stride;
                const int8_t va4 = *a4; a4 += a_stride;
                const int8_t va5 = *a5; a5 += a_stride;
                const int8_t va6 = *a6; a6 += a_stride;
                register vint16m2_t vb asm("v28") =
                    __riscv_vwcvt_x_x_v_i16m2(__riscv_vle8_v_i8m1(ww, vl), vl);
                ww += nr;
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, va0, vb, vl);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, va1, vb, vl);
                vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, va2, vb, vl);
                vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, va3, vb, vl);
                vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, va4, vb, vl);
                vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, va5, vb, vl);
                vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, va6, vb, vl);
                k--;
            } while (k != 0);

            a0-=kc*a_stride; a1-=kc*a_stride; a2-=kc*a_stride; a3-=kc*a_stride;
            a4-=kc*a_stride; a5-=kc*a_stride; a6-=kc*a_stride;

            __riscv_vse32_v_i32m4(cbase3,                                      vacc0, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase3 +   cm_stride), vacc1, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase3 + 2*cm_stride), vacc2, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase3 + 3*cm_stride), vacc3, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase3 + 4*cm_stride), vacc4, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase3 + 5*cm_stride), vacc5, vl);
            __riscv_vse32_v_i32m4((int32_t*)((uintptr_t)cbase3 + 6*cm_stride), vacc6, vl);
        }
        cbase3 += vl;

        // Extract OPE result for rows 28-35 of this column chunk.
        // fence w,r inside _op_ext_stride confirms all kc OP_ACC_Ls are done.
        OP_EXT_STRIDE(cbase_ope, (int)(cm_stride / sizeof(int32_t)), OPE_EXT_FLIP);
        cbase_ope += vl;

        ww = w_next;
    } while (nc_rem != 0);
}