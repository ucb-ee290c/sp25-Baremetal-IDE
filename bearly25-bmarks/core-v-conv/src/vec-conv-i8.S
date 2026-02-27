// See LICENSE for license details.

//**************************************************************************
// Vectorized 2D 3x3 convolution (int8 -> int16)
//--------------------------------------------------------------------------

    .text
    .balign 4

    .global vec_conv
/*
 * Calling convention:
 *     a0: size_t rows
 *     a1: size_t cols
 *     a2: size_t a_stride    (in bytes; for int8 inputs, no scaling)
 *     a3: size_t b_stride    (in int16 elements; multiply by 2 for bytes)
 *     a4: const int8_t *k
 *     a5: const int8_t *a
 *     a6: int16_t *b
 */

#define rows        a0
#define cols        a1
#define a_stride    a2
#define b_stride    a3
#define k           a4
#define a           a5
#define b           a6

#define ap          t0
#define bp          t1
#define vlen        t2
#define row_count   t3
#define VLEN_stride t4
#define ap_1        t5
#define ap_2        t6

#define row_check   s0
#define rows_odd    s1

// Use integer registers for kernel coefficients.
// We reserve t7,t8,t9 and s2-s7 for these 9 values.
#define k0          s8
#define k1          s9
#define k2          s10
#define k3          s2
#define k4          s3
#define k5          s4
#define k6          s5
#define k7          s6
#define k8          s7

// Vector registers for loading and accumulation.
// For int8 loads, we use vle8.v and for widening arithmetic, we use vwmul.vx and vwmacc.vx.
// The accumulators (vrow0, vrow1) are int16 vectors.
#define vload0      v0
#define vload1      v4
#define vload2      v8
#define vrow0       v16
#define vrow1       v20

// We need to save s0, s1, and s2-s7 (8 registers total). (8*8=64 bytes)
#define FRAMESIZE 88

vec_conv:
    addi    sp, sp, -FRAMESIZE
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    sd      s7, 56(sp)
    sd      s8, 64(sp)
    sd      s9, 72(sp)
    sd      s10, 80(sp)

    // Load the 3x3 kernel coefficients as int8 values into integer registers.
    lb      k0, 0(k)
    lb      k1, 1(k)
    lb      k2, 2(k)
    lb      k3, 3(k)
    lb      k4, 4(k)
    lb      k5, 5(k)
    lb      k6, 6(k)
    lb      k7, 7(k)
    lb      k8, 8(k)

    // For int8 inputs, a_stride is already in bytes.
    // For int16 outputs, multiply b_stride by 2 to convert element count to bytes.
    slli    b_stride, b_stride, 1

    mv      row_check, rows
    addi    row_check, row_check, -2

    andi    rows_odd, rows, 1

////////////////////////////////////////////////////////////////////////
// Prolog: Setup pointers and initial row loads.
////////////////////////////////////////////////////////////////////////
loop_prolog:
    mv      ap, a
    addi    ap_1, ap, 1      // offset by 4 bytes for int8 elements
    addi    ap_2, ap, 2
    mv      bp, b
    mv      row_count, row_check

    // Set vector length using e8 elements.
    vsetvli vlen, cols, e8, m2, ta, ma
    slli    VLEN_stride, vlen, 1   // 1 byte per int8 element

    // Load first row and compute horizontal convolution.
    vle8.v  vload0, (ap)
    vwmul.vx vrow0, vload0, k0
    vle8.v  vload1, (ap_1)
    vwmacc.vx vrow0, k1, vload1
    vle8.v  vload2, (ap_2)
    vwmacc.vx vrow0, k2, vload2

    add     ap, ap, a_stride
    addi    ap_1, ap, 1
    addi    ap_2, ap, 2

    // Load second row and accumulate.
    vle8.v  vload0, (ap)
    vwmacc.vx vrow0, k3, vload0
    vle8.v  vload1, (ap_1)
    vwmacc.vx vrow0, k4, vload1
    vle8.v  vload2, (ap_2)
    vwmacc.vx vrow0, k5, vload2
    add     ap, ap, a_stride

    // Load third row and start computing second output row.
    vwmul.vx vrow1, vload0, k0
    vle8.v  vload0, (ap)
    addi    ap_1, ap, 1
    vwmacc.vx vrow1, k1, vload1
    vle8.v  vload1, (ap_1)
    vwmacc.vx vrow1, k2, vload2
    addi    ap_2, ap, 2
    vle8.v  vload2, (ap_2)

////////////////////////////////////////////////////////////////////////
// Main Loop: Process blocks of 2 output rows.
////////////////////////////////////////////////////////////////////////
conv_loop:
    vwmacc.vx vrow0, k6, vload0
    add     ap, ap, a_stride
    vwmacc.vx vrow0, k7, vload1
    addi    ap_1, ap, 1
    vwmacc.vx vrow0, k8, vload2
    addi    ap_2, ap, 2

    vse16.v vrow0, (bp)

    vwmacc.vx vrow1, k3, vload0
    vwmacc.vx vrow1, k4, vload1
    vwmacc.vx vrow1, k5, vload2

    vwmul.vx vrow0, vload0, k0
    vle8.v  vload0, (ap)
    vwmacc.vx vrow0, k1, vload1
    vle8.v  vload1, (ap_1)
    add     bp, bp, b_stride
    vwmacc.vx vrow0, k2, vload2

    vle8.v  vload2, (ap_2)

    vwmacc.vx vrow1, k6, vload0
    add     ap, ap, a_stride
    vwmacc.vx vrow1, k7, vload1
    addi    ap_1, ap, 1
    vwmacc.vx vrow1, k8, vload2
    addi    ap_2, ap, 2

    vwmacc.vx vrow0, k3, vload0
    vwmacc.vx vrow0, k4, vload1
    vwmacc.vx vrow0, k5, vload2

    vse16.v vrow1, (bp) 

    vwmul.vx vrow1, vload0, k0
    vle8.v  vload0, (ap)
    vwmacc.vx vrow1, k1, vload1
    vle8.v  vload1, (ap_1)
    vwmacc.vx vrow1, k2, vload2
    vle8.v  vload2, (ap_2)

    add     bp, bp, b_stride
    addi    row_count, row_count, -2

    bgtz    row_count, conv_loop

epilog:
    vwmacc.vx vrow0, k6, vload0 
    vwmacc.vx vrow0, k7, vload1
    vwmacc.vx vrow0, k8, vload2
    vse16.v vrow0, (bp)

    bnez    rows_odd, row_loop_complete

    vwmacc.vx vrow1, k3, vload0
    vwmacc.vx vrow1, k4, vload1
    vwmacc.vx vrow1, k5, vload2

    add     ap, ap, a_stride
    addi    ap_1, ap, 1
    addi    ap_2, ap, 2
    add     bp, bp, b_stride

    vle8.v  vload0, (ap)
    vwmacc.vx vrow1, k6, vload0
    vle8.v  vload1, (ap_1)
    vwmacc.vx vrow1, k7, vload1
    vle8.v  vload2, (ap_2)
    vwmacc.vx vrow1, k8, vload2

    vse16.v vrow1, (bp)
    
row_loop_complete:
    add     a, a, vlen
    add     b, b, VLEN_stride

    sub     cols, cols, vlen
    bnez    cols, loop_prolog

exit:
    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    ld      s6, 48(sp)
    ld      s7, 56(sp)
    ld      s8, 64(sp)
    ld      s9, 72(sp)
    ld      s10, 80(sp)
    addi    sp, sp, FRAMESIZE

    ret