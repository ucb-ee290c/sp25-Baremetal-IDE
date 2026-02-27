//
// Vectorized 2D 5x5 convolution (illustrative example)
// Follows same structure, logic, and pipeline approach as the 3x3 code.
//
    .text
    .balign 4

    .global vec_conv_5x5
/*
 * Calling convention (same as original):
 *     a0: size_t rows
 *     a1: size_t cols
 *     a2: size_t a_stride   (in int8_ts)
 *     a3: size_t b_stride   (in int8_ts)
 *     a4: const int8_t *k    (25 int8_ts for the 5x5 kernel)
 *     a5: const int8_t *a    (input image)
 *     a6: int16_t *b          (output image)
 */

#define rows         a0
#define cols         a1
#define a_stride     a2
#define b_stride     a3
#define k            a4
#define a            a5
#define b            a6

// We need more pointers for loads at +0,+4,+8,+12,+16
#define ap           t0
#define bp           t1
#define ap_1         t5
#define ap_2         t6
// For +12 and +16, we will reuse s2 and s3 (since the code already saves/restores s0, s1):
#define ap_3        s2
#define ap_4        s3

#define vlen         t2
#define row_count    t3
#define VLEN_stride  t4

#define row_check    s0

// 25 scalar registers for the 5×5 kernel:
#define k0  s4
#define k1  s5
#define k2  s6
#define k3  s7
#define k4  s8
#define k5  s9
#define k6  s10
#define k7  s11
#define k8  a7
#define k9  s1

// Vector registers for loading and partial sums
// Just as in the 3×3 code, we use v0..v4 for loading, and v16..v20 for partial sums
#define vload0       v0
#define vload1       v4
#define vload2       v8
#define vload3       v12
#define vload4       v2
#define vrow0        v16
#define vrow1        v20
#define vrow2        v24
#define vrow3        v28

// Increase frame size to save/restore s2, s3 as well
#define FRAMESIZE 96   // 32 was for s0,s1; add 16 more bytes for s2,s3

//----------------------------------------------------------------------------
// vec_conv_5x5
//----------------------------------------------------------------------------
vec_conv_5x5:
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
    sd      s11, 88(sp)

    slli b_stride, b_stride, 1


    // For a 5×5 conv, valid output rows = (rows - 4).
    // We still process 2 output rows per iteration => row_check = rows - 4
    mv     row_check, rows
    addi   row_check, row_check, -4


////////////////////////////////////////////////////////////////////////
//  Prolog: set up pointers, load first 4 input rows into vrow0, then
//          partially load the 5th row to begin vrow1, etc.  We keep
//          the same "two-row" pipeline style from the 3×3 version.
////////////////////////////////////////////////////////////////////////
loop_prolog:
    
    ////////////////////////////////////
    // Load 10 out of the 25 kernel coefficients
    ////////////////////////////////////
    lb  k0,   0(k)
    lb  k1,   1(k)
    lb  k2,   2(k)
    lb  k3,  3(k)
    lb  k4,  4(k)
    lb  k5,  5(k)
    lb  k6,  6(k)
    lb  k7,  7(k)
    lb  k8,  8(k)
    lb  k9,  9(k)

    // ap -> row pointer
    mv    ap, a
    addi  ap_1,  ap,  1
    addi  ap_2,  ap,  2
    addi  ap_3,  ap, 3
    addi  ap_4,  ap, 4

    // bp -> output pointer
    mv    bp, b

    // row_count tracks how many pairs of output rows remain
    mv    row_count, row_check

    // Set vector length
    vsetvli  vlen, cols, e8, m2, ta, ma
    slli     VLEN_stride, vlen, 1

    ////////////////////////////////////////////
    // 1) Load row0 => partial accumulate in vrow0 with k0..k4
    ////////////////////////////////////////////
    vle8.v   vload0, (ap)       // row0[+0]
    vwmul.vx  vrow0, vload0, k0

    vle8.v   vload1, (ap_1)     // row0[+4]
    vwmacc.vx vrow0, k1, vload1

    vle8.v   vload2, (ap_2)     // row0[+8]
    vwmacc.vx vrow0, k2, vload2

    vle8.v   vload3, (ap_3)    // row0[+12]
    vwmacc.vx vrow0, k3, vload3

    vle8.v   vload4, (ap_4)    // row0[+16]
    vwmacc.vx vrow0, k4, vload4

    // advance pointers to row1
    add   ap, ap, a_stride
    addi  ap_1,  ap,  1
    addi  ap_2,  ap,  2
    addi  ap_3,  ap,  3
    addi  ap_4,  ap,  4

    ////////////////////////////////////////////
    // 2) Load row1 => accumulate in vrow0 with k5..k9
    ////////////////////////////////////////////
    vle8.v   vload0, (ap)
    vwmacc.vx vrow0, k5, vload0
    vwmul.vx  vrow1, vload0, k0

    vle8.v   vload1, (ap_1)
    vwmacc.vx vrow0, k6, vload1
    vwmacc.vx vrow1, k1, vload1

    vle8.v   vload2, (ap_2)
    vwmacc.vx vrow0, k7, vload2
    vwmacc.vx vrow1, k2, vload2

    vle8.v   vload3, (ap_3)
    vwmacc.vx vrow0, k8, vload3
    vwmacc.vx vrow1, k3, vload3

    vle8.v   vload4, (ap_4)
    vwmacc.vx vrow0, k9, vload4
    vwmacc.vx vrow1, k4, vload4
    

    // advance pointers to row2
    add   ap, ap, a_stride

    ////////////////////////////////////////////
    // 3) Load row2 => accumulate in vrow0 with k10..k14
    ////////////////////////////////////////////
    vle8.v   vload0, (ap)
    addi  ap_1,  ap,  1
    vwmacc.vx vrow1, k5, vload0
    vwmul.vx  vrow2, vload0, k0

    vle8.v   vload1, (ap_1)
    addi  ap_2,  ap,  2
    vwmacc.vx vrow2, k1, vload1
    vwmacc.vx vrow1, k6, vload1

    vle8.v   vload2, (ap_2)
    addi  ap_3,  ap,  3
    vwmacc.vx vrow2, k2, vload2
    vwmacc.vx vrow1, k7, vload2

    vle8.v   vload3, (ap_3)
    addi  ap_4,  ap,  4
    vwmacc.vx vrow2, k3, vload3
    vwmacc.vx vrow1, k8, vload3

    vle8.v   vload4, (ap_4)
    add   ap, ap, a_stride
    vwmacc.vx vrow2, k4, vload4
    vwmacc.vx vrow1, k9, vload4

    // advance pointers to row3


    ////////////////////////////////////////////
    // 4) Load row3 => accumulate in vrow0 with k15..k19
    ////////////////////////////////////////////
    lb k9, 10(k)
    vwmacc.vx vrow0, k9, vload0 // k10
    vle8.v   vload0, (ap)
    addi  ap_1,  ap,  1
    vwmacc.vx vrow2, k5, vload0
    vwmul.vx vrow3, vload0, k0
    lb k5, 15(k)
    vwmacc.vx vrow1, k9, vload0 // k10
    vwmacc.vx vrow0, k5, vload0 // k15


    lb k8, 11(k)
    vwmacc.vx vrow0, k8, vload1 // k11
    vle8.v   vload1, (ap_1)
    addi  ap_2,  ap,  2
    vwmacc.vx vrow3, k1, vload1
    vwmacc.vx vrow2, k6, vload1
    lb k6, 16(k)
    vwmacc.vx vrow1, k8, vload1 // k11
    vwmacc.vx vrow0, k6, vload1 // k16


    lb k0, 12(k)
    vwmacc.vx vrow0, k0, vload2 // k12
    vle8.v   vload2, (ap_2)
    addi  ap_3,  ap,  3
    vwmacc.vx vrow3, k2, vload2
    vwmacc.vx vrow2, k7, vload2
    lb k7, 17(k)
    vwmacc.vx vrow1, k0, vload2 // k12
    vwmacc.vx vrow0, k7, vload2 // k17


    lb k1, 13(k)
    vwmacc.vx vrow0, k1, vload3 // k13
    vle8.v   vload3, (ap_3)
    addi  ap_4,  ap,  4
    vwmacc.vx vrow3, k3, vload3
    lb k2, 8(k)
    vwmacc.vx vrow2, k2, vload3 // k8
    lb k3, 18(k)
    vwmacc.vx vrow0, k3, vload3 // k18
    vwmacc.vx vrow1, k1, vload3 // k13
    
    lb k2, 14(k)
    vwmacc.vx vrow0, k2, vload4 // k14
    vle8.v   vload4, (ap_4)
    vwmacc.vx vrow3, k4, vload4
    vwmacc.vx vrow1, k2, vload4 // k14
    lb k4, 19(k)
    lb k9, 9(k)
    vwmacc.vx vrow0, k4, vload4 // k19
    vwmacc.vx vrow2, k9, vload4
    

    // advance pointers to row4
    add   ap, ap, a_stride

    ////////////////////////////////////////////
    // 5) Load row4 => begin partial sums in vrow1 with k0..k4
    ////////////////////////////////////////////
    vle8.v   vload0, (ap)
    addi  ap_1,  ap,  1

    vle8.v   vload1, (ap_1)
    addi  ap_2,  ap,  2

    vle8.v   vload2, (ap_2)
    addi  ap_3,  ap,  3

    vle8.v   vload3, (ap_3)
    addi  ap_4,  ap,  4
    lb k9, 10(k)

    vle8.v   vload4, (ap_4)
    beqz row_count, epilog

    
//////////////////////////////////////////////////////////////////////
//  Main Loop: process pairs of output rows until row_count <= 0
//////////////////////////////////////////////////////////////////////
conv_loop:
    //------------------------------------------------
    // Finish vrow0 with row4 => k20..k24
    //------------------------------------------------
    vwmacc.vx  vrow2, k9, vload0 // k10
    vwmacc.vx  vrow2, k8, vload1 // k11
    vwmacc.vx  vrow2, k0, vload2 // k12
    vwmacc.vx  vrow2, k1, vload3 // k13
    vwmacc.vx  vrow2, k2, vload4 // k14
    vwmacc.vx  vrow1, k5, vload0 // k15
    vwmacc.vx  vrow1, k6, vload1 // k16
    vwmacc.vx  vrow1, k7, vload2 // k17
    vwmacc.vx  vrow1, k3, vload3 // k18
    vwmacc.vx  vrow1, k4, vload4 // k19

    lb k0, 20(k)
    vwmacc.vx  vrow0, k0, vload0 // k20
    lb k1, 21(k)
    vwmacc.vx  vrow0, k1, vload1
    lb k2, 22(k)
    vwmacc.vx  vrow0, k2, vload2
    lb k3, 23(k)
    vwmacc.vx  vrow0, k3, vload3
    lb k4, 24(k)
    vwmacc.vx  vrow0, k4, vload4
    
    lb  k0,   0(k)
    add ap, ap, a_stride
    lb  k1,   1(k)
    addi  ap_1,  ap,  1
    lb  k2,   2(k)
    addi  ap_2,  ap,  2
    lb  k3,  3(k)
    addi  ap_3,  ap,  3
    lb  k4,  4(k)
    addi  ap_4,  ap,  4
    lb  k5,  5(k)
    lb  k6,  6(k)
    lb  k7,  7(k)
    vwmacc.vx  vrow3, k5, vload0
    lb  k8,  8(k)
    vwmacc.vx  vrow3, k7, vload2
    vwmacc.vx  vrow3, k8, vload3
    lb  k9,  9(k)
    vwmacc.vx  vrow3, k6, vload1
    vwmacc.vx  vrow3, k9, vload4
    
    // store vrow0 => first of the two output rows
    vse16.v   vrow0, (bp)


    // load row5 => partial sums for vrow0 with k0..k4
    vwmul.vx vrow0, vload0, k0
    vle8.v   vload0, (ap)
    vwmacc.vx vrow0, k1, vload1
    vle8.v vload1, (ap_1)
    add bp, bp, b_stride
    vwmacc.vx vrow0, k2, vload2
    vle8.v vload2, (ap_2)
    vwmacc.vx vrow0, k3, vload3
    vle8.v vload3, (ap_3)
    vwmacc.vx vrow0, k4, vload4
    vle8.v vload4, (ap_4)


    vwmacc.vx vrow0, k5, vload0
    vwmacc.vx vrow0, k6, vload1
    vwmacc.vx vrow0, k7, vload2
    vwmacc.vx vrow0, k8, vload3
    vwmacc.vx vrow0, k9, vload4

    lb  k0,   10(k)
    lb  k1,   11(k)
    vwmacc.vx vrow3, k0, vload0
    lb  k2,   12(k)
    vwmacc.vx  vrow3, k1, vload1
    lb  k3,  13(k)
    vwmacc.vx  vrow3, k2, vload2
    lb  k4,  14(k)
    vwmacc.vx  vrow3, k3, vload3
    lb  k5,  15(k)
    vwmacc.vx  vrow3, k4, vload4
    lb  k6,  16(k)
    vwmacc.vx vrow2, k5, vload0
    lb  k7,  17(k)
    vwmacc.vx  vrow2, k6, vload1
    lb  k8,  18(k)
    vwmacc.vx  vrow2, k7, vload2
    lb  k9,  19(k)
    vwmacc.vx  vrow2, k8, vload3
    vwmacc.vx  vrow2, k9, vload4


    lb  k5,  20(k)
    add ap, ap, a_stride
    vwmacc.vx  vrow1, k5, vload0
    lb  k6,  21(k)
    addi ap_1, ap, 1
    vwmacc.vx  vrow1, k6, vload1
    lb  k7,  22(k)
    addi ap_2, ap, 2
    vwmacc.vx  vrow1, k7, vload2
    lb  k8,  23(k)
    addi ap_3,  ap,  3
    vwmacc.vx  vrow1, k8, vload3
    lb  k9,  24(k)
    addi ap_4,  ap,  4
    vwmacc.vx  vrow1, k9, vload4

    lb  k0,   0(k)
    lb  k1,   1(k)
    lb  k2,   2(k)
    lb  k3,   3(k)
    lb  k4,   4(k) 

    vse16.v vrow1, (bp)

    vwmul.vx vrow1, vload0, k0
    vle8.v   vload0, (ap)
    vwmacc.vx vrow1, k1, vload1
    vle8.v vload1, (ap_1)
    add bp, bp, b_stride
    vwmacc.vx vrow1, k2, vload2
    vle8.v vload2, (ap_2)
    vwmacc.vx vrow1, k3, vload3
    vle8.v vload3, (ap_3)
    vwmacc.vx vrow1, k4, vload4
    vle8.v vload4, (ap_4)

    vwmacc.vx vrow2, k5, vload0 //k20
    vwmacc.vx  vrow2, k6, vload1 //k21
    vwmacc.vx  vrow2, k7, vload2 //k22
    vwmacc.vx  vrow2, k8, vload3 //k23
    vwmacc.vx  vrow2, k9, vload4 //k24

    lb  k0,   10(k)
    lb  k1,   11(k)
    vwmacc.vx vrow0, k0, vload0
    lb  k2,   12(k)
    vwmacc.vx vrow0, k1, vload1
    lb  k3,  13(k)
    vwmacc.vx vrow0, k2, vload2
    lb  k4,  14(k)
    vwmacc.vx vrow0, k3, vload3
    lb  k5,  15(k)
    vwmacc.vx vrow0, k4, vload4
    lb  k6,  16(k)
    vwmacc.vx vrow3, k5, vload0
    lb  k7,  17(k)
    vwmacc.vx  vrow3, k6, vload1
    lb  k8,  18(k)
    vwmacc.vx  vrow3, k7, vload2
    lb  k9,  19(k)
    vwmacc.vx  vrow3, k8, vload3
    vwmacc.vx  vrow3, k9, vload4

    vse16.v vrow2, (bp)

    lb  k0,   0(k)
    add ap, ap, a_stride
    lb  k1,   1(k)
    addi ap_1, ap, 1
    lb  k2,   2(k)
    addi ap_2, ap, 2
    lb  k3,  3(k)
    addi ap_3,  ap,  3
    lb  k4,  4(k)
    addi ap_4,  ap,  4
    lb  k5,  5(k)
    lb  k6,  6(k)
    vwmacc.vx vrow1, k5, vload0
    lb  k7,  7(k)
    vwmacc.vx  vrow1, k6, vload1
    lb  k8,  8(k)
    vwmacc.vx  vrow1, k7, vload2
    lb  k9,  9(k)
    vwmacc.vx  vrow1, k8, vload3
    vwmacc.vx  vrow1, k9, vload4

    // missing the elements here to place, k0-k9 are already loaded

    vwmul.vx vrow2, vload0, k0
    vle8.v   vload0, (ap)
    vwmacc.vx vrow2, k1, vload1
    vle8.v vload1, (ap_1)
    add bp, bp, b_stride
    vwmacc.vx vrow2, k2, vload2
    vle8.v vload2, (ap_2)
    vwmacc.vx vrow2, k3, vload3
    vle8.v vload3, (ap_3)
    vwmacc.vx vrow2, k4, vload4
    vle8.v vload4, (ap_4)

    vwmacc.vx vrow2, k5, vload0
    vwmacc.vx  vrow2, k6, vload1
    vwmacc.vx  vrow2, k7, vload2
    vwmacc.vx  vrow2, k8, vload3
    vwmacc.vx  vrow2, k9, vload4

    lb  k0,   10(k)
    lb  k1,   11(k)
    vwmacc.vx vrow1, k0, vload0
    lb  k2,   12(k)
    vwmacc.vx  vrow1, k1, vload1
    lb  k3,  13(k)
    vwmacc.vx  vrow1, k2, vload2
    lb  k4,  14(k)
    vwmacc.vx  vrow1, k3, vload3
    lb  k5,  15(k)
    vwmacc.vx  vrow1, k4, vload4
    lb  k6,  16(k)
    vwmacc.vx vrow0, k5, vload0
    lb  k7,  17(k)
    vwmacc.vx vrow0, k6, vload1
    lb  k8,  18(k)
    vwmacc.vx vrow0, k7, vload2
    lb  k9,  19(k)
    vwmacc.vx vrow0, k8, vload3
    vwmacc.vx vrow0, k9, vload4


    lb  k0,   0(k)
    add ap, ap, a_stride
    lb  k1,   1(k)
    addi ap_1, ap, 1
    lb  k2,   2(k)
    addi ap_2, ap, 2
    lb  k3,  3(k)
    addi ap_3,  ap,  3
    lb  k4,  4(k)
    addi ap_4,  ap,  4
    lb  k5,  20(k)
    lb  k6,  21(k)
    vwmacc.vx vrow3, k5, vload0
    lb  k7,  22(k)
    vwmacc.vx  vrow3, k6, vload1
    lb  k8,  23(k)
    vwmacc.vx  vrow3, k7, vload2
    lb  k9,  24(k)
    vwmacc.vx  vrow3, k8, vload3
    vwmacc.vx  vrow3, k9, vload4


    vse16.v vrow3, (bp)

    vwmul.vx vrow3, vload0, k0
    vle8.v   vload0, (ap)
    vwmacc.vx vrow3, k1, vload1
    vle8.v vload1, (ap_1)
    add bp, bp, b_stride
    vwmacc.vx vrow3, k2, vload2
    vle8.v vload2, (ap_2)
    vwmacc.vx vrow3, k3, vload3
    vle8.v vload3, (ap_3)
    vwmacc.vx vrow3, k4, vload4
    vle8.v vload4, (ap_4)

	addi row_count, row_count, -4

    lb  k0,  12(k)
    lb  k1,  13(k)
    lb  k2,  14(k)
    lb  k3,  18(k)
    lb  k4,  19(k)
    lb  k5,  15(k)
    lb  k6,  16(k)
    lb  k7,  17(k)
    lb  k8,  11(k)
    lb  k9,  10(k)

	bgtz row_count, conv_loop



//////////////////////////////////////////////////////////////////////
//  Epilog: handle finishing up the pipeline plus leftover row if rows_odd != 0
//////////////////////////////////////////////////////////////////////

epilog:
    vwmacc.vx vrow2, k9, vload0
    vwmacc.vx vrow2, k8, vload1
    vwmacc.vx vrow2, k0, vload2
    vwmacc.vx vrow2, k1, vload3
    vwmacc.vx vrow2, k2, vload4
    vwmacc.vx vrow1, k5, vload0
    vwmacc.vx vrow1, k6, vload1
    vwmacc.vx vrow1, k7, vload2
    vwmacc.vx vrow1, k3, vload3
    vwmacc.vx vrow1, k4, vload4

    lb  k0,  20(k)
    lb  k1,  21(k)
    vwmacc.vx vrow0, k0, vload0
    lb  k2,  22(k)
    vwmacc.vx vrow0, k1, vload1
    lb  k3,  23(k)
    vwmacc.vx vrow0, k2, vload2
    lb  k4,  24(k)
    vwmacc.vx vrow0, k3, vload3
    lb  k5,  5(k)
    vwmacc.vx vrow0, k4, vload4
    lb  k6,  6(k)
    vwmacc.vx vrow3, k5, vload0
    lb  k7,  7(k)
    vwmacc.vx  vrow3, k6, vload1
    lb  k8,  8(k)
    vwmacc.vx  vrow3, k7, vload2
    lb  k9,  9(k)
    vwmacc.vx  vrow3, k8, vload3
    vwmacc.vx  vrow3, k9, vload4

    add ap, ap, a_stride
    addi ap_1, ap, 1
    addi ap_2, ap, 2
    addi ap_3,  ap,  3
    addi ap_4,  ap,  4

    vse16.v vrow0, (bp)
    add bp, bp, b_stride
    vle8.v vload0, (ap)
    vle8.v vload1, (ap_1)
    vle8.v vload2, (ap_2)
    vle8.v vload3, (ap_3)
    vle8.v vload4, (ap_4)

    vwmacc.vx vrow1, k0, vload0
    vwmacc.vx vrow1, k1, vload1
    vwmacc.vx vrow1, k2, vload2
    vwmacc.vx vrow1, k3, vload3
    vwmacc.vx vrow1, k4, vload4

    lb  k0,  10(k)
    lb  k1,  11(k)
    vwmacc.vx vrow3, k0, vload0
    lb  k2,  12(k)
    vwmacc.vx  vrow3, k1, vload1
    lb  k3,  13(k)
    vwmacc.vx  vrow3, k2, vload2
    lb  k4,  14(k)
    vwmacc.vx  vrow3, k3, vload3
    lb  k5,  15(k)
    vwmacc.vx  vrow3, k4, vload4
    lb  k6,  16(k)
    vwmacc.vx vrow2, k5, vload0
    lb  k7,  17(k)
    vwmacc.vx  vrow2, k6, vload1
    lb  k8,  18(k)
    vwmacc.vx  vrow2, k7, vload2
    lb  k9,  19(k)
    vwmacc.vx  vrow2, k8, vload3
    vwmacc.vx  vrow2, k9, vload4

    add ap, ap, a_stride
    addi ap_1, ap, 1
    addi ap_2, ap, 2
    addi ap_3,  ap,  3
    addi ap_4,  ap,  4

    vse16.v vrow1, (bp)
    add bp, bp, b_stride
    vle8.v vload0, (ap)
    vle8.v vload1, (ap_1)
    vle8.v vload2, (ap_2)
    vle8.v vload3, (ap_3)
    vle8.v vload4, (ap_4)

    lb  k0,  20(k)
    vwmacc.vx vrow3, k5, vload0
    lb  k1,  21(k)
    vwmacc.vx  vrow3, k6, vload1
    lb  k2,  22(k)
    vwmacc.vx  vrow3, k7, vload2
    lb  k3,  23(k)
    vwmacc.vx  vrow3, k8, vload3
    lb  k4,  24(k)
    vwmacc.vx  vrow3, k9, vload4

    vwmacc.vx vrow2, k0, vload0
    add ap, ap, a_stride
    vwmacc.vx  vrow2, k1, vload1
    addi ap_1, ap, 1
    vwmacc.vx  vrow2, k2, vload2
    addi ap_2, ap, 2
    vwmacc.vx  vrow2, k3, vload3
    addi ap_3,  ap,  3
    vwmacc.vx  vrow2, k4, vload4
    addi ap_4,  ap,  4

    vse16.v vrow2, (bp)
    add bp, bp, b_stride
    vle8.v vload0, (ap)
    vle8.v vload1, (ap_1)
    vle8.v vload2, (ap_2)
    vle8.v vload3, (ap_3)
    vle8.v vload4, (ap_4)


    vwmacc.vx vrow3, k0, vload0
    vwmacc.vx  vrow3, k1, vload1
    vwmacc.vx  vrow3, k2, vload2
    vwmacc.vx  vrow3, k3, vload3
    vwmacc.vx  vrow3, k4, vload4

    vse16.v vrow3, (bp)


row_loop_complete:
    // Move a,b pointers to next column-block (just like the 3×3 code),
    // then decrement 'cols', repeating if needed
    add a, a, vlen
    add b, b, VLEN_stride

    sub cols, cols, vlen
    bnez cols, loop_prolog

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
    ld      s11, 88(sp)
    addi sp, sp, FRAMESIZE
    ret