#if defined(__riscv_zvfh) && (__riscv_zvfh > 0)

    .text
    .align 2
    .global riscv_rfft_stage_f16_rvv_asm
    .type riscv_rfft_stage_f16_rvv_asm, @function
riscv_rfft_stage_f16_rvv_asm:
    beqz a5, .Lrfft_done
    li t3, 4
    neg t4, t3
    la t5, .Lrfft_half
    flh ft0, 0(t5)

.Lrfft_loop:
    vsetvli t0, a5, e16, m2, ta, ma

    vlse16.v v0, (a0), t3
    addi t1, a0, 2
    vlse16.v v2, (t1), t3

    vlse16.v v4, (a1), t4
    addi t1, a1, 2
    vlse16.v v6, (t1), t4

    vlse16.v v8, (a2), t3
    vlse16.v v10, (a3), t3

    vfsub.vv v12, v4, v0
    vfadd.vv v14, v6, v2

    vfmul.vv v16, v8, v12
    vfmul.vv v18, v10, v12
    vfmul.vv v20, v8, v14
    vfmul.vv v22, v10, v14

    vfadd.vv v24, v0, v4
    vfadd.vv v24, v24, v16
    vfadd.vv v24, v24, v22
    vfmul.vf v24, v24, ft0

    vfsub.vv v26, v2, v6
    vfadd.vv v26, v26, v18
    vfsub.vv v26, v26, v20
    vfmul.vf v26, v26, ft0

    vsse16.v v24, (a4), t3
    addi t1, a4, 2
    vsse16.v v26, (t1), t3

    slli t2, t0, 2
    add a0, a0, t2
    sub a1, a1, t2
    add a2, a2, t2
    add a3, a3, t2
    add a4, a4, t2

    sub a5, a5, t0
    bnez a5, .Lrfft_loop

.Lrfft_done:
    ret

    .section .rodata
    .align 1
.Lrfft_half:
    .half 0x3800

#else

    .text
    .align 2
    .global riscv_rfft_stage_f16_rvv_asm
    .type riscv_rfft_stage_f16_rvv_asm, @function
riscv_rfft_stage_f16_rvv_asm:
    ret

#endif
