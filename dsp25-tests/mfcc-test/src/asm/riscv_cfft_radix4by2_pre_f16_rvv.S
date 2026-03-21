#if defined(__riscv_zvfh) && (__riscv_zvfh > 0)

    .text
    .align 2
    .global riscv_cfft_radix4by2_pre_f16_rvv_asm
    .type riscv_cfft_radix4by2_pre_f16_rvv_asm, @function
riscv_cfft_radix4by2_pre_f16_rvv_asm:
    beqz a6, .Lrad_done
    li t3, 4

.Lrad_loop:
    vsetvli t0, a6, e16, m2, ta, ma

    vlse16.v v0, (a0), t3
    vlse16.v v2, (a1), t3
    vlse16.v v4, (a2), t3
    vlse16.v v6, (a3), t3
    vlse16.v v8, (a4), t3
    vlse16.v v10, (a5), t3

    vfadd.vv v12, v0, v4
    vfsub.vv v14, v0, v4
    vfsub.vv v16, v2, v6
    vfadd.vv v18, v6, v2

    vfmul.vv v20, v14, v8
    vfmul.vv v22, v16, v10
    vfmul.vv v24, v16, v8
    vfmul.vv v26, v14, v10

    vfadd.vv v28, v20, v22
    vfsub.vv v30, v24, v26

    vsse16.v v12, (a0), t3
    vsse16.v v18, (a1), t3
    vsse16.v v28, (a2), t3
    vsse16.v v30, (a3), t3

    slli t2, t0, 2
    add a0, a0, t2
    add a1, a1, t2
    add a2, a2, t2
    add a3, a3, t2
    add a4, a4, t2
    add a5, a5, t2

    sub a6, a6, t0
    bnez a6, .Lrad_loop

.Lrad_done:
    ret

#else

    .text
    .align 2
    .global riscv_cfft_radix4by2_pre_f16_rvv_asm
    .type riscv_cfft_radix4by2_pre_f16_rvv_asm, @function
riscv_cfft_radix4by2_pre_f16_rvv_asm:
    ret

#endif
