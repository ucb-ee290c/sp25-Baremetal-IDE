#if defined(__riscv_zvfh) && (__riscv_zvfh > 0)

    .text
    .align 2
    .global riscv_cfft_radix4by2_pre_f16_rvv_asm
    .type riscv_cfft_radix4by2_pre_f16_rvv_asm, @function
riscv_cfft_radix4by2_pre_f16_rvv_asm:
    beqz a6, .Lrad_done
    li t3, 4

.Lrad_loop:
    vsetvli t0, a6, e16, m8, ta, ma

    vlse16.v v0, (a0), t3
    vlse16.v v1, (a1), t3
    vlse16.v v2, (a2), t3
    vlse16.v v3, (a3), t3
    vlse16.v v4, (a4), t3
    vlse16.v v5, (a5), t3

    vfadd.vv v6, v0, v2
    vfsub.vv v7, v0, v2
    vfsub.vv v8, v1, v3
    vfadd.vv v9, v3, v1

    vfmul.vv v10, v7, v4
    vfmul.vv v11, v8, v5
    vfmul.vv v12, v8, v4
    vfmul.vv v13, v7, v5

    vfadd.vv v14, v10, v11
    vfsub.vv v15, v12, v13

    vsse16.v v6, (a0), t3
    vsse16.v v9, (a1), t3
    vsse16.v v14, (a2), t3
    vsse16.v v15, (a3), t3

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
