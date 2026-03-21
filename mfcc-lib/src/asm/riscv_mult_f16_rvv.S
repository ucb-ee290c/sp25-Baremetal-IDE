#if defined(__riscv_zvfh) && (__riscv_zvfh > 0)

    .text
    .align 2
    .global riscv_mult_f16_rvv_asm
    .type riscv_mult_f16_rvv_asm, @function
riscv_mult_f16_rvv_asm:
    beqz a3, .Ldone
.Lloop:
    vsetvli t0, a3, e16, m8, ta, ma
    vle16.v v0, (a0)
    vle16.v v8, (a1)
    vfmul.vv v16, v0, v8
    vse16.v v16, (a2)

    slli t1, t0, 1
    add a0, a0, t1
    add a1, a1, t1
    add a2, a2, t1
    sub a3, a3, t0
    bnez a3, .Lloop
.Ldone:
    ret

#else

    .text
    .align 2
    .global riscv_mult_f16_rvv_asm
    .type riscv_mult_f16_rvv_asm, @function
riscv_mult_f16_rvv_asm:
    ret

#endif

