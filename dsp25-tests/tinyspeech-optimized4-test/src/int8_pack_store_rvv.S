#if defined(__riscv_vector)

    .text
    .align 2
    .global tinyspeech_int8_pack_store_u7_rvv
    .type tinyspeech_int8_pack_store_u7_rvv, @function
tinyspeech_int8_pack_store_u7_rvv:
    beqz a2, .Lpack_done

.Lpack_loop:
    vsetvli t0, a2, e32, m4, ta, ma
    vle32.v v8, (a0)

    vsetvli zero, t0, e16, m2, ta, ma
    vnclip.wi v10, v8, 0

    vsetvli zero, t0, e8, m1, ta, ma
    vnclip.wi v12, v10, 0
    vse8.v v12, (a1)

    slli t1, t0, 2
    add a0, a0, t1
    add a1, a1, t0
    sub a2, a2, t0
    bnez a2, .Lpack_loop

.Lpack_done:
    ret

#else

    .text
    .align 2
    .global tinyspeech_int8_pack_store_u7_rvv
    .type tinyspeech_int8_pack_store_u7_rvv, @function
tinyspeech_int8_pack_store_u7_rvv:
    ret

#endif
