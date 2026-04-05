#ifndef BORAI_TEST_TINY_GEMM_I8_RVV_H
#define BORAI_TEST_TINY_GEMM_I8_RVV_H

#include <stdint.h>

/* Returns 1 if a specialized tiny-shape kernel handled this matmul, 0 otherwise.
 * Expected B_pack layout: [(n_in+1) x n_out] int8, with row 0 as bias row. */
int borai_tiny_matmul_t_i8_fout(
    const int8_t* xq,
    const int8_t* w_t_pack,
    float* xout,
    int n_in,
    int n_out,
    float scale);

#endif
