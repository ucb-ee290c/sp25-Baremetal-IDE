/*
 * bench_fill.h - Input generation and reference helpers for core-v-ope benchmarks.
*/

#ifndef CORE_V_OPE_BENCH_FILL_H
#define CORE_V_OPE_BENCH_FILL_H

#include <stdint.h>

void bench_fill_int8_small(int8_t *data, int rows, int cols, int stride);
void bench_fill_int32_zero(int32_t *data, int rows, int cols, int stride);
void bench_fill_int8_zero(int8_t *data, int rows, int cols, int stride);

void bench_ref_gemm_i8_i8_i32(const int8_t *A, const int8_t *B,
                              int32_t *C,
                              int M, int N, int K,
                              int ldA, int ldB, int ldC);

void bench_ref_quant_i32_to_i8(const int32_t *src, int8_t *dst,
                               int rows, int cols,
                               int ld_src, int ld_dst,
                               const float *scale,
                               int32_t zero_point);

int bench_compare_i32(const int32_t *got, int ld_got,
                      const int32_t *ref, int ld_ref,
                      int M, int N, int verbose);

int bench_compare_i8(const int8_t *got, int ld_got,
                     const int8_t *ref, int ld_ref,
                     int M, int N, int verbose);

#endif // CORE_V_OPE_BENCH_FILL_H
