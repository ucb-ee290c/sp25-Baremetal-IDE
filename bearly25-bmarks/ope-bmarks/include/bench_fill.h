#ifndef BENCH_FILL_H
#define BENCH_FILL_H

#include "bench_config.h"

// Fill an int8 matrix with a deterministic pseudo-random pattern
void bench_fill_int8_pattern(int8_t *data, int rows, int cols, int stride);

// Zero an int32 matrix (rows x cols, stride)
void bench_fill_int32_zero(int32_t *data, int rows, int cols, int stride);

// CPU reference GEMM: C = A^T * B
// A: MxK, row-major with ldA
// B: KxN, row-major with ldB
// C_ref: MxN, row-major with ldC
void bench_ref_gemm_AT_i8i8_i32(const int8_t *A, const int8_t *B,
                                int32_t *C,
                                int M, int N, int K,
                                int ldA, int ldB, int ldC);

// Compare OPE vs CPU result
int bench_compare_results(const int32_t *C_ope, int ldc_ope,
                          const int32_t *C_ref, int ldc_ref,
                          int M, int N, int verbose);

#endif // BENCH_FILL_H
