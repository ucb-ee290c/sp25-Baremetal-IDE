/*
 * bench_kernel.h - i8->i32 matmul kernel entry point.
 *
 * A is [M x K] int8, B is [K x N] int8, C is [M x N] int32.
 * No bias. For this benchmark M=N=K=64.
 */
#ifndef MATMUL_BENCH_KERNEL_H
#define MATMUL_BENCH_KERNEL_H

#include <stddef.h>
#include <stdint.h>

void i8_i32_matmul(size_t M, size_t N, size_t K,
                   const int8_t *A, size_t a_row_stride,
                   const int8_t *B,
                   int32_t *C, size_t c_row_stride, size_t c_col_stride);

#endif // MATMUL_BENCH_KERNEL_H
