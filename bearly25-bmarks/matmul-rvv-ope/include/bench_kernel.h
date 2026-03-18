/*
 * bench_kernel.h - i8->i32 matmul kernel entry points.
 *
 * A_T is [K x M] int8 (A transposed), B is [K x N] int8, C is [M x N] int32.
 * No bias. For this benchmark M=N=K=60.
 */
#ifndef MATMUL_BENCH_KERNEL_H
#define MATMUL_BENCH_KERNEL_H

#include <stddef.h>
#include <stdint.h>

// Pure RVV kernel (baseline): A_T [K x M], a_row_stride = M
void i8_i32_matmul(size_t M, size_t N, size_t K,
                   const int8_t *A_T, size_t a_row_stride,
                   const int8_t *B,
                   int32_t *C, size_t c_row_stride, size_t c_col_stride);

// Interleaved RVV+OPE kernel: 15-row tiles (7 RVV + 8 OPE).
//   A_T      - [K x M] transposed A, a_row_stride = M
//              RVV reads cols 0-6, OPE reads cols 7-14 of each 15-row tile.
//              Both access A_T directly; no separate remapped A needed.
//   B        - [K x N] raw B, row-major, used by RVV
//   B_ope    - [N_ope_tiles x K x 8] remapped B for OPE
//              B_ope[j*K*8 + k*8 + e] = B[k*N + j*8 + e]  (0 if out of bounds)
//   N_ope_tiles - ceil(N/8)
void i8_i32_matmul_interleaved(size_t M, size_t N, size_t K,
                                const int8_t *A_T, size_t a_row_stride,
                                const int8_t *B,
                                const int8_t *B_ope, size_t N_ope_tiles,
                                int32_t *C, size_t c_row_stride);

#endif // MATMUL_BENCH_KERNEL_H
