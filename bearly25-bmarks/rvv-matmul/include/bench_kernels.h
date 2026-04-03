/*
 * bench_kernels.h - RVV matmul kernel entry points.
 *
 * Definitions are provided by src/bench_impl.c which includes
 * the RVV kernel source files.
 */
#ifndef RVV_BENCH_KERNELS_H
#define RVV_BENCH_KERNELS_H

#include <stddef.h>
#include <stdint.h>

void f32_gemm(size_t M, size_t N, size_t K,
              const float *A, size_t a_row_stride,
              const float *B,
              float *C, size_t c_row_stride, size_t c_col_stride);

void f32_gemm_packed(size_t M, size_t N, size_t K,
                     const float *A, size_t a_row_stride,
                     const float *B,
                     float *C, size_t c_row_stride, size_t c_col_stride);

void pack_weight_matrix_f32(size_t K, size_t N,
                            const float *B, float *B_packed);

void int8_int16_gemm(size_t M, size_t N, size_t K,
                     const int8_t *A, size_t a_row_stride,
                     const int8_t *B,
                     int16_t *C, size_t c_row_stride, size_t c_col_stride);

void int8_int16_gemm_packed(size_t M, size_t N, size_t K,
                            const int8_t *A, size_t a_row_stride,
                            const int8_t *B,
                            int16_t *C, size_t c_row_stride, size_t c_col_stride);

void pack_weight_matrix_i8i16(size_t K, size_t N,
                              const int8_t *B, int8_t *B_packed);

void int8_int32_gemm(size_t M, size_t N, size_t K,
                     const int8_t *A, size_t a_row_stride,
                     const int8_t *B,
                     int32_t *C, size_t c_row_stride, size_t c_col_stride);

void int8_int32_gemm_packed(size_t M, size_t N, size_t K,
                            const int8_t *A, size_t a_row_stride,
                            const int8_t *B,
                            int32_t *C, size_t c_row_stride, size_t c_col_stride);

void pack_weight_matrix_i8i32(size_t K, size_t N,
                              const int8_t *B, int8_t *B_packed);

void int32_gemm(size_t M, size_t N, size_t K,
                const int32_t *A, size_t a_row_stride,
                const int32_t *B,
                int32_t *C, size_t c_row_stride, size_t c_col_stride);

void int32_gemm_packed(size_t M, size_t N, size_t K,
                       const int32_t *A, size_t a_row_stride,
                       const int32_t *B,
                       int32_t *C, size_t c_row_stride, size_t c_col_stride);

void pack_weight_matrix_i32(size_t K, size_t N,
                            const int32_t *B, int32_t *B_packed);

void int8_int8_gemm(size_t M, size_t N, size_t K,
                    const int8_t *A, size_t a_row_stride,
                    const int8_t *B,
                    int8_t *C, size_t c_row_stride, size_t c_col_stride);

void int8_int8_gemm_packed(size_t M, size_t N, size_t K,
                           const int8_t *A, size_t a_row_stride,
                           const int8_t *B,
                           int8_t *C, size_t c_row_stride, size_t c_col_stride);

void pack_weight_matrix_i8i8(size_t K, size_t N,
                             const int8_t *B, int8_t *B_packed);

#endif // RVV_BENCH_KERNELS_H
