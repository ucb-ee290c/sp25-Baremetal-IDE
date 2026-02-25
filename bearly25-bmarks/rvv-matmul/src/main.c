#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "simple_setup.h"

#define BENCH_RUNS      10
#define TARGET_FREQ     500000000UL

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

void int8_gemm(size_t M, size_t N, size_t K,
    const int8_t *A, size_t a_row_stride,
    const int8_t *B,
    int32_t *C, size_t c_row_stride, size_t c_col_stride);

void int8_gemm_packed(size_t M, size_t N, size_t K,
    const int8_t *A, size_t a_row_stride,
    const int8_t *B,
    int32_t *C, size_t c_row_stride, size_t c_col_stride);

void pack_weight_matrix_i8i32(size_t K, size_t N,
    const int8_t *B, int8_t *B_packed);

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

static void bench_f32(size_t N) {
    float *A        = malloc(N * N * sizeof(float));
    float *B        = malloc((N + 1) * N * sizeof(float));
    float *B_packed = malloc((N + 1) * N * sizeof(float));
    float *C        = malloc(N * N * sizeof(float));

    for (size_t i = 0; i < N * N; i++)       A[i] = (float)(i % 127);
    for (size_t i = 0; i < (N + 1) * N; i++) B[i] = (float)(i % 63);
    pack_weight_matrix_f32(N, N, B, B_packed);

    uint64_t sum_unpacked = 0, sum_packed = 0;
    for (int r = 0; r < BENCH_RUNS; r++) {
        memset(C, 0, N * N * sizeof(float));
        uint64_t t0 = rdcycle64();
        f32_gemm(N, N, N, A, N, B, C, N, 1);
        sum_unpacked += rdcycle64() - t0;
    }
    for (int r = 0; r < BENCH_RUNS; r++) {
        memset(C, 0, N * N * sizeof(float));
        uint64_t t0 = rdcycle64();
        f32_gemm_packed(N, N, N, A, N, B_packed, C, N, 1);
        sum_packed += rdcycle64() - t0;
    }

    printf("f32_gemm        %zux%zu: avg=%llu cycles\n",
           N, N, (unsigned long long)(sum_unpacked / BENCH_RUNS));
    printf("f32_gemm_packed %zux%zu: avg=%llu cycles\n",
           N, N, (unsigned long long)(sum_packed   / BENCH_RUNS));

    free(A); free(B); free(B_packed); free(C);
}

static void bench_i8i16(size_t N) {
    int8_t  *A        = malloc(N * N * sizeof(int8_t));
    int8_t  *B        = malloc((N + 1) * N * sizeof(int8_t));
    int8_t  *B_packed = malloc((N + 1) * N * sizeof(int8_t));
    int16_t *C        = malloc(N * N * sizeof(int16_t));

    for (size_t i = 0; i < N * N; i++)           A[i] = (int8_t)(i % 127);
    for (size_t i = 0; i < (N + 1) * N; i++)     B[i] = (int8_t)(i % 63);
    pack_weight_matrix_i8i16(N, N, B, B_packed);

    uint64_t sum_unpacked = 0, sum_packed = 0;
    for (int r = 0; r < BENCH_RUNS; r++) {
        memset(C, 0, N * N * sizeof(int16_t));
        uint64_t t0 = rdcycle64();
        int8_int16_gemm(N, N, N, A, N, B, C, N, 1);
        sum_unpacked += rdcycle64() - t0;
    }
    for (int r = 0; r < BENCH_RUNS; r++) {
        memset(C, 0, N * N * sizeof(int16_t));
        uint64_t t0 = rdcycle64();
        int8_int16_gemm_packed(N, N, N, A, N, B_packed, C, N, 1);
        sum_packed += rdcycle64() - t0;
    }

    printf("i8i16_gemm        %zux%zu: avg=%llu cycles\n",
           N, N, (unsigned long long)(sum_unpacked / BENCH_RUNS));
    printf("i8i16_gemm_packed %zux%zu: avg=%llu cycles\n",
           N, N, (unsigned long long)(sum_packed   / BENCH_RUNS));

    free(A); free(B); free(B_packed); free(C);
}

static void bench_i8i32(size_t N) {
    int8_t  *A        = malloc(N * N * sizeof(int8_t));
    int8_t  *B        = malloc((N + 1) * N * sizeof(int8_t));
    int8_t  *B_packed = malloc((N + 1) * N * sizeof(int8_t));
    int32_t *C        = malloc(N * N * sizeof(int32_t));

    for (size_t i = 0; i < N * N; i++)           A[i] = (int8_t)(i % 127);
    for (size_t i = 0; i < (N + 1) * N; i++)     B[i] = (int8_t)(i % 63);
    pack_weight_matrix_i8i32(N, N, B, B_packed);

    uint64_t sum_unpacked = 0, sum_packed = 0;
    for (int r = 0; r < BENCH_RUNS; r++) {
        memset(C, 0, N * N * sizeof(int32_t));
        uint64_t t0 = rdcycle64();
        int8_gemm(N, N, N, A, N, B, C, N, 1);
        sum_unpacked += rdcycle64() - t0;
    }
    for (int r = 0; r < BENCH_RUNS; r++) {
        memset(C, 0, N * N * sizeof(int32_t));
        uint64_t t0 = rdcycle64();
        int8_gemm_packed(N, N, N, A, N, B_packed, C, N, 1);
        sum_packed += rdcycle64() - t0;
    }

    printf("i8i32_gemm        %zux%zu: avg=%llu cycles\n",
           N, N, (unsigned long long)(sum_unpacked / BENCH_RUNS));
    printf("i8i32_gemm_packed %zux%zu: avg=%llu cycles\n",
           N, N, (unsigned long long)(sum_packed   / BENCH_RUNS));

    free(A); free(B); free(B_packed); free(C);
}

int main(void) {
    init_test(TARGET_FREQ);

    printf("=== RVV MATMUL BENCHMARKS ===\n");
    printf("runs per case: %d\n\n", BENCH_RUNS);

    printf("--- 32x32 ---\n");
    bench_f32(32);
    bench_i8i16(32);
    bench_i8i32(32);

    printf("\n--- 64x64 ---\n");
    bench_f32(64);
    bench_i8i16(64);
    bench_i8i32(64);

    printf("\n=== DONE ===\n");
    return 0;
}
