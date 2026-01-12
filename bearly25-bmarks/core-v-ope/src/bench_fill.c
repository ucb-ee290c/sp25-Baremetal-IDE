/*
 * bench_fill.c - Input generation and reference math for core-v-ope benchmarks.
*/

#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline int32_t round_float_to_i32(float x) {
  return (int32_t)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

void bench_fill_int8_small(int8_t *data, int rows, int cols, int stride) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int v = (i * 3 + j * 5 + 1) % 13;
      v -= 6;
      data[i * stride + j] = (int8_t)v;
    }
  }
}

void bench_fill_int32_zero(int32_t *data, int rows, int cols, int stride) {
  for (int i = 0; i < rows; ++i) {
    memset(&data[i * stride], 0, (size_t)cols * sizeof(int32_t));
  }
}

void bench_fill_int8_zero(int8_t *data, int rows, int cols, int stride) {
  for (int i = 0; i < rows; ++i) {
    memset(&data[i * stride], 0, (size_t)cols * sizeof(int8_t));
  }
}

void bench_ref_gemm_i8_i8_i32(const int8_t *A, const int8_t *B,
                              int32_t *C,
                              int M, int N, int K,
                              int ldA, int ldB, int ldC) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += (int32_t)A[i * ldA + k] * (int32_t)B[k * ldB + j];
      }
      C[i * ldC + j] = acc;
    }
  }
}

void bench_ref_quant_i32_to_i8(const int32_t *src, int8_t *dst,
                               int rows, int cols,
                               int ld_src, int ld_dst,
                               const float *scale,
                               int32_t zero_point) {
  const int32_t min_less_zero = -128 - zero_point;
  const int32_t max_less_zero = 127 - zero_point;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float scaled = (float)src[i * ld_src + j] * scale[j];
      int32_t q = round_float_to_i32(scaled);
      q = clamp_i32(q, min_less_zero, max_less_zero);
      q += zero_point;
      q = clamp_i32(q, -128, 127);
      dst[i * ld_dst + j] = (int8_t)q;
    }
  }
}

int bench_compare_i32(const int32_t *got, int ld_got,
                      const int32_t *ref, int ld_ref,
                      int M, int N, int verbose) {
  int errors = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t a = got[i * ld_got + j];
      int32_t b = ref[i * ld_ref + j];
      if (a != b) {
        ++errors;
        if (verbose && errors < 16) {
          printf("  MISMATCH at (%d,%d): got=%d ref=%d\n", i, j, (int)a, (int)b);
        }
      }
    }
  }
  if (verbose) {
    if (errors == 0) {
      printf("  MATRIX OUTPUT MATCHES (M=%d, N=%d)\n", M, N);
    } else {
      printf("  MATRIX OUTPUT MISMATCH: %d incorrect elements out of %d\n",
             errors, M * N);
    }
  }
  return errors;
}

int bench_compare_i8(const int8_t *got, int ld_got,
                     const int8_t *ref, int ld_ref,
                     int M, int N, int verbose) {
  int errors = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int8_t a = got[i * ld_got + j];
      int8_t b = ref[i * ld_ref + j];
      if (a != b) {
        ++errors;
        if (verbose && errors < 16) {
          printf("  MISMATCH at (%d,%d): got=%d ref=%d\n", i, j, (int)a, (int)b);
        }
      }
    }
  }
  if (verbose) {
    if (errors == 0) {
      printf("  MATRIX OUTPUT MATCHES (M=%d, N=%d)\n", M, N);
    } else {
      printf("  MATRIX OUTPUT MISMATCH: %d incorrect elements out of %d\n",
             errors, M * N);
    }
  }
  return errors;
}
