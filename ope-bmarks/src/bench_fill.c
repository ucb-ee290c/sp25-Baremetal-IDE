#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "bench_fill.h"

void bench_fill_int8_pattern(int8_t *data, int rows, int cols, int stride) {
  // Simple deterministic pattern: f(i,j) = (i*13 + j*7) mod 127
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int v = (i * 13 + j * 7) & 0x7F;
      if (v & 0x40) v -= 0x80;
      data[i * stride + j] = (int8_t)v;
    }
  }
}

void bench_fill_int32_zero(int32_t *data, int rows, int cols, int stride) {
  for (int i = 0; i < rows; ++i) {
    memset(&data[i * stride], 0, (size_t)cols * sizeof(int32_t));
  }
}

void bench_ref_gemm_AT_i8i8_i32(const int8_t *A, const int8_t *B,
                                int32_t *C,
                                int M, int N, int K,
                                int ldA, int ldB, int ldC) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        // A is treated as A^T: we index A[k + i*ldA]
        acc += (int32_t)A[k + i * ldA] * (int32_t)B[k * ldB + j];
      }
      C[i * ldC + j] = acc;
    }
  }
}

int bench_compare_results(const int32_t *C_ope, int ldc_ope,
                          const int32_t *C_ref, int ldc_ref,
                          int M, int N, int verbose) {
  int errors = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t a = C_ope[i * ldc_ope + j];
      int32_t b = C_ref[i * ldc_ref + j];
      if (a != b) {
        ++errors;
        if (verbose && errors < 16) {
          printf("  MISMATCH at (%d,%d): OPE=%d, REF=%d\n", i, j, (int)a, (int)b);
        }
      }
    }
  }
  if (verbose) {
    if (errors == 0) {
      printf("  MATRIX OUTPUT MATCHES (M=%d, N=%d)\n", M, N);
    } else {
      printf("  MATRIX OUTPUT MISMATCH: %d incorrect elements out of %d\n", errors, M * N);
    }
  }
  return errors;
}
