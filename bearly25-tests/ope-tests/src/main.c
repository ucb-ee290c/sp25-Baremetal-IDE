/* =========================================================================
 * main.c — Simple tests for OPE matmul
 * ========================================================================= */
#include "main.h"
#include <data/inputs.h>

/* ---------- Helpers ---------- */
static inline size_t ru8(size_t x) {
  return (x + 7u) & ~7u;
}

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

/* Scalar reference: C = A^T * B (A: MxK, B: KxN, C: MxN) */
static void ref_gemm_AT_i8i8_i32(const int8_t* A, const int8_t* B,
                                 int32_t* C, int M, int N, int K,
                                 int ldA, int ldB, int ldC) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += (int32_t)A[k + i * ldA] * (int32_t)B[k * ldB + j];
      }
      C[i * ldC + j] = acc;
    }
  }
}

static void print_matrix_i8(const char* name, const int8_t* matrix, 
                            int rows, int cols, int stride) {
  printf("\n%s (%dx%d):\n", name, rows, cols);
  for (int i = 0; i < rows; ++i) {
    printf("  [");
    for (int j = 0; j < cols; ++j) {
        printf("%4d", matrix[i * stride + j]);
        if (j < cols - 1) {
          printf(", ");
        }
    }
    printf("]\n");
  }
  printf("\n");
}

static void print_matrix_i32(const char* name, const int32_t* mat, int rows, int cols, int stride) {
  printf("\n%s (%dx%d):\n", name, rows, cols);
  for (int i = 0; i < rows; ++i) {
    printf("  [");
    for (int j = 0; j < cols; ++j) {
      printf("%6d", mat[i * stride + j]);
      if (j < cols - 1) printf(", ");
    }
    printf("]\n");
  }
  printf("\n");
}

static void print_diff_matrix(const int32_t* actual, const int32_t* expected,
                             int rows, int cols, int stride_actual, int stride_expected) {
  printf("\nDifference Matrix (Actual - Expected) (%dx%d):\n", rows, cols);
  for (int i = 0; i < rows; ++i) {
    printf("  [");
    for (int j = 0; j < cols; ++j) {
      int32_t diff = actual[i * stride_actual + j] - expected[i * stride_expected + j];
      printf("%6d", diff);
      if (j < cols - 1) {
        printf(", ");
      }
    }
    printf("]\n");
  }
  printf("\n");
}

static int compare_results(const int32_t* C, int ldc,
                           const int32_t* Cref, int ldcref,
                           int M, int N) {
    int errors = 0;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        int32_t a = C[i * ldc + j];
        int32_t b = Cref[i * ldcref + j];
        if (a != b) {
          if (errors == 0) {
            printf("First mismatch at (%d, %d): got %d, expected %d\n",
                    i, j, a, b);
          }
          errors++;
        }
      }
    }

    if (errors > 0) {
      printf("MATRIX OUTPUT MISMATCH: %d incorrect elements out of %d\n",
              errors, M * N);
      print_matrix_i32("Actual Output (OPE)", C, M, N, ldc);
      print_matrix_i32("Expected Output (CPU)", Cref, M, N, ldcref);
      return -1;
    } else {
      #if PRINT_SUCCESS_MATRICES
      printf("MATRIX OUTPUT MATCHES\n");
      print_matrix_i32("OPE Output", C, M, N, ldc);
      #endif
      return 0;
    }
  }

static int run_aligned_case(const OpeInputCase *tc) {
  const int M = tc->M;
  const int N = tc->N;
  const int K = tc->K;

  printf("\n=== Running test case: %s ===\n", tc->name);
  printf("Matrix dims: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

  // Allocate buffers (stack-local)
  int32_t C_ope[M * N];
  int32_t C_ref[M * N];
  memset(C_ope, 0, sizeof(C_ope));
  memset(C_ref, 0, sizeof(C_ref));

  #if PRINT_INPUT_MATRICES
  print_matrix_i8("Input Matrix A", tc->A, M, K, K);
  print_matrix_i8("Input Matrix B", tc->B, K, N, N);
  #endif

  // CPU reference
  ref_gemm_AT_i8i8_i32(tc->A, tc->B, C_ref, M, N, K, K, N, N);

  // Run OPE Accelerator
  printf("Running OPE Accelerator...\n");
  uint64_t t0 = rdcycle64();
  ope_matmul_m8m8(tc->A, tc->B, C_ope,
                                      M, N, K,
                                      K, N, N);
  uint64_t t1 = rdcycle64();
  printf("Aligned Execution time: %ld cycles\n", (unsigned long long)(t1 - t0));

  // Compare
  int ok = compare_results(C_ope, N, C_ref, N, M, N);
  printf("Result: %s\n", ok == 0 ? "PASS" : "FAIL");
  return ok;
}

static int run_unaligned_case(const OpeInputCase *tc) {
  const int M = tc->M;
  const int N = tc->N;
  const int K = tc->K;

  printf("\n=== Running test case: %s ===\n", tc->name);
  printf("Matrix dims: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

  // Allocate buffers
  int32_t C_ope[M * N];
  int32_t C_ref[M * N];
  memset(C_ope, 0, sizeof(C_ope));
  memset(C_ref, 0, sizeof(C_ref));

  #if PRINT_INPUT_MATRICES
  print_matrix_i8("Input Matrix A", tc->A, M, K, K);
  print_matrix_i8("Input Matrix B", tc->B, K, N, N);
  #endif

  // CPU reference
  ref_gemm_AT_i8i8_i32(tc->A, tc->B, C_ref, M, N, K, K, N, N);

  // Run OPE Accelerator
  printf("Running OPE Accelerator...\n");
  uint64_t t0 = rdcycle64();
  ope_matmul(tc->A, tc->B, C_ope,
                                      M, N, K,
                                      K, N, N);
  uint64_t t1 = rdcycle64();
  printf("Unaligned Execution time: %ld cycles\n", (unsigned long long)(t1 - t0));

  // Compare
  int ok = compare_results(C_ope, N, C_ref, N, M, N);
  printf("Result: %s\n", ok == 0 ? "PASS" : "FAIL");
  return ok;
}

static int run_case_prepacked(const OpeInputCase *tc)
{
  const int M = tc->M;
  const int N = tc->N;
  const int K = tc->K;

  printf("\n=== Running test case: %s ===\n", tc->name);
  printf("Matrix dims: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

  // Buffers
  int32_t C_ope[M * N];
  int32_t C_ref[M * N];
  memset(C_ope, 0, sizeof(C_ope));
  memset(C_ref, 0, sizeof(C_ref));

  // CPU reference
  ref_gemm_AT_i8i8_i32(tc->A, tc->B, C_ref, M, N, K, K, N, N);

  const size_t need = ope_pack_workspace_size(M, N, K);
  static uint8_t ws[OPE_PACK_WS_MAX_BYTES];

  printf("Running OPE Accelerator...\n");
  uint64_t t0 = rdcycle64();
  if (need <= sizeof(ws)) {
    ope_matmul_prepacked(tc->A, tc->B, C_ope,
                         M, N, K, K, N, N,
                         ws, sizeof(ws),
                        true);
  } else {
    printf("Falling Back to default OPE");
    ope_matmul(tc->A, tc->B, C_ope,
              M, N, K, K, N, N);
  }
  uint64_t t1 = rdcycle64();
  printf("Prepacked Execution time: %llu cycles\n", (unsigned long long)(t1 - t0));

  // Compare
  int ok = compare_results(C_ope, N, C_ref, N, M, N);
  printf("Result: %s\n", ok == 0 ? "PASS" : "FAIL");
  return ok;
}

void app_init(void) {}

void app_main(void) {
  printf("=== OPE MATMUL TESTS ===\n");
  printf("Debug settings: PRINT_INPUT_MATRICES=%d, PRINT_SUCCESS_MATRICES=%d\n",
          PRINT_INPUT_MATRICES, PRINT_SUCCESS_MATRICES);

  int total = 0, failed = 0;

  printf("\n--- ALIGNED TEST CASES ---\n");
  for (int i = 0; i < NUM_ALIGNED_CASES; ++i) {
    total++;
    if (run_unaligned_case(&OPE_CASES_ALIGNED[i]) != 0){
      failed++;
    }
    if (run_aligned_case(&OPE_CASES_ALIGNED[i]) != 0){
      failed++;
    }
    if (run_case_prepacked(&OPE_CASES_ALIGNED[i]) != 0){
      failed++;
    }
  }

  printf("\n--- UNALIGNED TEST CASES ---\n");
  for (int i = 0; i < NUM_UNALIGNED_CASES; ++i) {
    total++;
    if (run_unaligned_case(&OPE_CASES_UNALIGNED[i]) != 0){
      failed++;
    } 
    if (run_case_prepacked(&OPE_CASES_UNALIGNED[i]) != 0){
      failed++;
    }
  }

  printf("\n=== FINAL SUMMARY ===\n");
  printf("Total: %d, Passed: %d, Failed: %d\n",
          total, total - failed, failed);
  printf("OVERALL RESULT: %s\n", failed ? "FAIL" : "PASS");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
