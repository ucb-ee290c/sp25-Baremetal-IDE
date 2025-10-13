/* =========================================================================
 * main.c — Simple tests for OPE matmul
 * ========================================================================= */
#include "main.h"
#include <data/inputs.h>

/* ---------- Helpers ---------- */
static inline size_t ru8(size_t x){ return (x + 7u) & ~7u; }

static inline uint64_t rdcycle64(void) {
  uint64_t x; asm volatile("rdcycle %0" : "=r"(x)); return x;
}

/* Scalar reference: C = A^T * B (A: MxK, B: KxN, C: MxN) */
static void ref_gemm_AT_i8i8_i32(const int8_t* A, const int8_t* B,
                                 int32_t* C, int M, int N, int K,
                                 int ldA, int ldB, int ldC) {
  for (int i=0;i<M;++i){
    for (int j=0;j<N;++j){
      int32_t acc=0;
      for (int k=0;k<K;++k){
        acc += (int32_t)A[k + i*ldA] * (int32_t)B[k*ldB + j];
      }
      C[i*ldC + j] = acc;
    }
  }
}

// Manual padding/transpose (no RVV)
static void transpose_pad_i8_to_AT_padded(const int8_t* A, size_t M, size_t K,
                                          int8_t* AT, size_t M8, size_t K8) {
  for (size_t i=0;i<K8*M8;++i) AT[i]=0;
  for (size_t c=0;c<K;++c){
    int8_t* out = AT + c*M8;
    for (size_t r=0;r<M;++r){
      out[r] = A[r*K + c];
    }
  }
}

// B (KxN) -> B8 (K8xN8), bottom/right padded
static void pad_i8_bottom_right(const int8_t* B, size_t K, size_t N,
                                int8_t* B8, size_t K8, size_t N8) {
  for (size_t i=0;i<K8*N8;++i) B8[i]=0;
  for (size_t r=0;r<K;++r){
    memcpy(B8 + r*N8, B + r*N, N);
  }
}

// C (MxN, i32) -> C8 (M8xN8)
static void pad_i32_bottom_right(const int32_t* C, size_t M, size_t N,
                                 int32_t* C8, size_t M8, size_t N8) {
  for (size_t i=0;i<M8*N8;++i) C8[i]=0;
  for (size_t r=0;r<M;++r){
    memcpy(C8 + r*N8, C + r*N, N*sizeof(int32_t));
  }
}

static int compare_C_top_left(const int32_t* C, int ldc,
                              const int32_t* Cref, int ldcref,
                              int M, int N) {
  for (int i=0;i<M;++i){
    for (int j=0;j<N;++j){
      if (C[i*ldc + j] != Cref[i*ldcref + j]) {
        printf("Mismatch at (%d,%d): got %d, exp %d\n",
               i, j, C[i*ldc+j], Cref[i*ldcref+j]);
        return -1;
      }
    }
  }
  return 0;
}

#define MAX_M8  16
#define MAX_N8  16
#define MAX_K8  32
static int8_t  AT8_buf[MAX_K8 * MAX_M8];
static int8_t  B8_buf [MAX_K8 * MAX_N8];
static int32_t C8_buf [MAX_M8 * MAX_N8];
static int32_t Cref_buf[8*8];

static int run_case(const OpeInputCase *tc, int load_existing) {
  const int M=tc->M, N=tc->N, K=tc->K;
  const size_t M8=ru8((size_t)M), N8=ru8((size_t)N), K8=ru8((size_t)K);

  if (M8>MAX_M8 || N8>MAX_N8 || K8>MAX_K8) {
    printf("Buffers too small for case %s (M8=%u N8=%u K8=%u)\n",
           tc->name, (unsigned)M8,(unsigned)N8,(unsigned)K8);
    return -1;
  }

  for (int i=0;i<M*N;++i) Cref_buf[i]=0;
  ref_gemm_AT_i8i8_i32(tc->A, tc->B, Cref_buf, M,N,K, K, N, N);

  // Preprocess for OPE
  transpose_pad_i8_to_AT_padded(tc->A, M, K, AT8_buf, M8, K8);
  pad_i8_bottom_right(tc->B, K, N, B8_buf, K8, N8);
  if (load_existing) {
    static int32_t C0[8*8];
    for (int i=0;i<M*N;++i) C0[i] = (i%13) - 6;
    pad_i32_bottom_right(C0, M, N, C8_buf, M8, N8);
  } else {
    for (size_t i=0;i<M8*N8;++i) C8_buf[i]=0;
  }

  uint64_t t0 = rdcycle64();
  ope_matmul_i8i8_i32_AT(AT8_buf, (int)M8, B8_buf, (int)N8, C8_buf, (int)N8,
                         (int)M8, (int)N8, (int)K8,
                         (bool)load_existing);
  uint64_t t1 = rdcycle64();

  int ok = compare_C_top_left(C8_buf, (int)N8, Cref_buf, N, M, N);
  printf("  %-18s (M=%d N=%d K=%d, load=%d)  cycles=%llu  -> %s\n",
         tc->name, M,N,K, load_existing,
         (unsigned long long)(t1 - t0), ok==0 ? "PASS" : "FAIL");
  return ok;
}

void app_init(void) {}

void app_main(void) {
  printf("=== OPE matmul tests (static inputs, no malloc) ===\n");

  int any_fail = 0;

  // 1) aligned cases
  for (int i=0;i<NUM_ALIGNED_CASES;++i) {
    any_fail |= (run_case(&OPE_CASES_ALIGNED[i], 0) != 0);
  }

  // 2) unaligned cases (manually padded by the runner)
  for (int i=0;i<NUM_UNALIGNED_CASES;++i) {
    any_fail |= (run_case(&OPE_CASES_UNALIGNED[i], 0) != 0);
    any_fail |= (run_case(&OPE_CASES_UNALIGNED[i], 1) != 0);
  }

  printf("\nRESULT: %s\n", any_fail ? "FAIL" : "PASS");
}

int main(void) { app_init(); app_main(); return 0; }
