/* =========================================================================
 * main.c — Simple tests for OPE matmul
 * ========================================================================= */
#include "main.h"


static inline size_t round_up8(size_t x) { return (x + 7u) & ~7u; }

static inline uint64_t rdcycle64(void) {
  uint64_t x; asm volatile("rdcycle %0" : "=r"(x)); return x;
}

static void* xaligned_alloc(size_t alignment, size_t bytes) {
  void* p = malloc(bytes + alignment);
  if (!p) return NULL;
  uintptr_t raw = (uintptr_t)p;
  uintptr_t aligned = (raw + alignment - 1) & ~(alignment - 1);
  return (void*)aligned;
}

static void fill_rand_i8(int8_t* buf, size_t n, uint32_t* seed) {
  uint32_t s = *seed;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    buf[i] = (int8_t)((int32_t)s >> 24);
  }
  *seed = s;
}
static void fill_rand_i32(int32_t* buf, size_t n, uint32_t* seed) {
  uint32_t s = *seed;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    buf[i] = (int32_t)(s & 0x7FF) - 1024;
  }
  *seed = s;
}

/* ---------------------------
 * Scalar reference: C = A^T * B
 *  A: MxK (row-major)
 *  B: KxN (row-major)
 *  C: MxN (row-major, int32)
 * --------------------------- */
static void ref_gemm_AT_i8i8_i32(const int8_t* A, const int8_t* B,
                                 int32_t* C,
                                 int M, int N, int K,
                                 int ldA, int ldB, int ldC)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        int8_t a = A[k + i * ldA];
        int8_t b = B[k * ldB + j];
        acc += (int32_t)a * (int32_t)b;
      }
      C[i * ldC + j] = acc;
    }
  }
}


// A (MxK row-major) -> AT_padded (K8 x M8 row-major), transposed + bottom/right padded
static void transpose_pad_i8_to_AT_padded(const int8_t* A, size_t M, size_t K,
                                          int8_t* AT, size_t M8, size_t K8)
{
  memset(AT, 0, K8 * M8 * sizeof(int8_t));

  for (size_t c = 0; c < K; ++c) {
    int8_t* out_row = AT + c * M8;
    for (size_t r = 0; r < M; ++r) {
      out_row[r] = A[r * K + c];
    }
  }
}

// B (KxN row-major) -> B_padded (K8 x N8 row-major), bottom/right padded
static void pad_i8_bottom_right(const int8_t* B, size_t K, size_t N,
                                int8_t* B8, size_t K8, size_t N8)
{
  memset(B8, 0, K8 * N8 * sizeof(int8_t));
  for (size_t r = 0; r < K; ++r) {
    memcpy(B8 + r * N8, B + r * N, N * sizeof(int8_t));
  }
}

// C (MxN row-major int32) -> C_padded (M8 x N8), bottom/right padded
static void pad_i32_bottom_right(const int32_t* C, size_t M, size_t N,
                                 int32_t* C8, size_t M8, size_t N8)
{
  memset(C8, 0, M8 * N8 * sizeof(int32_t));
  for (size_t r = 0; r < M; ++r) {
    memcpy(C8 + r * N8, C + r * N, N * sizeof(int32_t));
  }
}

static int compare_C_top_left(const int32_t* C, int ldc,
                              const int32_t* Cref, int ldcref,
                              int M, int N)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t got = C[i*ldc + j];
      int32_t exp = Cref[i*ldcref + j];
      if (got != exp) {
        printf("Mismatch at (%d,%d): got %d, exp %d\n", i, j, (int)got, (int)exp);
        return -1;
      }
    }
  }
  return 0;
}


static int run_one_case(int M, int N, int K, int load_existing, uint32_t seed)
{
  printf("Case M=%d N=%d K=%d (load_existing=%d)\n", M, N, K, load_existing);

  // Allocate unpadded A,B,C0
  int8_t*  A  = (int8_t*) xaligned_alloc(64, (size_t)M * K);
  int8_t*  B  = (int8_t*) xaligned_alloc(64, (size_t)K * N);
  int32_t* C0 = (int32_t*)xaligned_alloc(64, (size_t)M * N);
  if (!A || !B || !C0) { printf("alloc failed\n"); return -1; }

  fill_rand_i8(A,  (size_t)M * K, &seed);
  fill_rand_i8(B,  (size_t)K * N, &seed);
  fill_rand_i32(C0, (size_t)M * N, &seed);

  // Reference result on original sizes
  int32_t* Cref = (int32_t*)xaligned_alloc(64, (size_t)M * N * sizeof(int32_t));
  if (!Cref) { printf("alloc failed\n"); return -1; }
  memset(Cref, 0, (size_t)M * N * sizeof(int32_t));
  ref_gemm_AT_i8i8_i32(A, B, Cref, M, N, K, K, N, N);

  // Padded sizes and buffers (for OPE)
  size_t M8 = round_up8((size_t)M);
  size_t N8 = round_up8((size_t)N);
  size_t K8 = round_up8((size_t)K);

  int8_t*  AT8 = (int8_t*) xaligned_alloc(64, K8 * M8);
  int8_t*  B8  = (int8_t*) xaligned_alloc(64, K8 * N8);
  int32_t* C8  = (int32_t*)xaligned_alloc(64, M8 * N8 * sizeof(int32_t));
  if (!AT8 || !B8 || !C8) { printf("alloc failed\n"); return -1; }

  // Manual padding/transforms
  transpose_pad_i8_to_AT_padded(A, (size_t)M, (size_t)K, AT8, M8, K8);
  pad_i8_bottom_right(B, (size_t)K, (size_t)N, B8, K8, N8);
  if (load_existing) {
    pad_i32_bottom_right(C0, (size_t)M, (size_t)N, C8, M8, N8);
  } else {
    memset(C8, 0, M8 * N8 * sizeof(int32_t));
  }

  // Call the OPE kernel (expects multiples of 8)
  uint64_t t0 = rdcycle64();
  ope_matmul_i8i8_i32_AT(AT8, (int)M8, B8, (int)N8, C8, (int)N8,
                         (int)M8, (int)N8, (int)K8,
                         (bool)load_existing);
  uint64_t t1 = rdcycle64();

  // Validate top-left region against reference
  int ok = compare_C_top_left(C8, (int)N8, Cref, N, M, N);
  printf("Cycles: %llu -> %s\n\n",
         (unsigned long long)(t1 - t0),
         ok == 0 ? "PASS" : "FAIL");

  free(A); free(B); free(C0);
  free(AT8); free(B8); free(C8); free(Cref);
  return ok;
}


typedef struct { int M,N,K; } dims_t;

static const dims_t aligned_cases[] = {
  { 8, 8, 8 },
  { 16, 8, 24 },
  { 32, 24, 16 },
  { 64, 64, 64 },
};

static const dims_t unaligned_cases[] = {
  { 7, 5, 9 },
  { 13, 11, 29 },
  { 31, 17, 63 },
  { 63, 63, 63 },
};


void app_init(void) {
  // nothing to init here
}

void app_main(void) {
  printf("=== OPE matmul tests (no RVV) ===\n\n");

  int any_fail = 0;
  uint32_t seed = 12345;

  printf("-- Aligned cases --\n");
  for (size_t i = 0; i < sizeof(aligned_cases)/sizeof(aligned_cases[0]); ++i) {
    dims_t d = aligned_cases[i];
    any_fail |= (run_one_case(d.M, d.N, d.K, 0, seed + (uint32_t)i*7) != 0);
  }

  printf("-- Unaligned cases (manual pad) --\n");
  for (size_t i = 0; i < sizeof(unaligned_cases)/sizeof(unaligned_cases[0]); ++i) {
    dims_t d = unaligned_cases[i];
    // Try both zero C, and accumulating into an existing C
    any_fail |= (run_one_case(d.M, d.N, d.K, 0, seed + 111 + (uint32_t)i*5) != 0);
    any_fail |= (run_one_case(d.M, d.N, d.K, 1, seed + 333 + (uint32_t)i*5) != 0);
  }

  printf("\nRESULT: %s\n", any_fail ? "FAIL" : "PASS");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
