#include <stdio.h>
#include <stdint.h>

//I ran this directly in main before, might need some editing now it is its own file

//EDit this to select
// 0 = CPU naive matmul
// 1 = OPE matmul_arb
#define RUN_OPE 0

// Max dimension for static buffers (must be >= largest n you test)
#define MAXN 128

// Pick your sizes here
typedef struct {
  const char *name;
  int n;
} Case;

static const Case cases[] = {
  { "sq_57", 57 },
  // { "sq_64", 64 },
  // { "sq_96", 96 },
};

//---------- RISC-V cycle counter ----------
static inline uint64_t rdcycle64(void) {
  uint64_t cycles;
  asm volatile ("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// ---------- Data buffers (static, no malloc) ---------- 
static int8_t  A[MAXN][MAXN];
static int8_t  B[MAXN][MAXN];
static int32_t C[MAXN][MAXN];

//---------- Deterministic fill ----------
static void fill_A_B(int n) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      A[i][k] = (int8_t)((i + 3*k) & 0x7);
    }
  }
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      B[k][j] = (int8_t)((k + 5*j) & 0x7);
    }
  }
}

static void zero_C(int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i][j] = 0;
    }
  }
}

//---------- Naive CPU matmul ---------- */
static void matmul_naive(int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < n; ++k) {
        sum += (int32_t)A[i][k] * (int32_t)B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

//Prevent dead-code elimination: tiny checksum. */
static int32_t checksum_C(int n) {
  int32_t s = 0;
  int step = (n > 16) ? (n / 16) : 1;
  for (int i = 0; i < n; i += step) {
    for (int j = 0; j < n; j += step) {
      s ^= C[i][j];
    }
  }
  return s;
}

//=========================================================
//   OPE OPTION (only compiled if RUN_OPE=1)

#if RUN_OPE
#include "bench_config.h"  // provides ope_mat8_t/ope_mat32_t + init/free
#include "bench_fill.h"    // if you want to reuse their fill; but we already fill above

static long matmul_ope_once(int n) {
  // Allocate OPE matrices
  ope_mat8_t  *Am = ope_mat8_init(n, n, OPE_MAT_ZERO);
  ope_mat8_t  *Bm = ope_mat8_init(n, n, OPE_MAT_ZERO);
  ope_mat32_t *Cm = ope_mat32_init(n, n, OPE_MAT_ZERO);
  if (!Am || !Bm || !Cm) {
    printf("ERROR: ope_mat*_init failed\n");
    return -1;
  }

  // Copy our static A/B into OPE matrices (row-major, with their stride)
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      Am->data[i * Am->colsU + k] = A[i][k];
    }
  }
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      Bm->data[k * Bm->colsU + j] = B[k][j];
    }
  }

  // Zero output
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Cm->data[i * Cm->colsU + j] = 0;
    }
  }

  // Run OPE
  long cycles_ope = ope_matmul_arb(Am, Bm, Cm);

  // Copy result back into our C so checksum works the same
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i][j] = Cm->data[i * Cm->colsU + j];
    }
  }

  ope_mat8_free(Am);
  ope_mat8_free(Bm);
  ope_mat32_free(Cm);

  return cycles_ope; // this is "OPE cycles" (not necessarily rdcycle total)
}
#endif

/* =========================
   PUBLIC ENTRY (no params)
   ========================= */

void run_matmul_bench(void) {
#if RUN_OPE
  printf("=== MATMUL BENCH (OPE matmul_arb) ===\n");
#else
  printf("=== MATMUL BENCH (CPU naive, NO OPE) ===\n");
#endif

  const int num_cases = (int)(sizeof(cases) / sizeof(cases[0]));

  for (int t = 0; t < num_cases; ++t) {
    int n = cases[t].n;

    if (n > MAXN) {
      printf("case=%s n=%d: ERROR (MAXN=%d too small)\n", cases[t].name, n, MAXN);
      continue;
    }

    fill_A_B(n);
    zero_C(n);

#if RUN_OPE
    // Time using rdcycle around the call (total CPU time for OPE invocation too)
    uint64_t t0 = rdcycle64();
    long ope_cycles = matmul_ope_once(n);
    uint64_t t1 = rdcycle64();
    uint64_t total_cycles = t1 - t0;

    int32_t cs = checksum_C(n);

    printf("case=%-8s  n=%3d  total_cycles=%llu  ope_cycles=%ld  checksum=%ld\n",
           cases[t].name, n,
           (unsigned long long)total_cycles,
           (long)ope_cycles,
           (long)cs);
#else
    uint64_t t0 = rdcycle64();
    matmul_naive(n);
    uint64_t t1 = rdcycle64();

    uint64_t cycles = t1 - t0;
    int32_t cs = checksum_C(n);

    printf("case=%-8s  n=%3d  cycles=%llu  checksum=%ld\n",
           cases[t].name, n,
           (unsigned long long)cycles,
           (long)cs);
#endif
  }

  printf("=== DONE ===\n");
}

void run_ope_square_sweep(void) {
  printf("=== SIMPLE SQUARE SWEEP (ARB, MINIMAL) ===\n");

  const OpeSizeCase cases[] = {
    { "sq_8",    8,   8,   8 },
    { "sq_8",    8,   8,   8 },
    { "sq_16",  16,  16,  16 },
    { "sq_16",  16,  16,  16 },
    { "sq_32",  32,  32,  32 },
    { "sq_32",  32,  32,  32 },
    { "sq_48",  48,  48,  48 },
    { "sq_48",  48,  48,  48 },
    { "sq_57",  57,  57,  57 },
    { "sq_57",  57,  57,  57 },
    { "sq_64",  64,  64,  64 },
    { "sq_64",  64,  64,  64 },
    { "sq_96",  96,  96,  96 },
    { "sq_96",  96,  96,  96 },
    { "sq_128", 128, 128, 128 },
    { "sq_128", 128, 128, 128 },
  };

  const int num_cases = (int)(sizeof(cases) / sizeof(cases[0]));
  for (int i = 0; i < num_cases; ++i) {
    const OpeSizeCase *cs = &cases[i];
    long cycles = bench_run_arb_once(cs);

    printf("square case %-8s  M=%d N=%d K=%d  cycles=%ld\n",
           cs->name, cs->M, cs->N, cs->K, cycles);
  }

  printf("=== SWEEP DONE ===\n");
}
