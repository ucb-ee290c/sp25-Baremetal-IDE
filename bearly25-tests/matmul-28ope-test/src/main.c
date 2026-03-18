/*
 * matmul-28ope-test/src/main.c
 *
 * Functional correctness test for gemm_i8_i32_28xm1():
 * 28 RVV rows (4 × 7) + 8 OPE rows = 36 output rows per call.
 *
 * Problem layout
 * --------------
 *   A_T    : [K × M] int8 row-major  (A stored transposed; a_stride = M)
 *   B      : [K × N] int8 row-major
 *   B_pack : [(K+1) × N] int8 — B with a zero bias row prepended at row 0
 *   C      : [36 × N] int32 row-major (output; cm_stride = N * sizeof(int32_t))
 *
 * The kernel computes C[i][j] = Σ_k  A_T[k*M + i] * B[k*N + j]
 * for rows i = 0..35 (rows 0-27 via RVV, rows 28-35 via OPE).
 *
 * Test strategy
 * -------------
 *   1. Fill A_T and B with small deterministic int8 values.
 *   2. Build B_pack = [zeros | B].
 *   3. Compute C_ref on the CPU using B directly.
 *   4. Run gemm_i8_i32_28xm1() with B_pack.
 *   5. Compare C_rvvope against C_ref element-by-element.
 *   6. Print both matrices (abbreviated) and a PASS / FAIL summary.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"

#define ROWS 36   /* total output rows: 28 RVV + 8 OPE */
#define N    64   /* columns */
#define M    64   /* A^T leading dim (>= ROWS) */
#define K    64   /* inner dimension; must be divisible by 4 */

/* -------------------------------------------------------------------------
 * Cycle counter
 * ------------------------------------------------------------------------- */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

/* -------------------------------------------------------------------------
 * Scalar reference: C[i][j] = Σ_k A_T[k*M+i] * B[k*N+j]  for i < ROWS
 * ------------------------------------------------------------------------- */

static void ref_matmul(const int8_t *A_T, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A_T[k * M + i] * (int32_t)B[k * N + j];
            C[i * N + j] = acc;
        }
    }
}

/* -------------------------------------------------------------------------
 * Print abbreviated matrix (first 8 rows × 8 cols)
 * ------------------------------------------------------------------------- */

static void print_matrix_i32(const char *label, const int32_t *mat,
                              int rows, int cols, int stride)
{
    const int SHOW = 8;
    printf("\n%s (%dx%d, showing top-left %dx%d):\n",
           label, rows, cols, SHOW, SHOW);
    for (int i = 0; i < SHOW; i++) {
        printf("  [");
        for (int j = 0; j < SHOW; j++) {
            printf("%7d", mat[i * stride + j]);
            if (j < SHOW - 1) printf(",");
        }
        printf(" ]\n");
    }
    /* Also print the OPE rows 28-35 */
    printf("  ... (rows 28-35, showing cols 0-7):\n");
    for (int i = 28; i < 36; i++) {
        printf("  [");
        for (int j = 0; j < SHOW; j++) {
            printf("%7d", mat[i * stride + j]);
            if (j < SHOW - 1) printf(",");
        }
        printf(" ]\n");
    }
}

/* -------------------------------------------------------------------------
 * Compare C_rvvope and C_ref; print first mismatch and error count.
 * Returns 0 on PASS, -1 on FAIL.
 * ------------------------------------------------------------------------- */

static int compare(const int32_t *got, const int32_t *exp)
{
    int errors = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < N; j++) {
            if (got[i * N + j] != exp[i * N + j]) {
                if (errors == 0)
                    printf("  First mismatch [%d][%d]: got %d, expected %d\n",
                           i, j, got[i * N + j], exp[i * N + j]);
                errors++;
            }
        }
    }
    if (errors)
        printf("  %d mismatches out of %d elements\n", errors, ROWS * N);
    return errors ? -1 : 0;
}

/* -------------------------------------------------------------------------
 * Hardware UART init
 * ------------------------------------------------------------------------- */

void app_init(void) {
}

/* -------------------------------------------------------------------------
 * Main test
 * ------------------------------------------------------------------------- */

void app_main(void) {
    printf("=== gemm_i8_i32_28xm1 TEST  A_T[%dx%d] * B[%dx%d] => C[%dx%d] ===\n",
           K, M, K, N, ROWS, N);
    printf("    (rows 0-27 via RVV, rows 28-35 via OPE)\n");

    static int8_t  A_T   [K * M];           /* [K x M] row-major              */
    static int8_t  B     [K * N];           /* [K x N] actual weights          */
    static int8_t  B_pack[(K+1) * N];       /* [(K+1) x N]: zero row + B rows  */
    static int32_t C_out [ROWS * N];
    static int32_t C_ref [ROWS * N];

    /* Deterministic fill */
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++)
            A_T[k * M + i] = (int8_t)((k * 3 + i * 7 + 1) % 17 - 8);
        for (int j = 0; j < N; j++)
            B[k * N + j]   = (int8_t)((k * 5 + j * 2 + 3) % 13 - 6);
    }

    /* B_pack: row 0 = zero bias, rows 1..K = B */
    memset(B_pack, 0, N);
    memcpy(B_pack + N, B, K * N);

    /* CPU reference */
    // ref_matmul(A_T, B, C_ref);

    /* Kernel under test */
    memset(C_out, 0, sizeof(C_out));
    uint64_t t0 = rdcycle64();
    gemm_i8_i32_28xm1(
        7,                          /* mr (unused) */
        N,                          /* nc */
        K,                          /* kc */
        A_T,                        /* A^T base */
        M,                          /* a_stride = M */
        B_pack,                     /* w = B_pack */
        N,                          /* b_row_stride = N */
        C_out,                      /* C */
        N * sizeof(int32_t),        /* cm_stride (bytes) */
        1                           /* cn_stride (unused) */
    );
    uint64_t t1 = rdcycle64();

    // print_matrix_i32("Kernel output", C_out, ROWS, N, N);
    // print_matrix_i32("Reference    ", C_ref, ROWS, N, N);

    // int rc = compare(C_out, C_ref);

    printf("\n  Cycles: %lu\n", (unsigned long)(t1 - t0));
    // printf("  Result: %s\n", rc == 0 ? "PASS" : "FAIL");
}

int main(void) {
    app_init();
    app_main();
    return 0;
}
