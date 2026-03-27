/*
 * matmul-28ope-test/src/main.c
 *
 * Functional correctness test for gemm_i8_i32_28ope():
 * 28 RVV rows (4 x 7) + 8 OPE rows = 36 output rows per call.
 *
 * Problem layout
 * --------------
 *   A_T    : [K x M] int8 row-major  (A stored transposed; a_stride = M)
 *   B      : [K x N] int8 row-major
 *   B_pack : [(K+1) x N] int8 — B with a zero bias row prepended (for RVV)
 *   A_ope  : [K x 8] int8 — rows 28-35 of A^T, repacked column-major in
 *            8-element groups (matching ope_remap_matrix_A format)
 *   B_ope  : [(N/8) x K x 8] int8 — B repacked into 8-column tiles
 *            (matching ope_remap_matrix_B format)
 *   C      : [36 x N] int32 row-major output
 *
 * The kernel computes C[i][j] = sum_k A_T[k*M + i] * B[k*N + j]
 * for rows i = 0..35 (rows 0-27 via RVV, rows 28-35 via OPE).
 *
 * OPE inputs are pre-packed here before calling the kernel so the OPE
 * receives large OP_ACC_L batches (L up to 32) for full throughput.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"

#define ROWS 36   /* total output rows: 28 RVV + 8 OPE */
#define N    64   /* columns (must be multiple of 8) */
#define M    64   /* A^T leading dim (>= ROWS) */
#define K    64   /* inner dimension */

#define N_OPE_TILES (N / 8)

/* -------------------------------------------------------------------------
 * Cycle counter
 * ------------------------------------------------------------------------- */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

/* -------------------------------------------------------------------------
 * Scalar reference: C[i][j] = sum_k A_T[k*M+i] * B[k*N+j]  for i < ROWS
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
 * OPE input packing (done once before the kernel call)
 * ------------------------------------------------------------------------- */

/*
 * Pack rows 28-35 of A^T into OPE column-major-within-8 format.
 * Output: A_ope[k*8 + r] = A_T[k*M + 28 + r]  for r=0..7, k=0..K-1
 * Matches the layout produced by ope_remap_matrix_A for a single 8-row chunk.
 */
static void pack_ope_A(const int8_t *A_T, int8_t *A_ope)
{
    for (int k = 0; k < K; k++) {
        for (int r = 0; r < 8; r++) {
            A_ope[k * 8 + r] = A_T[k * M + 28 + r];
        }
    }
}

/*
 * Pack B into OPE tile format: 8-column tiles, each contiguous across K.
 * Output: B_ope[j*K*8 + k*8 + c] = B[k*N + j*8 + c]
 * Matches the layout produced by ope_remap_matrix_B.
 */
static void pack_ope_B(const int8_t *B, int8_t *B_ope)
{
    for (int j = 0; j < N_OPE_TILES; j++) {
        for (int k = 0; k < K; k++) {
            for (int c = 0; c < 8; c++) {
                B_ope[j * K * 8 + k * 8 + c] = B[k * N + j * 8 + c];
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Print abbreviated matrix (first 8 rows x 8 cols, plus OPE rows 28-35)
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
 * Compare C_out and C_ref; print first mismatch and error count.
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
 * External kernel (in opervv.c)
 * ------------------------------------------------------------------------- */

extern void gemm_i8_i32_28ope(
    size_t nc,
    size_t kc,
    const int8_t *a,
    size_t a_stride,
    const int8_t *w,
    size_t b_row_stride,
    int32_t *c,
    size_t cm_stride,
    const int8_t *a_ope,
    const int8_t *b_ope
);

/* -------------------------------------------------------------------------
 * Hardware UART init
 * ------------------------------------------------------------------------- */

void app_init(void) {
}

/* -------------------------------------------------------------------------
 * Main test
 * ------------------------------------------------------------------------- */

void app_main(void) {
    printf("=== gemm_i8_i32_28ope TEST  A_T[%dx%d] * B[%dx%d] => C[%dx%d] ===\n",
           K, M, K, N, ROWS, N);
    printf("    (rows 0-27 via RVV, rows 28-35 via OPE, separate pre-packed inputs)\n");

    static int8_t  A_T    [K * M];           /* [K x M] row-major              */
    static int8_t  B      [K * N];           /* [K x N] actual weights          */
    static int8_t  B_pack [(K+1) * N];       /* [(K+1) x N]: zero row + B rows  */
    static int8_t  A_ope  [K * 8] __attribute__((aligned(8)));
    static int8_t  B_ope  [N_OPE_TILES * K * 8] __attribute__((aligned(8)));
    static int32_t C_out  [ROWS * N];
    static int32_t C_ref  [ROWS * N];

    /* Deterministic fill */
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++)
            A_T[k * M + i] = (int8_t)((k * 3 + i * 7 + 1) % 17 - 8);
        for (int j = 0; j < N; j++)
            B[k * N + j]   = (int8_t)((k * 5 + j * 2 + 3) % 13 - 6);
    }

    /* B_pack for RVV: row 0 = zero bias, rows 1..K = B */
    memset(B_pack, 0, N);
    memcpy(B_pack + N, B, K * N);

    /* Pre-pack OPE inputs (done before kernel, no impact on timed region) */
    pack_ope_A(A_T, A_ope);
    pack_ope_B(B, B_ope);

    /* CPU reference */
    ref_matmul(A_T, B, C_ref);

    /* Kernel under test */
    memset(C_out, 0, sizeof(C_out));
    uint64_t t0 = rdcycle64();
    gemm_i8_i32_28ope(
        N,                          /* nc */
        K,                          /* kc */
        A_T,                        /* A^T base (RVV rows 0-27) */
        M,                          /* a_stride = M */
        B_pack,                     /* w = B_pack (RVV) */
        N,                          /* b_row_stride = N */
        C_out,                      /* C output */
        N * sizeof(int32_t),        /* cm_stride (bytes) */
        A_ope,                      /* pre-packed OPE A */
        B_ope                       /* pre-packed OPE B */
    );
    uint64_t t1 = rdcycle64();

    print_matrix_i32("Kernel output", C_out, ROWS, N, N);
    print_matrix_i32("Reference    ", C_ref, ROWS, N, N);

    int rc = compare(C_out, C_ref);

    printf("\n  Cycles: %lu\n", (unsigned long)(t1 - t0));
    printf("  Result: %s\n", rc == 0 ? "PASS" : "FAIL");
}

int main(void) {
    app_init();
    app_main();
    return 0;
}
