/*
 * matmul-28ope-test/src/main.c
 *
 * Performance test for gemm_i8_i32_28ope():
 * 28 RVV rows (4 x 7) + 8 OPE rows = 36 output rows per call.
 *
 * Runs the kernel once to warm the cache, then once more to measure.
 * Define CHECK_CORRECTNESS=1 at compile time to enable scalar reference
 * comparison (disabled by default for benchmarking).
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"

#ifndef CHECK_CORRECTNESS
#define CHECK_CORRECTNESS 0
#endif

#define ROWS 36   /* total output rows: 28 RVV + 8 OPE */
#define N    72   /* columns (must be multiple of 8) */
#define M    72   /* A^T leading dim (>= ROWS) */
#define K    72   /* inner dimension */

#define N_OPE_TILES (N / 8)

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

/* ---- OPE input packing ---- */

static void pack_ope_A(const int8_t *A_T, int8_t *A_ope)
{
    for (int k = 0; k < K; k++)
        for (int r = 0; r < 8; r++)
            A_ope[k * 8 + r] = A_T[k * M + 28 + r];
}

static void pack_ope_B(const int8_t *B, int8_t *B_ope)
{
    for (int j = 0; j < N_OPE_TILES; j++)
        for (int k = 0; k < K; k++)
            for (int c = 0; c < 8; c++)
                B_ope[j * K * 8 + k * 8 + c] = B[k * N + j * 8 + c];
}

/* ---- Correctness helpers (compiled out unless CHECK_CORRECTNESS=1) ---- */

#if CHECK_CORRECTNESS
static void ref_matmul(const int8_t *A_T, const int8_t *B, int32_t *C)
{
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A_T[k * M + i] * (int32_t)B[k * N + j];
            C[i * N + j] = acc;
        }
}

static int compare(const int32_t *got, const int32_t *exp)
{
    int errors = 0;
    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < N; j++)
            if (got[i * N + j] != exp[i * N + j]) {
                if (errors == 0)
                    printf("  First mismatch [%d][%d]: got %d, expected %d\n",
                           i, j, got[i * N + j], exp[i * N + j]);
                errors++;
            }
    if (errors)
        printf("  %d mismatches out of %d elements\n", errors, ROWS * N);
    return errors ? -1 : 0;
}
#endif

/* ---- External kernel (in opervv.c) ---- */

extern void gemm_i8_i32_28ope(
    size_t nc, size_t kc,
    const int8_t *a, size_t a_stride,
    const int8_t *w, size_t b_row_stride,
    int32_t *c, size_t cm_stride,
    const int8_t *a_ope, const int8_t *b_ope
);

void app_init(void) {}

void app_main(void) {
    printf("=== gemm_i8_i32_28ope  A_T[%dx%d] * B[%dx%d] => C[%dx%d] ===\n",
           K, M, K, N, ROWS, N);

    static int8_t  A_T    [K * M];
    static int8_t  B      [K * N];
    static int8_t  B_pack [(K+1) * N];
    static int8_t  A_ope  [K * 8] __attribute__((aligned(8)));
    static int8_t  B_ope  [N_OPE_TILES * K * 8] __attribute__((aligned(8)));
    static int32_t C_out  [ROWS * N];

    printf("[dbg] filling inputs...\n");
    /* Deterministic fill */
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++)
            A_T[k * M + i] = (int8_t)((k * 3 + i * 7 + 1) % 17 - 8);
        for (int j = 0; j < N; j++)
            B[k * N + j]   = (int8_t)((k * 5 + j * 2 + 3) % 13 - 6);
    }

    memset(B_pack, 0, N);
    memcpy(B_pack + N, B, K * N);
    printf("[dbg] packing OPE A...\n");
    pack_ope_A(A_T, A_ope);
    printf("[dbg] packing OPE B (%d tiles)...\n", N_OPE_TILES);
    pack_ope_B(B, B_ope);

#define RUN_KERNEL() \
    gemm_i8_i32_28ope( \
        N, K, A_T, M, B_pack, N, \
        C_out, N * sizeof(int32_t), \
        A_ope, B_ope)

    /* Warm-up run (fills caches) */
    printf("[dbg] warm-up run...\n");
    memset(C_out, 0, sizeof(C_out));
    RUN_KERNEL();
    printf("[dbg] warm-up done\n");

    /* Timed run */
    printf("[dbg] timed run...\n");
    memset(C_out, 0, sizeof(C_out));
    uint64_t t0 = rdcycle64();
    RUN_KERNEL();
    uint64_t t1 = rdcycle64();
    printf("[dbg] timed run done\n");

    printf("  Cycles (hot): %lu\n", (unsigned long)(t1 - t0));

#if CHECK_CORRECTNESS
    static int32_t C_ref[ROWS * N];
    ref_matmul(A_T, B, C_ref);
    int rc = compare(C_out, C_ref);
    printf("  Result: %s\n", rc == 0 ? "PASS" : "FAIL");
#endif
}

int main(void) {
    app_init();
    app_main();
    return 0;
}
