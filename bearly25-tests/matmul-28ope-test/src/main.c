/*
 * matmul-28ope-test/src/main.c
 *
 * Simple OPE matmul test using hal_ope API directly
 * (mimics ope-bmarks to verify OPE works in this test context).
 *
 * Once this works, we can layer in RVV + OPE fusion.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"
#include "hal_ope.h"

#define DIM 72   /* 72 = 9 * 8, multiple of 8 */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

/* Scalar reference: C = A * B (A is MxK row-major, B is KxN row-major) */
static void ref_matmul(const int8_t *A, int Aldc,
                       const int8_t *B, int Bldc,
                       int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i * Aldc + k] * (int32_t)B[k * Bldc + j];
            C[i * N + j] = acc;
        }
}

void app_init(void) {}

void app_main(void) {
    printf("=== OPE-only test  %dx%d matmul (hal_ope API) ===\n", DIM, DIM);

    /* Allocate matrices via hal_ope */
    ope_mat8_t  *A = ope_mat8_init(DIM, DIM, OPE_MAT_ZERO);
    ope_mat8_t  *B = ope_mat8_init(DIM, DIM, OPE_MAT_ZERO);
    ope_mat32_t *C = ope_mat32_init(DIM, DIM, OPE_MAT_ZERO);

    if (!A || !B || !C) {
        printf("ERROR: allocation failed\n");
        return;
    }

    printf("[dbg] rowsU=%d colsU=%d\n", A->rowsU, A->colsU);

    /* Deterministic fill */
    printf("[dbg] filling inputs...\n");
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            A->data[i * A->colsU + j] = (int8_t)((i * 3 + j * 7 + 1) % 17 - 8);
            B->data[i * B->colsU + j] = (int8_t)((i * 5 + j * 2 + 3) % 13 - 6);
        }

    /* Pre-allocate workspace (like ope-bmarks does) */
    printf("[dbg] init workspace...\n");
    ope_init_workspace(DIM, DIM, DIM);

    /* Warm-up run */
    printf("[dbg] warm-up run...\n");
    memset(C->data, 0, (size_t)C->rowsU * C->colsU * sizeof(int32_t));
    long cyc_warmup = ope_matmul_arb(A, B, C);
    printf("[dbg] warm-up done, cycles=%ld\n", cyc_warmup);

    /* Timed run */
    printf("[dbg] timed run...\n");
    memset(C->data, 0, (size_t)C->rowsU * C->colsU * sizeof(int32_t));
    uint64_t t0 = rdcycle64();
    long cyc_ope = ope_matmul_arb(A, B, C);
    uint64_t t1 = rdcycle64();
    printf("[dbg] timed run done\n");

    printf("  OPE cycles: %ld\n", cyc_ope);
    printf("  Total cycles: %lu\n", (unsigned long)(t1 - t0));

    /* Quick correctness check */
    printf("[dbg] reference matmul...\n");
    static int32_t C_ref[DIM * DIM];
    ref_matmul(A->data, A->colsU, B->data, B->colsU, C_ref, DIM, DIM, DIM);

    /* OPE with EXT_FLIP=1 outputs tiles transposed; unflip for comparison */
    int errors = 0;
    for (int tr = 0; tr < C->rowsU; tr += 8) {
        for (int tc = 0; tc < C->colsU; tc += 8) {
            for (int r = 0; r < 8 && (tr + r) < DIM; r++) {
                for (int c = 0; c < 8 && (tc + c) < DIM; c++) {
                    /* With EXT_FLIP, tile is transposed: row r, col c stored at [r][c]
                       but represents [c][r] of the logical tile */
                    int32_t got = C->data[(tr + r) * C->colsU + (tc + c)];
                    /* The transposed tile means got is actually result[tr+c][tc+r] */
                    int32_t exp = C_ref[(tr + c) * DIM + (tc + r)];
                    if (got != exp) {
                        if (errors == 0)
                            printf("  First mismatch tile(%d,%d) r=%d c=%d: got %d, exp %d\n",
                                   tr, tc, r, c, got, exp);
                        errors++;
                    }
                }
            }
        }
    }

    if (errors)
        printf("  %d mismatches\n", errors);
    else
        printf("  Result: PASS\n");

    ope_free_workspace();
    ope_mat8_free(A);
    ope_mat8_free(B);
    ope_mat32_free(C);
}

int main(void) {
    app_init();
    app_main();
    return 0;
}
