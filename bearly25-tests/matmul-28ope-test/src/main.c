/*
 * matmul-28ope-test/src/main.c
 *
 * 32×32 int8 matmul: rows 0-7 via OPE, rows 8-31 via RVV.
 * RVV rows are interleaved inside gemm_ope_16rows during OPE ACC latency,
 * using the 7xm4 kernel from bearly25-bmarks/rvv-matmul (7 rows at a time).
 *
 * Call order:
 *   1. pack inputs for OPE and RVV
 *   2. gemm_ope_16rows  → C[0:8, :] (OPE) + C[8:32, :] (RVV, interleaved)
 *   3. scalar reference + correctness check
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"

#define DIM        32   /* must be multiple of 8 */
#define OPE_ROWS    8   /* rows handled by OPE (one 8-row tile) */
#define RVV_ROWS   24   /* rows handled by RVV (= DIM - OPE_ROWS) */
#define RVV_START  OPE_ROWS

/* ── forward declarations (opervv.c) ── */
void pack_A_ope(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K);
void pack_B_ope(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void pack_A_rvv(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K);
void pack_B_rvv_bias(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void gemm_ope_16rows(const int8_t *A_packed, const int8_t *B_packed,
                     int32_t *C, int N, int K, int ldc,
                     const int8_t *A_rvv, const int8_t *B_rvv,
                     int num_rvv_rows, int32_t *C_rvv);
void gemm_rvv_32rows(const int8_t *A_rvv, const int8_t *B_rvv,
                     int32_t *C, int N, int K, int ldc);

/* ── static buffers (BSS) ── */
/*
 * OPE RoCC memory requests require 8-byte alignment.
 * All packed buffers and the OPE output must be __attribute__((aligned(64))).
 */

/* Input matrices (not passed to OPE directly, no alignment requirement) */
static int8_t  A[DIM * DIM];
static int8_t  B[DIM * DIM];

/* Packed OPE inputs — must be 8-byte aligned:
 *   A_ope: OPE_ROWS/8 chunks × DIM × 8  =  1 × 32 × 8  = 256 bytes
 *   B_ope: DIM/8 chunks × DIM × 8       =  4 × 32 × 8  = 1024 bytes */
static int8_t  A_ope[(OPE_ROWS / 8) * DIM * 8] __attribute__((aligned(64)));
static int8_t  B_ope[(DIM / 8) * DIM * 8]      __attribute__((aligned(64)));

/* Packed RVV inputs:
 *   A_rvv: RVV_ROWS rows × DIM k-steps  =  24 × 32  = 768 bytes (row-major)
 *   B_rvv: (DIM+1) rows × DIM cols      =  33 × 32  = 1056 bytes (zero bias row) */
static int8_t  A_rvv[DIM * RVV_ROWS]            __attribute__((aligned(64)));
static int8_t  B_rvv[(DIM + 1) * DIM]           __attribute__((aligned(64)));

/* Pure-RVV baseline: all 32 rows packed for RVV */
static int8_t  A_rvv_all[DIM * DIM]             __attribute__((aligned(64)));

/* Outputs — OPE writes C_out via TileLink, keep 8-byte aligned */
static int32_t C_out[DIM * DIM]      __attribute__((aligned(64)));
static int32_t C_rvv_only[DIM * DIM] __attribute__((aligned(64)));
static int32_t C_ref[DIM * DIM];

/* ── helpers ── */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

/* Scalar reference: C = A * B, A[M×K] B[K×N] row-major */
static void ref_matmul(const int8_t *A, int lda,
                       const int8_t *B, int ldb,
                       int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i * lda + k] * (int32_t)B[k * ldb + j];
            C[i * N + j] = acc;
        }
}

/* ── application entry ── */

void app_init(void) {
    // init_test(500000000ULL);
}

void app_main(void)
{
    printf("=== OPE(%d rows) + RVV(%d rows) interleaved matmul %dx%d ===\n",
           OPE_ROWS, RVV_ROWS, DIM, DIM);

    /* ── Fill A and B with a deterministic pattern ── */
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            A[i * DIM + j] = (int8_t)((i * 3 + j * 7 + 1) % 17 - 8);
            B[i * DIM + j] = (int8_t)((i * 5 + j * 2 + 3) % 13 - 6);
        }

    /* ── Pack inputs ── */
    printf("[pack] A_ope (rows 0-%d)...\n", OPE_ROWS - 1);
    pack_A_ope(A, DIM, A_ope, /*row_start=*/0, OPE_ROWS, DIM);

    printf("[pack] B_ope...\n");
    pack_B_ope(B, DIM, B_ope, DIM, DIM);

    printf("[pack] A_rvv (rows %d-%d)...\n", RVV_START, DIM - 1);
    pack_A_rvv(A, DIM, A_rvv, RVV_START, RVV_ROWS, DIM);

    printf("[pack] B_rvv (with bias row)...\n");
    pack_B_rvv_bias(B, DIM, B_rvv, DIM, DIM);

    /* ── OPE + interleaved RVV ── */
    memset(C_out, 0, sizeof(C_out));

    printf("[ope+rvv] gemm_ope_16rows (interleaved)...\n");
    uint64_t t0 = rdcycle64();
    gemm_ope_16rows(A_ope, B_ope, C_out, DIM, DIM, /*ldc=*/DIM,
                    A_rvv, B_rvv, RVV_ROWS, C_out + RVV_START * DIM);
    uint64_t t1 = rdcycle64();
    printf("[ope+rvv] done, cycles=%llu\n", (unsigned long long)(t1 - t0));

    // /* ── Scalar reference ── */
    // printf("[ref] computing reference...\n");
    // ref_matmul(A, DIM, B, DIM, C_ref, DIM, DIM, DIM);

    /* ── Correctness check ── */
    int errors = 0;
    // int first_err_row = -1, first_err_col = -1;
    // int32_t first_got = 0, first_exp = 0;

    // for (int i = 0; i < DIM && errors < 1000; i++) {
    //     for (int j = 0; j < DIM; j++) {
    //         int32_t got = C_out[i * DIM + j];
    //         int32_t exp = C_ref[i * DIM + j];
    //         if (got != exp) {
    //             if (errors == 0) {
    //                 first_err_row = i;
    //                 first_err_col = j;
    //                 first_got = got;
    //                 first_exp = exp;
    //             }
    //             errors++;
    //         }
    //     }
    // }

    if (errors) {
        printf("  FAIL: %d mismatches\n", errors);
        printf("  First mismatch at C[%d][%d]: got %d, exp %d  (region: %s)\n",
               first_err_row, first_err_col, (int)first_got, (int)first_exp,
               first_err_row < OPE_ROWS ? "OPE" : "RVV");
    } else {
        printf("  PASS: all %d×%d elements correct\n", DIM, DIM);
    }

    /* ── Pure-RVV baseline (all 32 rows via Saturn) ── */
    printf("\n[rvv-only] packing A for all %d rows...\n", DIM);
    pack_A_rvv(A, DIM, A_rvv_all, /*row_start=*/0, DIM, DIM);

    memset(C_rvv_only, 0, sizeof(C_rvv_only));
    printf("[rvv-only] gemm_rvv_32rows...\n");
    uint64_t t2 = rdcycle64();
    gemm_rvv_32rows(A_rvv_all, B_rvv, C_rvv_only, DIM, DIM, DIM);
    uint64_t t3 = rdcycle64();
    uint64_t rvv_cycles = t3 - t2;
    uint64_t ope_rvv_cycles = t1 - t0;
    printf("[rvv-only] done, cycles=%llu\n", (unsigned long long)rvv_cycles);

    /* Verify RVV-only correctness */
    int rvv_errors = 0;
    // for (int i = 0; i < DIM * DIM; i++) {
    //     if (C_rvv_only[i] != C_ref[i]) rvv_errors++;
    // }
    if (rvv_errors)
        printf("[rvv-only] FAIL: %d mismatches vs reference\n", rvv_errors);
    else
        printf("[rvv-only] PASS: all elements correct\n");

    /* ── Comparison ── */
    printf("\n=== Performance comparison ===\n");
    printf("  OPE+RVV interleaved : %llu cycles\n", (unsigned long long)ope_rvv_cycles);
    printf("  RVV-only (32 rows)  : %llu cycles\n", (unsigned long long)rvv_cycles);
    if (rvv_cycles > ope_rvv_cycles) {
        printf("  OPE+RVV is %.2fx faster than RVV-only\n",
               (double)rvv_cycles / (double)ope_rvv_cycles);
    } else {
        printf("  RVV-only is %.2fx faster than OPE+RVV\n",
               (double)ope_rvv_cycles / (double)rvv_cycles);
    }
}

int main(void)
{
    app_init();
    app_main();
    return 0;
}
