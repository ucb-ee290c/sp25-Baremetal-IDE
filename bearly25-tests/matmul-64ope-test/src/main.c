/*
 * matmul-64ope-test/src/main.c
 *
 * 64×64 int8 matmul: rows 0-7 via OPE, rows 8-63 via RVV.
 * ACC stride capped at 32 (K=64 → two ACC(32) calls per tile).
 * 8 col-tiles × 7 RVV rows each = 56 RVV rows — perfect distribution.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"

#define DIM        64   /* must be multiple of 8 */
#define OPE_ROWS    8   /* rows handled by OPE (one 8-row tile) */
#define RVV_ROWS   (DIM - OPE_ROWS)   /* 56 */
#define RVV_START  OPE_ROWS

/* ── forward declarations (opervv.c) ── */
void pack_A_ope(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K);
void pack_B_ope(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void pack_B_rvv_bias(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void gemm_ope_rvv(const int8_t *A_packed, const int8_t *B_packed,
                  int32_t *C, int N, int K, int ldc,
                  const int8_t *A_rvv, const int8_t *B_rvv,
                  int num_rvv_rows, int32_t *C_rvv);
void gemm_rvv_all(const int8_t *A_rvv, const int8_t *B_rvv,
                  int32_t *C, int M, int N, int K, int ldc);

/* ── static buffers (BSS) ── */

static int8_t  A[DIM * DIM];
static int8_t  B[DIM * DIM];

/* Packed OPE inputs — 8-byte aligned:
 *   A_ope: 1 chunk × K × 8  = 64 × 8 = 512 bytes
 *   B_ope: 8 chunks × K × 8 = 8 × 64 × 8 = 4096 bytes */
static int8_t  A_ope[(OPE_ROWS / 8) * DIM * 8] __attribute__((aligned(64)));
static int8_t  B_ope[(DIM / 8) * DIM * 8]      __attribute__((aligned(64)));

/* B_rvv: (K+1) rows × N cols = 65 × 64 = 4160 bytes */
static int8_t  B_rvv[(DIM + 1) * DIM]           __attribute__((aligned(64)));

/* Outputs — page-align C_out for deterministic cache set mapping */
static int32_t C_out[DIM * DIM]      __attribute__((aligned(4096)));
static int32_t C_rvv_only[DIM * DIM] __attribute__((aligned(64)));
static int32_t C_ref[DIM * DIM];

/* ── helpers ── */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

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

void app_init(void) {}

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
    pack_A_ope(A, DIM, A_ope, 0, OPE_ROWS, DIM);

    printf("[pack] B_ope...\n");
    pack_B_ope(B, DIM, B_ope, DIM, DIM);

    printf("[pack] B_rvv (with bias row)...\n");
    pack_B_rvv_bias(B, DIM, B_rvv, DIM, DIM);

    /* ── OPE + interleaved RVV ── */
    memset(C_out, 0, sizeof(C_out));

    /* Prime all data into L1 before timing begins. */
    {
        volatile int8_t sink;
        volatile int32_t sink32;
        /* OPE output region (8 rows × DIM cols) */
        for (int r = 0; r < OPE_ROWS; r++)
            for (int c = 0; c < DIM; c += 16)
                sink32 = C_out[r * DIM + c];
        /* OPE ACC inputs */
        for (int off = 0; off < (OPE_ROWS / 8) * DIM * 8; off += 64)
            sink = A_ope[off];
        for (int off = 0; off < (DIM / 8) * DIM * 8; off += 64)
            sink = B_ope[off];
        /* RVV inputs */
        for (int off = 0; off < RVV_ROWS * DIM; off += 64)
            sink = (A + RVV_START * DIM)[off];
        for (int off = 0; off < (DIM + 1) * DIM; off += 64)
            sink = B_rvv[off];
        /* RVV output region */
        for (int r = 0; r < RVV_ROWS; r += 4)
            sink32 = C_out[(RVV_START + r) * DIM];
        asm volatile("fence r, rw" ::: "memory");
    }

    printf("[ope+rvv] gemm_ope_rvv (interleaved)...\n");
    uint64_t t0 = rdcycle64();
    gemm_ope_rvv(A_ope, B_ope, C_out, DIM, DIM, DIM,
                 A + RVV_START * DIM, B_rvv, RVV_ROWS,
                 C_out + RVV_START * DIM);
    uint64_t t1 = rdcycle64();
    printf("[ope+rvv] done, cycles=%llu\n", (unsigned long long)(t1 - t0));

    /* ── Pure-RVV baseline ── */
    memset(C_rvv_only, 0, sizeof(C_rvv_only));

    /* Prime RVV-only data before timing */
    {
        volatile int8_t sink;
        volatile int32_t sink32;
        for (int off = 0; off < DIM * DIM; off += 64)
            sink = A[off];
        for (int off = 0; off < (DIM + 1) * DIM; off += 64)
            sink = B_rvv[off];
        for (int r = 0; r < DIM; r += 4)
            sink32 = C_rvv_only[r * DIM];
        asm volatile("fence r, rw" ::: "memory");
    }

    printf("[rvv-only] gemm_rvv_all (%d rows)...\n", DIM);
    uint64_t t2 = rdcycle64();
    gemm_rvv_all(A, B_rvv, C_rvv_only, DIM, DIM, DIM, DIM);
    uint64_t t3 = rdcycle64();
    uint64_t rvv_cycles = t3 - t2;
    uint64_t ope_rvv_cycles = t1 - t0;
    printf("[rvv-only] done, cycles=%llu\n", (unsigned long long)rvv_cycles);

    /* ── Comparison ── */
    printf("\n=== Performance comparison (%dx%d) ===\n", DIM, DIM);
    printf("  OPE+RVV interleaved : %llu cycles\n", (unsigned long long)ope_rvv_cycles);
    printf("  RVV-only (%d rows)  : %llu cycles\n", DIM, (unsigned long long)rvv_cycles);
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
