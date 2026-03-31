/*
 * matmul-64ope-test/src/main.c
 *
 * 64×64 int8 matmul via 32×32 blocking.
 *
 * C[i,j] = Σ_k A[i,k] × B[k,j]   (i,j,k ∈ {0,1}, each block 32×32)
 * = 8 sub-multiplies, each using the proven 32×32 OPE+RVV kernel.
 *
 * All OPE parameters stay at 32: K=32, N=32, ldc=32, EXT stride=32.
 * Partial products are accumulated outside the timed region.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"

#define DIM        64
#define BLK        32   /* sub-block size — matches proven 32×32 kernel */
#define NBLK       (DIM / BLK)  /* 2 */
#define OPE_ROWS    8
#define RVV_ROWS   (BLK - OPE_ROWS)  /* 24 */

/* ── forward declarations (opervv.c) ── */
void pack_A_ope(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K);
void pack_B_ope(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void pack_A_rvv(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K);
void pack_B_rvv_bias(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void prime_32x32(int32_t *C, int ldc);
void gemm_ope_rvv_32x32(const int8_t *A_packed, const int8_t *B_packed,
                         int32_t *C,
                         const int8_t *A_rvv, const int8_t *B_rvv,
                         int num_rvv_rows);
void gemm_rvv_all(const int8_t *A, const int8_t *B_rvv,
                  int32_t *C, int M, int N, int K, int ldc);

/* ── static buffers ── */

static int8_t  A[DIM * DIM];
static int8_t  B[DIM * DIM];

/*
 * Pre-packed sub-block data: 4 A-blocks × (bi,bk), 4 B-blocks × (bk,bj).
 * Index: blk_idx = row_blk * NBLK + col_blk.
 */
static int8_t  all_A_ope[4][(OPE_ROWS / 8) * BLK * 8] __attribute__((aligned(64)));
static int8_t  all_B_ope[4][(BLK / 8) * BLK * 8]      __attribute__((aligned(64)));
static int8_t  all_A_rvv[4][RVV_ROWS * BLK]            __attribute__((aligned(64)));
static int8_t  all_B_rvv[4][(BLK + 1) * BLK]           __attribute__((aligned(64)));

/*
 * 8 pre-populated work buffer sets, one per sub-multiply.
 * Page-aligned (4096) so every set has identical cache set mapping:
 *   A_ope at page offset 0, B_ope at 256, A_rvv at 1280, B_rvv at 2336.
 * Each set occupies exactly one page → max 3 lines per cache set
 * (2 from sub_out + 1 from work), safe with 4-way associativity.
 */
typedef struct {
    int8_t A_ope[(OPE_ROWS / 8) * BLK * 8];  /* 256 B,  offset 0    */
    int8_t B_ope[(BLK / 8) * BLK * 8];       /* 1024 B, offset 256  */
    int8_t A_rvv[RVV_ROWS * BLK];            /* 768 B,  offset 1280 */
    int8_t B_rvv[(BLK + 1) * BLK];           /* 1056 B, offset 2336 */
    int8_t _pad[4096 - 256 - 1024 - 768 - 1056]; /* pad to 4096     */
} work_set_t;

static work_set_t work_sets[NBLK * NBLK * NBLK] __attribute__((aligned(4096)));

/* Sub-kernel outputs: one per sub-multiply (8 total = 2×2×2).
 * Page-align for deterministic OPE cache set mapping.
 * Index: sub_out[bi][bj][bk] */
static int32_t sub_out[NBLK][NBLK][NBLK][BLK * BLK] __attribute__((aligned(4096)));

/* Final output */
static int32_t C_out[DIM * DIM];
static int32_t C_rvv_only[DIM * DIM];

/* Full-size B_rvv for pure-RVV baseline (K=64, N=64) */
static int8_t  B_rvv_full[(DIM + 1) * DIM] __attribute__((aligned(64)));

/* ── helpers ── */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

/* Scatter-accumulate: C_out[bi*32+r][bj*32+c] += src[r*32+c] */
static void accum_block(const int32_t *src, int32_t *dst,
                        int bi, int bj, int ldc)
{
    for (int r = 0; r < BLK; r++)
        for (int c = 0; c < BLK; c++)
            dst[(bi * BLK + r) * ldc + bj * BLK + c] += src[r * BLK + c];
}

/* ── application entry ── */

void app_init(void) {}

void app_main(void)
{
    printf("=== 64x64 matmul via 32x32 blocking (OPE+RVV) ===\n");

    /* ── Fill A and B ── */
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            A[i * DIM + j] = (int8_t)((i * 3 + j * 7 + 1) % 17 - 8);
            B[i * DIM + j] = (int8_t)((i * 5 + j * 2 + 3) % 13 - 6);
        }

    /* ── Pre-pack all 4 A-blocks and 4 B-blocks ── */
    printf("[pack] pre-packing all sub-blocks...\n");
    for (int bi = 0; bi < NBLK; bi++) {
        for (int bk = 0; bk < NBLK; bk++) {
            int a_idx = bi * NBLK + bk;
            const int8_t *A_sub = A + bk * BLK;
            pack_A_ope(A_sub, DIM, all_A_ope[a_idx], bi * BLK, OPE_ROWS, BLK);
            pack_A_rvv(A_sub, DIM, all_A_rvv[a_idx], bi * BLK + OPE_ROWS, RVV_ROWS, BLK);
        }
    }
    for (int bk = 0; bk < NBLK; bk++) {
        for (int bj = 0; bj < NBLK; bj++) {
            int b_idx = bk * NBLK + bj;
            const int8_t *B_sub = B + bk * BLK * DIM + bj * BLK;
            pack_B_ope(B_sub, DIM, all_B_ope[b_idx], BLK, BLK);
            pack_B_rvv_bias(B_sub, DIM, all_B_rvv[b_idx], BLK, BLK);
        }
    }

    /* ── Full-size B_rvv for pure-RVV baseline ── */
    pack_B_rvv_bias(B, DIM, B_rvv_full, DIM, DIM);

    /* ── Pre-populate work sets (before timing) ── */
    for (int bi = 0; bi < NBLK; bi++)
        for (int bj = 0; bj < NBLK; bj++)
            for (int bk = 0; bk < NBLK; bk++) {
                int idx = bi * NBLK * NBLK + bj * NBLK + bk;
                int a_idx = bi * NBLK + bk;
                int b_idx = bk * NBLK + bj;
                memcpy(work_sets[idx].A_ope, all_A_ope[a_idx], sizeof(work_sets[0].A_ope));
                memcpy(work_sets[idx].B_ope, all_B_ope[b_idx], sizeof(work_sets[0].B_ope));
                memcpy(work_sets[idx].A_rvv, all_A_rvv[a_idx], sizeof(work_sets[0].A_rvv));
                memcpy(work_sets[idx].B_rvv, all_B_rvv[b_idx], sizeof(work_sets[0].B_rvv));
            }

    /* ── OPE+RVV: 8 sub-multiplies ── */
    memset(C_out, 0, sizeof(C_out));
    memset(sub_out, 0, sizeof(sub_out));

    printf("[ope+rvv] running 8 sub-multiplies (32x32 each)...\n");
    uint64_t t0 = rdcycle64();

    for (int bi = 0; bi < NBLK; bi++) {
        for (int bj = 0; bj < NBLK; bj++) {
            for (int bk = 0; bk < NBLK; bk++) {
                int idx = bi * NBLK * NBLK + bj * NBLK + bk;
                work_set_t *w = &work_sets[idx];

                /* Prime OPE EXT output into L1 (must not miss) */
                prime_32x32(sub_out[bi][bj][bk], BLK);

                /* Run the proven 32×32 kernel */
                gemm_ope_rvv_32x32(
                    w->A_ope, w->B_ope,
                    sub_out[bi][bj][bk],
                    w->A_rvv, w->B_rvv,
                    RVV_ROWS);
            }
        }
    }

    uint64_t t1 = rdcycle64();
    printf("[ope+rvv] done, cycles=%llu\n", (unsigned long long)(t1 - t0));

    /* Accumulate sub-block results into C_out (outside timed region) */
    for (int bi = 0; bi < NBLK; bi++)
        for (int bj = 0; bj < NBLK; bj++)
            for (int bk = 0; bk < NBLK; bk++)
                accum_block(sub_out[bi][bj][bk], C_out, bi, bj, DIM);

    /* ── Pure-RVV baseline (no blocking needed) ── */
    memset(C_rvv_only, 0, sizeof(C_rvv_only));

    /* Prime */
    {
        volatile int8_t sink;
        volatile int32_t sink32;
        for (int off = 0; off < DIM * DIM; off += 64)
            sink = A[off];
        for (int off = 0; off < (DIM + 1) * DIM; off += 64)
            sink = B_rvv_full[off];
        for (int r = 0; r < DIM; r += 4)
            sink32 = C_rvv_only[r * DIM];
        asm volatile("fence r, rw" ::: "memory");
    }

    printf("[rvv-only] gemm_rvv_all (%d rows)...\n", DIM);
    uint64_t t2 = rdcycle64();
    gemm_rvv_all(A, B_rvv_full, C_rvv_only, DIM, DIM, DIM, DIM);
    uint64_t t3 = rdcycle64();
    uint64_t rvv_cycles = t3 - t2;
    uint64_t ope_rvv_cycles = t1 - t0;
    printf("[rvv-only] done, cycles=%llu\n", (unsigned long long)rvv_cycles);

    /* ── Comparison ── */
    printf("\n=== Performance comparison (64x64) ===\n");
    printf("  OPE+RVV (8x 32x32) : %llu cycles\n", (unsigned long long)ope_rvv_cycles);
    printf("  RVV-only (64 rows)  : %llu cycles\n", (unsigned long long)rvv_cycles);
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
