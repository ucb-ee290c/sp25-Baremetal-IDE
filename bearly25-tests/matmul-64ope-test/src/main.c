/*
 * matmul-64ope-test/src/main.c
 *
 * 64×64 int8 matmul via 32×32 blocking.
 *
 * C[i,j] = Σ_k A[i,k] × B[k,j]   (i,j,k ∈ {0,1}, each block 32×32)
 * = 8 sub-multiplies, each using the proven 32×32 OPE+RVV kernel.
 *
 * Double-ACC: ZERO+ACC(k0)+ACC(k1) per col-tile reduces 8→4 sub-kernels.
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
/*
 * Inline: drain previous OPE EXT writes, then prime next sub_out.
 * Drain fence is here (not at end of kernel) so OPE writes overlap
 * with loop iteration overhead.
 */
static inline void prime_32x32(int32_t *C, int ldc)
{
    /* Drain previous OPE EXT writes before touching sub_out cache sets */
    asm volatile("fence w, rw" ::: "memory");
    volatile int32_t sink32;
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 32; c += 16)
            sink32 = C[r * ldc + c];
    asm volatile("fence r, rw" ::: "memory");
}

void gemm_ope_rvv_32x32_2k(
    const int8_t *A_ope_k0, const int8_t *B_ope_k0,
    const int8_t *A_ope_k1, const int8_t *B_ope_k1,
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

/*
 * Double-ACC work sets: one per (bi,bj) pair (4 total).
 * Each holds BOTH k-blocks' OPE data plus K=64 RVV data.
 * Page-aligned so every set has identical cache set mapping.
 *
 * A_ope_k0/k1 (256 B each) and B_ope_k0/k1 (1024 B each) are the
 * OPE inputs for k=0:31 and k=32:63.  A_rvv (24×64=1536 B) and
 * B_rvv (65×32=2080 B) are the RVV inputs for the full K=64.
 *
 * Total: 256+1024+256+1024+1536+2080 = 6176 B → pad to 8192 (2 pages).
 */
typedef struct {
    int8_t A_ope_k0[(OPE_ROWS / 8) * BLK * 8];  /* 256 B  */
    int8_t B_ope_k0[(BLK / 8) * BLK * 8];       /* 1024 B */
    int8_t A_ope_k1[(OPE_ROWS / 8) * BLK * 8];  /* 256 B  */
    int8_t B_ope_k1[(BLK / 8) * BLK * 8];       /* 1024 B */
    int8_t A_rvv[RVV_ROWS * DIM];               /* 1536 B (24×64) */
    int8_t B_rvv[(DIM + 1) * BLK];              /* 2080 B (65×32) */
    int8_t _pad[8192 - 256 - 1024 - 256 - 1024 - 1536 - 2080];
} work_set_t;

static work_set_t work_sets[NBLK * NBLK] __attribute__((aligned(4096)));

/* Sub-kernel outputs: one per (bi,bj) pair (4 total).
 * Page-align for deterministic OPE cache set mapping. */
static int32_t sub_out[NBLK][NBLK][BLK * BLK] __attribute__((aligned(4096)));

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

    /* ── Pre-pack OPE blocks (4 A-blocks, 4 B-blocks) ── */
    printf("[pack] pre-packing all sub-blocks...\n");
    for (int bi = 0; bi < NBLK; bi++)
        for (int bk = 0; bk < NBLK; bk++) {
            int a_idx = bi * NBLK + bk;
            const int8_t *A_sub = A + bk * BLK;
            pack_A_ope(A_sub, DIM, all_A_ope[a_idx], bi * BLK, OPE_ROWS, BLK);
        }
    for (int bk = 0; bk < NBLK; bk++)
        for (int bj = 0; bj < NBLK; bj++) {
            int b_idx = bk * NBLK + bj;
            const int8_t *B_sub = B + bk * BLK * DIM + bj * BLK;
            pack_B_ope(B_sub, DIM, all_B_ope[b_idx], BLK, BLK);
        }

    /* ── Full-size B_rvv for pure-RVV baseline ── */
    pack_B_rvv_bias(B, DIM, B_rvv_full, DIM, DIM);

    /* ── Pre-populate double-ACC work sets (before timing) ──
     * Each work_set[bi*2+bj] holds both k-blocks' OPE data and
     * K=64 RVV data for the full (bi,bj) output block. */
    for (int bi = 0; bi < NBLK; bi++)
        for (int bj = 0; bj < NBLK; bj++) {
            int idx = bi * NBLK + bj;
            work_set_t *w = &work_sets[idx];
            /* OPE data for k=0:31 */
            int a0 = bi * NBLK + 0;
            int b0 = 0 * NBLK + bj;
            memcpy(w->A_ope_k0, all_A_ope[a0], sizeof(w->A_ope_k0));
            memcpy(w->B_ope_k0, all_B_ope[b0], sizeof(w->B_ope_k0));
            /* OPE data for k=32:63 */
            int a1 = bi * NBLK + 1;
            int b1 = 1 * NBLK + bj;
            memcpy(w->A_ope_k1, all_A_ope[a1], sizeof(w->A_ope_k1));
            memcpy(w->B_ope_k1, all_B_ope[b1], sizeof(w->B_ope_k1));
            /* RVV A: pack rows [bi*32+8, bi*32+32) with full K=64 */
            pack_A_rvv(A, DIM, w->A_rvv,
                        bi * BLK + OPE_ROWS, RVV_ROWS, DIM);
            /* RVV B: pack B columns [bj*32, bj*32+32) with full K=64 */
            const int8_t *B_col = B + bj * BLK;
            pack_B_rvv_bias(B_col, DIM, w->B_rvv, DIM, BLK);
        }

    /* ── OPE+RVV: 4 sub-kernels (double ACC merges bk loop) ── */
    memset(C_out, 0, sizeof(C_out));
    memset(sub_out, 0, sizeof(sub_out));

    printf("[ope+rvv] running 4 sub-kernels (double-ACC 32x32)...\n");
    uint64_t t0 = rdcycle64();

    for (int bi = 0; bi < NBLK; bi++) {
        for (int bj = 0; bj < NBLK; bj++) {
            int idx = bi * NBLK + bj;
            work_set_t *w = &work_sets[idx];

            /* Prime OPE EXT output into L1 */
            prime_32x32(sub_out[bi][bj], BLK);

            /* Double-ACC kernel: ZERO+ACC(k0)+ACC(k1) per col-tile */
            gemm_ope_rvv_32x32_2k(
                w->A_ope_k0, w->B_ope_k0,
                w->A_ope_k1, w->B_ope_k1,
                sub_out[bi][bj],
                w->A_rvv, w->B_rvv,
                RVV_ROWS);
        }
    }

    uint64_t t1 = rdcycle64();
    printf("[ope+rvv] done, cycles=%llu\n", (unsigned long long)(t1 - t0));

    /* Accumulate sub-block results into C_out (outside timed region) */
    for (int bi = 0; bi < NBLK; bi++)
        for (int bj = 0; bj < NBLK; bj++)
            accum_block(sub_out[bi][bj], C_out, bi, bj, DIM);

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
