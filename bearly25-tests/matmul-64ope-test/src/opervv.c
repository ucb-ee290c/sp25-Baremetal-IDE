/*
 * opervv.c — 64×64 int8 matmul via 32×32 blocking
 *
 * The 64×64 problem is decomposed into 8 sub-multiplies of 32×32,
 * each using the exact same OPE+RVV kernel proven at 32×32:
 *   - K=32, N=32, ldc=32, EXT stride=32
 *   - Single ACC(32) per col-tile
 *   - 4 col-tiles × [7,7,7,3] RVV rows = 24 interleaved rows
 *
 * C[i,j] = Σ_k A[i,k] × B[k,j]   (i,j,k ∈ {0,1}, each block 32×32)
 *
 * RVV uses gemm_i8_i32_7xm4 / gemm_i8_i32_1xm4 from bearly25-bmarks/rvv-matmul.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "riscv_vector.h"
#include "rocc.h"

/* ──────────────────────── OPE low-level interface ──────────────────────── */

#ifndef OPE_CUSTOM
#define OPE_CUSTOM 0
#endif

#ifndef OPE_EXT_FLIP
#define OPE_EXT_FLIP 1
#endif

#define FCTN7_ACC     0b00
#define FCTN7_EXTRACT 0b01
#define FCTN7_ZERO    0b10

#define OP_ZERO() ROCC_INSTRUCTION(OPE_CUSTOM, FCTN7_ZERO)

static inline void OP_ACC_L(int8_t *U, int8_t *V, int L) {
  register uint64_t rs1 asm("x11") = (uint64_t)U;
  register uint64_t rs2 asm("x12") = (uint64_t)V;
  switch (L) {
    case  1: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 0<<2)); break;
    case  2: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 1<<2)); break;
    case  3: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 2<<2)); break;
    case  4: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 3<<2)); break;
    case  5: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 4<<2)); break;
    case  6: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 5<<2)); break;
    case  7: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 6<<2)); break;
    case  8: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 7<<2)); break;
    case  9: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 8<<2)); break;
    case 10: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|( 9<<2)); break;
    case 11: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(10<<2)); break;
    case 12: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(11<<2)); break;
    case 13: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(12<<2)); break;
    case 14: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(13<<2)); break;
    case 15: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(14<<2)); break;
    case 16: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(15<<2)); break;
    case 17: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(16<<2)); break;
    case 18: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(17<<2)); break;
    case 19: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(18<<2)); break;
    case 20: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(19<<2)); break;
    case 21: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(20<<2)); break;
    case 22: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(21<<2)); break;
    case 23: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(22<<2)); break;
    case 24: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(23<<2)); break;
    case 25: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(24<<2)); break;
    case 26: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(25<<2)); break;
    case 27: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(26<<2)); break;
    case 28: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(27<<2)); break;
    case 29: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(28<<2)); break;
    case 30: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(29<<2)); break;
    case 31: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(30<<2)); break;
    case 32: ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC|(31<<2)); break;
  }
}

#define _OP_EXT_S_T(rs1, rs2)  \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_EXTRACT|(1<<2)|(1<<3))
#define _OP_EXT_NS_T(rs2)      \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, 0,   rs2, FCTN7_EXTRACT|(1<<2)|(0<<3))
#define _OP_EXT_S_NT(rs1, rs2) \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_EXTRACT|(0<<2)|(1<<3))
#define _OP_EXT_NS_NT(rs2)     \
  ROCC_INSTRUCTION_SS(OPE_CUSTOM, 0,   rs2, FCTN7_EXTRACT|(0<<2)|(0<<3))

/*
 * Re-prime just the 8 cache lines for one EXT tile immediately before EXT.
 * Each tile row lands on exactly one 64-byte cache line.
 * stride_elements is in int32 units.
 */
static inline void ope_prime_tile_lines(int32_t *arr, int stride_elements) {
    volatile int32_t sink;
    for (int r = 0; r < 8; r++) {
        sink = arr[r * stride_elements];
    }
    asm volatile("fence r, rw" ::: "memory");
}

/*
 * Prime the ACC input vectors immediately before OP_ACC_L to prevent nacks
 * on OPE's cache reads of U and V.
 *
 * Each OP_ACC_L(ap, bp, L) causes the OPE to issue reads over L*8 bytes from
 * both ap and bp.  Touch one address per 64-byte cache line in each range.
 */
static inline void ope_prime_acc_inputs(const int8_t *ap, const int8_t *bp, int L) {
    volatile int8_t sink;
    int bytes = L * 8;
    for (int off = 0; off < bytes; off += 64) {
        sink = ap[off];
        sink = bp[off];
    }
    asm volatile("fence r, rw" ::: "memory");
}

/* ──────────────────────── end OPE interface ──────────────────────── */

/* Forward declarations for RVV kernels from bearly25-bmarks/rvv-matmul */
void gemm_i8_i32_7xm4(size_t mr, size_t nc, size_t kc,
                       const int8_t *a, size_t a_stride,
                       const int8_t *w,
                       int32_t *c, size_t cm_stride, size_t cn_stride);
void gemm_i8_i32_1xm4(size_t mr, size_t nc, size_t kc,
                       const int8_t *a, size_t a_stride,
                       const int8_t *w,
                       int32_t *c, size_t cm_stride, size_t cn_stride);

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

/* ====================================================================
 * Packing helpers
 * ==================================================================== */

void pack_A_ope(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K)
{
    for (int chunk = 0; chunk < num_rows / 8; chunk++) {
        for (int r = 0; r < 8; r++) {
            int row = row_start + chunk * 8 + r;
            for (int k = 0; k < K; k++)
                A_packed[chunk * K * 8 + k * 8 + r] = A[row * lda + k];
        }
    }
}

void pack_B_ope(const int8_t *B, int ldb, int8_t *B_packed, int K, int N)
{
    for (int chunk = 0; chunk < N / 8; chunk++) {
        for (int k = 0; k < K; k++) {
            for (int c = 0; c < 8; c++)
                B_packed[chunk * K * 8 + k * 8 + c] = B[k * ldb + chunk * 8 + c];
        }
    }
}

void pack_A_rvv(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K)
{
    for (int r = 0; r < num_rows; r++)
        for (int k = 0; k < K; k++)
            A_packed[r * K + k] = A[(row_start + r) * lda + k];
}

void pack_B_rvv_bias(const int8_t *B, int ldb, int8_t *B_packed, int K, int N)
{
    memset(B_packed, 0, (size_t)N);  /* bias row */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B_packed[(k + 1) * N + j] = B[k * ldb + j];
}

/* ====================================================================
 * rvv_compute_rows — compute a batch of RVV rows using 7xm4 + 1xm4.
 * ==================================================================== */
static void rvv_compute_rows(
    const int8_t *a, int num_rows,
    size_t N, size_t K,
    const int8_t *B_rvv,
    int32_t *C_row, int ldc)
{
    size_t cm_stride = (size_t)ldc * sizeof(int32_t);
    int r = 0;
    while (r + 7 <= num_rows) {
        gemm_i8_i32_7xm4(7, N, K,
                          a + (size_t)r * K, K,
                          B_rvv,
                          C_row + r * ldc, cm_stride, 0);
        r += 7;
    }
    while (r < num_rows) {
        gemm_i8_i32_1xm4(1, N, K,
                          a + (size_t)r * K, K,
                          B_rvv,
                          C_row + r * ldc, cm_stride, 0);
        r++;
    }
}

/* ====================================================================
 * prime_32x32 — bring OPE data for one 32×32 sub-kernel into L1.
 *
 * Only primes OPE ACC inputs and EXT output region.
 * RVV inputs/outputs are NOT primed — Saturn handles cache misses
 * gracefully, and with page-aligned work sets the RVV data occupies
 * different cache sets than OPE data (no eviction risk).
 * ==================================================================== */
void prime_32x32(int32_t *C, int ldc)
{
    volatile int32_t sink32;
    /* OPE EXT output (8 rows × 32 cols × 4B = 1024B = 16 lines).
     * EXT writes MUST hit L1 — misses cause nacks that hang the FSM.
     * ACC read misses are slow but safe (no hang), so skip A_ope/B_ope. */
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 32; c += 16)
            sink32 = C[r * ldc + c];
    asm volatile("fence r, rw" ::: "memory");
}

/* ====================================================================
 * gemm_ope_rvv_32x32 — 32×32 sub-kernel, IDENTICAL to proven 32×32.
 *
 * All parameters fixed at 32×32:
 *   N=32, K=32, ldc=32, EXT stride=32
 *   Single ACC(32) per col-tile, 4 col-tiles
 *   RVV: [7,7,7,3] rows interleaved
 *
 * Caller must prime data into L1 before calling.
 * A_rvv must be contiguously packed (stride = K = 32).
 * ==================================================================== */
void gemm_ope_rvv_32x32(
    const int8_t *A_packed,   /* OPE-packed, 1×32×8 = 256 bytes */
    const int8_t *B_packed,   /* OPE-packed, 4×32×8 = 1024 bytes */
    int32_t *C,               /* output 32×32, ldc=32 */
    const int8_t *A_rvv,      /* contiguous 24×32, stride=32 */
    const int8_t *B_rvv,      /* (33)×32 with bias row */
    int num_rvv_rows)         /* 24 */
{
    const int N = 32;
    const int K = 32;
    const int ldc = 32;
    int rvv_row_cursor = 0;

    for (int j = 0; j < N / 8; j++) {
        const int8_t *ap = A_packed;
        const int8_t *bp = B_packed + j * K * 8;

        /* ── ZERO + single ACC(32) ── */
        OP_ZERO();
        OP_ACC_L((int8_t *)ap, (int8_t *)bp, K);

        /* ── RVV batch — hides ACC latency ── */
        int rvv_count = MIN(7, num_rvv_rows - rvv_row_cursor);
        if (rvv_count > 0) {
            rvv_compute_rows(
                A_rvv + (size_t)rvv_row_cursor * K,
                rvv_count, N, K,
                B_rvv,
                C + (8 + rvv_row_cursor) * ldc, ldc);
            rvv_row_cursor += rvv_count;
        }

        /* ── EXT with stride=32 ── */
        {
            register uint64_t rs1 asm("x11") = (uint64_t)ldc;
            register uint64_t rs2 asm("x12") = (uint64_t)(C + j * 8);
            if (OPE_EXT_FLIP) { _OP_EXT_S_T(rs1, rs2);  }
            else              { _OP_EXT_S_NT(rs1, rs2); }
        }
    }

    /* Drain all in-flight OPE EXT writes */
    asm volatile("fence w, rw" ::: "memory");
}

/* ====================================================================
 * gemm_rvv_all — pure RVV baseline for M rows (no OPE, no blocking).
 * ==================================================================== */
void gemm_rvv_all(const int8_t *A, const int8_t *B_rvv,
                  int32_t *C, int M, int N, int K, int ldc)
{
    size_t cm_stride = (size_t)ldc * sizeof(int32_t);
    int r = 0;
    while (r + 7 <= M) {
        gemm_i8_i32_7xm4(7, N, K,
                          A + (size_t)r * K, K,
                          B_rvv,
                          C + r * ldc, cm_stride, 0);
        r += 7;
    }
    while (r < M) {
        gemm_i8_i32_1xm4(1, N, K,
                          A + (size_t)r * K, K,
                          B_rvv,
                          C + r * ldc, cm_stride, 0);
        r++;
    }
}
