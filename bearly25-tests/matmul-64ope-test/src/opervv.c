/*
 * opervv.c — OPE 8-row + RVV (DIM-8)-row kernels for DIM×DIM int8 matmul
 *
 * Generalized for arbitrary DIM (multiple of 8).
 * OPE handles rows 0-7, RVV handles the rest.
 * ACC stride capped at 32: K is broken into ceil(K/32) ACC calls per tile.
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

void pack_B_rvv_bias(const int8_t *B, int ldb, int8_t *B_packed, int K, int N)
{
    memset(B_packed, 0, (size_t)N);  /* bias row */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B_packed[(k + 1) * N + j] = B[k * ldb + j];
}

/* ====================================================================
 * rvv_compute_rows — compute a batch of RVV output rows.
 * Processes in batches of 7 (7xm4), then singles (1xm4) for remainder.
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
 * gemm_ope_rvv — OPE 8 rows + RVV (num_rvv_rows) rows, interleaved.
 *
 * Generalized for any DIM (multiple of 8).
 * ACC stride capped at 32: K is broken into ceil(K/32) ACC calls.
 * RVV rows are distributed across N/8 col-tiles, 7 per tile.
 *
 * Priming is done by the caller before timing begins.
 * ==================================================================== */
void gemm_ope_rvv(
    const int8_t *A_packed,
    const int8_t *B_packed,
    int32_t *C,
    int N, int K, int ldc,
    const int8_t *A_rvv,
    const int8_t *B_rvv,
    int num_rvv_rows,
    int32_t *C_rvv)
{
    int rvv_row_cursor = 0;

    for (int j = 0; j < N / 8; j++) {
        const int8_t *ap = A_packed;
        const int8_t *bp = B_packed + j * K * 8;

        /* ── 1. Fire ZERO + ACC (multiple ACC calls if K > 32) ── */
        OP_ZERO();
        int k_rem = K;
        while (k_rem > 0) {
            int L = MIN(32, k_rem);
            OP_ACC_L((int8_t *)ap, (int8_t *)bp, L);
            ap += L * 8;
            bp += L * 8;
            k_rem -= L;
        }

        /* ── 2. RVV batch — hides ACC latency ── */
        int rvv_count = MIN(7, num_rvv_rows - rvv_row_cursor);
        if (rvv_count > 0) {
            rvv_compute_rows(
                A_rvv + (size_t)rvv_row_cursor * K,
                rvv_count, N, K,
                B_rvv,
                C_rvv + rvv_row_cursor * ldc, ldc);
            rvv_row_cursor += rvv_count;
        }

        /* ── 3. Fire EXT ── */
        {
            register uint64_t rs1 asm("x11") = (uint64_t)ldc;
            register uint64_t rs2 asm("x12") = (uint64_t)(C + j * 8);
            if (OPE_EXT_FLIP) { _OP_EXT_S_T(rs1, rs2);  }
            else              { _OP_EXT_S_NT(rs1, rs2); }
        }
    }

    /* Compute any remaining RVV rows not covered during OPE tiles */
    if (rvv_row_cursor < num_rvv_rows) {
        rvv_compute_rows(
            A_rvv + (size_t)rvv_row_cursor * K,
            num_rvv_rows - rvv_row_cursor, N, K,
            B_rvv,
            C_rvv + rvv_row_cursor * ldc, ldc);
    }

    /* Drain all in-flight OPE EXT writes and RVV stores. */
    asm volatile("fence w, rw" ::: "memory");
}

/* ====================================================================
 * gemm_rvv_all — pure RVV baseline for all M rows (no OPE).
 * ==================================================================== */
void gemm_rvv_all(const int8_t *A_rvv, const int8_t *B_rvv,
                  int32_t *C, int M, int N, int K, int ldc)
{
    size_t cm_stride = (size_t)ldc * sizeof(int32_t);
    int r = 0;
    while (r + 7 <= M) {
        gemm_i8_i32_7xm4(7, N, K,
                          A_rvv + (size_t)r * K, K,
                          B_rvv,
                          C + r * ldc, cm_stride, 0);
        r += 7;
    }
    while (r < M) {
        gemm_i8_i32_1xm4(1, N, K,
                          A_rvv + (size_t)r * K, K,
                          B_rvv,
                          C + r * ldc, cm_stride, 0);
        r++;
    }
}
