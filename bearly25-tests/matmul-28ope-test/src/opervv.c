/*
 * opervv.c — OPE 8-row + RVV 24-row kernels for 32×32 int8 matmul
 *
 * Split: rows  0-7   → OPE  (1 tile of 8 rows × 4 col-tiles)
 *        rows 8-31   → RVV  (3 groups of 7 rows + 3 remaining = 24 rows)
 *
 * B layout for RVV kernel: (K+1)×N row-major, row 0 = zero bias.
 * A layout for RVV kernel: row-major — a_packed[r*K + k] = A[row_start+r][k].
 * A layout for OPE kernel: a_packed[chunk*K*8 + k*8 + r] = A[chunk*8+r][k].
 * B layout for OPE kernel: b_packed[chunk*K*8 + k*8 + c] = B[k][chunk*8+c].
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
 * Prime every cache line in the OPE output region before the tile loop.
 *
 * Why: HellaCache can issue an s2_nack for any write that conflicts with an
 * in-progress MSHR fill. SimpleHellaCacheIF replays nacked requests but also
 * passes the replay response directly back to the requestor (OPE) via the
 * same io.requestor.resp path, with req.ready held low during the replay.
 * This can cause the OPE's EXTRACT FSM to fire resp before the corresponding
 * req.fire, advancing the minor/major counter out of sync and leaving the FSM
 * stuck waiting for a response that will never arrive.
 *
 * By touching every 64-byte cache line in the entire output region before
 * issuing any OPE instruction, all writes become guaranteed L1 hits so no
 * nacks are ever issued during EXTRACT.
 *
 * C       : pointer to row 0, col 0 of the OPE output region
 * num_rows: number of rows (= num_row_tiles * 8)
 * N       : number of columns
 * ldc     : row stride in int32 elements
 */
static void ope_prime_output_region(int32_t *C, int num_rows, int N, int ldc)
{
    volatile int32_t sink;
    /* One load per 64-byte cache line.
     * 64 bytes / sizeof(int32_t) = 16 elements per line. */
    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < N; c += 16) {
            sink = C[r * ldc + c];
        }
    }
    asm volatile("fence r, rw" ::: "memory");
}

/*
 * Prime the ACC input vectors immediately before OP_ACC_L to prevent nacks
 * on OPE's cache reads of U and V.
 *
 * Each OP_ACC_L(ap, bp, L) causes the OPE to issue reads over L*8 bytes from
 * both ap and bp.  Touch one address per 64-byte cache line in each range,
 * then fence to ensure the lines are resident before the instruction fires.
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

/*
 * Re-prime just the 8 cache lines for one tile immediately before EXT.
 * Each tile row lands on exactly one 64-byte cache line (4 minor-pair writes
 * × 8 bytes = 32 bytes, all within a single aligned 64-byte line).
 * stride_elements is in int32 units.
 */
static inline void ope_prime_tile_lines(int32_t *arr, int stride_elements) {
    volatile int32_t sink;
    for (int r = 0; r < 8; r++) {
        sink = arr[r * stride_elements];
    }
    asm volatile("fence r, rw" ::: "memory");
}

static inline void OP_EXT_STRIDE(int32_t *arr, int stride_elements, int transposed) {
  ope_prime_tile_lines(arr, stride_elements == 0 ? 8 : stride_elements);
  register uint64_t rs2 asm("x12") = (uint64_t)arr;
  if (stride_elements == 0 || stride_elements == 8) {
    if (transposed) { _OP_EXT_NS_T(rs2);  }
    else            { _OP_EXT_NS_NT(rs2); }
  } else {
    register uint64_t rs1 asm("x11") = (uint64_t)stride_elements;
    if (transposed) { _OP_EXT_S_T(rs1, rs2);  }
    else            { _OP_EXT_S_NT(rs1, rs2); }
  }
  asm volatile("fence w, r" ::: "memory");
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

/*
 * pack_A_ope — pack num_rows rows of A (starting at row_start) for OPE.
 *
 * num_rows must be a multiple of 8.
 * Out layout: A_packed[chunk * K * 8 + k * 8 + r]  =  A[(row_start + chunk*8 + r) * lda + k]
 */
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

/*
 * pack_B_ope — pack B (K rows, N cols) for OPE column tiles.
 *
 * N must be a multiple of 8.
 * Out layout: B_packed[chunk * K * 8 + k * 8 + c]  =  B[k * ldb + chunk*8 + c]
 */
void pack_B_ope(const int8_t *B, int ldb, int8_t *B_packed, int K, int N)
{
    for (int chunk = 0; chunk < N / 8; chunk++) {
        for (int k = 0; k < K; k++) {
            for (int c = 0; c < 8; c++)
                B_packed[chunk * K * 8 + k * 8 + c] = B[k * ldb + chunk * 8 + c];
        }
    }
}

/*
 * pack_A_rvv — pack num_rows rows of A (starting at row_start) for RVV kernel.
 *
 * Out layout: A_packed[r * K + k]  =  A[(row_start + r) * lda + k]  (row-major)
 * Pass a_stride = K (bytes) to gemm_i8_i32_7xm4 / gemm_i8_i32_1xm4.
 */
void pack_A_rvv(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K)
{
    for (int r = 0; r < num_rows; r++)
        for (int k = 0; k < K; k++)
            A_packed[r * K + k] = A[(row_start + r) * lda + k];
}

/*
 * pack_B_rvv_bias — pack B (K rows, N cols) for RVV kernel with zero bias row.
 *
 * Output is (K+1)×N row-major: row 0 = zeros, rows 1..K = B[0..K-1].
 * Pass b_row_stride = N and this buffer as w to the kernel.
 */
void pack_B_rvv_bias(const int8_t *B, int ldb, int8_t *B_packed, int K, int N)
{
    memset(B_packed, 0, (size_t)N);  /* bias row */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B_packed[(k + 1) * N + j] = B[k * ldb + j];
}

/* ====================================================================
 * rvv_compute_rows — compute a batch of RVV output rows using the 7xm4 kernel.
 *
 * Fills the OPE ACC latency window with useful vector work.
 * Processes num_rows rows in batches of 7 (7xm4), then 1 (1xm4) for remainder.
 *
 * a:         A_rvv + rvv_row_cursor*K  (row-major, K bytes stride between rows)
 * num_rows:  number of rows to compute
 * B_rvv:     (K+1)×N, row 0 = zero bias, rows 1..K = B[0..K-1]
 * C_row:     C_rvv + rvv_row_cursor*ldc  (pointer to first output row)
 *
 * A_rvv and B_rvv are in different memory regions from A_ope/B_ope, so these
 * loads do not evict the cache lines the OPE is concurrently fetching.
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
 * gemm_ope_16rows — OPE kernel for rows 0-7, interleaved with RVV rows 8-31
 *
 * OPE: 1 row-tile × (N/8) col-tiles of 8×8.
 *   A_packed: [1][K][8]      (pack_A_ope, row_start=0, num_rows=8)
 *   B_packed: [(N/8)][K][8]  (pack_B_ope)
 *   EXTRACT uses stride=ldc (≤32) directly into C — no tmp buffer or scatter.
 *
 * RVV interleaving: distributes num_rvv_rows rows in batches of up to 7 across
 *   the (N/8) OPE col-tiles, using gemm_i8_i32_7xm4 / gemm_i8_i32_1xm4 from
 *   bearly25-bmarks/rvv-matmul. For N=32 and num_rvv_rows=24 this gives
 *   [7, 7, 7, 3] rows per tile.
 *
 *   A_rvv:    row-major (r*K + k), K bytes stride  — different region from A_ope/B_ope
 *   B_rvv:    (K+1)×N, row 0 = zero bias           — compatible with 7xm4 kernel
 *   C_rvv:    C + RVV_START*ldc
 *
 * Interleaving strategy:
 *   Each col-tile follows the pattern:
 *     prime ACC inputs → ZERO → ACC(s) → RVV batch (hides ACC latency) →
 *     re-prime EXT output lines → fire EXT (no per-tile fence)
 *   A single fence at the end drains all in-flight EXT writes.  This lets
 *   the CPU issue ZERO+ACC for tile j+1 into the OPE command queue
 *   immediately after firing EXT for tile j, so the OPE is never idle
 *   between col-tiles.
 *
 * L1 nack avoidance:
 *   ope_prime_output_region touches every cache line of the OPE output
 *   region at the start, before any EXT.  ope_prime_acc_inputs touches
 *   every cache line of U and V before each ACC.  ope_prime_tile_lines
 *   re-primes the 8 per-row lines immediately before each EXT fires.
 *   Together these guarantee every OPE TileLink request hits in L1.
 * ==================================================================== */
void gemm_ope_16rows(
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

    /*
     * Pre-prime every cache line in the OPE output region (all 8 rows × N cols).
     * With ldc=32 each row is 128 bytes = 2 cache lines; ope_prime_output_region
     * strides by 16 int32s (64 bytes) so both lines per row are touched.
     * This ensures all subsequent EXTRACT writes are guaranteed L1 hits regardless
     * of which cache lines the RVV kernel may evict between here and the EXT.
     */
    ope_prime_output_region(C, 8, N, ldc);

    /* OPE: 1 row-tile (i=0), N/8 col-tiles */
    for (int j = 0; j < N / 8; j++) {
        const int8_t *ap = A_packed;               /* row-tile 0, fixed across j */
        const int8_t *bp = B_packed + j * K * 8;

        /*
         * Prime ACC input vectors before issuing ZERO/ACC to prevent L1 nacks
         * on the OPE's TileLink reads of U (ap) and V (bp).
         * bytes = K*8; for K=32 this primes 256 bytes at 64-byte intervals
         * covering all cache lines the upcoming OP_ACC_L will access.
         */
        ope_prime_acc_inputs(ap, bp, K);

        OP_ZERO();
        int k_rem = K;
        while (k_rem > 0) {
            int L = MIN(32, k_rem);
            OP_ACC_L((int8_t *)ap, (int8_t *)bp, L);
            ap += L * 8;
            bp += L * 8;
            k_rem -= L;
        }

        /*
         * Interleave up to 7 RVV rows during the OPE ACC latency window.
         * A_rvv/B_rvv are in different memory regions from A_packed/B_packed
         * so these vector loads cannot evict the OPE's ACC input cache lines.
         * OPE rows 0-7 and RVV rows 8-31 occupy disjoint regions of C,
         * so RVV stores cannot conflict with OPE's pending EXT writes.
         */
        int rvv_count = MIN(7, num_rvv_rows - rvv_row_cursor);
        if (rvv_count > 0) {
            rvv_compute_rows(
                A_rvv + (size_t)rvv_row_cursor * K,
                rvv_count, N, K,
                B_rvv,
                C_rvv + rvv_row_cursor * ldc, ldc);
            rvv_row_cursor += rvv_count;
        }

        /*
         * Re-prime this tile's 8 output cache lines immediately before EXTRACT.
         * RVV stores to rows 8-31 are to different cache sets but re-priming
         * here is cheap insurance against any L1 eviction since the upfront prime.
         * stride=ldc=32 (≤ 32 limit): each of the 8 rows is 128 bytes apart so
         * every load touches a distinct 64-byte cache line.
         */
        ope_prime_tile_lines(C + j * 8, ldc);

        /*
         * Fire EXTRACT directly into C with stride=ldc (≤ 32), writing
         *   C[r*ldc + j*8 .. j*8+7]  for r = 0..7
         * No per-tile fence: EXT is a fire-and-forget RoCC instruction (xd=0).
         * The OPE command queue ensures EXT executes after all preceding ACCs,
         * and ZERO/ACC for tile j+1 (issued next iteration) execute after EXT.
         * A single fence at the end of the loop drains all in-flight EXT writes.
         */
        {
            register uint64_t rs1 asm("x11") = (uint64_t)ldc;
            register uint64_t rs2 asm("x12") = (uint64_t)(C + j * 8);
            if (OPE_EXT_FLIP) { _OP_EXT_S_T(rs1, rs2);  }
            else              { _OP_EXT_S_NT(rs1, rs2); }
        }
    }

    /* Drain all in-flight OPE EXT writes and RVV stores before returning. */
    asm volatile("fence w, rw" ::: "memory");
}

