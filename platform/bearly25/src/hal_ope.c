#include "hal_ope.h"
#include <stddef.h>
#include "rocc.h"

// ===== Fences =====
static inline void _fence_all(void){ asm volatile("fence iorw, iorw" ::: "memory"); }
static inline void _fence_wr(void) { asm volatile("fence w, r"      ::: "memory"); }
static inline void _fence_rw(void) { asm volatile("fence r, w"      ::: "memory"); }

void ope_fence(void) { _fence_all(); }

// ===== Functional Operations =====

// funct7 tags (see BearlyML’25 OPE ISA)
#define FCTN7_ACC     0b00
#define FCTN7_EXTRACT 0b01
#define FCTN7_ZERO    0b10
#define FCTN7_LOAD    0b11

// Rocket source register numbers for RoCC interface
#define ROCC_RS1_REG_N 11
#define ROCC_RS2_REG_N 12

#define REG_STR_HELPER(x) #x
#define REG_STR(x)        REG_STR_HELPER(x)
#define REG_VAR(name, reg_num) register uint64_t name asm("x" REG_STR(reg_num))

// ===== Low-Level Driver Functions =====

/* -------------------------- ZERO ---------------------------- */
void ope_zero(void) {
  _fence_all();
  ROCC_INSTRUCTION(OPE_CUSTOM, FCTN7_ZERO);
  _fence_all();
}

/* --------------------------- LOAD --------------------------- */
#define _OP_LOAD_S_T(rs1, rs2)  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1,rs2, FCTN7_LOAD | (1 << 2) | (1 << 3))
#define _OP_LOAD_NS_T(rs2)      ROCC_INSTRUCTION_SS(OPE_CUSTOM,   0,rs2, FCTN7_LOAD | (1 << 2) | (0 << 3))
#define _OP_LOAD_S_NT(rs1, rs2) ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1,rs2, FCTN7_LOAD | (0 << 2) | (1 << 3))
#define _OP_LOAD_NS_NT(rs2)     ROCC_INSTRUCTION_SS(OPE_CUSTOM,   0,rs2, FCTN7_LOAD | (0 << 2) | (0 << 3))

void ope_load(uint64_t mem_base_phys, uint16_t stride_elems, uint8_t transpose, uint8_t use_stride)
{
  REG_VAR(rs2, ROCC_RS2_REG_N) = mem_base_phys;
  if (!use_stride) {
    if (transpose) { _OP_LOAD_NS_T(rs2);  }
    else           { _OP_LOAD_NS_NT(rs2); }
  } else {
    REG_VAR(rs1, ROCC_RS1_REG_N) = (uint64_t)stride_elems;
    if (transpose) { _OP_LOAD_S_T(rs1, rs2);  }
    else           { _OP_LOAD_S_NT(rs1, rs2); }
  }
  _fence_rw();
}

/* -------------------------- EXTRACT ------------------------- */
#define _OP_EXT_S_T(rs1, rs2)   ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1,rs2, FCTN7_EXTRACT | (1 << 2) | (1 << 3))
#define _OP_EXT_NS_T(rs2)       ROCC_INSTRUCTION_SS(OPE_CUSTOM,   0,rs2, FCTN7_EXTRACT | (1 << 2) | (0 << 3))
#define _OP_EXT_S_NT(rs1, rs2)  ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1,rs2, FCTN7_EXTRACT | (0 << 2) | (1 << 3))
#define _OP_EXT_NS_NT(rs2)      ROCC_INSTRUCTION_SS(OPE_CUSTOM,   0,rs2, FCTN7_EXTRACT | (0 << 2) | (0 << 3))

void ope_extract(uint64_t mem_base_phys, uint16_t stride_elems,
                 uint8_t transpose, uint8_t use_stride)
{
  REG_VAR(rs2, ROCC_RS2_REG_N) = mem_base_phys;

  if (!use_stride) {
    if (transpose) { _OP_EXT_NS_T(rs2);  }
    else           { _OP_EXT_NS_NT(rs2); }
    _fence_wr();
  } else {
    REG_VAR(rs1, ROCC_RS1_REG_N) = (uint64_t)stride_elems;
    if (transpose) { _OP_EXT_S_T(rs1, rs2);  }
    else           { _OP_EXT_S_NT(rs1, rs2); }
    _fence_wr();
  }
}

/* --------------------------- ACC ---------------------------- */
void ope_acc(uint64_t a_base_phys, uint64_t b_base_phys, uint8_t L_elems)
{
  if (L_elems < 1)  L_elems = 1;
  if (L_elems > 32) L_elems = 32;

  REG_VAR(rs1, ROCC_RS1_REG_N) = a_base_phys;
  REG_VAR(rs2, ROCC_RS2_REG_N) = b_base_phys;

  // Group 1: 1-4
  if (L_elems <= 4) {
    if (L_elems == 1) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (0 << 2)); }
    else if (L_elems == 2) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (1 << 2)); }
    else if (L_elems == 3) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (2 << 2)); }
    else if (L_elems == 4) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (3 << 2)); }
  }
  // Group 2: 5-8
  else if (L_elems <= 8) {
    if (L_elems == 5) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (4 << 2)); }
    else if (L_elems == 6) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (5 << 2)); }
    else if (L_elems == 7) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (6 << 2)); }
    else if (L_elems == 8) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (7 << 2)); }
  }
  // Group 3: 9-12
  else if (L_elems <= 12) {
    if (L_elems == 9) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (8 << 2)); }
    else if (L_elems == 10) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (9 << 2)); }
    else if (L_elems == 11) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (10 << 2)); }
    else if (L_elems == 12) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (11 << 2)); }
  }
  // Group 4: 13-16
  else if (L_elems <= 16) {
    if (L_elems == 13) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (12 << 2)); }
    else if (L_elems == 14) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (13 << 2)); }
    else if (L_elems == 15) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (14 << 2)); }
    else if (L_elems == 16) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (15 << 2)); }
  }
  // Group 5: 17-20
  else if (L_elems <= 20) {
    if (L_elems == 17) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (16 << 2)); }
    else if (L_elems == 18) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (17 << 2)); }
    else if (L_elems == 19) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (18 << 2)); }
    else if (L_elems == 20) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (19 << 2)); }
  }
  // Group 6: 21-24
  else if (L_elems <= 24) {
    if (L_elems == 21) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (20 << 2)); }
    else if (L_elems == 22) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (21 << 2)); }
    else if (L_elems == 23) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (22 << 2)); }
    else if (L_elems == 24) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (23 << 2)); }
  }
  // Group 7: 25-28
  else if (L_elems <= 28) {
    if (L_elems == 25) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (24 << 2)); }
    else if (L_elems == 26) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (25 << 2)); }
    else if (L_elems == 27) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (26 << 2)); }
    else if (L_elems == 28) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (27 << 2)); }
  }
  // Group 8: 29-32
  else {
    if (L_elems == 29) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (28 << 2)); }
    else if (L_elems == 30) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (29 << 2)); }
    else if (L_elems == 31) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (30 << 2)); }
    else if (L_elems == 32) { ROCC_INSTRUCTION_SS(OPE_CUSTOM, rs1, rs2, FCTN7_ACC | (31 << 2)); }
  }
}

// ===== Matmul Driver Helpers=====

#define OP_ZERO()            ROCC_INSTRUCTION(OPE_CUSTOM, FCTN7_ZERO)
#define OP_ACC(a, b)         ROCC_INSTRUCTION_SS(OPE_CUSTOM, a, b, FCTN7_ACC)
#define OP_EXTRACT_DIRECT(r) ROCC_INSTRUCTION_SS(OPE_CUSTOM, r, 0, FCTN7_EXTRACT)

__attribute__((noinline))
void ope_tile(const int8_t* A, const int8_t* B, int32_t* C,
              int i0, int j0, int K, int lda, int ldb, int ldc)
{
  // 1) Prepack K×8 panels (contiguous, no branching in the ACC loop)
  int8_t  A_panel[K * 8];
  int8_t  B_panel[K * 8];
  for (int k = 0; k < K; ++k) {
    // Pack 8 rows from column k of A (A is MxK, op computes A^T * B)
    for (int r = 0; r < 8; ++r)
      A_panel[k*8 + r] = A[(i0 + r) * lda + k];
    // Pack 8 cols from row k of B
    for (int c = 0; c < 8; ++c)
      B_panel[k*8 + c] = B[k * ldb + (j0 + c)];
  }
  // 2) Fresh tile
  ope_zero();
  // 3) Stream K in chunks of up to 32 using address-based ACC
  for (int k0 = 0; k0 < K; k0 += 32) {
    const uint8_t L = (uint8_t)((K - k0) > 32 ? 32 : (K - k0));
    const uint64_t a_chunk = (uint64_t)(A_panel + k0*8);
    const uint64_t b_chunk = (uint64_t)(B_panel + k0*8);
    ope_acc(a_chunk, b_chunk, L);
  }
  // 4) Extract the 8×8 tile directly into C with stride (elements)
  const uint64_t c_tile_base   = (uint64_t)&C[(size_t)i0 * ldc + j0];
  const uint16_t c_stride_elem = (uint16_t)ldc;
  const uint8_t  transpose_c   = 0;
  const uint8_t  use_stride    = 1;

  ope_extract(c_tile_base, c_stride_elem, transpose_c, use_stride);
}

__attribute__((noinline))
void ope_tile_buffer(const int8_t* A, const int8_t* B, int32_t* C,
              int i0, int j0, int K, int lda, int ldb, int ldc)
{
  // 1) Prepack K×8 panels (contiguous) and ensure 8-byte alignment
  __attribute__((aligned(8))) int8_t  A_panel_stack[K*8];
  __attribute__((aligned(8))) int8_t  B_panel_stack[K*8];
  int8_t* A_panel = A_panel_stack;
  int8_t* B_panel = B_panel_stack;

  for (int k = 0; k < K; ++k) {
    // Pack 8 rows of A at column k
    for (int r = 0; r < 8; ++r)
      A_panel[k*8 + r] = A[(i0 + r) * lda + k];
    // Pack 8 cols of B at row k
    for (int c = 0; c < 8; ++c)
      B_panel[k*8 + c] = B[k * ldb + (j0 + c)];
  }

  // 2) Fresh tile
  ope_zero();

  // 3) ACC in L<=32 chunks
  for (int k0 = 0; k0 < K; k0 += 32) {
    const uint8_t L = (uint8_t)((K - k0) > 32 ? 32 : (K - k0));
    const uint64_t a_chunk = (uint64_t)(A_panel + k0*8);
    const uint64_t b_chunk = (uint64_t)(B_panel + k0*8);
    ope_acc(a_chunk, b_chunk, L);
  }

  // 4) Extract to a local 8×8 with stride 8 (safe layout), then copy to C
  __attribute__((aligned(8))) int32_t out_tile[8 * 8];
  const uint64_t tmp_base      = (uint64_t)&out_tile[0];
  const uint16_t tmp_stride_el = 8;
  const uint8_t  transpose_c   = 0, use_stride = 1;
  ope_extract(tmp_base, tmp_stride_el, transpose_c, use_stride);

  // Copy 8×8 block into C (row-major)
  for (int r = 0; r < 8; ++r) {
    int32_t* dst = &C[(size_t)(i0 + r) * ldc + j0];
    const int32_t* src = &out_tile[r * 8];
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    dst[4] = src[4]; dst[5] = src[5]; dst[6] = src[6]; dst[7] = src[7];
  }
}

__attribute__((noinline))
void ope_tile_partial(const int8_t* A, const int8_t* B, int32_t* C,
                      int i0, int j0, int i_size, int j_size,
                      int K, int lda, int ldb, int ldc)
{
  // 1) Prepack K×8 panels with zero-padding beyond valid rows/cols
  int8_t  A_panel[K * 8];
  int8_t  B_panel[K * 8];
  for (int k = 0; k < K; ++k) {
    // Rows of A
    int r = 0;
    for (; r < i_size; ++r) A_panel[k*8 + r] = A[(i0 + r) * lda + k];
    for (; r < 8;      ++r) A_panel[k*8 + r] = 0;
    // Cols of B
    int c = 0;
    for (; c < j_size; ++c) B_panel[k*8 + c] = B[k * ldb + (j0 + c)];
    for (; c < 8;      ++c) B_panel[k*8 + c] = 0;
  }
  // 2) Fresh tile
  ope_zero();
  // 3) ACC in L<=32 chunks
  for (int k0 = 0; k0 < K; k0 += 32) {
    const uint8_t L = (uint8_t)((K - k0) > 32 ? 32 : (K - k0));
    const uint64_t a_chunk = (uint64_t)(A_panel + k0*8);
    const uint64_t b_chunk = (uint64_t)(B_panel + k0*8);
    ope_acc(a_chunk, b_chunk, L);
  }
  // 4) Extract to a local 8×8 buffer (stride 8), then copy only i_size×j_size
  int32_t out_tile[8 * 8];
  const uint64_t tmp_base      = (uint64_t)&out_tile[0];
  const uint16_t tmp_stride_el = 8;
  const uint8_t  transpose_c   = 0;
  const uint8_t  use_stride    = 1;
  ope_extract(tmp_base, tmp_stride_el, transpose_c, use_stride);
  // Scatter valid region into C
  for (int r = 0; r < i_size; ++r) {
    const int32_t* src = &out_tile[r * 8];
    int32_t*       dst = &C[(size_t)(i0 + r) * ldc + j0];
    for (int c = 0; c < j_size; ++c) {
      dst[c] = src[c];
    }
  }
}

// ===== Matmul Driver Functions =====

__attribute__((noinline))
void ope_matmul_m8m8(const int8_t* A, const int8_t* B, int32_t* C,
                int M, int N, int K, int lda, int ldb, int ldc)
{
  const int rem_M = M & 7;
  const int rem_N = N & 7;

  for (int i = 0; i < M; i += 8) {
    for (int j = 0; j < N; j += 8) {
      ope_tile(A, B, C, i, j, K, lda, ldb, ldc);
    }
  }
}

__attribute__((noinline))
void ope_matmul(const int8_t* A, const int8_t* B, int32_t* C,
                int M, int N, int K, int lda, int ldb, int ldc)
{
  const int full_M = M & ~7;
  const int full_N = N & ~7;
  const int rem_M  = M & 7;
  const int rem_N  = N & 7;

  // Core 8×8 blocks
  for (int i = 0; i < full_M; i += 8) {
    for (int j = 0; j < full_N; j += 8) {
      ope_tile_buffer(A, B, C, i, j, K, lda, ldb, ldc);
    }
  }
  // Right edge (width = rem_N)
  if (rem_N) {
    for (int i = 0; i < full_M; i += 8) {
      ope_tile_partial(A, B, C, i, full_N, 8, rem_N, K, lda, ldb, ldc);
    }
  }
  // Bottom edge (height = rem_M)
  if (rem_M) {
    for (int j = 0; j < full_N; j += 8) {
      ope_tile_partial(A, B, C, full_M, j, rem_M, 8, K, lda, ldb, ldc);
    }
  }
  // Bottom-right corner (rem_M × rem_N)
  if (rem_M && rem_N) {
    ope_tile_partial(A, B, C, full_M, full_N, rem_M, rem_N, K, lda, ldb, ldc);
  }
}