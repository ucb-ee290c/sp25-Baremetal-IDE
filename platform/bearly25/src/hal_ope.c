#include "hal_ope.h"
#include <stddef.h>
#include "rocc.h"

#define REG_STR_HELPER(x) #x
#define REG_STR(x)        REG_STR_HELPER(x)
#define REG_VAR(name, reg_num) register uint64_t name asm("x" REG_STR(reg_num))

#define ROCC_RS1_REG_N 11
#define ROCC_RS2_REG_N 12

// funct7 tags
// funct[1:0] selects op-class, funct[6:2] carries parameters
#define FCTN7_ACC     0b00
#define FCTN7_EXTRACT 0b01
#define FCTN7_ZERO    0b10
#define FCTN7_LOAD    0b11

// ===== Fences =====
static inline void _fence_all(void){ asm volatile("fence iorw, iorw" ::: "memory"); }
static inline void _fence_wr(void) { asm volatile("fence w, r"      ::: "memory"); }
static inline void _fence_rw(void) { asm volatile("fence r, w"      ::: "memory"); }

void ope_fence(void) { _fence_all(); }

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

/* --------------------- Tile + Matmul helpers ---------------- */

// One 8x8 tile step
void ope_tile_outer_product(uint64_t a_tile_phys,
                            uint64_t b_tile_phys,
                            uint64_t c_tile_phys,
                            uint8_t  L_iters8,
                            uint16_t stride_elems,
                            uint8_t  transpose,
                            uint8_t  use_stride,
                            bool     load_existing)
{
  if (load_existing) {
    ope_load(c_tile_phys, stride_elems, transpose, use_stride);
  } else {
    ope_zero();
  }
  ope_acc(a_tile_phys, b_tile_phys, L_iters8);
  ope_extract(c_tile_phys, stride_elems, transpose, use_stride);
}

// Multiply an M×K matrix A^T by a K×N matrix B, accumulate into an M×N C
// Precondition: M, N, K are multiples of 8
void ope_matmul_i8i8_i32_AT(const int8_t* AT_phys, int ldAT,
                            const int8_t* B_phys,  int ldb,
                            int32_t*      C_phys,  int ldc,
                            int M, int N, int K,
                            bool load_existing)
{
  const uint16_t c_stride = (uint16_t)ldc;
  const uint8_t  use_stride_c = 1;
  const uint8_t  transpose_c  = 0;

  for (int i0 = 0; i0 < M; i0 += 8) {
    for (int j0 = 0; j0 < N; j0 += 8) {

      // Base pointers for the (i0, j0) 8×8 tile
      const uint64_t c_tile = (uint64_t)(C_phys  + i0 + (size_t)j0 * ldc);
      const uint64_t a_tile = (uint64_t)(AT_phys + i0 * (size_t)ldAT);
      const uint64_t b_tile = (uint64_t)(B_phys  + j0);

      // Bring C tile into accumulators (or zero them)
      if (load_existing) {
        ope_load(c_tile, c_stride, transpose_c, use_stride_c);
      } else {
        ope_zero();
      }

      // Accumulate over K using as large an L as possible (<= 32)
      int remaining_k8 = K / 8;
      uint64_t a_step  = a_tile;
      uint64_t b_step  = b_tile;

      while (remaining_k8 > 0) {
        uint8_t L = (remaining_k8 > 32) ? 32 : (uint8_t)remaining_k8;
        ope_acc(a_step, b_step, L);

        a_step += (uint64_t)8 * L;          // advance A^T by L lanes
        b_step += (uint64_t)ldb * 8 * L;    // advance B by L rows (each 8-wide)
        remaining_k8 -= L;
      }
      ope_extract(c_tile, c_stride, transpose_c, use_stride_c);
    }
  }
}