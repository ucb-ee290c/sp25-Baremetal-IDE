#ifndef HAL_OPE_H
#define HAL_OPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "chip_config.h"

#ifndef OPE_CUSTOM
#define OPE_CUSTOM 0
#endif

// Orders all CPU memory and I/O operations
void ope_fence(void);

// Clears the accelerator’s internal accumulators to zero.
void ope_zero(void);

// Load an 8x8 int32 tile from memory into the accelerator
void ope_load(uint64_t mem_base_phys, uint16_t stride_elems,
              uint8_t transpose, uint8_t use_stride);

// Store the current 8x8 int32 accumulator tile to memory
void ope_extract(uint64_t mem_base_phys, uint16_t stride_elems,
                 uint8_t transpose, uint8_t use_stride);

// Outer-product accumulate of length L_elems in [1..32] using int8* A and B
void ope_acc(uint64_t a_base_phys, uint64_t b_base_phys, uint8_t L_elems);

// One 8x8 tile step of GEMM/conv
void ope_tile_outer_product(uint64_t a_tile_phys,
                            uint64_t b_tile_phys,
                            uint64_t c_tile_phys,
                            uint8_t  L_iters8,
                            uint16_t stride_elems,
                            uint8_t  transpose,
                            uint8_t  use_stride,
                            bool     load_existing);

// Top-level tiled GEMM: C += A^T * B (A is provided as AT).
void ope_matmul_i8i8_i32_AT(const int8_t* AT_phys, int ldAT,
                            const int8_t* B_phys,  int ldb,
                            int32_t*      C_phys,  int ldc,
                            int M, int N, int K,
                            bool load_existing);

#ifdef __cplusplus
}
#endif

#endif // HAL_OPE_H
