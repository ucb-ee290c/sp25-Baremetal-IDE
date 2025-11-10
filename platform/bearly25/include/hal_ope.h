#ifndef HAL_OPE_H
#define HAL_OPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
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

// Compute one full 8x8 tile of C = A^T * B (no partial tiles)
void ope_tile(const int8_t* A, const int8_t* B, int32_t* C,
              int i0, int j0, int K, int lda, int ldb, int ldc);

// Compute one full 8x8 tile of C = A^T * B but with a temporary buffer for unaligned matrices
void ope_tile_buffer(const int8_t* A, const int8_t* B, int32_t* C,
                     int i0, int j0, int K, int lda, int ldb, int ldc);

// Compute a partial tile (for edges, when M/N not multiple of 8)
void ope_tile_partial(const int8_t* A, const int8_t* B, int32_t* C,
                      int i0, int j0, int i_size, int j_size,
                      int K, int lda, int ldb, int ldc);

// Top-level tiled multiply using OPE RoCC Accelerator
void ope_matmul_m8m8(const int8_t* A, const int8_t* B, int32_t* C,
                     int M, int N, int K, int lda, int ldb, int ldc);

void ope_matmul(const int8_t* A, const int8_t* B, int32_t* C,
                int M, int N, int K, int lda, int ldb, int ldc);                 

#ifdef __cplusplus
}
#endif

#endif // HAL_OPE_H
