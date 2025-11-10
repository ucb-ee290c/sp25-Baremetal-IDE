#ifndef HAL_OPE_H
#define HAL_OPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "chip_config.h"
#include "hal_ope.h"

#ifndef OPE_MAX_L
#define OPE_MAX_L 32
#endif

#ifndef OPE_DEFAULT_BUFFERED_EXTRACT
#define OPE_DEFAULT_BUFFERED_EXTRACT 1
#endif

typedef struct {
  int M, N, K;
  int itiles, jtiles;
  int8_t* A_pack;
  int8_t* B_pack;
} ope_pack_plan;

size_t ope_pack_workspace_size(int M, int N, int K);
int    ope_pack_plan_init(ope_pack_plan* p,
                          int M, int N, int K,
                          void* workspace, size_t workspace_bytes);

void   ope_pack_all_A(const int8_t* A, int lda, const ope_pack_plan* p);
void   ope_pack_all_B(const int8_t* B, int ldb, const ope_pack_plan* p);

void ope_matmul_prepacked(const int8_t* A, const int8_t* B, int32_t* C,
                          int M, int N, int K, int lda, int ldb, int ldc,
                          void* pack_workspace, size_t pack_workspace_bytes,
                          int buffered_full_tiles);                     

#ifdef __cplusplus
}
#endif

#endif // HAL_OPE_H
