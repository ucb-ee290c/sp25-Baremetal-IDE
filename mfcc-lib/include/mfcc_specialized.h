#ifndef MFCC_SPECIALIZED_H
#define MFCC_SPECIALIZED_H

#ifdef __cplusplus
extern "C" {
#endif

#include "riscv_math.h"

#define MFCC_TINYSPEECH_FFT_LEN 1024U
#define MFCC_TINYSPEECH_NUM_MEL 23U
#define MFCC_TINYSPEECH_NUM_DCT 12U

void mfcc_tinyspeech_1024_23_12_f32(const riscv_mfcc_instance_f32 *S,
                                   float32_t *pSrc,
                                   float32_t *pDst,
                                   float32_t *pTmp);

#if defined(RISCV_FLOAT16_SUPPORTED)
void mfcc_tinyspeech_1024_23_12_f16(const riscv_mfcc_instance_f16 *S,
                                   float16_t *pSrc,
                                   float16_t *pDst,
                                   float16_t *pTmp);
#endif

#ifdef __cplusplus
}
#endif

#endif
