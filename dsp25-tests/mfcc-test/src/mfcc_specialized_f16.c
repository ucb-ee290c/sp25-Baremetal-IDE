#include "mfcc_specialized.h"

#if defined(RISCV_FLOAT16_SUPPORTED)

#include "dsp/basic_math_functions_f16.h"
#include "dsp/complex_math_functions_f16.h"
#include "dsp/fast_math_functions_f16.h"
#include "dsp/matrix_functions_f16.h"
#include "dsp/statistics_functions_f16.h"
#include "dsp/transform_functions_f16.h"

#ifndef MFCC_SPECIALIZED_RUNTIME_CHECK
#define MFCC_SPECIALIZED_RUNTIME_CHECK 0
#endif

#ifndef MFCC_F16_ACCUM_MODE
#define MFCC_F16_ACCUM_MODE 0
#endif

__STATIC_FORCEINLINE void mfcc_tinyspeech_apply_mel_f16(const riscv_mfcc_instance_f16 *S,
                                                         const float16_t *spectrum,
                                                         float16_t *mel_out)
{
  const float16_t *coef = S->filterCoefs;

  for (uint32_t i = 0; i < MFCC_TINYSPEECH_NUM_MEL; i++) {
    const float16_t *pMel = spectrum + S->filterPos[i];
    const float16_t *pCoef = coef;
    uint32_t len = S->filterLengths[i];
    float16_t result;
#if (MFCC_F16_ACCUM_MODE == 1)
    float32_t acc32 = 0.0f;
    for (uint32_t n = 0; n < len; n++) {
      acc32 += (float32_t)pMel[n] * (float32_t)pCoef[n];
    }
    result = (float16_t)acc32;
#else
  #if defined(RISCV_MATH_VECTOR_F16)
    if (len <= 16U) {
      uint32_t n = len;
      result = 0.0f16;
      while (n > 0U) {
        result += (_Float16)(*pMel++) * (_Float16)(*pCoef++);
        n--;
      }
    } else
  #endif
    {
      riscv_dot_prod_f16(pMel, pCoef, len, &result);
    }
#endif

    coef += len;
    mel_out[i] = result;
  }
}

__STATIC_FORCEINLINE void mfcc_tinyspeech_apply_dct_f16(const riscv_mfcc_instance_f16 *S,
                                                         const float16_t *mel,
                                                         float16_t *out)
{
#if (MFCC_F16_ACCUM_MODE == 1)
  for (uint32_t i = 0; i < MFCC_TINYSPEECH_NUM_DCT; i++) {
    const float16_t *pDct = S->dctCoefs + (i * MFCC_TINYSPEECH_NUM_MEL);
    float32_t acc32 = 0.0f;
    for (uint32_t j = 0; j < MFCC_TINYSPEECH_NUM_MEL; j++) {
      acc32 += (float32_t)pDct[j] * (float32_t)mel[j];
    }
    out[i] = (float16_t)acc32;
  }
#else
  riscv_matrix_instance_f16 pDctMat;
  pDctMat.numRows = MFCC_TINYSPEECH_NUM_DCT;
  pDctMat.numCols = MFCC_TINYSPEECH_NUM_MEL;
  pDctMat.pData = (float16_t *)S->dctCoefs;
  riscv_mat_vec_mult_f16(&pDctMat, mel, out);
#endif
}

void mfcc_tinyspeech_1024_23_12_f16(const riscv_mfcc_instance_f16 *S,
                                   float16_t *pSrc,
                                   float16_t *pDst,
                                   float16_t *pTmp)
{
  float16_t maxValue;
  uint32_t index;

#if (MFCC_SPECIALIZED_RUNTIME_CHECK == 1)
  if ((S->fftLen != MFCC_TINYSPEECH_FFT_LEN) ||
      (S->nbMelFilters != MFCC_TINYSPEECH_NUM_MEL) ||
      (S->nbDctOutputs != MFCC_TINYSPEECH_NUM_DCT)) {
    riscv_mfcc_f16(S, pSrc, pDst, pTmp);
    return;
  }
#endif

  riscv_absmax_f16(pSrc, S->fftLen, &maxValue, &index);

  if ((_Float16)maxValue != 0.0f16) {
    riscv_scale_f16(pSrc, 1.0f16 / (_Float16)maxValue, pSrc, S->fftLen);
  }

  riscv_mult_f16(pSrc, S->windowCoefs, pSrc, S->fftLen);

#if defined(RISCV_MFCC_CFFT_BASED)
  for (uint32_t i = 0; i < S->fftLen; i++) {
    pTmp[2U * i] = pSrc[i];
    pTmp[(2U * i) + 1U] = 0.0f16;
  }
  riscv_cfft_f16(&(S->cfft), pTmp, 0, 1);
#else
  riscv_rfft_fast_f16(&(S->rfft), pSrc, pTmp, 0);
  pTmp[S->fftLen] = pTmp[1];
  pTmp[S->fftLen + 1U] = 0.0f16;
  pTmp[1] = 0.0f16;
#endif

  riscv_cmplx_mag_f16(pTmp, pSrc, S->fftLen);

  if ((_Float16)maxValue != 0.0f16) {
    riscv_scale_f16(pSrc, maxValue, pSrc, S->fftLen);
  }

  mfcc_tinyspeech_apply_mel_f16(S, pSrc, pTmp);

  riscv_offset_f16(pTmp, 1.0e-4f16, pTmp, MFCC_TINYSPEECH_NUM_MEL);
  riscv_vlog_f16(pTmp, pTmp, MFCC_TINYSPEECH_NUM_MEL);

  mfcc_tinyspeech_apply_dct_f16(S, pTmp, pDst);
}

#endif
