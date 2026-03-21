#include "mfcc_specialized.h"

#include "dsp/basic_math_functions.h"
#include "dsp/complex_math_functions.h"
#include "dsp/fast_math_functions.h"
#include "dsp/matrix_functions.h"
#include "dsp/statistics_functions.h"
#include "dsp/transform_functions.h"

#ifndef MFCC_SPECIALIZED_RUNTIME_CHECK
#define MFCC_SPECIALIZED_RUNTIME_CHECK 0
#endif

static __STATIC_FORCEINLINE void mfcc_tinyspeech_apply_mel_f32(const riscv_mfcc_instance_f32 *S,
                                                                const float32_t *spectrum,
                                                                float32_t *mel_out)
{
  const float32_t *coef = S->filterCoefs;

  for (uint32_t i = 0; i < MFCC_TINYSPEECH_NUM_MEL; i++) {
    float32_t result;
    uint32_t len = S->filterLengths[i];
    riscv_dot_prod_f32(spectrum + S->filterPos[i], coef, len, &result);
    mel_out[i] = result;
    coef += len;
  }
}

static __STATIC_FORCEINLINE void mfcc_tinyspeech_apply_dct_f32(const riscv_mfcc_instance_f32 *S,
                                                                const float32_t *mel,
                                                                float32_t *out)
{
  riscv_matrix_instance_f32 pDctMat;
  pDctMat.numRows = MFCC_TINYSPEECH_NUM_DCT;
  pDctMat.numCols = MFCC_TINYSPEECH_NUM_MEL;
  pDctMat.pData = (float32_t *)S->dctCoefs;
  riscv_mat_vec_mult_f32(&pDctMat, mel, out);
}

void mfcc_tinyspeech_256_23_12_f32(const riscv_mfcc_instance_f32 *S,
                                   float32_t *pSrc,
                                   float32_t *pDst,
                                   float32_t *pTmp)
{
  float32_t maxValue;
  uint32_t index;

#if (MFCC_SPECIALIZED_RUNTIME_CHECK == 1)
  if ((S->fftLen != MFCC_TINYSPEECH_FFT_LEN) ||
      (S->nbMelFilters != MFCC_TINYSPEECH_NUM_MEL) ||
      (S->nbDctOutputs != MFCC_TINYSPEECH_NUM_DCT)) {
    riscv_mfcc_f32(S, pSrc, pDst, pTmp);
    return;
  }
#endif

  riscv_absmax_f32(pSrc, S->fftLen, &maxValue, &index);

  if (maxValue != 0.0f) {
    riscv_scale_f32(pSrc, 1.0f / maxValue, pSrc, S->fftLen);
  }

  riscv_mult_f32(pSrc, S->windowCoefs, pSrc, S->fftLen);

#if defined(RISCV_MFCC_CFFT_BASED)
  for (uint32_t i = 0; i < S->fftLen; i++) {
    pTmp[2U * i] = pSrc[i];
    pTmp[(2U * i) + 1U] = 0.0f;
  }
  riscv_cfft_f32(&(S->cfft), pTmp, 0, 1);
#else
  riscv_rfft_fast_f32(&(S->rfft), pSrc, pTmp, 0);
  pTmp[S->fftLen] = pTmp[1];
  pTmp[S->fftLen + 1U] = 0.0f;
  pTmp[1] = 0.0f;
#endif

  riscv_cmplx_mag_f32(pTmp, pSrc, S->fftLen);

  if (maxValue != 0.0f) {
    riscv_scale_f32(pSrc, maxValue, pSrc, S->fftLen);
  }

  mfcc_tinyspeech_apply_mel_f32(S, pSrc, pTmp);

  riscv_offset_f32(pTmp, 1.0e-6f, pTmp, MFCC_TINYSPEECH_NUM_MEL);
  riscv_vlog_f32(pTmp, pTmp, MFCC_TINYSPEECH_NUM_MEL);

  mfcc_tinyspeech_apply_dct_f32(S, pTmp, pDst);
}
