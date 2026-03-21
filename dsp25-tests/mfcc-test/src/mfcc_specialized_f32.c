#include "mfcc_specialized.h"

#include "dsp/basic_math_functions.h"
#include "dsp/complex_math_functions.h"
#include "dsp/fast_math_functions.h"
#include "dsp/statistics_functions.h"
#include "dsp/transform_functions.h"

static void mfcc_tinyspeech_apply_mel_f32(const riscv_mfcc_instance_f32 *S,
                                          const float32_t *spectrum,
                                          float32_t *mel_out)
{
  const float32_t *coef = S->filterCoefs;

  for (uint32_t i = 0; i < MFCC_TINYSPEECH_NUM_MEL; i++) {
    const float32_t *spec = spectrum + S->filterPos[i];
    uint32_t len = S->filterLengths[i];
    float32_t acc0 = 0.0f;
    float32_t acc1 = 0.0f;
    float32_t acc2 = 0.0f;
    float32_t acc3 = 0.0f;
    uint32_t n = 0;

    for (; (n + 3U) < len; n += 4U) {
      acc0 += spec[n] * coef[n];
      acc1 += spec[n + 1U] * coef[n + 1U];
      acc2 += spec[n + 2U] * coef[n + 2U];
      acc3 += spec[n + 3U] * coef[n + 3U];
    }

    {
      float32_t acc = (acc0 + acc1) + (acc2 + acc3);
      for (; n < len; n++) {
        acc += spec[n] * coef[n];
      }
      mel_out[i] = acc;
    }

    coef += len;
  }
}

static void mfcc_tinyspeech_apply_dct_f32(const riscv_mfcc_instance_f32 *S,
                                          const float32_t *mel,
                                          float32_t *out)
{
  const float32_t *dct = S->dctCoefs;

  for (uint32_t row = 0; row < MFCC_TINYSPEECH_NUM_DCT; row++) {
    const float32_t *d = dct + (row * MFCC_TINYSPEECH_NUM_MEL);
    float32_t acc0 = 0.0f;
    float32_t acc1 = 0.0f;
    float32_t acc2 = 0.0f;
    float32_t acc3 = 0.0f;
    uint32_t n = 0;

    for (; (n + 3U) < MFCC_TINYSPEECH_NUM_MEL; n += 4U) {
      acc0 += d[n] * mel[n];
      acc1 += d[n + 1U] * mel[n + 1U];
      acc2 += d[n + 2U] * mel[n + 2U];
      acc3 += d[n + 3U] * mel[n + 3U];
    }

    {
      float32_t acc = (acc0 + acc1) + (acc2 + acc3);
      for (; n < MFCC_TINYSPEECH_NUM_MEL; n++) {
        acc += d[n] * mel[n];
      }
      out[row] = acc;
    }
  }
}

void mfcc_tinyspeech_256_23_12_f32(const riscv_mfcc_instance_f32 *S,
                                   float32_t *pSrc,
                                   float32_t *pDst,
                                   float32_t *pTmp)
{
  float32_t maxValue;
  uint32_t index;

  if ((S->fftLen != MFCC_TINYSPEECH_FFT_LEN) ||
      (S->nbMelFilters != MFCC_TINYSPEECH_NUM_MEL) ||
      (S->nbDctOutputs != MFCC_TINYSPEECH_NUM_DCT)) {
    riscv_mfcc_f32(S, pSrc, pDst, pTmp);
    return;
  }

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
