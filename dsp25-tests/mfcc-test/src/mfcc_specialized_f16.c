#include "mfcc_specialized.h"

#if defined(RISCV_FLOAT16_SUPPORTED)

#include "dsp/basic_math_functions_f16.h"
#include "dsp/complex_math_functions_f16.h"
#include "dsp/fast_math_functions_f16.h"
#include "dsp/statistics_functions_f16.h"
#include "dsp/transform_functions_f16.h"

#ifndef MFCC_F16_ACCUM_MODE
#define MFCC_F16_ACCUM_MODE 0
#endif

static void mfcc_tinyspeech_apply_mel_f16(const riscv_mfcc_instance_f16 *S,
                                          const float16_t *spectrum,
                                          float16_t *mel_out)
{
  const float16_t *coef = S->filterCoefs;

  for (uint32_t i = 0; i < MFCC_TINYSPEECH_NUM_MEL; i++) {
    const float16_t *spec = spectrum + S->filterPos[i];
    uint32_t len = S->filterLengths[i];
#if (MFCC_F16_ACCUM_MODE == 1)
    float32_t acc0 = 0.0f;
    float32_t acc1 = 0.0f;
    float32_t acc2 = 0.0f;
    float32_t acc3 = 0.0f;
#else
    float16_t acc0 = 0.0f16;
    float16_t acc1 = 0.0f16;
    float16_t acc2 = 0.0f16;
    float16_t acc3 = 0.0f16;
#endif
    uint32_t n = 0;

    for (; (n + 3U) < len; n += 4U) {
#if (MFCC_F16_ACCUM_MODE == 1)
      acc0 += (float32_t)spec[n] * (float32_t)coef[n];
      acc1 += (float32_t)spec[n + 1U] * (float32_t)coef[n + 1U];
      acc2 += (float32_t)spec[n + 2U] * (float32_t)coef[n + 2U];
      acc3 += (float32_t)spec[n + 3U] * (float32_t)coef[n + 3U];
#else
      acc0 += (_Float16)spec[n] * (_Float16)coef[n];
      acc1 += (_Float16)spec[n + 1U] * (_Float16)coef[n + 1U];
      acc2 += (_Float16)spec[n + 2U] * (_Float16)coef[n + 2U];
      acc3 += (_Float16)spec[n + 3U] * (_Float16)coef[n + 3U];
#endif
    }

#if (MFCC_F16_ACCUM_MODE == 1)
    {
      float32_t acc = (acc0 + acc1) + (acc2 + acc3);
      for (; n < len; n++) {
        acc += (float32_t)spec[n] * (float32_t)coef[n];
      }
      mel_out[i] = (float16_t)acc;
    }
#else
    {
      float16_t acc = (float16_t)((_Float16)(acc0 + acc1) + (_Float16)(acc2 + acc3));
      for (; n < len; n++) {
        acc += (_Float16)spec[n] * (_Float16)coef[n];
      }
      mel_out[i] = acc;
    }
#endif

    coef += len;
  }
}

static void mfcc_tinyspeech_apply_dct_f16(const riscv_mfcc_instance_f16 *S,
                                          const float16_t *mel,
                                          float16_t *out)
{
#if defined(RISCV_MATH_VECTOR_F16)
  uint32_t rows_remaining = MFCC_TINYSPEECH_NUM_DCT;
  const float16_t *dct_base = S->dctCoefs;
  float16_t *dst = out;
  const ptrdiff_t stride = (ptrdiff_t)(MFCC_TINYSPEECH_NUM_MEL * sizeof(float16_t));

  while (rows_remaining > 0U) {
    size_t l = __riscv_vsetvl_e16m8(rows_remaining);
    vfloat16m8_t acc = __riscv_vfmv_v_f_f16m8(0.0f, l);

    for (uint32_t c = 0; c < MFCC_TINYSPEECH_NUM_MEL; c++) {
      vfloat16m8_t vc = __riscv_vlse16_v_f16m8(dct_base + c, stride, l);
      acc = __riscv_vfmacc_vf_f16m8(acc, mel[c], vc, l);
    }

    __riscv_vse16_v_f16m8(dst, acc, l);
    dst += l;
    dct_base += l * MFCC_TINYSPEECH_NUM_MEL;
    rows_remaining -= (uint32_t)l;
  }
#else
  const float16_t *dct = S->dctCoefs;

  for (uint32_t row = 0; row < MFCC_TINYSPEECH_NUM_DCT; row++) {
    const float16_t *d = dct + (row * MFCC_TINYSPEECH_NUM_MEL);
#if (MFCC_F16_ACCUM_MODE == 1)
    float32_t acc0 = 0.0f;
    float32_t acc1 = 0.0f;
    float32_t acc2 = 0.0f;
    float32_t acc3 = 0.0f;
#else
    float16_t acc0 = 0.0f16;
    float16_t acc1 = 0.0f16;
    float16_t acc2 = 0.0f16;
    float16_t acc3 = 0.0f16;
#endif
    uint32_t n = 0;

    for (; (n + 3U) < MFCC_TINYSPEECH_NUM_MEL; n += 4U) {
#if (MFCC_F16_ACCUM_MODE == 1)
      acc0 += (float32_t)d[n] * (float32_t)mel[n];
      acc1 += (float32_t)d[n + 1U] * (float32_t)mel[n + 1U];
      acc2 += (float32_t)d[n + 2U] * (float32_t)mel[n + 2U];
      acc3 += (float32_t)d[n + 3U] * (float32_t)mel[n + 3U];
#else
      acc0 += (_Float16)d[n] * (_Float16)mel[n];
      acc1 += (_Float16)d[n + 1U] * (_Float16)mel[n + 1U];
      acc2 += (_Float16)d[n + 2U] * (_Float16)mel[n + 2U];
      acc3 += (_Float16)d[n + 3U] * (_Float16)mel[n + 3U];
#endif
    }

#if (MFCC_F16_ACCUM_MODE == 1)
    {
      float32_t acc = (acc0 + acc1) + (acc2 + acc3);
      for (; n < MFCC_TINYSPEECH_NUM_MEL; n++) {
        acc += (float32_t)d[n] * (float32_t)mel[n];
      }
      out[row] = (float16_t)acc;
    }
#else
    {
      float16_t acc = (float16_t)((_Float16)(acc0 + acc1) + (_Float16)(acc2 + acc3));
      for (; n < MFCC_TINYSPEECH_NUM_MEL; n++) {
        acc += (_Float16)d[n] * (_Float16)mel[n];
      }
      out[row] = acc;
    }
#endif
  }
#endif
}

void mfcc_tinyspeech_256_23_12_f16(const riscv_mfcc_instance_f16 *S,
                                   float16_t *pSrc,
                                   float16_t *pDst,
                                   float16_t *pTmp)
{
  float16_t maxValue;
  uint32_t index;

  if ((S->fftLen != MFCC_TINYSPEECH_FFT_LEN) ||
      (S->nbMelFilters != MFCC_TINYSPEECH_NUM_MEL) ||
      (S->nbDctOutputs != MFCC_TINYSPEECH_NUM_DCT)) {
    riscv_mfcc_f16(S, pSrc, pDst, pTmp);
    return;
  }

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
