#include "mfcc_driver_mc.h"

#include "dsp/basic_math_functions.h"
#include "dsp/complex_math_functions.h"
#include "dsp/fast_math_functions.h"
#include "dsp/statistics_functions.h"
#include "dsp/transform_functions.h"
#if defined(RISCV_FLOAT16_SUPPORTED)
#include "dsp/basic_math_functions_f16.h"
#include "dsp/complex_math_functions_f16.h"
#include "dsp/fast_math_functions_f16.h"
#include "dsp/statistics_functions_f16.h"
#include "dsp/transform_functions_f16.h"
#endif

void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);

typedef struct {
  const riscv_mfcc_instance_f32 *S;
  const float32_t *spectrum;
  float32_t *mel_out;
  uint32_t mel_start;
  uint32_t mel_end;
  uint32_t coef_offset;
} mfcc_mc_mel_job_f32_t;

typedef struct {
  const riscv_mfcc_instance_f32 *S;
  const float32_t *mel;
  float32_t *out;
  uint32_t dct_start;
  uint32_t dct_end;
} mfcc_mc_dct_job_f32_t;

#if defined(RISCV_FLOAT16_SUPPORTED)
typedef struct {
  const riscv_mfcc_instance_f16 *S;
  const float16_t *spectrum;
  float16_t *mel_out;
  uint32_t mel_start;
  uint32_t mel_end;
  uint32_t coef_offset;
} mfcc_mc_mel_job_f16_t;

typedef struct {
  const riscv_mfcc_instance_f16 *S;
  const float16_t *mel;
  float16_t *out;
  uint32_t dct_start;
  uint32_t dct_end;
} mfcc_mc_dct_job_f16_t;
#endif

static inline uint32_t mfcc_mc_hart_id(void) {
  uint64_t x;
  asm volatile("csrr %0, mhartid" : "=r"(x));
  return (uint32_t)x;
}

static inline uint64_t mfcc_mc_rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

static uint32_t mfcc_mc_coef_offset(const uint32_t *lengths, uint32_t start) {
  uint32_t offset = 0U;
  for (uint32_t i = 0; i < start; i++) {
    offset += lengths[i];
  }
  return offset;
}

static void mfcc_mc_mel_worker_f32(void *arg) {
  mfcc_mc_mel_job_f32_t *job = (mfcc_mc_mel_job_f32_t *)arg;
  const float32_t *coef = job->S->filterCoefs + job->coef_offset;

  for (uint32_t i = job->mel_start; i < job->mel_end; i++) {
    const uint32_t len = job->S->filterLengths[i];
    float32_t result = 0.0f;
    riscv_dot_prod_f32(job->spectrum + job->S->filterPos[i], coef, len, &result);
    job->mel_out[i] = result;
    coef += len;
  }
}

static void mfcc_mc_dct_worker_f32(void *arg) {
  mfcc_mc_dct_job_f32_t *job = (mfcc_mc_dct_job_f32_t *)arg;
  for (uint32_t r = job->dct_start; r < job->dct_end; r++) {
    const float32_t *row = job->S->dctCoefs + (r * MFCC_TINYSPEECH_NUM_MEL);
    float32_t result = 0.0f;
    riscv_dot_prod_f32(row, job->mel, MFCC_TINYSPEECH_NUM_MEL, &result);
    job->out[r] = result;
  }
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static void mfcc_mc_mel_worker_f16(void *arg) {
  mfcc_mc_mel_job_f16_t *job = (mfcc_mc_mel_job_f16_t *)arg;
  const float16_t *coef = job->S->filterCoefs + job->coef_offset;

  for (uint32_t i = job->mel_start; i < job->mel_end; i++) {
    const uint32_t len = job->S->filterLengths[i];
    float16_t result = 0.0f16;
    riscv_dot_prod_f16(job->spectrum + job->S->filterPos[i], coef, len, &result);
    job->mel_out[i] = result;
    coef += len;
  }
}

static void mfcc_mc_dct_worker_f16(void *arg) {
  mfcc_mc_dct_job_f16_t *job = (mfcc_mc_dct_job_f16_t *)arg;
  for (uint32_t r = job->dct_start; r < job->dct_end; r++) {
    const float16_t *row = job->S->dctCoefs + (r * MFCC_TINYSPEECH_NUM_MEL);
    float16_t result = 0.0f16;
    riscv_dot_prod_f16(row, job->mel, MFCC_TINYSPEECH_NUM_MEL, &result);
    job->out[r] = result;
  }
}
#endif

mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f32_mc(mfcc_driver_t *ctx,
                                                          const float32_t *input,
                                                          float32_t *output,
                                                          uint64_t *cycles) {
  const riscv_mfcc_instance_f32 *S;
  float32_t *pSrc;
  float32_t *pTmp;
  float32_t maxValue = 0.0f;
  uint32_t index = 0U;
  uint64_t t0 = 0U;
  uint64_t t1 = 0U;
  const uint32_t mel_mid = (MFCC_TINYSPEECH_NUM_MEL + 1U) / 2U;
  const uint32_t dct_mid = (MFCC_TINYSPEECH_NUM_DCT + 1U) / 2U;
  mfcc_mc_mel_job_f32_t mel_h1;
  mfcc_mc_mel_job_f32_t mel_h0;
  mfcc_mc_dct_job_f32_t dct_h1;
  mfcc_mc_dct_job_f32_t dct_h0;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  if (mfcc_mc_hart_id() != 0U) {
    return mfcc_driver_run_sp1024x23x12_f32(ctx, input, output, cycles);
  }

  S = &ctx->mfcc_f32;
  pSrc = ctx->input_f32;
  pTmp = ctx->tmp_f32;
  memcpy(pSrc, input, sizeof(ctx->input_f32));

  t0 = mfcc_mc_rdcycle64();
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

  mel_h1.S = S;
  mel_h1.spectrum = pSrc;
  mel_h1.mel_out = pTmp;
  mel_h1.mel_start = mel_mid;
  mel_h1.mel_end = MFCC_TINYSPEECH_NUM_MEL;
  mel_h1.coef_offset = mfcc_mc_coef_offset(S->filterLengths, mel_mid);

  mel_h0.S = S;
  mel_h0.spectrum = pSrc;
  mel_h0.mel_out = pTmp;
  mel_h0.mel_start = 0U;
  mel_h0.mel_end = mel_mid;
  mel_h0.coef_offset = 0U;

  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1, mfcc_mc_mel_worker_f32, &mel_h1);
  mfcc_mc_mel_worker_f32(&mel_h0);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");

  riscv_offset_f32(pTmp, 1.0e-6f, pTmp, MFCC_TINYSPEECH_NUM_MEL);
  riscv_vlog_f32(pTmp, pTmp, MFCC_TINYSPEECH_NUM_MEL);

  dct_h1.S = S;
  dct_h1.mel = pTmp;
  dct_h1.out = output;
  dct_h1.dct_start = dct_mid;
  dct_h1.dct_end = MFCC_TINYSPEECH_NUM_DCT;

  dct_h0.S = S;
  dct_h0.mel = pTmp;
  dct_h0.out = output;
  dct_h0.dct_start = 0U;
  dct_h0.dct_end = dct_mid;

  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1, mfcc_mc_dct_worker_f32, &dct_h1);
  mfcc_mc_dct_worker_f32(&dct_h0);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");
  t1 = mfcc_mc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return MFCC_DRIVER_OK;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f16_mc(mfcc_driver_t *ctx,
                                                          const float32_t *input,
                                                          float16_t *output,
                                                          uint64_t *cycles) {
  const riscv_mfcc_instance_f16 *S;
  float16_t *pSrc;
  float16_t *pTmp;
  float16_t maxValue = 0.0f16;
  uint32_t index = 0U;
  uint64_t t0 = 0U;
  uint64_t t1 = 0U;
  const uint32_t mel_mid = (MFCC_TINYSPEECH_NUM_MEL + 1U) / 2U;
  const uint32_t dct_mid = (MFCC_TINYSPEECH_NUM_DCT + 1U) / 2U;
  mfcc_mc_mel_job_f16_t mel_h1;
  mfcc_mc_mel_job_f16_t mel_h0;
  mfcc_mc_dct_job_f16_t dct_h1;
  mfcc_mc_dct_job_f16_t dct_h0;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  if (mfcc_mc_hart_id() != 0U) {
    return mfcc_driver_run_sp1024x23x12_f16(ctx, input, output, cycles);
  }

  S = &ctx->mfcc_f16;
  pSrc = ctx->input_f16;
  pTmp = ctx->tmp_f16;
  for (uint32_t i = 0; i < MFCC_DRIVER_FFT_LEN; i++) {
    pSrc[i] = (float16_t)input[i];
  }

  t0 = mfcc_mc_rdcycle64();
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

  mel_h1.S = S;
  mel_h1.spectrum = pSrc;
  mel_h1.mel_out = pTmp;
  mel_h1.mel_start = mel_mid;
  mel_h1.mel_end = MFCC_TINYSPEECH_NUM_MEL;
  mel_h1.coef_offset = mfcc_mc_coef_offset(S->filterLengths, mel_mid);

  mel_h0.S = S;
  mel_h0.spectrum = pSrc;
  mel_h0.mel_out = pTmp;
  mel_h0.mel_start = 0U;
  mel_h0.mel_end = mel_mid;
  mel_h0.coef_offset = 0U;

  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1, mfcc_mc_mel_worker_f16, &mel_h1);
  mfcc_mc_mel_worker_f16(&mel_h0);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");

  riscv_offset_f16(pTmp, 1.0e-4f16, pTmp, MFCC_TINYSPEECH_NUM_MEL);
  riscv_vlog_f16(pTmp, pTmp, MFCC_TINYSPEECH_NUM_MEL);

  dct_h1.S = S;
  dct_h1.mel = pTmp;
  dct_h1.out = output;
  dct_h1.dct_start = dct_mid;
  dct_h1.dct_end = MFCC_TINYSPEECH_NUM_DCT;

  dct_h0.S = S;
  dct_h0.mel = pTmp;
  dct_h0.out = output;
  dct_h0.dct_start = 0U;
  dct_h0.dct_end = dct_mid;

  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1, mfcc_mc_dct_worker_f16, &dct_h1);
  mfcc_mc_dct_worker_f16(&dct_h0);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");
  t1 = mfcc_mc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return MFCC_DRIVER_OK;
}
#endif
