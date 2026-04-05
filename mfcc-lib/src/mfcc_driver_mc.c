#include "mfcc_driver_mc.h"

#include <stddef.h>
#include <string.h>

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

typedef enum {
  MFCC_MC_STAGE_NONE = 0U,
  MFCC_MC_STAGE_ABSMAX = 1U,
  MFCC_MC_STAGE_SCALE_WINDOW = 2U,
  MFCC_MC_STAGE_RFFT_STAGE = 3U,
  MFCC_MC_STAGE_CMPLX_MAG = 4U,
  MFCC_MC_STAGE_RESCALE = 5U,
  MFCC_MC_STAGE_MEL = 6U,
  MFCC_MC_STAGE_DCT = 7U,
  MFCC_MC_STAGE_EXIT = 8U
} mfcc_mc_stage_t;

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

typedef struct {
  volatile uint32_t cmd;
  volatile uint32_t ack;
  const riscv_mfcc_instance_f32 *S;
  float32_t *pSrc;
  float32_t *pTmp;
  float32_t *pDst;
  float32_t maxValue;
  float32_t localMax;
  float32_t normScale;
  const riscv_rfft_fast_instance_f32 *rfft;
  uint32_t rfft_l_start;
  uint32_t rfft_l_end;
  uint32_t half;
  uint32_t mel_mid;
  uint32_t dct_mid;
} mfcc_mc_worker_f32_t;

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

typedef struct {
  volatile uint32_t cmd;
  volatile uint32_t ack;
  const riscv_mfcc_instance_f16 *S;
  float16_t *pSrc;
  float16_t *pTmp;
  float16_t *pDst;
  float16_t maxValue;
  float16_t localMax;
  float16_t normScale;
  const riscv_rfft_fast_instance_f16 *rfft;
  uint32_t rfft_l_start;
  uint32_t rfft_l_end;
  uint32_t half;
  uint32_t mel_mid;
  uint32_t dct_mid;
} mfcc_mc_worker_f16_t;
#endif

typedef enum {
  MFCC_MC_WORKER_NONE = 0U,
  MFCC_MC_WORKER_F32 = 1U,
  MFCC_MC_WORKER_F16 = 2U
} mfcc_mc_worker_kind_t;

static mfcc_mc_worker_kind_t g_mfcc_mc_worker_kind = MFCC_MC_WORKER_NONE;
static mfcc_mc_worker_f32_t g_mfcc_mc_worker_f32;
#if defined(RISCV_FLOAT16_SUPPORTED)
static mfcc_mc_worker_f16_t g_mfcc_mc_worker_f16;
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

static inline void mfcc_mc_stage_launch(volatile uint32_t *cmd,
                                        volatile uint32_t *ack,
                                        uint32_t stage) {
  __atomic_store_n(ack, MFCC_MC_STAGE_NONE, __ATOMIC_RELEASE);
  __sync_synchronize();
  __atomic_store_n(cmd, stage, __ATOMIC_RELEASE);
}

static inline void mfcc_mc_stage_wait(volatile uint32_t *ack, uint32_t stage) {
  while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != stage) {
    asm volatile("nop");
  }
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

static void mfcc_mc_rfft_stage_f32_range(const riscv_rfft_fast_instance_f32 *R,
                                         const float32_t *p,
                                         float32_t *pOut,
                                         uint32_t l_start,
                                         uint32_t l_end) {
  if ((R == NULL) || (p == NULL) || (pOut == NULL) || (l_end <= l_start)) {
    return;
  }

#if defined(RISCV_MATH_VECTOR)
  {
    uint32_t blkCnt = l_end - l_start;
    size_t vl;
    ptrdiff_t cplxStride = (ptrdiff_t)(2U * sizeof(float32_t));
    ptrdiff_t revStride = -cplxStride;
    const float32_t *pAcur = p + (2U * l_start);
    const float32_t *pBcur = p + (2U * ((R->Sint).fftLen - l_start));
    const float32_t *pTwR = R->pTwiddleRFFT + (2U * l_start);
    const float32_t *pTwI = pTwR + 1;
    ptrdiff_t twStride = cplxStride;
    float32_t *pOutCur = pOut + (2U * l_start);
    const float32_t half = 0.5f;

    while ((vl = __riscv_vsetvl_e32m8(blkCnt)) > 0) {
      vfloat32m8_t vAR = __riscv_vlse32_v_f32m8(pAcur, cplxStride, vl);
      vfloat32m8_t vAI = __riscv_vlse32_v_f32m8(pAcur + 1, cplxStride, vl);
      vfloat32m8_t vBR = __riscv_vlse32_v_f32m8(pBcur, revStride, vl);
      vfloat32m8_t vBI = __riscv_vlse32_v_f32m8(pBcur + 1, revStride, vl);
      vfloat32m8_t vTwR = __riscv_vlse32_v_f32m8(pTwR, twStride, vl);
      vfloat32m8_t vTwI = __riscv_vlse32_v_f32m8(pTwI, twStride, vl);

      vfloat32m8_t vT1a = __riscv_vfsub_vv_f32m8(vBR, vAR, vl);
      vfloat32m8_t vT1b = __riscv_vfadd_vv_f32m8(vBI, vAI, vl);

      vfloat32m8_t vP0 = __riscv_vfmul_vv_f32m8(vTwR, vT1a, vl);
      vfloat32m8_t vP1 = __riscv_vfmul_vv_f32m8(vTwI, vT1a, vl);
      vfloat32m8_t vP2 = __riscv_vfmul_vv_f32m8(vTwR, vT1b, vl);
      vfloat32m8_t vP3 = __riscv_vfmul_vv_f32m8(vTwI, vT1b, vl);

      vfloat32m8_t vOutR = __riscv_vfadd_vv_f32m8(vAR, vBR, vl);
      vOutR = __riscv_vfadd_vv_f32m8(vOutR, vP0, vl);
      vOutR = __riscv_vfadd_vv_f32m8(vOutR, vP3, vl);
      vOutR = __riscv_vfmul_vf_f32m8(vOutR, half, vl);

      vfloat32m8_t vOutI = __riscv_vfsub_vv_f32m8(vAI, vBI, vl);
      vOutI = __riscv_vfadd_vv_f32m8(vOutI, vP1, vl);
      vOutI = __riscv_vfsub_vv_f32m8(vOutI, vP2, vl);
      vOutI = __riscv_vfmul_vf_f32m8(vOutI, half, vl);

      __riscv_vsse32_v_f32m8(pOutCur, cplxStride, vOutR, vl);
      __riscv_vsse32_v_f32m8(pOutCur + 1, cplxStride, vOutI, vl);

      pOutCur += (uint32_t)(2U * vl);
      pAcur += (uint32_t)(2U * vl);
      pBcur -= (uint32_t)(2U * vl);
      pTwR += (uint32_t)(2U * vl);
      pTwI += (uint32_t)(2U * vl);
      blkCnt -= (uint32_t)vl;
    }
  }
#else
  for (uint32_t l = l_start; l < l_end; l++) {
    const uint32_t ia = 2U * l;
    const uint32_t ib = 2U * ((R->Sint).fftLen - l);
    const uint32_t itw = 2U * l;
    const float32_t xAR = p[ia];
    const float32_t xAI = p[ia + 1U];
    const float32_t xBR = p[ib];
    const float32_t xBI = p[ib + 1U];
    const float32_t twR = R->pTwiddleRFFT[itw];
    const float32_t twI = R->pTwiddleRFFT[itw + 1U];
    const float32_t t1a = xBR - xAR;
    const float32_t t1b = xBI + xAI;
    const float32_t p0 = twR * t1a;
    const float32_t p1 = twI * t1a;
    const float32_t p2 = twR * t1b;
    const float32_t p3 = twI * t1b;
    pOut[ia] = 0.5f * (xAR + xBR + p0 + p3);
    pOut[ia + 1U] = 0.5f * (xAI - xBI + p1 - p2);
  }
#endif
}

static void mfcc_mc_intra_worker_f32(void *arg) {
  mfcc_mc_worker_f32_t *w = (mfcc_mc_worker_f32_t *)arg;
  uint32_t last = MFCC_MC_STAGE_NONE;
  uint32_t idx = 0U;

  while (1) {
    const uint32_t stage = __atomic_load_n(&w->cmd, __ATOMIC_ACQUIRE);
    if ((stage == MFCC_MC_STAGE_NONE) || (stage == last)) {
      asm volatile("nop");
      continue;
    }

    if (stage == MFCC_MC_STAGE_ABSMAX) {
      riscv_absmax_f32(w->pSrc + w->half, w->S->fftLen - w->half, &w->localMax, &idx);
    } else if (stage == MFCC_MC_STAGE_SCALE_WINDOW) {
      if (w->normScale != 1.0f) {
        riscv_scale_f32(w->pSrc + w->half, w->normScale, w->pSrc + w->half, w->S->fftLen - w->half);
      }
      riscv_mult_f32(w->pSrc + w->half,
                     w->S->windowCoefs + w->half,
                     w->pSrc + w->half,
                     w->S->fftLen - w->half);
    } else if (stage == MFCC_MC_STAGE_RFFT_STAGE) {
      mfcc_mc_rfft_stage_f32_range(w->rfft, w->pSrc, w->pTmp, w->rfft_l_start, w->rfft_l_end);
    } else if (stage == MFCC_MC_STAGE_CMPLX_MAG) {
      riscv_cmplx_mag_f32(w->pTmp + (2U * w->half), w->pSrc + w->half, w->S->fftLen - w->half);
    } else if (stage == MFCC_MC_STAGE_RESCALE) {
      if (w->maxValue != 0.0f) {
        riscv_scale_f32(w->pSrc + w->half, w->maxValue, w->pSrc + w->half, w->S->fftLen - w->half);
      }
    } else if (stage == MFCC_MC_STAGE_MEL) {
      mfcc_mc_mel_job_f32_t job;
      job.S = w->S;
      job.spectrum = w->pSrc;
      job.mel_out = w->pTmp;
      job.mel_start = w->mel_mid;
      job.mel_end = MFCC_TINYSPEECH_NUM_MEL;
      job.coef_offset = mfcc_mc_coef_offset(w->S->filterLengths, w->mel_mid);
      mfcc_mc_mel_worker_f32(&job);
    } else if (stage == MFCC_MC_STAGE_DCT) {
      mfcc_mc_dct_job_f32_t job;
      job.S = w->S;
      job.mel = w->pTmp;
      job.out = w->pDst;
      job.dct_start = w->dct_mid;
      job.dct_end = MFCC_TINYSPEECH_NUM_DCT;
      mfcc_mc_dct_worker_f32(&job);
    }

    __sync_synchronize();
    __atomic_store_n(&w->ack, stage, __ATOMIC_RELEASE);
    last = stage;
    if (stage == MFCC_MC_STAGE_EXIT) {
      return;
    }
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

static void mfcc_mc_rfft_stage_f16_range(const riscv_rfft_fast_instance_f16 *R,
                                         const float16_t *p,
                                         float16_t *pOut,
                                         uint32_t l_start,
                                         uint32_t l_end) {
  if ((R == NULL) || (p == NULL) || (pOut == NULL) || (l_end <= l_start)) {
    return;
  }

#if defined(RISCV_MATH_VECTOR_F16)
  {
    uint32_t blkCnt = l_end - l_start;
    size_t vl;
    ptrdiff_t cplxStride = (ptrdiff_t)(2U * sizeof(float16_t));
    ptrdiff_t revStride = -cplxStride;
    const float16_t *pAcur = p + (2U * l_start);
    const float16_t *pBcur = p + (2U * ((R->Sint).fftLen - l_start));
    const float16_t *pTwR = R->pTwiddleRFFT + (2U * l_start);
    const float16_t *pTwI = pTwR + 1;
    ptrdiff_t twStride = cplxStride;
    float16_t *pOutCur = pOut + (2U * l_start);
    const float16_t half = 0.5f16;

    while ((vl = __riscv_vsetvl_e16m8(blkCnt)) > 0) {
      vfloat16m8_t vAR = __riscv_vlse16_v_f16m8(pAcur, cplxStride, vl);
      vfloat16m8_t vAI = __riscv_vlse16_v_f16m8(pAcur + 1, cplxStride, vl);
      vfloat16m8_t vBR = __riscv_vlse16_v_f16m8(pBcur, revStride, vl);
      vfloat16m8_t vBI = __riscv_vlse16_v_f16m8(pBcur + 1, revStride, vl);
      vfloat16m8_t vTwR = __riscv_vlse16_v_f16m8(pTwR, twStride, vl);
      vfloat16m8_t vTwI = __riscv_vlse16_v_f16m8(pTwI, twStride, vl);

      vfloat16m8_t vT1a = __riscv_vfsub_vv_f16m8(vBR, vAR, vl);
      vfloat16m8_t vT1b = __riscv_vfadd_vv_f16m8(vBI, vAI, vl);

      vfloat16m8_t vP0 = __riscv_vfmul_vv_f16m8(vTwR, vT1a, vl);
      vfloat16m8_t vP1 = __riscv_vfmul_vv_f16m8(vTwI, vT1a, vl);
      vfloat16m8_t vP2 = __riscv_vfmul_vv_f16m8(vTwR, vT1b, vl);
      vfloat16m8_t vP3 = __riscv_vfmul_vv_f16m8(vTwI, vT1b, vl);

      vfloat16m8_t vOutR = __riscv_vfadd_vv_f16m8(vAR, vBR, vl);
      vOutR = __riscv_vfadd_vv_f16m8(vOutR, vP0, vl);
      vOutR = __riscv_vfadd_vv_f16m8(vOutR, vP3, vl);
      vOutR = __riscv_vfmul_vf_f16m8(vOutR, half, vl);

      vfloat16m8_t vOutI = __riscv_vfsub_vv_f16m8(vAI, vBI, vl);
      vOutI = __riscv_vfadd_vv_f16m8(vOutI, vP1, vl);
      vOutI = __riscv_vfsub_vv_f16m8(vOutI, vP2, vl);
      vOutI = __riscv_vfmul_vf_f16m8(vOutI, half, vl);

      __riscv_vsse16_v_f16m8(pOutCur, cplxStride, vOutR, vl);
      __riscv_vsse16_v_f16m8(pOutCur + 1, cplxStride, vOutI, vl);

      pOutCur += (uint32_t)(2U * vl);
      pAcur += (uint32_t)(2U * vl);
      pBcur -= (uint32_t)(2U * vl);
      pTwR += (uint32_t)(2U * vl);
      pTwI += (uint32_t)(2U * vl);
      blkCnt -= (uint32_t)vl;
    }
  }
#else
  for (uint32_t l = l_start; l < l_end; l++) {
    const uint32_t ia = 2U * l;
    const uint32_t ib = 2U * ((R->Sint).fftLen - l);
    const uint32_t itw = 2U * l;
    const float16_t xAR = p[ia];
    const float16_t xAI = p[ia + 1U];
    const float16_t xBR = p[ib];
    const float16_t xBI = p[ib + 1U];
    const float16_t twR = R->pTwiddleRFFT[itw];
    const float16_t twI = R->pTwiddleRFFT[itw + 1U];
    const _Float16 t1a = (_Float16)xBR - (_Float16)xAR;
    const _Float16 t1b = (_Float16)xBI + (_Float16)xAI;
    const _Float16 p0 = (_Float16)twR * t1a;
    const _Float16 p1 = (_Float16)twI * t1a;
    const _Float16 p2 = (_Float16)twR * t1b;
    const _Float16 p3 = (_Float16)twI * t1b;
    pOut[ia] = (float16_t)(0.5f16 * ((_Float16)xAR + (_Float16)xBR + p0 + p3));
    pOut[ia + 1U] = (float16_t)(0.5f16 * ((_Float16)xAI - (_Float16)xBI + p1 - p2));
  }
#endif
}

static void mfcc_mc_intra_worker_f16(void *arg) {
  mfcc_mc_worker_f16_t *w = (mfcc_mc_worker_f16_t *)arg;
  uint32_t last = MFCC_MC_STAGE_NONE;
  uint32_t idx = 0U;

  while (1) {
    const uint32_t stage = __atomic_load_n(&w->cmd, __ATOMIC_ACQUIRE);
    if ((stage == MFCC_MC_STAGE_NONE) || (stage == last)) {
      asm volatile("nop");
      continue;
    }

    if (stage == MFCC_MC_STAGE_ABSMAX) {
      riscv_absmax_f16(w->pSrc + w->half, w->S->fftLen - w->half, &w->localMax, &idx);
    } else if (stage == MFCC_MC_STAGE_SCALE_WINDOW) {
      if ((_Float16)w->normScale != 1.0f16) {
        riscv_scale_f16(w->pSrc + w->half, w->normScale, w->pSrc + w->half, w->S->fftLen - w->half);
      }
      riscv_mult_f16(w->pSrc + w->half,
                     w->S->windowCoefs + w->half,
                     w->pSrc + w->half,
                     w->S->fftLen - w->half);
    } else if (stage == MFCC_MC_STAGE_RFFT_STAGE) {
      mfcc_mc_rfft_stage_f16_range(w->rfft, w->pSrc, w->pTmp, w->rfft_l_start, w->rfft_l_end);
    } else if (stage == MFCC_MC_STAGE_CMPLX_MAG) {
      riscv_cmplx_mag_f16(w->pTmp + (2U * w->half), w->pSrc + w->half, w->S->fftLen - w->half);
    } else if (stage == MFCC_MC_STAGE_RESCALE) {
      if ((_Float16)w->maxValue != 0.0f16) {
        riscv_scale_f16(w->pSrc + w->half, w->maxValue, w->pSrc + w->half, w->S->fftLen - w->half);
      }
    } else if (stage == MFCC_MC_STAGE_MEL) {
      mfcc_mc_mel_job_f16_t job;
      job.S = w->S;
      job.spectrum = w->pSrc;
      job.mel_out = w->pTmp;
      job.mel_start = w->mel_mid;
      job.mel_end = MFCC_TINYSPEECH_NUM_MEL;
      job.coef_offset = mfcc_mc_coef_offset(w->S->filterLengths, w->mel_mid);
      mfcc_mc_mel_worker_f16(&job);
    } else if (stage == MFCC_MC_STAGE_DCT) {
      mfcc_mc_dct_job_f16_t job;
      job.S = w->S;
      job.mel = w->pTmp;
      job.out = w->pDst;
      job.dct_start = w->dct_mid;
      job.dct_end = MFCC_TINYSPEECH_NUM_DCT;
      mfcc_mc_dct_worker_f16(&job);
    }

    __sync_synchronize();
    __atomic_store_n(&w->ack, stage, __ATOMIC_RELEASE);
    last = stage;
    if (stage == MFCC_MC_STAGE_EXIT) {
      return;
    }
  }
}
#endif

static void mfcc_mc_stop_active_worker(void) {
  if (g_mfcc_mc_worker_kind == MFCC_MC_WORKER_F32) {
    mfcc_mc_stage_launch(&g_mfcc_mc_worker_f32.cmd, &g_mfcc_mc_worker_f32.ack, MFCC_MC_STAGE_EXIT);
    mfcc_mc_stage_wait(&g_mfcc_mc_worker_f32.ack, MFCC_MC_STAGE_EXIT);
    hthread_join(1U);
    g_mfcc_mc_worker_kind = MFCC_MC_WORKER_NONE;
    return;
  }

#if defined(RISCV_FLOAT16_SUPPORTED)
  if (g_mfcc_mc_worker_kind == MFCC_MC_WORKER_F16) {
    mfcc_mc_stage_launch(&g_mfcc_mc_worker_f16.cmd, &g_mfcc_mc_worker_f16.ack, MFCC_MC_STAGE_EXIT);
    mfcc_mc_stage_wait(&g_mfcc_mc_worker_f16.ack, MFCC_MC_STAGE_EXIT);
    hthread_join(1U);
    g_mfcc_mc_worker_kind = MFCC_MC_WORKER_NONE;
    return;
  }
#endif
}

static mfcc_mc_worker_f32_t *mfcc_mc_ensure_worker_f32(void) {
  if (g_mfcc_mc_worker_kind == MFCC_MC_WORKER_F32) {
    return &g_mfcc_mc_worker_f32;
  }

  mfcc_mc_stop_active_worker();
  memset(&g_mfcc_mc_worker_f32, 0, sizeof(g_mfcc_mc_worker_f32));
  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1U, mfcc_mc_intra_worker_f32, &g_mfcc_mc_worker_f32);
  g_mfcc_mc_worker_kind = MFCC_MC_WORKER_F32;
  return &g_mfcc_mc_worker_f32;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static mfcc_mc_worker_f16_t *mfcc_mc_ensure_worker_f16(void) {
  if (g_mfcc_mc_worker_kind == MFCC_MC_WORKER_F16) {
    return &g_mfcc_mc_worker_f16;
  }

  mfcc_mc_stop_active_worker();
  memset(&g_mfcc_mc_worker_f16, 0, sizeof(g_mfcc_mc_worker_f16));
  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1U, mfcc_mc_intra_worker_f16, &g_mfcc_mc_worker_f16);
  g_mfcc_mc_worker_kind = MFCC_MC_WORKER_F16;
  return &g_mfcc_mc_worker_f16;
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
  float32_t maxValueLo = 0.0f;
  uint32_t index = 0U;
  uint64_t t0 = 0U;
  uint64_t t1 = 0U;
  mfcc_mc_worker_f32_t *w;
  mfcc_mc_mel_job_f32_t mel_h0;
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

  w = mfcc_mc_ensure_worker_f32();
  w->S = S;
  w->pSrc = pSrc;
  w->pTmp = pTmp;
  w->pDst = output;
  w->half = S->fftLen / 2U;
  w->mel_mid = (MFCC_TINYSPEECH_NUM_MEL + 1U) / 2U;
  w->dct_mid = (MFCC_TINYSPEECH_NUM_DCT + 1U) / 2U;
  w->localMax = 0.0f;
  w->normScale = 1.0f;
  w->cmd = MFCC_MC_STAGE_NONE;
  w->ack = MFCC_MC_STAGE_NONE;

  t0 = mfcc_mc_rdcycle64();

  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_ABSMAX);
  riscv_absmax_f32(pSrc, w->half, &maxValueLo, &index);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_ABSMAX);
  maxValue = (maxValueLo > w->localMax) ? maxValueLo : w->localMax;
  if (maxValue != 0.0f) {
    w->normScale = 1.0f / maxValue;
  }
  w->maxValue = maxValue;

  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_SCALE_WINDOW);
  if (w->normScale != 1.0f) {
    riscv_scale_f32(pSrc, w->normScale, pSrc, w->half);
  }
  riscv_mult_f32(pSrc, S->windowCoefs, pSrc, w->half);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_SCALE_WINDOW);

#if defined(RISCV_MFCC_CFFT_BASED)
  for (uint32_t i = 0; i < S->fftLen; i++) {
    pTmp[2U * i] = pSrc[i];
    pTmp[(2U * i) + 1U] = 0.0f;
  }
  riscv_cfft_f32(&(S->cfft), pTmp, 0, 1);
#else
  {
    const uint32_t n2 = (S->rfft.Sint).fftLen;
    const uint32_t bins = n2 - 1U;
    const uint32_t l_mid = 1U + (bins >> 1);
    riscv_cfft_f32(&(S->rfft.Sint), pSrc, 0, 1);
    const float32_t xAR = pSrc[0];
    const float32_t xAI = pSrc[1];

    pTmp[0] = xAR + xAI;
    pTmp[1] = xAR - xAI;

    w->rfft = &(S->rfft);
    w->rfft_l_start = l_mid;
    w->rfft_l_end = n2;
    mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_RFFT_STAGE);
    mfcc_mc_rfft_stage_f32_range(&(S->rfft), pSrc, pTmp, 1U, l_mid);
    mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_RFFT_STAGE);
  }
  pTmp[S->fftLen] = pTmp[1];
  pTmp[S->fftLen + 1U] = 0.0f;
  pTmp[1] = 0.0f;
#endif

  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_CMPLX_MAG);
  riscv_cmplx_mag_f32(pTmp, pSrc, w->half);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_CMPLX_MAG);

  if (maxValue != 0.0f) {
    mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_RESCALE);
    riscv_scale_f32(pSrc, maxValue, pSrc, w->half);
    mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_RESCALE);
  }

  mel_h0.S = S;
  mel_h0.spectrum = pSrc;
  mel_h0.mel_out = pTmp;
  mel_h0.mel_start = 0U;
  mel_h0.mel_end = w->mel_mid;
  mel_h0.coef_offset = 0U;
  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_MEL);
  mfcc_mc_mel_worker_f32(&mel_h0);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_MEL);

  riscv_offset_f32(pTmp, 1.0e-6f, pTmp, MFCC_TINYSPEECH_NUM_MEL);
  riscv_vlog_f32(pTmp, pTmp, MFCC_TINYSPEECH_NUM_MEL);

  dct_h0.S = S;
  dct_h0.mel = pTmp;
  dct_h0.out = output;
  dct_h0.dct_start = 0U;
  dct_h0.dct_end = w->dct_mid;
  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_DCT);
  mfcc_mc_dct_worker_f32(&dct_h0);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_DCT);
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
  float16_t maxValueLo = 0.0f16;
  uint32_t index = 0U;
  uint64_t t0 = 0U;
  uint64_t t1 = 0U;
  mfcc_mc_worker_f16_t *w;
  mfcc_mc_mel_job_f16_t mel_h0;
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

  w = mfcc_mc_ensure_worker_f16();
  w->S = S;
  w->pSrc = pSrc;
  w->pTmp = pTmp;
  w->pDst = output;
  w->half = S->fftLen / 2U;
  w->mel_mid = (MFCC_TINYSPEECH_NUM_MEL + 1U) / 2U;
  w->dct_mid = (MFCC_TINYSPEECH_NUM_DCT + 1U) / 2U;
  w->localMax = 0.0f16;
  w->normScale = 1.0f16;
  w->cmd = MFCC_MC_STAGE_NONE;
  w->ack = MFCC_MC_STAGE_NONE;

  t0 = mfcc_mc_rdcycle64();

  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_ABSMAX);
  riscv_absmax_f16(pSrc, w->half, &maxValueLo, &index);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_ABSMAX);
  maxValue = ((_Float16)maxValueLo > (_Float16)w->localMax) ? maxValueLo : w->localMax;
  if ((_Float16)maxValue != 0.0f16) {
    w->normScale = (float16_t)(1.0f16 / (_Float16)maxValue);
  }
  w->maxValue = maxValue;

  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_SCALE_WINDOW);
  if ((_Float16)w->normScale != 1.0f16) {
    riscv_scale_f16(pSrc, w->normScale, pSrc, w->half);
  }
  riscv_mult_f16(pSrc, S->windowCoefs, pSrc, w->half);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_SCALE_WINDOW);

#if defined(RISCV_MFCC_CFFT_BASED)
  for (uint32_t i = 0; i < S->fftLen; i++) {
    pTmp[2U * i] = pSrc[i];
    pTmp[(2U * i) + 1U] = 0.0f16;
  }
  riscv_cfft_f16(&(S->cfft), pTmp, 0, 1);
#else
  {
    const uint32_t n2 = (S->rfft.Sint).fftLen;
    const uint32_t bins = n2 - 1U;
    const uint32_t l_mid = 1U + (bins >> 1);
    riscv_cfft_f16(&(S->rfft.Sint), pSrc, 0, 1);
    const float16_t xAR = pSrc[0];
    const float16_t xAI = pSrc[1];

    pTmp[0] = (float16_t)((_Float16)xAR + (_Float16)xAI);
    pTmp[1] = (float16_t)((_Float16)xAR - (_Float16)xAI);

    w->rfft = &(S->rfft);
    w->rfft_l_start = l_mid;
    w->rfft_l_end = n2;
    mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_RFFT_STAGE);
    mfcc_mc_rfft_stage_f16_range(&(S->rfft), pSrc, pTmp, 1U, l_mid);
    mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_RFFT_STAGE);
  }
  pTmp[S->fftLen] = pTmp[1];
  pTmp[S->fftLen + 1U] = 0.0f16;
  pTmp[1] = 0.0f16;
#endif

  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_CMPLX_MAG);
  riscv_cmplx_mag_f16(pTmp, pSrc, w->half);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_CMPLX_MAG);

  if ((_Float16)maxValue != 0.0f16) {
    mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_RESCALE);
    riscv_scale_f16(pSrc, maxValue, pSrc, w->half);
    mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_RESCALE);
  }

  mel_h0.S = S;
  mel_h0.spectrum = pSrc;
  mel_h0.mel_out = pTmp;
  mel_h0.mel_start = 0U;
  mel_h0.mel_end = w->mel_mid;
  mel_h0.coef_offset = 0U;
  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_MEL);
  mfcc_mc_mel_worker_f16(&mel_h0);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_MEL);

  riscv_offset_f16(pTmp, 1.0e-4f16, pTmp, MFCC_TINYSPEECH_NUM_MEL);
  riscv_vlog_f16(pTmp, pTmp, MFCC_TINYSPEECH_NUM_MEL);

  dct_h0.S = S;
  dct_h0.mel = pTmp;
  dct_h0.out = output;
  dct_h0.dct_start = 0U;
  dct_h0.dct_end = w->dct_mid;
  mfcc_mc_stage_launch(&w->cmd, &w->ack, MFCC_MC_STAGE_DCT);
  mfcc_mc_dct_worker_f16(&dct_h0);
  mfcc_mc_stage_wait(&w->ack, MFCC_MC_STAGE_DCT);
  t1 = mfcc_mc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return MFCC_DRIVER_OK;
}
#endif

void mfcc_driver_mc_shutdown(void) {
  if (mfcc_mc_hart_id() != 0U) {
    return;
  }
  mfcc_mc_stop_active_worker();
}
