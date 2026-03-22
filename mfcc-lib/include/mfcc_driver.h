#ifndef MFCC_DRIVER_H
#define MFCC_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "mfcc_specialized.h"
#include "riscv_math.h"

#define MFCC_DRIVER_SAMPLE_RATE_HZ 16000.0f
#define MFCC_DRIVER_FFT_LEN MFCC_TINYSPEECH_FFT_LEN
#define MFCC_DRIVER_NUM_MEL MFCC_TINYSPEECH_NUM_MEL
#define MFCC_DRIVER_NUM_DCT MFCC_TINYSPEECH_NUM_DCT
#define MFCC_DRIVER_NUM_FFT_BINS ((MFCC_DRIVER_FFT_LEN / 2U) + 1U)
#define MFCC_DRIVER_MAX_FILTER_COEFS (MFCC_DRIVER_NUM_MEL * MFCC_DRIVER_NUM_FFT_BINS)

typedef enum {
  MFCC_DRIVER_OK = 0,
  MFCC_DRIVER_ERR_BAD_ARG = -1,
  MFCC_DRIVER_ERR_INIT = -2,
  MFCC_DRIVER_ERR_F16_UNSUPPORTED = -3,
} mfcc_driver_status_t;

typedef struct {
  uint32_t initialized;

  riscv_mfcc_instance_f32 mfcc_f32;
  riscv_mfcc_instance_q31 mfcc_q31;
  riscv_mfcc_instance_q15 mfcc_q15;
#if defined(RISCV_FLOAT16_SUPPORTED)
  riscv_mfcc_instance_f16 mfcc_f16;
#endif

  float32_t window_f32[MFCC_DRIVER_FFT_LEN];
  q31_t window_q31[MFCC_DRIVER_FFT_LEN];
  q15_t window_q15[MFCC_DRIVER_FFT_LEN] __attribute__((aligned(4)));
#if defined(RISCV_FLOAT16_SUPPORTED)
  float16_t window_f16[MFCC_DRIVER_FFT_LEN];
#endif

  float32_t dct_f32[MFCC_DRIVER_NUM_DCT * MFCC_DRIVER_NUM_MEL];
  q31_t dct_q31[MFCC_DRIVER_NUM_DCT * MFCC_DRIVER_NUM_MEL];
  q15_t dct_q15[MFCC_DRIVER_NUM_DCT * MFCC_DRIVER_NUM_MEL] __attribute__((aligned(4)));
#if defined(RISCV_FLOAT16_SUPPORTED)
  float16_t dct_f16[MFCC_DRIVER_NUM_DCT * MFCC_DRIVER_NUM_MEL];
#endif

  uint32_t filter_pos[MFCC_DRIVER_NUM_MEL];
  uint32_t filter_lengths[MFCC_DRIVER_NUM_MEL];
  uint32_t filter_coef_count;

  float32_t filter_f32[MFCC_DRIVER_MAX_FILTER_COEFS];
  q31_t filter_q31[MFCC_DRIVER_MAX_FILTER_COEFS];
  q15_t filter_q15[MFCC_DRIVER_MAX_FILTER_COEFS] __attribute__((aligned(4)));
#if defined(RISCV_FLOAT16_SUPPORTED)
  float16_t filter_f16[MFCC_DRIVER_MAX_FILTER_COEFS];
#endif

  float32_t input_f32[MFCC_DRIVER_FFT_LEN];
  q31_t input_q31[MFCC_DRIVER_FFT_LEN];
  q15_t input_q15[MFCC_DRIVER_FFT_LEN] __attribute__((aligned(4)));
#if defined(RISCV_FLOAT16_SUPPORTED)
  float16_t input_f16[MFCC_DRIVER_FFT_LEN];
#endif

  float32_t tmp_f32[2 * MFCC_DRIVER_FFT_LEN];
  q31_t tmp_q31[2 * MFCC_DRIVER_FFT_LEN];
  q31_t tmp_q15_as_q31[2 * MFCC_DRIVER_FFT_LEN];
#if defined(RISCV_FLOAT16_SUPPORTED)
  float16_t tmp_f16[2 * MFCC_DRIVER_FFT_LEN];
#endif
} mfcc_driver_t;

mfcc_driver_status_t mfcc_driver_init(mfcc_driver_t *ctx);

mfcc_driver_status_t mfcc_driver_run_f32(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         float32_t *output,
                                         uint64_t *cycles);
mfcc_driver_status_t mfcc_driver_run_q31(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         q31_t *output,
                                         uint64_t *cycles);
mfcc_driver_status_t mfcc_driver_run_q15(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         q15_t *output,
                                         uint64_t *cycles);
#if defined(RISCV_FLOAT16_SUPPORTED)
mfcc_driver_status_t mfcc_driver_run_f16(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         float16_t *output,
                                         uint64_t *cycles);
#endif
mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f32(mfcc_driver_t *ctx,
                                                      const float32_t *input,
                                                      float32_t *output,
                                                      uint64_t *cycles);
#if defined(RISCV_FLOAT16_SUPPORTED)
mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f16(mfcc_driver_t *ctx,
                                                      const float32_t *input,
                                                      float16_t *output,
                                                      uint64_t *cycles);
#endif

static inline float32_t mfcc_driver_q31_to_float(q31_t x) {
  return ((float32_t)x) / 8388608.0f;
}

static inline float32_t mfcc_driver_q15_to_float(q15_t x) {
  return ((float32_t)x) / 128.0f;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static inline float32_t mfcc_driver_f16_to_float(float16_t x) {
  return (float32_t)x;
}
#endif

const char *mfcc_driver_status_str(mfcc_driver_status_t status);

#ifdef __cplusplus
}
#endif

#endif
