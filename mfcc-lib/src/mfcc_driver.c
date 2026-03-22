#include "mfcc_driver.h"

#include <math.h>
#include <string.h>

static inline uint64_t mfcc_rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

static float32_t clampf_local(float32_t x, float32_t lo, float32_t hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static q31_t to_q31(float32_t x) {
  const float32_t s = clampf_local(x, -1.0f, 0.9999999f) * 2147483647.0f;
  if (s >= 0.0f) {
    return (q31_t)(s + 0.5f);
  }
  return (q31_t)(s - 0.5f);
}

static q15_t to_q15(float32_t x) {
  const float32_t s = clampf_local(x, -1.0f, 0.9999695f) * 32767.0f;
  if (s >= 0.0f) {
    return (q15_t)(s + 0.5f);
  }
  return (q15_t)(s - 0.5f);
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static float16_t to_f16(float32_t x) {
  return (float16_t)clampf_local(x, -1.0f, 1.0f);
}
#endif

static float32_t hz_to_mel(float32_t hz) {
  return 2595.0f * log10f(1.0f + (hz / 700.0f));
}

static float32_t mel_to_hz(float32_t mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void generate_window(mfcc_driver_t *ctx) {
  const float32_t kPi = 3.14159265358979323846f;
  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; n++) {
    const float32_t w =
        0.54f - (0.46f * cosf((2.0f * kPi * (float32_t)n) / (float32_t)(MFCC_DRIVER_FFT_LEN - 1U)));
    ctx->window_f32[n] = w;
    ctx->window_q31[n] = to_q31(w);
    ctx->window_q15[n] = to_q15(w);
#if defined(RISCV_FLOAT16_SUPPORTED)
    ctx->window_f16[n] = to_f16(w);
#endif
  }
}

static void generate_dct(mfcc_driver_t *ctx) {
  const float32_t kPi = 3.14159265358979323846f;
  const float32_t m = (float32_t)MFCC_DRIVER_NUM_MEL;

  for (uint32_t k = 0; k < MFCC_DRIVER_NUM_DCT; k++) {
    const float32_t alpha = (k == 0U) ? sqrtf(1.0f / m) : sqrtf(2.0f / m);
    for (uint32_t n = 0; n < MFCC_DRIVER_NUM_MEL; n++) {
      const float32_t c = alpha * cosf((kPi / m) * ((float32_t)n + 0.5f) * (float32_t)k);
      const uint32_t idx = (k * MFCC_DRIVER_NUM_MEL) + n;
      ctx->dct_f32[idx] = c;
      ctx->dct_q31[idx] = to_q31(c);
      ctx->dct_q15[idx] = to_q15(c);
#if defined(RISCV_FLOAT16_SUPPORTED)
      ctx->dct_f16[idx] = to_f16(c);
#endif
    }
  }
}

static mfcc_driver_status_t generate_mel_filterbank(mfcc_driver_t *ctx) {
  const float32_t f_min_hz = 20.0f;
  const float32_t f_max_hz = 4000.0f;
  const float32_t mel_min = hz_to_mel(f_min_hz);
  const float32_t mel_max = hz_to_mel(f_max_hz);

  float32_t mel_points[MFCC_DRIVER_NUM_MEL + 2U];
  uint32_t bins[MFCC_DRIVER_NUM_MEL + 2U];

  for (uint32_t i = 0; i < (MFCC_DRIVER_NUM_MEL + 2U); i++) {
    const float32_t frac = ((float32_t)i) / ((float32_t)(MFCC_DRIVER_NUM_MEL + 1U));
    mel_points[i] = mel_min + frac * (mel_max - mel_min);

    float32_t hz = mel_to_hz(mel_points[i]);
    float32_t bin_f = ((float32_t)MFCC_DRIVER_FFT_LEN + 1.0f) * hz / MFCC_DRIVER_SAMPLE_RATE_HZ;

    if (bin_f < 0.0f) {
      bin_f = 0.0f;
    }
    if (bin_f > (float32_t)(MFCC_DRIVER_NUM_FFT_BINS - 1U)) {
      bin_f = (float32_t)(MFCC_DRIVER_NUM_FFT_BINS - 1U);
    }
    bins[i] = (uint32_t)bin_f;
  }

  ctx->filter_coef_count = 0U;
  for (uint32_t m = 0; m < MFCC_DRIVER_NUM_MEL; m++) {
    uint32_t left = bins[m];
    uint32_t center = bins[m + 1U];
    uint32_t right = bins[m + 2U];

    if (center <= left) {
      center = left + 1U;
    }
    if (right <= center) {
      right = center + 1U;
    }
    if (right >= MFCC_DRIVER_NUM_FFT_BINS) {
      right = MFCC_DRIVER_NUM_FFT_BINS - 1U;
    }

    uint32_t start = 0U;
    uint32_t count = 0U;

    for (uint32_t k = left; k <= right; k++) {
      float32_t v = 0.0f;

      if (k < center) {
        v = ((float32_t)k - (float32_t)left) / ((float32_t)center - (float32_t)left);
      } else if (k > center) {
        v = ((float32_t)right - (float32_t)k) / ((float32_t)right - (float32_t)center);
      } else {
        v = 1.0f;
      }

      if (v > 0.0f) {
        if (count == 0U) {
          start = k;
        }
        if (ctx->filter_coef_count >= MFCC_DRIVER_MAX_FILTER_COEFS) {
          return MFCC_DRIVER_ERR_INIT;
        }

        ctx->filter_f32[ctx->filter_coef_count] = v;
        ctx->filter_q31[ctx->filter_coef_count] = to_q31(v);
        ctx->filter_q15[ctx->filter_coef_count] = to_q15(v);
#if defined(RISCV_FLOAT16_SUPPORTED)
        ctx->filter_f16[ctx->filter_coef_count] = to_f16(v);
#endif
        ctx->filter_coef_count++;
        count++;
      }
    }

    ctx->filter_pos[m] = start;
    ctx->filter_lengths[m] = count;
  }

  return MFCC_DRIVER_OK;
}

mfcc_driver_status_t mfcc_driver_init(mfcc_driver_t *ctx) {
  riscv_status st;

  if (ctx == NULL) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  memset(ctx, 0, sizeof(*ctx));
  generate_window(ctx);
  generate_dct(ctx);
  if (generate_mel_filterbank(ctx) != MFCC_DRIVER_OK) {
    return MFCC_DRIVER_ERR_INIT;
  }

  st = riscv_mfcc_init_f32(&ctx->mfcc_f32,
                           MFCC_DRIVER_FFT_LEN,
                           MFCC_DRIVER_NUM_MEL,
                           MFCC_DRIVER_NUM_DCT,
                           ctx->dct_f32,
                           ctx->filter_pos,
                           ctx->filter_lengths,
                           ctx->filter_f32,
                           ctx->window_f32);
  if (st != RISCV_MATH_SUCCESS) {
    return MFCC_DRIVER_ERR_INIT;
  }

  st = riscv_mfcc_init_q31(&ctx->mfcc_q31,
                           MFCC_DRIVER_FFT_LEN,
                           MFCC_DRIVER_NUM_MEL,
                           MFCC_DRIVER_NUM_DCT,
                           ctx->dct_q31,
                           ctx->filter_pos,
                           ctx->filter_lengths,
                           ctx->filter_q31,
                           ctx->window_q31);
  if (st != RISCV_MATH_SUCCESS) {
    return MFCC_DRIVER_ERR_INIT;
  }

  st = riscv_mfcc_init_q15(&ctx->mfcc_q15,
                           MFCC_DRIVER_FFT_LEN,
                           MFCC_DRIVER_NUM_MEL,
                           MFCC_DRIVER_NUM_DCT,
                           ctx->dct_q15,
                           ctx->filter_pos,
                           ctx->filter_lengths,
                           ctx->filter_q15,
                           ctx->window_q15);
  if (st != RISCV_MATH_SUCCESS) {
    return MFCC_DRIVER_ERR_INIT;
  }

#if defined(RISCV_FLOAT16_SUPPORTED)
  st = riscv_mfcc_init_f16(&ctx->mfcc_f16,
                           MFCC_DRIVER_FFT_LEN,
                           MFCC_DRIVER_NUM_MEL,
                           MFCC_DRIVER_NUM_DCT,
                           ctx->dct_f16,
                           ctx->filter_pos,
                           ctx->filter_lengths,
                           ctx->filter_f16,
                           ctx->window_f16);
  if (st != RISCV_MATH_SUCCESS) {
    return MFCC_DRIVER_ERR_INIT;
  }
#endif

  ctx->initialized = 1U;
  return MFCC_DRIVER_OK;
}

mfcc_driver_status_t mfcc_driver_run_f32(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         float32_t *output,
                                         uint64_t *cycles) {
  uint64_t t0;
  uint64_t t1;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  memcpy(ctx->input_f32, input, sizeof(ctx->input_f32));
  t0 = mfcc_rdcycle64();
  riscv_mfcc_f32(&ctx->mfcc_f32, ctx->input_f32, output, ctx->tmp_f32);
  t1 = mfcc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return MFCC_DRIVER_OK;
}

mfcc_driver_status_t mfcc_driver_run_q31(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         q31_t *output,
                                         uint64_t *cycles) {
  uint64_t t0;
  uint64_t t1;
  riscv_status st;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  for (uint32_t i = 0; i < MFCC_DRIVER_FFT_LEN; i++) {
    ctx->input_q31[i] = to_q31(input[i]);
  }

  t0 = mfcc_rdcycle64();
  st = riscv_mfcc_q31(&ctx->mfcc_q31, ctx->input_q31, output, ctx->tmp_q31);
  t1 = mfcc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return (st == RISCV_MATH_SUCCESS) ? MFCC_DRIVER_OK : MFCC_DRIVER_ERR_INIT;
}

mfcc_driver_status_t mfcc_driver_run_q15(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         q15_t *output,
                                         uint64_t *cycles) {
  uint64_t t0;
  uint64_t t1;
  riscv_status st;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  for (uint32_t i = 0; i < MFCC_DRIVER_FFT_LEN; i++) {
    ctx->input_q15[i] = to_q15(input[i]);
  }

  t0 = mfcc_rdcycle64();
  st = riscv_mfcc_q15(&ctx->mfcc_q15, ctx->input_q15, output, ctx->tmp_q15_as_q31);
  t1 = mfcc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return (st == RISCV_MATH_SUCCESS) ? MFCC_DRIVER_OK : MFCC_DRIVER_ERR_INIT;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
mfcc_driver_status_t mfcc_driver_run_f16(mfcc_driver_t *ctx,
                                         const float32_t *input,
                                         float16_t *output,
                                         uint64_t *cycles) {
  uint64_t t0;
  uint64_t t1;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  for (uint32_t i = 0; i < MFCC_DRIVER_FFT_LEN; i++) {
    ctx->input_f16[i] = to_f16(input[i]);
  }

  t0 = mfcc_rdcycle64();
  riscv_mfcc_f16(&ctx->mfcc_f16, ctx->input_f16, output, ctx->tmp_f16);
  t1 = mfcc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return MFCC_DRIVER_OK;
}
#endif

mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f32(mfcc_driver_t *ctx,
                                                      const float32_t *input,
                                                      float32_t *output,
                                                      uint64_t *cycles) {
  uint64_t t0;
  uint64_t t1;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  memcpy(ctx->input_f32, input, sizeof(ctx->input_f32));
  t0 = mfcc_rdcycle64();
  mfcc_tinyspeech_1024_23_12_f32(&ctx->mfcc_f32, ctx->input_f32, output, ctx->tmp_f32);
  t1 = mfcc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return MFCC_DRIVER_OK;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
mfcc_driver_status_t mfcc_driver_run_sp1024x23x12_f16(mfcc_driver_t *ctx,
                                                      const float32_t *input,
                                                      float16_t *output,
                                                      uint64_t *cycles) {
  uint64_t t0;
  uint64_t t1;

  if ((ctx == NULL) || (input == NULL) || (output == NULL) || (ctx->initialized == 0U)) {
    return MFCC_DRIVER_ERR_BAD_ARG;
  }

  for (uint32_t i = 0; i < MFCC_DRIVER_FFT_LEN; i++) {
    ctx->input_f16[i] = to_f16(input[i]);
  }

  t0 = mfcc_rdcycle64();
  mfcc_tinyspeech_1024_23_12_f16(&ctx->mfcc_f16, ctx->input_f16, output, ctx->tmp_f16);
  t1 = mfcc_rdcycle64();

  if (cycles != NULL) {
    *cycles = t1 - t0;
  }
  return MFCC_DRIVER_OK;
}
#endif

const char *mfcc_driver_status_str(mfcc_driver_status_t status) {
  switch (status) {
    case MFCC_DRIVER_OK:
      return "OK";
    case MFCC_DRIVER_ERR_BAD_ARG:
      return "BAD_ARG";
    case MFCC_DRIVER_ERR_INIT:
      return "INIT_ERROR";
    case MFCC_DRIVER_ERR_F16_UNSUPPORTED:
      return "F16_UNSUPPORTED";
    default:
      return "UNKNOWN";
  }
}
