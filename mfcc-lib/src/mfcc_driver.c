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

/* torchaudio MelSpectrogram default: a Hann window of win_length samples, placed centered inside the
 * n_fft frame and zero-padded elsewhere (win_length < n_fft, center=False). win_length = 30 ms @ 16
 * kHz = 480. torch.hann_window is periodic by default: w[m] = 0.5*(1 - cos(2*pi*m/win_length)). The
 * previous window here was a Hamming spanning the full 1024-sample frame, which analyzed a 64 ms
 * window instead of torchaudio's 30 ms -> a systematic feature mismatch. See /CLAUDE.md KWS notes. */
#define MFCC_DRIVER_WIN_LEN 480U
static void generate_window(mfcc_driver_t *ctx) {
  const float32_t kPi = 3.14159265358979323846f;
  const uint32_t pad_left = (MFCC_DRIVER_FFT_LEN - MFCC_DRIVER_WIN_LEN) / 2U; /* center the window */
  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; n++) {
    float32_t w = 0.0f;
    if ((n >= pad_left) && (n < (pad_left + MFCC_DRIVER_WIN_LEN))) {
      const uint32_t m = n - pad_left;
      w = 0.5f - (0.5f * cosf((2.0f * kPi * (float32_t)m) / (float32_t)MFCC_DRIVER_WIN_LEN));
    }
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

/* torchaudio-matching mel filterbank: htk mel scale over f_min=0 .. f_max=sr/2 (=8000 Hz), with
 * continuous triangular weights evaluated at each FFT bin's centre frequency (norm=None, so peak-1
 * triangles — no Slaney area normalization). Matches torch's melscale_fbanks(). The previous version
 * spanned only 20..4000 Hz and quantized filter edges to integer bins, placing the 23 filters on the
 * wrong frequencies entirely -> the dominant on-chip-vs-reference feature divergence. */
static mfcc_driver_status_t generate_mel_filterbank(mfcc_driver_t *ctx) {
  const float32_t f_min_hz = 0.0f;
  const float32_t f_max_hz = MFCC_DRIVER_SAMPLE_RATE_HZ / 2.0f; /* 8000 */
  const uint32_t n_freqs = MFCC_DRIVER_NUM_FFT_BINS;            /* 513 unique RFFT bins */
  const float32_t bin_hz = f_max_hz / (float32_t)(n_freqs - 1U);
  const float32_t mel_min = hz_to_mel(f_min_hz);
  const float32_t mel_max = hz_to_mel(f_max_hz);

  /* NUM_MEL+2 band edges (in Hz), equally spaced on the mel scale. */
  float32_t f_pts[MFCC_DRIVER_NUM_MEL + 2U];
  for (uint32_t i = 0; i < (MFCC_DRIVER_NUM_MEL + 2U); i++) {
    const float32_t frac = ((float32_t)i) / ((float32_t)(MFCC_DRIVER_NUM_MEL + 1U));
    f_pts[i] = mel_to_hz(mel_min + frac * (mel_max - mel_min));
  }

  ctx->filter_coef_count = 0U;
  for (uint32_t m = 0; m < MFCC_DRIVER_NUM_MEL; m++) {
    const float32_t f_left = f_pts[m];
    const float32_t f_center = f_pts[m + 1U];
    const float32_t f_right = f_pts[m + 2U];
    uint32_t start = 0U;
    uint32_t count = 0U;
    int started = 0;

    for (uint32_t k = 0; k < n_freqs; k++) {
      const float32_t f = (float32_t)k * bin_hz;
      const float32_t up = (f - f_left) / (f_center - f_left);      /* rising edge */
      const float32_t down = (f_right - f) / (f_right - f_center);  /* falling edge */
      float32_t v = (up < down) ? up : down;
      if (v < 0.0f) {
        v = 0.0f;
      }

      if (v > 0.0f) {
        if (!started) {
          start = k;
          started = 1;
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
      } else if (started) {
        break; /* triangular support is contiguous in k -> done with this filter */
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
