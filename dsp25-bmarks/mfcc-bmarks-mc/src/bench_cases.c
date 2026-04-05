#include "bench_cases.h"

#include <math.h>

static float32_t clampf_local(float32_t x, float32_t lo, float32_t hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

void mfcc_bench_prepare_cases(mfcc_bench_case_t *cases, uint32_t num_cases) {
  uint32_t lcg = 0x12345678u;
  const float32_t kPi = 3.14159265358979323846f;

  if ((cases == 0) || (num_cases < MFCC_BENCH_NUM_CASES)) {
    return;
  }

  cases[0].name = "silence";
  cases[1].name = "impulse";
  cases[2].name = "alt_sign";
  cases[3].name = "sine_440hz";
  cases[4].name = "sine_3khz";
  cases[5].name = "chirp_100_to_3k";
  cases[6].name = "noise";
  cases[7].name = "hard_clipped_mix";

  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; n++) {
    const float32_t t = (float32_t)n / MFCC_DRIVER_SAMPLE_RATE_HZ;
    const float32_t frac = (float32_t)n / (float32_t)(MFCC_DRIVER_FFT_LEN - 1U);

    cases[0].samples[n] = 0.0f;
    cases[1].samples[n] = (n == 0U) ? 1.0f : 0.0f;
    cases[2].samples[n] = (n & 1U) ? -0.9f : 0.9f;
    cases[3].samples[n] = 0.9f * sinf(2.0f * kPi * 440.0f * t);
    cases[4].samples[n] = 0.9f * sinf(2.0f * kPi * 3000.0f * t);

    {
      const float32_t chirp_hz = 100.0f + (2900.0f * frac);
      cases[5].samples[n] = 0.85f * sinf(2.0f * kPi * chirp_hz * t);
    }

    lcg = (1664525u * lcg) + 1013904223u;
    {
      const float32_t u = ((float32_t)(lcg & 0x00FFFFFFu) / 8388607.5f) - 1.0f;
      cases[6].samples[n] = 0.8f * u;
    }

    {
      const float32_t mix =
          1.4f * sinf(2.0f * kPi * 700.0f * t) + 1.1f * sinf(2.0f * kPi * 1200.0f * t);
      cases[7].samples[n] = clampf_local(mix, -0.7f, 0.7f);
    }
  }
}
