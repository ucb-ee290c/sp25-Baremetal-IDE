#include "main.h"
#include "mfcc_reference_data.h"
#include "mfcc_specialized.h"
#include "simple_setup.h"

#define MFCC_TEST_SAMPLE_RATE_HZ 16000.0f
#define MFCC_TEST_FFT_LEN MFCC_TINYSPEECH_FFT_LEN
#define MFCC_TEST_NUM_MEL 23
#define MFCC_TEST_NUM_DCT 12

#ifndef MFCC_REF_FFT_LEN
#define MFCC_REF_FFT_LEN MFCC_TEST_FFT_LEN
#endif

#define MFCC_TEST_NUM_CASES 8
#define MFCC_TEST_PRINT_INPUT_N 16

#define MFCC_TEST_NUM_FFT_BINS ((MFCC_TEST_FFT_LEN / 2) + 1)
#define MFCC_TEST_MAX_FILTER_COEFS (MFCC_TEST_NUM_MEL * MFCC_TEST_NUM_FFT_BINS)

#define MFCC_TEST_Q31_SCALE 2147483647.0f
#define MFCC_TEST_Q15_SCALE 32767.0f

#if defined(RISCV_FLOAT16_SUPPORTED)
#define MFCC_TEST_ENABLE_F16 1
#else
#define MFCC_TEST_ENABLE_F16 0
#endif

typedef struct {
  const char *name;
  float32_t samples[MFCC_TEST_FFT_LEN];
} mfcc_case_t;

static float32_t g_window_f32[MFCC_TEST_FFT_LEN];
static q31_t g_window_q31[MFCC_TEST_FFT_LEN];
static q15_t g_window_q15[MFCC_TEST_FFT_LEN] __attribute__((aligned(4)));
#if MFCC_TEST_ENABLE_F16
static float16_t g_window_f16[MFCC_TEST_FFT_LEN];
#endif

static float32_t g_dct_f32[MFCC_TEST_NUM_DCT * MFCC_TEST_NUM_MEL];
static q31_t g_dct_q31[MFCC_TEST_NUM_DCT * MFCC_TEST_NUM_MEL];
static q15_t g_dct_q15[MFCC_TEST_NUM_DCT * MFCC_TEST_NUM_MEL] __attribute__((aligned(4)));
#if MFCC_TEST_ENABLE_F16
static float16_t g_dct_f16[MFCC_TEST_NUM_DCT * MFCC_TEST_NUM_MEL];
#endif

static uint32_t g_filter_pos[MFCC_TEST_NUM_MEL];
static uint32_t g_filter_lengths[MFCC_TEST_NUM_MEL];
static uint32_t g_filter_coef_count = 0;

static float32_t g_filter_f32[MFCC_TEST_MAX_FILTER_COEFS];
static q31_t g_filter_q31[MFCC_TEST_MAX_FILTER_COEFS];
static q15_t g_filter_q15[MFCC_TEST_MAX_FILTER_COEFS] __attribute__((aligned(4)));
#if MFCC_TEST_ENABLE_F16
static float16_t g_filter_f16[MFCC_TEST_MAX_FILTER_COEFS];
#endif

static float32_t g_input_f32[MFCC_TEST_FFT_LEN];
static float32_t g_input_f32_specialized[MFCC_TEST_FFT_LEN];
static q31_t g_input_q31[MFCC_TEST_FFT_LEN];
static q15_t g_input_q15[MFCC_TEST_FFT_LEN] __attribute__((aligned(4)));
#if MFCC_TEST_ENABLE_F16
static float16_t g_input_f16[MFCC_TEST_FFT_LEN];
static float16_t g_input_f16_specialized[MFCC_TEST_FFT_LEN];
#endif

static float32_t g_tmp_f32[2 * MFCC_TEST_FFT_LEN];
static q31_t g_tmp_q31[2 * MFCC_TEST_FFT_LEN];
static q31_t g_tmp_q15_as_q31[2 * MFCC_TEST_FFT_LEN];
#if MFCC_TEST_ENABLE_F16
static float16_t g_tmp_f16[2 * MFCC_TEST_FFT_LEN];
#endif

static float32_t g_out_f32[MFCC_TEST_NUM_DCT];
static float32_t g_out_f32_specialized[MFCC_TEST_NUM_DCT];
static q31_t g_out_q31[MFCC_TEST_NUM_DCT];
static q15_t g_out_q15[MFCC_TEST_NUM_DCT] __attribute__((aligned(4)));
#if MFCC_TEST_ENABLE_F16
static float16_t g_out_f16[MFCC_TEST_NUM_DCT];
static float16_t g_out_f16_specialized[MFCC_TEST_NUM_DCT];
#endif

static riscv_mfcc_instance_f32 g_mfcc_f32;
static riscv_mfcc_instance_q31 g_mfcc_q31;
static riscv_mfcc_instance_q15 g_mfcc_q15;
#if MFCC_TEST_ENABLE_F16
static riscv_mfcc_instance_f16 g_mfcc_f16;
#endif

static mfcc_case_t g_cases[MFCC_TEST_NUM_CASES];

static void print_kernel_mode_summary(void) {
  printf("  kernel-mode summary:\n");

#if defined(RISCV_MATH_VECTOR)
  printf("    f32 path : vector-enabled (RVV)\n");
#else
  printf("    f32 path : scalar\n");
#if defined(__riscv_vector)
  printf("    note     : RVV present but disabled (missing __RISCV_VXRM_RNU API)\n");
#endif
#endif
  printf("    sp1024x23x12_f32 path: enabled (fixed-shape TinySpeech helper)\n");

#if defined(MFCC_VLOG_VEC_APPROX) && (MFCC_VLOG_VEC_APPROX == 1)
  printf("    vlog mode: RVV polynomial approximation\n");
#else
  printf("    vlog mode: scalar reference implementation\n");
#endif

#if defined(RISCV_MATH_VECTOR) && defined(__RISCV_XLEN) && (__RISCV_XLEN == 64)
  printf("    q31 path : vector-enabled (RVV, XLEN=64)\n");
  printf("    q15 path : vector-enabled (RVV, XLEN=64)\n");
#else
#if defined(RISCV_MATH_DSP)
  printf("    q31 path : scalar + DSP intrinsics\n");
  printf("    q15 path : scalar + DSP intrinsics\n");
#else
  printf("    q31 path : scalar\n");
  printf("    q15 path : scalar\n");
#endif
#endif

#if MFCC_TEST_ENABLE_F16
#if defined(RISCV_MATH_VECTOR_F16)
  printf("    f16 path : vector-enabled (RVV + Zvfh)\n");
#else
  printf("    f16 path : scalar (no Zvfh RVV support)\n");
#endif
  printf("    sp1024x23x12_f16 path: enabled (fixed-shape TinySpeech helper)\n");
#if defined(MFCC_F16_ASM_MULT)
  printf("    f16 mult  : asm microkernel enabled\n");
#else
  printf("    f16 mult  : C/RVV kernel\n");
#endif
#if defined(MFCC_F16_ASM_RADIX4BY2)
  printf("    f16 radix4by2: asm microkernel enabled\n");
#else
  printf("    f16 radix4by2: C/RVV kernel\n");
#endif
#if defined(MFCC_F16_ASM_RFFT_STAGE)
  printf("    f16 rfft stage: asm microkernel enabled\n");
#else
  printf("    f16 rfft stage: C/RVV kernel\n");
#endif
#if defined(MFCC_F16_ACCUM_MODE) && (MFCC_F16_ACCUM_MODE == 1)
  printf("    f16 accum : mixed f32 accumulation\n");
#else
  printf("    f16 accum : native f16 accumulation\n");
#endif
#if defined(MFCC_F16_ULTRA_OPT_ENABLED)
  printf("    f16 build : ULTRA_OPT enabled\n");
#else
  printf("    f16 build : standard O3 hotspot mode\n");
#endif
#if defined(RISCV_MFCC_F16_RADIX4_VEC_EXPERIMENTAL)
  printf("    f16 radix4: experimental RVV enabled\n");
#else
  printf("    f16 radix4: scalar-safe (experimental RVV block disabled)\n");
#endif
#else
  printf("    f16 path : disabled (no Zfh scalar support)\n");
#endif
}

static inline uint64_t rdcycle64(void) {
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
  float32_t s = clampf_local(x, -1.0f, 0.9999999f) * MFCC_TEST_Q31_SCALE;
  if (s >= 0.0f) {
    return (q31_t)(s + 0.5f);
  }
  return (q31_t)(s - 0.5f);
}

static q15_t to_q15(float32_t x) {
  float32_t s = clampf_local(x, -1.0f, 0.9999695f) * MFCC_TEST_Q15_SCALE;
  if (s >= 0.0f) {
    return (q15_t)(s + 0.5f);
  }
  return (q15_t)(s - 0.5f);
}

static float32_t q31_to_float(q31_t x) {
  return ((float32_t)x) / 8388608.0f; /* q8.23 -> float */
}

static float32_t q15_to_float(q15_t x) {
  return ((float32_t)x) / 128.0f; /* q8.7 -> float */
}

#if MFCC_TEST_ENABLE_F16
static float16_t to_f16(float32_t x) {
  return (float16_t)clampf_local(x, -1.0f, 1.0f);
}

static float32_t f16_to_float(float16_t x) {
  return (float32_t)x;
}
#endif

static float32_t hz_to_mel(float32_t hz) {
  return 2595.0f * log10f(1.0f + (hz / 700.0f));
}

static float32_t mel_to_hz(float32_t mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void generate_window(void) {
  for (uint32_t n = 0; n < MFCC_TEST_FFT_LEN; n++) {
    float32_t w = 0.54f - (0.46f * cosf((2.0f * PI * (float32_t)n) /
                                        (float32_t)(MFCC_TEST_FFT_LEN - 1)));
    g_window_f32[n] = w;
    g_window_q31[n] = to_q31(w);
    g_window_q15[n] = to_q15(w);
#if MFCC_TEST_ENABLE_F16
    g_window_f16[n] = to_f16(w);
#endif
  }
}

static void generate_dct(void) {
  const float32_t m = (float32_t)MFCC_TEST_NUM_MEL;

  for (uint32_t k = 0; k < MFCC_TEST_NUM_DCT; k++) {
    float32_t alpha = (k == 0) ? sqrtf(1.0f / m) : sqrtf(2.0f / m);
    for (uint32_t n = 0; n < MFCC_TEST_NUM_MEL; n++) {
      float32_t c = alpha *
                    cosf((PI / m) * ((float32_t)n + 0.5f) * (float32_t)k);
      uint32_t idx = (k * MFCC_TEST_NUM_MEL) + n;
      g_dct_f32[idx] = c;
      g_dct_q31[idx] = to_q31(c);
      g_dct_q15[idx] = to_q15(c);
#if MFCC_TEST_ENABLE_F16
      g_dct_f16[idx] = to_f16(c);
#endif
    }
  }
}

static void generate_mel_filterbank(void) {
  const float32_t f_min_hz = 20.0f;
  const float32_t f_max_hz = 4000.0f;
  const float32_t mel_min = hz_to_mel(f_min_hz);
  const float32_t mel_max = hz_to_mel(f_max_hz);

  float32_t mel_points[MFCC_TEST_NUM_MEL + 2];
  uint32_t bins[MFCC_TEST_NUM_MEL + 2];

  for (uint32_t i = 0; i < MFCC_TEST_NUM_MEL + 2; i++) {
    float32_t frac = ((float32_t)i) / ((float32_t)(MFCC_TEST_NUM_MEL + 1));
    mel_points[i] = mel_min + frac * (mel_max - mel_min);

    float32_t hz = mel_to_hz(mel_points[i]);
    float32_t bin_f = ((float32_t)MFCC_TEST_FFT_LEN + 1.0f) *
                      hz / MFCC_TEST_SAMPLE_RATE_HZ;

    if (bin_f < 0.0f) {
      bin_f = 0.0f;
    }
    if (bin_f > (float32_t)(MFCC_TEST_NUM_FFT_BINS - 1)) {
      bin_f = (float32_t)(MFCC_TEST_NUM_FFT_BINS - 1);
    }
    bins[i] = (uint32_t)bin_f;
  }

  g_filter_coef_count = 0;

  for (uint32_t m = 0; m < MFCC_TEST_NUM_MEL; m++) {
    uint32_t left = bins[m];
    uint32_t center = bins[m + 1];
    uint32_t right = bins[m + 2];

    if (center <= left) {
      center = left + 1;
    }
    if (right <= center) {
      right = center + 1;
    }
    if (right >= MFCC_TEST_NUM_FFT_BINS) {
      right = MFCC_TEST_NUM_FFT_BINS - 1;
    }

    uint32_t start = 0;
    uint32_t count = 0;

    for (uint32_t k = left; k <= right; k++) {
      float32_t v = 0.0f;

      if (k < center) {
        v = ((float32_t)k - (float32_t)left) /
            ((float32_t)center - (float32_t)left);
      } else if (k > center) {
        v = ((float32_t)right - (float32_t)k) /
            ((float32_t)right - (float32_t)center);
      } else {
        v = 1.0f;
      }

      if (v > 0.0f) {
        if (count == 0) {
          start = k;
        }

        if (g_filter_coef_count < MFCC_TEST_MAX_FILTER_COEFS) {
          g_filter_f32[g_filter_coef_count] = v;
          g_filter_q31[g_filter_coef_count] = to_q31(v);
          g_filter_q15[g_filter_coef_count] = to_q15(v);
#if MFCC_TEST_ENABLE_F16
          g_filter_f16[g_filter_coef_count] = to_f16(v);
#endif
          g_filter_coef_count++;
          count++;
        }
      }
    }

    g_filter_pos[m] = start;
    g_filter_lengths[m] = count;
  }
}

static void prepare_cases(void) {
  uint32_t lcg = 0x12345678u;

  g_cases[0].name = "silence";
  g_cases[1].name = "impulse";
  g_cases[2].name = "alt_sign";
  g_cases[3].name = "sine_440hz";
  g_cases[4].name = "sine_3khz";
  g_cases[5].name = "chirp_100_to_3k";
  g_cases[6].name = "noise";
  g_cases[7].name = "hard_clipped_mix";

  for (uint32_t n = 0; n < MFCC_TEST_FFT_LEN; n++) {
    float32_t t = (float32_t)n / MFCC_TEST_SAMPLE_RATE_HZ;
    float32_t frac = (float32_t)n / (float32_t)(MFCC_TEST_FFT_LEN - 1);

    g_cases[0].samples[n] = 0.0f;
    g_cases[1].samples[n] = (n == 0) ? 1.0f : 0.0f;
    g_cases[2].samples[n] = (n & 1U) ? -0.9f : 0.9f;
    g_cases[3].samples[n] = 0.9f * sinf(2.0f * PI * 440.0f * t);
    g_cases[4].samples[n] = 0.9f * sinf(2.0f * PI * 3000.0f * t);

    float32_t chirp_hz = 100.0f + (2900.0f * frac);
    g_cases[5].samples[n] = 0.85f * sinf(2.0f * PI * chirp_hz * t);

    lcg = (1664525u * lcg) + 1013904223u;
    float32_t u = ((float32_t)(lcg & 0x00FFFFFFu) / 8388607.5f) - 1.0f;
    g_cases[6].samples[n] = 0.8f * u;

    float32_t mix = 1.4f * sinf(2.0f * PI * 700.0f * t) +
                    1.1f * sinf(2.0f * PI * 1200.0f * t);
    g_cases[7].samples[n] = clampf_local(mix, -0.7f, 0.7f);
  }
}

static void print_input_preview(const float32_t *x) {
  printf("    input[0:%d] =", MFCC_TEST_PRINT_INPUT_N - 1);
  for (uint32_t i = 0; i < MFCC_TEST_PRINT_INPUT_N; i++) {
    printf(" %0.4f", x[i]);
  }
  printf("\n");
}

static void print_output_f32(const float32_t *x) {
  printf("    f32_mfcc     =");
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    printf(" %0.5f", x[i]);
  }
  printf("\n");
}

static void print_output_f32_specialized(const float32_t *x) {
  printf("    sp1024x23x12_f32_mfcc=");
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    printf(" %0.5f", x[i]);
  }
  printf("\n");
}

static void print_output_q31(const q31_t *x) {
  printf("    q31_mfcc(q8.23)=");
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    printf(" %0.5f", q31_to_float(x[i]));
  }
  printf("\n");
}

static void print_output_q15(const q15_t *x) {
  printf("    q15_mfcc(q8.7) =");
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    printf(" %0.5f", q15_to_float(x[i]));
  }
  printf("\n");
}

#if MFCC_TEST_ENABLE_F16
static void print_output_f16(const float16_t *x) {
  printf("    f16_mfcc     =");
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    printf(" %0.5f", f16_to_float(x[i]));
  }
  printf("\n");
}

static void print_output_f16_specialized(const float16_t *x) {
  printf("    sp1024x23x12_f16_mfcc=");
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    printf(" %0.5f", f16_to_float(x[i]));
  }
  printf("\n");
}
#endif

static int compare_output_f32_to_reference(uint32_t case_idx, const float32_t *x) {
#if !MFCC_REF_HAS_F32
  (void)case_idx;
  (void)x;
  printf("    ref[f32]    = SKIP (f32 golden disabled)\n");
  return -1;
#else
  if ((MFCC_REF_NUM_CASES != MFCC_TEST_NUM_CASES) ||
      (MFCC_REF_NUM_DCT != MFCC_TEST_NUM_DCT) ||
      (MFCC_REF_FFT_LEN != MFCC_TEST_FFT_LEN)) {
    printf("    ref[f32]    = SKIP (shape/fft mismatch)\n");
    return -1;
  }

  if (strcmp(g_cases[case_idx].name, g_mfcc_ref_case_names[case_idx]) != 0) {
    printf("    ref[f32]    = SKIP (case-name mismatch: test=%s ref=%s)\n",
           g_cases[case_idx].name,
           g_mfcc_ref_case_names[case_idx]);
    return -1;
  }

  float32_t max_abs_err = 0.0f;
  uint32_t max_abs_idx = 0;
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    float32_t err = fabsf(x[i] - g_mfcc_ref_f32[case_idx][i]);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_abs_idx = i;
    }
  }

  int pass = (max_abs_err <= MFCC_REF_F32_TOL);
  printf("    ref[f32]    = %s max_abs_err=%0.6f idx=%lu\n",
         pass ? "PASS" : "FAIL",
         max_abs_err,
         (unsigned long)max_abs_idx);
  return pass ? 1 : 0;
#endif
}

static int compare_output_f32_specialized_to_reference(uint32_t case_idx,
                                                       const float32_t *x) {
#if !MFCC_REF_HAS_F32
  (void)case_idx;
  (void)x;
  printf("    ref[sp1024x23x12_f32] = SKIP (f32 golden disabled)\n");
  return -1;
#else
  if ((MFCC_REF_NUM_CASES != MFCC_TEST_NUM_CASES) ||
      (MFCC_REF_NUM_DCT != MFCC_TEST_NUM_DCT) ||
      (MFCC_REF_FFT_LEN != MFCC_TEST_FFT_LEN)) {
    printf("    ref[sp1024x23x12_f32] = SKIP (shape/fft mismatch)\n");
    return -1;
  }

  if (strcmp(g_cases[case_idx].name, g_mfcc_ref_case_names[case_idx]) != 0) {
    printf("    ref[sp1024x23x12_f32] = SKIP (case-name mismatch: test=%s ref=%s)\n",
           g_cases[case_idx].name,
           g_mfcc_ref_case_names[case_idx]);
    return -1;
  }

  float32_t max_abs_err = 0.0f;
  uint32_t max_abs_idx = 0;
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    float32_t err = fabsf(x[i] - g_mfcc_ref_f32[case_idx][i]);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_abs_idx = i;
    }
  }

  int pass = (max_abs_err <= MFCC_REF_F32_TOL);
  printf("    ref[sp1024x23x12_f32] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
         pass ? "PASS" : "FAIL",
         max_abs_err,
         (unsigned long)max_abs_idx,
         MFCC_REF_F32_TOL);
  return pass ? 1 : 0;
#endif
}

static int compare_output_q31_to_reference(uint32_t case_idx, const q31_t *x_q31) {
#if !MFCC_REF_HAS_Q31
  (void)case_idx;
  (void)x_q31;
  printf("    ref[q31]    = SKIP (q31 golden disabled)\n");
  return -1;
#else
  if ((MFCC_REF_NUM_CASES != MFCC_TEST_NUM_CASES) ||
      (MFCC_REF_NUM_DCT != MFCC_TEST_NUM_DCT) ||
      (MFCC_REF_FFT_LEN != MFCC_TEST_FFT_LEN)) {
    printf("    ref[q31]    = SKIP (shape/fft mismatch)\n");
    return -1;
  }
  if (strcmp(g_cases[case_idx].name, g_mfcc_ref_case_names[case_idx]) != 0) {
    printf("    ref[q31]    = SKIP (case-name mismatch: test=%s ref=%s)\n",
           g_cases[case_idx].name,
           g_mfcc_ref_case_names[case_idx]);
    return -1;
  }

  float32_t max_abs_err = 0.0f;
  uint32_t max_abs_idx = 0;
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    float32_t err = fabsf(q31_to_float(x_q31[i]) - g_mfcc_ref_q31[case_idx][i]);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_abs_idx = i;
    }
  }
  int pass = (max_abs_err <= MFCC_REF_Q31_TOL);
  printf("    ref[q31]    = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
         pass ? "PASS" : "FAIL",
         max_abs_err,
         (unsigned long)max_abs_idx,
         MFCC_REF_Q31_TOL);
  return pass ? 1 : 0;
#endif
}

static int compare_output_q15_to_reference(uint32_t case_idx, const q15_t *x_q15) {
#if !MFCC_REF_HAS_Q15
  (void)case_idx;
  (void)x_q15;
  printf("    ref[q15]    = SKIP (q15 golden disabled)\n");
  return -1;
#else
  if ((MFCC_REF_NUM_CASES != MFCC_TEST_NUM_CASES) ||
      (MFCC_REF_NUM_DCT != MFCC_TEST_NUM_DCT) ||
      (MFCC_REF_FFT_LEN != MFCC_TEST_FFT_LEN)) {
    printf("    ref[q15]    = SKIP (shape/fft mismatch)\n");
    return -1;
  }
  if (strcmp(g_cases[case_idx].name, g_mfcc_ref_case_names[case_idx]) != 0) {
    printf("    ref[q15]    = SKIP (case-name mismatch: test=%s ref=%s)\n",
           g_cases[case_idx].name,
           g_mfcc_ref_case_names[case_idx]);
    return -1;
  }

  float32_t max_abs_err = 0.0f;
  uint32_t max_abs_idx = 0;
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    float32_t err = fabsf(q15_to_float(x_q15[i]) - g_mfcc_ref_q15[case_idx][i]);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_abs_idx = i;
    }
  }
  int pass = (max_abs_err <= MFCC_REF_Q15_TOL);
  printf("    ref[q15]    = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
         pass ? "PASS" : "FAIL",
         max_abs_err,
         (unsigned long)max_abs_idx,
         MFCC_REF_Q15_TOL);
  return pass ? 1 : 0;
#endif
}

#if MFCC_TEST_ENABLE_F16
static int compare_output_f16_to_reference(uint32_t case_idx, const float16_t *x_f16) {
#if !MFCC_REF_HAS_F16
  (void)case_idx;
  (void)x_f16;
  printf("    ref[f16]    = SKIP (f16 golden disabled)\n");
  return -1;
#else
  if ((MFCC_REF_NUM_CASES != MFCC_TEST_NUM_CASES) ||
      (MFCC_REF_NUM_DCT != MFCC_TEST_NUM_DCT) ||
      (MFCC_REF_FFT_LEN != MFCC_TEST_FFT_LEN)) {
    printf("    ref[f16]    = SKIP (shape/fft mismatch)\n");
    return -1;
  }
  if (strcmp(g_cases[case_idx].name, g_mfcc_ref_case_names[case_idx]) != 0) {
    printf("    ref[f16]    = SKIP (case-name mismatch: test=%s ref=%s)\n",
           g_cases[case_idx].name,
           g_mfcc_ref_case_names[case_idx]);
    return -1;
  }

  float32_t max_abs_err = 0.0f;
  uint32_t max_abs_idx = 0;
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    float32_t err = fabsf(f16_to_float(x_f16[i]) - g_mfcc_ref_f16[case_idx][i]);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_abs_idx = i;
    }
  }
  int pass = (max_abs_err <= MFCC_REF_F16_TOL);
  printf("    ref[f16]    = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
         pass ? "PASS" : "FAIL",
         max_abs_err,
         (unsigned long)max_abs_idx,
         MFCC_REF_F16_TOL);
  return pass ? 1 : 0;
#endif
}

static int compare_output_f16_specialized_to_reference(uint32_t case_idx,
                                                       const float16_t *x_f16) {
#if !MFCC_REF_HAS_F16
  (void)case_idx;
  (void)x_f16;
  printf("    ref[sp1024x23x12_f16] = SKIP (f16 golden disabled)\n");
  return -1;
#else
  if ((MFCC_REF_NUM_CASES != MFCC_TEST_NUM_CASES) ||
      (MFCC_REF_NUM_DCT != MFCC_TEST_NUM_DCT) ||
      (MFCC_REF_FFT_LEN != MFCC_TEST_FFT_LEN)) {
    printf("    ref[sp1024x23x12_f16] = SKIP (shape/fft mismatch)\n");
    return -1;
  }
  if (strcmp(g_cases[case_idx].name, g_mfcc_ref_case_names[case_idx]) != 0) {
    printf("    ref[sp1024x23x12_f16] = SKIP (case-name mismatch: test=%s ref=%s)\n",
           g_cases[case_idx].name,
           g_mfcc_ref_case_names[case_idx]);
    return -1;
  }

  float32_t max_abs_err = 0.0f;
  uint32_t max_abs_idx = 0;
  for (uint32_t i = 0; i < MFCC_TEST_NUM_DCT; i++) {
    float32_t err = fabsf(f16_to_float(x_f16[i]) - g_mfcc_ref_f16[case_idx][i]);
    if (err > max_abs_err) {
      max_abs_err = err;
      max_abs_idx = i;
    }
  }
  int pass = (max_abs_err <= MFCC_REF_F16_TOL);
  printf("    ref[sp1024x23x12_f16] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
         pass ? "PASS" : "FAIL",
         max_abs_err,
         (unsigned long)max_abs_idx,
         MFCC_REF_F16_TOL);
  return pass ? 1 : 0;
#endif
}
#endif

static int init_mfcc_instances(void) {
  riscv_status st;

  st = riscv_mfcc_init_f32(&g_mfcc_f32,
                           MFCC_TEST_FFT_LEN,
                           MFCC_TEST_NUM_MEL,
                           MFCC_TEST_NUM_DCT,
                           g_dct_f32,
                           g_filter_pos,
                           g_filter_lengths,
                           g_filter_f32,
                           g_window_f32);
  if (st != RISCV_MATH_SUCCESS) {
    printf("MFCC f32 init failed: %d\n", (int)st);
    return -1;
  }

  st = riscv_mfcc_init_q31(&g_mfcc_q31,
                           MFCC_TEST_FFT_LEN,
                           MFCC_TEST_NUM_MEL,
                           MFCC_TEST_NUM_DCT,
                           g_dct_q31,
                           g_filter_pos,
                           g_filter_lengths,
                           g_filter_q31,
                           g_window_q31);
  if (st != RISCV_MATH_SUCCESS) {
    printf("MFCC q31 init failed: %d\n", (int)st);
    return -1;
  }

  st = riscv_mfcc_init_q15(&g_mfcc_q15,
                           MFCC_TEST_FFT_LEN,
                           MFCC_TEST_NUM_MEL,
                           MFCC_TEST_NUM_DCT,
                           g_dct_q15,
                           g_filter_pos,
                           g_filter_lengths,
                           g_filter_q15,
                           g_window_q15);
  if (st != RISCV_MATH_SUCCESS) {
    printf("MFCC q15 init failed: %d\n", (int)st);
    return -1;
  }

#if MFCC_TEST_ENABLE_F16
  st = riscv_mfcc_init_f16(&g_mfcc_f16,
                           MFCC_TEST_FFT_LEN,
                           MFCC_TEST_NUM_MEL,
                           MFCC_TEST_NUM_DCT,
                           g_dct_f16,
                           g_filter_pos,
                           g_filter_lengths,
                           g_filter_f16,
                           g_window_f16);
  if (st != RISCV_MATH_SUCCESS) {
    printf("MFCC f16 init failed: %d\n", (int)st);
    return -1;
  }
#endif

  return 0;
}

void app_init(void) {
  printf("MFCC test init\n");
  // init_test(500000000ULL);
}

void app_main(void) {
  printf("=== DSP25 MFCC Kernel Bring-up ===\n");
  printf("  FFT_LEN=%d MEL=%d DCT=%d\n",
         MFCC_TEST_FFT_LEN, MFCC_TEST_NUM_MEL, MFCC_TEST_NUM_DCT);
#if MFCC_TEST_ENABLE_F16
  printf("  f16 path enabled (RISCV_FLOAT16_SUPPORTED)\n");
#else
  printf("  f16 path disabled (toolchain missing __riscv_zfh)\n");
#endif
  print_kernel_mode_summary();

  generate_window();
  generate_dct();
  generate_mel_filterbank();
  prepare_cases();

  printf("  filter coef count=%lu\n", (unsigned long)g_filter_coef_count);

  if (init_mfcc_instances() != 0) {
    printf("Initialization failed.\n");
    return;
  }

  uint32_t ref_pass = 0;
  uint32_t ref_fail = 0;
  uint32_t ref_skip = 0;
  uint32_t ref_q31_pass = 0;
  uint32_t ref_q31_fail = 0;
  uint32_t ref_q31_skip = 0;
  uint32_t ref_q15_pass = 0;
  uint32_t ref_q15_fail = 0;
  uint32_t ref_q15_skip = 0;
  uint32_t ref_sp_f32_pass = 0;
  uint32_t ref_sp_f32_fail = 0;
  uint32_t ref_sp_f32_skip = 0;
#if MFCC_TEST_ENABLE_F16
  uint32_t ref_f16_pass = 0;
  uint32_t ref_f16_fail = 0;
  uint32_t ref_f16_skip = 0;
  uint32_t ref_sp_f16_pass = 0;
  uint32_t ref_sp_f16_fail = 0;
  uint32_t ref_sp_f16_skip = 0;
#endif

  for (uint32_t tc = 0; tc < MFCC_TEST_NUM_CASES; tc++) {
    const float32_t *in = g_cases[tc].samples;

    memcpy(g_input_f32, in, sizeof(g_input_f32));
    memcpy(g_input_f32_specialized, in, sizeof(g_input_f32_specialized));
    for (uint32_t i = 0; i < MFCC_TEST_FFT_LEN; i++) {
      g_input_q31[i] = to_q31(in[i]);
      g_input_q15[i] = to_q15(in[i]);
#if MFCC_TEST_ENABLE_F16
      g_input_f16[i] = to_f16(in[i]);
      g_input_f16_specialized[i] = g_input_f16[i];
#endif
    }

    printf("\n[CASE %lu] %s\n", (unsigned long)tc, g_cases[tc].name);
    print_input_preview(in);

    uint64_t f0 = rdcycle64();
    riscv_mfcc_f32(&g_mfcc_f32, g_input_f32, g_out_f32, g_tmp_f32);
    uint64_t f1 = rdcycle64();
    uint64_t sf0 = rdcycle64();
    mfcc_tinyspeech_1024_23_12_f32(&g_mfcc_f32,
                                  g_input_f32_specialized,
                                  g_out_f32_specialized,
                                  g_tmp_f32);
    uint64_t sf1 = rdcycle64();

    uint64_t q0 = rdcycle64();
    riscv_status st_q31 = riscv_mfcc_q31(&g_mfcc_q31,
                                         g_input_q31,
                                         g_out_q31,
                                         g_tmp_q31);
    uint64_t q1 = rdcycle64();

    uint64_t h0 = rdcycle64();
    riscv_status st_q15 = riscv_mfcc_q15(&g_mfcc_q15,
                                         g_input_q15,
                                         g_out_q15,
                                         g_tmp_q15_as_q31);
    uint64_t h1 = rdcycle64();

#if MFCC_TEST_ENABLE_F16
    uint64_t e0 = rdcycle64();
    riscv_mfcc_f16(&g_mfcc_f16, g_input_f16, g_out_f16, g_tmp_f16);
    uint64_t e1 = rdcycle64();
    uint64_t se0 = rdcycle64();
    mfcc_tinyspeech_1024_23_12_f16(&g_mfcc_f16,
                                  g_input_f16_specialized,
                                  g_out_f16_specialized,
                                  g_tmp_f16);
    uint64_t se1 = rdcycle64();
#endif

    print_output_f32(g_out_f32);
    print_output_f32_specialized(g_out_f32_specialized);
    print_output_q31(g_out_q31);
    print_output_q15(g_out_q15);
#if MFCC_TEST_ENABLE_F16
    print_output_f16(g_out_f16);
    print_output_f16_specialized(g_out_f16_specialized);
#endif

    printf("    cycles[f32]=%lu\n", (unsigned long)(f1 - f0));
    printf("    cycles[sp1024x23x12_f32]=%lu\n", (unsigned long)(sf1 - sf0));
    printf("    cycles[q31]=%lu status=%d\n", (unsigned long)(q1 - q0), (int)st_q31);
    printf("    cycles[q15]=%lu status=%d\n", (unsigned long)(h1 - h0), (int)st_q15);
#if MFCC_TEST_ENABLE_F16
    printf("    cycles[f16]=%lu\n", (unsigned long)(e1 - e0));
    printf("    cycles[sp1024x23x12_f16]=%lu\n", (unsigned long)(se1 - se0));
#endif

    int ref_status = compare_output_f32_to_reference(tc, g_out_f32);
    if (ref_status > 0) {
      ref_pass++;
    } else if (ref_status == 0) {
      ref_fail++;
    } else {
      ref_skip++;
    }

    int q31_status = compare_output_q31_to_reference(tc, g_out_q31);
    if (q31_status > 0) {
      ref_q31_pass++;
    } else if (q31_status == 0) {
      ref_q31_fail++;
    } else {
      ref_q31_skip++;
    }

    int q15_status = compare_output_q15_to_reference(tc, g_out_q15);
    if (q15_status > 0) {
      ref_q15_pass++;
    } else if (q15_status == 0) {
      ref_q15_fail++;
    } else {
      ref_q15_skip++;
    }

    int sp_f32_status = compare_output_f32_specialized_to_reference(tc, g_out_f32_specialized);
    if (sp_f32_status > 0) {
      ref_sp_f32_pass++;
    } else if (sp_f32_status == 0) {
      ref_sp_f32_fail++;
    } else {
      ref_sp_f32_skip++;
    }

#if MFCC_TEST_ENABLE_F16
    int f16_status = compare_output_f16_to_reference(tc, g_out_f16);
    if (f16_status > 0) {
      ref_f16_pass++;
    } else if (f16_status == 0) {
      ref_f16_fail++;
    } else {
      ref_f16_skip++;
    }

    int sp_f16_status = compare_output_f16_specialized_to_reference(tc, g_out_f16_specialized);
    if (sp_f16_status > 0) {
      ref_sp_f16_pass++;
    } else if (sp_f16_status == 0) {
      ref_sp_f16_fail++;
    } else {
      ref_sp_f16_skip++;
    }
#endif
  }

  if ((MFCC_REF_NUM_CASES == MFCC_TEST_NUM_CASES) &&
      (MFCC_REF_NUM_DCT == MFCC_TEST_NUM_DCT) &&
      (MFCC_REF_FFT_LEN == MFCC_TEST_FFT_LEN)) {
    printf("\nReference summary: pass=%lu fail=%lu skip=%lu tol=%0.6f\n",
           (unsigned long)ref_pass,
           (unsigned long)ref_fail,
           (unsigned long)ref_skip,
           MFCC_REF_F32_TOL);
  }

  printf("Reference summary[q31]: pass=%lu fail=%lu skip=%lu tol=%0.6f\n",
         (unsigned long)ref_q31_pass,
         (unsigned long)ref_q31_fail,
         (unsigned long)ref_q31_skip,
         MFCC_REF_Q31_TOL);
  printf("Reference summary[q15]: pass=%lu fail=%lu skip=%lu tol=%0.6f\n",
         (unsigned long)ref_q15_pass,
         (unsigned long)ref_q15_fail,
         (unsigned long)ref_q15_skip,
         MFCC_REF_Q15_TOL);
  printf("Reference summary[sp1024x23x12_f32]: pass=%lu fail=%lu skip=%lu tol=%0.6f\n",
         (unsigned long)ref_sp_f32_pass,
         (unsigned long)ref_sp_f32_fail,
         (unsigned long)ref_sp_f32_skip,
         MFCC_REF_F32_TOL);
#if MFCC_TEST_ENABLE_F16
  printf("Reference summary[f16]: pass=%lu fail=%lu skip=%lu tol=%0.6f\n",
         (unsigned long)ref_f16_pass,
         (unsigned long)ref_f16_fail,
         (unsigned long)ref_f16_skip,
         MFCC_REF_F16_TOL);
  printf("Reference summary[sp1024x23x12_f16]: pass=%lu fail=%lu skip=%lu tol=%0.6f\n",
         (unsigned long)ref_sp_f16_pass,
         (unsigned long)ref_sp_f16_fail,
         (unsigned long)ref_sp_f16_skip,
         MFCC_REF_F16_TOL);
#endif

  printf("\nMFCC test complete.\n");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}
