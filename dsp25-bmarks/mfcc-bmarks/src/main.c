#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bench_cache.h"
#include "bench_cases.h"
#include "bench_config.h"
#include "mfcc_driver.h"
#include "mfcc_reference_data.h"
#include "simple_setup.h"

#ifndef MFCC_REF_FFT_LEN
#define MFCC_REF_FFT_LEN MFCC_DRIVER_FFT_LEN
#endif

typedef enum {
  MFCC_VAR_F32 = 0,
  MFCC_VAR_Q31,
  MFCC_VAR_Q15,
  MFCC_VAR_F16,
  MFCC_VAR_SP_F32,
  MFCC_VAR_SP_F16,
  MFCC_VAR_COUNT
} mfcc_variant_t;

typedef struct {
  uint64_t sum;
  uint64_t best;
  uint64_t worst;
  uint32_t runs;
} cycle_stats_t;

typedef struct {
  uint32_t pass;
  uint32_t fail;
  uint32_t skip;
} check_stats_t;

typedef struct {
  float32_t out_f32[MFCC_DRIVER_NUM_DCT];
  q31_t out_q31[MFCC_DRIVER_NUM_DCT];
  q15_t out_q15[MFCC_DRIVER_NUM_DCT];
#if defined(RISCV_FLOAT16_SUPPORTED)
  float16_t out_f16[MFCC_DRIVER_NUM_DCT];
#endif
  float32_t out_sp_f32[MFCC_DRIVER_NUM_DCT];
#if defined(RISCV_FLOAT16_SUPPORTED)
  float16_t out_sp_f16[MFCC_DRIVER_NUM_DCT];
#endif
  uint8_t has_f32;
  uint8_t has_q31;
  uint8_t has_q15;
  uint8_t has_f16;
  uint8_t has_sp_f32;
  uint8_t has_sp_f16;
} case_outputs_t;

static uint64_t target_frequency = MFCC_BENCH_TARGET_FREQUENCY_HZ;
static mfcc_driver_t g_driver;
static mfcc_bench_case_t g_cases[MFCC_BENCH_NUM_CASES];
static uint8_t g_setup_done = 0U;

static check_stats_t g_check_stats[MFCC_VAR_COUNT];
static cycle_stats_t g_total_cold[MFCC_VAR_COUNT];
static cycle_stats_t g_total_warm[MFCC_VAR_COUNT];

static const char *k_var_names[MFCC_VAR_COUNT] = {
    "f32",
    "q31",
    "q15",
    "f16",
    "sp1024x23x12_f32",
    "sp1024x23x12_f16",
};

static void stats_init(cycle_stats_t *s) {
  s->sum = 0U;
  s->best = ~0ULL;
  s->worst = 0U;
  s->runs = 0U;
}

static void stats_update(cycle_stats_t *s, uint64_t cycles) {
  s->sum += cycles;
  if (cycles < s->best) {
    s->best = cycles;
  }
  if (cycles > s->worst) {
    s->worst = cycles;
  }
  s->runs += 1U;
}

static uint64_t stats_avg(const cycle_stats_t *s) {
  if (s->runs == 0U) {
    return 0U;
  }
  return s->sum / (uint64_t)s->runs;
}

static void print_input_preview(const float32_t *x) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }
  printf("    input[0:%d] =", MFCC_BENCH_PRINT_INPUT_N - 1);
  for (uint32_t i = 0; i < MFCC_BENCH_PRINT_INPUT_N; i++) {
    printf(" %0.4f", x[i]);
  }
  printf("\n");
}

static void print_vec_f32(const char *label, const float32_t *x) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }
  printf("      %s =", label);
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    printf(" %0.5f", x[i]);
  }
  printf("\n");
}

static void print_vec_q31(const char *label, const q31_t *x) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }
  printf("      %s =", label);
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    printf(" %0.5f", mfcc_driver_q31_to_float(x[i]));
  }
  printf("\n");
}

static void print_vec_q15(const char *label, const q15_t *x) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }
  printf("      %s =", label);
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    printf(" %0.5f", mfcc_driver_q15_to_float(x[i]));
  }
  printf("\n");
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static void print_vec_f16(const char *label, const float16_t *x) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }
  printf("      %s =", label);
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    printf(" %0.5f", mfcc_driver_f16_to_float(x[i]));
  }
  printf("\n");
}
#endif

static int run_f32_mode(const float32_t *input,
                        float32_t *output,
                        const char *cache_name,
                        uint8_t is_cold,
                        cycle_stats_t *total_stats) {
  cycle_stats_t local;
  mfcc_driver_status_t st = MFCC_DRIVER_OK;
  stats_init(&local);

  for (uint32_t iter = 0; iter < MFCC_BENCH_NUM_ITERATIONS; iter++) {
    uint64_t cycles = 0U;
    if (is_cold) {
      bench_cache_flush();
    }
    st = mfcc_driver_run_f32(&g_driver, input, output, &cycles);
    if (st != MFCC_DRIVER_OK) {
      if (mfcc_bench_is_print_hart()) {
        printf("      f32 run failed: %s\n", mfcc_driver_status_str(st));
      }
      return -1;
    }
    stats_update(&local, cycles);
    stats_update(total_stats, cycles);
    (void)iter;
  }

  if (mfcc_bench_is_print_hart()) {
    printf("      summary[f32][%s]: runs=%lu best=%llu avg=%llu worst=%llu\n",
           cache_name,
           (unsigned long)local.runs,
           (unsigned long long)local.best,
           (unsigned long long)stats_avg(&local),
           (unsigned long long)local.worst);
    print_vec_f32("f32_mfcc", output);
  }
  return 0;
}

static int run_q31_mode(const float32_t *input,
                        q31_t *output,
                        const char *cache_name,
                        uint8_t is_cold,
                        cycle_stats_t *total_stats) {
  cycle_stats_t local;
  mfcc_driver_status_t st = MFCC_DRIVER_OK;
  stats_init(&local);

  for (uint32_t iter = 0; iter < MFCC_BENCH_NUM_ITERATIONS; iter++) {
    uint64_t cycles = 0U;
    if (is_cold) {
      bench_cache_flush();
    }
    st = mfcc_driver_run_q31(&g_driver, input, output, &cycles);
    if (st != MFCC_DRIVER_OK) {
      if (mfcc_bench_is_print_hart()) {
        printf("      q31 run failed: %s\n", mfcc_driver_status_str(st));
      }
      return -1;
    }
    stats_update(&local, cycles);
    stats_update(total_stats, cycles);
    (void)iter;
  }

  if (mfcc_bench_is_print_hart()) {
    printf("      summary[q31][%s]: runs=%lu best=%llu avg=%llu worst=%llu\n",
           cache_name,
           (unsigned long)local.runs,
           (unsigned long long)local.best,
           (unsigned long long)stats_avg(&local),
           (unsigned long long)local.worst);
    print_vec_q31("q31_mfcc", output);
  }
  return 0;
}

static int run_q15_mode(const float32_t *input,
                        q15_t *output,
                        const char *cache_name,
                        uint8_t is_cold,
                        cycle_stats_t *total_stats) {
  cycle_stats_t local;
  mfcc_driver_status_t st = MFCC_DRIVER_OK;
  stats_init(&local);

  for (uint32_t iter = 0; iter < MFCC_BENCH_NUM_ITERATIONS; iter++) {
    uint64_t cycles = 0U;
    if (is_cold) {
      bench_cache_flush();
    }
    st = mfcc_driver_run_q15(&g_driver, input, output, &cycles);
    if (st != MFCC_DRIVER_OK) {
      if (mfcc_bench_is_print_hart()) {
        printf("      q15 run failed: %s\n", mfcc_driver_status_str(st));
      }
      return -1;
    }
    stats_update(&local, cycles);
    stats_update(total_stats, cycles);
    (void)iter;
  }

  if (mfcc_bench_is_print_hart()) {
    printf("      summary[q15][%s]: runs=%lu best=%llu avg=%llu worst=%llu\n",
           cache_name,
           (unsigned long)local.runs,
           (unsigned long long)local.best,
           (unsigned long long)stats_avg(&local),
           (unsigned long long)local.worst);
    print_vec_q15("q15_mfcc", output);
  }
  return 0;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static int run_f16_mode(const float32_t *input,
                        float16_t *output,
                        const char *cache_name,
                        uint8_t is_cold,
                        cycle_stats_t *total_stats) {
  cycle_stats_t local;
  mfcc_driver_status_t st = MFCC_DRIVER_OK;
  stats_init(&local);

  for (uint32_t iter = 0; iter < MFCC_BENCH_NUM_ITERATIONS; iter++) {
    uint64_t cycles = 0U;
    if (is_cold) {
      bench_cache_flush();
    }
    st = mfcc_driver_run_f16(&g_driver, input, output, &cycles);
    if (st != MFCC_DRIVER_OK) {
      if (mfcc_bench_is_print_hart()) {
        printf("      f16 run failed: %s\n", mfcc_driver_status_str(st));
      }
      return -1;
    }
    stats_update(&local, cycles);
    stats_update(total_stats, cycles);
    (void)iter;
  }

  if (mfcc_bench_is_print_hart()) {
    printf("      summary[f16][%s]: runs=%lu best=%llu avg=%llu worst=%llu\n",
           cache_name,
           (unsigned long)local.runs,
           (unsigned long long)local.best,
           (unsigned long long)stats_avg(&local),
           (unsigned long long)local.worst);
    print_vec_f16("f16_mfcc", output);
  }
  return 0;
}
#endif

static int run_sp_f32_mode(const float32_t *input,
                           float32_t *output,
                           const char *cache_name,
                           uint8_t is_cold,
                           cycle_stats_t *total_stats) {
  cycle_stats_t local;
  mfcc_driver_status_t st = MFCC_DRIVER_OK;
  stats_init(&local);

  for (uint32_t iter = 0; iter < MFCC_BENCH_NUM_ITERATIONS; iter++) {
    uint64_t cycles = 0U;
    if (is_cold) {
      bench_cache_flush();
    }
    st = mfcc_driver_run_sp1024x23x12_f32(&g_driver, input, output, &cycles);
    if (st != MFCC_DRIVER_OK) {
      if (mfcc_bench_is_print_hart()) {
        printf("      sp f32 run failed: %s\n", mfcc_driver_status_str(st));
      }
      return -1;
    }
    stats_update(&local, cycles);
    stats_update(total_stats, cycles);
    (void)iter;
  }

  if (mfcc_bench_is_print_hart()) {
    printf("      summary[sp1024x23x12_f32][%s]: runs=%lu best=%llu avg=%llu worst=%llu\n",
           cache_name,
           (unsigned long)local.runs,
           (unsigned long long)local.best,
           (unsigned long long)stats_avg(&local),
           (unsigned long long)local.worst);
    print_vec_f32("sp1024x23x12_f32_mfcc", output);
  }
  return 0;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static int run_sp_f16_mode(const float32_t *input,
                           float16_t *output,
                           const char *cache_name,
                           uint8_t is_cold,
                           cycle_stats_t *total_stats) {
  cycle_stats_t local;
  mfcc_driver_status_t st = MFCC_DRIVER_OK;
  stats_init(&local);

  for (uint32_t iter = 0; iter < MFCC_BENCH_NUM_ITERATIONS; iter++) {
    uint64_t cycles = 0U;
    if (is_cold) {
      bench_cache_flush();
    }
    st = mfcc_driver_run_sp1024x23x12_f16(&g_driver, input, output, &cycles);
    if (st != MFCC_DRIVER_OK) {
      if (mfcc_bench_is_print_hart()) {
        printf("      sp f16 run failed: %s\n", mfcc_driver_status_str(st));
      }
      return -1;
    }
    stats_update(&local, cycles);
    stats_update(total_stats, cycles);
    (void)iter;
  }

  if (mfcc_bench_is_print_hart()) {
    printf("      summary[sp1024x23x12_f16][%s]: runs=%lu best=%llu avg=%llu worst=%llu\n",
           cache_name,
           (unsigned long)local.runs,
           (unsigned long long)local.best,
           (unsigned long long)stats_avg(&local),
           (unsigned long long)local.worst);
    print_vec_f16("sp1024x23x12_f16_mfcc", output);
  }
  return 0;
}
#endif

static void update_check_stats(mfcc_variant_t v, int status) {
  if (status > 0) {
    g_check_stats[v].pass++;
  } else if (status == 0) {
    g_check_stats[v].fail++;
  } else {
    g_check_stats[v].skip++;
  }
}

static int compare_f32_arrays(const float32_t *a,
                              const float32_t *b,
                              float32_t tol,
                              float32_t *max_err,
                              uint32_t *max_idx) {
  float32_t err_max = 0.0f;
  uint32_t idx = 0U;
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    const float32_t err = fabsf(a[i] - b[i]);
    if (err > err_max) {
      err_max = err;
      idx = i;
    }
  }
  if (max_err != NULL) {
    *max_err = err_max;
  }
  if (max_idx != NULL) {
    *max_idx = idx;
  }
  return (err_max <= tol) ? 1 : 0;
}

static int compare_q31_to_f32(const q31_t *a,
                              const float32_t *b,
                              float32_t tol,
                              float32_t *max_err,
                              uint32_t *max_idx) {
  float32_t err_max = 0.0f;
  uint32_t idx = 0U;
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    const float32_t err = fabsf(mfcc_driver_q31_to_float(a[i]) - b[i]);
    if (err > err_max) {
      err_max = err;
      idx = i;
    }
  }
  if (max_err != NULL) {
    *max_err = err_max;
  }
  if (max_idx != NULL) {
    *max_idx = idx;
  }
  return (err_max <= tol) ? 1 : 0;
}

static int compare_q15_to_f32(const q15_t *a,
                              const float32_t *b,
                              float32_t tol,
                              float32_t *max_err,
                              uint32_t *max_idx) {
  float32_t err_max = 0.0f;
  uint32_t idx = 0U;
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    const float32_t err = fabsf(mfcc_driver_q15_to_float(a[i]) - b[i]);
    if (err > err_max) {
      err_max = err;
      idx = i;
    }
  }
  if (max_err != NULL) {
    *max_err = err_max;
  }
  if (max_idx != NULL) {
    *max_idx = idx;
  }
  return (err_max <= tol) ? 1 : 0;
}

#if defined(RISCV_FLOAT16_SUPPORTED)
static int compare_f16_to_f32(const float16_t *a,
                              const float32_t *b,
                              float32_t tol,
                              float32_t *max_err,
                              uint32_t *max_idx) {
  float32_t err_max = 0.0f;
  uint32_t idx = 0U;
  for (uint32_t i = 0; i < MFCC_DRIVER_NUM_DCT; i++) {
    const float32_t err = fabsf(mfcc_driver_f16_to_float(a[i]) - b[i]);
    if (err > err_max) {
      err_max = err;
      idx = i;
    }
  }
  if (max_err != NULL) {
    *max_err = err_max;
  }
  if (max_idx != NULL) {
    *max_idx = idx;
  }
  return (err_max <= tol) ? 1 : 0;
}
#endif

static int ref_case_is_usable(uint32_t case_idx, const char *label) {
  if ((MFCC_REF_NUM_CASES != MFCC_BENCH_NUM_CASES) ||
      (MFCC_REF_NUM_DCT != MFCC_DRIVER_NUM_DCT) ||
      (MFCC_REF_FFT_LEN != MFCC_DRIVER_FFT_LEN)) {
    if (mfcc_bench_is_print_hart()) {
      printf("    check[%s] = SKIP (reference shape/fft mismatch)\n", label);
    }
    return 0;
  }

  if (strcmp(g_cases[case_idx].name, g_mfcc_ref_case_names[case_idx]) != 0) {
    if (mfcc_bench_is_print_hart()) {
      printf("    check[%s] = SKIP (case-name mismatch: test=%s ref=%s)\n",
             label,
             g_cases[case_idx].name,
             g_mfcc_ref_case_names[case_idx]);
    }
    return 0;
  }

  return 1;
}

static void run_correctness_checks(uint32_t case_idx, const case_outputs_t *out) {
  float32_t max_err = 0.0f;
  uint32_t max_idx = 0U;

  if (mfcc_bench_is_print_hart()) {
    printf("    correctness reference = per-type goldens\n");
  }

#if MFCC_BENCH_ENABLE_F32
  if (out->has_f32) {
#if MFCC_REF_HAS_F32
    if (ref_case_is_usable(case_idx, "f32")) {
      int pass = compare_f32_arrays(
          out->out_f32, g_mfcc_ref_f32[case_idx], MFCC_REF_F32_TOL, &max_err, &max_idx);
      update_check_stats(MFCC_VAR_F32, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[f32] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F32_TOL);
      }
    } else {
      update_check_stats(MFCC_VAR_F32, -1);
    }
#else
    update_check_stats(MFCC_VAR_F32, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[f32] = SKIP (f32 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(MFCC_VAR_F32, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_Q31
  if (out->has_q31) {
#if MFCC_REF_HAS_Q31
    if (ref_case_is_usable(case_idx, "q31")) {
      int pass = compare_q31_to_f32(
          out->out_q31, g_mfcc_ref_q31[case_idx], MFCC_REF_Q31_TOL, &max_err, &max_idx);
      update_check_stats(MFCC_VAR_Q31, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[q31] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_Q31_TOL);
      }
    } else {
      update_check_stats(MFCC_VAR_Q31, -1);
    }
#else
    update_check_stats(MFCC_VAR_Q31, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[q31] = SKIP (q31 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(MFCC_VAR_Q31, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_Q15
  if (out->has_q15) {
#if MFCC_REF_HAS_Q15
    if (ref_case_is_usable(case_idx, "q15")) {
      int pass = compare_q15_to_f32(
          out->out_q15, g_mfcc_ref_q15[case_idx], MFCC_REF_Q15_TOL, &max_err, &max_idx);
      update_check_stats(MFCC_VAR_Q15, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[q15] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_Q15_TOL);
      }
    } else {
      update_check_stats(MFCC_VAR_Q15, -1);
    }
#else
    update_check_stats(MFCC_VAR_Q15, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[q15] = SKIP (q15 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(MFCC_VAR_Q15, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
  if (out->has_f16) {
#if MFCC_REF_HAS_F16
    if (ref_case_is_usable(case_idx, "f16")) {
      int pass = compare_f16_to_f32(
          out->out_f16, g_mfcc_ref_f16[case_idx], MFCC_REF_F16_TOL, &max_err, &max_idx);
      update_check_stats(MFCC_VAR_F16, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[f16] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F16_TOL);
      }
    } else {
      update_check_stats(MFCC_VAR_F16, -1);
    }
#else
    update_check_stats(MFCC_VAR_F16, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[f16] = SKIP (f16 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(MFCC_VAR_F16, -1);
  }
#else
  update_check_stats(MFCC_VAR_F16, -1);
#endif
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F32
  if (out->has_sp_f32) {
#if MFCC_REF_HAS_F32
    if (ref_case_is_usable(case_idx, "sp1024x23x12_f32")) {
      int pass = compare_f32_arrays(
          out->out_sp_f32, g_mfcc_ref_f32[case_idx], MFCC_REF_F32_TOL, &max_err, &max_idx);
      update_check_stats(MFCC_VAR_SP_F32, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[sp1024x23x12_f32] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F32_TOL);
      }
    } else {
      update_check_stats(MFCC_VAR_SP_F32, -1);
    }
#else
    update_check_stats(MFCC_VAR_SP_F32, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[sp1024x23x12_f32] = SKIP (f32 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(MFCC_VAR_SP_F32, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
  if (out->has_sp_f16) {
#if MFCC_REF_HAS_F16
    if (ref_case_is_usable(case_idx, "sp1024x23x12_f16")) {
      int pass = compare_f16_to_f32(
          out->out_sp_f16, g_mfcc_ref_f16[case_idx], MFCC_REF_F16_TOL, &max_err, &max_idx);
      update_check_stats(MFCC_VAR_SP_F16, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[sp1024x23x12_f16] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F16_TOL);
      }
    } else {
      update_check_stats(MFCC_VAR_SP_F16, -1);
    }
#else
    update_check_stats(MFCC_VAR_SP_F16, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[sp1024x23x12_f16] = SKIP (f16 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(MFCC_VAR_SP_F16, -1);
  }
#else
  update_check_stats(MFCC_VAR_SP_F16, -1);
#endif
#endif
}

static void run_case(const mfcc_bench_case_t *cs, uint32_t case_idx) {
  case_outputs_t out;
  memset(&out, 0, sizeof(out));

  if (mfcc_bench_is_print_hart()) {
    printf("\n[CASE %lu] %s\n", (unsigned long)case_idx, cs->name);
  }
  print_input_preview(cs->samples);

#if MFCC_BENCH_ENABLE_F32
#if MFCC_BENCH_RUN_COLD
  if (run_f32_mode(cs->samples, out.out_f32, "cold", 1U, &g_total_cold[MFCC_VAR_F32]) == 0) {
    out.has_f32 = 1U;
  }
#endif
#if MFCC_BENCH_RUN_WARM
  if (run_f32_mode(cs->samples, out.out_f32, "warm", 0U, &g_total_warm[MFCC_VAR_F32]) == 0) {
    out.has_f32 = 1U;
  }
#endif
#endif

#if MFCC_BENCH_ENABLE_Q31
#if MFCC_BENCH_RUN_COLD
  if (run_q31_mode(cs->samples, out.out_q31, "cold", 1U, &g_total_cold[MFCC_VAR_Q31]) == 0) {
    out.has_q31 = 1U;
  }
#endif
#if MFCC_BENCH_RUN_WARM
  if (run_q31_mode(cs->samples, out.out_q31, "warm", 0U, &g_total_warm[MFCC_VAR_Q31]) == 0) {
    out.has_q31 = 1U;
  }
#endif
#endif

#if MFCC_BENCH_ENABLE_Q15
#if MFCC_BENCH_RUN_COLD
  if (run_q15_mode(cs->samples, out.out_q15, "cold", 1U, &g_total_cold[MFCC_VAR_Q15]) == 0) {
    out.has_q15 = 1U;
  }
#endif
#if MFCC_BENCH_RUN_WARM
  if (run_q15_mode(cs->samples, out.out_q15, "warm", 0U, &g_total_warm[MFCC_VAR_Q15]) == 0) {
    out.has_q15 = 1U;
  }
#endif
#endif

#if MFCC_BENCH_ENABLE_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
#if MFCC_BENCH_RUN_COLD
  if (run_f16_mode(cs->samples, out.out_f16, "cold", 1U, &g_total_cold[MFCC_VAR_F16]) == 0) {
    out.has_f16 = 1U;
  }
#endif
#if MFCC_BENCH_RUN_WARM
  if (run_f16_mode(cs->samples, out.out_f16, "warm", 0U, &g_total_warm[MFCC_VAR_F16]) == 0) {
    out.has_f16 = 1U;
  }
#endif
#else
  if (mfcc_bench_is_print_hart()) {
    printf("      f16 disabled at compile time (RISCV_FLOAT16_SUPPORTED not set)\n");
  }
#endif
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F32
#if MFCC_BENCH_RUN_COLD
  if (run_sp_f32_mode(cs->samples,
                      out.out_sp_f32,
                      "cold",
                      1U,
                      &g_total_cold[MFCC_VAR_SP_F32]) == 0) {
    out.has_sp_f32 = 1U;
  }
#endif
#if MFCC_BENCH_RUN_WARM
  if (run_sp_f32_mode(cs->samples,
                      out.out_sp_f32,
                      "warm",
                      0U,
                      &g_total_warm[MFCC_VAR_SP_F32]) == 0) {
    out.has_sp_f32 = 1U;
  }
#endif
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
#if MFCC_BENCH_RUN_COLD
  if (run_sp_f16_mode(cs->samples,
                      out.out_sp_f16,
                      "cold",
                      1U,
                      &g_total_cold[MFCC_VAR_SP_F16]) == 0) {
    out.has_sp_f16 = 1U;
  }
#endif
#if MFCC_BENCH_RUN_WARM
  if (run_sp_f16_mode(cs->samples,
                      out.out_sp_f16,
                      "warm",
                      0U,
                      &g_total_warm[MFCC_VAR_SP_F16]) == 0) {
    out.has_sp_f16 = 1U;
  }
#endif
#else
  if (mfcc_bench_is_print_hart()) {
    printf("      sp1024x23x12_f16 disabled at compile time (RISCV_FLOAT16_SUPPORTED not set)\n");
  }
#endif
#endif

  run_correctness_checks(case_idx, &out);
}

static void print_global_cycle_summary(void) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }

  printf("\n=== MFCC Cycle Summary ===\n");
  for (uint32_t v = 0; v < MFCC_VAR_COUNT; v++) {
    const cycle_stats_t *cold = &g_total_cold[v];
    const cycle_stats_t *warm = &g_total_warm[v];
    printf("  %-18s COLD(runs=%lu best=%llu avg=%llu worst=%llu) "
           "WARM(runs=%lu best=%llu avg=%llu worst=%llu)\n",
           k_var_names[v],
           (unsigned long)cold->runs,
           (unsigned long long)(cold->runs ? cold->best : 0ULL),
           (unsigned long long)stats_avg(cold),
           (unsigned long long)cold->worst,
           (unsigned long)warm->runs,
           (unsigned long long)(warm->runs ? warm->best : 0ULL),
           (unsigned long long)stats_avg(warm),
           (unsigned long long)warm->worst);
  }
}

static void print_global_correctness_summary(void) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }

  printf("\n=== MFCC Correctness Summary ===\n");
  for (uint32_t v = 0; v < MFCC_VAR_COUNT; v++) {
    printf("  %-18s pass=%lu fail=%lu skip=%lu\n",
           k_var_names[v],
           (unsigned long)g_check_stats[v].pass,
           (unsigned long)g_check_stats[v].fail,
           (unsigned long)g_check_stats[v].skip);
  }
}

static void reset_aggregate_stats(void) {
  for (uint32_t i = 0; i < MFCC_VAR_COUNT; i++) {
    stats_init(&g_total_cold[i]);
    stats_init(&g_total_warm[i]);
    g_check_stats[i].pass = 0U;
    g_check_stats[i].fail = 0U;
    g_check_stats[i].skip = 0U;
  }
}

static int setup_once(void) {
  if (g_setup_done) {
    return 0;
  }

  bench_cache_init();
  mfcc_bench_prepare_cases(g_cases, MFCC_BENCH_NUM_CASES);
  if (mfcc_driver_init(&g_driver) != MFCC_DRIVER_OK) {
    if (mfcc_bench_is_print_hart()) {
      printf("MFCC driver init failed\n");
    }
    return -1;
  }

  reset_aggregate_stats();

  g_setup_done = 1U;
  return 0;
}

static void run_suite_for_frequency(uint64_t frequency_hz) {
  reset_aggregate_stats();

  if (mfcc_bench_is_print_hart()) {
    printf("\n=== MFCC Driver Benchmark @ %llu Hz ===\n", (unsigned long long)frequency_hz);
    printf("  iterations=%d cold=%d warm=%d\n",
           MFCC_BENCH_NUM_ITERATIONS,
           MFCC_BENCH_RUN_COLD,
           MFCC_BENCH_RUN_WARM);
    printf("  enabled: f32=%d q31=%d q15=%d f16=%d sp_f32=%d sp_f16=%d\n",
           MFCC_BENCH_ENABLE_F32,
           MFCC_BENCH_ENABLE_Q31,
           MFCC_BENCH_ENABLE_Q15,
           MFCC_BENCH_ENABLE_F16,
           MFCC_BENCH_ENABLE_SP1024X23X12_F32,
           MFCC_BENCH_ENABLE_SP1024X23X12_F16);
  }

  for (uint32_t tc = 0; tc < MFCC_BENCH_NUM_CASES; tc++) {
    run_case(&g_cases[tc], tc);
  }

  print_global_cycle_summary();
  print_global_correctness_summary();
}

void app_init(void) {
  // init_test(target_frequency);
  (void)setup_once();
}

void app_main(void) {
  run_suite_for_frequency(target_frequency);
}

#if MFCC_BENCH_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = {MFCC_BENCH_PLL_FREQ_LIST};
#endif

int main(void) {
#if MFCC_BENCH_ENABLE_PLL_SWEEP
  const size_t num_freqs = sizeof(k_pll_sweep_freqs_hz) / sizeof(k_pll_sweep_freqs_hz[0]);
  if (num_freqs == 0u) {
    return 0;
  }

  target_frequency = k_pll_sweep_freqs_hz[0];
  init_test(target_frequency);
  if (setup_once() != 0) {
    return -1;
  }
  run_suite_for_frequency(target_frequency);

  for (size_t i = 1; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, MFCC_BENCH_PLL_SWEEP_SLEEP_MS);
    run_suite_for_frequency(target_frequency);
  }
  return 0;
#else
  app_init();
  app_main();
  return 0;
#endif
}

int __main(void) {
#if MFCC_BENCH_ENABLE_PLL_SWEEP
  return main();
#else
  app_init();
  app_main();
  return 0;
#endif
}
