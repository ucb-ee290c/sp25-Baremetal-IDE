#include <math.h>
#include <stdio.h>
#include <string.h>

#include "bench_cache.h"
#include "bench_cases.h"
#include "bench_config.h"
#include "chip_config.h"
#include "hthread.h"
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

/* Per-hart state for the multicore benchmark. */
typedef struct {
  mfcc_driver_t driver;
  check_stats_t check_stats[MFCC_VAR_COUNT];
  cycle_stats_t total_cold[MFCC_VAR_COUNT];
  cycle_stats_t total_warm[MFCC_VAR_COUNT];
  uint32_t hart_id;
} hart_state_t;

typedef struct {
  hart_state_t *hs;
  const mfcc_bench_case_t *cs;
  case_outputs_t *out;
  uint32_t variant_mask;
  int status;
} case_job_t;

static uint64_t target_frequency = MFCC_BENCH_TARGET_FREQUENCY_HZ;
static mfcc_bench_case_t g_cases[MFCC_BENCH_NUM_CASES];
static uint8_t g_setup_done = 0U;

/* Two hart states: index 0 for hart 0, index 1 for hart 1. */
static hart_state_t g_hart[2];

static const char *k_var_names[MFCC_VAR_COUNT] = {
    "f32",
    "q31",
    "q15",
    "f16",
    "sp1024x23x12_f32",
    "sp1024x23x12_f16",
};

#define MFCC_VAR_BIT(v) (1u << (v))

/* Balance the per-case workload across the two harts. */
static const uint32_t k_hart0_variant_mask =
    MFCC_VAR_BIT(MFCC_VAR_F32) |
    MFCC_VAR_BIT(MFCC_VAR_Q15) |
    MFCC_VAR_BIT(MFCC_VAR_SP_F32);
static const uint32_t k_hart1_variant_mask =
    MFCC_VAR_BIT(MFCC_VAR_Q31) |
    MFCC_VAR_BIT(MFCC_VAR_F16) |
    MFCC_VAR_BIT(MFCC_VAR_SP_F16);

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

static void stats_merge(cycle_stats_t *dst, const cycle_stats_t *src) {
  dst->sum += src->sum;
  if (src->runs > 0U && src->best < dst->best) {
    dst->best = src->best;
  }
  if (src->worst > dst->worst) {
    dst->worst = src->worst;
  }
  dst->runs += src->runs;
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

static int run_f32_mode(hart_state_t *hs,
                        const float32_t *input,
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
      bench_cache_flush(hs->hart_id);
    }
    st = mfcc_driver_run_f32(&hs->driver, input, output, &cycles);
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

static int run_q31_mode(hart_state_t *hs,
                        const float32_t *input,
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
      bench_cache_flush(hs->hart_id);
    }
    st = mfcc_driver_run_q31(&hs->driver, input, output, &cycles);
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

static int run_q15_mode(hart_state_t *hs,
                        const float32_t *input,
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
      bench_cache_flush(hs->hart_id);
    }
    st = mfcc_driver_run_q15(&hs->driver, input, output, &cycles);
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
static int run_f16_mode(hart_state_t *hs,
                        const float32_t *input,
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
      bench_cache_flush(hs->hart_id);
    }
    st = mfcc_driver_run_f16(&hs->driver, input, output, &cycles);
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

static int run_sp_f32_mode(hart_state_t *hs,
                           const float32_t *input,
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
      bench_cache_flush(hs->hart_id);
    }
    st = mfcc_driver_run_sp1024x23x12_f32(&hs->driver, input, output, &cycles);
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
static int run_sp_f16_mode(hart_state_t *hs,
                           const float32_t *input,
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
      bench_cache_flush(hs->hart_id);
    }
    st = mfcc_driver_run_sp1024x23x12_f16(&hs->driver, input, output, &cycles);
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

static void update_check_stats(hart_state_t *hs, mfcc_variant_t v, int status) {
  if (status > 0) {
    hs->check_stats[v].pass++;
  } else if (status == 0) {
    hs->check_stats[v].fail++;
  } else {
    hs->check_stats[v].skip++;
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

static void run_correctness_checks(hart_state_t *hs,
                                   uint32_t case_idx,
                                   const case_outputs_t *out) {
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
      update_check_stats(hs, MFCC_VAR_F32, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[f32] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F32_TOL);
      }
    } else {
      update_check_stats(hs, MFCC_VAR_F32, -1);
    }
#else
    update_check_stats(hs, MFCC_VAR_F32, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[f32] = SKIP (f32 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(hs, MFCC_VAR_F32, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_Q31
  if (out->has_q31) {
#if MFCC_REF_HAS_Q31
    if (ref_case_is_usable(case_idx, "q31")) {
      int pass = compare_q31_to_f32(
          out->out_q31, g_mfcc_ref_q31[case_idx], MFCC_REF_Q31_TOL, &max_err, &max_idx);
      update_check_stats(hs, MFCC_VAR_Q31, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[q31] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_Q31_TOL);
      }
    } else {
      update_check_stats(hs, MFCC_VAR_Q31, -1);
    }
#else
    update_check_stats(hs, MFCC_VAR_Q31, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[q31] = SKIP (q31 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(hs, MFCC_VAR_Q31, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_Q15
  if (out->has_q15) {
#if MFCC_REF_HAS_Q15
    if (ref_case_is_usable(case_idx, "q15")) {
      int pass = compare_q15_to_f32(
          out->out_q15, g_mfcc_ref_q15[case_idx], MFCC_REF_Q15_TOL, &max_err, &max_idx);
      update_check_stats(hs, MFCC_VAR_Q15, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[q15] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_Q15_TOL);
      }
    } else {
      update_check_stats(hs, MFCC_VAR_Q15, -1);
    }
#else
    update_check_stats(hs, MFCC_VAR_Q15, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[q15] = SKIP (q15 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(hs, MFCC_VAR_Q15, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
  if (out->has_f16) {
#if MFCC_REF_HAS_F16
    if (ref_case_is_usable(case_idx, "f16")) {
      int pass = compare_f16_to_f32(
          out->out_f16, g_mfcc_ref_f16[case_idx], MFCC_REF_F16_TOL, &max_err, &max_idx);
      update_check_stats(hs, MFCC_VAR_F16, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[f16] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F16_TOL);
      }
    } else {
      update_check_stats(hs, MFCC_VAR_F16, -1);
    }
#else
    update_check_stats(hs, MFCC_VAR_F16, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[f16] = SKIP (f16 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(hs, MFCC_VAR_F16, -1);
  }
#else
  update_check_stats(hs, MFCC_VAR_F16, -1);
#endif
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F32
  if (out->has_sp_f32) {
#if MFCC_REF_HAS_F32
    if (ref_case_is_usable(case_idx, "sp1024x23x12_f32")) {
      int pass = compare_f32_arrays(
          out->out_sp_f32, g_mfcc_ref_f32[case_idx], MFCC_REF_F32_TOL, &max_err, &max_idx);
      update_check_stats(hs, MFCC_VAR_SP_F32, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[sp1024x23x12_f32] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F32_TOL);
      }
    } else {
      update_check_stats(hs, MFCC_VAR_SP_F32, -1);
    }
#else
    update_check_stats(hs, MFCC_VAR_SP_F32, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[sp1024x23x12_f32] = SKIP (f32 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(hs, MFCC_VAR_SP_F32, -1);
  }
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
  if (out->has_sp_f16) {
#if MFCC_REF_HAS_F16
    if (ref_case_is_usable(case_idx, "sp1024x23x12_f16")) {
      int pass = compare_f16_to_f32(
          out->out_sp_f16, g_mfcc_ref_f16[case_idx], MFCC_REF_F16_TOL, &max_err, &max_idx);
      update_check_stats(hs, MFCC_VAR_SP_F16, pass);
      if (mfcc_bench_is_print_hart()) {
        printf("    check[sp1024x23x12_f16] = %s max_abs_err=%0.6f idx=%lu tol=%0.3f\n",
               pass ? "PASS" : "FAIL",
               max_err,
               (unsigned long)max_idx,
               MFCC_REF_F16_TOL);
      }
    } else {
      update_check_stats(hs, MFCC_VAR_SP_F16, -1);
    }
#else
    update_check_stats(hs, MFCC_VAR_SP_F16, -1);
    if (mfcc_bench_is_print_hart()) {
      printf("    check[sp1024x23x12_f16] = SKIP (f16 golden disabled)\n");
    }
#endif
  } else {
    update_check_stats(hs, MFCC_VAR_SP_F16, -1);
  }
#else
  update_check_stats(hs, MFCC_VAR_SP_F16, -1);
#endif
#endif
}

static int run_case_variants(hart_state_t *hs,
                             const mfcc_bench_case_t *cs,
                             case_outputs_t *out,
                             uint32_t variant_mask) {
  int status = 0;

#if MFCC_BENCH_ENABLE_F32
  if ((variant_mask & MFCC_VAR_BIT(MFCC_VAR_F32)) != 0U) {
#if MFCC_BENCH_RUN_COLD
    if (run_f32_mode(hs, cs->samples, out->out_f32, "cold", 1U, &hs->total_cold[MFCC_VAR_F32]) == 0) {
      out->has_f32 = 1U;
    } else {
      status = -1;
    }
#endif
#if MFCC_BENCH_RUN_WARM
    if (run_f32_mode(hs, cs->samples, out->out_f32, "warm", 0U, &hs->total_warm[MFCC_VAR_F32]) == 0) {
      out->has_f32 = 1U;
    } else {
      status = -1;
    }
#endif
  }
#endif

#if MFCC_BENCH_ENABLE_Q31
  if ((variant_mask & MFCC_VAR_BIT(MFCC_VAR_Q31)) != 0U) {
#if MFCC_BENCH_RUN_COLD
    if (run_q31_mode(hs, cs->samples, out->out_q31, "cold", 1U, &hs->total_cold[MFCC_VAR_Q31]) == 0) {
      out->has_q31 = 1U;
    } else {
      status = -1;
    }
#endif
#if MFCC_BENCH_RUN_WARM
    if (run_q31_mode(hs, cs->samples, out->out_q31, "warm", 0U, &hs->total_warm[MFCC_VAR_Q31]) == 0) {
      out->has_q31 = 1U;
    } else {
      status = -1;
    }
#endif
  }
#endif

#if MFCC_BENCH_ENABLE_Q15
  if ((variant_mask & MFCC_VAR_BIT(MFCC_VAR_Q15)) != 0U) {
#if MFCC_BENCH_RUN_COLD
    if (run_q15_mode(hs, cs->samples, out->out_q15, "cold", 1U, &hs->total_cold[MFCC_VAR_Q15]) == 0) {
      out->has_q15 = 1U;
    } else {
      status = -1;
    }
#endif
#if MFCC_BENCH_RUN_WARM
    if (run_q15_mode(hs, cs->samples, out->out_q15, "warm", 0U, &hs->total_warm[MFCC_VAR_Q15]) == 0) {
      out->has_q15 = 1U;
    } else {
      status = -1;
    }
#endif
  }
#endif

#if MFCC_BENCH_ENABLE_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
  if ((variant_mask & MFCC_VAR_BIT(MFCC_VAR_F16)) != 0U) {
#if MFCC_BENCH_RUN_COLD
    if (run_f16_mode(hs, cs->samples, out->out_f16, "cold", 1U, &hs->total_cold[MFCC_VAR_F16]) == 0) {
      out->has_f16 = 1U;
    } else {
      status = -1;
    }
#endif
#if MFCC_BENCH_RUN_WARM
    if (run_f16_mode(hs, cs->samples, out->out_f16, "warm", 0U, &hs->total_warm[MFCC_VAR_F16]) == 0) {
      out->has_f16 = 1U;
    } else {
      status = -1;
    }
#endif
  }
#else
  if (((variant_mask & MFCC_VAR_BIT(MFCC_VAR_F16)) != 0U) && mfcc_bench_is_print_hart()) {
    printf("      f16 disabled at compile time (RISCV_FLOAT16_SUPPORTED not set)\n");
  }
#endif
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F32
  if ((variant_mask & MFCC_VAR_BIT(MFCC_VAR_SP_F32)) != 0U) {
#if MFCC_BENCH_RUN_COLD
    if (run_sp_f32_mode(hs, cs->samples,
                        out->out_sp_f32,
                        "cold",
                        1U,
                        &hs->total_cold[MFCC_VAR_SP_F32]) == 0) {
      out->has_sp_f32 = 1U;
    } else {
      status = -1;
    }
#endif
#if MFCC_BENCH_RUN_WARM
    if (run_sp_f32_mode(hs, cs->samples,
                        out->out_sp_f32,
                        "warm",
                        0U,
                        &hs->total_warm[MFCC_VAR_SP_F32]) == 0) {
      out->has_sp_f32 = 1U;
    } else {
      status = -1;
    }
#endif
  }
#endif

#if MFCC_BENCH_ENABLE_SP1024X23X12_F16
#if defined(RISCV_FLOAT16_SUPPORTED)
  if ((variant_mask & MFCC_VAR_BIT(MFCC_VAR_SP_F16)) != 0U) {
#if MFCC_BENCH_RUN_COLD
    if (run_sp_f16_mode(hs, cs->samples,
                        out->out_sp_f16,
                        "cold",
                        1U,
                        &hs->total_cold[MFCC_VAR_SP_F16]) == 0) {
      out->has_sp_f16 = 1U;
    } else {
      status = -1;
    }
#endif
#if MFCC_BENCH_RUN_WARM
    if (run_sp_f16_mode(hs, cs->samples,
                        out->out_sp_f16,
                        "warm",
                        0U,
                        &hs->total_warm[MFCC_VAR_SP_F16]) == 0) {
      out->has_sp_f16 = 1U;
    } else {
      status = -1;
    }
#endif
  }
#else
  if (((variant_mask & MFCC_VAR_BIT(MFCC_VAR_SP_F16)) != 0U) && mfcc_bench_is_print_hart()) {
    printf("      sp1024x23x12_f16 disabled at compile time (RISCV_FLOAT16_SUPPORTED not set)\n");
  }
#endif
#endif

  return status;
}

static void reset_hart_stats(hart_state_t *hs) {
  for (uint32_t i = 0; i < MFCC_VAR_COUNT; i++) {
    stats_init(&hs->total_cold[i]);
    stats_init(&hs->total_warm[i]);
    hs->check_stats[i].pass = 0U;
    hs->check_stats[i].fail = 0U;
    hs->check_stats[i].skip = 0U;
  }
}

/* __main() for secondary harts is provided by threadlib. */

static void hart1_worker(void *arg) {
  case_job_t *job = (case_job_t *)arg;
  job->status = run_case_variants(job->hs, job->cs, job->out, job->variant_mask);
}

static void mc_nop(void *arg) {
  (void)arg;
}

static void run_case_multicore(uint32_t case_idx) {
  case_outputs_t out;
  case_job_t job;
  int h0_status = 0;

  memset(&out, 0, sizeof(out));
  memset(&job, 0, sizeof(job));

  if (mfcc_bench_is_print_hart()) {
    printf("\n[CASE %lu] %s\n", (unsigned long)case_idx, g_cases[case_idx].name);
  }
  print_input_preview(g_cases[case_idx].samples);

  job.hs = &g_hart[1];
  job.cs = &g_cases[case_idx];
  job.out = &out;
  job.variant_mask = k_hart1_variant_mask;
  job.status = 0;

  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1, hart1_worker, &job);

  h0_status = run_case_variants(&g_hart[0], &g_cases[case_idx], &out, k_hart0_variant_mask);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");

  if ((h0_status != 0 || job.status != 0) && mfcc_bench_is_print_hart()) {
    printf("    warning: runtime errors seen (hart0=%d hart1=%d), correctness may be skipped per variant\n",
           h0_status,
           job.status);
  }

  run_correctness_checks(&g_hart[0], case_idx, &out);
}

static void print_global_cycle_summary(const cycle_stats_t total_cold[MFCC_VAR_COUNT],
                                       const cycle_stats_t total_warm[MFCC_VAR_COUNT]) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }

  printf("\n=== MFCC Cycle Summary (Multicore) ===\n");
  for (uint32_t v = 0; v < MFCC_VAR_COUNT; v++) {
    const cycle_stats_t *cold = &total_cold[v];
    const cycle_stats_t *warm = &total_warm[v];
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

static void print_global_correctness_summary(const check_stats_t check_stats[MFCC_VAR_COUNT]) {
  if (!mfcc_bench_is_print_hart()) {
    return;
  }

  printf("\n=== MFCC Correctness Summary (Multicore) ===\n");
  for (uint32_t v = 0; v < MFCC_VAR_COUNT; v++) {
    printf("  %-18s pass=%lu fail=%lu skip=%lu\n",
           k_var_names[v],
           (unsigned long)check_stats[v].pass,
           (unsigned long)check_stats[v].fail,
           (unsigned long)check_stats[v].skip);
  }
}

static int setup_once(void) {
  if (g_setup_done) {
    return 0;
  }

  bench_cache_init();
  mfcc_bench_prepare_cases(g_cases, MFCC_BENCH_NUM_CASES);

  g_hart[0].hart_id = 0U;
  g_hart[1].hart_id = 1U;

  /* Initialize both driver instances. */
  if (mfcc_driver_init(&g_hart[0].driver) != MFCC_DRIVER_OK) {
    if (mfcc_bench_is_print_hart()) {
      printf("MFCC driver init failed (hart 0)\n");
    }
    return -1;
  }
  if (mfcc_driver_init(&g_hart[1].driver) != MFCC_DRIVER_OK) {
    if (mfcc_bench_is_print_hart()) {
      printf("MFCC driver init failed (hart 1)\n");
    }
    return -1;
  }

  reset_hart_stats(&g_hart[0]);
  reset_hart_stats(&g_hart[1]);

  g_setup_done = 1U;
  return 0;
}

static void run_suite_for_frequency(uint64_t frequency_hz) {
  reset_hart_stats(&g_hart[0]);
  reset_hart_stats(&g_hart[1]);

  if (mfcc_bench_is_print_hart()) {
    printf("\n=== MFCC Driver Benchmark (Multicore) @ %llu Hz ===\n",
           (unsigned long long)frequency_hz);
    printf("  mode: 2-core per-case variant split\n");
    printf("  hart0 variants: f32, q15, sp1024x23x12_f32\n");
    printf("  hart1 variants: q31, f16, sp1024x23x12_f16\n");
    printf("  reference header: FFT_LEN=%d CASES=%d DCT=%d\n",
           MFCC_REF_FFT_LEN, MFCC_REF_NUM_CASES, MFCC_REF_NUM_DCT);
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
    run_case_multicore(tc);
  }

  /* Merge stats from both harts. */
  cycle_stats_t merged_cold[MFCC_VAR_COUNT];
  cycle_stats_t merged_warm[MFCC_VAR_COUNT];
  check_stats_t merged_check[MFCC_VAR_COUNT];
  for (uint32_t v = 0; v < MFCC_VAR_COUNT; v++) {
    merged_cold[v] = g_hart[0].total_cold[v];
    stats_merge(&merged_cold[v], &g_hart[1].total_cold[v]);
    merged_warm[v] = g_hart[0].total_warm[v];
    stats_merge(&merged_warm[v], &g_hart[1].total_warm[v]);
    merged_check[v].pass = g_hart[0].check_stats[v].pass;
    merged_check[v].fail = g_hart[0].check_stats[v].fail;
    merged_check[v].skip = g_hart[0].check_stats[v].skip;
  }

  print_global_cycle_summary(merged_cold, merged_warm);
  print_global_correctness_summary(merged_check);
}

void app_init(void) {
  init_test(target_frequency);
  hthread_init();
  /* Warm hart 1 once so threadlib secondary loop is active before timed work. */
  printf("Warming hart 1...\n");
  hthread_issue(1, mc_nop, NULL);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");
  printf("Hart 1 warmed, entering main loop.\n");
  if (setup_once() != 0) {
    if (mfcc_bench_is_print_hart()) {
      printf("MFCC benchmark setup failed\n");
    }
    while (1) {
      asm volatile("wfi");
    }
  }
}

void app_main(void) {
  run_suite_for_frequency(target_frequency);
  // printf("MFCC benchmark main loop not implemented (PLL sweep disabled)\n");
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
  hthread_init();
  hthread_issue(1, mc_nop, NULL);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");
  if (setup_once() != 0) {
    return -1;
  }
  run_suite_for_frequency(target_frequency);

  for (size_t i = 1; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, MFCC_BENCH_PLL_SWEEP_SLEEP_MS);
    hthread_issue(1, mc_nop, NULL);
    hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");
    run_suite_for_frequency(target_frequency);
  }
  return 0;
#else
  app_init();
  app_main();
  return 0;
#endif
}
