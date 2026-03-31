#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bench_cache.h"
#include "bench_config.h"
#include "chip_config.h"
#include "hal_conv.h"
#include "simple_setup.h"

#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0]))

#define CONV_BENCH_MAX_REF_OUTPUT_WORDS \
  (CONV_BENCH_MAX_N + ((CONV_BENCH_MAX_K - 1u) * CONV_BENCH_MAX_DILATION) + 8u)

#define CONV_BENCH_MAX_HW_OUTPUT_WORDS \
  (CONV_BENCH_MAX_N + CONV_BENCH_MAX_K + 8u)

typedef enum {
  CONV_CACHE_COLD = 0,
  CONV_CACHE_WARM = 1,
} conv_cache_mode_t;

typedef struct {
  uint64_t best;
  uint64_t worst;
  uint64_t sum;
  uint32_t runs;
  uint32_t status_fail;
} cycle_stats_t;

typedef struct {
  uint32_t samples;
  uint32_t mismatches;
  float max_abs;
  double sum_abs;
  double sum_sq;
} accuracy_stats_t;

typedef struct {
  const char *name;
  uintptr_t input_base;
  uintptr_t kernel_base;
  uintptr_t output_base;
  uint32_t input_capacity_bytes;
  uint32_t kernel_capacity_bytes;
  uint32_t output_capacity_bytes;
  uint8_t enabled;
} mem_mode_t;

typedef struct {
  uint32_t total_cases;
  uint32_t run_cases;
  uint32_t skipped_cases;
  uint32_t total_status_fails;
  uint32_t total_mismatches;
} suite_summary_t;

static uint64_t target_frequency = CONV_BENCH_TARGET_FREQUENCY_HZ;

static const uint32_t k_sweep_n[] = { CONV_BENCH_N_LIST };
static const uint32_t k_sweep_k[] = { CONV_BENCH_K_LIST };
static const uint32_t k_sweep_d[] = { CONV_BENCH_DILATION_LIST };

static const mem_mode_t k_mem_modes[] = {
  {
      "dram_l2",
      CONV_BENCH_DRAM_INPUT_BASE,
      CONV_BENCH_DRAM_KERNEL_BASE,
      CONV_BENCH_DRAM_OUTPUT_BASE,
      CONV_BENCH_DRAM_REGION_BYTES,
      CONV_BENCH_DRAM_REGION_BYTES,
      CONV_BENCH_DRAM_REGION_BYTES,
      CONV_BENCH_ENABLE_MEMMODE_DRAM_L2 ? 1u : 0u,
  },
  {
      "scratchpad",
      CONV_BENCH_SCRATCH_INPUT_BASE,
      CONV_BENCH_SCRATCH_KERNEL_BASE,
      CONV_BENCH_SCRATCH_OUTPUT_BASE,
      CONV_BENCH_SCRATCH_REGION_BYTES,
      CONV_BENCH_SCRATCH_REGION_BYTES,
      CONV_BENCH_SCRATCH_REGION_BYTES,
      CONV_BENCH_ENABLE_MEMMODE_SCRATCHPAD ? 1u : 0u,
  },
};

static uint32_t g_hw_capture[CONV_BENCH_MAX_HW_OUTPUT_WORDS];
static float g_ref_output[CONV_BENCH_MAX_REF_OUTPUT_WORDS];

static const char *cache_mode_name(conv_cache_mode_t mode) {
  return (mode == CONV_CACHE_COLD) ? "cold" : "warm";
}

static const char *reuse_mode_name(bool reuse_mode) {
  return reuse_mode ? "reuse_on" : "reuse_off";
}

static const char *status_to_name(uint8_t status) {
  if (status == 0u) {
    return "idle";
  }
  if (status == STATUS_COMPL) {
    return "complete";
  }
  if (status & STATUS_ERROR) {
    return "error";
  }
  if (status & STATUS_INVALID) {
    return "invalid";
  }
  if (status & STATUS_INFINITE) {
    return "infinite";
  }
  if (status & STATUS_OVERFLOW) {
    return "overflow";
  }
  if (status & STATUS_UNDERFLOW) {
    return "underflow";
  }
  if (status & STATUS_BUSY) {
    return "busy";
  }
  return "other";
}

static inline float bits_to_f32(uint32_t x) {
  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = x;
  return conv.f;
}

static inline void full_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline uint32_t hw_output_words(uint32_t n, uint32_t k) {
  return ((n / 2u) + (k / 2u)) * 2u;
}

static inline uint32_t ref_output_words(uint32_t n, uint32_t k, uint32_t dilation) {
  return n + ((k - 1u) * dilation);
}

static bool is_status_ok(uint8_t status) {
  const uint8_t fatal_mask = STATUS_ERROR | STATUS_INVALID | STATUS_INFINITE | STATUS_OVERFLOW | STATUS_UNDERFLOW;
  if ((status & fatal_mask) != 0u) {
    return false;
  }
  if (status == STATUS_BUSY) {
    return false;
  }
  return true;
}

static void stats_init(cycle_stats_t *stats) {
  stats->best = UINT64_MAX;
  stats->worst = 0u;
  stats->sum = 0u;
  stats->runs = 0u;
  stats->status_fail = 0u;
}

static void stats_record(cycle_stats_t *stats, uint64_t cycles) {
  if (cycles < stats->best) {
    stats->best = cycles;
  }
  if (cycles > stats->worst) {
    stats->worst = cycles;
  }
  stats->sum += cycles;
  stats->runs += 1u;
}

static uint64_t stats_avg(const cycle_stats_t *stats) {
  if (stats->runs == 0u) {
    return 0u;
  }
  return stats->sum / (uint64_t)stats->runs;
}

static void accuracy_init(accuracy_stats_t *acc) {
  acc->samples = 0u;
  acc->mismatches = 0u;
  acc->max_abs = 0.0f;
  acc->sum_abs = 0.0;
  acc->sum_sq = 0.0;
}

static void accuracy_update(accuracy_stats_t *acc,
                            const uint32_t *hw,
                            uint32_t hw_words,
                            const float *ref,
                            uint32_t ref_words) {
  uint32_t n = hw_words < ref_words ? hw_words : ref_words;
  for (uint32_t i = 0; i < n; ++i) {
    float hw_f = bits_to_f32(hw[i]);
    float ref_f = ref[i];

    float abs_err;
    float rel_err;

    if (!isfinite(hw_f) || !isfinite(ref_f)) {
      abs_err = INFINITY;
      rel_err = INFINITY;
      acc->mismatches += 1u;
      acc->samples += 1u;
      acc->sum_abs += 1.0;
      acc->sum_sq += 1.0;
      if (acc->max_abs < 1.0f) {
        acc->max_abs = 1.0f;
      }
      continue;
    }

    abs_err = fabsf(hw_f - ref_f);
    rel_err = abs_err / (fabsf(ref_f) + CONV_BENCH_ACC_EPSILON);

    if (abs_err > acc->max_abs) {
      acc->max_abs = abs_err;
    }
    acc->sum_abs += (double)abs_err;
    acc->sum_sq += (double)abs_err * (double)abs_err;
    if (abs_err > CONV_BENCH_ACCURACY_ABS_TOL && rel_err > CONV_BENCH_ACCURACY_REL_TOL) {
      acc->mismatches += 1u;
    }
    acc->samples += 1u;
  }

  if (ref_words > n) {
    uint32_t missing = ref_words - n;
    acc->mismatches += missing;
    acc->samples += missing;
    acc->sum_abs += (double)missing;
    acc->sum_sq += (double)missing;
    if (acc->max_abs < 1.0f) {
      acc->max_abs = 1.0f;
    }
  }
}

static float accuracy_rmse(const accuracy_stats_t *acc) {
  if (acc->samples == 0u) {
    return 0.0f;
  }
  return (float)sqrt(acc->sum_sq / (double)acc->samples);
}

static float accuracy_mean_abs(const accuracy_stats_t *acc) {
  if (acc->samples == 0u) {
    return 0.0f;
  }
  return (float)(acc->sum_abs / (double)acc->samples);
}

static void clear_u32(volatile uint32_t *dst, uint32_t words) {
  for (uint32_t i = 0; i < words; ++i) {
    dst[i] = 0u;
  }
}

static void copy_u32(volatile uint32_t *dst, const uint32_t *src, uint32_t words) {
  for (uint32_t i = 0; i < words; ++i) {
    dst[i] = src[i];
  }
}

static void copy_from_volatile_u32(uint32_t *dst, volatile const uint32_t *src, uint32_t words) {
  for (uint32_t i = 0; i < words; ++i) {
    dst[i] = src[i];
  }
}

static void stream_touch(volatile const uint32_t *buf, uint32_t words, uint32_t stride_words) {
  if (stride_words == 0u) {
    stride_words = 16u;
  }
  volatile uint32_t sink = 0u;
  for (uint32_t i = 0; i < words; i += stride_words) {
    sink ^= buf[i];
  }
  asm volatile("" :: "r"(sink) : "memory");
}

static void prepare_cache_mode(conv_cache_mode_t cache_mode,
                               volatile const uint32_t *in,
                               uint32_t in_words,
                               volatile const uint32_t *kernel,
                               uint32_t kernel_words,
                               volatile const uint32_t *out,
                               uint32_t out_words) {
  if (cache_mode == CONV_CACHE_COLD) {
    bench_cache_flush();
    return;
  }

  stream_touch(in, in_words, 16u);
  stream_touch(kernel, kernel_words, 4u);
  stream_touch(out, out_words, 16u);
}

static bool case_supported(uint32_t n, uint32_t k, uint32_t dilation, char *reason, size_t reason_cap) {
  if (n == 0u || k == 0u || dilation == 0u) {
    snprintf(reason, reason_cap, "N/K/D must be non-zero");
    return false;
  }

  if ((n & 1u) != 0u) {
    snprintf(reason, reason_cap, "N must be even (driver streams 2 fp32 per packet)");
    return false;
  }

  if (k != 8u && k != 16u) {
    snprintf(reason, reason_cap, "K=%u unsupported by driver (supported: 8,16)", k);
    return false;
  }

  if (n > CONV_BENCH_MAX_N) {
    snprintf(reason, reason_cap, "N=%u exceeds CONV_BENCH_MAX_N=%u", n, (unsigned)CONV_BENCH_MAX_N);
    return false;
  }

  if (k > CONV_BENCH_MAX_K) {
    snprintf(reason, reason_cap, "K=%u exceeds CONV_BENCH_MAX_K=%u", k, (unsigned)CONV_BENCH_MAX_K);
    return false;
  }

  if (dilation > CONV_BENCH_MAX_DILATION) {
    snprintf(reason, reason_cap, "D=%u exceeds CONV_BENCH_MAX_DILATION=%u", dilation, (unsigned)CONV_BENCH_MAX_DILATION);
    return false;
  }

  if (hw_output_words(n, k) > CONV_BENCH_MAX_HW_OUTPUT_WORDS) {
    snprintf(reason, reason_cap, "HW output bound too small");
    return false;
  }

  if (ref_output_words(n, k, dilation) > CONV_BENCH_MAX_REF_OUTPUT_WORDS) {
    snprintf(reason, reason_cap, "Reference output bound too small");
    return false;
  }

  reason[0] = '\0';
  return true;
}

static bool memory_mode_fits(const mem_mode_t *mode,
                             uint32_t n,
                             uint32_t k,
                             uint32_t hw_out_words,
                             char *reason,
                             size_t reason_cap) {
  uint32_t in_bytes = n * (uint32_t)sizeof(uint32_t);
  uint32_t k_bytes = k * (uint32_t)sizeof(uint32_t);
  uint32_t out_bytes = hw_out_words * (uint32_t)sizeof(uint32_t);

  if (in_bytes > mode->input_capacity_bytes) {
    snprintf(reason, reason_cap, "input bytes=%u exceeds mode capacity=%u", in_bytes, mode->input_capacity_bytes);
    return false;
  }
  if (k_bytes > mode->kernel_capacity_bytes) {
    snprintf(reason, reason_cap, "kernel bytes=%u exceeds mode capacity=%u", k_bytes, mode->kernel_capacity_bytes);
    return false;
  }
  if (out_bytes > mode->output_capacity_bytes) {
    snprintf(reason, reason_cap, "output bytes=%u exceeds mode capacity=%u", out_bytes, mode->output_capacity_bytes);
    return false;
  }

  reason[0] = '\0';
  return true;
}

static void select_case_data(uint32_t dataset_idx,
                             bool reuse_mode,
                             uint32_t run_idx,
                             uint32_t n,
                             uint32_t k,
                             volatile uint32_t *input_dst,
                             volatile uint32_t *kernel_dst) {
  uint32_t ds = dataset_idx % CONV_BENCH_GENERATED_NUM_DATASETS;

  uint32_t in_shift = 0u;
  uint32_t k_shift = 0u;

  if (!reuse_mode) {
    uint32_t max_in_shift = CONV_BENCH_GENERATED_MAX_N - n;
    uint32_t max_k_shift = CONV_BENCH_GENERATED_MAX_K - k;

    if (max_in_shift > 0u) {
      in_shift = (run_idx * CONV_BENCH_NO_REUSE_SHIFT_STRIDE) % (max_in_shift + 1u);
    }
    if (max_k_shift > 0u) {
      k_shift = (run_idx * (CONV_BENCH_NO_REUSE_SHIFT_STRIDE + 7u)) % (max_k_shift + 1u);
    }
  }

  copy_u32(input_dst, &g_conv_bench_generated_inputs[ds][in_shift], n);
  copy_u32(kernel_dst, &g_conv_bench_generated_kernels[ds][k_shift], k);
}

static void print_result_line(uint64_t frequency_hz,
                              const mem_mode_t *mem_mode,
                              conv_cache_mode_t cache_mode,
                              bool reuse_mode,
                              uint32_t dataset_idx,
                              uint32_t n,
                              uint32_t k,
                              uint32_t dilation,
                              uint32_t runs_requested,
                              const cycle_stats_t *cy,
                              const accuracy_stats_t *acc,
                              uint32_t hw_words,
                              uint32_t ref_words) {
  uint64_t avg_cycles = stats_avg(cy);

  double elem_per_cycle = (avg_cycles == 0u) ? 0.0 : ((double)hw_words / (double)avg_cycles);
  double mac_per_cycle = (avg_cycles == 0u) ? 0.0 : (((double)n * (double)k) / (double)avg_cycles);

  uint64_t moved_bytes = (uint64_t)(n + k + hw_words) * (uint64_t)sizeof(uint32_t);
  double mbps = (avg_cycles == 0u || frequency_hz == 0u)
                    ? 0.0
                    : ((double)moved_bytes * (double)frequency_hz) / ((double)avg_cycles * 1000000.0);

  printf("RESULT,"
         "pll_hz=%" PRIu64 ","
         "mem=%s,"
         "cache=%s,"
         "reuse=%s,"
         "dataset=%u(%s),"
         "N=%u,K=%u,D=%u,"
         "runs_req=%u,runs_ok=%u,status_fail=%u,"
         "best=%" PRIu64 ",avg=%" PRIu64 ",worst=%" PRIu64 ","
         "elem_per_cycle=%.6f,mac_per_cycle=%.6f,mbps=%.3f,"
         "compare_len=%u,ref_len=%u,mismatch=%u,max_abs=%.6e,mean_abs=%.6e,rmse=%.6e\n",
         frequency_hz,
         mem_mode->name,
         cache_mode_name(cache_mode),
         reuse_mode_name(reuse_mode),
         dataset_idx,
         g_conv_bench_dataset_names[dataset_idx % CONV_BENCH_GENERATED_NUM_DATASETS],
         n,
         k,
         dilation,
         runs_requested,
         cy->runs,
         cy->status_fail,
         (cy->runs == 0u) ? 0u : cy->best,
         avg_cycles,
         cy->worst,
         elem_per_cycle,
         mac_per_cycle,
         mbps,
         (hw_words < ref_words) ? hw_words : ref_words,
         ref_words,
         acc->mismatches,
         (double)acc->max_abs,
         (double)accuracy_mean_abs(acc),
         (double)accuracy_rmse(acc));
}

static void run_one_case(uint64_t frequency_hz,
                         const mem_mode_t *mem_mode,
                         conv_cache_mode_t cache_mode,
                         bool reuse_mode,
                         uint32_t dataset_idx,
                         uint32_t n,
                         uint32_t k,
                         uint32_t dilation,
                         suite_summary_t *summary) {
  char reason[160];
  uint32_t out_words_hw = hw_output_words(n, k);
  uint32_t out_words_ref = ref_output_words(n, k, dilation);

  summary->total_cases += 1u;

  if (!case_supported(n, k, dilation, reason, sizeof(reason))) {
    summary->skipped_cases += 1u;
    if (conv_bench_is_print_hart() && CONV_BENCH_PRINT_CASE_DETAILS) {
      printf("SKIP,mem=%s,cache=%s,reuse=%s,dataset=%u,N=%u,K=%u,D=%u,reason=%s\n",
             mem_mode->name,
             cache_mode_name(cache_mode),
             reuse_mode_name(reuse_mode),
             dataset_idx,
             n,
             k,
             dilation,
             reason);
    }
    return;
  }

  if (!memory_mode_fits(mem_mode, n, k, out_words_hw, reason, sizeof(reason))) {
    summary->skipped_cases += 1u;
    if (conv_bench_is_print_hart() && CONV_BENCH_PRINT_CASE_DETAILS) {
      printf("SKIP,mem=%s,cache=%s,reuse=%s,dataset=%u,N=%u,K=%u,D=%u,reason=%s\n",
             mem_mode->name,
             cache_mode_name(cache_mode),
             reuse_mode_name(reuse_mode),
             dataset_idx,
             n,
             k,
             dilation,
             reason);
    }
    return;
  }

  volatile uint32_t *input_ptr = (volatile uint32_t *)mem_mode->input_base;
  volatile uint32_t *kernel_ptr = (volatile uint32_t *)mem_mode->kernel_base;
  volatile uint32_t *output_ptr = (volatile uint32_t *)mem_mode->output_base;

  uint32_t runs = (cache_mode == CONV_CACHE_COLD) ? CONV_BENCH_RUNS_COLD : CONV_BENCH_RUNS_WARM;
  cycle_stats_t cycles;
  accuracy_stats_t acc;

  stats_init(&cycles);
  accuracy_init(&acc);

  for (uint32_t run = 0; run < runs; ++run) {
    if (!reuse_mode || run == 0u) {
      select_case_data(dataset_idx, reuse_mode, run, n, k, input_ptr, kernel_ptr);
    }

    clear_u32(output_ptr, out_words_hw);
    prepare_cache_mode(cache_mode, input_ptr, n, kernel_ptr, k, output_ptr, out_words_hw);

    full_fence();
    uint64_t t0 = conv_bench_rdcycle64();
    uint8_t status = perform_convolution_1D((uint32_t *)input_ptr,
                                            n,
                                            (uint32_t *)kernel_ptr,
                                            (uint8_t)k,
                                            (uint32_t *)output_ptr,
                                            (uint16_t)dilation);
    uint64_t elapsed = conv_bench_rdcycle64() - t0;
    full_fence();

    if (!is_status_ok(status)) {
      cycles.status_fail += 1u;
      if (conv_bench_is_print_hart() && CONV_BENCH_PRINT_CASE_DETAILS) {
        printf("WARN,status_fail,mem=%s,cache=%s,reuse=%s,dataset=%u,N=%u,K=%u,D=%u,run=%u,status=0x%02X(%s)\n",
               mem_mode->name,
               cache_mode_name(cache_mode),
               reuse_mode_name(reuse_mode),
               dataset_idx,
               n,
               k,
               dilation,
               run,
               status,
               status_to_name(status));
      }
      continue;
    }

    stats_record(&cycles, elapsed);

    copy_from_volatile_u32(g_hw_capture, output_ptr, out_words_hw);
    perform_naive_convolution_1D((uint32_t *)input_ptr,
                                 n,
                                 (uint32_t *)kernel_ptr,
                                 k,
                                 dilation,
                                 g_ref_output);
    accuracy_update(&acc, g_hw_capture, out_words_hw, g_ref_output, out_words_ref);
  }

  summary->run_cases += 1u;
  summary->total_status_fails += cycles.status_fail;
  summary->total_mismatches += acc.mismatches;

  if (conv_bench_is_print_hart()) {
    print_result_line(frequency_hz,
                      mem_mode,
                      cache_mode,
                      reuse_mode,
                      dataset_idx,
                      n,
                      k,
                      dilation,
                      runs,
                      &cycles,
                      &acc,
                      out_words_hw,
                      out_words_ref);
  }
}

static uint32_t active_dataset_count(void) {
  uint32_t n = CONV_BENCH_DATASET_COUNT;
  if (n > CONV_BENCH_GENERATED_NUM_DATASETS) {
    n = CONV_BENCH_GENERATED_NUM_DATASETS;
  }
  return n;
}

static void run_suite_for_frequency(uint64_t frequency_hz) {
  suite_summary_t summary;
  summary.total_cases = 0u;
  summary.run_cases = 0u;
  summary.skipped_cases = 0u;
  summary.total_status_fails = 0u;
  summary.total_mismatches = 0u;

  const uint32_t dataset_count = active_dataset_count();

  if (conv_bench_is_print_hart()) {
    printf("\n=== 1D CONV BENCH @ %" PRIu64 " Hz ===\n", frequency_hz);
    printf("CONFIG,N_list=%zu,K_list=%zu,D_list=%zu,datasets=%u,runs_cold=%u,runs_warm=%u\n",
           (size_t)ARRAY_LEN(k_sweep_n),
           (size_t)ARRAY_LEN(k_sweep_k),
           (size_t)ARRAY_LEN(k_sweep_d),
           dataset_count,
           (unsigned)CONV_BENCH_RUNS_COLD,
           (unsigned)CONV_BENCH_RUNS_WARM);
    printf("CONFIG,memmodes,dram_l2=%d,scratchpad=%d\n",
           CONV_BENCH_ENABLE_MEMMODE_DRAM_L2,
           CONV_BENCH_ENABLE_MEMMODE_SCRATCHPAD);
    printf("CONFIG,cachemodes,cold=%d,warm=%d,reuse_off=%d,reuse_on=%d\n",
           CONV_BENCH_ENABLE_CACHE_COLD,
           CONV_BENCH_ENABLE_CACHE_WARM,
           CONV_BENCH_ENABLE_REUSE_OFF,
           CONV_BENCH_ENABLE_REUSE_ON);
    printf("CONFIG,accuracy,abs_tol=%g,rel_tol=%g\n",
           (double)CONV_BENCH_ACCURACY_ABS_TOL,
           (double)CONV_BENCH_ACCURACY_REL_TOL);
  }

  for (size_t mem_i = 0; mem_i < ARRAY_LEN(k_mem_modes); ++mem_i) {
    const mem_mode_t *mem_mode = &k_mem_modes[mem_i];
    if (!mem_mode->enabled) {
      continue;
    }

    for (size_t dsi = 0; dsi < dataset_count; ++dsi) {
      for (size_t ni = 0; ni < ARRAY_LEN(k_sweep_n); ++ni) {
        for (size_t ki = 0; ki < ARRAY_LEN(k_sweep_k); ++ki) {
          for (size_t di = 0; di < ARRAY_LEN(k_sweep_d); ++di) {
            uint32_t n = k_sweep_n[ni];
            uint32_t k = k_sweep_k[ki];
            uint32_t dilation = k_sweep_d[di];

#if CONV_BENCH_ENABLE_CACHE_COLD
#if CONV_BENCH_ENABLE_REUSE_OFF
            run_one_case(frequency_hz, mem_mode, CONV_CACHE_COLD, false, (uint32_t)dsi, n, k, dilation, &summary);
#endif
#if CONV_BENCH_ENABLE_REUSE_ON
            run_one_case(frequency_hz, mem_mode, CONV_CACHE_COLD, true, (uint32_t)dsi, n, k, dilation, &summary);
#endif
#endif

#if CONV_BENCH_ENABLE_CACHE_WARM
#if CONV_BENCH_ENABLE_REUSE_OFF
            run_one_case(frequency_hz, mem_mode, CONV_CACHE_WARM, false, (uint32_t)dsi, n, k, dilation, &summary);
#endif
#if CONV_BENCH_ENABLE_REUSE_ON
            run_one_case(frequency_hz, mem_mode, CONV_CACHE_WARM, true, (uint32_t)dsi, n, k, dilation, &summary);
#endif
#endif
          }
        }
      }
    }
  }

  if (conv_bench_is_print_hart()) {
    printf("SUMMARY,pll_hz=%" PRIu64 ",total_cases=%u,run_cases=%u,skipped_cases=%u,status_fail=%u,total_mismatch=%u\n",
           frequency_hz,
           summary.total_cases,
           summary.run_cases,
           summary.skipped_cases,
           summary.total_status_fails,
           summary.total_mismatches);
    printf("=== 1D CONV BENCH DONE @ %" PRIu64 " Hz ===\n", frequency_hz);
  }
}

void app_init(void) {
  init_test(target_frequency);
  bench_cache_init();
}

void app_main(void) {
  run_suite_for_frequency(target_frequency);
}

#if CONV_BENCH_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = {
  CONV_BENCH_PLL_FREQ_LIST
};
#endif

int main(void) {
#if CONV_BENCH_ENABLE_PLL_SWEEP
  const size_t num_freqs = ARRAY_LEN(k_pll_sweep_freqs_hz);
  if (num_freqs == 0u) {
    return 0;
  }

  target_frequency = k_pll_sweep_freqs_hz[0];
  init_test(target_frequency);
  bench_cache_init();
  run_suite_for_frequency(target_frequency);

  for (size_t i = 1; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, CONV_BENCH_PLL_SWEEP_SLEEP_MS);
    run_suite_for_frequency(target_frequency);
  }
  return 0;
#else
  app_init();
  app_main();
  return 0;
#endif
}
