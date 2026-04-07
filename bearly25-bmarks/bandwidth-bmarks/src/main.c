/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Memory-bandwidth benchmark with transfer/cache-state sweeps
  ******************************************************************************
  */
/* USER CODE END Header */

#include "main.h"
#include "mtwister.h"
#include "simple_setup.h"
#include <limits.h>
#include <riscv_vector.h>

#ifndef BW_USE_THREADLIB
#define BW_USE_THREADLIB 0
#endif

#if BW_USE_THREADLIB
#include "hthread.h"
#endif

typedef struct {
  uint64_t summary_cycles;
  bool correct;
} bw_suite_result_t;

typedef enum {
  BW_REGION_DRAM = 0,
  BW_REGION_SCRATCHPAD,
  BW_REGION_LOCAL_TCM,
  BW_REGION_REMOTE_TCM
} bw_region_t;

typedef enum {
  BW_CACHE_COLD = 0,
  BW_CACHE_WARM_SRC,
  BW_CACHE_WARM_DST,
  BW_CACHE_WARM_BOTH,
  BW_CACHE_HOT_REPEAT,
  BW_CACHE_STATE_COUNT
} bw_cache_state_t;

typedef struct {
  const char *name;
  bw_region_t src;
  bw_region_t dst;
  uint32_t bytes;
} bw_transfer_case_t;

typedef struct {
  uint64_t best_cycles;
  uint64_t sum_cycles;
  uint32_t runs;
} bw_stats_t;

typedef void (*bw_copy_fn_t)(volatile uint8_t *dst, volatile uint8_t *src, uint32_t bytes);

typedef struct {
  volatile uint8_t *src;
  volatile uint8_t *dst;
  uint32_t bytes;
  bool vector;
} bw_worker_arg_t;

static volatile uint8_t *g_dram_src = (volatile uint8_t *)BW_DRAM_SRC_BASE;
static volatile uint8_t *g_dram_dst = (volatile uint8_t *)BW_DRAM_DST_BASE;
static uint8_t g_cache_evict[BW_CACHE_EVICT_BYTES] __attribute__((aligned(BW_CACHE_LINE_BYTES)));
uint64_t target_frequency = BW_TARGET_FREQUENCY_HZ;

static const char *k_cache_state_name[BW_CACHE_STATE_COUNT] = {
  "COLD",
  "WARM_SRC",
  "WARM_DST",
  "WARM_BOTH",
  "HOT_REPEAT"
};

static const bw_transfer_case_t k_transfer_cases[] = {
  { "DRAM->DRAM",        BW_REGION_DRAM,       BW_REGION_DRAM,       BW_DRAM_BYTES    },
  // { "DRAM->Scratchpad",  BW_REGION_DRAM,       BW_REGION_SCRATCHPAD, BW_SCRATCH_BYTES },
  // { "Scratchpad->DRAM",  BW_REGION_SCRATCHPAD, BW_REGION_DRAM,       BW_SCRATCH_BYTES },
  // { "DRAM->LocalTCM",    BW_REGION_DRAM,       BW_REGION_LOCAL_TCM,  BW_TCM_BYTES     },
  // { "LocalTCM->DRAM",    BW_REGION_LOCAL_TCM,  BW_REGION_DRAM,       BW_TCM_BYTES     },
  // { "DRAM->RemoteTCM",   BW_REGION_DRAM,       BW_REGION_REMOTE_TCM, BW_TCM_BYTES     },
  // { "RemoteTCM->DRAM",   BW_REGION_REMOTE_TCM, BW_REGION_DRAM,       BW_TCM_BYTES     },
};

static inline uint64_t rdcycle64(void) {
  return get_cycles();
}

static inline void bw_fence_all(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline volatile uint8_t *bw_region_base(bw_region_t region, bool is_dst_side) {
  switch (region) {
    case BW_REGION_DRAM:
      return is_dst_side ? g_dram_dst : g_dram_src;
    case BW_REGION_SCRATCHPAD:
      return (volatile uint8_t *)BW_SCRATCHPAD_BASE;
    case BW_REGION_LOCAL_TCM:
      return (volatile uint8_t *)BW_CORE0_TCM_BASE;
    case BW_REGION_REMOTE_TCM:
      return (volatile uint8_t *)BW_CORE1_TCM_BASE;
    default:
      return is_dst_side ? g_dram_dst : g_dram_src;
  }
}

static void init_buffer(volatile uint8_t *buf, uint32_t size, uint32_t seed) {
  MTRand r = seedRand(seed);
  volatile uint32_t *w = (volatile uint32_t *)buf;
  uint32_t words = size / (uint32_t)sizeof(uint32_t);
  for (uint32_t i = 0; i < words; i++) {
    w[i] = genRandLong(&r);
  }
}

static bool check_buffer(volatile uint8_t *buf, uint32_t size, uint32_t seed) {
  MTRand r = seedRand(seed);
  volatile uint32_t *w = (volatile uint32_t *)buf;
  uint32_t words = size / (uint32_t)sizeof(uint32_t);
  for (uint32_t i = 0; i < words; i++) {
    if (w[i] != genRandLong(&r)) {
      return false;
    }
  }
  return true;
}

static void bw_stream_touch(volatile uint8_t *buf, uint32_t size, uint32_t stride_bytes) {
  volatile uint8_t sink = 0;
  if (stride_bytes == 0) {
    stride_bytes = BW_CACHE_LINE_BYTES;
  }
  for (uint32_t i = 0; i < size; i += stride_bytes) {
    sink ^= buf[i];
  }
  asm volatile("" :: "r"(sink) : "memory");
}

static void bw_evict_caches(void) {
  bw_stream_touch((volatile uint8_t *)g_cache_evict, BW_CACHE_EVICT_BYTES, BW_CACHE_LINE_BYTES);
}

static void bw_prepare_cache_state(bw_cache_state_t state,
                                   volatile uint8_t *src,
                                   volatile uint8_t *dst,
                                   uint32_t bytes) {
  if (state == BW_CACHE_HOT_REPEAT) {
    return;
  }

  bw_evict_caches();

  if (state == BW_CACHE_WARM_SRC || state == BW_CACHE_WARM_BOTH) {
    bw_stream_touch(src, bytes, BW_CACHE_LINE_BYTES);
  }
  if (state == BW_CACHE_WARM_DST || state == BW_CACHE_WARM_BOTH) {
    bw_stream_touch(dst, bytes, BW_CACHE_LINE_BYTES);
  }

  bw_fence_all();
}

static inline void bw_stats_init(bw_stats_t *s) {
  s->best_cycles = UINT64_MAX;
  s->sum_cycles = 0;
  s->runs = 0;
}

static inline void bw_stats_record(bw_stats_t *s, uint64_t cycles) {
  if (cycles < s->best_cycles) {
    s->best_cycles = cycles;
  }
  s->sum_cycles += cycles;
  s->runs++;
}

static inline uint64_t bw_stats_avg(const bw_stats_t *s) {
  if (s->runs == 0) {
    return 0;
  }
  return s->sum_cycles / (uint64_t)s->runs;
}

static inline double bw_cycles_to_ns(uint64_t cycles, uint64_t frequency_hz) {
  if (frequency_hz == 0u) {
    return 0.0;
  }
  return ((double)cycles * 1000000000.0) / (double)frequency_hz;
}

static inline double bw_cycles_to_mbps(uint32_t bytes, uint64_t cycles, uint64_t frequency_hz) {
  if (frequency_hz == 0u || cycles == 0u) {
    return 0.0;
  }
  // Report decimal MB/s (1 MB = 1,000,000 bytes)
  return ((double)bytes * (double)frequency_hz) / ((double)cycles * 1000000.0);
}

static void bw_copy_cpu(volatile uint8_t *dst, volatile uint8_t *src, uint32_t bytes) {
  volatile uint64_t *src64 = (volatile uint64_t *)src;
  volatile uint64_t *dst64 = (volatile uint64_t *)dst;
  uint32_t words64 = bytes / 8u;

  for (uint32_t i = 0; i < words64; i++) {
    dst64[i] = src64[i];
  }

  for (uint32_t i = words64 * 8u; i < bytes; i++) {
    dst[i] = src[i];
  }
}

static void bw_copy_glibc(volatile uint8_t *dst, volatile uint8_t *src, uint32_t bytes) {
  memcpy((void *)dst, (const void *)src, bytes);
}

static void bw_copy_rvv(volatile uint8_t *dst, volatile uint8_t *src, uint32_t bytes) {
  const uint8_t *src_ptr = (const uint8_t *)src;
  uint8_t *dst_ptr = (uint8_t *)dst;
  size_t remaining = bytes;

  while (remaining > 0) {
    size_t vl = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t vec_src = __riscv_vle8_v_u8m8(src_ptr, vl);
    __riscv_vse8_v_u8m8(dst_ptr, vec_src, vl);
    src_ptr += vl;
    dst_ptr += vl;
    remaining -= vl;
  }
}

#if BW_USE_THREADLIB
static void bw_worker_copy(void *arg_) {
  bw_worker_arg_t *arg = (bw_worker_arg_t *)arg_;
  if (arg->vector) {
    bw_copy_rvv(arg->dst, arg->src, arg->bytes);
  } else {
    bw_copy_cpu(arg->dst, arg->src, arg->bytes);
  }
}

static void bw_worker_nop(void *arg) {
  (void)arg;
}

static void bw_threadlib_init(void) {
  hthread_init();

  /* Warm thread-lib once so first measured run avoids cold startup effects. */
  for (uint32_t i = 0; i < N_HARTS; ++i) {
    hthread_dispatch(bw_worker_nop, NULL);
  }
  for (uint32_t i = 1; i < N_HARTS; ++i) {
    hthread_join(i);
  }
}
#else
static void bw_threadlib_init(void) {}
#endif

static void bw_copy_mp_common(volatile uint8_t *dst, volatile uint8_t *src, uint32_t bytes, bool vector) {
#if BW_USE_THREADLIB
  const uint32_t workers = N_HARTS;
  bw_worker_arg_t args[N_HARTS];
  uint32_t base_chunk;
  uint32_t rem;
  uint32_t offset = 0;

  if (workers < 2u || bytes == 0u) {
    if (vector) {
      bw_copy_rvv(dst, src, bytes);
    } else {
      bw_copy_cpu(dst, src, bytes);
    }
    return;
  }

  base_chunk = bytes / workers;
  rem = bytes % workers;

  for (uint32_t i = 0; i < workers; i++) {
    uint32_t chunk = base_chunk + ((i == workers - 1u) ? rem : 0u);
    args[i].src = src + offset;
    args[i].dst = dst + offset;
    args[i].bytes = chunk;
    args[i].vector = vector;
    offset += chunk;
  }

  for (uint32_t i = 0; i < workers; ++i) {
    hthread_dispatch(bw_worker_copy, args + i);
  }

  for (uint32_t i = 1; i < workers; ++i) {
    hthread_join(i);
  }
#else
  if (vector) {
    bw_copy_rvv(dst, src, bytes);
  } else {
    bw_copy_cpu(dst, src, bytes);
  }
#endif
}

static void bw_copy_cpu_mp(volatile uint8_t *dst, volatile uint8_t *src, uint32_t bytes) {
  bw_copy_mp_common(dst, src, bytes, false);
}

static void bw_copy_rvv_mp(volatile uint8_t *dst, volatile uint8_t *src, uint32_t bytes) {
  bw_copy_mp_common(dst, src, bytes, true);
}

static bool bw_run_state(const bw_transfer_case_t *tc,
                         bw_cache_state_t state,
                         bw_copy_fn_t copy_fn,
                         uint32_t seed_base,
                         bw_stats_t *stats_out) {
  volatile uint8_t *src = bw_region_base(tc->src, false);
  volatile uint8_t *dst = bw_region_base(tc->dst, true);
  bool ok = true;

  bw_stats_init(stats_out);

  if (state == BW_CACHE_HOT_REPEAT) {
    uint32_t seed = seed_base + 0x5a5au;
    init_buffer(src, tc->bytes, seed);
    bw_prepare_cache_state(BW_CACHE_WARM_BOTH, src, dst, tc->bytes);

    // Warm-up copy not measured, then measure back-to-back repeats.
    copy_fn(dst, src, tc->bytes);
    bw_fence_all();

    for (uint32_t run = 0; run < BW_NUM_RUNS; run++) {
      uint64_t t0 = rdcycle64();
      copy_fn(dst, src, tc->bytes);
      bw_fence_all();
      uint64_t cycles = rdcycle64() - t0;
      bw_stats_record(stats_out, cycles);
    }

    ok = check_buffer(dst, tc->bytes, seed);
    return ok;
  }

  for (uint32_t run = 0; run < BW_NUM_RUNS; run++) {
    uint32_t seed = seed_base + (uint32_t)(state * 131u + run);

    init_buffer(src, tc->bytes, seed);
    bw_prepare_cache_state(state, src, dst, tc->bytes);

    uint64_t t0 = rdcycle64();
    copy_fn(dst, src, tc->bytes);
    bw_fence_all();
    uint64_t cycles = rdcycle64() - t0;

    bw_stats_record(stats_out, cycles);
    ok &= check_buffer(dst, tc->bytes, seed);
  }

  return ok;
}

static bw_suite_result_t run_bandwidth_suite(const char *kernel_name,
                                             bw_copy_fn_t copy_fn,
                                             uint32_t seed_base,
                                             uint64_t frequency_hz) {
  bw_suite_result_t result;
  result.summary_cycles = 0;
  result.correct = true;

  printf("\n=== Memory Bandwidth Sweep: %s ===\n", kernel_name);
  printf("runs/state=%u, DRAM=%u B, Scratchpad=%u B, TCM=%u B\n",
         BW_NUM_RUNS, BW_DRAM_BYTES, BW_SCRATCH_BYTES, BW_TCM_BYTES);

  for (uint32_t tc_i = 0; tc_i < (uint32_t)(sizeof(k_transfer_cases) / sizeof(k_transfer_cases[0])); tc_i++) {
    const bw_transfer_case_t *tc = &k_transfer_cases[tc_i];
    printf("\n[%s] bytes=%u\n", tc->name, tc->bytes);

    for (uint32_t st = 0; st < BW_CACHE_STATE_COUNT; st++) {
      bw_stats_t stats;
      bool ok = bw_run_state(tc, (bw_cache_state_t)st, copy_fn,
                             seed_base + tc_i * 0x1000u, &stats);
      uint64_t avg = bw_stats_avg(&stats);
      double best_ns = bw_cycles_to_ns(stats.best_cycles, frequency_hz);
      double avg_ns = bw_cycles_to_ns(avg, frequency_hz);
      double best_mbps = bw_cycles_to_mbps(tc->bytes, stats.best_cycles, frequency_hz);
      double avg_mbps = bw_cycles_to_mbps(tc->bytes, avg, frequency_hz);

      printf("  %-10s best=%10llu cyc (%12.2f ns, %10.2f MB/s)  "
             "avg=%10llu cyc (%12.2f ns, %10.2f MB/s)  %s\n",
             k_cache_state_name[st],
             (unsigned long long)stats.best_cycles,
             best_ns,
             best_mbps,
             (unsigned long long)avg,
             avg_ns,
             avg_mbps,
             ok ? "PASS" : "FAIL");

      if (tc_i == 0 && st == BW_CACHE_COLD) {
        // Summary cycles for default DRAM->DRAM COLD condition.
        result.summary_cycles = avg;
      }
      result.correct &= ok;
    }
  }

  {
    const uint32_t summary_bytes = k_transfer_cases[0].bytes;
    const double summary_ns = bw_cycles_to_ns(result.summary_cycles, frequency_hz);
    const double summary_mbps = bw_cycles_to_mbps(summary_bytes, result.summary_cycles, frequency_hz);
    printf("\nSummary: cycles=%llu, time=%0.2f ns, bw=%0.2f MB/s "
           "(DRAM->DRAM COLD avg), pass=%d\n",
           (unsigned long long)result.summary_cycles,
           summary_ns,
           summary_mbps,
           result.correct ? 1 : 0);
  }
  return result;
}

static bool cpu_memcpy_suite(uint32_t seed, uint64_t frequency_hz) {
  bw_suite_result_t result = run_bandwidth_suite("CPU memcpy", bw_copy_cpu, seed, frequency_hz);
  return result.correct;
}

static bool glibc_memcpy_suite(uint32_t seed, uint64_t frequency_hz) {
  bw_suite_result_t result = run_bandwidth_suite("glibc memcpy", bw_copy_glibc, seed, frequency_hz);
  return result.correct;
}

static bool rvv_memcpy_suite(uint32_t seed, uint64_t frequency_hz) {
  bw_suite_result_t result = run_bandwidth_suite("RVV memcpy", bw_copy_rvv, seed, frequency_hz);
  return result.correct;
}

static bool cpu_memcpy_mp_suite(uint32_t seed, uint64_t frequency_hz) {
  bw_suite_result_t result = run_bandwidth_suite("CPU memcpy (multi-hart)", bw_copy_cpu_mp, seed, frequency_hz);
  return result.correct;
}

static bool rvv_memcpy_mp_suite(uint32_t seed, uint64_t frequency_hz) {
  bw_suite_result_t result = run_bandwidth_suite("RVV memcpy (multi-hart)", bw_copy_rvv_mp, seed, frequency_hz);
  return result.correct;
}

static void run_suite_for_frequency(uint64_t frequency_hz) {
  bool all_ok = true;
  uint32_t seed = BW_BASE_SEED;

  printf("\n=== Bearly25 Bandwidth Benchmark Sweep @ %llu Hz ===\n",
         (unsigned long long)frequency_hz);
  printf("runs/state=%u\n", BW_NUM_RUNS);

#if BW_ENABLE_CPU
  all_ok &= cpu_memcpy_suite(seed + 0x1000u, frequency_hz);
#endif

#if BW_ENABLE_GLIBC
  all_ok &= glibc_memcpy_suite(seed + 0x2000u, frequency_hz);
#endif

#if BW_ENABLE_RVV
  all_ok &= rvv_memcpy_suite(seed + 0x3000u, frequency_hz);
#endif

#if BW_ENABLE_CPU_MP
#if BW_USE_THREADLIB
  all_ok &= cpu_memcpy_mp_suite(seed + 0x4000u, frequency_hz);
#else
  printf("[bandwidth] BW_ENABLE_CPU_MP=1 but thread-lib is unavailable; skipping CPU memcpy (multi-hart)\n");
#endif
#endif

#if BW_ENABLE_RVV_MP
#if BW_USE_THREADLIB
  all_ok &= rvv_memcpy_mp_suite(seed + 0x5000u, frequency_hz);
#else
  printf("[bandwidth] BW_ENABLE_RVV_MP=1 but thread-lib is unavailable; skipping RVV memcpy (multi-hart)\n");
#endif
#endif

  printf("\n=== Bandwidth sweep complete @ %llu Hz: %s ===\n",
         (unsigned long long)frequency_hz, all_ok ? "PASS" : "FAIL");
}

void app_init(void) {
  init_test(target_frequency);
  bw_threadlib_init();
}

void app_main(void) {
  run_suite_for_frequency(target_frequency);
}

#if BW_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = {
  BW_PLL_FREQ_LIST
};
#endif

int main(void) {
#if BW_ENABLE_PLL_SWEEP
  const size_t num_freqs =
      sizeof(k_pll_sweep_freqs_hz) / sizeof(k_pll_sweep_freqs_hz[0]);
  if (num_freqs == 0u) {
    return 0;
  }

  target_frequency = k_pll_sweep_freqs_hz[0];
  init_test(target_frequency);
  bw_threadlib_init();
  run_suite_for_frequency(target_frequency);

  for (size_t i = 1; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, BW_PLL_SWEEP_SLEEP_MS);
    run_suite_for_frequency(target_frequency);
  }
  return 0;
#else
  app_init();
  app_main();
  return 0;
#endif
}

/*
 * Main function for secondary harts.
 *
 * Multi-threaded programs may override this.
 */
void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile ("wfi");
  }
}
