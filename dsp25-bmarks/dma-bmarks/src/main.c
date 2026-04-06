#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "bench_config.h"
#include "hal_dma.h"
#include "simple_setup.h"

#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0]))

typedef enum {
  DMA_REGION_DRAM = 0,
  DMA_REGION_SCRATCHPAD,
} dma_region_t;

typedef enum {
  DMA_CACHE_COLD = 0,
  DMA_CACHE_WARM_SRC,
  DMA_CACHE_WARM_DST,
  DMA_CACHE_WARM_BOTH,
  DMA_CACHE_HOT_REPEAT,
  DMA_CACHE_STATE_COUNT,
} dma_cache_state_t;

typedef struct {
  const char *name;
  dma_region_t src;
  dma_region_t dst;
  uint32_t bytes;
  bool enabled;
} dma_transfer_case_t;

typedef struct {
  const char *name;
  bool enabled;
  uint32_t requested_channels;
} dma_mode_t;

typedef struct {
  uint64_t best;
  uint64_t sum;
  uint32_t runs;
} metric_stats_t;

typedef struct {
  metric_stats_t total_cycles;
  metric_stats_t setup_cycles;
  metric_stats_t transfer_cycles;
} dma_stats_t;

typedef struct {
  uint64_t total_cycles;
  uint64_t setup_cycles;
  uint64_t transfer_cycles;
  bool success;
} dma_copy_result_t;

static uint8_t g_cache_evict[DMA_BENCH_CACHE_EVICT_BYTES]
    __attribute__((aligned(DMA_BENCH_CACHE_LINE_BYTES)));

static uint64_t target_frequency = DMA_BENCH_TARGET_FREQUENCY_HZ;

static const char *k_cache_state_name[DMA_CACHE_STATE_COUNT] = {
    "COLD", "WARM_SRC", "WARM_DST", "WARM_BOTH", "HOT_REPEAT"};

static const dma_transfer_case_t k_transfer_cases[] = {
    {"DRAM->DRAM", DMA_REGION_DRAM, DMA_REGION_DRAM,
     DMA_BENCH_DRAM_TO_DRAM_BYTES,
     DMA_BENCH_ENABLE_CASE_DRAM_TO_DRAM ? true : false},
    {"DRAM->Scratchpad", DMA_REGION_DRAM, DMA_REGION_SCRATCHPAD,
     DMA_BENCH_DRAM_TO_SCRATCH_BYTES,
     DMA_BENCH_ENABLE_CASE_DRAM_TO_SCRATCH ? true : false},
    {"Scratchpad->DRAM", DMA_REGION_SCRATCHPAD, DMA_REGION_DRAM,
     DMA_BENCH_SCRATCH_TO_DRAM_BYTES,
     DMA_BENCH_ENABLE_CASE_SCRATCH_TO_DRAM ? true : false},
};

static const dma_mode_t k_dma_modes[] = {
    {"single-channel", DMA_BENCH_ENABLE_SINGLE_CHANNEL ? true : false, 1u},
    {"multi-channel", DMA_BENCH_ENABLE_MULTI_CHANNEL ? true : false,
     DMA_BENCH_MULTI_CHANNELS},
};

static inline void dma_full_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline volatile uint8_t *dma_region_base(dma_region_t region,
                                                bool is_dst_side) {
  if (region == DMA_REGION_DRAM) {
    return (volatile uint8_t *)(is_dst_side ? DMA_BENCH_DRAM_DST_BASE
                                            : DMA_BENCH_DRAM_SRC_BASE);
  }
  return (volatile uint8_t *)DMA_BENCH_SCRATCHPAD_BASE;
}

static inline uint32_t dma_pattern_word(uint32_t seed, uint32_t idx) {
  return (seed * 0x9e3779b1u) ^ (uint32_t)(idx * 0x45d9f3bu + seed);
}

static void dma_init_buffer(volatile uint8_t *buf, uint32_t bytes, uint32_t seed) {
  volatile uint32_t *w = (volatile uint32_t *)buf;
  uint32_t words = bytes / 4u;
  uint32_t i;

  for (i = 0; i < words; ++i) {
    w[i] = dma_pattern_word(seed, i);
  }

  for (i = words * 4u; i < bytes; ++i) {
    buf[i] = (uint8_t)(seed ^ i);
  }
}

static void dma_zero_buffer(volatile uint8_t *buf, uint32_t bytes) {
  uint32_t i;
  for (i = 0; i < bytes; ++i) {
    buf[i] = 0u;
  }
}

static bool dma_check_buffer(volatile uint8_t *buf, uint32_t bytes, uint32_t seed) {
  volatile uint32_t *w = (volatile uint32_t *)buf;
  uint32_t words = bytes / 4u;
  uint32_t i;

  for (i = 0; i < words; ++i) {
    if (w[i] != dma_pattern_word(seed, i)) {
      return false;
    }
  }

  for (i = words * 4u; i < bytes; ++i) {
    if (buf[i] != (uint8_t)(seed ^ i)) {
      return false;
    }
  }

  return true;
}

static void dma_stream_touch(volatile uint8_t *buf,
                             uint32_t bytes,
                             uint32_t stride_bytes) {
  volatile uint8_t sink = 0u;
  uint32_t i;

  if (stride_bytes == 0u) {
    stride_bytes = DMA_BENCH_CACHE_LINE_BYTES;
  }

  for (i = 0; i < bytes; i += stride_bytes) {
    sink ^= buf[i];
  }

  asm volatile("" : : "r"(sink) : "memory");
}

static void dma_evict_caches(void) {
  dma_stream_touch((volatile uint8_t *)g_cache_evict,
                   DMA_BENCH_CACHE_EVICT_BYTES,
                   DMA_BENCH_CACHE_LINE_BYTES);
}

static void dma_prepare_cache_state(dma_cache_state_t state,
                                    volatile uint8_t *src,
                                    volatile uint8_t *dst,
                                    uint32_t bytes) {
  if (state == DMA_CACHE_HOT_REPEAT) {
    return;
  }

  dma_evict_caches();

  if (state == DMA_CACHE_WARM_SRC || state == DMA_CACHE_WARM_BOTH) {
    dma_stream_touch(src, bytes, DMA_BENCH_CACHE_LINE_BYTES);
  }
  if (state == DMA_CACHE_WARM_DST || state == DMA_CACHE_WARM_BOTH) {
    dma_stream_touch(dst, bytes, DMA_BENCH_CACHE_LINE_BYTES);
  }

  dma_full_fence();
}

static inline void metric_stats_init(metric_stats_t *stats) {
  stats->best = UINT64_MAX;
  stats->sum = 0u;
  stats->runs = 0u;
}

static inline void metric_stats_record(metric_stats_t *stats, uint64_t value) {
  if (value < stats->best) {
    stats->best = value;
  }
  stats->sum += value;
  stats->runs += 1u;
}

static inline uint64_t metric_stats_avg(const metric_stats_t *stats) {
  if (stats->runs == 0u) {
    return 0u;
  }
  return stats->sum / (uint64_t)stats->runs;
}

static inline void dma_stats_init(dma_stats_t *stats) {
  metric_stats_init(&stats->total_cycles);
  metric_stats_init(&stats->setup_cycles);
  metric_stats_init(&stats->transfer_cycles);
}

static inline void dma_stats_record(dma_stats_t *stats,
                                    const dma_copy_result_t *copy_result) {
  metric_stats_record(&stats->total_cycles, copy_result->total_cycles);
  metric_stats_record(&stats->setup_cycles, copy_result->setup_cycles);
  metric_stats_record(&stats->transfer_cycles, copy_result->transfer_cycles);
}

static inline double cycles_to_ns(uint64_t cycles, uint64_t frequency_hz) {
  if (frequency_hz == 0u) {
    return 0.0;
  }
  return ((double)cycles * 1000000000.0) / (double)frequency_hz;
}

static inline double cycles_to_mbps(uint32_t bytes,
                                    uint64_t cycles,
                                    uint64_t frequency_hz) {
  if (cycles == 0u || frequency_hz == 0u) {
    return 0.0;
  }
  return ((double)bytes * (double)frequency_hz) / ((double)cycles * 1000000.0);
}

static bool is_cache_state_enabled(dma_cache_state_t state) {
  switch (state) {
    case DMA_CACHE_COLD:
      return DMA_BENCH_ENABLE_CACHE_COLD ? true : false;
    case DMA_CACHE_WARM_SRC:
      return DMA_BENCH_ENABLE_CACHE_WARM_SRC ? true : false;
    case DMA_CACHE_WARM_DST:
      return DMA_BENCH_ENABLE_CACHE_WARM_DST ? true : false;
    case DMA_CACHE_WARM_BOTH:
      return DMA_BENCH_ENABLE_CACHE_WARM_BOTH ? true : false;
    case DMA_CACHE_HOT_REPEAT:
      return DMA_BENCH_ENABLE_CACHE_HOT_REPEAT ? true : false;
    default:
      return false;
  }
}

static uint16_t dma_next_tid(void) {
  static uint16_t g_tid = 0x100u;
  uint16_t out = g_tid;

  g_tid = (uint16_t)(g_tid + 1u);
  if (g_tid == 0u) {
    g_tid = 0x100u;
  }

  return out;
}

static uint32_t clamp_channel_count(uint32_t requested, uint32_t total_packets) {
  uint32_t active = requested;

  if (active == 0u) {
    active = 1u;
  }
  if (active > DMA_CORE_CHANNEL_COUNT) {
    active = DMA_CORE_CHANNEL_COUNT;
  }
  if (active > total_packets) {
    active = total_packets;
  }

  return active;
}

static dma_copy_result_t dma_copy_buffer(volatile uint8_t *dst,
                                         volatile uint8_t *src,
                                         uint32_t bytes,
                                         uint32_t requested_channels) {
  dma_copy_result_t result;
  uint32_t packet_bytes = (uint32_t)(1u << DMA_BENCH_LOGW);
  uint32_t total_packets;
  uint32_t channels;
  uint32_t packet_cursor = 0u;
  uint32_t ch;
  uint16_t tids[DMA_CORE_CHANNEL_COUNT];
  uint64_t t_setup_start;
  uint64_t t_setup_end;
  uint64_t t_done;

  result.total_cycles = 0u;
  result.setup_cycles = 0u;
  result.transfer_cycles = 0u;
  result.success = false;

  if (packet_bytes == 0u || bytes == 0u) {
    return result;
  }

  if ((bytes % packet_bytes) != 0u) {
    return result;
  }

  total_packets = bytes / packet_bytes;
  if (total_packets == 0u) {
    return result;
  }

  channels = clamp_channel_count(requested_channels, total_packets);
  if (channels == 0u) {
    return result;
  }

  t_setup_start = rdcycle();

  for (ch = 0u; ch < channels; ++ch) {
    dma_transaction_t tx;
    uint32_t packets_this = total_packets / channels;
    uint32_t rem = total_packets % channels;
    uintptr_t addr_offset;

    if (ch < rem) {
      packets_this += 1u;
    }

    if (packets_this == 0u || packets_this > UINT16_MAX) {
      dma_reset();
      return result;
    }

    addr_offset = (uintptr_t)(packet_cursor * packet_bytes);

    tx.core = DMA_BENCH_CORE_ID;
    tx.transaction_id = dma_next_tid();
    tx.transaction_priority = DMA_BENCH_PRIORITY;
    tx.peripheral_id = 0u;
    tx.addr_r = (uintptr_t)(src + addr_offset);
    tx.addr_w = (uintptr_t)(dst + addr_offset);
    tx.inc_r = (uint16_t)packet_bytes;
    tx.inc_w = (uint16_t)packet_bytes;
    tx.len = (uint16_t)packets_this;
    tx.logw = (uint8_t)DMA_BENCH_LOGW;
    tx.do_interrupt = false;
    tx.do_address_gate = false;

    if (!set_DMA_C(ch, tx, true)) {
      dma_reset();
      return result;
    }

    tids[ch] = tx.transaction_id;
    packet_cursor += packets_this;
  }

  for (ch = 0u; ch < channels; ++ch) {
    start_DMA(ch, tids[ch], NULL);
  }

  t_setup_end = rdcycle();

  dma_wait_till_inactive(DMA_BENCH_IDLE_SPIN_CYCLES);
  t_done = rdcycle();

  dma_reset();

  result.setup_cycles = t_setup_end - t_setup_start;
  result.transfer_cycles = t_done - t_setup_end;
  result.total_cycles = t_done - t_setup_start;
  result.success = true;
  return result;
}

static bool dma_run_state(const dma_transfer_case_t *tc,
                          dma_cache_state_t state,
                          const dma_mode_t *mode,
                          uint32_t seed_base,
                          dma_stats_t *stats_out) {
  volatile uint8_t *src = dma_region_base(tc->src, false);
  volatile uint8_t *dst = dma_region_base(tc->dst, true);
  bool ok = true;
  uint32_t channels = mode->requested_channels;

  dma_stats_init(stats_out);

  if (state == DMA_CACHE_HOT_REPEAT) {
    uint32_t run;
    uint32_t seed = seed_base + 0x5a5au;

    dma_init_buffer(src, tc->bytes, seed);
    dma_zero_buffer(dst, tc->bytes);
    dma_prepare_cache_state(DMA_CACHE_WARM_BOTH, src, dst, tc->bytes);

    {
      dma_copy_result_t warmup =
          dma_copy_buffer(dst, src, tc->bytes, channels);
      ok &= warmup.success;
      ok &= dma_check_buffer(dst, tc->bytes, seed);
    }

    for (run = 0u; run < DMA_BENCH_HOT_REPEAT_RUNS; ++run) {
      dma_copy_result_t copy_result =
          dma_copy_buffer(dst, src, tc->bytes, channels);
      dma_stats_record(stats_out, &copy_result);
      ok &= copy_result.success;
      if (copy_result.success) {
        ok &= dma_check_buffer(dst, tc->bytes, seed);
      }
    }

    return ok;
  }

  {
    uint32_t run;
    for (run = 0u; run < DMA_BENCH_NUM_RUNS; ++run) {
      uint32_t seed = seed_base + (uint32_t)(state * 131u + run);
      dma_copy_result_t copy_result;

      dma_init_buffer(src, tc->bytes, seed);
      dma_zero_buffer(dst, tc->bytes);
      dma_prepare_cache_state(state, src, dst, tc->bytes);

      copy_result = dma_copy_buffer(dst, src, tc->bytes, channels);
      dma_stats_record(stats_out, &copy_result);

      ok &= copy_result.success;
      if (copy_result.success) {
        ok &= dma_check_buffer(dst, tc->bytes, seed);
      }
    }
  }

  return ok;
}

static void print_state_row(const dma_transfer_case_t *tc,
                            dma_cache_state_t state,
                            const dma_stats_t *stats,
                            uint64_t frequency_hz,
                            bool ok) {
  uint64_t best_total = stats->total_cycles.best;
  uint64_t avg_total = metric_stats_avg(&stats->total_cycles);
  uint64_t avg_setup = metric_stats_avg(&stats->setup_cycles);
  uint64_t avg_transfer = metric_stats_avg(&stats->transfer_cycles);
  double best_ns;
  double avg_ns;
  double best_mbps;
  double avg_mbps;

  if (stats->total_cycles.runs == 0u || best_total == UINT64_MAX) {
    best_total = 0u;
  }

  best_ns = cycles_to_ns(best_total, frequency_hz);
  avg_ns = cycles_to_ns(avg_total, frequency_hz);
  best_mbps = cycles_to_mbps(tc->bytes, best_total, frequency_hz);
  avg_mbps = cycles_to_mbps(tc->bytes, avg_total, frequency_hz);

  printf("  %-10s best=%10llu cyc (%11.2f ns, %9.2f MB/s)  "
         "avg=%10llu cyc (%11.2f ns, %9.2f MB/s)  "
         "setup_avg=%8llu  xfer_avg=%8llu  %s\n",
         k_cache_state_name[state],
         (unsigned long long)best_total,
         best_ns,
         best_mbps,
         (unsigned long long)avg_total,
         avg_ns,
         avg_mbps,
         (unsigned long long)avg_setup,
         (unsigned long long)avg_transfer,
         ok ? "PASS" : "FAIL");
}

static bool run_suite_for_frequency(uint64_t frequency_hz) {
  bool all_ok = true;
  uint32_t mode_i;

  printf("\n=== DSP25 DMA Benchmark Suite @ %llu Hz ===\n",
         (unsigned long long)frequency_hz);
  printf("runs=%u, hot_runs=%u, logw=%u, packet=%u B\n",
         (unsigned)DMA_BENCH_NUM_RUNS,
         (unsigned)DMA_BENCH_HOT_REPEAT_RUNS,
         (unsigned)DMA_BENCH_LOGW,
         (unsigned)(1u << DMA_BENCH_LOGW));

  for (mode_i = 0u; mode_i < (uint32_t)ARRAY_LEN(k_dma_modes); ++mode_i) {
    const dma_mode_t *mode = &k_dma_modes[mode_i];
    uint32_t tc_i;
    uint32_t mode_channels_cap;

    if (!mode->enabled) {
      continue;
    }

    mode_channels_cap = mode->requested_channels;
    if (mode_channels_cap == 0u) {
      mode_channels_cap = 1u;
    }
    if (mode_channels_cap > DMA_CORE_CHANNEL_COUNT) {
      mode_channels_cap = DMA_CORE_CHANNEL_COUNT;
    }
    if (mode_channels_cap == 0u) {
      mode_channels_cap = 1u;
    }

    printf("\n--- Mode: %s (requested=%u, capped=%u, hw_max=%u) ---\n",
           mode->name,
           (unsigned)mode->requested_channels,
           (unsigned)mode_channels_cap,
           (unsigned)DMA_CORE_CHANNEL_COUNT);

    for (tc_i = 0u; tc_i < (uint32_t)ARRAY_LEN(k_transfer_cases); ++tc_i) {
      const dma_transfer_case_t *tc = &k_transfer_cases[tc_i];
      dma_cache_state_t state;

      if (!tc->enabled) {
        continue;
      }

      printf("\n[%s] bytes=%u\n", tc->name, (unsigned)tc->bytes);

      for (state = DMA_CACHE_COLD; state < DMA_CACHE_STATE_COUNT;
           state = (dma_cache_state_t)(state + 1)) {
        dma_stats_t stats;
        bool ok;

        if (!is_cache_state_enabled(state)) {
          continue;
        }

        ok = dma_run_state(tc,
                           state,
                           mode,
                           DMA_BENCH_BASE_SEED +
                               (mode_i * 0x10000u) +
                               (tc_i * 0x1000u),
                           &stats);

        print_state_row(tc, state, &stats, frequency_hz, ok);
        all_ok &= ok;
      }
    }
  }

  printf("\n=== DMA suite complete @ %llu Hz: %s ===\n",
         (unsigned long long)frequency_hz,
         all_ok ? "PASS" : "FAIL");

  return all_ok;
}

#if DMA_BENCH_ENABLE_PLL_SWEEP
static const uint64_t k_pll_sweep_freqs_hz[] = {
    DMA_BENCH_PLL_FREQ_LIST,
};
#endif

int main(void) {
#if DMA_BENCH_ENABLE_PLL_SWEEP
  size_t i;
  size_t num_freqs = ARRAY_LEN(k_pll_sweep_freqs_hz);

  if (!dma_bench_is_print_hart()) {
    while (1) {
      asm volatile("wfi");
    }
  }

  if (num_freqs == 0u) {
    return 0;
  }

  target_frequency = k_pll_sweep_freqs_hz[0];
  init_test(target_frequency);
  (void)run_suite_for_frequency(target_frequency);

  for (i = 1u; i < num_freqs; ++i) {
    target_frequency = k_pll_sweep_freqs_hz[i];
    reconfigure_pll(target_frequency, DMA_BENCH_PLL_SWEEP_SLEEP_MS);
    (void)run_suite_for_frequency(target_frequency);
  }

  return 0;
#else
  if (!dma_bench_is_print_hart()) {
    while (1) {
      asm volatile("wfi");
    }
  }

  target_frequency = DMA_BENCH_TARGET_FREQUENCY_HZ;
  init_test(target_frequency);
  (void)run_suite_for_frequency(target_frequency);
  return 0;
#endif
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    asm volatile("wfi");
  }
}
