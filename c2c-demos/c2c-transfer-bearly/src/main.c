#include "main.h"

_Static_assert((C2C_TRANSFER_BEARLY_CACHE_LINE_BYTES & (C2C_TRANSFER_BEARLY_CACHE_LINE_BYTES - 1u)) == 0u,
               "C2C_TRANSFER_BEARLY_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((C2C_TRANSFER_BEARLY_CACHE_EVICT_BYTES >= C2C_TRANSFER_BEARLY_CACHE_LINE_BYTES),
               "C2C_TRANSFER_BEARLY_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((C2C_TRANSFER_BEARLY_CACHE_EVICT_BYTES % C2C_TRANSFER_BEARLY_CACHE_LINE_BYTES) == 0u,
               "C2C_TRANSFER_BEARLY_CACHE_EVICT_BYTES must be a multiple of cache line size.");

typedef struct {
  uint64_t tx_cycle;
  uint64_t rx_cycle;
  uint64_t delta;
  uint32_t poll_loops;
  uint8_t immediate_hit;
} rx_sample_t;

static transfer_cycle_word_t *const g_cycle_word =
    (transfer_cycle_word_t *)(uintptr_t)C2C_TRANSFER_BEARLY_MAILBOX_ADDR;

static uint8_t g_cache_evict[C2C_TRANSFER_BEARLY_CACHE_EVICT_BYTES]
    __attribute__((aligned(0x8000)));
static volatile uint8_t g_cache_sink;

uint64_t target_frequency = C2C_TRANSFER_BEARLY_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t pass = 0; pass < C2C_TRANSFER_BEARLY_CACHE_EVICT_PASSES; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)C2C_TRANSFER_BEARLY_CACHE_EVICT_BYTES; i += C2C_TRANSFER_BEARLY_CACHE_LINE_BYTES) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    transfer_fence_rw();
  }

  g_cache_sink = sink;
  transfer_fence_rw();
}

static inline void refresh_shared(void) {
  cache_evict_all();
  transfer_fence_rw();
}

static void clear_shared_if_enabled(void) {
#if C2C_TRANSFER_BEARLY_CLEAR_SHM_ON_BOOT
  refresh_shared();
  if (*g_cycle_word != 0u) {
    C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] skip clear: cycle word already non-zero (%llu)\n",
                            (unsigned long long)(*g_cycle_word));
    return;
  }

  *g_cycle_word = 0u;
  transfer_fence_rw();
  cache_evict_all();
  transfer_fence_rw();
  C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] cleared cycle word at 0x%08lx\n",
                          (unsigned long)C2C_TRANSFER_BEARLY_MAILBOX_ADDR);
#endif
}

static void wait_until_cycle(uint64_t target_cycle) {
  while (rdcycle64() < target_cycle) {
    __asm__ volatile("nop");
  }
}

static rx_sample_t poll_for_new_cycle(uint64_t prev_value, uint8_t require_strictly_increasing) {
  rx_sample_t out;
  uint32_t loops = 0u;

  while (1) {
    uint64_t v;

    refresh_shared();
    v = *g_cycle_word;

    if ((v != prev_value) && (v != 0u) &&
        ((!require_strictly_increasing) || (v > prev_value))) {
      out.tx_cycle = v;
      out.rx_cycle = rdcycle64();
      out.delta = out.rx_cycle - out.tx_cycle;
      out.poll_loops = loops;
      out.immediate_hit = (loops == 0u) ? 1u : 0u;
      return out;
    }

    loops++;
  }
}

void app_init(void) {
  init_test(target_frequency);
  g_cache_sink = 0u;

  clear_shared_if_enabled();

  C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] init cycle_addr=0x%08lx packets=%u interval=%llu\n",
                          (unsigned long)C2C_TRANSFER_BEARLY_MAILBOX_ADDR,
                          (unsigned)C2C_TRANSFER_BEARLY_NUM_PACKETS,
                          (unsigned long long)C2C_TRANSFER_BEARLY_INTERVAL_CYCLES);
}

void app_main(void) {
  uint64_t interval = C2C_TRANSFER_BEARLY_INTERVAL_CYCLES;
  uint32_t total_packets = C2C_TRANSFER_BEARLY_NUM_PACKETS;
  uint64_t slack;
  uint64_t step;
  uint64_t prev_tx;
  uint64_t prev_rx;
  uint64_t delta_min;
  uint64_t delta_max;
  uint64_t delta_sum;
  uint64_t est_link_delay;
  uint64_t baseline;
  int prev_dir = 0;
  rx_sample_t first_sample;

  slack = C2C_TRANSFER_BEARLY_SEARCH_INITIAL_SLACK_CYCLES;
  if (slack > interval) {
    slack = interval;
  }

  step = C2C_TRANSFER_BEARLY_SEARCH_INITIAL_STEP_CYCLES;
  if (step > interval) {
    step = interval;
  }
  if (step < C2C_TRANSFER_BEARLY_SEARCH_MIN_STEP_CYCLES) {
    step = C2C_TRANSFER_BEARLY_SEARCH_MIN_STEP_CYCLES;
  }

  C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] search start packets=%u interval=%llu initial_slack=%llu initial_step=%llu\n",
                          (unsigned)total_packets,
                          (unsigned long long)interval,
                          (unsigned long long)slack,
                          (unsigned long long)step);

  refresh_shared();
  baseline = *g_cycle_word;
  C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] baseline cycle word=%llu; waiting for first non-zero change\n",
                          (unsigned long long)baseline);

  first_sample = poll_for_new_cycle(baseline, 0u);
  prev_tx = first_sample.tx_cycle;
  prev_rx = first_sample.rx_cycle;
  delta_min = first_sample.delta;
  delta_max = first_sample.delta;
  delta_sum = first_sample.delta;
  est_link_delay = delta_min;

  C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] recv seq=1 tx_cycle=%llu rx_cycle=%llu delta=%llu loops=%u immediate=%u est_link_delay=%llu\n",
                          (unsigned long long)first_sample.tx_cycle,
                          (unsigned long long)first_sample.rx_cycle,
                          (unsigned long long)first_sample.delta,
                          (unsigned)first_sample.poll_loops,
                          (unsigned)first_sample.immediate_hit,
                          (unsigned long long)est_link_delay);

  for (uint32_t seq = 2u; seq <= total_packets; ++seq) {
    uint64_t predicted_rx = prev_rx + interval;
    uint64_t poll_start = (predicted_rx > slack) ? (predicted_rx - slack) : 0u;
    rx_sample_t sample;

    C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] seq=%u predict_rx=%llu poll_start=%llu slack=%llu step=%llu\n",
                            (unsigned)seq,
                            (unsigned long long)predicted_rx,
                            (unsigned long long)poll_start,
                            (unsigned long long)slack,
                            (unsigned long long)step);

    wait_until_cycle(poll_start);
    sample = poll_for_new_cycle(prev_tx, 1u);

    if (sample.delta < delta_min) {
      delta_min = sample.delta;
    }
    if (sample.delta > delta_max) {
      delta_max = sample.delta;
    }
    delta_sum += sample.delta;
    est_link_delay = delta_min;

    C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] recv seq=%u tx_cycle=%llu rx_cycle=%llu delta=%llu loops=%u immediate=%u est_link_delay=%llu\n",
                            (unsigned)seq,
                            (unsigned long long)sample.tx_cycle,
                            (unsigned long long)sample.rx_cycle,
                            (unsigned long long)sample.delta,
                            (unsigned)sample.poll_loops,
                            (unsigned)sample.immediate_hit,
                            (unsigned long long)est_link_delay);

    {
      int dir = sample.immediate_hit ? +1 : -1;

      if ((prev_dir != 0) && (dir != prev_dir) && (step > C2C_TRANSFER_BEARLY_SEARCH_MIN_STEP_CYCLES)) {
        step >>= 1;
        if (step < C2C_TRANSFER_BEARLY_SEARCH_MIN_STEP_CYCLES) {
          step = C2C_TRANSFER_BEARLY_SEARCH_MIN_STEP_CYCLES;
        }
      }

      if (dir > 0) {
        uint64_t next_slack = slack + step;
        slack = (next_slack > interval) ? interval : next_slack;
      } else {
        slack = (slack > step) ? (slack - step) : 0u;
      }

      prev_dir = dir;

      C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] adjust seq=%u dir=%s new_slack=%llu new_step=%llu\n",
                              (unsigned)seq,
                              (dir > 0) ? "late(packet already there)" : "early(packet not yet there)",
                              (unsigned long long)slack,
                              (unsigned long long)step);
    }

    prev_tx = sample.tx_cycle;
    prev_rx = sample.rx_cycle;
  }

  C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] summary packets=%u delta_min=%llu delta_avg=%llu delta_max=%llu est_link_delay=%llu final_slack=%llu final_step=%llu\n",
                          (unsigned)total_packets,
                          (unsigned long long)delta_min,
                          (unsigned long long)(delta_sum / (uint64_t)total_packets),
                          (unsigned long long)delta_max,
                          (unsigned long long)est_link_delay,
                          (unsigned long long)slack,
                          (unsigned long long)step);

  C2C_TRANSFER_BEARLY_LOG("[c2c-transfer-bearly] entering wfi\n");
  while (1) {
    __asm__ volatile("wfi");
  }
}

int main(void) {
  app_init();
  app_main();
  return 0;
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    __asm__ volatile("wfi");
  }
}
