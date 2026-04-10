#include "main.h"

#include "tensor.h"

#if KWS_BEARLY_USE_THREADLIB
/* Use explicit threadlib API declarations to avoid picking a legacy hthread.h. */
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
#endif

_Static_assert((KWS_BEARLY_SHM_BYTES > sizeof(kws_mailbox_t)),
               "KWS_BEARLY_SHM_BYTES must be larger than mailbox size.");
_Static_assert((KWS_BEARLY_CACHE_LINE_BYTES & (KWS_BEARLY_CACHE_LINE_BYTES - 1u)) == 0u,
               "KWS_BEARLY_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((KWS_BEARLY_CACHE_EVICT_BYTES >= KWS_BEARLY_CACHE_LINE_BYTES),
               "KWS_BEARLY_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((KWS_BEARLY_CACHE_EVICT_BYTES % KWS_BEARLY_CACHE_LINE_BYTES) == 0u,
               "KWS_BEARLY_CACHE_EVICT_BYTES must be a multiple of cache line size.");

typedef struct {
  uint32_t inferences;
  uint64_t cycle_sum;
  uint64_t cycle_min;
  uint64_t cycle_max;
  uint32_t class_hist[TINYSPEECH_NUM_CLASSES];
} inference_stats_t;

typedef struct {
  uint32_t consumed_frames;
  uint32_t dropped_frames;
  uint32_t seq_errors;
  uint32_t last_seq;
  uint32_t last_case_id;
  uint32_t last_frame_idx;
} stream_stats_t;

typedef struct {
  uint32_t cases;
  uint64_t link_cycle_sum;
  uint64_t link_cycle_min;
  uint64_t link_cycle_max;
  uint64_t tx_cycle_sum;
  uint64_t tx_cycle_min;
  uint64_t tx_cycle_max;
  uint64_t first_tx_cycle;
  uint64_t first_rx_cycle;
  uint64_t last_rx_cycle;
  uint8_t have_first;
} link_stats_t;

typedef struct {
  uint8_t have_case;
  uint16_t case_id;
  uint8_t next_frame_idx;
  int8_t window[KWS_FRAMES_PER_CASE * KWS_MFCC_DIM];
} case_assembler_t;

static volatile kws_mailbox_t *const g_mbox =
    (volatile kws_mailbox_t *)(uintptr_t)KWS_BEARLY_MAILBOX_ADDR;
static volatile kws_ring_slot_t *const g_ring_safe =
    (volatile kws_ring_slot_t *)(uintptr_t)KWS_BEARLY_RING_ADDR;
static volatile kws_fast_case_slot_t *const g_ring_fast =
    (volatile kws_fast_case_slot_t *)(uintptr_t)KWS_BEARLY_RING_ADDR;

static inference_stats_t g_stats;
static stream_stats_t g_stream;
static link_stats_t g_link;
static case_assembler_t g_assembler;
static uint8_t g_summary_printed;
static uint32_t g_mode;
static uint32_t g_cons_frames;
static uint32_t g_cons_cases;
static uint8_t g_cache_evict[KWS_BEARLY_CACHE_EVICT_BYTES]
    __attribute__((aligned(KWS_BEARLY_CACHE_LINE_BYTES)));
static volatile uint8_t g_cache_sink;
static uint32_t g_mailbox_poll_count;
static uint32_t g_wait_poll_count;

uint64_t target_frequency = KWS_BEARLY_TARGET_FREQUENCY_HZ;

#if KWS_BEARLY_USE_THREADLIB
static void mc_nop_worker(void *arg) {
  (void)arg;
}
#endif

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline double cycles_to_us(uint64_t cycles) {
  if (target_frequency == 0u) {
    return 0.0;
  }
  return ((double)cycles * 1000000.0) / (double)target_frequency;
}

static inline double payload_mb_per_s(uint32_t payload_bytes, uint64_t cycles) {
  if ((target_frequency == 0u) || (cycles == 0u)) {
    return 0.0;
  }
  return ((double)payload_bytes * (double)target_frequency) / ((double)cycles * 1000000.0);
}

static inline double payload_mbit_per_s(uint32_t payload_bytes, uint64_t cycles) {
  return 8.0 * payload_mb_per_s(payload_bytes, cycles);
}

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t i = 0; i < (uint32_t)KWS_BEARLY_CACHE_EVICT_BYTES; i += KWS_BEARLY_CACHE_LINE_BYTES) {
    sink ^= buf[i];
    buf[i] = (uint8_t)(sink + (uint8_t)i);
  }

  g_cache_sink = sink;
  kws_fence_rw();
}

static inline void refresh_mailbox(void) {
  if ((KWS_BEARLY_MAILBOX_EVICT_EVERY <= 1u) ||
      ((g_mailbox_poll_count++ % KWS_BEARLY_MAILBOX_EVICT_EVERY) == 0u)) {
    cache_evict_all();
  }
  kws_fence_rw();
}

#if KWS_BEARLY_CLEAR_SHM_ON_BOOT
static void clear_shared_scratchpad(void) {
  volatile uint8_t *shm = (volatile uint8_t *)(uintptr_t)KWS_BEARLY_SHM_BASE;
  for (uint32_t i = 0; i < (uint32_t)KWS_BEARLY_SHM_BYTES; ++i) {
    shm[i] = 0u;
  }
  kws_fence_rw();
  cache_evict_all();
  kws_fence_rw();
}

static inline uint8_t mailbox_writer_ready_seen(void) {
  return (uint8_t)((g_mbox->magic == KWS_MAILBOX_MAGIC) &&
                   (g_mbox->version == KWS_PROTO_VERSION) &&
                   ((g_mbox->flags & KWS_FLAG_WRITER_READY) != 0u));
}
#endif

static void maybe_clear_shared_scratchpad(void) {
#if KWS_BEARLY_CLEAR_SHM_ON_BOOT
  refresh_mailbox();
  if (mailbox_writer_ready_seen()) {
    KWS_BEARLY_LOG("[bearly-kws] skip clear: writer already active magic=0x%08x flags=0x%08x mode=%u prod_frames=%u prod_cases=%u\n",
                   (unsigned)g_mbox->magic,
                   (unsigned)g_mbox->flags,
                   (unsigned)g_mbox->mode,
                   (unsigned)g_mbox->produced_frames,
                   (unsigned)g_mbox->produced_cases);
    return;
  }
  clear_shared_scratchpad();
  KWS_BEARLY_LOG("[bearly-kws] cleared shared scratchpad base=0x%08lx bytes=%u\n",
                 (unsigned long)KWS_BEARLY_SHM_BASE,
                 (unsigned)KWS_BEARLY_SHM_BYTES);
#endif
}

static void reset_case_assembler(void) {
  g_assembler.have_case = 0u;
  g_assembler.case_id = 0u;
  g_assembler.next_frame_idx = 0u;
  memset(g_assembler.window, 0, sizeof(g_assembler.window));
}

static void reset_stats(void) {
  memset(&g_stats, 0, sizeof(g_stats));
  memset(&g_stream, 0, sizeof(g_stream));
  memset(&g_link, 0, sizeof(g_link));
  g_stats.cycle_min = UINT64_MAX;
  g_link.link_cycle_min = UINT64_MAX;
  g_link.tx_cycle_min = UINT64_MAX;
}

static void run_inference_for_current_case(void) {
  uint8_t shape[4] = {1, 1, KWS_MFCC_DIM, KWS_FRAMES_PER_CASE};
  Tensor input = create_tensor(shape, 4);
  uint64_t t0;
  uint64_t t1;
  uint64_t cycles;
  Tensor probs;
  float max_prob = 0.0f;
  int32_t pred;

  memcpy(input.data, g_assembler.window, sizeof(g_assembler.window));

  t0 = rdcycle64();
  probs = tinyspeech_run_inference(&input);
  pred = tinyspeech_argmax(&probs, &max_prob);
  t1 = rdcycle64();
  cycles = t1 - t0;

  g_stats.inferences++;
  g_stats.cycle_sum += cycles;
  if (cycles < g_stats.cycle_min) {
    g_stats.cycle_min = cycles;
  }
  if (cycles > g_stats.cycle_max) {
    g_stats.cycle_max = cycles;
  }
  if ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) {
    g_stats.class_hist[pred]++;
  }

  if ((g_stats.inferences % KWS_BEARLY_PROGRESS_EVERY) == 0u) {
    const uint64_t avg = g_stats.cycle_sum / g_stats.inferences;
    KWS_BEARLY_LOG("[bearly-kws] infer=%u case=%u pred=%ld score=%.4f cycles=%llu avg=%llu min=%llu max=%llu\n",
                   (unsigned)g_stats.inferences,
                   (unsigned)g_assembler.case_id,
                   (long)pred,
                   max_prob,
                   (unsigned long long)cycles,
                   (unsigned long long)avg,
                   (unsigned long long)g_stats.cycle_min,
                   (unsigned long long)g_stats.cycle_max);
  }

  free_tensor(&probs);
  free_tensor(&input);
}

static void run_inference_for_case_payload(uint16_t case_id, const volatile int8_t *payload) {
  for (uint32_t i = 0; i < KWS_CASE_PAYLOAD_BYTES; ++i) {
    g_assembler.window[i] = payload[i];
  }
  g_assembler.case_id = case_id;
  run_inference_for_current_case();
}

static uint32_t wait_for_writer_mode(void) {
  while (1) {
    g_wait_poll_count++;
    refresh_mailbox();

    if ((g_mbox->magic != KWS_MAILBOX_MAGIC) ||
        (g_mbox->version != KWS_PROTO_VERSION) ||
        ((g_mbox->flags & KWS_FLAG_WRITER_READY) == 0u)) {
#if KWS_BEARLY_WAIT_LOG_EVERY
      if ((g_wait_poll_count % KWS_BEARLY_WAIT_LOG_EVERY) == 0u) {
        KWS_BEARLY_LOG("[bearly-kws] polling wait=%u magic=0x%08x ver=%u flags=0x%08x mode=%u prod_frames=%u prod_cases=%u ring_slots=%u slot_bytes=%u\n",
                       (unsigned)g_wait_poll_count,
                       (unsigned)g_mbox->magic,
                       (unsigned)g_mbox->version,
                       (unsigned)g_mbox->flags,
                       (unsigned)g_mbox->mode,
                       (unsigned)g_mbox->produced_frames,
                       (unsigned)g_mbox->produced_cases,
                       (unsigned)g_mbox->ring_slots,
                       (unsigned)g_mbox->slot_bytes);
      }
#endif
      __asm__ volatile("nop");
      continue;
    }

    if ((g_mbox->mode != KWS_LINK_MODE_SAFE) &&
        (g_mbox->mode != KWS_LINK_MODE_FAST)) {
      __asm__ volatile("nop");
      continue;
    }

    if ((KWS_BEARLY_EXPECTED_MODE != 0xFFFFFFFFu) &&
        (g_mbox->mode != KWS_BEARLY_EXPECTED_MODE)) {
      KWS_BEARLY_LOG("[bearly-kws] mode mismatch expected=%u got=%u\n",
                     (unsigned)KWS_BEARLY_EXPECTED_MODE,
                     (unsigned)g_mbox->mode);
      while (1) {
        __asm__ volatile("wfi");
      }
    }

    return g_mbox->mode;
  }
}

static void process_safe(void) {
  uint32_t prod;
  uint32_t available;
  uint32_t consumed_now = 0u;
  uint32_t ring_slots = g_mbox->ring_slots;

  if ((ring_slots == 0u) || (g_mbox->slot_bytes != (uint32_t)sizeof(kws_ring_slot_t))) {
    return;
  }

  refresh_mailbox();
  prod = g_mbox->prod_idx;
  available = prod - g_cons_frames;
  if (available == 0u) {
    return;
  }

  cache_evict_all();
  refresh_mailbox();
  prod = g_mbox->prod_idx;
  available = prod - g_cons_frames;
  if (available == 0u) {
    return;
  }

  if (available > ring_slots) {
    const uint32_t overflow = available - ring_slots;
    g_stream.dropped_frames += overflow;
    g_cons_frames = prod - ring_slots;
    available = ring_slots;
    g_stream.seq_errors++;
  }

  while (consumed_now < available) {
    const uint32_t seq = g_cons_frames + consumed_now + 1u;
    const uint32_t slot_idx = (g_cons_frames + consumed_now) % ring_slots;
    volatile kws_ring_slot_t *slot = &g_ring_safe[slot_idx];

    if (!slot->valid) {
      g_stream.seq_errors++;
      break;
    }

    if (!g_assembler.have_case) {
      g_assembler.have_case = 1u;
      g_assembler.case_id = slot->case_id;
      g_assembler.next_frame_idx = 0u;
      memset(g_assembler.window, 0, sizeof(g_assembler.window));
    }

    if ((slot->case_id != g_assembler.case_id) ||
        (slot->frame_idx != g_assembler.next_frame_idx)) {
      g_stream.seq_errors++;
      g_assembler.have_case = 1u;
      g_assembler.case_id = slot->case_id;
      g_assembler.next_frame_idx = 0u;
      memset(g_assembler.window, 0, sizeof(g_assembler.window));
      if (slot->frame_idx != 0u) {
        g_stream.dropped_frames++;
        g_stream.last_seq = seq;
        g_stream.last_case_id = slot->case_id;
        g_stream.last_frame_idx = slot->frame_idx;
        consumed_now++;
        continue;
      }
    }

    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      g_assembler.window[((uint32_t)slot->frame_idx * KWS_MFCC_DIM) + k] = slot->mfcc[k];
    }
    g_assembler.next_frame_idx++;

    if (g_assembler.next_frame_idx >= KWS_FRAMES_PER_CASE) {
      run_inference_for_current_case();
      reset_case_assembler();
    }

    g_stream.last_seq = seq;
    g_stream.last_case_id = slot->case_id;
    g_stream.last_frame_idx = slot->frame_idx;
    consumed_now++;
  }

  g_cons_frames += consumed_now;
  g_stream.consumed_frames += consumed_now;
}

static void process_fast(void) {
  uint32_t produced_cases;
  uint32_t available_cases;
  uint32_t ring_slots = g_mbox->ring_slots;

  if ((ring_slots == 0u) || (g_mbox->slot_bytes != (uint32_t)sizeof(kws_fast_case_slot_t))) {
    return;
  }

  refresh_mailbox();
  produced_cases = g_mbox->produced_cases;
  available_cases = produced_cases - g_cons_cases;
  if (available_cases == 0u) {
    return;
  }

  if (available_cases > ring_slots) {
    const uint32_t overflow_cases = available_cases - ring_slots;
    g_cons_cases += overflow_cases;
    g_stream.dropped_frames += overflow_cases * KWS_FRAMES_PER_CASE;
    g_stream.seq_errors++;
    available_cases = ring_slots;
  }

  cache_evict_all();
  refresh_mailbox();
  produced_cases = g_mbox->produced_cases;

  while (g_cons_cases < produced_cases) {
    const uint32_t expected_seq = g_cons_cases + 1u;
    const uint32_t slot_idx = g_cons_cases % ring_slots;
    volatile kws_fast_case_slot_t *slot = &g_ring_fast[slot_idx];
    const uint32_t commit_seq = slot->commit_seq;
    uint64_t rx_cycle;
    uint64_t tx_cycle_start;
    uint64_t tx_cycle_commit;
    uint64_t link_cycles;
    uint64_t tx_cycles;

    if (commit_seq < expected_seq) {
      break;
    }
    if (commit_seq > expected_seq) {
      const uint32_t missed = commit_seq - expected_seq;
      g_cons_cases += missed;
      g_stream.dropped_frames += missed * KWS_FRAMES_PER_CASE;
      g_stream.seq_errors++;
      continue;
    }

    kws_fence_rw();
    rx_cycle = rdcycle64();
    tx_cycle_start = slot->tx_cycle_start;
    tx_cycle_commit = slot->tx_cycle_commit;
    link_cycles = rx_cycle - tx_cycle_start;
    tx_cycles = tx_cycle_commit - tx_cycle_start;

    if (!g_link.have_first) {
      g_link.have_first = 1u;
      g_link.first_tx_cycle = tx_cycle_start;
      g_link.first_rx_cycle = rx_cycle;
    }
    g_link.last_rx_cycle = rx_cycle;
    g_link.cases++;
    g_link.link_cycle_sum += link_cycles;
    g_link.tx_cycle_sum += tx_cycles;
    if (link_cycles < g_link.link_cycle_min) {
      g_link.link_cycle_min = link_cycles;
    }
    if (link_cycles > g_link.link_cycle_max) {
      g_link.link_cycle_max = link_cycles;
    }
    if (tx_cycles < g_link.tx_cycle_min) {
      g_link.tx_cycle_min = tx_cycles;
    }
    if (tx_cycles > g_link.tx_cycle_max) {
      g_link.tx_cycle_max = tx_cycles;
    }

    run_inference_for_case_payload(slot->case_id, slot->mfcc);
    g_cons_cases++;
    g_stream.consumed_frames += KWS_FRAMES_PER_CASE;
    g_stream.last_seq = g_cons_cases * KWS_FRAMES_PER_CASE;
    g_stream.last_case_id = slot->case_id;
    g_stream.last_frame_idx = KWS_FRAMES_PER_CASE - 1u;
  }
}

void app_init(void) {
  init_test(target_frequency);

#if KWS_BEARLY_USE_THREADLIB
  hthread_init();
  /* Warm hart1 once so the first multicore inference is not penalized. */
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
#endif

  reset_stats();
  reset_case_assembler();
  g_summary_printed = 0u;
  g_mode = KWS_LINK_MODE_SAFE;
  g_cons_frames = 0u;
  g_cons_cases = 0u;
  g_cache_sink = 0u;
  g_mailbox_poll_count = 0u;
  g_wait_poll_count = 0u;
  maybe_clear_shared_scratchpad();

  KWS_BEARLY_LOG("[bearly-kws] startup one-way reader: shm=0x%08lx bytes=%u cache_evict_bytes=%u mailbox_evict_every=%u\n",
                 (unsigned long)KWS_BEARLY_SHM_BASE,
                 (unsigned)KWS_BEARLY_SHM_BYTES,
                 (unsigned)KWS_BEARLY_CACHE_EVICT_BYTES,
                 (unsigned)KWS_BEARLY_MAILBOX_EVICT_EVERY);

  KWS_BEARLY_LOG("[bearly-kws] preparing TinySpeech runtime...\n");
  tinyspeech_prepare_runtime();
  KWS_BEARLY_LOG("[bearly-kws] TinySpeech runtime ready\n");
}

void app_main(void) {
  g_mode = wait_for_writer_mode();
  KWS_BEARLY_LOG("[bearly-kws] writer detected mode=%s ring_slots=%u slot_bytes=%u num_cases=%u\n",
                 (g_mode == KWS_LINK_MODE_FAST) ? "FAST" : "SAFE",
                 (unsigned)g_mbox->ring_slots,
                 (unsigned)g_mbox->slot_bytes,
                 (unsigned)g_mbox->num_cases);

  while (1) {
    if (g_mode == KWS_LINK_MODE_FAST) {
      process_fast();
    } else {
      process_safe();
    }

    refresh_mailbox();
    if (!g_summary_printed && ((g_mbox->flags & KWS_FLAG_STREAM_DONE) != 0u)) {
      uint8_t done = 0u;
      if (g_mode == KWS_LINK_MODE_FAST) {
        done = (g_cons_cases == g_mbox->produced_cases);
      } else {
        done = (g_cons_frames == g_mbox->prod_idx);
      }

      if (done) {
        g_summary_printed = 1u;
        KWS_BEARLY_LOG("[bearly-kws] stream-end mode=%s produced_frames=%u produced_cases=%u consumed_frames=%u drops=%u seq_err=%u mfcc_fail=%u infer=%u\n",
                       (g_mode == KWS_LINK_MODE_FAST) ? "FAST" : "SAFE",
                       (unsigned)g_mbox->produced_frames,
                       (unsigned)g_mbox->produced_cases,
                       (unsigned)g_stream.consumed_frames,
                       (unsigned)g_stream.dropped_frames,
                       (unsigned)g_stream.seq_errors,
                       (unsigned)g_mbox->mfcc_failures,
                       (unsigned)g_stats.inferences);
        if ((g_mode == KWS_LINK_MODE_FAST) && (g_link.cases > 0u)) {
          const uint64_t link_avg = g_link.link_cycle_sum / g_link.cases;
          const uint64_t tx_avg = g_link.tx_cycle_sum / g_link.cases;
          const uint64_t first_msg_to_last_case = g_link.last_rx_cycle - g_link.first_tx_cycle;
          const uint64_t first_rx_to_last_case = g_link.last_rx_cycle - g_link.first_rx_cycle;
          KWS_BEARLY_LOG("[bearly-kws] link-summary: payload=%uB/case link_cycles(best/avg/worst)=%llu/%llu/%llu tx_cycles(best/avg/worst)=%llu/%llu/%llu\n",
                         (unsigned)KWS_CASE_PAYLOAD_BYTES,
                         (unsigned long long)g_link.link_cycle_min,
                         (unsigned long long)link_avg,
                         (unsigned long long)g_link.link_cycle_max,
                         (unsigned long long)g_link.tx_cycle_min,
                         (unsigned long long)tx_avg,
                         (unsigned long long)g_link.tx_cycle_max);
          KWS_BEARLY_LOG("[bearly-kws] link-speed: payloadMBps(best/avg)=%.3f/%.3f\n",
                         payload_mb_per_s(KWS_CASE_PAYLOAD_BYTES, g_link.link_cycle_min),
                         payload_mb_per_s(KWS_CASE_PAYLOAD_BYTES, link_avg));
          KWS_BEARLY_LOG("[bearly-kws] link-speed: payloadMbit/s(best/avg)=%.3f/%.3f\n",
                         payload_mbit_per_s(KWS_CASE_PAYLOAD_BYTES, g_link.link_cycle_min),
                         payload_mbit_per_s(KWS_CASE_PAYLOAD_BYTES, link_avg));
          KWS_BEARLY_LOG("[bearly-kws] span: first-msg->last-case cycles=%llu (%.3f us), first-rx->last-case cycles=%llu (%.3f us)\n",
                         (unsigned long long)first_msg_to_last_case,
                         cycles_to_us(first_msg_to_last_case),
                         (unsigned long long)first_rx_to_last_case,
                         cycles_to_us(first_rx_to_last_case));
        }
        if (g_stats.inferences > 0u) {
          KWS_BEARLY_LOG("[bearly-kws] cycle-summary: avg=%llu min=%llu max=%llu\n",
                         (unsigned long long)(g_stats.cycle_sum / g_stats.inferences),
                         (unsigned long long)g_stats.cycle_min,
                         (unsigned long long)g_stats.cycle_max);
        }
      }
    }

    __asm__ volatile("nop");
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
