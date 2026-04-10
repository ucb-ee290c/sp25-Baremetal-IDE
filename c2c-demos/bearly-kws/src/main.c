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
_Static_assert((KWS_BEARLY_RING_SLOTS > 0u),
               "KWS_BEARLY_RING_SLOTS must be greater than 0.");
_Static_assert(((KWS_BEARLY_CACHE_LINE_BYTES & (KWS_BEARLY_CACHE_LINE_BYTES - 1u)) == 0u),
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
  uint8_t have_case;
  uint16_t case_id;
  uint8_t next_frame_idx;
  int8_t window[KWS_FRAMES_PER_CASE * KWS_MFCC_DIM];
} case_assembler_t;

static volatile kws_mailbox_t *const g_mbox =
    (volatile kws_mailbox_t *)(uintptr_t)KWS_BEARLY_MAILBOX_ADDR;
static volatile kws_ring_slot_t *const g_ring =
    (volatile kws_ring_slot_t *)(uintptr_t)KWS_BEARLY_RING_ADDR;

static inference_stats_t g_stats;
static case_assembler_t g_assembler;
static uint8_t g_summary_printed;
static uint32_t g_cons_local;
static uint8_t g_cache_evict[KWS_BEARLY_CACHE_EVICT_BYTES]
    __attribute__((aligned(KWS_BEARLY_CACHE_LINE_BYTES)));
static volatile uint8_t g_cache_sink;
static uint32_t g_mailbox_poll_count;

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

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t i = 0; i < (uint32_t)KWS_BEARLY_CACHE_EVICT_BYTES; i += KWS_BEARLY_CACHE_LINE_BYTES) {
    /* Read+write each cache line to force capacity eviction across sets/ways. */
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

static inline void publish_mailbox(void) {
  kws_fence_rw();
  cache_evict_all();
  kws_fence_rw();
}

static void reset_case_assembler(void) {
  g_assembler.have_case = 0u;
  g_assembler.case_id = 0u;
  g_assembler.next_frame_idx = 0u;
  memset(g_assembler.window, 0, sizeof(g_assembler.window));
}

static void reset_stats(void) {
  memset(&g_stats, 0, sizeof(g_stats));
  g_stats.cycle_min = UINT64_MAX;
}

static void init_mailbox_and_ring(void) {
  g_mbox->magic = KWS_MAILBOX_MAGIC;
  g_mbox->version = KWS_PROTO_VERSION;
  g_mbox->ring_slots = KWS_BEARLY_RING_SLOTS;
  g_mbox->slot_bytes = (uint32_t)sizeof(kws_ring_slot_t);

  g_mbox->prod_idx = 0u;
  g_mbox->produced_frames = 0u;
  g_mbox->mfcc_failures = 0u;
  g_mbox->last_seq = 0u;
  g_mbox->last_case_id = 0u;
  g_mbox->last_frame_idx = 0u;
  g_mbox->producer_wait_loops = 0u;

  g_mbox->cons_idx = 0u;
  g_mbox->consumed_frames = 0u;
  g_mbox->dropped_frames = 0u;
  g_mbox->seq_errors = 0u;

  g_mbox->flags = KWS_FLAG_READER_READY;

  for (uint32_t i = 0; i < KWS_BEARLY_RING_SLOTS; ++i) {
    g_ring[i].case_id = 0u;
    g_ring[i].frame_idx = 0u;
    g_ring[i].valid = 0u;
    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      g_ring[i].mfcc[k] = 0;
    }
  }

  publish_mailbox();
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

static void process_ring(void) {
  uint32_t prod;
  uint32_t available;
  uint32_t consumed_now = 0u;

  refresh_mailbox();
  prod = g_mbox->prod_idx;
  available = prod - g_cons_local;

  if (available == 0u) {
    return;
  }

  /* Ensure slot payload reads observe producer updates in non-coherent caches. */
  cache_evict_all();
  refresh_mailbox();
  prod = g_mbox->prod_idx;
  available = prod - g_cons_local;
  if (available == 0u) {
    return;
  }

  if (available > KWS_BEARLY_RING_SLOTS) {
    const uint32_t overflow = available - KWS_BEARLY_RING_SLOTS;
    g_mbox->dropped_frames += overflow;
    g_cons_local = prod - KWS_BEARLY_RING_SLOTS;
    available = KWS_BEARLY_RING_SLOTS;
    g_mbox->seq_errors++;
  }

  while (consumed_now < available) {
    const uint32_t seq = g_cons_local + consumed_now + 1u;
    const uint32_t slot_idx = (g_cons_local + consumed_now) % KWS_BEARLY_RING_SLOTS;
    volatile kws_ring_slot_t *slot = &g_ring[slot_idx];

    if (!slot->valid) {
      g_mbox->seq_errors++;
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
      g_mbox->seq_errors++;
      g_assembler.have_case = 1u;
      g_assembler.case_id = slot->case_id;
      g_assembler.next_frame_idx = 0u;
      memset(g_assembler.window, 0, sizeof(g_assembler.window));

      if (slot->frame_idx != 0u) {
        g_mbox->dropped_frames++;
        g_mbox->last_seq = seq;
        g_mbox->last_case_id = slot->case_id;
        g_mbox->last_frame_idx = slot->frame_idx;
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

    g_mbox->last_seq = seq;
    g_mbox->last_case_id = slot->case_id;
    g_mbox->last_frame_idx = slot->frame_idx;
    consumed_now++;
  }

  g_cons_local += consumed_now;
  g_mbox->cons_idx = g_cons_local;
  g_mbox->consumed_frames += consumed_now;
  publish_mailbox();
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
  init_mailbox_and_ring();
  g_summary_printed = 0u;
  g_cons_local = 0u;
  g_cache_sink = 0u;
  g_mailbox_poll_count = 0u;

  KWS_BEARLY_LOG("[bearly-kws] startup: shm=0x%08lx bytes=%u ring_slots=%u slot_bytes=%u cache_evict_bytes=%u mailbox_evict_every=%u\n",
                 (unsigned long)KWS_BEARLY_SHM_BASE,
                 (unsigned)KWS_BEARLY_SHM_BYTES,
                 (unsigned)KWS_BEARLY_RING_SLOTS,
                 (unsigned)sizeof(kws_ring_slot_t),
                 (unsigned)KWS_BEARLY_CACHE_EVICT_BYTES,
                 (unsigned)KWS_BEARLY_MAILBOX_EVICT_EVERY);

  KWS_BEARLY_LOG("[bearly-kws] preparing TinySpeech runtime...\n");
  tinyspeech_prepare_runtime();
  KWS_BEARLY_LOG("[bearly-kws] TinySpeech runtime ready\n");
}

void app_main(void) {
  while (1) {
    process_ring();

    refresh_mailbox();
    if (!g_summary_printed &&
        ((g_mbox->flags & KWS_FLAG_STREAM_DONE) != 0u) &&
        (g_cons_local == g_mbox->prod_idx)) {
      g_summary_printed = 1u;
      KWS_BEARLY_LOG("[bearly-kws] stream-end: produced=%u consumed=%u drops=%u seq_err=%u mfcc_fail=%u infer=%u\n",
                     (unsigned)g_mbox->produced_frames,
                     (unsigned)g_mbox->consumed_frames,
                     (unsigned)g_mbox->dropped_frames,
                     (unsigned)g_mbox->seq_errors,
                     (unsigned)g_mbox->mfcc_failures,
                     (unsigned)g_stats.inferences);
      if (g_stats.inferences > 0u) {
        KWS_BEARLY_LOG("[bearly-kws] cycle-summary: avg=%llu min=%llu max=%llu\n",
                       (unsigned long long)(g_stats.cycle_sum / g_stats.inferences),
                       (unsigned long long)g_stats.cycle_min,
                       (unsigned long long)g_stats.cycle_max);
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
