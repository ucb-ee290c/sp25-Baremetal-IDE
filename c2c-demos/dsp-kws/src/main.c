#include "main.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if KWS_DSP_USE_THREADLIB
#include "mfcc_driver_mc.h"
/* Use explicit threadlib API declarations to avoid picking a legacy hthread.h. */
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
#endif

#define KWS_TEMPLATE_CASES 8u

_Static_assert((KWS_DSP_SHARED_BYTES > sizeof(kws_mailbox_t)),
               "KWS_DSP_SHARED_BYTES must be larger than mailbox size.");
_Static_assert((KWS_DSP_REMOTE_RING_BYTES > 0u),
               "KWS_DSP_REMOTE_RING_BYTES must be greater than 0.");

static mfcc_driver_t g_mfcc;
static float32_t g_templates[KWS_TEMPLATE_CASES][MFCC_DRIVER_FFT_LEN];
static float32_t g_input_window[MFCC_DRIVER_FFT_LEN];

static volatile kws_mailbox_t *const g_mbox =
    (volatile kws_mailbox_t *)(uintptr_t)KWS_DSP_REMOTE_MAILBOX_ADDR;
static volatile kws_ring_slot_t *const g_ring_safe =
    (volatile kws_ring_slot_t *)(uintptr_t)KWS_DSP_REMOTE_RING_ADDR;
static volatile kws_fast_case_slot_t *const g_ring_fast =
    (volatile kws_fast_case_slot_t *)(uintptr_t)KWS_DSP_REMOTE_RING_ADDR;

uint64_t target_frequency = KWS_DSP_TARGET_FREQUENCY_HZ;
static uint32_t g_mfcc_fail_local;

#if KWS_DSP_USE_THREADLIB
static void mc_nop_worker(void *arg) {
  (void)arg;
}
#endif

static float32_t clampf_local(float32_t x, float32_t lo, float32_t hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static void prepare_templates(void) {
  uint32_t lcg = 0x12345678u;
  const float32_t kPi = (float32_t)M_PI;

  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; ++n) {
    const float32_t t = (float32_t)n / MFCC_DRIVER_SAMPLE_RATE_HZ;
    const float32_t frac = (float32_t)n / (float32_t)(MFCC_DRIVER_FFT_LEN - 1u);

    g_templates[0][n] = 0.0f;
    g_templates[1][n] = (n == 0u) ? 1.0f : 0.0f;
    g_templates[2][n] = (n & 1u) ? -0.9f : 0.9f;
    g_templates[3][n] = 0.9f * sinf(2.0f * kPi * 440.0f * t);
    g_templates[4][n] = 0.9f * sinf(2.0f * kPi * 3000.0f * t);

    {
      const float32_t chirp_hz = 100.0f + (2900.0f * frac);
      g_templates[5][n] = 0.85f * sinf(2.0f * kPi * chirp_hz * t);
    }

    lcg = (1664525u * lcg) + 1013904223u;
    {
      const float32_t u = ((float32_t)(lcg & 0x00FFFFFFu) / 8388607.5f) - 1.0f;
      g_templates[6][n] = 0.8f * u;
    }

    {
      const float32_t mix =
          1.4f * sinf(2.0f * kPi * 700.0f * t) + 1.1f * sinf(2.0f * kPi * 1200.0f * t);
      g_templates[7][n] = clampf_local(mix, -0.7f, 0.7f);
    }
  }
}

static void build_window(uint16_t case_id, uint8_t frame_idx, float32_t *dst) {
  const uint32_t template_idx = ((uint32_t)case_id + (uint32_t)frame_idx) % KWS_TEMPLATE_CASES;
  const uint32_t shift = (((uint32_t)case_id * 17u) + ((uint32_t)frame_idx * 11u)) % MFCC_DRIVER_FFT_LEN;
  const float32_t *src = g_templates[template_idx];

  for (uint32_t i = 0; i < MFCC_DRIVER_FFT_LEN; ++i) {
    dst[i] = src[(i + shift) % MFCC_DRIVER_FFT_LEN];
  }
}

static int8_t quantize_mfcc(float32_t x) {
  float32_t qf = (x * (float32_t)KWS_DSP_MFCC_QUANT_SCALE) + (float32_t)KWS_DSP_MFCC_QUANT_ZERO;
  int32_t qi = (int32_t)lrintf(qf);
  if (qi > 127) {
    qi = 127;
  }
  if (qi < -127) {
    qi = -127;
  }
  return (int8_t)qi;
}

static void set_flags(uint32_t flags) {
  g_mbox->flags = flags;
  kws_fence_rw();
}

static void init_writer_mailbox(uint32_t mode, uint32_t ring_slots, uint32_t slot_bytes) {
  g_mbox->magic = KWS_MAILBOX_MAGIC;
  g_mbox->version = KWS_PROTO_VERSION;
  g_mbox->mode = mode;
  g_mbox->ring_slots = ring_slots;
  g_mbox->slot_bytes = slot_bytes;
  g_mbox->num_cases = KWS_DSP_NUM_CASES;
  g_mbox->frames_per_case = KWS_DSP_FRAMES_PER_CASE;
  g_mbox->mfcc_dim = KWS_MFCC_DIM;

  g_mbox->prod_idx = 0u;
  g_mbox->produced_frames = 0u;
  g_mbox->produced_cases = 0u;
  g_mbox->mfcc_failures = 0u;
  g_mbox->last_seq = 0u;
  g_mbox->last_case_id = 0u;
  g_mbox->last_frame_idx = 0u;
  g_mbox->producer_wait_loops = 0u;
  g_mfcc_fail_local = 0u;

  set_flags(KWS_FLAG_WRITER_READY);
}

static mfcc_driver_status_t run_one_mfcc(uint16_t case_id,
                                         uint8_t frame_idx,
                                         int8_t *mfcc_q_out,
                                         uint64_t *mfcc_cycles_sum_io) {
  float32_t mfcc_f32[MFCC_DRIVER_NUM_DCT];
  uint64_t mfcc_cycles = 0;
  mfcc_driver_status_t st;

  build_window(case_id, frame_idx, g_input_window);

#if KWS_DSP_USE_THREADLIB
  st = mfcc_driver_run_sp1024x23x12_f32_mc(&g_mfcc, g_input_window, mfcc_f32, &mfcc_cycles);
  if (st != MFCC_DRIVER_OK) {
    st = mfcc_driver_run_sp1024x23x12_f32(&g_mfcc, g_input_window, mfcc_f32, &mfcc_cycles);
  }
#else
  st = mfcc_driver_run_sp1024x23x12_f32(&g_mfcc, g_input_window, mfcc_f32, &mfcc_cycles);
#endif
  if (st != MFCC_DRIVER_OK) {
    st = mfcc_driver_run_f32(&g_mfcc, g_input_window, mfcc_f32, &mfcc_cycles);
  }

  if (st != MFCC_DRIVER_OK) {
    memset(mfcc_q_out, 0, KWS_MFCC_DIM);
    g_mfcc_fail_local++;
    g_mbox->mfcc_failures = g_mfcc_fail_local;
    KWS_DSP_LOG("[dsp-kws] MFCC failed case=%u frame=%u err=%s\n",
                (unsigned)case_id,
                (unsigned)frame_idx,
                mfcc_driver_status_str(st));
    return st;
  }

  for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
    mfcc_q_out[k] = quantize_mfcc(mfcc_f32[k]);
  }
  *mfcc_cycles_sum_io += mfcc_cycles;
  return MFCC_DRIVER_OK;
}

static void stream_safe(void) {
  const uint32_t ring_slots = KWS_DSP_REMOTE_RING_BYTES / (uint32_t)sizeof(kws_ring_slot_t);
  uint32_t seq = 1u;
  uint32_t prod = 0u;
  uint64_t total_mfcc_cycles = 0u;

  if (ring_slots == 0u) {
    KWS_DSP_LOG("[dsp-kws] SAFE mode invalid ring_slots=0\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

  init_writer_mailbox(KWS_LINK_MODE_SAFE, ring_slots, (uint32_t)sizeof(kws_ring_slot_t));
  for (uint32_t i = 0; i < ring_slots; ++i) {
    g_ring_safe[i].valid = 0u;
  }
  kws_fence_rw();

  for (uint16_t case_id = 0; case_id < (uint16_t)KWS_DSP_NUM_CASES; ++case_id) {
    for (uint8_t frame_idx = 0; frame_idx < (uint8_t)KWS_DSP_FRAMES_PER_CASE; ++frame_idx) {
      int8_t mfcc_q[KWS_MFCC_DIM];
      const uint32_t slot_idx = prod % ring_slots;
      volatile kws_ring_slot_t *slot = &g_ring_safe[slot_idx];

      (void)run_one_mfcc(case_id, frame_idx, mfcc_q, &total_mfcc_cycles);

      slot->valid = 0u;
      kws_fence_rw();
      slot->case_id = case_id;
      slot->frame_idx = frame_idx;
      for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
        slot->mfcc[k] = mfcc_q[k];
      }
      kws_fence_rw();
      slot->valid = 1u;

      prod++;
      g_mbox->prod_idx = prod;
      g_mbox->produced_frames = prod;
      g_mbox->last_seq = seq++;
      g_mbox->last_case_id = case_id;
      g_mbox->last_frame_idx = frame_idx;
    }

    g_mbox->produced_cases = (uint32_t)case_id + 1u;
    if (KWS_DSP_LOG_ENABLE && (((uint32_t)case_id + 1u) % KWS_DSP_PROGRESS_EVERY_CASES == 0u)) {
      KWS_DSP_LOG("[dsp-kws] SAFE streamed %u/%u cases\n",
                  (unsigned)((uint32_t)case_id + 1u),
                  (unsigned)KWS_DSP_NUM_CASES);
    }
  }

  set_flags(KWS_FLAG_WRITER_READY | KWS_FLAG_STREAM_DONE);
  KWS_DSP_LOG("[dsp-kws] SAFE done: cases=%u produced_frames=%u mfcc_fail=%u mfcc_cycles=%llu\n",
              (unsigned)KWS_DSP_NUM_CASES,
              (unsigned)prod,
              (unsigned)g_mfcc_fail_local,
              (unsigned long long)total_mfcc_cycles);
}

static void stream_fast(void) {
  const uint32_t fast_slots = KWS_DSP_REMOTE_RING_BYTES / (uint32_t)sizeof(kws_fast_case_slot_t);
  uint64_t total_mfcc_cycles = 0u;

  if (fast_slots == 0u) {
    KWS_DSP_LOG("[dsp-kws] FAST mode invalid fast_slots=0\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

  init_writer_mailbox(KWS_LINK_MODE_FAST, fast_slots, (uint32_t)sizeof(kws_fast_case_slot_t));
  for (uint32_t i = 0; i < fast_slots; ++i) {
    g_ring_fast[i].commit_seq = 0u;
  }
  kws_fence_rw();

  for (uint16_t case_id = 0; case_id < (uint16_t)KWS_DSP_NUM_CASES; ++case_id) {
    volatile kws_fast_case_slot_t *slot = &g_ring_fast[((uint32_t)case_id) % fast_slots];
    const uint32_t commit_seq = (uint32_t)case_id + 1u;
    uint32_t case_seq = (uint32_t)case_id * KWS_DSP_FRAMES_PER_CASE;

    slot->commit_seq = 0u;
    kws_fence_rw();
    slot->case_id = case_id;
    slot->reserved = 0u;

    for (uint8_t frame_idx = 0; frame_idx < (uint8_t)KWS_DSP_FRAMES_PER_CASE; ++frame_idx) {
      int8_t mfcc_q[KWS_MFCC_DIM];
      const uint32_t base = ((uint32_t)frame_idx) * KWS_MFCC_DIM;

      (void)run_one_mfcc(case_id, frame_idx, mfcc_q, &total_mfcc_cycles);
      for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
        slot->mfcc[base + k] = mfcc_q[k];
      }
      case_seq++;
    }

    kws_fence_rw();
    slot->commit_seq = commit_seq;
    kws_fence_rw();

    g_mbox->prod_idx = commit_seq;
    g_mbox->produced_cases = commit_seq;
    g_mbox->produced_frames = commit_seq * KWS_DSP_FRAMES_PER_CASE;
    g_mbox->last_seq = case_seq;
    g_mbox->last_case_id = case_id;
    g_mbox->last_frame_idx = KWS_DSP_FRAMES_PER_CASE - 1u;

    if (KWS_DSP_LOG_ENABLE && (((uint32_t)case_id + 1u) % KWS_DSP_PROGRESS_EVERY_CASES == 0u)) {
      KWS_DSP_LOG("[dsp-kws] FAST streamed %u/%u cases\n",
                  (unsigned)((uint32_t)case_id + 1u),
                  (unsigned)KWS_DSP_NUM_CASES);
    }
  }

  set_flags(KWS_FLAG_WRITER_READY | KWS_FLAG_STREAM_DONE);
  KWS_DSP_LOG("[dsp-kws] FAST done: cases=%u produced_cases=%u mfcc_fail=%u mfcc_cycles=%llu\n",
              (unsigned)KWS_DSP_NUM_CASES,
              (unsigned)KWS_DSP_NUM_CASES,
              (unsigned)g_mfcc_fail_local,
              (unsigned long long)total_mfcc_cycles);
}

void app_init(void) {
  init_test(target_frequency);

#if KWS_DSP_USE_THREADLIB
  hthread_init();
  /* Warm hart1 once so steady-state dispatch has no cold-start hiccup. */
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
#endif

  prepare_templates();
  if (mfcc_driver_init(&g_mfcc) != MFCC_DRIVER_OK) {
    KWS_DSP_LOG("[dsp-kws] MFCC init failed\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

  KWS_DSP_LOG("[dsp-kws] one-way writer mode=%u mailbox=0x%08lx data=0x%08lx bytes=%u\n",
              (unsigned)KWS_DSP_LINK_MODE,
              (unsigned long)KWS_DSP_REMOTE_MAILBOX_ADDR,
              (unsigned long)KWS_DSP_REMOTE_RING_ADDR,
              (unsigned)KWS_DSP_REMOTE_RING_BYTES);
}

void app_main(void) {
  if (KWS_DSP_LINK_MODE == KWS_LINK_MODE_FAST) {
    stream_fast();
  } else {
    stream_safe();
  }

#if KWS_DSP_USE_THREADLIB
  mfcc_driver_mc_shutdown();
#endif

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
