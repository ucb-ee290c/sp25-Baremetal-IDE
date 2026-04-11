#include "main.h"
#include "yes_test_005_signal.h"

#if KWS_DSP_ROLLING_USE_THREADLIB
#include "mfcc_driver_mc.h"
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

_Static_assert((KWS_DSP_ROLLING_FRAME_ADDR >= KWS_DSP_ROLLING_SHARED_BASE),
               "KWS_DSP_ROLLING_FRAME_ADDR must be inside shared region.");
_Static_assert(((KWS_DSP_ROLLING_FRAME_ADDR - KWS_DSP_ROLLING_SHARED_BASE) + KWS_ROLLING_FRAME_BYTES) <= KWS_DSP_ROLLING_SHARED_BYTES,
               "Rolling frame payload does not fit in shared region.");
_Static_assert((KWS_DSP_ROLLING_CACHE_LINE_BYTES & (KWS_DSP_ROLLING_CACHE_LINE_BYTES - 1u)) == 0u,
               "KWS_DSP_ROLLING_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((KWS_DSP_ROLLING_CACHE_EVICT_BYTES >= KWS_DSP_ROLLING_CACHE_LINE_BYTES),
               "KWS_DSP_ROLLING_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((KWS_DSP_ROLLING_CACHE_EVICT_BYTES % KWS_DSP_ROLLING_CACHE_LINE_BYTES) == 0u,
               "KWS_DSP_ROLLING_CACHE_EVICT_BYTES must be a multiple of cache line size.");
_Static_assert(KWS_DSP_YES005_NUM_SAMPLES >= ((((uint32_t)KWS_DSP_ROLLING_FRAMES_PER_CASE - 1u) * KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES) + MFCC_DRIVER_FFT_LEN),
               "Embedded yes_test_005 signal does not cover all requested MFCC frames.");

static mfcc_driver_t g_mfcc;
static float32_t g_input_window[MFCC_DRIVER_FFT_LEN];
static uint8_t g_cache_evict[KWS_DSP_ROLLING_CACHE_EVICT_BYTES]
    __attribute__((aligned(0x8000)));
static volatile uint8_t g_cache_sink;
static uint32_t g_mfcc_fail_local;
static uint32_t g_commit_seq_local;

static volatile uint32_t *const g_commit_seq =
    (volatile uint32_t *)(uintptr_t)KWS_DSP_ROLLING_COMMIT_SEQ_ADDR;
static volatile int8_t *const g_frame =
    (volatile int8_t *)(uintptr_t)KWS_DSP_ROLLING_FRAME_ADDR;

uint64_t target_frequency = KWS_DSP_ROLLING_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline void kws_fence_rw_local(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline void cache_writeback_pressure(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t pass = 0; pass < 3u; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)KWS_DSP_ROLLING_CACHE_EVICT_BYTES; i += KWS_DSP_ROLLING_CACHE_LINE_BYTES) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    kws_fence_rw_local();
  }

  g_cache_sink = sink;
  kws_fence_rw_local();
}

static void load_yes005_window(uint8_t frame_idx, float32_t *dst) {
  const uint32_t start = ((uint32_t)frame_idx) * KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES;

  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; ++n) {
    const uint32_t idx = start + n;
    dst[n] = (idx < KWS_DSP_YES005_NUM_SAMPLES) ? g_kws_dsp_yes005_signal[idx] : 0.0f;
  }
}

static int8_t quantize_mfcc(float32_t x) {
  float32_t qf = (x * (float32_t)KWS_DSP_ROLLING_MFCC_QUANT_SCALE) + (float32_t)KWS_DSP_ROLLING_MFCC_QUANT_ZERO;
  int32_t qi = (int32_t)lrintf(qf);
  if (qi > 127) {
    qi = 127;
  }
  if (qi < -127) {
    qi = -127;
  }
  return (int8_t)qi;
}

static mfcc_driver_status_t compute_one_mfcc_frame(uint8_t frame_idx,
                                                    int8_t *mfcc_q_out,
                                                    uint64_t *mfcc_cycles_out) {
  float32_t mfcc_f32[MFCC_DRIVER_NUM_DCT];
  uint64_t mfcc_cycles = 0;
  mfcc_driver_status_t st;

  load_yes005_window(frame_idx, g_input_window);

#if KWS_DSP_ROLLING_USE_THREADLIB
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
    *mfcc_cycles_out = mfcc_cycles;
    return st;
  }

  for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
    mfcc_q_out[k] = quantize_mfcc(mfcc_f32[k]);
  }

  *mfcc_cycles_out = mfcc_cycles;
  return MFCC_DRIVER_OK;
}

static void publish_one_frame(const int8_t *mfcc_q) {
  for (uint32_t i = 0; i < KWS_ROLLING_FRAME_BYTES; ++i) {
    g_frame[i] = mfcc_q[i];
  }

  kws_fence_rw_local();
  cache_writeback_pressure();

  g_commit_seq_local++;
  *g_commit_seq = g_commit_seq_local;

  kws_fence_rw_local();
  cache_writeback_pressure();
  kws_fence_rw_local();
}

static void maybe_wait_send_interval(uint64_t *next_send_cycle_io) {
#if KWS_DSP_ROLLING_SEND_INTERVAL_CYCLES
  uint64_t now;

  while ((now = rdcycle64()) < *next_send_cycle_io) {
    (void)now;
    __asm__ volatile("nop");
  }

  *next_send_cycle_io += KWS_DSP_ROLLING_SEND_INTERVAL_CYCLES;
#else
  (void)next_send_cycle_io;
#endif
}

void app_init(void) {
  init_test(target_frequency);
  g_cache_sink = 0u;
  g_mfcc_fail_local = 0u;
  g_commit_seq_local = 0u;

#if KWS_DSP_ROLLING_USE_THREADLIB
  hthread_init();
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
#endif

  if (mfcc_driver_init(&g_mfcc) != MFCC_DRIVER_OK) {
    KWS_DSP_ROLLING_LOG("[dsp-kws-rolling] MFCC init failed\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

  *g_commit_seq = 0u;
  kws_fence_rw_local();
  cache_writeback_pressure();

  KWS_DSP_ROLLING_LOG("[dsp-kws-rolling] init seq_addr=0x%08lx frame_addr=0x%08lx frame_bytes=%u signal=%s\n",
                      (unsigned long)KWS_DSP_ROLLING_COMMIT_SEQ_ADDR,
                      (unsigned long)KWS_DSP_ROLLING_FRAME_ADDR,
                      (unsigned)KWS_ROLLING_FRAME_BYTES,
                      KWS_DSP_YES005_MEMBER);
}

void app_main(void) {
  uint64_t total_frames = 0u;
  uint64_t total_mfcc_cycles = 0u;
  uint64_t next_send_cycle = rdcycle64();

#if KWS_DSP_ROLLING_DEBUG_WRITE_ENABLE
  {
    volatile uint32_t *debug_word =
        (volatile uint32_t *)(uintptr_t)KWS_DSP_ROLLING_COMMIT_SEQ_ADDR;
    *debug_word = 0xDEADBEEFu;
    kws_fence_rw_local();
    cache_writeback_pressure();
    kws_fence_rw_local();
    KWS_DSP_ROLLING_LOG(
        "[dsp-kws-rolling] DEBUG wrote 0xDEADBEEF to seq addr=0x%08lx\n",
        (unsigned long)KWS_DSP_ROLLING_COMMIT_SEQ_ADDR);
    *debug_word = 0u;
    kws_fence_rw_local();
    cache_writeback_pressure();
    kws_fence_rw_local();
  }
#endif

  KWS_DSP_ROLLING_LOG("[dsp-kws-rolling] priming %u frames\n",
                      (unsigned)KWS_DSP_ROLLING_FRAMES_PER_CASE);

  for (uint8_t frame_idx = 0; frame_idx < (uint8_t)KWS_DSP_ROLLING_FRAMES_PER_CASE; ++frame_idx) {
    int8_t mfcc_q[KWS_MFCC_DIM];
    uint64_t mfcc_cycles = 0u;
    mfcc_driver_status_t st;

    maybe_wait_send_interval(&next_send_cycle);
    st = compute_one_mfcc_frame(frame_idx, mfcc_q, &mfcc_cycles);
    publish_one_frame(mfcc_q);

    total_frames++;
    total_mfcc_cycles += mfcc_cycles;

    if ((KWS_DSP_ROLLING_LOG_EVERY != 0u) &&
        ((total_frames % KWS_DSP_ROLLING_LOG_EVERY) == 0u)) {
      KWS_DSP_ROLLING_LOG("[dsp-kws-rolling] prime seq=%u frame_idx=%u mfcc_cycles=%llu mfcc_status=%s avg_mfcc_cycles/frame=%llu\n",
                          (unsigned)g_commit_seq_local,
                          (unsigned)frame_idx,
                          (unsigned long long)mfcc_cycles,
                          mfcc_driver_status_str(st),
                          (unsigned long long)(total_mfcc_cycles / total_frames));
    }
  }

  KWS_DSP_ROLLING_LOG("[dsp-kws-rolling] steady-state frame_idx=%u (recomputed+sent repeatedly)\n",
                      (unsigned)KWS_DSP_ROLLING_STEADY_FRAME_IDX);

  while (1) {
    int8_t mfcc_q[KWS_MFCC_DIM];
    uint64_t mfcc_cycles = 0u;
    mfcc_driver_status_t st;

    maybe_wait_send_interval(&next_send_cycle);
    st = compute_one_mfcc_frame((uint8_t)KWS_DSP_ROLLING_STEADY_FRAME_IDX, mfcc_q, &mfcc_cycles);
    publish_one_frame(mfcc_q);

    total_frames++;
    total_mfcc_cycles += mfcc_cycles;

    if ((KWS_DSP_ROLLING_LOG_EVERY != 0u) &&
        ((total_frames % KWS_DSP_ROLLING_LOG_EVERY) == 0u)) {
      KWS_DSP_ROLLING_LOG("[dsp-kws-rolling] steady seq=%u mfcc_cycles/frame=%llu mfcc_status=%s avg_mfcc_cycles/frame=%llu fails=%u\n",
                          (unsigned)g_commit_seq_local,
                          (unsigned long long)mfcc_cycles,
                          mfcc_driver_status_str(st),
                          (unsigned long long)(total_mfcc_cycles / total_frames),
                          (unsigned)g_mfcc_fail_local);
    }
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
