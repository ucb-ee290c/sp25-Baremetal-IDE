#include "main.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if KWS_DSP_USE_THREADLIB
#include "mfcc_driver_mc.h"
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

_Static_assert((KWS_DSP_SIMPLE_PAYLOAD_ADDR >= KWS_DSP_SHARED_BASE),
               "KWS_DSP_SIMPLE_PAYLOAD_ADDR must be inside shared region.");
_Static_assert(((KWS_DSP_SIMPLE_PAYLOAD_ADDR - KWS_DSP_SHARED_BASE) + KWS_CASE_PAYLOAD_BYTES) <= KWS_DSP_SHARED_BYTES,
               "KWS payload does not fit in shared region.");
_Static_assert((KWS_DSP_CACHE_LINE_BYTES & (KWS_DSP_CACHE_LINE_BYTES - 1u)) == 0u,
               "KWS_DSP_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((KWS_DSP_CACHE_EVICT_BYTES >= KWS_DSP_CACHE_LINE_BYTES),
               "KWS_DSP_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((KWS_DSP_CACHE_EVICT_BYTES % KWS_DSP_CACHE_LINE_BYTES) == 0u,
               "KWS_DSP_CACHE_EVICT_BYTES must be a multiple of cache line size.");

static mfcc_driver_t g_mfcc;
static float32_t g_input_window[MFCC_DRIVER_FFT_LEN];
static uint8_t g_cache_evict[KWS_DSP_CACHE_EVICT_BYTES]
    __attribute__((aligned(KWS_DSP_CACHE_LINE_BYTES)));
static volatile uint8_t g_cache_sink;
static uint32_t g_mfcc_fail_local;

static volatile uint32_t *const g_marker =
    (volatile uint32_t *)(uintptr_t)KWS_DSP_SIMPLE_MARKER_ADDR;
static volatile int8_t *const g_payload =
    (volatile int8_t *)(uintptr_t)KWS_DSP_SIMPLE_PAYLOAD_ADDR;

uint64_t target_frequency = KWS_DSP_TARGET_FREQUENCY_HZ;

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

  for (uint32_t i = 0; i < (uint32_t)KWS_DSP_CACHE_EVICT_BYTES; i += KWS_DSP_CACHE_LINE_BYTES) {
    sink ^= buf[i];
    buf[i] = (uint8_t)(sink + (uint8_t)i);
  }

  g_cache_sink = sink;
  kws_fence_rw_local();
}

static float32_t clampf_local(float32_t x, float32_t lo, float32_t hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/*
 * Generate a deterministic "yes-like" utterance frame:
 * - early quiet region
 * - voiced section with low harmonics/formants ("ye")
 * - late high-frequency fricative tail ("s")
 */
static void build_yes_like_window(uint8_t frame_idx, float32_t *dst) {
  const float32_t kPi = (float32_t)M_PI;
  const float32_t p = (float32_t)frame_idx / (float32_t)(KWS_DSP_FRAMES_PER_CASE - 1u);
  const float32_t voiced_center = 0.48f;
  const float32_t voiced_width = 0.23f;
  float32_t voiced_env = expf(-((p - voiced_center) * (p - voiced_center)) / (2.0f * voiced_width * voiced_width));
  float32_t fric_env = 0.0f;
  uint32_t lcg = 0x9E3779B9u ^ ((uint32_t)frame_idx * 0x85EBCA6Bu);

  if (p > 0.62f) {
    const float32_t u = (p - 0.62f) / 0.38f;
    fric_env = clampf_local(u, 0.0f, 1.0f);
  }

  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; ++n) {
    const float32_t t = (float32_t)n / MFCC_DRIVER_SAMPLE_RATE_HZ;
    const float32_t f0 = 165.0f + (25.0f * sinf(2.0f * kPi * p));
    const float32_t voiced =
        0.85f * sinf(2.0f * kPi * f0 * t) +
        0.35f * sinf(2.0f * kPi * (2.0f * f0) * t) +
        0.22f * sinf(2.0f * kPi * 520.0f * t) +
        0.17f * sinf(2.0f * kPi * 2100.0f * t);
    float32_t fric = 0.0f;

    lcg = (1664525u * lcg) + 1013904223u;
    {
      const float32_t noise = ((float32_t)(lcg & 0x00FFFFFFu) / 8388607.5f) - 1.0f;
      fric = noise * sinf(2.0f * kPi * 3600.0f * t);
    }

    dst[n] = clampf_local((0.70f * voiced_env * voiced) + (0.40f * fric_env * fric), -1.0f, 1.0f);
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

static mfcc_driver_status_t run_one_mfcc(uint16_t case_id,
                                         uint8_t frame_idx,
                                         int8_t *mfcc_q_out,
                                         uint64_t *mfcc_cycles_sum_io) {
  float32_t mfcc_f32[MFCC_DRIVER_NUM_DCT];
  uint64_t mfcc_cycles = 0;
  mfcc_driver_status_t st;

  (void)case_id;
  build_yes_like_window(frame_idx, g_input_window);

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

static void build_case_payload(uint16_t case_id, int8_t *payload_out, uint64_t *total_mfcc_cycles) {
  for (uint8_t frame_idx = 0; frame_idx < (uint8_t)KWS_DSP_FRAMES_PER_CASE; ++frame_idx) {
    int8_t mfcc_q[KWS_MFCC_DIM];
    const uint32_t base = ((uint32_t)frame_idx) * KWS_MFCC_DIM;

    (void)run_one_mfcc(case_id, frame_idx, mfcc_q, total_mfcc_cycles);
    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      payload_out[base + k] = mfcc_q[k];
    }
  }
}

static void publish_case_payload(const int8_t *payload, uint16_t case_id) {
  *g_marker = 0u;
  kws_fence_rw_local();
  cache_writeback_pressure();

  for (uint32_t i = 0; i < KWS_CASE_PAYLOAD_BYTES; ++i) {
    g_payload[i] = payload[i];
  }

  kws_fence_rw_local();
  cache_writeback_pressure();
  *g_marker = (uint32_t)KWS_DSP_SIMPLE_MARKER_VALUE;
  kws_fence_rw_local();
  cache_writeback_pressure();
  kws_fence_rw_local();

  KWS_DSP_LOG("[dsp-kws] published case=%u marker=0x%08lx marker_addr=0x%08lx payload_addr=0x%08lx bytes=%u\n",
              (unsigned)case_id,
              (unsigned long)KWS_DSP_SIMPLE_MARKER_VALUE,
              (unsigned long)KWS_DSP_SIMPLE_MARKER_ADDR,
              (unsigned long)KWS_DSP_SIMPLE_PAYLOAD_ADDR,
              (unsigned)KWS_CASE_PAYLOAD_BYTES);
}

void app_init(void) {
  init_test(target_frequency);
  g_cache_sink = 0u;
  g_mfcc_fail_local = 0u;

#if KWS_DSP_USE_THREADLIB
  hthread_init();
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
#endif

  if (mfcc_driver_init(&g_mfcc) != MFCC_DRIVER_OK) {
    KWS_DSP_LOG("[dsp-kws] MFCC init failed\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

  /* Clear marker so reader does not start early on stale value. */
  *g_marker = 0u;
  kws_fence_rw_local();
  cache_writeback_pressure();

  KWS_DSP_LOG("[dsp-kws] simple mode init marker=0x%08lx payload=0x%08lx bytes=%u (signal=yes-like synthetic)\n",
              (unsigned long)KWS_DSP_SIMPLE_MARKER_ADDR,
              (unsigned long)KWS_DSP_SIMPLE_PAYLOAD_ADDR,
              (unsigned)KWS_CASE_PAYLOAD_BYTES);
}

void app_main(void) {
  int8_t payload[KWS_CASE_PAYLOAD_BYTES];
  uint64_t total_mfcc_cycles = 0u;
  uint64_t t0 = rdcycle64();

  memset(payload, 0, sizeof(payload));
  build_case_payload(0u, payload, &total_mfcc_cycles);
  publish_case_payload(payload, 0u);

  {
    uint64_t t1 = rdcycle64();
    KWS_DSP_LOG("[dsp-kws] one-case complete mfcc_fail=%u mfcc_cycles=%llu total_cycles=%llu\n",
                (unsigned)g_mfcc_fail_local,
                (unsigned long long)total_mfcc_cycles,
                (unsigned long long)(t1 - t0));
  }

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
