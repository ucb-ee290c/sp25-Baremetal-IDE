#include "main.h"

#include "tensor.h"

#if KWS_BEARLY_USE_THREADLIB
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

_Static_assert((KWS_BEARLY_SIMPLE_PAYLOAD_ADDR >= KWS_BEARLY_SHM_BASE),
               "KWS_BEARLY_SIMPLE_PAYLOAD_ADDR must be inside shared region.");
_Static_assert(((KWS_BEARLY_SIMPLE_PAYLOAD_ADDR - KWS_BEARLY_SHM_BASE) + KWS_CASE_PAYLOAD_BYTES) <= KWS_BEARLY_SHM_BYTES,
               "KWS payload does not fit in shared region.");
_Static_assert((KWS_BEARLY_CACHE_LINE_BYTES & (KWS_BEARLY_CACHE_LINE_BYTES - 1u)) == 0u,
               "KWS_BEARLY_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((KWS_BEARLY_CACHE_EVICT_BYTES >= KWS_BEARLY_CACHE_LINE_BYTES),
               "KWS_BEARLY_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((KWS_BEARLY_CACHE_EVICT_BYTES % KWS_BEARLY_CACHE_LINE_BYTES) == 0u,
               "KWS_BEARLY_CACHE_EVICT_BYTES must be a multiple of cache line size.");

static volatile uint32_t *const g_marker =
    (volatile uint32_t *)(uintptr_t)KWS_BEARLY_SIMPLE_MARKER_ADDR;
static volatile int8_t *const g_payload =
    (volatile int8_t *)(uintptr_t)KWS_BEARLY_SIMPLE_PAYLOAD_ADDR;
static const char *g_labels[TINYSPEECH_NUM_CLASSES] = {
    "yes", "no", "on", "off", "stop", "go"
};

static uint8_t g_cache_evict[KWS_BEARLY_CACHE_EVICT_BYTES]
    __attribute__((aligned(0x8000)));
static volatile uint8_t g_cache_sink;
static uint32_t g_wait_poll_count;
static uint8_t g_int8_calibrated;

uint64_t target_frequency = KWS_BEARLY_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline void kws_fence_rw_local(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t pass = 0; pass < 3u; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)KWS_BEARLY_CACHE_EVICT_BYTES; i += KWS_BEARLY_CACHE_LINE_BYTES) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    kws_fence_rw_local();
  }

  g_cache_sink = sink;
  kws_fence_rw_local();
}

static inline void refresh_shared(void) {
  cache_evict_all();
  kws_fence_rw_local();
}

static void clear_shared_if_safe(void) {
#if KWS_BEARLY_CLEAR_SHM_ON_BOOT
  volatile uint8_t *shm = (volatile uint8_t *)(uintptr_t)KWS_BEARLY_SHM_BASE;

  refresh_shared();
  if (*g_marker == (uint32_t)KWS_BEARLY_SIMPLE_MARKER_VALUE) {
    KWS_BEARLY_LOG("[bearly-kws] skip clear: marker already set (0x%08lx)\n",
                   (unsigned long)*g_marker);
    return;
  }

  for (uint32_t i = 0; i < (uint32_t)KWS_BEARLY_SHM_BYTES; ++i) {
    shm[i] = 0u;
  }
  kws_fence_rw_local();
  cache_evict_all();
  kws_fence_rw_local();

  KWS_BEARLY_LOG("[bearly-kws] cleared shared region base=0x%08lx bytes=%u\n",
                 (unsigned long)KWS_BEARLY_SHM_BASE,
                 (unsigned)KWS_BEARLY_SHM_BYTES);
#endif
}

static void wait_for_marker(void) {
  while (1) {
    uint32_t marker;
    g_wait_poll_count++;
    refresh_shared();
    marker = *g_marker;

    if (marker == (uint32_t)KWS_BEARLY_SIMPLE_MARKER_VALUE) {
      KWS_BEARLY_LOG("[bearly-kws] marker detected: 0x%08lx after polls=%u\n",
                     (unsigned long)marker,
                     (unsigned)g_wait_poll_count);
      return;
    }

    if ((KWS_BEARLY_WAIT_LOG_EVERY != 0u) &&
        ((g_wait_poll_count % KWS_BEARLY_WAIT_LOG_EVERY) == 0u)) {
      KWS_BEARLY_LOG("[bearly-kws] polling marker=0x%08lx polls=%u payload0=%d payload1=%d payload2=%d payload3=%d\n",
                     (unsigned long)marker,
                     (unsigned)g_wait_poll_count,
                     (int)g_payload[0],
                     (int)g_payload[1],
                     (int)g_payload[2],
                     (int)g_payload[3]);
    }

    __asm__ volatile("nop");
  }
}

static void run_one_inference_from_shared(void) {
  int8_t payload[KWS_CASE_PAYLOAD_BYTES];
  uint8_t shape[4] = {1, 1, KWS_MFCC_DIM, KWS_FRAMES_PER_CASE};
  Tensor input;
  Tensor warm;
  Tensor probs;
  const tinyspeech_cycle_profile_t *profile = NULL;
  float max_prob = 0.0f;
  int32_t pred;
  uint64_t model_cycles;
  uint64_t t0;
  uint64_t t1;

  refresh_shared();
  for (uint32_t i = 0; i < KWS_CASE_PAYLOAD_BYTES; ++i) {
    payload[i] = g_payload[i];
  }

  input = create_tensor(shape, 4);
  memcpy(input.data, payload, sizeof(payload));

#if TINYSPEECH_INT8_PIPELINE
  if (!g_int8_calibrated) {
    int calib_ok;
    KWS_BEARLY_LOG("[bearly-kws] int8 calibration (single-case) begin\n");
    tinyspeech_int8_calibration_begin();
    warm = tinyspeech_run_inference(&input);
    free_tensor(&warm);
    calib_ok = tinyspeech_int8_calibration_end();
    g_int8_calibrated = 1u;
    KWS_BEARLY_LOG("[bearly-kws] int8 calibration %s\n", calib_ok ? "done" : "failed");

    /* Warm one fixed-path run so measured cycles are steady-state. */
    KWS_BEARLY_LOG("[bearly-kws] warm-up inference begin\n");
    warm = tinyspeech_run_inference(&input);
    free_tensor(&warm);
  }
#endif

  KWS_BEARLY_LOG("[bearly-kws] measured inference begin\n");
  t0 = rdcycle64();
  probs = tinyspeech_run_inference(&input);
  t1 = rdcycle64();
  pred = tinyspeech_argmax(&probs, &max_prob);
  profile = tinyspeech_last_cycle_profile();
  model_cycles = (profile != NULL) ? profile->total : (t1 - t0);

  KWS_BEARLY_LOG("[bearly-kws] one-case inference pred=%ld (%s) score=%.4f model_cycles=%llu wall_cycles=%llu payload0=%d payload1=%d payload2=%d payload3=%d\n",
                 (long)pred,
                 ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) ? g_labels[pred] : "out-of-range",
                 max_prob,
                 (unsigned long long)model_cycles,
                 (unsigned long long)(t1 - t0),
                 (int)payload[0],
                 (int)payload[1],
                 (int)payload[2],
                 (int)payload[3]);

  if (profile != NULL) {
    KWS_BEARLY_LOG("[bearly-kws] layer_cycles input_cast=%llu conv1_pool1=%llu conv2_pool2=%llu conv3_gap=%llu fc=%llu softmax=%llu\n",
                   (unsigned long long)profile->input_cast,
                   (unsigned long long)profile->conv1_pool1,
                   (unsigned long long)profile->conv2_pool2,
                   (unsigned long long)profile->conv3_gap,
                   (unsigned long long)profile->fc_logits,
                   (unsigned long long)profile->softmax);
  }

  free_tensor(&probs);
  free_tensor(&input);
}

void app_init(void) {
  init_test(target_frequency);
  g_cache_sink = 0u;
  g_wait_poll_count = 0u;
  g_int8_calibrated = 0u;

#if KWS_BEARLY_USE_THREADLIB
  hthread_init();
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
#endif

  clear_shared_if_safe();

  KWS_BEARLY_LOG("[bearly-kws] simple mode init marker=0x%08lx payload=0x%08lx bytes=%u\n",
                 (unsigned long)KWS_BEARLY_SIMPLE_MARKER_ADDR,
                 (unsigned long)KWS_BEARLY_SIMPLE_PAYLOAD_ADDR,
                 (unsigned)KWS_CASE_PAYLOAD_BYTES);

  KWS_BEARLY_LOG("[bearly-kws] preparing TinySpeech runtime...\n");
  tinyspeech_prepare_runtime();
  KWS_BEARLY_LOG("[bearly-kws] TinySpeech runtime ready\n");
}

void app_main(void) {
  wait_for_marker();
  run_one_inference_from_shared();
  KWS_BEARLY_LOG("[bearly-kws] done; entering wfi loop\n");

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
