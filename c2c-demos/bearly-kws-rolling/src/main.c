#include "main.h"

#include <limits.h>

#include "tensor.h"

#if KWS_BEARLY_ROLLING_USE_THREADLIB
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

_Static_assert((KWS_BEARLY_ROLLING_FRAME_ADDR >= KWS_BEARLY_ROLLING_SHM_BASE),
               "KWS_BEARLY_ROLLING_FRAME_ADDR must be inside shared region.");
_Static_assert(
    ((KWS_BEARLY_ROLLING_FRAME_ADDR - KWS_BEARLY_ROLLING_SHM_BASE) +
     KWS_ROLLING_FRAME_BYTES) <= KWS_BEARLY_ROLLING_SHM_BYTES,
    "Rolling frame payload does not fit in shared region.");
_Static_assert((KWS_BEARLY_ROLLING_CACHE_LINE_BYTES &
                (KWS_BEARLY_ROLLING_CACHE_LINE_BYTES - 1u)) == 0u,
               "KWS_BEARLY_ROLLING_CACHE_LINE_BYTES must be a power of two.");
_Static_assert(
    (KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES >= KWS_BEARLY_ROLLING_CACHE_LINE_BYTES),
    "KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES %
                KWS_BEARLY_ROLLING_CACHE_LINE_BYTES) == 0u,
               "KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES must be a multiple of cache "
               "line size.");
_Static_assert((KWS_BEARLY_ROLLING_TCM_WINDOW_OFFSET + KWS_ROLLING_WINDOW_BYTES) <=
                   KWS_BEARLY_ROLLING_TCM_BYTES,
               "KWS rolling TCM window exceeds configured TCM bytes.");

static volatile uint32_t *const g_commit_seq =
    (volatile uint32_t *)(uintptr_t)KWS_BEARLY_ROLLING_COMMIT_SEQ_ADDR;
static volatile int8_t *const g_frame =
    (volatile int8_t *)(uintptr_t)KWS_BEARLY_ROLLING_FRAME_ADDR;
static volatile int8_t *const g_window_tcm =
    (volatile int8_t *)(uintptr_t)KWS_BEARLY_ROLLING_TCM_WINDOW_ADDR;

static const char *g_labels[TINYSPEECH_NUM_CLASSES] = {
    "yes", "no", "on", "off", "stop", "go"
};

static uint8_t g_cache_evict[KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES]
    __attribute__((aligned(0x8000)));
static volatile uint8_t g_cache_sink;

static uint32_t g_last_seq;
static uint32_t g_window_frames;
static uint8_t g_int8_calibrated;

static uint64_t g_poll_count;
static uint64_t g_recv_count;
static uint64_t g_drop_count;
static uint64_t g_infer_count;
static uint64_t g_infer_sum;
static uint64_t g_infer_best;

uint64_t target_frequency = KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ;

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
    for (uint32_t i = 0; i < (uint32_t)KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES;
         i += KWS_BEARLY_ROLLING_CACHE_LINE_BYTES) {
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
#if KWS_BEARLY_ROLLING_CLEAR_SHM_ON_BOOT
  volatile uint8_t *shm = (volatile uint8_t *)(uintptr_t)KWS_BEARLY_ROLLING_SHM_BASE;

  refresh_shared();
  if (*g_commit_seq != 0u) {
    KWS_BEARLY_ROLLING_LOG(
        "[bearly-kws-rolling] skip clear: seq already non-zero (%u)\n",
        (unsigned)*g_commit_seq);
    return;
  }

  for (uint32_t i = 0; i < (uint32_t)KWS_BEARLY_ROLLING_SHM_BYTES; ++i) {
    shm[i] = 0u;
  }
  kws_fence_rw_local();
  cache_evict_all();
  kws_fence_rw_local();

  KWS_BEARLY_ROLLING_LOG(
      "[bearly-kws-rolling] cleared shared region base=0x%08lx bytes=%u\n",
      (unsigned long)KWS_BEARLY_ROLLING_SHM_BASE,
      (unsigned)KWS_BEARLY_ROLLING_SHM_BYTES);
#endif
}

static void clear_tcm_window(void) {
  for (uint32_t i = 0; i < KWS_ROLLING_WINDOW_BYTES; ++i) {
    g_window_tcm[i] = 0;
  }
  kws_fence_rw_local();
}

static void append_frame_to_tcm(const int8_t *frame) {
  if (g_window_frames < KWS_FRAMES_PER_CASE) {
    uint32_t base = g_window_frames * KWS_MFCC_DIM;
    for (uint32_t i = 0; i < KWS_MFCC_DIM; ++i) {
      g_window_tcm[base + i] = frame[i];
    }
    g_window_frames++;
    kws_fence_rw_local();
    return;
  }

  for (uint32_t i = 0; i < (KWS_ROLLING_WINDOW_BYTES - KWS_MFCC_DIM); ++i) {
    g_window_tcm[i] = g_window_tcm[i + KWS_MFCC_DIM];
  }
  for (uint32_t i = 0; i < KWS_MFCC_DIM; ++i) {
    g_window_tcm[(KWS_ROLLING_WINDOW_BYTES - KWS_MFCC_DIM) + i] = frame[i];
  }
  kws_fence_rw_local();
}

static void copy_window_to_tensor(Tensor *input) {
  for (uint32_t i = 0; i < KWS_ROLLING_WINDOW_BYTES; ++i) {
    input->data[i] = g_window_tcm[i];
  }
}

static void poll_next_frame(int8_t *frame_out,
                            uint32_t *seq_out,
                            uint64_t *rx_cycle_out,
                            uint32_t *polls_out) {
  uint32_t local_polls = 0u;

  while (1) {
    uint32_t seq0;
    uint32_t seq1;

    local_polls++;
    g_poll_count++;
    refresh_shared();
    seq0 = *g_commit_seq;

    if (seq0 == g_last_seq) {
      if ((KWS_BEARLY_ROLLING_WAIT_LOG_EVERY != 0u) &&
          ((g_poll_count % KWS_BEARLY_ROLLING_WAIT_LOG_EVERY) == 0u)) {
        KWS_BEARLY_ROLLING_LOG(
            "[bearly-kws-rolling] polling seq=%u polls=%llu frame0=%d frame1=%d "
            "frame2=%d frame3=%d\n",
            (unsigned)seq0,
            (unsigned long long)g_poll_count,
            (int)g_frame[0],
            (int)g_frame[1],
            (int)g_frame[2],
            (int)g_frame[3]);
      }
      __asm__ volatile("nop");
      continue;
    }

    for (uint32_t i = 0; i < KWS_ROLLING_FRAME_BYTES; ++i) {
      frame_out[i] = g_frame[i];
    }
    kws_fence_rw_local();

    refresh_shared();
    seq1 = *g_commit_seq;
    if (seq1 != seq0) {
      continue;
    }

    *seq_out = seq1;
    *rx_cycle_out = rdcycle64();
    *polls_out = local_polls;
    return;
  }
}

static void run_inference_from_tcm(uint32_t seq, uint64_t rx_cycle, uint32_t polls) {
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

  input = create_tensor(shape, 4);
  copy_window_to_tensor(&input);

#if TINYSPEECH_INT8_PIPELINE
  if (!g_int8_calibrated) {
    int calib_ok;
    KWS_BEARLY_ROLLING_LOG(
        "[bearly-kws-rolling] int8 calibration (rolling) begin\n");
    tinyspeech_int8_calibration_begin();
    warm = tinyspeech_run_inference(&input);
    free_tensor(&warm);
    calib_ok = tinyspeech_int8_calibration_end();
    g_int8_calibrated = 1u;
    KWS_BEARLY_ROLLING_LOG(
        "[bearly-kws-rolling] int8 calibration %s\n",
        calib_ok ? "done" : "failed");

    KWS_BEARLY_ROLLING_LOG("[bearly-kws-rolling] warm-up inference begin\n");
    warm = tinyspeech_run_inference(&input);
    free_tensor(&warm);
  }
#endif

  t0 = rdcycle64();
  probs = tinyspeech_run_inference(&input);
  t1 = rdcycle64();
  pred = tinyspeech_argmax(&probs, &max_prob);
  profile = tinyspeech_last_cycle_profile();
  model_cycles = (profile != NULL) ? profile->total : (t1 - t0);

  g_infer_count++;
  g_infer_sum += model_cycles;
  if (model_cycles < g_infer_best) {
    g_infer_best = model_cycles;
  }

  if ((KWS_BEARLY_ROLLING_INFER_LOG_EVERY != 0u) &&
      ((g_infer_count % KWS_BEARLY_ROLLING_INFER_LOG_EVERY) == 0u)) {
    KWS_BEARLY_ROLLING_LOG(
        "[bearly-kws-rolling] infer seq=%u pred=%ld (%s) score=%.4f "
        "model_cycles=%llu wall_cycles=%llu best=%llu avg=%llu recv=%llu "
        "drops=%llu rx_cycle=%llu polls=%u\n",
        (unsigned)seq,
        (long)pred,
        ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) ? g_labels[pred]
                                                          : "out-of-range",
        max_prob,
        (unsigned long long)model_cycles,
        (unsigned long long)(t1 - t0),
        (unsigned long long)g_infer_best,
        (unsigned long long)(g_infer_sum / g_infer_count),
        (unsigned long long)g_recv_count,
        (unsigned long long)g_drop_count,
        (unsigned long long)rx_cycle,
        (unsigned)polls);
  }

#if KWS_BEARLY_ROLLING_PRINT_LAYER_CYCLES
  if ((profile != NULL) && (KWS_BEARLY_ROLLING_INFER_LOG_EVERY != 0u) &&
      ((g_infer_count % KWS_BEARLY_ROLLING_INFER_LOG_EVERY) == 0u)) {
    KWS_BEARLY_ROLLING_LOG(
        "[bearly-kws-rolling] layer_cycles input_cast=%llu conv1_pool1=%llu "
        "conv2_pool2=%llu conv3_gap=%llu fc=%llu softmax=%llu\n",
        (unsigned long long)profile->input_cast,
        (unsigned long long)profile->conv1_pool1,
        (unsigned long long)profile->conv2_pool2,
        (unsigned long long)profile->conv3_gap,
        (unsigned long long)profile->fc_logits,
        (unsigned long long)profile->softmax);
  }
#endif

  free_tensor(&probs);
  free_tensor(&input);
}

void app_init(void) {
  init_test(target_frequency);

  g_cache_sink = 0u;
  g_last_seq = 0u;
  g_window_frames = 0u;
  g_int8_calibrated = 0u;
  g_poll_count = 0u;
  g_recv_count = 0u;
  g_drop_count = 0u;
  g_infer_count = 0u;
  g_infer_sum = 0u;
  g_infer_best = UINT64_MAX;

#if KWS_BEARLY_ROLLING_USE_THREADLIB
  hthread_init();
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
#endif

  clear_shared_if_safe();
  refresh_shared();
  g_last_seq = *g_commit_seq;
  clear_tcm_window();

  KWS_BEARLY_ROLLING_LOG(
      "[bearly-kws-rolling] init seq_addr=0x%08lx frame_addr=0x%08lx "
      "frame_bytes=%u baseline_seq=%u window_tcm=0x%08lx window_bytes=%u\n",
      (unsigned long)KWS_BEARLY_ROLLING_COMMIT_SEQ_ADDR,
      (unsigned long)KWS_BEARLY_ROLLING_FRAME_ADDR,
      (unsigned)KWS_ROLLING_FRAME_BYTES,
      (unsigned)g_last_seq,
      (unsigned long)KWS_BEARLY_ROLLING_TCM_WINDOW_ADDR,
      (unsigned)KWS_ROLLING_WINDOW_BYTES);

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-rolling] preparing TinySpeech runtime...\n");
  tinyspeech_prepare_runtime();
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-rolling] TinySpeech runtime ready\n");
}

void app_main(void) {
  while (1) {
    int8_t frame[KWS_MFCC_DIM];
    uint32_t seq;
    uint64_t rx_cycle;
    uint32_t polls;

    poll_next_frame(frame, &seq, &rx_cycle, &polls);

    if (seq > (g_last_seq + 1u)) {
      g_drop_count += (uint64_t)(seq - g_last_seq - 1u);
    }
    g_last_seq = seq;
    g_recv_count++;

    append_frame_to_tcm(frame);

    if ((KWS_BEARLY_ROLLING_RX_LOG_EVERY != 0u) &&
        ((g_recv_count % KWS_BEARLY_ROLLING_RX_LOG_EVERY) == 0u)) {
      KWS_BEARLY_ROLLING_LOG(
          "[bearly-kws-rolling] recv seq=%u frame0=%d frame1=%d frame2=%d "
          "frame3=%d recv=%llu drops=%llu polls=%u\n",
          (unsigned)seq,
          (int)frame[0],
          (int)frame[1],
          (int)frame[2],
          (int)frame[3],
          (unsigned long long)g_recv_count,
          (unsigned long long)g_drop_count,
          (unsigned)polls);
    }

    if (g_window_frames == KWS_FRAMES_PER_CASE) {
      if (g_recv_count == KWS_FRAMES_PER_CASE) {
        KWS_BEARLY_ROLLING_LOG(
            "[bearly-kws-rolling] rolling window primed with %u frames\n",
            (unsigned)KWS_FRAMES_PER_CASE);
      }
      run_inference_from_tcm(seq, rx_cycle, polls);
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
