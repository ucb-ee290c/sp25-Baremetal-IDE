#include "main.h"

#if KWS_DSP_ROLLING_USE_MIC
/* Live I2S mic: the audio source is g_mic_audio, filled per case by mic_capture_case(). No embedded
 * waveform header is compiled in. */
#define KWS_DSP_SIGNAL_COUNT   1u
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
#include "kws_dsp_signals.h"   /* generated: g_kws_dsp_signals[], KWS_DSP_NUM_SIGNALS */
#define KWS_DSP_SIGNAL_COUNT   KWS_DSP_NUM_SIGNALS
#else
#include "yes_test_005_signal.h"
#define KWS_DSP_SIGNAL_COUNT   1u
#endif

#include "c2c_shm.h"
#include "c2c_turnsync.h"
#include "kws_stream_proto.h"

#if KWS_DSP_ROLLING_USE_MIC
#include "rocketcore.h"
#include "hal_mmio.h"
#include "hal_i2s.h"

/* One live-audio window (raw 24-bit samples scaled to float ~[-1,1], DC removed). Reused as the
 * "signal" buffer for every case, so all the downstream MFCC code is unchanged. */
static float32_t g_mic_audio[KWS_DSP_ROLLING_MIC_NUM_SAMPLES];

/* Mic config: RX + internal clock generator on, 32-bit, DAC off. Mirrors the proven dsp-i2s-test
 * i2s_params_mic. clkdiv is a placeholder; set_I2S_sample_freq() overrides it for exactly 16 kHz at
 * the demo's operating frequency (app_init). */
static i2s_params_t g_i2s_params_mic = {
    .tx_en         = 1,
    .rx_en         = 1,
    .bitdepth_tx   = I2S_BITDEPTH_32,
    .bitdepth_rx   = I2S_BITDEPTH_32,
    .clkgen        = 1,
    .dacen         = 0,
    .ws_len        = 3,
    .clkdiv        = 8,
    .tx_fp         = 0,
    .rx_fp         = 0,
    .tx_force_left = 0,
    .rx_force_left = 0,
};

#if KWS_DSP_ROLLING_VAD_ENABLE
/* Pre-roll ring of the most-recent monitoring samples (scaled float), so the window can start
 * shortly BEFORE the detected onset. */
static float32_t g_vad_ring[KWS_DSP_ROLLING_VAD_PREROLL_SAMPLES];
_Static_assert((KWS_DSP_ROLLING_VAD_FRAME_SAMPLES % 2u) == 0u, "VAD frame must be even (I2S reads pairs).");
_Static_assert((KWS_DSP_ROLLING_VAD_PREROLL_SAMPLES % KWS_DSP_ROLLING_VAD_FRAME_SAMPLES) == 0u,
               "VAD pre-roll must be a whole number of frames.");
_Static_assert(KWS_DSP_ROLLING_VAD_PREROLL_SAMPLES < KWS_DSP_ROLLING_MIC_NUM_SAMPLES,
               "VAD pre-roll must be shorter than the capture window.");
#endif
#endif

/* One signal source per streamed case. In single mode this is the embedded yes_test_005 waveform;
 * in multi mode we index the generated table. Every case labels the payload with a ground-truth
 * class so BML can score pred-vs-expected. */
static const float *signal_samples(uint32_t s) {
#if KWS_DSP_ROLLING_USE_MIC
  (void)s;
  return g_mic_audio;
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
  return g_kws_dsp_signals[s].samples;
#else
  (void)s;
  return g_kws_dsp_yes005_signal;
#endif
}
static uint32_t signal_num_samples(uint32_t s) {
#if KWS_DSP_ROLLING_USE_MIC
  (void)s;
  return KWS_DSP_ROLLING_MIC_NUM_SAMPLES;
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
  (void)s;
  return KWS_DSP_SIGNAL_NUM_SAMPLES;
#else
  (void)s;
  return KWS_DSP_YES005_NUM_SAMPLES;
#endif
}
static int32_t signal_label(uint32_t s) {
#if KWS_DSP_ROLLING_USE_MIC
  (void)s;
  return -1; /* live audio: no ground truth */
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
  return g_kws_dsp_signals[s].expected_label;
#else
  (void)s;
  return 0; /* yes_test_005 -> class 0 (yes) */
#endif
}
static const char *signal_name(uint32_t s) {
#if KWS_DSP_ROLLING_USE_MIC
  (void)s;
  return "mic";
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
  return g_kws_dsp_signals[s].name;
#else
  (void)s;
  return KWS_DSP_YES005_MEMBER;
#endif
}
static int32_t signal_ref_index(uint32_t s) {
#if KWS_DSP_ROLLING_USE_MIC
  (void)s;
  return -1; /* live audio: no matching reference case */
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
  return g_kws_dsp_signals[s].ref_case_index;
#else
  (void)s;
  return 5; /* yes_test_005 = ./yes/0cb74144_nohash_2.wav */
#endif
}

/* Multi-signal and live-mic stream a fresh payload every case, so they must always full-publish.
 * Single embedded mode honours the static-payload optimization. */
#if KWS_DSP_ROLLING_MULTI_SIGNAL || KWS_DSP_ROLLING_USE_MIC
#define KWS_DSP_ROLLING_FULL_PUBLISH 1
#else
#define KWS_DSP_ROLLING_FULL_PUBLISH (KWS_DSP_ROLLING_STATIC_PAYLOAD == 0)
#endif

#if !KWS_DSP_ROLLING_USE_MIC
/* Signal used by case index `idx` (1-based): round-robin over the table in multi mode. (Unused in
 * mic mode — the source is always the freshly captured g_mic_audio.) */
static uint32_t signal_for_case(uint32_t idx) {
  return (idx - 1u) % (uint32_t)KWS_DSP_SIGNAL_COUNT;
}
#endif

#if KWS_DSP_ROLLING_USE_THREADLIB
#include "mfcc_driver_mc.h"
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

#if KWS_DSP_ROLLING_USE_MIC
_Static_assert(KWS_DSP_ROLLING_MIC_NUM_SAMPLES >= ((((uint32_t)KWS_DSP_ROLLING_FRAMES_PER_CASE - 1u) * KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES) + MFCC_DRIVER_FFT_LEN),
               "Mic capture window does not cover all requested MFCC frames.");
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
_Static_assert(KWS_DSP_SIGNAL_NUM_SAMPLES >= ((((uint32_t)KWS_DSP_ROLLING_FRAMES_PER_CASE - 1u) * KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES) + MFCC_DRIVER_FFT_LEN),
               "Generated signal length does not cover all requested MFCC frames.");
#else
_Static_assert(KWS_DSP_YES005_NUM_SAMPLES >= ((((uint32_t)KWS_DSP_ROLLING_FRAMES_PER_CASE - 1u) * KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES) + MFCC_DRIVER_FFT_LEN),
               "Embedded yes_test_005 signal does not cover all requested MFCC frames.");
#endif
_Static_assert((KWS_DSP_ROLLING_FRAMES_PER_CASE * KWS_MFCC_DIM) == KWS_CASE_PAYLOAD_BYTES,
               "Frames*dim must equal the case payload size.");

/* Turn-taking sync (see c2c_turnsync.h / /CLAUDE.md). DSP produces cases; roles: DSP's turn =
 * publish the next case + hand off to BML; BML's turn = read+verify+infer + hand back.
 *   - own spad (0xC): DSP local-reads acks and local-writes its OWN turn register.
 *   - peer spad (BML, reached at 0x1_D000_0000): DSP remote-writes payload + BML's turn register. */
static kws_stream_dsp_spad_t *const g_dsp =
    (kws_stream_dsp_spad_t *)(uintptr_t)KWS_STREAM_DSP_SPAD_BASE;   /* own, local reads */
static kws_stream_bml_spad_t *const g_bml =
    (kws_stream_bml_spad_t *)(uintptr_t)KWS_STREAM_BML_SPAD_PEER;   /* peer, cross-link writes */

static mfcc_driver_t g_mfcc;
static float32_t g_input_window[MFCC_DRIVER_FFT_LEN];
static int8_t g_case[KWS_CASE_PAYLOAD_BYTES];
static float32_t g_case_f32[KWS_CASE_PAYLOAD_BYTES]; /* full float MFCC map before quantization */
static uint32_t g_mfcc_fail_local;
static uint32_t g_case_index;
static uint32_t g_case_checksum;
static int32_t  g_case_expected_label;  /* ground-truth class for the case currently in g_case */
static int32_t  g_case_ref_index;       /* matching tinyspeech_inputs.h index for the case */

uint64_t target_frequency = KWS_DSP_ROLLING_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static void load_signal_window(uint32_t sig_idx, uint8_t frame_idx, float32_t *dst) {
  const float *samples = signal_samples(sig_idx);
  const uint32_t len = signal_num_samples(sig_idx);
  const uint32_t start = ((uint32_t)frame_idx) * KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES;

  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; ++n) {
    const uint32_t idx = start + n;
    dst[n] = (idx < len) ? samples[idx] : 0.0f;
  }
}

static int8_t clip_i8(int32_t qi) {
  if (qi > 127) {
    qi = 127;
  }
  if (qi < -127) {
    qi = -127;
  }
  return (int8_t)qi;
}

static int8_t quantize_mfcc_fixed(float32_t x) {
  float32_t qf = (x * (float32_t)KWS_DSP_ROLLING_MFCC_QUANT_SCALE) + (float32_t)KWS_DSP_ROLLING_MFCC_QUANT_ZERO;
  return clip_i8((int32_t)lrintf(qf));
}

/* Fill mfcc_f32_out with KWS_MFCC_DIM float coefficients for one frame (no quantization — the whole
 * case is quantized together so per-case normalization can match the reference recipe). */
static mfcc_driver_status_t compute_one_mfcc_frame(uint32_t sig_idx,
                                                    uint8_t frame_idx,
                                                    float32_t *mfcc_f32_out,
                                                    uint64_t *mfcc_cycles_out) {
  float32_t mfcc_f32[MFCC_DRIVER_NUM_DCT];
  uint64_t mfcc_cycles = 0;
  mfcc_driver_status_t st;

  load_signal_window(sig_idx, frame_idx, g_input_window);

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
    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      mfcc_f32_out[k] = 0.0f;
    }
    g_mfcc_fail_local++;
    *mfcc_cycles_out = mfcc_cycles;
    return st;
  }

  for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
    mfcc_f32_out[k] = mfcc_f32[k];
  }

  *mfcc_cycles_out = mfcc_cycles;
  return MFCC_DRIVER_OK;
}

/* Compute a full 94-frame case for signal sig_idx into g_case (coeff-major). All frames' float
 * MFCCs are buffered first so the whole case can be quantized together — this lets the default
 * per-case normalization match the reference recipe (q = clip(round(x * 127/max|x|))). */
static void compute_full_case(uint32_t sig_idx) {
  uint64_t total_mfcc_cycles = 0u;
  float32_t case_amax = 0.0f;

  g_case_expected_label = signal_label(sig_idx);
  g_case_ref_index = signal_ref_index(sig_idx);
  g_mfcc_fail_local = 0u;

  for (uint8_t frame_idx = 0; frame_idx < (uint8_t)KWS_DSP_ROLLING_FRAMES_PER_CASE; ++frame_idx) {
    float32_t mfcc_f32[KWS_MFCC_DIM];
    uint64_t mfcc_cycles = 0u;
    (void)compute_one_mfcc_frame(sig_idx, frame_idx, mfcc_f32, &mfcc_cycles);

    /* Coefficient-major layout to match the model input {1,1,H=12,W=94}: element (coeff k, frame f)
     * lives at index k*FRAMES + f. (DSP previously wrote frame-major [f*DIM + k], which the model
     * read transposed -> scrambled features. See the INPUT-CMP diagnostic.) */
    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      float32_t v = mfcc_f32[k];
      g_case_f32[(k * (uint32_t)KWS_DSP_ROLLING_FRAMES_PER_CASE) + (uint32_t)frame_idx] = v;
      float32_t a = (v < 0.0f) ? -v : v;
      if (a > case_amax) {
        case_amax = a;
      }
    }
    total_mfcc_cycles += mfcc_cycles;
  }

  /* Quantize the whole map. */
#if KWS_DSP_ROLLING_MFCC_NORMALIZE
  float32_t norm_scale = (case_amax < 1e-12f) ? 0.0f : (127.0f / case_amax);
  for (uint32_t i = 0; i < (uint32_t)KWS_CASE_PAYLOAD_BYTES; ++i) {
    g_case[i] = clip_i8((int32_t)lrintf(g_case_f32[i] * norm_scale));
  }
#else
  for (uint32_t i = 0; i < (uint32_t)KWS_CASE_PAYLOAD_BYTES; ++i) {
    g_case[i] = quantize_mfcc_fixed(g_case_f32[i]);
  }
#endif

  g_case_checksum = c2c_checksum(g_case, KWS_CASE_PAYLOAD_BYTES);

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] case computed sig=%u name=%s expected=%ld ref=%ld frames=%u amax=%d/1000 checksum=0x%08lx avg_mfcc_cycles/frame=%llu fails=%u\n",
                      (unsigned)sig_idx, signal_name(sig_idx),
                      (long)g_case_expected_label, (long)g_case_ref_index,
                      (unsigned)KWS_DSP_ROLLING_FRAMES_PER_CASE,
                      (int)lrintf(case_amax * 1000.0f),
                      (unsigned long)g_case_checksum,
                      (unsigned long long)(total_mfcc_cycles / KWS_DSP_ROLLING_FRAMES_PER_CASE),
                      (unsigned)g_mfcc_fail_local);
}

/* Identity (magic/version/payload_bytes) into BML's spad. Cross-link write -> app_main only. */
static void publish_identity(void) {
  c2c_remote_write_u32(&g_bml->magic, KWS_STREAM_MAGIC_BML);
  c2c_remote_write_u32(&g_bml->version, KWS_STREAM_PROTO_VERSION);
  c2c_remote_write_u32(&g_bml->payload_bytes, (uint32_t)KWS_CASE_PAYLOAD_BYTES);
}

/* Full publish: whole payload + checksum + tx_cycle, then case_index (written before the turn). */
static void publish_case_full(uint32_t idx) {
  uint64_t tx_cycle = rdcycle64();

  c2c_remote_write_block(g_bml->case_payload, g_case, KWS_CASE_PAYLOAD_BYTES);
  c2c_remote_write_u32(&g_bml->payload_checksum, g_case_checksum);
  c2c_remote_write_u32((volatile uint32_t *)&g_bml->expected_label, (uint32_t)g_case_expected_label);
  c2c_remote_write_u32((volatile uint32_t *)&g_bml->ref_case_index, (uint32_t)g_case_ref_index);
  c2c_remote_write_block(&g_bml->dsp_tx_cycle, &tx_cycle, sizeof(tx_cycle));
  c2c_remote_write_u32(&g_bml->case_index, idx);
}

/* Static-payload fast path: payload+checksum already resident in BML's spad; just bump case_index
 * to re-trigger inference on the same data. Turn-taking serializes access, so even a full publish
 * every round is now safe — this is purely a throughput optimization. */
static void publish_case_recommit(uint32_t idx) {
  c2c_remote_write_u32(&g_bml->case_index, idx);
}

/* Hand the turn to BML: publish the case data, then flip the turn register (peer spad = commit,
 * so data is resident before BML sees its turn; then our own spad = "not ours"), then wake BML. */
static void handoff_to_bml(uint32_t idx, int full) {
  if (full) {
    publish_case_full(idx);
  } else {
    publish_case_recommit(idx);
  }
  c2c_remote_write_u32(&g_bml->turn, C2C_TURN_BML); /* commit: BML's turn, in BML's spad */
  c2c_local_write_u32(&g_dsp->turn, C2C_TURN_BML);  /* our own spad: no longer our turn */
  c2c_wake_peer();
  g_case_index = idx;
}

/* Boot barrier: poll our LOCAL 0xC for bml_ready. No cross-link writes until we see it (a write
 * into a still-booting BML kills it). */
static void wait_for_bml_ready(void) {
  uint32_t loops = 0u;

  /* Wipe stale bml_ready from a previous run (spad SRAM survives a chip-only reset) — local write. */
  c2c_local_write_u32(&g_dsp->bml_ready, 0u);
  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] cleared stale bml_ready in 0xC before barrier\n");

  while (1) {
    uint32_t ready = c2c_local_read_u32(&g_dsp->bml_ready);
    if (ready == KWS_STREAM_READY_MAGIC) {
      KWS_DSP_ROLLING_LOG("[dsp-kws-stream] bml_ready seen after %u polls\n", (unsigned)loops);
      return;
    }
    if ((loops % 100000u) == 0u) {
      KWS_DSP_ROLLING_LOG("[dsp-kws-stream] waiting for bml_ready (0xC=0x%08lx) polls=%u\n",
                          (unsigned long)ready, (unsigned)loops);
    }
    loops++;
  }
}

#if KWS_DSP_ROLLING_USE_MIC
static inline int32_t mic_extract_sample(uint32_t slot) {
  /* 24-bit sample in the top of the 32-bit slot; sign-extend via arithmetic shift. */
  return ((int32_t)slot) >> KWS_DSP_ROLLING_MIC_SAMPLE_SHIFT;
}

/* Pop one 64-bit RX block from the mic (LEFT) = two consecutive time samples, scaled to float
 * ~[-1,1]. Mirrors the proven dsp-i2s-test read model; the mic's internal clkgen free-runs, so a
 * blind read blocks until a sample is available. */
static inline void mic_read_pair(float32_t *s0, float32_t *s1) {
  uint64_t v = read_I2S_rx(KWS_DSP_ROLLING_MIC_CHANNEL, I2S_LEFT);
  *s0 = (float32_t)mic_extract_sample((uint32_t)(v & 0xFFFFFFFFu)) * (1.0f / KWS_DSP_ROLLING_MIC_FULLSCALE);
  *s1 = (float32_t)mic_extract_sample((uint32_t)(v >> 32)) * (1.0f / KWS_DSP_ROLLING_MIC_FULLSCALE);
}

/* Capture one ~1 s window of live mic audio into g_mic_audio (scaled float, DC removed) — the same
 * shape as an embedded waveform, so all the downstream MFCC code is unchanged. With the VAD gate
 * enabled, monitors short-frame AC energy and starts capturing only once a speech onset crosses the
 * threshold, keeping a short pre-roll so the word's attack is included. */
static void mic_capture_case(void) {
  const uint32_t N = KWS_DSP_ROLLING_MIC_NUM_SAMPLES;
  uint32_t w = 0u;

#if KWS_DSP_ROLLING_VAD_ENABLE
  const uint32_t FR = KWS_DSP_ROLLING_VAD_FRAME_SAMPLES;
  const uint32_t PRE = KWS_DSP_ROLLING_VAD_PREROLL_SAMPLES;
  uint32_t ring_pos = 0u;      /* next write slot in the pre-roll ring */
  uint32_t ring_filled = 0u;   /* samples currently held (<= PRE) */
  uint32_t frame_ctr = 0u;

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] mic: listening (VAD thresh=%d/1e6; speak now)...\n",
                      (int)lrintf((float32_t)KWS_DSP_ROLLING_VAD_THRESHOLD * 1.0e6f));

  for (;;) {
    float32_t sum = 0.0f;
    float32_t sumsq = 0.0f;
    for (uint32_t i = 0; i < FR; i += 2u) {
      float32_t a, b;
      mic_read_pair(&a, &b);
      g_vad_ring[ring_pos] = a; ring_pos = (ring_pos + 1u) % PRE;
      g_vad_ring[ring_pos] = b; ring_pos = (ring_pos + 1u) % PRE;
      sum += a + b;
      sumsq += (a * a) + (b * b);
    }
    if (ring_filled < PRE) {
      ring_filled += FR;
      if (ring_filled > PRE) {
        ring_filled = PRE;
      }
    }
    float32_t mean = sum / (float32_t)FR;
    float32_t energy = (sumsq / (float32_t)FR) - (mean * mean); /* AC variance (DC removed) */
    if (energy < 0.0f) {
      energy = 0.0f;
    }

#if KWS_DSP_ROLLING_VAD_LOG_EVERY
    if ((frame_ctr % KWS_DSP_ROLLING_VAD_LOG_EVERY) == 0u) {
      KWS_DSP_ROLLING_LOG("[dsp-kws-stream] mic: vad frame=%u energy=%d/1e6\n",
                          (unsigned)frame_ctr, (int)lrintf(energy * 1.0e6f));
    }
#endif
    frame_ctr++;

    if (energy >= (float32_t)KWS_DSP_ROLLING_VAD_THRESHOLD) {
      KWS_DSP_ROLLING_LOG("[dsp-kws-stream] mic: onset frame=%u energy=%d/1e6 -> capturing\n",
                          (unsigned)frame_ctr, (int)lrintf(energy * 1.0e6f));
      break;
    }
  }

  /* Emit the pre-roll (oldest -> newest) into the window first, so the word's attack is included. */
  uint32_t start = (ring_filled < PRE) ? 0u : ring_pos;
  for (uint32_t k = 0; (k < ring_filled) && (w < N); ++k) {
    g_mic_audio[w++] = g_vad_ring[(start + k) % PRE];
  }
#endif /* KWS_DSP_ROLLING_VAD_ENABLE */

  /* Fill the remainder of the window with fresh samples. */
  while (w < N) {
    float32_t a, b;
    mic_read_pair(&a, &b);
    g_mic_audio[w++] = a;
    if (w < N) {
      g_mic_audio[w++] = b;
    }
  }

  /* Remove the (large, near-constant) DC offset and gather a stats line for tuning. */
  float32_t sum = 0.0f;
  for (uint32_t i = 0; i < N; ++i) {
    sum += g_mic_audio[i];
  }
  float32_t dc = sum / (float32_t)N;
  float32_t vmin = 1.0f, vmax = -1.0f, absmean = 0.0f;
  for (uint32_t i = 0; i < N; ++i) {
    float32_t x = g_mic_audio[i] - dc;
    g_mic_audio[i] = x;
    if (x < vmin) vmin = x;
    if (x > vmax) vmax = x;
    absmean += (x < 0.0f) ? -x : x;
  }
  absmean /= (float32_t)N;

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] mic: captured %u samples dc=%d/1e6 min=%d/1e6 max=%d/1e6 absmean=%d/1e6\n",
                      (unsigned)N, (int)lrintf(dc * 1.0e6f), (int)lrintf(vmin * 1.0e6f),
                      (int)lrintf(vmax * 1.0e6f), (int)lrintf(absmean * 1.0e6f));
}
#endif /* KWS_DSP_ROLLING_USE_MIC */

void app_init(void) {
  init_test(target_frequency);
  g_mfcc_fail_local = 0u;
  g_case_index = 0u;

  /* NOTE: app_init does NO cross-link (spad) writes. All peer writes happen in app_main. */

#if KWS_DSP_ROLLING_USE_THREADLIB
  hthread_init();
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
#endif

  if (mfcc_driver_init(&g_mfcc) != MFCC_DRIVER_OK) {
    KWS_DSP_ROLLING_LOG("[dsp-kws-stream] MFCC init failed\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

#if KWS_DSP_ROLLING_USE_MIC
  /* Configure the I2S mic (channel 0, per the validated pinout) and clock it at exactly 16 kHz for
   * the operating frequency, so the capture rate matches the MFCC front-end's assumption without
   * any software resampling. No cross-link writes here (app_init rule) — I2S is a local peripheral. */
  config_I2S(KWS_DSP_ROLLING_MIC_CHANNEL, &g_i2s_params_mic);
  set_I2S_sample_freq(KWS_DSP_ROLLING_MIC_CHANNEL, (uint64_t)target_frequency,
                      (uint64_t)KWS_DSP_ROLLING_MIC_SAMPLE_RATE_HZ, (uint8_t)KWS_DSP_ROLLING_MIC_BITDEPTH);
  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] I2S mic configured ch=%d rate=%uHz bits=%u config=0x%04x\n",
                      KWS_DSP_ROLLING_MIC_CHANNEL, (unsigned)KWS_DSP_ROLLING_MIC_SAMPLE_RATE_HZ,
                      (unsigned)KWS_DSP_ROLLING_MIC_BITDEPTH,
                      (unsigned)reg_read16(I2S_CONFIG(KWS_DSP_ROLLING_MIC_CHANNEL)));
#endif

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] init own_spad=0x%08lx peer_spad=0x%09llx payload_bytes=%u signals=%u mode=%s\n",
                      (unsigned long)KWS_STREAM_DSP_SPAD_BASE,
                      (unsigned long long)KWS_STREAM_BML_SPAD_PEER,
                      (unsigned)KWS_CASE_PAYLOAD_BYTES,
                      (unsigned)KWS_DSP_SIGNAL_COUNT,
                      KWS_DSP_ROLLING_USE_MIC ? "mic" : (KWS_DSP_ROLLING_MULTI_SIGNAL ? "multi" : "single"));
}

void app_main(void) {
#if !KWS_DSP_ROLLING_USE_MIC
  /* Embedded audio: the first case can be computed up front (pure-local MFCC, no link). For mic
   * mode we defer the first capture until AFTER the boot barrier, so listening (which blocks on the
   * user speaking) doesn't stall the link handshake. */
  compute_full_case(signal_for_case(1u));
#endif

  /* Local boot-clear of our OWN 0xC control block: turn = BML (not ours until we grant it after
   * the first publish), and clear the stale barrier flag. Local writes — safe. */
  c2c_local_write_u32(&g_dsp->turn, C2C_TURN_BML);

  /* Boot barrier: do NOT write BML's spad until BML says it has booted. */
  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] waiting for BML boot barrier before touching peer spad\n");
  wait_for_bml_ready();

  /* Arm MSIP + timer wake (turn-taking safety net) before any wfi. */
  c2c_arm_wake();

#if KWS_DSP_ROLLING_USE_MIC
  /* Now that the link is up, capture the first live utterance and build case 1 from it. */
  mic_capture_case();
  compute_full_case(0u);
#endif

  /* DSP is the initiator: publish case 1 and hand the first turn to BML (no await — nobody grants
   * DSP the first turn). Identity is folded into this first quiet window. */
  publish_identity();
  handoff_to_bml(1u, /*full=*/1);
  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] published case_index=1 -> BML; entering turn-taking loop\n");

  uint32_t n = 1u; /* the case currently granted to BML (== g_case_index) */

  while (1) {
    /* Wait until BML has acked case n. On any idle TIMER tick with no ack, RE-GRANT n: this
     * self-heals a dropped grant, a dropped return-ack, or a dropped wake in either direction.
     * Re-grants are idempotent — BML ignores a duplicate (case_index <= its last_consumed), so
     * there is never a double inference. Steady state (ack arrives via MSIP before the ~50ms tick)
     * fires no re-grant, so link traffic is unchanged. */
    for (;;) {
      if (c2c_local_read_u32(&g_dsp->ack_index) >= n) {
        break;
      }
      c2c_sleep_until_tick(); /* wake on BML's ack MSIP or the periodic timer */
      if (c2c_local_read_u32(&g_dsp->ack_index) >= n) {
        break;
      }
      KWS_DSP_ROLLING_LOG("[dsp-kws-stream] re-grant case_index=%u (ack_index=%u) [self-heal]\n",
                          (unsigned)n, (unsigned)g_dsp->ack_index);
      handoff_to_bml(n, KWS_DSP_ROLLING_FULL_PUBLISH);
    }

    /* Acked -> read result telemetry. */
    c2c_full_flush();
    KWS_DSP_ROLLING_LOG("[dsp-kws-stream] case_index=%u acked; pred=%u score_q=%u -> next\n",
                        (unsigned)n, (unsigned)g_dsp->bml_pred_class,
                        (unsigned)g_dsp->bml_pred_score_q);

#if KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES && !KWS_DSP_ROLLING_USE_MIC
    /* Fixed inter-case pacing. In mic mode the VAD gate paces the stream (it blocks until the next
     * utterance), so this dead window is skipped — we go straight back to listening. */
    {
      uint64_t q0 = rdcycle64();
      while ((rdcycle64() - q0) < (uint64_t)KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES) {
        __asm__ volatile("nop");
      }
    }
#endif

    /* Produce + publish the next case. Live mic captures a fresh utterance (VAD-gated) and always
     * full-publishes; multi-signal recomputes a fresh MFCC case for the next recording (round-robin)
     * and always full-publishes; single static payload -> recommit only (payload already resident). */
    uint32_t idx = n + 1u;
    int full = KWS_DSP_ROLLING_FULL_PUBLISH;
#if KWS_DSP_ROLLING_USE_MIC
    mic_capture_case();
    compute_full_case(0u);
#elif KWS_DSP_ROLLING_MULTI_SIGNAL
    compute_full_case(signal_for_case(idx));
#endif
    KWS_DSP_ROLLING_LOG("[dsp-kws-stream] handing case_index=%u to BML (%s)\n",
                        (unsigned)idx, full ? "full payload" : "recommit");
    handoff_to_bml(idx, full);
    n = idx;

#if KWS_DSP_ROLLING_PUBLISH_ONCE
    KWS_DSP_ROLLING_LOG("[dsp-kws-stream] PUBLISH_ONCE: done; entering wfi (link idle)\n");
    while (1) {
      __asm__ volatile("wfi");
    }
#endif
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
