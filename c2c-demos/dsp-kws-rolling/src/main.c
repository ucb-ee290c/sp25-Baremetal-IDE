#include "main.h"
#include "yes_test_005_signal.h"

#include "c2c_shm.h"
#include "kws_stream_proto.h"

#if KWS_DSP_ROLLING_USE_THREADLIB
#include "mfcc_driver_mc.h"
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

_Static_assert(KWS_DSP_YES005_NUM_SAMPLES >= ((((uint32_t)KWS_DSP_ROLLING_FRAMES_PER_CASE - 1u) * KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES) + MFCC_DRIVER_FFT_LEN),
               "Embedded yes_test_005 signal does not cover all requested MFCC frames.");
_Static_assert((KWS_DSP_ROLLING_FRAMES_PER_CASE * KWS_MFCC_DIM) == KWS_CASE_PAYLOAD_BYTES,
               "Frames*dim must equal the case payload size.");

/* BML-adjacent spad (0xD): DSP remote-writes here.  DSP-adjacent spad (0xC): DSP local-reads. */
static kws_stream_bml_spad_t *const g_bml =
    (kws_stream_bml_spad_t *)(uintptr_t)KWS_STREAM_BML_SPAD_BASE;
static kws_stream_dsp_spad_t *const g_dsp =
    (kws_stream_dsp_spad_t *)(uintptr_t)KWS_STREAM_DSP_SPAD_BASE;

static mfcc_driver_t g_mfcc;
static float32_t g_input_window[MFCC_DRIVER_FFT_LEN];
static int8_t g_case[KWS_CASE_PAYLOAD_BYTES];
static uint32_t g_mfcc_fail_local;
static uint32_t g_case_index;
static uint32_t g_epoch;
static uint32_t g_case_checksum;

uint64_t target_frequency = KWS_DSP_ROLLING_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
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

/* Compute a full 94-frame case into g_case (frame-major: g_case[frame*DIM + k]). */
static void compute_full_case(void) {
  uint64_t total_mfcc_cycles = 0u;

  for (uint8_t frame_idx = 0; frame_idx < (uint8_t)KWS_DSP_ROLLING_FRAMES_PER_CASE; ++frame_idx) {
    int8_t mfcc_q[KWS_MFCC_DIM];
    uint64_t mfcc_cycles = 0u;
    (void)compute_one_mfcc_frame(frame_idx, mfcc_q, &mfcc_cycles);

    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      g_case[((uint32_t)frame_idx * KWS_MFCC_DIM) + k] = mfcc_q[k];
    }
    total_mfcc_cycles += mfcc_cycles;
  }

  g_case_checksum = c2c_checksum(g_case, KWS_CASE_PAYLOAD_BYTES);

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] case computed frames=%u checksum=0x%08lx avg_mfcc_cycles/frame=%llu fails=%u\n",
                      (unsigned)KWS_DSP_ROLLING_FRAMES_PER_CASE,
                      (unsigned long)g_case_checksum,
                      (unsigned long long)(total_mfcc_cycles / KWS_DSP_ROLLING_FRAMES_PER_CASE),
                      (unsigned)g_mfcc_fail_local);
}

/* First cross-link write: identity + epoch into BML's spad. Called from app_main (after boot),
 * NEVER from app_init — a cross-link write to an absent/wedged peer hangs the core. */
static void publish_identity(void) {
  c2c_remote_write_u32(&g_bml->magic, KWS_STREAM_MAGIC_BML);
  c2c_remote_write_u32(&g_bml->version, KWS_STREAM_PROTO_VERSION);
  c2c_remote_write_u32(&g_bml->payload_bytes, (uint32_t)KWS_CASE_PAYLOAD_BYTES);
  c2c_remote_write_u32(&g_bml->case_index, 0u);
  c2c_remote_write_u32(&g_bml->epoch, g_epoch); /* epoch announces the producer */
}

/* Full remote-commit: write the whole payload + checksum, then case_index (the commit) LAST. */
static void publish_case_full(uint32_t idx) {
  uint64_t tx_cycle = rdcycle64();

  c2c_remote_write_block(g_bml->case_payload, g_case, KWS_CASE_PAYLOAD_BYTES);
  c2c_remote_write_u32(&g_bml->payload_checksum, g_case_checksum);
  c2c_remote_write_block(&g_bml->dsp_tx_cycle, &tx_cycle, sizeof(tx_cycle));
  c2c_remote_write_u32(&g_bml->case_index, idx); /* commit */
}

/* Static-payload fast path: the payload + checksum are already in BML's spad from the first full
 * publish and never change, so only bump the commit word to make BML re-infer. This is ~1 word vs
 * ~4500 for a full publish — the difference between a link-quiet steady state and a flood that
 * eventually collides with BML's polling and wedges the link. */
static void publish_case_recommit(uint32_t idx) {
  c2c_remote_write_u32(&g_bml->case_index, idx); /* commit only */
}

/* Block (polling our LOCAL 0xC only — no cross-link writes) until BML announces it has booted.
 * We must not write BML's spad before this, or the write lands during BML's boot and kills it. */
static void wait_for_bml_ready(void) {
  uint32_t loops = 0u;

  /* Wipe any stale bml_ready left in OUR OWN 0xC spad from a previous run. Scratchpad SRAM
   * survives a chip-only reset, so leftover magic would satisfy the barrier at 0 polls and make
   * us write 0xD while BML is still booting — which kills BML (see boot-kill quirk in /CLAUDE.md).
   * This is a LOCAL write into our own spad (safe); only a genuinely fresh BML boot can now set it. */
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

/* Turn-taking gate. Block (polling our LOCAL 0xC only — safe, no link writes) until BML has ARMED:
 * rx_ready == KWS_STREAM_RX_READY_MAGIC means BML is about to PARK (go quiet on 0xD), so it is now
 * safe for us to write 0xD. We local-clear rx_ready as we consume the arm, so the next arm (BML
 * re-setting the magic) is distinguishable. Returns rx_seq = the case number BML wants. */
static uint32_t wait_for_rx_arm(void) {
  uint32_t loops = 0u;

  while (1) {
    c2c_full_flush();
    uint32_t ready = g_dsp->rx_ready;
    uint32_t seq = g_dsp->rx_seq;
    if (ready == KWS_STREAM_RX_READY_MAGIC) {
      /* Consume this arm locally so we don't re-fire on it; BML re-sets the magic to arm again. */
      c2c_local_write_u32(&g_dsp->rx_ready, 0u);
      return seq;
    }
    if ((KWS_DSP_ROLLING_RX_WAIT_LOG_EVERY != 0u) &&
        ((loops % KWS_DSP_ROLLING_RX_WAIT_LOG_EVERY) == 0u)) {
      KWS_DSP_ROLLING_LOG("[dsp-kws-stream] waiting for rx arm (0xC.rx_ready=0x%08lx seq=%u) polls=%u\n",
                          (unsigned long)ready, (unsigned)seq, (unsigned)loops);
    }
    loops++;
  }
}

void app_init(void) {
  init_test(target_frequency);
  g_mfcc_fail_local = 0u;
  g_case_index = 0u;

  /* Per-boot epoch nonce (nonzero) so BML rebaselines and ignores any prior-run state. */
  g_epoch = (uint32_t)rdcycle64();
  if (g_epoch == 0u) {
    g_epoch = 1u;
  }

  /* NOTE: app_init does NO cross-link (spad) writes. Touching the peer's spad before the peer is
   * up hangs the core. All 0xD writes happen in app_main, after we have fully booted. */

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

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] init bml_spad=0x%08lx dsp_spad=0x%08lx epoch=0x%08lx payload_bytes=%u signal=%s\n",
                      (unsigned long)KWS_STREAM_BML_SPAD_BASE,
                      (unsigned long)KWS_STREAM_DSP_SPAD_BASE,
                      (unsigned long)g_epoch,
                      (unsigned)KWS_CASE_PAYLOAD_BYTES,
                      KWS_DSP_YES005_MEMBER);
}

void app_main(void) {
  /* Milestone 3+: compute one case, then stream it with a turn-taking handshake. Each 0xD write
   * burst happens only while BML is PARKED (it armed rx_ready, then went quiet), so the payload
   * never races BML's polling. BML drives the cadence by arming for the next case it wants. */
  compute_full_case();

  /* Boot barrier: do NOT write BML's spad until BML says it has booted. */
  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] waiting for BML boot barrier before touching 0xD\n");
  wait_for_bml_ready();

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] streaming epoch=0x%08lx (turn-taking; waiting for first rx arm)\n",
                      (unsigned long)g_epoch);

  uint32_t last_seq = 0u;   /* the rx_seq we most recently served */
  int identity_sent = 0;

  while (1) {
    /* Wait for BML to arm + park. Only now is it safe to write 0xD. rx_seq = the case BML wants. */
    uint32_t seq = wait_for_rx_arm();
    if (seq == 0u) {
      seq = 1u; /* defensive: treat an armed-but-zero seq as "wants case 1" */
    }

    /* Identity (epoch/magic into 0xD) is itself a 0xD write, so it must also land inside a park.
     * Do it on the first arm, folded into the same quiet window as case 1. */
    if (!identity_sent) {
      publish_identity();
      identity_sent = 1;
    }

    /* Full payload for case 1, when the payload changed (non-static), or when BML re-requested the
     * same case (seq == last_seq -> its previous read was torn/late, so resend the whole thing).
     * Otherwise a one-word recommit is enough (payload already resident in BML's spad). */
    int full = (KWS_DSP_ROLLING_STATIC_PAYLOAD == 0) || (seq == 1u) || (seq == last_seq);
    KWS_DSP_ROLLING_LOG("[dsp-kws-stream] rx arm seq=%u -> publish %s into 0xD (park window)\n",
                        (unsigned)seq, full ? "full payload" : "recommit only");
    if (full) {
      publish_case_full(seq);
    } else {
      publish_case_recommit(seq);
    }
    g_case_index = seq;
    last_seq = seq;
    KWS_DSP_ROLLING_LOG("[dsp-kws-stream] published case_index=%u; awaiting next rx arm\n",
                        (unsigned)seq);

#if KWS_DSP_ROLLING_PUBLISH_ONCE
    /* Diagnostic: after the first published case, stop touching the link entirely. */
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
