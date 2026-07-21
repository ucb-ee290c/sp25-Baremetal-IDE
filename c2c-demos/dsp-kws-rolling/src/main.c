#include "main.h"
#include "yes_test_005_signal.h"

#include "c2c_shm.h"
#include "c2c_turnsync.h"
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
static uint32_t g_mfcc_fail_local;
static uint32_t g_case_index;
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

    /* Coefficient-major layout to match the model input {1,1,H=12,W=94}: element (coeff k, frame f)
     * lives at index k*FRAMES + f. (DSP previously wrote frame-major [f*DIM + k], which the model
     * read transposed -> scrambled features. See the INPUT-CMP diagnostic.) */
    for (uint32_t k = 0; k < KWS_MFCC_DIM; ++k) {
      g_case[(k * (uint32_t)KWS_DSP_ROLLING_FRAMES_PER_CASE) + (uint32_t)frame_idx] = mfcc_q[k];
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

  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] init own_spad=0x%08lx peer_spad=0x%09llx payload_bytes=%u signal=%s\n",
                      (unsigned long)KWS_STREAM_DSP_SPAD_BASE,
                      (unsigned long long)KWS_STREAM_BML_SPAD_PEER,
                      (unsigned)KWS_CASE_PAYLOAD_BYTES,
                      KWS_DSP_YES005_MEMBER);
}

void app_main(void) {
  compute_full_case();

  /* Local boot-clear of our OWN 0xC control block: turn = BML (not ours until we grant it after
   * the first publish), and clear the stale barrier flag. Local writes — safe. */
  c2c_local_write_u32(&g_dsp->turn, C2C_TURN_BML);

  /* Boot barrier: do NOT write BML's spad until BML says it has booted. */
  KWS_DSP_ROLLING_LOG("[dsp-kws-stream] waiting for BML boot barrier before touching peer spad\n");
  wait_for_bml_ready();

  /* Arm MSIP + timer wake (turn-taking safety net) before any wfi. */
  c2c_arm_wake();

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
      handoff_to_bml(n, (KWS_DSP_ROLLING_STATIC_PAYLOAD == 0));
    }

    /* Acked -> read result telemetry. */
    c2c_full_flush();
    KWS_DSP_ROLLING_LOG("[dsp-kws-stream] case_index=%u acked; pred=%u score_q=%u -> next\n",
                        (unsigned)n, (unsigned)g_dsp->bml_pred_class,
                        (unsigned)g_dsp->bml_pred_score_q);

#if KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES
    {
      uint64_t q0 = rdcycle64();
      while ((rdcycle64() - q0) < (uint64_t)KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES) {
        __asm__ volatile("nop");
      }
    }
#endif

    /* Produce + publish the next case. Static payload -> recommit only (payload already resident);
     * otherwise a full re-publish (e.g. once VAD makes each case distinct). */
    uint32_t idx = n + 1u;
    int full = (KWS_DSP_ROLLING_STATIC_PAYLOAD == 0);
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
