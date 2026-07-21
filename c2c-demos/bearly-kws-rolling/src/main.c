#include "main.h"

#include <limits.h>

#include "tensor.h"
#include "tinyspeech_int8.h" /* tinyspeech_int8_calib_get_max/set_max */

#include "c2c_shm.h"
#include "c2c_turnsync.h"
#include "kws_stream_proto.h"

#if KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE || KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT || KWS_BEARLY_ROLLING_CALIBRATE_FULL
#include "tinyspeech_inputs.h" /* g_tinyspeech_test_inputs[]: the exact int8 MFCC maps Spike ran on */
_Static_assert(TINYSPEECH_TEST_INPUT_SIZE == KWS_CASE_PAYLOAD_BYTES,
               "reference input size must match the case payload size.");
_Static_assert(KWS_BEARLY_ROLLING_REF_CASE_INDEX < TINYSPEECH_TEST_NUM_CASES,
               "REF_CASE_INDEX out of range.");
#endif

#if KWS_BEARLY_ROLLING_USE_THREADLIB
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

_Static_assert((KWS_FRAMES_PER_CASE * KWS_MFCC_DIM) == KWS_CASE_PAYLOAD_BYTES,
               "Frames*dim must equal the case payload size.");

/* Turn-taking sync (see c2c_turnsync.h / /CLAUDE.md). BML consumes cases; roles: BML's turn =
 * read+verify+infer + hand back; DSP's turn = publish the next case.
 *   - own spad (0xD): BML local-reads the payload and local-writes its OWN turn register.
 *   - peer spad (DSP, reached at 0x1_C000_0000): BML remote-writes acks + DSP's turn register. */
static kws_stream_bml_spad_t *const g_bml =
    (kws_stream_bml_spad_t *)(uintptr_t)KWS_STREAM_BML_SPAD_BASE;   /* own, local reads */
static kws_stream_dsp_spad_t *const g_dsp =
    (kws_stream_dsp_spad_t *)(uintptr_t)KWS_STREAM_DSP_SPAD_PEER;   /* peer, cross-link writes */

static const char *g_labels[TINYSPEECH_NUM_CLASSES] = {
    "yes", "no", "on", "off", "stop", "go"
};

static int8_t g_case[KWS_CASE_PAYLOAD_BYTES];
static uint8_t g_int8_calibrated;
static uint8_t g_input_compared; /* print the DSP-vs-reference input diff only once */

/* Compile-time tag so every infer line shows which input the model actually ran on. */
#if KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT
#define KWS_BEARLY_SRC_TAG " [GOLDEN]"
#else
#define KWS_BEARLY_SRC_TAG " [dsp]"
#endif

static uint32_t g_last_consumed;
static uint32_t g_last_pred_class;
static uint32_t g_last_pred_score_q;
static uint64_t g_infer_count;

uint64_t target_frequency = KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

/* Remote-write our ack (+ result) into the DSP spad; ack_index is written before the turn flip. */
static void send_ack(uint32_t idx) {
  uint64_t rx_cycle = rdcycle64();

  c2c_remote_write_u32(&g_dsp->bml_pred_class, g_last_pred_class);
  c2c_remote_write_u32(&g_dsp->bml_pred_score_q, g_last_pred_score_q);
  c2c_remote_write_block(&g_dsp->bml_rx_cycle, &rx_cycle, sizeof(rx_cycle));
  c2c_remote_write_u32(&g_dsp->ack_index, idx);
}

/* Hand the turn back to DSP: write our ack/result, then flip the turn register (peer spad = commit,
 * then our own spad = "not ours"), then wake DSP. */
static void handoff_to_dsp(uint32_t ack_idx) {
  send_ack(ack_idx);
  c2c_remote_write_u32(&g_dsp->turn, C2C_TURN_DSP); /* commit: DSP's turn, in DSP's spad */
  c2c_local_write_u32(&g_bml->turn, C2C_TURN_DSP);  /* our own spad: no longer our turn */
  c2c_wake_peer();
}

static void run_inference(uint32_t case_index) {
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
  for (uint32_t i = 0; i < KWS_CASE_PAYLOAD_BYTES; ++i) {
    input.data[i] = g_case[i];
  }

#if TINYSPEECH_INT8_PIPELINE
  if (!g_int8_calibrated) {
    int calib_ok;
    KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] int8 calibration begin\n");
    tinyspeech_int8_calibration_begin();
    warm = tinyspeech_run_inference(&input);
    free_tensor(&warm);
    calib_ok = tinyspeech_int8_calibration_end();
    g_int8_calibrated = 1u;
    KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] int8 calibration %s\n",
                           calib_ok ? "done" : "failed");
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
  g_last_pred_class = (uint32_t)pred;
  g_last_pred_score_q = (uint32_t)(int32_t)(max_prob * 10000.0f);

  if ((KWS_BEARLY_ROLLING_INFER_LOG_EVERY != 0u) &&
      ((g_infer_count % KWS_BEARLY_ROLLING_INFER_LOG_EVERY) == 0u)) {
    KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] infer%s case_index=%u pred=%ld (%s) "
                           "score=%.4f model_cycles=%llu wall_cycles=%llu infers=%llu\n",
                           KWS_BEARLY_SRC_TAG,
                           (unsigned)case_index,
                           (long)pred,
                           ((pred >= 0) && (pred < TINYSPEECH_NUM_CLASSES)) ? g_labels[pred]
                                                                            : "out-of-range",
                           max_prob,
                           (unsigned long long)model_cycles,
                           (unsigned long long)(t1 - t0),
                           (unsigned long long)g_infer_count);
  }

  free_tensor(&probs);
  free_tensor(&input);
}

#if KWS_BEARLY_ROLLING_CALIBRATE_FULL && TINYSPEECH_INT8_PIPELINE
/* Calibrate int8 over the full reference set once (like the validated standalone benchmark), freeze,
 * and PRINT the three activation maxima that constitute the entire data-dependent calibration — bake
 * these into a header to skip the pass next time (tinyspeech_int8_calib_set_max). Sets
 * g_int8_calibrated so run_inference() skips its single-sample inline path. */
static void calibrate_int8_over_reference(void) {
  uint8_t shape[4] = {1, 1, KWS_MFCC_DIM, KWS_FRAMES_PER_CASE};
  int32_t m1 = 0, m2 = 0, m3 = 0;
  int calib_ok;

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] int8 calibration over %u reference cases (begin)...\n",
                         (unsigned)TINYSPEECH_TEST_NUM_CASES);
  tinyspeech_int8_calibration_begin();
  for (uint32_t tc = 0; tc < (uint32_t)TINYSPEECH_TEST_NUM_CASES; ++tc) {
    Tensor in = create_tensor(shape, 4);
    for (uint32_t i = 0; i < KWS_CASE_PAYLOAD_BYTES; ++i) {
      in.data[i] = g_tinyspeech_test_inputs[tc].data[i];
    }
    Tensor logits = tinyspeech_run_inference(&in);
    free_tensor(&logits);
    free_tensor(&in);
  }
  calib_ok = tinyspeech_int8_calibration_end();
  tinyspeech_int8_calib_get_max(&m1, &m2, &m3);
  g_int8_calibrated = 1u;

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] int8 calibration %s\n", calib_ok ? "done" : "FAILED");
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] CALIB_MAX M1=%ld M2=%ld M3=%ld  <-- bake these to skip the pass\n",
                         (long)m1, (long)m2, (long)m3);
}
#endif

void app_init(void) {
  init_test(target_frequency);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG boot: core up, pre-sync (freq=%llu)\n",
                         (unsigned long long)target_frequency);

  g_int8_calibrated = 0u;
  g_last_consumed = 0u;
  g_last_pred_class = 0u;
  g_last_pred_score_q = 0u;
  g_infer_count = 0u;
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG state initialized; own_spad=0x%08lx peer_spad=0x%09llx\n",
                         (unsigned long)KWS_STREAM_BML_SPAD_BASE,
                         (unsigned long long)KWS_STREAM_DSP_SPAD_PEER);

#if KWS_BEARLY_ROLLING_USE_THREADLIB
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG hthread_init begin\n");
  hthread_init();
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG hthread_init done\n");
#endif

  /* NOTE: app_init does NO cross-link (spad) writes. All peer writes happen in app_main. */

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] preparing TinySpeech runtime...\n");
  tinyspeech_prepare_runtime();
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] TinySpeech runtime ready\n");

#if KWS_BEARLY_ROLLING_CALIBRATE_FULL && TINYSPEECH_INT8_PIPELINE
  calibrate_int8_over_reference();
#endif

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] init own_spad=0x%08lx peer_spad=0x%09llx payload_bytes=%u; waiting for producer\n",
                         (unsigned long)KWS_STREAM_BML_SPAD_BASE,
                         (unsigned long long)KWS_STREAM_DSP_SPAD_PEER,
                         (unsigned)KWS_CASE_PAYLOAD_BYTES);
}

void app_main(void) {
  /* Boot grace: no cross-link access while DSP is also coming up. */
  uint64_t start = rdcycle64();
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG entered app_main; startup grace %llu cycles (no link access)\n",
                         (unsigned long long)(uint64_t)KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES);
  while ((rdcycle64() - start) < (uint64_t)KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES) {
    __asm__ volatile("nop");
  }

  /* Local boot-clear of our OWN 0xD control block BEFORE announcing ready: turn = DSP (not ours,
   * so we stay parked until DSP hands off) and case_index = 0. Local writes — safe. Done before
   * bml_ready so it lands before DSP (which waits for bml_ready) writes anything into our spad,
   * which also removes the stale-SRAM / boot-order hazard. */
  c2c_local_write_u32(&g_bml->turn, C2C_TURN_DSP);
  c2c_local_write_u32(&g_bml->case_index, 0u);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG cleared own 0xD control (turn=DSP, case_index=0)\n");

  /* Arm MSIP + timer wake before any wfi. */
  c2c_arm_wake();

  /* Boot barrier: announce readiness by writing bml_ready into the DSP spad. DSP will not write our
   * spad until it sees this. */
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG grace done; announcing bml_ready -> DSP spad\n");
  c2c_remote_write_u32(&g_dsp->bml_ready, KWS_STREAM_READY_MAGIC);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG bml_ready announced; entering turn-taking loop\n");

  /* Turn-taking consumer loop. Wait for our turn AND a NEW case (case_index advanced). A duplicate
   * grant (turn==BML but case_index already consumed) means our previous ack was lost -> re-ack
   * WITHOUT re-inferring, so DSP's self-heal re-grant is absorbed idempotently. */
  while (1) {
    uint32_t idx = 0u;
    uint32_t checksum;

    for (;;) {
      uint32_t turn;
      c2c_full_flush();
      turn = g_bml->turn;
      idx = g_bml->case_index;
      if (turn == C2C_TURN_BML) {
        if (idx > g_last_consumed) {
          break; /* genuinely new case -> process it below */
        }
        /* Duplicate grant (a lost ack). Re-hand-back so DSP advances; do not re-infer. */
        KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] dup grant case_index=%u (last_consumed=%u); re-acking\n",
                               (unsigned)idx, (unsigned)g_last_consumed);
        handoff_to_dsp(g_last_consumed);
      }
      c2c_sleep_until_tick(); /* wake on DSP's grant MSIP or the periodic timer */
    }

    checksum = g_bml->payload_checksum;
    if (!c2c_local_read_block_verify(g_case, g_bml->case_payload, KWS_CASE_PAYLOAD_BYTES, checksum)) {
      /* Torn read after retries (rare). Re-ack last so DSP re-grants; static payload re-reads clean. */
      KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] verify FAILED case_index=%u checksum=0x%08lx; re-acking last=%u\n",
                             (unsigned)idx, (unsigned long)checksum, (unsigned)g_last_consumed);
      handoff_to_dsp(g_last_consumed);
      continue;
    }

#if KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE
    /* Once: compare the received (DSP-computed) MFCC map against the Spike reference feature map.
     * The link is already checksum-verified, so any mismatch here is the on-chip MFCC front-end
     * (window/hop, mel bank, log, DCT, or the quant scale/zero) diverging from the reference. */
    if (!g_input_compared) {
      const tinyspeech_test_input_case_t *ref =
          &g_tinyspeech_test_inputs[KWS_BEARLY_ROLLING_REF_CASE_INDEX];
      uint32_t mism = 0u;
      int32_t maxabs = 0;
      for (uint32_t i = 0; i < KWS_CASE_PAYLOAD_BYTES; ++i) {
        int32_t d = (int32_t)g_case[i] - (int32_t)ref->data[i];
        if (d != 0) {
          mism++;
        }
        if (d < 0) {
          d = -d;
        }
        if (d > maxabs) {
          maxabs = d;
        }
      }
      KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] INPUT-CMP ref=%s exp=%ld recv_cksum=0x%08lx ref_cksum=0x%08lx mism=%u/%u max_abs_diff=%ld\n",
                             ref->name, (long)ref->expected_label,
                             (unsigned long)c2c_checksum(g_case, KWS_CASE_PAYLOAD_BYTES),
                             (unsigned long)c2c_checksum(ref->data, KWS_CASE_PAYLOAD_BYTES),
                             (unsigned)mism, (unsigned)KWS_CASE_PAYLOAD_BYTES, (long)maxabs);
      KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] INPUT-CMP recv[0:16]=");
      for (uint32_t i = 0; i < 16u; ++i) {
        KWS_BEARLY_ROLLING_LOG(" %d", (int)g_case[i]);
      }
      KWS_BEARLY_ROLLING_LOG("\n[bearly-kws-stream] INPUT-CMP  ref[0:16]=");
      for (uint32_t i = 0; i < 16u; ++i) {
        KWS_BEARLY_ROLLING_LOG(" %d", (int)ref->data[i]);
      }
      KWS_BEARLY_ROLLING_LOG("\n");
      g_input_compared = 1u;
    }
#endif

#if KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT
    /* Isolation mode: infer on the known-good reference features instead of the received case, so
     * a wrong/varying prediction points at the model/inference rather than the MFCC front-end. */
    {
      static uint8_t g_golden_announced = 0u;
      if (!g_golden_announced) {
        KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] GOLDEN INPUT ACTIVE: overriding received case with ref=%s (expect=%ld yes) every inference\n",
                               g_tinyspeech_test_inputs[KWS_BEARLY_ROLLING_REF_CASE_INDEX].name,
                               (long)g_tinyspeech_test_inputs[KWS_BEARLY_ROLLING_REF_CASE_INDEX].expected_label);
        g_golden_announced = 1u;
      }
    }
    for (uint32_t i = 0; i < KWS_CASE_PAYLOAD_BYTES; ++i) {
      g_case[i] = g_tinyspeech_test_inputs[KWS_BEARLY_ROLLING_REF_CASE_INDEX].data[i];
    }
#endif

    run_inference(idx);
    g_last_consumed = idx;
    handoff_to_dsp(idx);
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
