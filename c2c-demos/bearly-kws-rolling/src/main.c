#include "main.h"

#include <limits.h>

#include "tensor.h"

#include "c2c_shm.h"
#include "kws_stream_proto.h"

#if KWS_BEARLY_ROLLING_USE_THREADLIB
void hthread_init(void);
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
static void mc_nop_worker(void *arg) { (void)arg; }
#endif

_Static_assert((KWS_FRAMES_PER_CASE * KWS_MFCC_DIM) == KWS_CASE_PAYLOAD_BYTES,
               "Frames*dim must equal the case payload size.");

/* BML-adjacent spad (0xD): BML local-reads.  DSP-adjacent spad (0xC): BML remote-writes. */
static kws_stream_bml_spad_t *const g_bml =
    (kws_stream_bml_spad_t *)(uintptr_t)KWS_STREAM_BML_SPAD_BASE;
static kws_stream_dsp_spad_t *const g_dsp =
    (kws_stream_dsp_spad_t *)(uintptr_t)KWS_STREAM_DSP_SPAD_BASE;

static const char *g_labels[TINYSPEECH_NUM_CLASSES] = {
    "yes", "no", "on", "off", "stop", "go"
};

static int8_t g_case[KWS_CASE_PAYLOAD_BYTES];
static uint8_t g_int8_calibrated;

static uint32_t g_last_epoch;
static uint32_t g_last_consumed;
static uint32_t g_last_pred_class;
static uint32_t g_last_pred_score_q;
static uint64_t g_poll_count;
static uint64_t g_infer_count;

uint64_t target_frequency = KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

/* Remote-write our ack (+ result) into the DSP spad; ack_index is written LAST. */
static void send_ack(uint32_t epoch, uint32_t idx) {
  uint64_t rx_cycle = rdcycle64();

  c2c_remote_write_u32(&g_dsp->epoch_echo, epoch);
  c2c_remote_write_u32(&g_dsp->bml_pred_class, g_last_pred_class);
  c2c_remote_write_u32(&g_dsp->bml_pred_score_q, g_last_pred_score_q);
  c2c_remote_write_block(&g_dsp->bml_rx_cycle, &rx_cycle, sizeof(rx_cycle));
  c2c_remote_write_u32(&g_dsp->ack_index, idx); /* ack commit */
}

static void run_inference(uint32_t epoch, uint32_t case_index) {
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
    KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] infer epoch=0x%08lx case_index=%u pred=%ld (%s) "
                           "score=%.4f model_cycles=%llu wall_cycles=%llu infers=%llu\n",
                           (unsigned long)epoch,
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

void app_init(void) {
  init_test(target_frequency);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG boot: core up, pre-sync (freq=%llu)\n",
                         (unsigned long long)target_frequency);

  g_int8_calibrated = 0u;
  g_last_epoch = 0u;
  g_last_consumed = 0u;
  g_last_pred_class = 0u;
  g_last_pred_score_q = 0u;
  g_poll_count = 0u;
  g_infer_count = 0u;
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG state initialized; dsp_spad=0x%08lx bml_spad=0x%08lx\n",
                         (unsigned long)KWS_STREAM_DSP_SPAD_BASE,
                         (unsigned long)KWS_STREAM_BML_SPAD_BASE);

#if KWS_BEARLY_ROLLING_USE_THREADLIB
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG hthread_init begin\n");
  hthread_init();
  hthread_issue(1, mc_nop_worker, NULL);
  hthread_join(1);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG hthread_init done\n");
#endif

  /* NOTE: app_init does NO cross-link (spad) writes. Touching the peer's spad (0xC) before the
   * peer is up hangs the core. BML's only 0xC writes are acks, in app_main, after full boot. */

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] preparing TinySpeech runtime...\n");
  tinyspeech_prepare_runtime();
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] TinySpeech runtime ready\n");

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] init bml_spad=0x%08lx dsp_spad=0x%08lx payload_bytes=%u; waiting for producer\n",
                         (unsigned long)KWS_STREAM_BML_SPAD_BASE,
                         (unsigned long)KWS_STREAM_DSP_SPAD_BASE,
                         (unsigned)KWS_CASE_PAYLOAD_BYTES);
}

void app_main(void) {
  /* Boot barrier: wait a grace period (no cross-link access) so DSP is also booted, THEN announce
   * readiness by writing bml_ready into the DSP spad. DSP will not write our spad until it sees
   * this — writing a still-booting chip's spad kills it. */
  uint64_t start = rdcycle64();
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG entered app_main; startup grace %llu cycles (no link access)\n",
                         (unsigned long long)(uint64_t)KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES);
  while ((rdcycle64() - start) < (uint64_t)KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES) {
    __asm__ volatile("nop");
  }

  /* Wipe stale commit signals left in OUR OWN 0xD spad from a previous run BEFORE announcing
   * ready. Scratchpad SRAM survives a chip-only reset, so a prior run's epoch/case_index/payload
   * are still here and self-consistent (same static signal -> same checksum) — without this, BML
   * would "receive" and infer a case the current DSP never sent. These are LOCAL writes into our
   * own spad (safe). Clearing epoch=0 alone makes the poll loop wait, but we clear case_index too
   * so a fresh epoch can't pair with a stale index. Done before bml_ready so it is guaranteed to
   * land before DSP (which waits for bml_ready) writes anything into 0xD. */
  c2c_local_write_u32(&g_bml->epoch, 0u);
  c2c_local_write_u32(&g_bml->case_index, 0u);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG cleared stale epoch/case_index in own 0xD spad\n");

  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG grace done; announcing bml_ready -> 0x%08lx\n",
                         (unsigned long)KWS_STREAM_DSP_SPAD_BASE);
  c2c_remote_write_u32(&g_dsp->bml_ready, KWS_STREAM_READY_MAGIC);
  KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] DEBUG bml_ready announced; polling own spad 0x%08lx\n",
                         (unsigned long)KWS_STREAM_BML_SPAD_BASE);

  /* Turn-taking consumer loop. Each round: ARM (tell DSP we want the next case and are parking) ->
   * PARK (go fully quiet so DSP's 0xD write lands uncontended) -> READ our own 0xD (now stable) ->
   * verify + infer -> ack. We touch 0xD ONLY after the park, never while DSP might be writing it,
   * which is what eliminates the payload/poll collision that was wedging the link. */
  while (1) {
    uint32_t want = g_last_consumed + 1u;
    uint32_t epoch;
    uint32_t idx;
    uint32_t checksum;

    g_poll_count++;

    /* ARM: writes into the DSP spad (0xC) only. rx_seq (which case we want) first, then rx_ready
     * (the magic DSP polls on) last. DSP will not write 0xD until it sees rx_ready. */
    c2c_remote_write_u32(&g_dsp->rx_seq, want);
    c2c_remote_write_u32(&g_dsp->rx_ready, KWS_STREAM_RX_READY_MAGIC);

    /* PARK: no 0xD access, no link access — pure spin — long enough for DSP to detect the arm and
     * finish its (possibly full-payload) write into our spad. */
    {
      uint64_t p0 = rdcycle64();
      while ((rdcycle64() - p0) < (uint64_t)KWS_BEARLY_ROLLING_RX_PARK_CYCLES) {
        __asm__ volatile("nop");
      }
    }

    /* READ: DSP has finished writing and is now polling its own 0xC for our next arm, so 0xD is
     * uncontended. One flush, then the commit signals. */
    c2c_full_flush();
    epoch = g_bml->epoch;
    idx = g_bml->case_index;
    checksum = g_bml->payload_checksum;

    if ((KWS_BEARLY_ROLLING_WAIT_LOG_EVERY != 0u) &&
        ((g_poll_count % KWS_BEARLY_ROLLING_WAIT_LOG_EVERY) == 0u)) {
      KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] arm want=%u -> park-read epoch=0x%08lx case_index=%u checksum=0x%08lx round=%llu\n",
                             (unsigned)want, (unsigned long)epoch, (unsigned)idx,
                             (unsigned long)checksum, (unsigned long long)g_poll_count);
    }

    if (epoch == 0u) {
      continue; /* DSP has not written identity yet; re-arm and park again */
    }

    /* New producer epoch -> rebaseline; restart the case sequence at 1 under the new epoch. */
    if (epoch != g_last_epoch) {
      g_last_epoch = epoch;
      g_last_consumed = 0u;
      KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] new epoch=0x%08lx (rebaselined)\n",
                             (unsigned long)epoch);
      continue;
    }

    if (idx != want) {
      /* DSP hasn't committed the case we asked for yet (slow write / missed the park). Re-arm the
       * same `want`; DSP resends the full payload when it sees a repeated request. */
      continue;
    }

    if (!c2c_local_read_block_verify(g_case, g_bml->case_payload, KWS_CASE_PAYLOAD_BYTES, checksum)) {
      /* Torn payload (rare now that reads are uncontended). Re-arm same want -> DSP resends full. */
      KWS_BEARLY_ROLLING_LOG("[bearly-kws-stream] verify FAILED want=%u checksum=0x%08lx; re-arming\n",
                             (unsigned)want, (unsigned long)checksum);
      continue;
    }

    run_inference(epoch, want);
    g_last_consumed = want;
    send_ack(epoch, want);
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
