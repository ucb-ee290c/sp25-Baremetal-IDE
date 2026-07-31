#ifndef C2C_BEARLY_KWS_ROLLING_CONFIG_H
#define C2C_BEARLY_KWS_ROLLING_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "kws_rolling_proto.h"
#include "kws_stream_proto.h"
#include "c2c_shm.h"

#ifndef KWS_BEARLY_ROLLING_LOG_ENABLE
#define KWS_BEARLY_ROLLING_LOG_ENABLE 1
#endif

#if KWS_BEARLY_ROLLING_LOG_ENABLE
#define KWS_BEARLY_ROLLING_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define KWS_BEARLY_ROLLING_LOG(...) do { } while (0)
#endif

#ifndef KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ
#define KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ 500000000ULL
#endif

#ifndef KWS_BEARLY_ROLLING_SHM_BASE
#define KWS_BEARLY_ROLLING_SHM_BASE KWS_ROLLING_SHARED_BASE_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_SHM_BYTES
#define KWS_BEARLY_ROLLING_SHM_BYTES KWS_ROLLING_SHARED_BYTES
#endif

#ifndef KWS_BEARLY_ROLLING_COMMIT_SEQ_ADDR
#define KWS_BEARLY_ROLLING_COMMIT_SEQ_ADDR KWS_ROLLING_COMMIT_SEQ_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_FRAME_ADDR
#define KWS_BEARLY_ROLLING_FRAME_ADDR KWS_ROLLING_FRAME_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_WAIT_LOG_EVERY
#define KWS_BEARLY_ROLLING_WAIT_LOG_EVERY 200000u
#endif

#ifndef KWS_BEARLY_ROLLING_RX_LOG_EVERY
#define KWS_BEARLY_ROLLING_RX_LOG_EVERY 1u
#endif

/* Re-ack an already-consumed case (covers a dropped ack) at most once per this many polls, so BML
 * does not continuously write 0xC — continuous cross-link writes block the peer from booting. */
#ifndef KWS_BEARLY_ROLLING_REACK_EVERY
#define KWS_BEARLY_ROLLING_REACK_EVERY 20000u
#endif

/* Turn-taking park window (core cycles). After arming (rx_ready -> DSP), BML spins quietly for this
 * long with NO 0xD/link access so DSP can write the next case uncontended, then reads its own spad.
 * Must exceed DSP's detect latency + its (possibly full-payload) write time. Generous by default for
 * bring-up reliability; tune DOWN once the link is proven stable to raise throughput. */
#ifndef KWS_BEARLY_ROLLING_RX_PARK_CYCLES
#define KWS_BEARLY_ROLLING_RX_PARK_CYCLES 50000000ULL
#endif

/* After booting, wait this many cycles before the first cross-link write (bml_ready -> 0xC), so
 * DSP has time to boot too. Writing a peer's spad while it is still booting kills it. */
#ifndef KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES
#define KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES 1000000000ULL
#endif

#ifndef KWS_BEARLY_ROLLING_INFER_LOG_EVERY
#define KWS_BEARLY_ROLLING_INFER_LOG_EVERY 1u
#endif

/* Confidence gate. The model runs with softmax OFF (TINYSPEECH_OUTPUT_SOFTMAX=0), so the prediction
 * "score" is the top raw LOGIT (not a 0..1 probability). A weak/uncertain window has a low top
 * logit. When the winning score is NOT greater than this threshold, the RESULT line reports
 * "(no word)" instead of announcing a keyword — filtering low-confidence noise (e.g. non-speech that
 * still tripped the mic VAD gate). Raise for stricter gating, lower to announce more. Only affects
 * the printed verdict; the raw pred/score are still logged on the infer line and sent to DSP. */
#ifndef KWS_BEARLY_ROLLING_MIN_SCORE
#define KWS_BEARLY_ROLLING_MIN_SCORE 2.0f
#endif

/* Score the prediction against the ground-truth expected_label DSP tags each case with (multi-
 * testcase mode, plan 003). When 1, BML logs a per-case PASS/FAIL and a running correct/total tally.
 * Cases arriving with expected_label < 0 (unknown) are logged but excluded from the tally. Harmless
 * with the single-signal DSP (every case is yes -> expected_label 0). */
#ifndef KWS_BEARLY_ROLLING_CHECK_EXPECTED
#define KWS_BEARLY_ROLLING_CHECK_EXPECTED 1
#endif

/* --- Debug: compare the received input against the Spike/reference feature map ------------------
 * The reference (tinyspeech_inputs.h) stores the exact int8 MFCC 12x94 map Spike ran on. DSP,
 * instead, computes MFCC on-chip. Set COMPARE=1 to print, once, DSP's received g_case vs the
 * reference input for REF_CASE_INDEX (mismatch count + previews) — tells us if the on-chip MFCC
 * front-end matches the reference. Set USE_GOLDEN_INPUT=1 to infer on the reference bytes INSTEAD
 * of the received case, isolating the model/inference from the MFCC front-end and the link. */
/* Default OFF since the 6-word retrain (2026-07-27): the shipped goldens (tinyspeech_inputs.h) are
 * still the old 8-word maps and are meaningless for live mic audio, so INPUT-CMP is just noise.
 * Set to 1 only if you regenerate the goldens for the current word set and want the diagnostic. */
#ifndef KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE
#define KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE 0
#endif
#ifndef KWS_BEARLY_ROLLING_REF_CASE_INDEX
#define KWS_BEARLY_ROLLING_REF_CASE_INDEX 5u /* yes_test_005 = ./yes/0cb74144_nohash_2.wav */
#endif
#ifndef KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT
#define KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT 0
#endif

/* Calibrate int8 over the FULL reference set at boot (like the validated standalone benchmark),
 * then freeze — instead of calibrating on a single received case (degenerate scales -> mispredicts).
 * Requires tinyspeech_inputs.h. Set 0 to revert to the old single-sample inline calibration. */
#ifndef KWS_BEARLY_ROLLING_CALIBRATE_FULL
#define KWS_BEARLY_ROLLING_CALIBRATE_FULL 1
#endif

#ifndef KWS_BEARLY_ROLLING_PRINT_LAYER_CYCLES
#define KWS_BEARLY_ROLLING_PRINT_LAYER_CYCLES 1u
#endif

#ifndef KWS_BEARLY_ROLLING_CLEAR_SHM_ON_BOOT
#define KWS_BEARLY_ROLLING_CLEAR_SHM_ON_BOOT 0u
#endif

#ifndef KWS_BEARLY_ROLLING_DONE_ADDR
#define KWS_BEARLY_ROLLING_DONE_ADDR KWS_ROLLING_BEARLY_DONE_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_CACHE_LINE_BYTES
#define KWS_BEARLY_ROLLING_CACHE_LINE_BYTES 64u
#endif

#ifndef KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES
#define KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES (256u * 1024u)
#endif

#ifndef KWS_BEARLY_ROLLING_USE_THREADLIB
#define KWS_BEARLY_ROLLING_USE_THREADLIB 1
#endif

#ifndef KWS_BEARLY_ROLLING_TCM_BASE
#define KWS_BEARLY_ROLLING_TCM_BASE TCM_BASE
#endif

#ifndef KWS_BEARLY_ROLLING_TCM_BYTES
#define KWS_BEARLY_ROLLING_TCM_BYTES 0x2000u
#endif

#ifndef KWS_BEARLY_ROLLING_TCM_WINDOW_OFFSET
#define KWS_BEARLY_ROLLING_TCM_WINDOW_OFFSET 0x1000u
#endif

#define KWS_BEARLY_ROLLING_TCM_WINDOW_ADDR \
  (KWS_BEARLY_ROLLING_TCM_BASE + KWS_BEARLY_ROLLING_TCM_WINDOW_OFFSET)

#endif /* C2C_BEARLY_KWS_ROLLING_CONFIG_H */
