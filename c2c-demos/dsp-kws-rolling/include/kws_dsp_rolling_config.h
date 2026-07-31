#ifndef C2C_DSP_KWS_ROLLING_CONFIG_H
#define C2C_DSP_KWS_ROLLING_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "kws_rolling_proto.h"
#include "kws_stream_proto.h"
#include "c2c_shm.h"

#ifndef KWS_DSP_ROLLING_LOG_ENABLE
#define KWS_DSP_ROLLING_LOG_ENABLE 1
#endif

#if KWS_DSP_ROLLING_LOG_ENABLE
#define KWS_DSP_ROLLING_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define KWS_DSP_ROLLING_LOG(...) do { } while (0)
#endif

#ifndef KWS_DSP_ROLLING_TARGET_FREQUENCY_HZ
#define KWS_DSP_ROLLING_TARGET_FREQUENCY_HZ 500000000ULL
#endif

#ifndef KWS_DSP_ROLLING_SHARED_BASE
#define KWS_DSP_ROLLING_SHARED_BASE KWS_ROLLING_SHARED_BASE_ADDR
#endif

#ifndef KWS_DSP_ROLLING_SHARED_BYTES
#define KWS_DSP_ROLLING_SHARED_BYTES KWS_ROLLING_SHARED_BYTES
#endif

#ifndef KWS_DSP_ROLLING_COMMIT_SEQ_ADDR
#define KWS_DSP_ROLLING_COMMIT_SEQ_ADDR KWS_ROLLING_COMMIT_SEQ_ADDR
#endif

#ifndef KWS_DSP_ROLLING_FRAME_ADDR
#define KWS_DSP_ROLLING_FRAME_ADDR KWS_ROLLING_FRAME_ADDR
#endif

#ifndef KWS_DSP_ROLLING_FRAMES_PER_CASE
#define KWS_DSP_ROLLING_FRAMES_PER_CASE KWS_FRAMES_PER_CASE
#endif

#ifndef KWS_DSP_ROLLING_STEADY_FRAME_IDX
#define KWS_DSP_ROLLING_STEADY_FRAME_IDX (KWS_DSP_ROLLING_FRAMES_PER_CASE - 1u)
#endif

/* MFCC int8 quantization recipe — MUST match how the deployed model's features were quantized.
 *  - NORMALIZE=1: per-case peak normalization q = clip(round(x * 127/max|x|)). This matched the
 *    ORIGINAL 6-word Speech-Commands reference set (tinyspeech_inputs.h generated that way).
 *  - NORMALIZE=0: fixed scale q = clip(round(x * QUANT_SCALE + QUANT_ZERO)) with SCALE/ZERO below.
 * The 2026-07-27 **6-word retrain** (go/bird/cat/dog/happy/tree, real Speech-Commands audio via
 * dsp25-tests/tinyspeech-test/scripts/rebuild_weights_simplecnn.py) quantizes each case with per-case
 * PEAK normalization (`q = clip(round(x * 127/max|x|))`, see `_quantize_mfcc_to_int8_like_runtime`),
 * so the matching DSP recipe is NORMALIZE=1. (The earlier 8-word TTS model used fixed scale 4.0 ->
 * NORMALIZE=0.) A mismatch here feeds the model a different int8 distribution than training. */
#ifndef KWS_DSP_ROLLING_MFCC_NORMALIZE
#define KWS_DSP_ROLLING_MFCC_NORMALIZE 1
#endif

#ifndef KWS_DSP_ROLLING_MFCC_QUANT_SCALE
#define KWS_DSP_ROLLING_MFCC_QUANT_SCALE 4.0f
#endif

#ifndef KWS_DSP_ROLLING_MFCC_QUANT_ZERO
#define KWS_DSP_ROLLING_MFCC_QUANT_ZERO 0.0f
#endif

#ifndef KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES
#define KWS_DSP_ROLLING_SIGNAL_HOP_SAMPLES 160u
#endif

#ifndef KWS_DSP_ROLLING_USE_THREADLIB
#define KWS_DSP_ROLLING_USE_THREADLIB 0
#endif

/* Multi-testcase mode (plan 003 axis B). 0 (default) = the original single embedded yes_test_005
 * waveform, streamed repeatedly (static payload). 1 = round-robin over the generated waveform set
 * in kws_dsp_signals.h (a few recordings per keyword), computing a fresh MFCC case per recording
 * and tagging each with its ground-truth expected_label so BML can score pred-vs-expected. Multi
 * mode forces a full payload re-publish every case (the payload changes) regardless of
 * KWS_DSP_ROLLING_STATIC_PAYLOAD. Generate kws_dsp_signals.h first via scripts/gen_dsp_signals.py. */
#ifndef KWS_DSP_ROLLING_MULTI_SIGNAL
#define KWS_DSP_ROLLING_MULTI_SIGNAL 0
#endif

#ifndef KWS_DSP_ROLLING_SEND_INTERVAL_CYCLES
#define KWS_DSP_ROLLING_SEND_INTERVAL_CYCLES 0ULL
#endif

#ifndef KWS_DSP_ROLLING_LOG_EVERY
#define KWS_DSP_ROLLING_LOG_EVERY 1u
#endif

/* Static-payload fast path: this milestone streams ONE precomputed case repeatedly. Writing the
 * full 1128-byte payload into 0xD every round (x C2C_SHM_WRITE_REPEATS ~= 4500 word-writes) floods
 * the link; over many rounds one of those cross-link writes eventually collides with BML's polling
 * and wedges the link (the "stops after several cases" symptom). With this set, DSP ships the
 * payload+checksum ONCE, then each round only re-commits case_index (one word) to re-trigger
 * inference on the payload already resident in BML's spad. Set 0 to restore full-payload-per-round
 * (required once the payload actually changes, e.g. the VAD milestone). Retries always resend full. */
#ifndef KWS_DSP_ROLLING_STATIC_PAYLOAD
#define KWS_DSP_ROLLING_STATIC_PAYLOAD 1
#endif

/* Pacing delay (core cycles) between receiving BML's ack and granting the next case. This spaces
 * out how often a new prediction is published so the stream is readable, and stays link-quiet
 * (DSP touches nothing during the spin; BML is parked awaiting its turn), so it cannot race the
 * handoff. ~500M cycles is ~1 s @ 500 MHz (~10 s @ 50 MHz); set 0 to run flat-out. Tune to taste. */
#ifndef KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES
#define KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES 500000000ULL
#endif

/* How often (in poll iterations) to log while waiting for BML to arm (rx_ready). 0 = never. */
#ifndef KWS_DSP_ROLLING_RX_WAIT_LOG_EVERY
#define KWS_DSP_ROLLING_RX_WAIT_LOG_EVERY 200000u
#endif

/* Settle margin after seeing bml_ready before the first cross-link write into 0xD. bml_ready only
 * means BML SET the flag, not that it has finished its first cache flush + reached its steady
 * read-only poll loop; writing 0xD inside that window collides with BML and wedges it (and then
 * hangs us on the wedged-peer write). This gap makes the handshake independent of how early DSP
 * happens to catch the flag. Increase if low-poll-count runs still wedge. */
#ifndef KWS_DSP_ROLLING_POST_READY_SETTLE_CYCLES
#define KWS_DSP_ROLLING_POST_READY_SETTLE_CYCLES 50000000ULL
#endif

/* Ack polls to wait after a publish before re-publishing the same case (self-heal on drops).
 * Between publishes DSP only reads its LOCAL 0xC spad (no link writes), so keep this large: the
 * peer must be able to boot + prep + run inference in the gap without DSP driving the link. */
#ifndef KWS_DSP_ROLLING_ACK_POLL_BUDGET
#define KWS_DSP_ROLLING_ACK_POLL_BUDGET 20000u
#endif

/* Diagnostic: publish exactly one case, then stop touching the link (wfi). Lets us test whether
 * a single transfer succeeds while BML is concurrently running, isolating link contention. */
#ifndef KWS_DSP_ROLLING_PUBLISH_ONCE
#define KWS_DSP_ROLLING_PUBLISH_ONCE 0
#endif

#ifndef KWS_DSP_ROLLING_DEBUG_WRITE_ENABLE
#define KWS_DSP_ROLLING_DEBUG_WRITE_ENABLE 0
#endif

#ifndef KWS_DSP_ROLLING_BEARLY_DONE_ADDR
#define KWS_DSP_ROLLING_BEARLY_DONE_ADDR KWS_ROLLING_BEARLY_DONE_ADDR
#endif

#ifndef KWS_DSP_ROLLING_CACHE_LINE_BYTES
#define KWS_DSP_ROLLING_CACHE_LINE_BYTES 64u
#endif

#ifndef KWS_DSP_ROLLING_CACHE_EVICT_BYTES
#define KWS_DSP_ROLLING_CACHE_EVICT_BYTES (256u * 1024u)
#endif

/* ------------------------------------------------------------------------------------------------
 * Live I2S microphone audio source (plan 001 P1). When KWS_DSP_ROLLING_USE_MIC=1 the demo drops the
 * embedded waveform table and instead captures ~1 s of live audio from the DSP I2S mic (channel 0,
 * proven in dsp-i2s-test / see the I2S validation note), then runs the SAME MFCC -> quantize ->
 * stream pipeline on it and reports BML's prediction. Ground truth is unknown for live audio, so
 * each case is tagged expected_label = -1 (BML's PASS/FAIL tally is meaningless in this mode). Set
 * via the CMake option KWS_DSP_ROLLING_USE_MIC (needs a clean reconfigure). Mutually exclusive with
 * KWS_DSP_ROLLING_MULTI_SIGNAL. ---------------------------------------------------------------- */
#ifndef KWS_DSP_ROLLING_USE_MIC
#define KWS_DSP_ROLLING_USE_MIC 0
#endif

#if KWS_DSP_ROLLING_USE_MIC && KWS_DSP_ROLLING_MULTI_SIGNAL
#error "KWS_DSP_ROLLING_USE_MIC and KWS_DSP_ROLLING_MULTI_SIGNAL are mutually exclusive."
#endif

/* I2S mic channel (0 = mic per the validated pinout: BCLK0/LRCLK0/SDIN0). */
#ifndef KWS_DSP_ROLLING_MIC_CHANNEL
#define KWS_DSP_ROLLING_MIC_CHANNEL 0
#endif

/* Target mic sample rate. The MFCC front-end assumes 16 kHz (hop 160 = 10 ms, 1024-pt FFT), so the
 * mic is clocked at 16 kHz directly via set_I2S_sample_freq() using the demo's operating frequency
 * (KWS_DSP_ROLLING_TARGET_FREQUENCY_HZ) — no software resampling. */
#ifndef KWS_DSP_ROLLING_MIC_SAMPLE_RATE_HZ
#define KWS_DSP_ROLLING_MIC_SAMPLE_RATE_HZ 16000u
#endif
#ifndef KWS_DSP_ROLLING_MIC_BITDEPTH
#define KWS_DSP_ROLLING_MIC_BITDEPTH 32u
#endif

/* Samples per captured case. Must cover all MFCC frames: (FRAMES-1)*HOP + FFT_LEN. 16000 (= 1 s @
 * 16 kHz) matches the embedded Speech Commands clips. */
#ifndef KWS_DSP_ROLLING_MIC_NUM_SAMPLES
#define KWS_DSP_ROLLING_MIC_NUM_SAMPLES 16000u
#endif

/* Mic delivers a 24-bit sample in the TOP of each 32-bit slot (low byte idle 0xFF). Recover the
 * signed 24-bit value with an arithmetic >>8, then scale to float ~[-1,1] by 1/2^23 to match the
 * range of the embedded float waveforms. A large near-constant DC offset is removed per capture. */
#ifndef KWS_DSP_ROLLING_MIC_SAMPLE_SHIFT
#define KWS_DSP_ROLLING_MIC_SAMPLE_SHIFT 8
#endif
#ifndef KWS_DSP_ROLLING_MIC_FULLSCALE
#define KWS_DSP_ROLLING_MIC_FULLSCALE 8388608.0f /* 2^23 */
#endif

/* --- Voice-activity gate (plan 001 §7). Monitor short-frame AC energy; when it crosses
 * KWS_DSP_ROLLING_VAD_THRESHOLD, capture the case starting from a short pre-roll before the onset so
 * the word's attack isn't clipped, then fill a full 1 s window. Set KWS_DSP_ROLLING_VAD_ENABLE=0 to
 * free-run (capture a window immediately, no gating). --------------------------------------------- */
#ifndef KWS_DSP_ROLLING_VAD_ENABLE
#define KWS_DSP_ROLLING_VAD_ENABLE 1
#endif
/* Monitoring frame length in samples (must be even; 320 = 20 ms @ 16 kHz). AC energy = mean of
 * (x - frame_mean)^2 over the frame (frame-mean removal = crude DC high-pass for the gate). */
#ifndef KWS_DSP_ROLLING_VAD_FRAME_SAMPLES
#define KWS_DSP_ROLLING_VAD_FRAME_SAMPLES 320u
#endif
/* Pre-roll captured ahead of the onset (must be even and a multiple of VAD_FRAME_SAMPLES; 3200 =
 * 200 ms). Ensures the onset frame + a little lead-in land inside the window. */
#ifndef KWS_DSP_ROLLING_VAD_PREROLL_SAMPLES
#define KWS_DSP_ROLLING_VAD_PREROLL_SAMPLES 3200u
#endif
/* Onset threshold on the frame AC energy (scaled-float variance units; samples are /2^23). The
 * noise floor vs speech level is mic/room dependent — monitoring-frame energy is logged (throttled)
 * so this can be tuned from a run. */
#ifndef KWS_DSP_ROLLING_VAD_THRESHOLD
#define KWS_DSP_ROLLING_VAD_THRESHOLD 5.0e-4f
#endif
/* Throttle: log one VAD monitoring energy line every N frames while listening (0 = never). */
#ifndef KWS_DSP_ROLLING_VAD_LOG_EVERY
#define KWS_DSP_ROLLING_VAD_LOG_EVERY 25u
#endif

#endif /* C2C_DSP_KWS_ROLLING_CONFIG_H */
