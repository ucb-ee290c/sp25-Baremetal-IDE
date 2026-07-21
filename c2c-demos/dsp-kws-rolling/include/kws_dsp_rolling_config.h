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

#endif /* C2C_DSP_KWS_ROLLING_CONFIG_H */
