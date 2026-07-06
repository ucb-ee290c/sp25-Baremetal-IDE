#ifndef C2C_KWS_STREAM_PROTO_H
#define C2C_KWS_STREAM_PROTO_H

/*
 * kws_stream_proto — two-scratchpad protocol for the full C2C KWS streaming demo.
 *
 * Hardware access rule (see /CLAUDE.md): a chip may READ only its own adjacent spad and may
 * WRITE to both. So each field a chip must read lives in that chip's own spad, placed there by
 * the other chip's cross-link (remote) write.
 *
 *   0xC0000000  DSP-adjacent spad  -> DSP reads locally; BML remote-writes (ack + result)
 *   0xD0000000  BML-adjacent spad  -> BML reads locally; DSP remote-writes (payload + case_index)
 *
 * Order-independent, stale-proof, self-healing handshake:
 *   - Each chip boot-clears the control block of the spad it writes into, wiping leftover state.
 *   - DSP picks a per-boot `epoch` nonce (nonzero) and stamps it in the BML spad. BML rebaselines
 *     whenever it sees a new epoch, so leftover state from a previous run can never be mistaken
 *     for fresh, regardless of which chip booted first.
 *   - `case_index` is monotonic within an epoch and is written LAST on each publish -> it is the
 *     commit signal. `payload_checksum` guards a torn/partial payload.
 *   - BML acks by writing `epoch_echo` + `ack_index` into the DSP spad. DSP considers a case
 *     delivered only when `epoch_echo == epoch` AND `ack_index >= case_index`; otherwise it
 *     re-publishes. This self-heals against unstable cross-link writes and dropped acks.
 *
 * All accesses go through c2c_shm.h helpers (repeat remote writes; flush-first local reads).
 */

#include <stdint.h>
#include <stddef.h>   /* offsetof */

#include "kws_proto.h"   /* KWS_MFCC_DIM, KWS_FRAMES_PER_CASE, KWS_CASE_PAYLOAD_BYTES */

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_STREAM_PROTO_VERSION 2u

/* Scratchpad bases. */
#ifndef KWS_STREAM_DSP_SPAD_BASE
#define KWS_STREAM_DSP_SPAD_BASE 0xC0000000UL
#endif
#ifndef KWS_STREAM_BML_SPAD_BASE
#define KWS_STREAM_BML_SPAD_BASE 0xD0000000UL
#endif

#define KWS_STREAM_MAGIC_BML 0x4B575344u /* 'KWSD' — BML spad (DSP -> BML) */
#define KWS_STREAM_MAGIC_DSP 0x4B575343u /* 'KWSC' — DSP spad (BML -> DSP) */

/* Boot barrier: BML writes this into the DSP spad (`bml_ready`) once it is fully booted. DSP must
 * NOT write BML's spad until it sees this — a cross-link write into a still-booting chip kills it. */
#define KWS_STREAM_READY_MAGIC 0x52454459u /* 'REDY' */

/* Bytes at the top of each spad to wipe on boot (control block, excludes payload). */
#define KWS_STREAM_CONTROL_CLEAR_BYTES 0x40u

/* BML-adjacent spad @ 0xD0000000 : DSP -> BML data path (DSP remote-writes, BML local-reads).
 * Base is 64-byte aligned by address; case_payload lands at offset 0x40 (its own cache line). */
typedef struct __attribute__((packed)) {
  volatile uint32_t magic;            /* 0x00  KWS_STREAM_MAGIC_BML */
  volatile uint32_t version;          /* 0x04  KWS_STREAM_PROTO_VERSION */
  volatile uint32_t epoch;            /* 0x08  DSP per-boot nonce (nonzero) */
  volatile uint32_t payload_bytes;    /* 0x0C  = KWS_CASE_PAYLOAD_BYTES */
  volatile uint32_t payload_checksum; /* 0x10  c2c_checksum over case_payload */
  volatile uint32_t case_index;       /* 0x14  ++ per case within epoch; COMMIT (written last) */
  volatile uint64_t dsp_tx_cycle;     /* 0x18  rdcycle at commit */
  volatile uint32_t reserved[8];      /* 0x20..0x3F */
  volatile int8_t   case_payload[KWS_CASE_PAYLOAD_BYTES]; /* 0x40 */
} kws_stream_bml_spad_t;

/* DSP-adjacent spad @ 0xC0000000 : BML -> DSP ack/back-channel (BML remote-writes, DSP local-reads). */
typedef struct __attribute__((packed)) {
  volatile uint32_t magic;            /* 0x00  KWS_STREAM_MAGIC_DSP */
  volatile uint32_t epoch_echo;       /* 0x04  epoch BML is currently serving */
  volatile uint32_t ack_index;        /* 0x08  last case_index BML consumed (written last on ack) */
  volatile uint32_t bml_pred_class;   /* 0x0C  last inference class (optional) */
  volatile uint32_t bml_pred_score_q; /* 0x10  quantized score (optional) */
  volatile uint32_t bml_ready;        /* 0x14  KWS_STREAM_READY_MAGIC once BML has booted */
  volatile uint64_t bml_rx_cycle;     /* 0x18  rdcycle at ack */
  volatile uint32_t reserved[8];      /* 0x20..0x3F */
} kws_stream_dsp_spad_t;

_Static_assert(sizeof(kws_stream_bml_spad_t) == (0x40u + KWS_CASE_PAYLOAD_BYTES),
               "kws_stream_bml_spad_t layout drifted from documented offsets.");
_Static_assert(offsetof(kws_stream_bml_spad_t, case_payload) == 0x40u,
               "case_payload must sit at offset 0x40.");
_Static_assert(offsetof(kws_stream_bml_spad_t, case_index) == 0x14u,
               "case_index must sit at offset 0x14.");
_Static_assert(offsetof(kws_stream_dsp_spad_t, ack_index) == 0x08u,
               "ack_index must sit at offset 0x08.");

#ifdef __cplusplus
}
#endif

#endif /* C2C_KWS_STREAM_PROTO_H */
