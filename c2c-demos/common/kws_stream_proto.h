#ifndef C2C_KWS_STREAM_PROTO_H
#define C2C_KWS_STREAM_PROTO_H

/*
 * kws_stream_proto — two-scratchpad protocol for the full C2C KWS streaming demo.
 *
 * Synchronization is the proven turn-taking pattern from hello-wfi (see /CLAUDE.md "Reliable C2C
 * turn-taking synchronization" and c2c-demos/common/c2c_turnsync.h): a **turn register** in each
 * spad decides whose turn it is, a CLINT timer wakes a sleeper periodically so a dropped MSIP is
 * recovered, and every cross-link write is hardened by repeats.
 *
 * Hardware access rule (see /CLAUDE.md): a chip may READ only its own adjacent spad and may WRITE
 * to both. To WRITE the peer's spad you must use the cross-link address (peer-local addr with a
 * leading 1); a local read uses the plain address.
 *
 *   0xC0000000    DSP-adjacent spad   -> DSP reads locally; BML writes it at 0x1_C000_0000
 *   0xD0000000    BML-adjacent spad   -> BML reads locally; DSP writes it at 0x1_D000_0000
 *
 * Turn register (offset 0x20 in BOTH spads): C2C_TURN_DSP (0) = DSP's turn to produce+publish the
 * next case; C2C_TURN_BML (1) = BML's turn to read+verify+infer+hand back. Each chip reads the turn
 * register from its OWN spad and acts only when it equals its own id. On handoff the sender sets the
 * turn (peer spad = commit, then own spad = "not mine") and rings the peer's MSIP.
 *
 * `case_index` is monotonic and `payload_checksum` guards a torn payload; both are written before
 * the turn register (the commit), so when the peer sees its turn the data is already resident.
 *
 * All accesses go through c2c_shm.h helpers (repeat remote writes; flush-first local reads).
 */

#include <stdint.h>
#include <stddef.h>   /* offsetof */

#include "kws_proto.h"   /* KWS_MFCC_DIM, KWS_FRAMES_PER_CASE, KWS_CASE_PAYLOAD_BYTES */

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_STREAM_PROTO_VERSION 3u   /* 3: turn-register sync (was 2: rx_ready/epoch handshake) */

/* Local scratchpad bases (used for LOCAL reads / local turn writes). */
#ifndef KWS_STREAM_DSP_SPAD_BASE
#define KWS_STREAM_DSP_SPAD_BASE 0xC0000000UL
#endif
#ifndef KWS_STREAM_BML_SPAD_BASE
#define KWS_STREAM_BML_SPAD_BASE 0xD0000000UL
#endif

/* Cross-link peer bases (used for REMOTE writes into the other chip's spad — leading 1). */
#ifndef KWS_STREAM_DSP_SPAD_PEER
#define KWS_STREAM_DSP_SPAD_PEER 0x1C0000000ULL /* BML writes DSP's spad here */
#endif
#ifndef KWS_STREAM_BML_SPAD_PEER
#define KWS_STREAM_BML_SPAD_PEER 0x1D0000000ULL /* DSP writes BML's spad here */
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
  volatile uint32_t payload_bytes;    /* 0x08  = KWS_CASE_PAYLOAD_BYTES */
  volatile uint32_t payload_checksum; /* 0x0C  c2c_checksum over case_payload */
  volatile uint32_t case_index;       /* 0x10  ++ per case; telemetry/verify */
  volatile uint32_t reserved0;        /* 0x14  pad (align dsp_tx_cycle) */
  volatile uint64_t dsp_tx_cycle;     /* 0x18  rdcycle at publish */
  volatile uint32_t turn;             /* 0x20  C2C_TURN_*: whose turn it is (the commit) */
  volatile int32_t  expected_label;   /* 0x24  ground-truth class for this case (-1 = unknown) */
  volatile int32_t  ref_case_index;   /* 0x28  matching tinyspeech_inputs.h index (-1 = none) */
  volatile uint32_t reserved1[5];     /* 0x2C..0x3F */
  volatile int8_t   case_payload[KWS_CASE_PAYLOAD_BYTES]; /* 0x40 */
} kws_stream_bml_spad_t;

/* DSP-adjacent spad @ 0xC0000000 : BML -> DSP ack/back-channel (BML remote-writes, DSP local-reads). */
typedef struct __attribute__((packed)) {
  volatile uint32_t magic;            /* 0x00  KWS_STREAM_MAGIC_DSP */
  volatile uint32_t ack_index;        /* 0x04  last case_index BML consumed */
  volatile uint32_t bml_pred_class;   /* 0x08  last inference class (optional) */
  volatile uint32_t bml_pred_score_q; /* 0x0C  quantized score (optional) */
  volatile uint32_t bml_ready;        /* 0x10  KWS_STREAM_READY_MAGIC once BML has booted */
  volatile uint32_t reserved0;        /* 0x14  pad (align bml_rx_cycle) */
  volatile uint64_t bml_rx_cycle;     /* 0x18  rdcycle at ack */
  volatile uint32_t turn;             /* 0x20  C2C_TURN_*: whose turn it is (the commit) */
  volatile uint32_t reserved1[7];     /* 0x24..0x3F */
} kws_stream_dsp_spad_t;

_Static_assert(sizeof(kws_stream_bml_spad_t) == (0x40u + KWS_CASE_PAYLOAD_BYTES),
               "kws_stream_bml_spad_t layout drifted from documented offsets.");
_Static_assert(offsetof(kws_stream_bml_spad_t, case_payload) == 0x40u,
               "case_payload must sit at offset 0x40.");
_Static_assert(offsetof(kws_stream_bml_spad_t, case_index) == 0x10u,
               "case_index must sit at offset 0x10.");
_Static_assert(offsetof(kws_stream_bml_spad_t, turn) == 0x20u,
               "bml turn register must sit at offset 0x20.");
_Static_assert(offsetof(kws_stream_bml_spad_t, expected_label) == 0x24u,
               "expected_label must sit at offset 0x24.");
_Static_assert(offsetof(kws_stream_bml_spad_t, ref_case_index) == 0x28u,
               "ref_case_index must sit at offset 0x28.");
_Static_assert(offsetof(kws_stream_dsp_spad_t, ack_index) == 0x04u,
               "ack_index must sit at offset 0x04.");
_Static_assert(offsetof(kws_stream_dsp_spad_t, bml_ready) == 0x10u,
               "bml_ready must sit at offset 0x10.");
_Static_assert(offsetof(kws_stream_dsp_spad_t, turn) == 0x20u,
               "dsp turn register must sit at offset 0x20.");
_Static_assert(sizeof(kws_stream_dsp_spad_t) == 0x40u,
               "kws_stream_dsp_spad_t control block must stay 0x40 bytes.");

#ifdef __cplusplus
}
#endif

#endif /* C2C_KWS_STREAM_PROTO_H */
