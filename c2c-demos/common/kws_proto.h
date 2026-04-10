#ifndef C2C_KWS_PROTO_H
#define C2C_KWS_PROTO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_PROTO_VERSION 1u

#define KWS_MFCC_DIM 12u
#define KWS_FRAMES_PER_CASE 94u

#ifndef KWS_SHARED_BASE_ADDR
#define KWS_SHARED_BASE_ADDR 0xC0000000UL
#endif

#ifndef KWS_SHARED_BYTES
#define KWS_SHARED_BYTES 16384u
#endif

#define KWS_MAILBOX_MAGIC 0x4B57534Du /* 'KWSM' */
#define KWS_FLAG_READER_READY (1u << 0)
#define KWS_FLAG_WRITER_READY (1u << 1)
#define KWS_FLAG_STREAM_DONE  (1u << 2)

typedef struct __attribute__((packed, aligned(64))) {
  volatile uint32_t magic;
  volatile uint32_t version;
  volatile uint32_t ring_slots;
  volatile uint32_t slot_bytes;

  /* Owned by producer/writer (DSP side). */
  volatile uint32_t prod_idx;
  volatile uint32_t produced_frames;
  volatile uint32_t mfcc_failures;
  volatile uint32_t last_seq;
  volatile uint32_t last_case_id;
  volatile uint32_t last_frame_idx;
  volatile uint32_t producer_wait_loops;

  /* Owned by consumer/reader (Bearly side). */
  volatile uint32_t cons_idx;
  volatile uint32_t consumed_frames;
  volatile uint32_t dropped_frames;
  volatile uint32_t seq_errors;

  /* Shared flags/diagnostics. */
  volatile uint32_t flags;
} kws_mailbox_t;

typedef struct __attribute__((packed, aligned(16))) {
  volatile uint16_t case_id;
  volatile uint8_t frame_idx;
  volatile uint8_t valid;
  volatile int8_t mfcc[KWS_MFCC_DIM];
} kws_ring_slot_t;

static inline void kws_fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

#ifdef __cplusplus
}
#endif

#endif /* C2C_KWS_PROTO_H */
