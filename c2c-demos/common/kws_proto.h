#ifndef C2C_KWS_PROTO_H
#define C2C_KWS_PROTO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_SYNC0 0xA5u
#define KWS_SYNC1 0x5Au
#define KWS_PROTO_VERSION 1u

#define KWS_MFCC_DIM 12u
#define KWS_FRAMES_PER_CASE 94u
#define KWS_PACKET_MAX_PAYLOAD KWS_MFCC_DIM

typedef enum {
  KWS_PKT_CASE_START = 1,
  KWS_PKT_FRAME = 2,
  KWS_PKT_CASE_END = 3,
  KWS_PKT_STREAM_END = 4,
} kws_packet_type_t;

typedef struct {
  uint8_t version;
  uint8_t type;
  uint32_t seq;
  uint16_t case_id;
  uint8_t frame_idx;
  uint8_t payload_len;
  int8_t payload[KWS_PACKET_MAX_PAYLOAD];
  uint32_t crc32;
} kws_packet_t;

#define KWS_MAILBOX_MAGIC 0x4B57534Du /* 'KWSM' */

typedef struct __attribute__((packed, aligned(64))) {
  volatile uint32_t magic;
  volatile uint32_t version;
  volatile uint32_t ring_slots;
  volatile uint32_t slot_bytes;

  volatile uint32_t prod_idx;
  volatile uint32_t cons_idx;

  volatile uint32_t total_packets;
  volatile uint32_t total_frames;
  volatile uint32_t dropped_frames;
  volatile uint32_t crc_errors;
  volatile uint32_t seq_errors;

  volatile uint32_t last_seq;
  volatile uint32_t last_case_id;
  volatile uint32_t last_frame_idx;
  volatile uint32_t flags;
} kws_mailbox_t;

typedef struct __attribute__((packed, aligned(16))) {
  volatile uint32_t seq;
  volatile uint16_t case_id;
  volatile uint8_t frame_idx;
  volatile uint8_t valid;
  volatile int8_t mfcc[KWS_MFCC_DIM];
} kws_ring_slot_t;

static inline void kws_fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline void kws_write_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xFFu);
  dst[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static inline void kws_write_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xFFu);
  dst[1] = (uint8_t)((v >> 8) & 0xFFu);
  dst[2] = (uint8_t)((v >> 16) & 0xFFu);
  dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static inline uint16_t kws_read_u16_le(const uint8_t *src) {
  return (uint16_t)src[0] | ((uint16_t)src[1] << 8);
}

static inline uint32_t kws_read_u32_le(const uint8_t *src) {
  return ((uint32_t)src[0]) |
         ((uint32_t)src[1] << 8) |
         ((uint32_t)src[2] << 16) |
         ((uint32_t)src[3] << 24);
}

static inline uint32_t kws_crc32(const uint8_t *data, size_t len) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= (uint32_t)data[i];
    for (uint32_t b = 0; b < 8u; ++b) {
      uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return ~crc;
}

/* Wire format sizes. */
#define KWS_WIRE_FIXED_HEADER_BYTES 12u
#define KWS_WIRE_CRC_BYTES 4u
#define KWS_WIRE_MAX_PACKET_BYTES \
  (KWS_WIRE_FIXED_HEADER_BYTES + KWS_PACKET_MAX_PAYLOAD + KWS_WIRE_CRC_BYTES)

#ifdef __cplusplus
}
#endif

#endif /* C2C_KWS_PROTO_H */
