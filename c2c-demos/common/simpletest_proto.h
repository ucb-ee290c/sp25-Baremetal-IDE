#ifndef C2C_SIMPLETEST_PROTO_H
#define C2C_SIMPLETEST_PROTO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIMPLETEST_PROTO_VERSION 1u

#ifndef SIMPLETEST_SHARED_BASE_ADDR
#define SIMPLETEST_SHARED_BASE_ADDR 0xC0000000UL
#endif

#ifndef SIMPLETEST_SHARED_BYTES
#define SIMPLETEST_SHARED_BYTES 16384u
#endif

#define SIMPLETEST_MAILBOX_MAGIC 0x53544D42u /* 'STMB' */

#define SIMPLETEST_FLAG_WRITER_READY (1u << 0)
#define SIMPLETEST_FLAG_STREAM_DONE  (1u << 1)

#define SIMPLETEST_MSG_MAX_BYTES 64u

typedef struct __attribute__((packed, aligned(64))) {
  volatile uint32_t magic;
  volatile uint32_t version;
  volatile uint32_t flags;
  volatile uint32_t ring_slots;
  volatile uint32_t slot_bytes;
  volatile uint32_t produced_msgs;
  volatile uint32_t last_seq;
  volatile uint64_t last_tx_cycle;
} simpletest_mailbox_t;

typedef struct __attribute__((packed, aligned(16))) {
  volatile uint32_t commit_seq; /* 0 while producer is filling; N once message N is committed. */
  volatile uint32_t msg_id;
  volatile uint16_t msg_len;
  volatile uint16_t reserved;
  volatile uint64_t tx_cycle;
  volatile char msg[SIMPLETEST_MSG_MAX_BYTES];
} simpletest_slot_t;

static inline void simpletest_fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

#ifdef __cplusplus
}
#endif

#endif /* C2C_SIMPLETEST_PROTO_H */
