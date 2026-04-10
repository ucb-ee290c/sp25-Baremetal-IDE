#ifndef C2C_TRANSFER_PROTO_H
#define C2C_TRANSFER_PROTO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRANSFER_PROTO_VERSION 1u

#ifndef TRANSFER_SHARED_BASE_ADDR
#define TRANSFER_SHARED_BASE_ADDR 0xC0000000UL
#endif

#ifndef TRANSFER_SHARED_BYTES
#define TRANSFER_SHARED_BYTES 16384u
#endif

#define TRANSFER_MAILBOX_MAGIC 0x54524652u /* 'TRFR' */

/* Single-address protocol: DSP repeatedly overwrites this word with tx rdcycle. */
typedef volatile uint64_t transfer_cycle_word_t;

static inline void transfer_fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

#ifdef __cplusplus
}
#endif

#endif /* C2C_TRANSFER_PROTO_H */
