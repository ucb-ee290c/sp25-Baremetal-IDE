#ifndef C2C_KWS_ROLLING_PROTO_H
#define C2C_KWS_ROLLING_PROTO_H

#include <stdint.h>

#include "kws_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef KWS_ROLLING_SHARED_BASE_ADDR
#define KWS_ROLLING_SHARED_BASE_ADDR KWS_SHARED_BASE_ADDR
#endif

#ifndef KWS_ROLLING_SHARED_BYTES
#define KWS_ROLLING_SHARED_BYTES KWS_SHARED_BYTES
#endif

#ifndef KWS_ROLLING_COMMIT_SEQ_ADDR
#define KWS_ROLLING_COMMIT_SEQ_ADDR KWS_ROLLING_SHARED_BASE_ADDR
#endif

#ifndef KWS_ROLLING_FRAME_ADDR
#define KWS_ROLLING_FRAME_ADDR (KWS_ROLLING_COMMIT_SEQ_ADDR + 4u)
#endif

#ifndef KWS_ROLLING_FRAME_BYTES
#define KWS_ROLLING_FRAME_BYTES KWS_MFCC_DIM
#endif

#define KWS_ROLLING_WINDOW_BYTES KWS_CASE_PAYLOAD_BYTES

static inline void kws_rolling_fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

#ifdef __cplusplus
}
#endif

#endif /* C2C_KWS_ROLLING_PROTO_H */
