#ifndef C2C_DSP_KWS_CONFIG_H
#define C2C_DSP_KWS_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "kws_proto.h"

#ifndef KWS_DSP_LOG_ENABLE
#define KWS_DSP_LOG_ENABLE 0
#endif

#if KWS_DSP_LOG_ENABLE
#define KWS_DSP_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define KWS_DSP_LOG(...) do { } while (0)
#endif

#ifndef KWS_DSP_TARGET_FREQUENCY_HZ
#define KWS_DSP_TARGET_FREQUENCY_HZ 500000000ULL
#endif

#ifndef KWS_DSP_SHARED_BASE
#define KWS_DSP_SHARED_BASE KWS_SHARED_BASE_ADDR
#endif

#ifndef KWS_DSP_SHARED_BYTES
#define KWS_DSP_SHARED_BYTES KWS_SHARED_BYTES
#endif

#ifndef KWS_DSP_REMOTE_MAILBOX_ADDR
#define KWS_DSP_REMOTE_MAILBOX_ADDR KWS_DSP_SHARED_BASE
#endif

#ifndef KWS_DSP_REMOTE_RING_ADDR
#define KWS_DSP_REMOTE_RING_ADDR (KWS_DSP_REMOTE_MAILBOX_ADDR + sizeof(kws_mailbox_t))
#endif

#ifndef KWS_DSP_REMOTE_RING_BYTES
#define KWS_DSP_REMOTE_RING_BYTES (KWS_DSP_SHARED_BYTES - sizeof(kws_mailbox_t))
#endif

#ifndef KWS_DSP_LINK_MODE
#define KWS_DSP_LINK_MODE KWS_LINK_MODE_SAFE
#endif

#ifndef KWS_DSP_NUM_CASES
#define KWS_DSP_NUM_CASES 100u
#endif

#ifndef KWS_DSP_FRAMES_PER_CASE
#define KWS_DSP_FRAMES_PER_CASE KWS_FRAMES_PER_CASE
#endif

#ifndef KWS_DSP_MFCC_QUANT_SCALE
#define KWS_DSP_MFCC_QUANT_SCALE 4.0f
#endif

#ifndef KWS_DSP_MFCC_QUANT_ZERO
#define KWS_DSP_MFCC_QUANT_ZERO 0.0f
#endif

#ifndef KWS_DSP_PROGRESS_EVERY_CASES
#define KWS_DSP_PROGRESS_EVERY_CASES 10u
#endif

#ifndef KWS_DSP_USE_THREADLIB
#define KWS_DSP_USE_THREADLIB 0
#endif

#endif /* C2C_DSP_KWS_CONFIG_H */
