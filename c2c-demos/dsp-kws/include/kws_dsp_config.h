#ifndef C2C_DSP_KWS_CONFIG_H
#define C2C_DSP_KWS_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "kws_proto.h"

#ifndef KWS_DSP_LOG_ENABLE
#define KWS_DSP_LOG_ENABLE 1
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

/* Simple marker-based synchronization (no mailbox/ring). */
#ifndef KWS_DSP_SIMPLE_MARKER_ADDR
#define KWS_DSP_SIMPLE_MARKER_ADDR KWS_DSP_SHARED_BASE
#endif

#ifndef KWS_DSP_SIMPLE_PAYLOAD_ADDR
#define KWS_DSP_SIMPLE_PAYLOAD_ADDR (KWS_DSP_SIMPLE_MARKER_ADDR + 4u)
#endif

#ifndef KWS_DSP_SIMPLE_MARKER_VALUE
#define KWS_DSP_SIMPLE_MARKER_VALUE 0xFFFFFFFFu
#endif

#ifndef KWS_DSP_SIMPLE_NUM_CASES
#define KWS_DSP_SIMPLE_NUM_CASES 1u
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

#ifndef KWS_DSP_SIGNAL_HOP_SAMPLES
#define KWS_DSP_SIGNAL_HOP_SAMPLES 160u
#endif

#ifndef KWS_DSP_USE_THREADLIB
#define KWS_DSP_USE_THREADLIB 0
#endif

#ifndef KWS_DSP_CACHE_LINE_BYTES
#define KWS_DSP_CACHE_LINE_BYTES 64u
#endif

#ifndef KWS_DSP_CACHE_EVICT_BYTES
#define KWS_DSP_CACHE_EVICT_BYTES (256u * 1024u)
#endif

#endif /* C2C_DSP_KWS_CONFIG_H */
