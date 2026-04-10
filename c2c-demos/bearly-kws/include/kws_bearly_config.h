#ifndef C2C_BEARLY_KWS_CONFIG_H
#define C2C_BEARLY_KWS_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "kws_proto.h"

#ifndef KWS_BEARLY_LOG_ENABLE
#define KWS_BEARLY_LOG_ENABLE 1
#endif

#if KWS_BEARLY_LOG_ENABLE
#define KWS_BEARLY_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define KWS_BEARLY_LOG(...) do { } while (0)
#endif

#ifndef KWS_BEARLY_TARGET_FREQUENCY_HZ
#define KWS_BEARLY_TARGET_FREQUENCY_HZ 500000000ULL
#endif

#ifndef KWS_BEARLY_SHM_BASE
#define KWS_BEARLY_SHM_BASE KWS_SHARED_BASE_ADDR
#endif

#ifndef KWS_BEARLY_SHM_BYTES
#define KWS_BEARLY_SHM_BYTES KWS_SHARED_BYTES
#endif

#ifndef KWS_BEARLY_MAILBOX_ADDR
#define KWS_BEARLY_MAILBOX_ADDR KWS_BEARLY_SHM_BASE
#endif

#ifndef KWS_BEARLY_RING_ADDR
#define KWS_BEARLY_RING_ADDR (KWS_BEARLY_MAILBOX_ADDR + sizeof(kws_mailbox_t))
#endif

#ifndef KWS_BEARLY_RING_BYTES
#define KWS_BEARLY_RING_BYTES (KWS_BEARLY_SHM_BYTES - sizeof(kws_mailbox_t))
#endif

#ifndef KWS_BEARLY_RING_SLOTS
#define KWS_BEARLY_RING_SLOTS (KWS_BEARLY_RING_BYTES / sizeof(kws_ring_slot_t))
#endif

#ifndef KWS_BEARLY_EXPECTED_MODE
#define KWS_BEARLY_EXPECTED_MODE 0xFFFFFFFFu /* Accept writer-selected mode by default. */
#endif

#ifndef KWS_BEARLY_CACHE_LINE_BYTES
#define KWS_BEARLY_CACHE_LINE_BYTES 64u
#endif

#ifndef KWS_BEARLY_CACHE_EVICT_BYTES
#define KWS_BEARLY_CACHE_EVICT_BYTES (256u * 1024u)
#endif

/*
 * Run one full software cache-eviction sweep every N mailbox polls.
 * 1 = sweep every poll (strongest coherence, highest overhead).
 */
#ifndef KWS_BEARLY_MAILBOX_EVICT_EVERY
#define KWS_BEARLY_MAILBOX_EVICT_EVERY 1u
#endif

#ifndef KWS_BEARLY_WAIT_LOG_EVERY
#define KWS_BEARLY_WAIT_LOG_EVERY 200000u
#endif

#ifndef KWS_BEARLY_CLEAR_SHM_ON_BOOT
#define KWS_BEARLY_CLEAR_SHM_ON_BOOT 1u
#endif

#ifndef KWS_BEARLY_PROGRESS_EVERY
#define KWS_BEARLY_PROGRESS_EVERY 10u
#endif

#ifndef KWS_BEARLY_USE_THREADLIB
#define KWS_BEARLY_USE_THREADLIB 0
#endif

#endif /* C2C_BEARLY_KWS_CONFIG_H */
