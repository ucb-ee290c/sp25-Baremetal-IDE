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

#ifndef KWS_BEARLY_RING_SLOTS
#define KWS_BEARLY_RING_SLOTS 16384u
#endif

#ifndef KWS_BEARLY_SHM_BASE
#define KWS_BEARLY_SHM_BASE 0x8FF00000UL
#endif

#ifndef KWS_BEARLY_MAILBOX_ADDR
#define KWS_BEARLY_MAILBOX_ADDR KWS_BEARLY_SHM_BASE
#endif

#ifndef KWS_BEARLY_RING_ADDR
#define KWS_BEARLY_RING_ADDR (KWS_BEARLY_SHM_BASE + 0x1000UL)
#endif

#ifndef KWS_BEARLY_PROGRESS_EVERY
#define KWS_BEARLY_PROGRESS_EVERY 10u
#endif

#endif /* C2C_BEARLY_KWS_CONFIG_H */
