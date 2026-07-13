#ifndef C2C_BEARLY_KWS_ROLLING_CONFIG_H
#define C2C_BEARLY_KWS_ROLLING_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "kws_rolling_proto.h"
#include "kws_stream_proto.h"
#include "c2c_shm.h"

#ifndef KWS_BEARLY_ROLLING_LOG_ENABLE
#define KWS_BEARLY_ROLLING_LOG_ENABLE 1
#endif

#if KWS_BEARLY_ROLLING_LOG_ENABLE
#define KWS_BEARLY_ROLLING_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define KWS_BEARLY_ROLLING_LOG(...) do { } while (0)
#endif

#ifndef KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ
#define KWS_BEARLY_ROLLING_TARGET_FREQUENCY_HZ 500000000ULL
#endif

#ifndef KWS_BEARLY_ROLLING_SHM_BASE
#define KWS_BEARLY_ROLLING_SHM_BASE KWS_ROLLING_SHARED_BASE_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_SHM_BYTES
#define KWS_BEARLY_ROLLING_SHM_BYTES KWS_ROLLING_SHARED_BYTES
#endif

#ifndef KWS_BEARLY_ROLLING_COMMIT_SEQ_ADDR
#define KWS_BEARLY_ROLLING_COMMIT_SEQ_ADDR KWS_ROLLING_COMMIT_SEQ_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_FRAME_ADDR
#define KWS_BEARLY_ROLLING_FRAME_ADDR KWS_ROLLING_FRAME_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_WAIT_LOG_EVERY
#define KWS_BEARLY_ROLLING_WAIT_LOG_EVERY 200000u
#endif

#ifndef KWS_BEARLY_ROLLING_RX_LOG_EVERY
#define KWS_BEARLY_ROLLING_RX_LOG_EVERY 1u
#endif

/* Re-ack an already-consumed case (covers a dropped ack) at most once per this many polls, so BML
 * does not continuously write 0xC — continuous cross-link writes block the peer from booting. */
#ifndef KWS_BEARLY_ROLLING_REACK_EVERY
#define KWS_BEARLY_ROLLING_REACK_EVERY 20000u
#endif

/* Turn-taking park window (core cycles). After arming (rx_ready -> DSP), BML spins quietly for this
 * long with NO 0xD/link access so DSP can write the next case uncontended, then reads its own spad.
 * Must exceed DSP's detect latency + its (possibly full-payload) write time. Generous by default for
 * bring-up reliability; tune DOWN once the link is proven stable to raise throughput. */
#ifndef KWS_BEARLY_ROLLING_RX_PARK_CYCLES
#define KWS_BEARLY_ROLLING_RX_PARK_CYCLES 50000000ULL
#endif

/* After booting, wait this many cycles before the first cross-link write (bml_ready -> 0xC), so
 * DSP has time to boot too. Writing a peer's spad while it is still booting kills it. */
#ifndef KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES
#define KWS_BEARLY_ROLLING_STARTUP_GRACE_CYCLES 1000000000ULL
#endif

#ifndef KWS_BEARLY_ROLLING_INFER_LOG_EVERY
#define KWS_BEARLY_ROLLING_INFER_LOG_EVERY 1u
#endif

#ifndef KWS_BEARLY_ROLLING_PRINT_LAYER_CYCLES
#define KWS_BEARLY_ROLLING_PRINT_LAYER_CYCLES 1u
#endif

#ifndef KWS_BEARLY_ROLLING_CLEAR_SHM_ON_BOOT
#define KWS_BEARLY_ROLLING_CLEAR_SHM_ON_BOOT 0u
#endif

#ifndef KWS_BEARLY_ROLLING_DONE_ADDR
#define KWS_BEARLY_ROLLING_DONE_ADDR KWS_ROLLING_BEARLY_DONE_ADDR
#endif

#ifndef KWS_BEARLY_ROLLING_CACHE_LINE_BYTES
#define KWS_BEARLY_ROLLING_CACHE_LINE_BYTES 64u
#endif

#ifndef KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES
#define KWS_BEARLY_ROLLING_CACHE_EVICT_BYTES (256u * 1024u)
#endif

#ifndef KWS_BEARLY_ROLLING_USE_THREADLIB
#define KWS_BEARLY_ROLLING_USE_THREADLIB 1
#endif

#ifndef KWS_BEARLY_ROLLING_TCM_BASE
#define KWS_BEARLY_ROLLING_TCM_BASE TCM_BASE
#endif

#ifndef KWS_BEARLY_ROLLING_TCM_BYTES
#define KWS_BEARLY_ROLLING_TCM_BYTES 0x2000u
#endif

#ifndef KWS_BEARLY_ROLLING_TCM_WINDOW_OFFSET
#define KWS_BEARLY_ROLLING_TCM_WINDOW_OFFSET 0x1000u
#endif

#define KWS_BEARLY_ROLLING_TCM_WINDOW_ADDR \
  (KWS_BEARLY_ROLLING_TCM_BASE + KWS_BEARLY_ROLLING_TCM_WINDOW_OFFSET)

#endif /* C2C_BEARLY_KWS_ROLLING_CONFIG_H */
