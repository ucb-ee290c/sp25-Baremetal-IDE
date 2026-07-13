#ifndef DSP_HELLO_WFI_CONFIG_H
#define DSP_HELLO_WFI_CONFIG_H

#include <stdio.h>

#include "chip_config.h"

#ifndef HELLO_WFI_LOG_ENABLE
#define HELLO_WFI_LOG_ENABLE 1
#endif

#if HELLO_WFI_LOG_ENABLE
#define HELLO_WFI_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define HELLO_WFI_LOG(...) do { } while (0)
#endif

#ifndef HELLO_WFI_TARGET_FREQUENCY_HZ
#define HELLO_WFI_TARGET_FREQUENCY_HZ 500000000ULL
#endif

/* ---- Cross-link address map (DSP side) --------------------------------------------------------
 * DSP's local scratchpad is 0xC000_0000. Its peer is BML, whose spad is 0xD000_0000, reached over
 * the link by prepending a leading 1 -> 0x1_D000_0000. The peer's CLINT MSIP is reached the same
 * way (own MSIP 0x0200_0000 -> 0x1_0200_0000). */

/* Own scratchpad (we local-read the baton here). */
#ifndef HELLO_WFI_LOCAL_SPAD_BASE
#define HELLO_WFI_LOCAL_SPAD_BASE 0xC0000000ULL
#endif

/* Peer (BML) scratchpad across the link (0xD000_0000 with a leading 1). */
#ifndef HELLO_WFI_PEER_SPAD_BASE
#define HELLO_WFI_PEER_SPAD_BASE 0x1D0000000ULL
#endif

/* Peer CLINT MSIP across the link (own MSIP with a leading 1). */
#ifndef HELLO_WFI_PEER_MSIP_ADDR
#define HELLO_WFI_PEER_MSIP_ADDR 0x102000000ULL
#endif

/* Own CLINT MSIP (hart 0, CLINT offset 0). */
#ifndef HELLO_WFI_OWN_MSIP_ADDR
#define HELLO_WFI_OWN_MSIP_ADDR (CLINT_BASE + 0x0000U)
#endif

/* Cache flush is force-eviction (buffer walk) in hello_wfi_link.h — writing the cache-controller
 * flush register did not evict on this silicon. Override HELLO_WFI_EVICT_BYTES / _PASSES /
 * _CACHE_LINE_BYTES here if the flush needs to be stronger. */

/* Turn register (spad offset 0x00): 0 = DSP's turn, 1 = Bearly's turn. Each chip reads it from its
 * OWN spad to decide whether a wake is really for it. This chip is DSP, so its turn value is 0. */
#ifndef HELLO_WFI_MY_TURN
#define HELLO_WFI_MY_TURN 0u   /* DSP */
#endif
#ifndef HELLO_WFI_PEER_TURN
#define HELLO_WFI_PEER_TURN 1u /* Bearly */
#endif

/* Byte offsets within a scratchpad: turn register at 0x00, baton (data) at 0x04. */
#ifndef HELLO_WFI_TURN_OFFSET
#define HELLO_WFI_TURN_OFFSET 0x00u
#endif
#ifndef HELLO_WFI_BATON_OFFSET
#define HELLO_WFI_BATON_OFFSET 0x04u
#endif

#endif /* DSP_HELLO_WFI_CONFIG_H */
