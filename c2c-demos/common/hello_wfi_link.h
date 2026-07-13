#ifndef HELLO_WFI_LINK_H
#define HELLO_WFI_LINK_H

#include <stdint.h>

/* Shared cross-link helpers for the hello-wfi interrupt ping-pong (dsp-hello-wfi / bearly-hello-wfi).
 *
 * Each including translation unit must define these BEFORE including this header (its per-chip
 * config does so):
 *   HELLO_WFI_LOCAL_SPAD_BASE   own scratchpad base (we local-read here; peer wrote it over the link)
 *   HELLO_WFI_PEER_SPAD_BASE    peer scratchpad via the cross-link (own spad addr with a leading 1)
 *   HELLO_WFI_PEER_MSIP_ADDR    peer CLINT MSIP via the cross-link (0x1_0200_0000)
 *   HELLO_WFI_OWN_MSIP_ADDR     own CLINT MSIP (0x0200_0000)
 *   HELLO_WFI_MY_TURN / _PEER_TURN   turn-register value for this chip / the peer (0=DSP, 1=Bearly)
 *   HELLO_WFI_TURN_OFFSET       byte offset of the turn register within a spad (0x00)
 *   HELLO_WFI_BATON_OFFSET      byte offset of the baton/data word within a spad (0x04)
 *
 * Hardware rules honored here (see /CLAUDE.md): the cross-link is not cache-coherent, so we flush
 * the whole cache around every cross-link touch — before every poll/read of our own spad (the peer
 * wrote it behind our cache) and after every cross-link write into the peer's spad / MSIP.
 *
 * The flush is a FORCE-EVICTION buffer walk (writing 1 to a cache-controller flush register did not
 * actually evict on this silicon): we touch one byte of every cache line across a buffer larger than
 * the cache, several passes, so every resident line is read + dirtied + pushed out. Same technique
 * as c2c_shm.c / c2c-measure.
 */

#ifndef HELLO_WFI_LOCAL_SPAD_BASE
#error "hello_wfi_link.h: define HELLO_WFI_LOCAL_SPAD_BASE (own scratchpad) before including."
#endif
#ifndef HELLO_WFI_PEER_SPAD_BASE
#error "hello_wfi_link.h: define HELLO_WFI_PEER_SPAD_BASE (cross-link peer scratchpad) first."
#endif
#ifndef HELLO_WFI_PEER_MSIP_ADDR
#error "hello_wfi_link.h: define HELLO_WFI_PEER_MSIP_ADDR (cross-link peer MSIP) first."
#endif
#ifndef HELLO_WFI_OWN_MSIP_ADDR
#error "hello_wfi_link.h: define HELLO_WFI_OWN_MSIP_ADDR (own MSIP) first."
#endif
#ifndef HELLO_WFI_MY_TURN
#error "hello_wfi_link.h: define HELLO_WFI_MY_TURN (this chip's turn-register value) first."
#endif
#ifndef HELLO_WFI_PEER_TURN
#error "hello_wfi_link.h: define HELLO_WFI_PEER_TURN (the peer's turn-register value) first."
#endif
#ifndef HELLO_WFI_TURN_OFFSET
#define HELLO_WFI_TURN_OFFSET 0x00u
#endif
#ifndef HELLO_WFI_BATON_OFFSET
#define HELLO_WFI_BATON_OFFSET 0x04u
#endif

/* Cross-link writes are UNSTABLE: a single store into the peer's spad / MSIP may not "take"
 * (see /CLAUDE.md). Repeat every cross-link write this many times (fenced) so it sticks. Both the
 * baton write and the MSIP-set are idempotent, so repeats are safe. A single wake write that drops
 * is unrecoverable here — the peer is asleep in wfi and nothing re-sends it — so bias this high. */
#ifndef HELLO_WFI_WRITE_REPEATS
#define HELLO_WFI_WRITE_REPEATS 2u
#endif

/* Safety-net timer period, in CLINT mtime ticks. Alongside the MSIP wake we arm a periodic machine
 * timer, so a chip parked in wfi re-checks its turn register at least this often even if the MSIP
 * that should have woken it was dropped. MTIME_FREQ is 50 kHz (20 us/tick) -> 2500 ticks ~= 50 ms. */
#ifndef HELLO_WFI_POLL_INTERVAL_TICKS
#define HELLO_WFI_POLL_INTERVAL_TICKS 2500ULL
#endif

/* Force-eviction geometry. EVICT_BYTES must comfortably exceed the largest cache being flushed;
 * 256 KiB / 64 B lines / 3 passes matches the other C2C demos. */
#ifndef HELLO_WFI_CACHE_LINE_BYTES
#define HELLO_WFI_CACHE_LINE_BYTES 64u
#endif
#ifndef HELLO_WFI_EVICT_BYTES
#define HELLO_WFI_EVICT_BYTES (256u * 1024u)
#endif
#ifndef HELLO_WFI_EVICT_PASSES
#define HELLO_WFI_EVICT_PASSES 3u
#endif

_Static_assert((HELLO_WFI_CACHE_LINE_BYTES & (HELLO_WFI_CACHE_LINE_BYTES - 1u)) == 0u,
               "HELLO_WFI_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((HELLO_WFI_EVICT_BYTES % HELLO_WFI_CACHE_LINE_BYTES) == 0u,
               "HELLO_WFI_EVICT_BYTES must be a multiple of the cache line size.");

static inline void hwfi_fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

/* Eviction scratch buffer. This header is included by exactly one TU per binary, so a file-scope
 * static here is a single buffer per executable. Aligned so the walk stride maps cleanly to sets. */
static uint8_t hwfi_evict_buf[HELLO_WFI_EVICT_BYTES] __attribute__((aligned(0x8000)));
static volatile uint8_t hwfi_evict_sink;

/* Full cache flush by force-eviction: read + write one byte of every cache line across a
 * larger-than-cache buffer, several passes, then fence. Mandatory around every cross-link access. */
static inline void hwfi_cache_flush(void) {
  volatile uint8_t *buf = (volatile uint8_t *)hwfi_evict_buf;
  volatile uint8_t sink = hwfi_evict_sink;

  for (uint32_t pass = 0; pass < HELLO_WFI_EVICT_PASSES; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)HELLO_WFI_EVICT_BYTES; i += HELLO_WFI_CACHE_LINE_BYTES) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    hwfi_fence_rw();
  }

  hwfi_evict_sink = sink;
  hwfi_fence_rw();
}

/* Reliable cross-link store: write val into a peer address HELLO_WFI_WRITE_REPEATS times (fenced),
 * then a single trailing flush. Repeats defeat the unstable-remote-write quirk; the single trailing
 * flush keeps the burst short so it lands before the woken peer can clear the bit (no double-wake). */
static inline void hwfi_remote_write_u32(volatile uint32_t *addr, uint32_t val) {
  for (uint32_t r = 0; r < (uint32_t)HELLO_WFI_WRITE_REPEATS; ++r) {
    *addr = val;
    hwfi_fence_rw();
  }
  hwfi_cache_flush();
}

/* Read the baton from our OWN spad (peer wrote it across the link). Flush first. */
static inline uint32_t hwfi_read_baton_local(void) {
  volatile uint32_t *baton =
      (volatile uint32_t *)(uintptr_t)(HELLO_WFI_LOCAL_SPAD_BASE + HELLO_WFI_BATON_OFFSET);
  hwfi_cache_flush();
  return *baton;
}

/* Write the baton into the PEER's spad across the link (repeated to defeat unstable writes). */
static inline void hwfi_write_baton_peer(uint32_t value) {
  volatile uint32_t *baton =
      (volatile uint32_t *)(uintptr_t)(HELLO_WFI_PEER_SPAD_BASE + HELLO_WFI_BATON_OFFSET);
  hwfi_remote_write_u32(baton, value);
}

/* Read the turn register from our OWN spad (flush first). 0 = DSP's turn, 1 = Bearly's turn. */
static inline uint32_t hwfi_read_local_turn(void) {
  volatile uint32_t *turn =
      (volatile uint32_t *)(uintptr_t)(HELLO_WFI_LOCAL_SPAD_BASE + HELLO_WFI_TURN_OFFSET);
  hwfi_cache_flush();
  return *turn;
}

/* Set the turn register in our OWN spad (local write). */
static inline void hwfi_set_local_turn(uint32_t value) {
  volatile uint32_t *turn =
      (volatile uint32_t *)(uintptr_t)(HELLO_WFI_LOCAL_SPAD_BASE + HELLO_WFI_TURN_OFFSET);
  *turn = value;
  hwfi_fence_rw();
  hwfi_cache_flush();
}

/* Set the turn register in the PEER's spad across the link (repeated to defeat unstable writes). */
static inline void hwfi_set_peer_turn(uint32_t value) {
  volatile uint32_t *turn =
      (volatile uint32_t *)(uintptr_t)(HELLO_WFI_PEER_SPAD_BASE + HELLO_WFI_TURN_OFFSET);
  hwfi_remote_write_u32(turn, value);
}

/* Boot init (local writes into our own spad — safe in app_init): clear the baton and mark the turn
 * as the PEER's, so that until the peer explicitly hands off to us, any early/spurious/timer wake
 * reads "not my turn" and goes back to sleep. Also defeats stale spad SRAM from a previous run. */
static inline void hwfi_boot_init(void) {
  volatile uint32_t *turn =
      (volatile uint32_t *)(uintptr_t)(HELLO_WFI_LOCAL_SPAD_BASE + HELLO_WFI_TURN_OFFSET);
  volatile uint32_t *baton =
      (volatile uint32_t *)(uintptr_t)(HELLO_WFI_LOCAL_SPAD_BASE + HELLO_WFI_BATON_OFFSET);
  *baton = 0u;
  *turn = (uint32_t)HELLO_WFI_PEER_TURN;
  hwfi_fence_rw();
  hwfi_cache_flush();
}

/* Read / clear our OWN CLINT MSIP (the peer set it via a cross-link write to HELLO_WFI_PEER_MSIP). */
static inline uint32_t hwfi_read_own_msip(void) {
  volatile uint32_t *msip = (volatile uint32_t *)(uintptr_t)(HELLO_WFI_OWN_MSIP_ADDR);
  hwfi_cache_flush();
  return *msip;
}

static inline void hwfi_clear_own_msip(void) {
  volatile uint32_t *msip = (volatile uint32_t *)(uintptr_t)(HELLO_WFI_OWN_MSIP_ADDR);
  *msip = 0u;
  hwfi_fence_rw();
  hwfi_cache_flush();
}

/* Wake the peer: cross-link write of 1 to its CLINT MSIP (repeated to defeat unstable writes — a
 * dropped wake is unrecoverable since the peer is asleep in wfi). */
static inline void hwfi_wake_peer(void) {
  volatile uint32_t *peer_msip = (volatile uint32_t *)(uintptr_t)(HELLO_WFI_PEER_MSIP_ADDR);
  hwfi_remote_write_u32(peer_msip, 1u);
}

/* ---- CLINT timer: safety-net wake -------------------------------------------------------------
 * Standard SiFive CLINT layout relative to CLINT_BASE (== own MSIP addr): mtimecmp @ +0x4000,
 * mtime @ +0xBFF8. RV64, so both are single 64-bit accesses. */
static inline uint64_t hwfi_mtime(void) {
  return *(volatile uint64_t *)(uintptr_t)(HELLO_WFI_OWN_MSIP_ADDR + 0xBFF8U);
}

static inline void hwfi_set_mtimecmp(uint64_t v) {
  *(volatile uint64_t *)(uintptr_t)(HELLO_WFI_OWN_MSIP_ADDR + 0x4000U) = v;
  hwfi_fence_rw();
}

/* Arm BOTH wake sources: MSIE (fast peer wake) and MTIE (periodic safety net) enabled in mie, with
 * mstatus.MIE cleared so neither is taken as a trap (no handler) — wfi just resumes and we poll. */
static inline void hwfi_arm_wake(void) {
  __asm__ volatile("csrc mstatus, %0" :: "r"(1UL << 3) : "memory");            /* mstatus.MIE = 0 */
  __asm__ volatile("csrs mie, %0" :: "r"((1UL << 3) | (1UL << 7)) : "memory"); /* MSIE + MTIE   */
}

/* Sleep in wfi until the next timer tick (interval from now) OR an MSIP wake, whichever is first.
 * Re-arming mtimecmp each call schedules the next tick and clears any pending MTIP. Clear our own
 * MSIP after waking (idempotent) so a stale pending bit can't spin us. */
static inline void hwfi_sleep_until_tick(uint64_t interval_ticks) {
  hwfi_set_mtimecmp(hwfi_mtime() + interval_ticks);
  __asm__ volatile("wfi");
  hwfi_clear_own_msip();
}

/* Wait until the turn register in our OWN spad says it is our turn. Each wake (MSIP or timer tick)
 * re-reads the turn register: if it is not yet ours we go back to sleep. The timer guarantees we
 * re-check periodically, so a dropped MSIP costs at most one interval of latency instead of a
 * deadlock — as long as the peer's turn-register write itself landed (hardened via repeats). */
static inline void hwfi_await_my_turn(void) {
  while (hwfi_read_local_turn() != (uint32_t)HELLO_WFI_MY_TURN) {
    hwfi_sleep_until_tick((uint64_t)HELLO_WFI_POLL_INTERVAL_TICKS);
  }
}

/* Hand off to the peer: publish the data (baton) into the peer's spad, then flip the turn register
 * to the peer BOTH in the peer's spad (so it may run) and in our own spad (so a later spurious/timer
 * wake of ours reads "not my turn" and sleeps), then raise the peer's MSIP. Order matters: baton
 * (data) is written and flushed before the turn register (the commit), so when the peer sees its
 * turn the data is already resident. */
static inline void hwfi_handoff(uint32_t baton_value) {
  hwfi_write_baton_peer(baton_value);
  hwfi_set_peer_turn((uint32_t)HELLO_WFI_PEER_TURN);
  hwfi_set_local_turn((uint32_t)HELLO_WFI_PEER_TURN);
  hwfi_wake_peer();
}

#endif /* HELLO_WFI_LINK_H */
