#ifndef C2C_SHM_H
#define C2C_SHM_H

/*
 * c2c_shm — shared-scratchpad access helpers for the SP25 C2C link.
 *
 * Hardware model (see /CLAUDE.md "C2C link & shared-memory model"):
 *   - Two scratchpads: 0xC0000000 is adjacent to DSP, 0xD0000000 is adjacent to BML.
 *   - A chip may READ only its OWN adjacent spad; it may WRITE to BOTH.
 *   - Cross-link (remote) writes are UNSTABLE: repeat them a few times so they stick.
 *   - Coherence is not automatic: before reading your own spad (which the remote wrote
 *     behind your cache) you must FULL-FLUSH the cache first.
 *
 * This module centralizes those rules so every demo uses one implementation. The API is
 * split by access class:
 *   - c2c_remote_write_*  : write into the OTHER chip's spad, repeated C2C_WRITE_REPEATS times.
 *   - c2c_local_write_u32 : write into your OWN spad once (e.g. clearing your own flag).
 *   - c2c_local_read_*    : read your OWN spad, flushing first.
 */

#include <stdint.h>

#ifndef NULL
#define NULL ((void *)0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Tunables (override with -D or before including) ------------------------------------ */

/* Cache size stand-in walked to force a full flush. */
#ifndef C2C_SHM_CACHE_LINE_BYTES
#define C2C_SHM_CACHE_LINE_BYTES 64u
#endif

#ifndef C2C_SHM_EVICT_BYTES
#define C2C_SHM_EVICT_BYTES (256u * 1024u)
#endif

#ifndef C2C_SHM_EVICT_PASSES
#define C2C_SHM_EVICT_PASSES 3u
#endif

/* How many times to repeat a remote (cross-link) write so it takes. Tune on silicon. */
#ifndef C2C_SHM_WRITE_REPEATS
#define C2C_SHM_WRITE_REPEATS 4u
#endif

/* How many flush+read attempts before a stable read / verified block read gives up. */
#ifndef C2C_SHM_READ_RETRIES
#define C2C_SHM_READ_RETRIES 8u
#endif

/* Canonical scratchpad bases (informational; callers pass explicit addresses). */
#ifndef C2C_SHM_DSP_SPAD_BASE
#define C2C_SHM_DSP_SPAD_BASE 0xC0000000UL /* adjacent to DSP; DSP reads, BML remote-writes */
#endif
#ifndef C2C_SHM_BML_SPAD_BASE
#define C2C_SHM_BML_SPAD_BASE 0xD0000000UL /* adjacent to BML; BML reads, DSP remote-writes */
#endif

/* ---- Primitives ------------------------------------------------------------------------- */

static inline void c2c_fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

/* Force the entire cache out by walking the internal evict buffer, then fence. */
void c2c_full_flush(void);

/* Remote (cross-link) writes: repeated C2C_SHM_WRITE_REPEATS times, then a full flush. */
void c2c_remote_write_u32(volatile uint32_t *addr, uint32_t val);
void c2c_remote_write_block(volatile void *dst, const void *src, uint32_t bytes);

/* Local write into your own spad (single write) + flush so it reaches spad memory. */
void c2c_local_write_u32(volatile uint32_t *addr, uint32_t val);

/* Local reads of your own spad: full-flush first so you see what the remote wrote. */
uint32_t c2c_local_read_u32(volatile const uint32_t *addr);

/* Flush+read repeatedly until two fresh reads agree (bounded by C2C_SHM_READ_RETRIES).
 * *stable is set to 1 if agreement was reached, 0 if it gave up (last value returned). */
uint32_t c2c_local_read_u32_stable(volatile const uint32_t *addr, int *stable);

/* Simple checksum over a byte buffer (order-sensitive). */
uint32_t c2c_checksum(const void *buf, uint32_t bytes);

/* Copy `bytes` from your own spad into `dst`, verifying against expect_checksum.
 * Retries (flush+recopy) up to C2C_SHM_READ_RETRIES. Returns 1 on match, 0 on give-up. */
int c2c_local_read_block_verify(void *dst, volatile const void *src, uint32_t bytes,
                                uint32_t expect_checksum);

#ifdef __cplusplus
}
#endif

#endif /* C2C_SHM_H */
