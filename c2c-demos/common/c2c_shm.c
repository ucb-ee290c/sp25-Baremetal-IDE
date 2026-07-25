#include "c2c_shm.h"

_Static_assert((C2C_SHM_CACHE_LINE_BYTES & (C2C_SHM_CACHE_LINE_BYTES - 1u)) == 0u,
               "C2C_SHM_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((C2C_SHM_EVICT_BYTES >= C2C_SHM_CACHE_LINE_BYTES),
               "C2C_SHM_EVICT_BYTES must be at least one cache line.");
_Static_assert((C2C_SHM_EVICT_BYTES % C2C_SHM_CACHE_LINE_BYTES) == 0u,
               "C2C_SHM_EVICT_BYTES must be a multiple of cache line size.");
_Static_assert((C2C_SHM_WRITE_REPEATS >= 1u), "C2C_SHM_WRITE_REPEATS must be >= 1.");
_Static_assert((C2C_SHM_READ_RETRIES >= 1u), "C2C_SHM_READ_RETRIES must be >= 1.");

static uint8_t g_c2c_evict[C2C_SHM_EVICT_BYTES] __attribute__((aligned(0x8000)));
static volatile uint8_t g_c2c_evict_sink;

void c2c_full_flush(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_c2c_evict;
  volatile uint8_t sink = g_c2c_evict_sink;

  for (uint32_t pass = 0; pass < C2C_SHM_EVICT_PASSES; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)C2C_SHM_EVICT_BYTES; i += C2C_SHM_CACHE_LINE_BYTES) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    c2c_fence_rw();
  }

  g_c2c_evict_sink = sink;
  c2c_fence_rw();
}

void c2c_remote_write_u32(volatile uint32_t *addr, uint32_t val) {
  for (uint32_t r = 0; r < C2C_SHM_WRITE_REPEATS; ++r) {
    *addr = val;
    c2c_fence_rw();
  }
  c2c_full_flush();
}

/* Spads are 32-bit-access-only: byte/half stores can trap or hang. Callers pass a 4-aligned dst
 * and a 4-byte-multiple length; we always write whole 32-bit words. `src` may be unaligned, so we
 * assemble each word from bytes (little-endian, matching the RISC-V cores on both sides). */
void c2c_remote_write_block(volatile void *dst, const void *src, uint32_t bytes) {
  volatile uint32_t *d = (volatile uint32_t *)dst;
  const uint8_t *s = (const uint8_t *)src;
  uint32_t words = bytes >> 2;

  for (uint32_t r = 0; r < C2C_SHM_WRITE_REPEATS; ++r) {
    for (uint32_t w = 0; w < words; ++w) {
      const uint8_t *b = &s[w << 2];
      uint32_t val = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
                     ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
      d[w] = val;
    }
    c2c_fence_rw();
  }
  c2c_full_flush();
}

void c2c_local_write_u32(volatile uint32_t *addr, uint32_t val) {
  *addr = val;
  c2c_fence_rw();
  c2c_full_flush();
}

uint32_t c2c_local_read_u32(volatile const uint32_t *addr) {
  c2c_full_flush();
  return *addr;
}

uint32_t c2c_local_read_u32_stable(volatile const uint32_t *addr, int *stable) {
  uint32_t prev = c2c_local_read_u32(addr);

  for (uint32_t r = 1; r < C2C_SHM_READ_RETRIES; ++r) {
    uint32_t cur = c2c_local_read_u32(addr);
    if (cur == prev) {
      if (stable != NULL) {
        *stable = 1;
      }
      return cur;
    }
    prev = cur;
  }

  if (stable != NULL) {
    *stable = 0;
  }
  return prev;
}

uint32_t c2c_checksum(const void *buf, uint32_t bytes) {
  const uint8_t *b = (const uint8_t *)buf;
  uint32_t acc = 0x811C9DC5u; /* FNV-1a offset basis */

  for (uint32_t i = 0; i < bytes; ++i) {
    acc ^= (uint32_t)b[i];
    acc *= 0x01000193u; /* FNV prime */
  }
  return acc;
}

/* Word-granular read from your own spad (32-bit-access-only). `dst` is local RAM (any alignment);
 * `src` is 4-aligned spad and `bytes` is a 4-byte multiple. */
int c2c_local_read_block_verify(void *dst, volatile const void *src, uint32_t bytes,
                                uint32_t expect_checksum) {
  volatile const uint32_t *s = (volatile const uint32_t *)src;
  uint8_t *d = (uint8_t *)dst;
  uint32_t words = bytes >> 2;

  for (uint32_t r = 0; r < C2C_SHM_READ_RETRIES; ++r) {
    c2c_full_flush();
    for (uint32_t w = 0; w < words; ++w) {
      uint32_t val = s[w];
      uint8_t *b = &d[w << 2];
      b[0] = (uint8_t)val;
      b[1] = (uint8_t)(val >> 8);
      b[2] = (uint8_t)(val >> 16);
      b[3] = (uint8_t)(val >> 24);
    }
    c2c_fence_rw();
    if (c2c_checksum(d, bytes) == expect_checksum) {
      return 1;
    }
  }
  return 0;
}
