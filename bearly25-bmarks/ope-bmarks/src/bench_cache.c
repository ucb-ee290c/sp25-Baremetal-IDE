/*
 * bench_cache.c - Cache thrash helpers for OPE benchmark cold runs.
 *
 * Initializes and touches a buffer larger than L2 to evict cached data.
 */
#include "bench_cache.h"

static uint8_t cache_thrash_buf[OPE_L2_BYTES * 2];

void bench_cache_init(void) {
  for (size_t i = 0; i < sizeof(cache_thrash_buf); ++i) {
    cache_thrash_buf[i] = (uint8_t)(i & 0xFFu);
  }
}

void bench_cache_flush(void) {
  volatile uint8_t *p = (volatile uint8_t *)cache_thrash_buf;
  for (size_t i = 0; i < sizeof(cache_thrash_buf); ++i) {
    p[i] ^= 0xAAu;
  }
}
