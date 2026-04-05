#include "bench_cache.h"

#define MC_HARTS 2u

static uint8_t cache_thrash_buf[MC_HARTS][MFCC_BENCH_L2_BYTES * 2u];

void bench_cache_init(void) {
  for (uint32_t h = 0; h < MC_HARTS; h++) {
    for (size_t i = 0; i < sizeof(cache_thrash_buf[0]); ++i) {
      cache_thrash_buf[h][i] = (uint8_t)(i & 0xFFu);
    }
  }
}

void bench_cache_flush(uint32_t hart_id) {
  volatile uint8_t *p = (volatile uint8_t *)cache_thrash_buf[hart_id];
  for (size_t i = 0; i < sizeof(cache_thrash_buf[0]); ++i) {
    p[i] ^= 0xA5u;
  }
}
