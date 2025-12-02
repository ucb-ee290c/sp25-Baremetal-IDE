#ifndef MEMLAT_PATTERNS_H
#define MEMLAT_PATTERNS_H

#include <stdint.h>
#include "memlat_config.h"

// Sweep over region [base, base + bytes) with fixed stride in bytes
void memlat_stream_region_32b(volatile uint32_t *base, uint32_t bytes, uint32_t stride_bytes);

// Warm up a single cacheline around addr by repeatedly hitting it
static inline void memlat_warm_single_line(volatile uint32_t *addr, uint32_t iters)
{
    volatile uint32_t tmp = 0;
    for (uint32_t i = 0; i < iters; ++i) {
        tmp ^= *addr;
    }
    // Prevent the compiler from optimizing tmp away
    asm volatile("" :: "r"(tmp) : "memory");
}

#endif // MEMLAT_PATTERNS_H
