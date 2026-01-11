/*
 * memlat_patterns.c - Streaming access patterns for cache/DRAM conditioning.
 */
#include "memlat_patterns.h"

// Walk a region on with a fixed stride
void memlat_stream_region_32b(volatile uint32_t *base,
                              uint32_t bytes,
                              uint32_t stride_bytes)
{
    if (stride_bytes < sizeof(uint32_t)) {
        stride_bytes = sizeof(uint32_t);
    }

    uint32_t steps = bytes / stride_bytes;
    volatile uint32_t *addr = base;
    volatile uint32_t acc = 1;

    for (uint32_t i = 0; i < steps; ++i) {
        acc += *addr;
        addr = (volatile uint32_t *)((uintptr_t)addr + stride_bytes);
    }

    asm volatile("" :: "r"(acc) : "memory");
}
