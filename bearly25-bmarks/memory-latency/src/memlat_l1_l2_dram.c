/*
 * memlat_l1_l2_dram.c - L1 hit, L2 local/remote hit, and DRAM latency tests.
 */

#include <stdint.h>
#include "memlat_core.h"
#include "memlat_config.h"
#include "memlat_addrs.h"

static uint8_t dram_pool[DRAM_POOL_BYTES] __attribute__((aligned(128)));
static uintptr_t node_addrs[MAX_NODES];


void memlat_test_l1_hit(void) {
    uintptr_t base = (uintptr_t)&dram_pool[0];

    for (uint32_t i = 0; i < L1_NUM_NODES; i++) {
        node_addrs[i] = base + (uintptr_t)i * CACHE_LINE_BYTES;
    }

    uintptr_t start = memlat_build_chase_ring(node_addrs, L1_NUM_NODES, 42);
    memlat_run_test("L1 Hit", start, L1_NUM_NODES);
}


void memlat_test_l2_local_hit(void) {
    // Align to 128 B so that base[6] = 0 ==>  base+64 has bit 6 = 1
    uintptr_t base = ((uintptr_t)&dram_pool[0] + 127u) & ~(uintptr_t)127u;

    for (uint32_t i = 0; i < L2_NUM_NODES; i++) {
        // base + 64 + i*128  →  bit 6 is always 1 (Bank 1, local)
        node_addrs[i] = base + CACHE_LINE_BYTES
                             + (uintptr_t)i * (2u * CACHE_LINE_BYTES);
    }

    uintptr_t start = memlat_build_chase_ring(node_addrs, L2_NUM_NODES, 123);
    memlat_run_test("L2 Local Hit", start, L2_NUM_NODES);
}

void memlat_test_l2_remote_hit(void) {
    uintptr_t raw = (uintptr_t)&dram_pool[DRAM_POOL_BYTES / 2];
    uintptr_t base = (raw + 127u) & ~(uintptr_t)127u;

    for (uint32_t i = 0; i < L2_NUM_NODES; i++) {
        // base + i*128  ==>  bit 6 is always 0 (Bank 0, remote)
        node_addrs[i] = base + (uintptr_t)i * (2u * CACHE_LINE_BYTES);
    }

    uintptr_t start = memlat_build_chase_ring(node_addrs, L2_NUM_NODES, 456);
    memlat_run_test("L2 Remote Hit", start, L2_NUM_NODES);
}

void memlat_test_dram(void) {
    uintptr_t base = (uintptr_t)&dram_pool[0];

    for (uint32_t i = 0; i < DRAM_NUM_NODES; i++) {
        node_addrs[i] = base + (uintptr_t)i * CACHE_LINE_BYTES;
    }

    uintptr_t start = memlat_build_chase_ring(node_addrs, DRAM_NUM_NODES, 789);
    memlat_run_test("DRAM", start, DRAM_NUM_NODES);
}
