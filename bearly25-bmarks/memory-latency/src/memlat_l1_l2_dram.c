#include <stdio.h>
#include "memlat_tests.h"
#include "memlat_patterns.h"
#include "memlat_addrs.h"
#include "memlat_config.h"

/*
 * This file covers:
 *   - L1 hit latency (using a DRAM-backed line that we warm)
 *   - DRAM cold-miss latency (by thrashing a big DRAM region)
 *   - Stubs for L2 local / remote tests (we'll fill in once
 *     we decide how to steer addresses to specific banks).
 */

// Buffer that lives in DRAM
static volatile uint32_t dram_region[MEMLAT_DRAM_REGION_BYTES / sizeof(uint32_t)]
    __attribute__((aligned(64)));

void memlat_run_l1_hit_test(int core_id)
{
    uint64_t samples[MEMLAT_NUM_SAMPLES + MEMLAT_WARMUP_SAMPLES];
    volatile uint32_t *addr = &dram_region[0];

    // Make sure location is in L1 by warming it repeatedly
    memlat_warm_single_line(addr, MEMLAT_L1_SIZE_BYTES / sizeof(uint32_t));

    uint64_t prev_mstatus = memlat_disable_irqs();
    for (uint32_t i = 0; i < MEMLAT_NUM_SAMPLES + MEMLAT_WARMUP_SAMPLES; ++i) {
        uint64_t cyc_per_load = memlat_measure_dep_chain(addr, MEMLAT_DEP_CHAIN_ITERS);
        samples[i] = cyc_per_load;
    }
    memlat_restore_irqs(prev_mstatus);

    uint64_t kept[MEMLAT_NUM_SAMPLES];
    for (uint32_t i = 0; i < MEMLAT_NUM_SAMPLES; ++i) {
        kept[i] = samples[i + MEMLAT_WARMUP_SAMPLES];
    }

    memlat_stats_t stats;
    memlat_compute_stats(kept, MEMLAT_NUM_SAMPLES, &stats);
    memlat_print_stats_line(core_id, "L1", "hit", &stats);
}

// Force cold DRAM by streaming through the whole region between sample
void memlat_run_dram_cold_miss_test(int core_id)
{
    uint64_t samples[MEMLAT_NUM_SAMPLES + MEMLAT_WARMUP_SAMPLES];
    const uint32_t stride_bytes = 64; // one cacheline per step
    volatile uint32_t *probe_addr = &dram_region[0];
    
    uint64_t prev_mstatus = memlat_disable_irqs();
    for (uint32_t i = 0; i < MEMLAT_NUM_SAMPLES + MEMLAT_WARMUP_SAMPLES; ++i) {
        // Thrash caches so probe_addr will not be present
        memlat_stream_region_32b(dram_region, MEMLAT_DRAM_REGION_BYTES, stride_bytes);
        uint64_t cyc = memlat_measure_single_load(probe_addr);
        samples[i] = cyc;
    }
    memlat_restore_irqs(prev_mstatus);

    uint64_t kept[MEMLAT_NUM_SAMPLES];
    for (uint32_t i = 0; i < MEMLAT_NUM_SAMPLES; ++i) {
        kept[i] = samples[i + MEMLAT_WARMUP_SAMPLES];
    }

    memlat_stats_t stats;
    memlat_compute_stats(kept, MEMLAT_NUM_SAMPLES, &stats);
    memlat_print_stats_line(core_id, "DRAM", "cold_miss", &stats);
}

void memlat_run_l2_local_hit_test(int core_id)
{
    (void)core_id;
    printf("L2 local hit test not implemented yet\n");
}

void memlat_run_l2_remote_hit_test(int core_id)
{
    (void)core_id;
    printf("L2 remote hit test not implemented yet\n");
}
