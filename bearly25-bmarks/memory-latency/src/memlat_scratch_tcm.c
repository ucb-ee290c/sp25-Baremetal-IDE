#include <stdio.h>
#include "memlat_tests.h"
#include "memlat_patterns.h"
#include "memlat_addrs.h"
#include "memlat_config.h"

void memlat_run_scratchpad_hit_test(int core_id)
{
    uint64_t samples[MEMLAT_NUM_SAMPLES + MEMLAT_WARMUP_SAMPLES];

    // First word in scratchpad and set it to a constant
    volatile uint32_t *addr = SCRATCHPAD_PTR(0);
    *addr = 0xCAFEF00D;

    memlat_warm_single_line(addr, 128);
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
    memlat_print_stats_line(core_id, "Scratchpad", "hit", &stats);
}

void memlat_run_tcm_hit_test(int core_id)
{
    uint64_t samples[MEMLAT_NUM_SAMPLES + MEMLAT_WARMUP_SAMPLES];
    volatile uint32_t *addr = TCM_PTR(0);
    *addr = 0xF00DF00D;

    memlat_warm_single_line(addr, 128);
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
    memlat_print_stats_line(core_id, "TCM", "hit", &stats);
}
