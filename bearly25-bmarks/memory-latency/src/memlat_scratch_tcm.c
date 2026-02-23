/* =========================================================================
 * memlat_scratch_tcm.c - Scratchpad and TCM latency tests.
 *
 * Scratchpad:  64 KB SRAM on MBUS at 0x08000000.
 *              Path: Core → SBUS NoC → L2 (pass-through) → MBUS NoC → SRAM
 *
 * Local TCM:   8 KB tightly-coupled memory inside the tile.
 *              Core 0 TCM at 0x08010000, Core 1 TCM at 0x08012000.
 *              Local access bypasses the NoC.
 *
 * Remote TCM:  Accessing the OTHER core's TCM.
 *              Core 0 → Core 1 TCM at 0x08012000.
 *              Path: Core 0 → SBUS NoC → Core 1 tile → TCM SRAM.
 * ========================================================================= */

#include <stdint.h>
#include "memlat_core.h"
#include "memlat_config.h"
#include "memlat_addrs.h"

static uintptr_t node_addrs[MAX_NODES];

static uintptr_t setup_sram_chase(uintptr_t region_base,
                                  uint32_t  num_nodes,
                                  uint32_t  seed) {
    for (uint32_t i = 0; i < num_nodes; i++) {
        node_addrs[i] = region_base + (uintptr_t)i * CACHE_LINE_BYTES;
    }
    return memlat_build_chase_ring(node_addrs, num_nodes, seed);
}

void memlat_test_scratchpad(void) {
    uintptr_t start = setup_sram_chase(SCRATCHPAD_BASE,
                                       SCRATCH_NUM_NODES, 111);
    memlat_run_test("Scratchpad", start, SCRATCH_NUM_NODES);
}

void memlat_test_local_tcm(void)
{
    uintptr_t start = setup_sram_chase(CORE0_TCM_BASE,
                                       TCM_NUM_NODES, 222);
    memlat_run_test("Local TCM (Core 0)", start, TCM_NUM_NODES);
}

void memlat_test_remote_tcm(void)
{
    uintptr_t start = setup_sram_chase(CORE1_TCM_BASE,
                                       TCM_NUM_NODES, 333);
    memlat_run_test("Remote TCM (Core 1)", start, TCM_NUM_NODES);
}
