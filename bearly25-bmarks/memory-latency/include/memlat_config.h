/*
 * memlat_config.h - Configuration constants for memory-latency microbenchmarks.
 *
 * Bearly25 Cache Hierarchy:
 *   L1D: 64 sets, 2 ways, 64B lines = 8 KB per core
 *   L2:  2 banks, 256 sets, 8 ways, 64B lines = 128 KB per bank (256 KB total)
 *        Bank index = addr[6]
 *.       Bank 1 is local to Core 0, Bank 0 is local to Core 1
 *   Scratchpad: 64 KB SRAM on MBUS at 0x08000000
 *   TCM: 8 KB per core (Core 0 @ 0x08010000, Core 1 @ 0x08012000)
 *   DRAM: behind SerialTL at 0x80000000+
 */
 
#ifndef MEMLAT_CONFIG_H
#define MEMLAT_CONFIG_H


#define CACHE_LINE_BYTES        64u
#define L1D_SETS                64u
#define L1D_WAYS                2u
#define L1D_CAPACITY_BYTES      (L1D_SETS * L1D_WAYS * CACHE_LINE_BYTES)  /* 8 KB */

#define L2_BANKS                2u
#define L2_SETS_PER_BANK        256u
#define L2_WAYS                 8u
#define L2_BANK_CAPACITY_BYTES  (L2_SETS_PER_BANK * L2_WAYS * CACHE_LINE_BYTES) /* 128 KB */
#define L2_TOTAL_BYTES          (L2_BANKS * L2_BANK_CAPACITY_BYTES)             /* 256 KB */

#define L2_BANK_BIT             6u

/*
 * L1 test:  64 nodes = 4 KB working set    (fits in 8 KB L1D)
 * L2 test:  1024 nodes on one bank = 64 KB (> L1D, < 128 KB bank)
 * DRAM:     16384 nodes = 1 MB             (>> 256 KB L2)
 * Scratch:  512 nodes = 32 KB              (< 64 KB scratchpad)
 * TCM:      64 nodes = 4 KB                (< 8 KB TCM)
*/

#define L1_NUM_NODES            64u
#define L2_NUM_NODES            1024u
#define DRAM_NUM_NODES          16384u
#define SCRATCH_NUM_NODES       512u
#define TCM_NUM_NODES           64u

#define MAX_NODES               DRAM_NUM_NODES

#define MIN_STEPS_PER_SAMPLE    1024u
#define NUM_SAMPLES             32u

#define WARMUP_PASSES           8u

#define DRAM_POOL_BYTES         (2u * 1024u * 1024u)  /* 2 MB */

#endif /* MEMLAT_CONFIG_H */
