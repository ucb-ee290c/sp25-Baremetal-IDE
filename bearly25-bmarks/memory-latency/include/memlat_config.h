/*
 * memlat_config.h - Configuration constants for memory-latency microbenchmarks.
 */
#ifndef MEMLAT_CONFIG_H
#define MEMLAT_CONFIG_H

#include <stdint.h>

#define MEMLAT_L1_SIZE_BYTES      (16u * 1024u)
#define MEMLAT_L2_TOTAL_BYTES     (256u * 1024u)

// Number of timing samples to collect for each experiment (after warmup)
#define MEMLAT_NUM_SAMPLES        256u

// How many initial samples to throw away to let things warmup
#define MEMLAT_WARMUP_SAMPLES     16u

// For dependency chains L1 / scratchpad / TCM hit tests
#define MEMLAT_DEP_CHAIN_ITERS    32u

// Size of DRAM region we stream over to ensure L1/L2 are cold
#define MEMLAT_DRAM_REGION_BYTES  (1u * 1024u * 1024u) // 1 MiB

#endif // MEMLAT_CONFIG_H
