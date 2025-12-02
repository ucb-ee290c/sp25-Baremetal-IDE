#ifndef MEMLAT_CORE_H
#define MEMLAT_CORE_H

#include <stdint.h>
#include "memlat_config.h"

typedef struct {
    double      mean_cycles;
    uint64_t    min_cycles;
    uint64_t    p95_cycles;
    uint64_t    p99_cycles;
} memlat_stats_t;

// Disable interrupts, return previous mstatus so we can restore it later
uint64_t memlat_disable_irqs(void);

// Restore mstatus saved by memlat_disable_irqs()
void memlat_restore_irqs(uint64_t prev_mstatus);

// Read the current hart ID (mhartid CSR)
uint32_t memlat_read_hartid(void);

// One dependent-load chain on a single address (cycles per load averaged over iters)
uint64_t memlat_measure_dep_chain(volatile uint32_t *addr, uint32_t iters);

// Single-load measurement (for cold DRAM misses)
uint64_t memlat_measure_single_load(volatile uint32_t *addr);

// Compute mean / min / p95 / p99 over the samples array (size = n)
void memlat_compute_stats(uint64_t *samples, uint32_t n, memlat_stats_t *out);

// Print a single CSV line with stats for this core / region / mode.
void memlat_print_stats_line(int core_id,
                             const char *region,
                             const char *mode,
                             const memlat_stats_t *s);

#endif // MEMLAT_CORE_H
