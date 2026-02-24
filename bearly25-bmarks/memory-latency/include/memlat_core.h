/*
 * memlat_core.h - Pointer-chasing and statistics for memory-latency tests
 *
 * Each test builds a random pointer-chase ring: an array of cache-line-aligned
 * locations where the first word of each location stores the address of the
 * next location in the ring.  Chasing the ring creates a load-to-use
 * data-dependency chain as each load address depends on the previous load
 * result, so the CPU cannot overlap or prefetch.
 *
 * Measured quantity:  cycles_per_access = total_cycles / num_steps
 */
 
#ifndef MEMLAT_CORE_H
#define MEMLAT_CORE_H

#include <stdint.h>
#include "memlat_config.h"

typedef struct {
    double mean;
    uint64_t min;
    uint64_t median;
    uint64_t max;
} memlat_stats_t;

static inline uint64_t memlat_rdcycle(void) {
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

static inline uint64_t memlat_disable_irqs(void) {
    uint64_t old;
    asm volatile("csrrc %0, mstatus, %1" : "=r"(old) : "r"(1ULL << 3) : "memory");
    return old;
}

static inline void memlat_restore_irqs(uint64_t prev) {
    asm volatile("csrw mstatus, %0" :: "r"(prev) : "memory");
}

static inline uint32_t memlat_hartid(void) {
    uint32_t id;
    asm volatile("csrr %0, mhartid" : "=r"(id));
    return id;
}

uintptr_t memlat_build_chase_ring(uintptr_t *addrs, uint32_t n, uint32_t seed);


uint64_t memlat_run_chase(uintptr_t start, uint64_t steps);

void memlat_run_test(const char *test_name,
                     uintptr_t  start,
                     uint32_t   num_nodes);

void memlat_compute_stats(uint64_t *samples, uint32_t n, memlat_stats_t *out);
void memlat_print_result(const char *name, const memlat_stats_t *s);

uint32_t memlat_verify_chase(uintptr_t start, uint32_t num_nodes,
                              uintptr_t region_base, uintptr_t region_size);
void memlat_print_integrity(uint32_t errors);

#endif /* MEMLAT_CORE_H */
