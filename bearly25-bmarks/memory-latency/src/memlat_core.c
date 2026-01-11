/*
 * memlat_core.c - Core timing utilities and stats aggregation.
 *
 * Provides cycle reads, IRQ masking, dependency-chain timing, and percentile
 * statistics for memory-latency measurements.
 */
#include <stdio.h>
#include <string.h>
#include "memlat_core.h"

#define MEMLAT_MAX_SAMPLES  (MEMLAT_NUM_SAMPLES)

static inline uint64_t memlat_read_cycle(void)
{
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

uint64_t memlat_disable_irqs(void)
{
    uint64_t old_mstatus;
    const uint64_t MSTATUS_MIE = (1ULL << 3);
    // clears bits set in rs1
    asm volatile("csrrc %0, mstatus, %1" : "=r"(old_mstatus) : "r"(MSTATUS_MIE) : "memory");
    return old_mstatus;
}

void memlat_restore_irqs(uint64_t prev_mstatus)
{
    asm volatile("csrw mstatus, %0" :: "r"(prev_mstatus) : "memory");
}

uint32_t memlat_read_hartid(void)
{
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return hartid;
}

// Simple dependency chain: acc_{i+1} = acc_i + *addr
uint64_t memlat_measure_dep_chain(volatile uint32_t *addr, uint32_t iters)
{
    uint64_t start, end;
    uint32_t acc = 1;

    start = memlat_read_cycle();
    for (uint32_t i = 0; i < iters; ++i) {
        acc += *addr;
    }
    end = memlat_read_cycle();
    asm volatile("" :: "r"(acc) : "memory");

    uint64_t total = end - start;
    if (iters == 0) return total;
    return total / (uint64_t)iters;
}

// Cold DRAM miss: only care about first access cost
uint64_t memlat_measure_single_load(volatile uint32_t *addr)
{
    uint64_t start, end;
    uint32_t value;

    start = memlat_read_cycle();
    value = *addr;
    end = memlat_read_cycle();

    asm volatile("" :: "r"(value) : "memory");
    return end - start;
}

static void insertion_sort_u64(uint64_t *arr, uint32_t n)
{
    for (uint32_t i = 1; i < n; ++i) {
        uint64_t key = arr[i];
        uint32_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            --j;
        }
        arr[j] = key;
    }
}

void memlat_compute_stats(uint64_t *samples, uint32_t n, memlat_stats_t *out)
{
    if (n == 0 || !out) return;

    // Compute mean and min first
    uint64_t sum = 0;
    uint64_t minv = samples[0];

    for (uint32_t i = 0; i < n; ++i) {
        sum += samples[i];
        if (samples[i] < minv) minv = samples[i];
    }

    insertion_sort_u64(samples, n);

    uint32_t idx95 = (uint32_t)((double)(n - 1) * 0.95);
    uint32_t idx99 = (uint32_t)((double)(n - 1) * 0.99);

    out->mean_cycles = (double)sum / (double)n;
    out->min_cycles = minv;
    out->p95_cycles = samples[idx95];
    out->p99_cycles = samples[idx99];
}

void memlat_print_stats_line(int core_id,
                             const char *region,
                             const char *mode,
                             const memlat_stats_t *s)
{
    printf("core=%d, region=%s, mode=%s, "
           "mean_cycles=%.2f, min_cycles=%llu, "
           "p95_cycles=%llu, p99_cycles=%llu\n",
           core_id,
           region,
           mode,
           (double)s->mean_cycles,
           (unsigned long long)s->min_cycles,
           (unsigned long long)s->p95_cycles,
           (unsigned long long)s->p99_cycles);
}
