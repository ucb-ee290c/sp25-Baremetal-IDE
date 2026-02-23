/*
 * memlat_core.c - Pointer-chase engine, statistics, and test driver.
 */

#include <stdio.h>
#include <string.h>
#include "memlat_core.h"

static uint32_t lcg_state;

static void lcg_seed(uint32_t s) {
    lcg_state = s ? s : 1u;
}

static uint32_t lcg_next(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

uintptr_t memlat_build_chase_ring(uintptr_t *addrs, uint32_t n, uint32_t seed) {
    lcg_seed(seed);

    // Fisher-Yates shuffle
    for (uint32_t i = n - 1; i > 0; i--) {
        uint32_t j = lcg_next() % (i + 1);
        uintptr_t tmp = addrs[i];
        addrs[i] = addrs[j];
        addrs[j] = tmp;
    }

    // Link each node to the next
    for (uint32_t i = 0; i < n; i++) {
        *(volatile uintptr_t *)addrs[i] = addrs[(i + 1) % n];
    }

    // Ensure all pointer writes are visible before any chase
    asm volatile("fence rw, rw" ::: "memory");

    return addrs[0];
}

uint64_t memlat_run_chase(uintptr_t start, uint64_t steps) {
    register uintptr_t p = start;
    uint64_t t0, t1;

    asm volatile("fence rw, rw" ::: "memory");
    t0 = memlat_rdcycle();

    for (uint64_t i = 0; i < steps; i++) {
        p = *(volatile uintptr_t *)p;
    }

    t1 = memlat_rdcycle();

    asm volatile("" :: "r"(p) : "memory");

    return t1 - t0;
}

static void sort_u64(uint64_t *a, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) {
        uint64_t key = a[i];
        uint32_t j = i;
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = key;
    }
}

void memlat_compute_stats(uint64_t *samples, uint32_t n, memlat_stats_t *out) {
    if (n == 0) return;

    uint64_t sum = 0, mn = samples[0], mx = samples[0];
    for (uint32_t i = 0; i < n; i++) {
        sum += samples[i];
        if (samples[i] < mn) mn = samples[i];
        if (samples[i] > mx) mx = samples[i];
    }

    sort_u64(samples, n);

    out->mean = (double)sum / (double)n;
    out->min = mn;
    out->median = samples[n / 2];
    out->max = mx;
}

void memlat_print_result(const char *name, const memlat_stats_t *s) {
    printf("  %-22s  min=%3llu  mean=%3llu  median=%3llu  max=%3llu  cycles/access\n",
           name,
           (unsigned long long)s->min,
           (unsigned long long)(uint64_t)(s->mean + 0.5),
           (unsigned long long)s->median,
           (unsigned long long)s->max);
}

void memlat_run_test(const char *test_name,
                     uintptr_t  start,
                     uint32_t   num_nodes) {
    uint32_t steps = num_nodes;
    while (steps < MIN_STEPS_PER_SAMPLE){
        steps += num_nodes;
    }

    for (uint32_t w = 0; w < WARMUP_PASSES; w++) {
        (void)memlat_run_chase(start, num_nodes);
    }

    uint64_t samples[NUM_SAMPLES];

    uint64_t saved = memlat_disable_irqs();

    for (uint32_t s = 0; s < NUM_SAMPLES; s++) {
        uint64_t total = memlat_run_chase(start, steps);
        samples[s] = total / (uint64_t)steps;
    }

    memlat_restore_irqs(saved);

    memlat_stats_t stats;
    memlat_compute_stats(samples, NUM_SAMPLES, &stats);
    memlat_print_result(test_name, &stats);
}
