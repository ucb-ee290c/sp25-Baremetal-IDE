/*
 * test_steal.c - Work-stealing distribution test across harts.
 */
#include "tests.h"
#include "hthread.h"
#include "riscv.h"
#include <stdio.h>
#include <stdint.h>

static volatile uint32_t steal_counts[N_HARTS];
static volatile uint32_t steal_total = 0;

static void stealing_task_fn(void *arg) {
    (void)arg;
    uint64_t mhartid = READ_CSR("mhartid");

    __sync_fetch_and_add(&steal_counts[mhartid], 1);
    __sync_fetch_and_add(&steal_total, 1);
}

// Simple check that all harts get *some* work and total is correct
void test_work_stealing_distribution(void) {
    printf("\n[TEST] Work stealing distribution across harts...\n");

    for (uint32_t i = 0; i < N_HARTS; i++) {
        steal_counts[i] = 0;
    }
    steal_total = 0;

    const uint32_t N = 128;

    // Dispatch tasks without specifying which hart; runtime load-balances
    for (uint32_t i = 0; i < N; i++) {
        hthread_dispatch(stealing_task_fn, NULL);
    }

    // Wait until all tasks have run
    while (steal_total != N) {
        asm volatile("nop");
    }

    // Simple heuristic: each hart should have executed at least one task
    int pass = 1;
    printf("[INFO] task distribution:\n");
    for (uint32_t i = 0; i < N_HARTS; i++) {
        printf("  hart %u -> %u tasks\n", i, steal_counts[i]);
        if (steal_counts[i] == 0) {
            pass = 0;
        }
    }

    if (pass) {
        printf("[PASS] work_stealing_distribution: all harts participated.\n");
    } else {
        printf("[FAIL] work_stealing_distribution: at least one hart did no work.\n");
    }
}
