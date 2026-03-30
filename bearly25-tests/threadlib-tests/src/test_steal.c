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
    g_threadlib_multicore_ok = 0;

    for (uint32_t i = 0; i < N_HARTS; i++) {
        steal_counts[i] = 0;
    }
    steal_total = 0;

    const uint32_t N = 128;

    // Dispatch tasks without specifying which hart; runtime load-balances
    for (uint32_t i = 0; i < N; i++) {
        hthread_dispatch(stealing_task_fn, NULL);
    }

    // Wait for autonomous execution first, then fall back to join-based cleanup.
    uint64_t spin = 0;
    const uint64_t spin_limit = 20000000ull;
    while ((steal_total != N) && (spin < spin_limit)) {
        asm volatile("nop");
        spin++;
    }

    if (steal_total != N) {
        printf("[WARN] dispatch completion timeout: total=%u expected=%u. Forcing join cleanup.\n",
               steal_total, N);
        for (uint32_t h = 0; h < N_HARTS; h++) {
            hthread_join(h);
        }
    }

    // Simple heuristic: each hart should have executed at least one task,
    // and all tasks must complete.
    int pass = 1;
    if (steal_total != N) {
        pass = 0;
        printf("[FAIL] work_stealing_distribution: total completed=%u expected=%u\n",
               steal_total, N);
    }

    printf("[INFO] task distribution:\n");
    for (uint32_t i = 0; i < N_HARTS; i++) {
        printf("  hart %u -> %u tasks\n", i, steal_counts[i]);
        if (steal_counts[i] == 0) {
            pass = 0;
        }
    }

    if (pass) {
        g_threadlib_multicore_ok = 1;
        printf("[PASS] work_stealing_distribution: all harts participated.\n");
    } else {
        printf("[FAIL] work_stealing_distribution: at least one hart did no work.\n");
    }
}
