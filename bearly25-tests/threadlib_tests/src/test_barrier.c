/*
 * test_barrier.c - Barrier synchronization correctness test.
 */
#include "tests.h"
#include "hthread.h"
#include "riscv.h"
#include <stdio.h>
#include <stdint.h>

static volatile uint32_t barrier_step[N_HARTS];

static void barrier_task_fn(void *arg) {
    (void)arg;
    uint64_t mhartid = READ_CSR("mhartid");

    // Phase 1: mark we've reached step 1
    barrier_step[mhartid] = 1;

    // Synchronize all harts
    hthread_barrier();

    // Phase 2: mark we've reached step 2
    barrier_step[mhartid] = 2;

    // Synchronize again so no one exits too early
    hthread_barrier();
}

void test_barrier_correctness(void) {
    printf("\n[TEST] Barrier correctness across %u harts...\n", (uint32_t)N_HARTS);

    for (uint32_t i = 0; i < N_HARTS; i++) {
        barrier_step[i] = 0;
    }

    // Schedule one barrier_task_fn per hart
    for (uint32_t h = 0; h < N_HARTS; h++) {
        hthread_issue(h, barrier_task_fn, NULL);
    }

    // Wait for all to finish
    for (uint32_t h = 0; h < N_HARTS; h++) {
        hthread_join(h);
    }

    // Check that every hart reached step 2
    int pass = 1;
    for (uint32_t i = 0; i < N_HARTS; i++) {
        if (barrier_step[i] != 2) {
            pass = 0;
            printf("[FAIL] barrier_correctness: hart %u finished with step=%u (expected 2)\n",
                   i, barrier_step[i]);
        }
    }

    if (pass) {
        printf("[PASS] barrier_correctness: all harts synchronized correctly.\n");
    }
}
