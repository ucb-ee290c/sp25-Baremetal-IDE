#include "tests.h"
#include "hthread.h"
#include "riscv.h"
#include <stdio.h>
#include <stdint.h>

// ----------------------
// Test 1: local queue count
// ----------------------

static volatile uint32_t basic_counter = 0;

static void basic_task_fn(void *arg) {
    (void)arg;
    __sync_fetch_and_add(&basic_counter, 1);
}

void test_basic_local_queue(void) {
    printf("\n[TEST] Basic local queue count on hart0...\n");

    basic_counter = 0;
    const uint32_t N = 16;

    // Issue all tasks to hart 0 only
    for (uint32_t i = 0; i < N; i++) {
        hthread_issue(0, basic_task_fn, NULL);
    }

    // Wait until hart0 finishes its queue
    hthread_join(0);

    if (basic_counter == N) {
        printf("[PASS] basic_local_queue: expected %u, got %u\n", N, basic_counter);
    } else {
        printf("[FAIL] basic_local_queue: expected %u, got %u\n", N, basic_counter);
    }
}

// ----------------------
// Test 2: local LIFO order (hart0 only, no stealing)
// ----------------------

#define LIFO_TEST_N 8

static volatile uint32_t lifo_exec[LIFO_TEST_N];
static volatile uint32_t lifo_index = 0;

// Each task records its ID in the next slot
static void lifo_task_fn(void *arg) {
    uint32_t id = (uint32_t)(uintptr_t)arg;
    uint32_t idx = lifo_index;
    lifo_exec[idx] = id;
    lifo_index = idx + 1;
}

void test_basic_lifo_order(void) {
    printf("\n[TEST] Basic LIFO ordering on hart0...\n");

    lifo_index = 0;
    for (uint32_t i = 0; i < LIFO_TEST_N; i++) {
        lifo_exec[i] = 0xFFFFFFFFu;
    }

    // Push tasks with IDs 0..N-1 in order, to hart 0.
    // Only hart 0 is woken here, so no stealing should occur.
    for (uint32_t i = 0; i < LIFO_TEST_N; i++) {
        hthread_issue(0, lifo_task_fn, (void *)(uintptr_t)i);
    }

    hthread_join(0);

    // Expected LIFO: executed sequence should be N-1, N-2, ..., 0
    int pass = 1;
    if (lifo_index != LIFO_TEST_N) {
        pass = 0;
        printf("[FAIL] lifo_order: executed %u tasks, expected %u\n",
               lifo_index, (uint32_t)LIFO_TEST_N);
    } else {
        for (uint32_t i = 0; i < LIFO_TEST_N; i++) {
            uint32_t expected = (LIFO_TEST_N - 1) - i;
            if (lifo_exec[i] != expected) {
                pass = 0;
                printf("[FAIL] lifo_order: position %u -> got %u, expected %u\n",
                       i, lifo_exec[i], expected);
            }
        }
    }

    if (pass) {
        printf("[PASS] basic_lifo_order: owner queue behaves as LIFO.\n");
    }
}
