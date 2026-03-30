/*
 * test_advanced.c - Additional correctness and stress tests for thread-lib.
 */
#include "tests.h"
#include "hthread.h"
#include "riscv.h"
#include <stdint.h>
#include <stdio.h>

// ----------------------
// Test: issue() hart affinity
// ----------------------

static volatile uint32_t issue_affinity_counts[N_HARTS];
static volatile uint32_t issue_affinity_bad = 0;

static void issue_affinity_task(void *arg) {
    uint32_t expected_hart = (uint32_t)(uintptr_t)arg;
    uint32_t mhartid = (uint32_t)READ_CSR("mhartid");

    if (mhartid < N_HARTS) {
        __sync_fetch_and_add(&issue_affinity_counts[mhartid], 1);
    }
    if (mhartid != expected_hart) {
        __sync_fetch_and_add(&issue_affinity_bad, 1);
    }
}

void test_issue_affinity(void) {
    printf("\n[TEST] issue() hart affinity...\n");

    for (uint32_t h = 0; h < N_HARTS; h++) {
        issue_affinity_counts[h] = 0;
    }
    issue_affinity_bad = 0;

    const uint32_t N = 64;
    for (uint32_t i = 0; i < N; i++) {
        hthread_issue(1, issue_affinity_task, (void *)(uintptr_t)1u);
    }
    hthread_join(1);

    int pass = 1;
    if (issue_affinity_counts[1] != N) {
        pass = 0;
        printf("[FAIL] issue_affinity: hart1 executed %u, expected %u\n",
               issue_affinity_counts[1], N);
    }
    if (issue_affinity_bad != 0) {
        pass = 0;
        printf("[FAIL] issue_affinity: %u task(s) ran on wrong hart\n",
               issue_affinity_bad);
    }

    if (pass) {
        printf("[PASS] issue_affinity: all tasks stayed on hart1.\n");
    }
}

// ----------------------
// Test: join() must wait for task completion
// ----------------------

static volatile uint32_t join_task_started = 0;
static volatile uint32_t join_task_done = 0;

static void join_completion_task(void *arg) {
    (void)arg;
    join_task_started = 1;

    for (volatile uint32_t i = 0; i < 3000000u; i++) {
        asm volatile("nop");
    }

    join_task_done = 1;
}

void test_join_completion_semantics(void) {
    printf("\n[TEST] join() completion semantics...\n");

    join_task_started = 0;
    join_task_done = 0;

    hthread_issue(1, join_completion_task, NULL);
    hthread_join(1);

    if (join_task_started && join_task_done) {
        printf("[PASS] join_completion_semantics: join returned after completion.\n");
    } else {
        printf("[FAIL] join_completion_semantics: started=%u done=%u\n",
               join_task_started, join_task_done);
    }
}

// ----------------------
// Test: dispatch capacity stress (> WSQ_SIZE)
// ----------------------

static volatile uint32_t dispatch_capacity_total = 0;

static void dispatch_capacity_task(void *arg) {
    (void)arg;
    __sync_fetch_and_add(&dispatch_capacity_total, 1);
}

void test_dispatch_capacity_stress(void) {
    printf("\n[TEST] dispatch() capacity stress...\n");

    dispatch_capacity_total = 0;
    const uint32_t N = (uint32_t)(WSQ_SIZE * 6u);

    for (uint32_t i = 0; i < N; i++) {
        hthread_dispatch(dispatch_capacity_task, NULL);
    }

    uint64_t spin = 0;
    const uint64_t spin_limit = 60000000ull;
    while ((dispatch_capacity_total != N) && (spin < spin_limit)) {
        asm volatile("nop");
        spin++;
    }

    if (dispatch_capacity_total != N) {
        printf("[WARN] dispatch_capacity timeout: total=%u expected=%u. Forcing join cleanup.\n",
               dispatch_capacity_total, N);
        for (uint32_t h = 0; h < N_HARTS; h++) {
            hthread_join(h);
        }
    }

    if (dispatch_capacity_total == N) {
        printf("[PASS] dispatch_capacity_stress: completed %u/%u tasks.\n", N, N);
    } else {
        printf("[FAIL] dispatch_capacity_stress: completed %u/%u tasks.\n",
               dispatch_capacity_total, N);
    }
}

// ----------------------
// Test: barrier repeated stress
// ----------------------

static volatile uint32_t barrier_repeated_phase[N_HARTS];
static volatile uint32_t barrier_repeated_failures = 0;

static void barrier_repeated_task(void *arg) {
    (void)arg;
    uint32_t self = (uint32_t)READ_CSR("mhartid");
    const uint32_t iters = 64u;

    for (uint32_t i = 0; i < iters; i++) {
        barrier_repeated_phase[self] = i + 1u;
        hthread_barrier();

        for (uint32_t h = 0; h < N_HARTS; h++) {
            if (barrier_repeated_phase[h] < (i + 1u)) {
                __sync_fetch_and_add(&barrier_repeated_failures, 1);
            }
        }

        hthread_barrier();
    }
}

void test_barrier_repeated_stress(void) {
    printf("\n[TEST] Barrier repeated stress...\n");

    for (uint32_t h = 0; h < N_HARTS; h++) {
        barrier_repeated_phase[h] = 0;
    }
    barrier_repeated_failures = 0;

    for (uint32_t h = 0; h < N_HARTS; h++) {
        hthread_issue(h, barrier_repeated_task, NULL);
    }
    for (uint32_t h = 0; h < N_HARTS; h++) {
        hthread_join(h);
    }

    if (barrier_repeated_failures == 0) {
        printf("[PASS] barrier_repeated_stress: all phases synchronized.\n");
    } else {
        printf("[FAIL] barrier_repeated_stress: observed %u barrier ordering failure(s).\n",
               barrier_repeated_failures);
    }
}

// ----------------------
// Test: mixed issue/dispatch/join stress
// ----------------------

static volatile uint32_t mixed_total = 0;
static volatile uint32_t mixed_issue_on_hart1 = 0;
static volatile uint32_t mixed_issue_off_hart1 = 0;
static volatile uint32_t mixed_dispatch_count = 0;

static void mixed_mode_task(void *arg) {
    uint32_t tag = (uint32_t)(uintptr_t)arg; // 1=issue, 0=dispatch
    uint32_t mhartid = (uint32_t)READ_CSR("mhartid");

    if (tag == 1u) {
        if (mhartid == 1u) {
            __sync_fetch_and_add(&mixed_issue_on_hart1, 1);
        } else {
            __sync_fetch_and_add(&mixed_issue_off_hart1, 1);
        }
    } else {
        __sync_fetch_and_add(&mixed_dispatch_count, 1);
    }

    __sync_fetch_and_add(&mixed_total, 1);
}

void test_mixed_mode_stress(void) {
    printf("\n[TEST] Mixed issue/dispatch/join stress...\n");

    mixed_total = 0;
    mixed_issue_on_hart1 = 0;
    mixed_issue_off_hart1 = 0;
    mixed_dispatch_count = 0;

    const uint32_t N_ISSUE = 64;
    const uint32_t N_DISPATCH = 192;
    const uint32_t N_TOTAL = N_ISSUE + N_DISPATCH;

    for (uint32_t i = 0; i < N_ISSUE; i++) {
        hthread_issue(1, mixed_mode_task, (void *)(uintptr_t)1u);
    }
    for (uint32_t i = 0; i < N_DISPATCH; i++) {
        hthread_dispatch(mixed_mode_task, (void *)(uintptr_t)0u);
    }

    uint64_t spin = 0;
    const uint64_t spin_limit = 60000000ull;
    while ((mixed_total != N_TOTAL) && (spin < spin_limit)) {
        asm volatile("nop");
        spin++;
    }

    if (mixed_total != N_TOTAL) {
        printf("[WARN] mixed_mode timeout: total=%u expected=%u. Forcing join cleanup.\n",
               mixed_total, N_TOTAL);
        for (uint32_t h = 0; h < N_HARTS; h++) {
            hthread_join(h);
        }
    }

    int pass = 1;
    if (mixed_total != N_TOTAL) {
        pass = 0;
    }
    if (mixed_issue_on_hart1 != N_ISSUE) {
        pass = 0;
    }
    if (mixed_issue_off_hart1 != 0) {
        pass = 0;
    }
    if (mixed_dispatch_count != N_DISPATCH) {
        pass = 0;
    }

    printf("[INFO] mixed totals: total=%u issue_on_h1=%u issue_off_h1=%u dispatch=%u\n",
           mixed_total, mixed_issue_on_hart1, mixed_issue_off_hart1, mixed_dispatch_count);

    if (pass) {
        printf("[PASS] mixed_mode_stress: mixed operations completed correctly.\n");
    } else {
        printf("[FAIL] mixed_mode_stress: expected total=%u issue_on_h1=%u issue_off_h1=0 dispatch=%u\n",
               N_TOTAL, N_ISSUE, N_DISPATCH);
    }
}

// ----------------------
// Test: API edge handling (invalid args / NULL dispatch)
// ----------------------

static volatile uint32_t edge_counter = 0;

static void edge_counter_task(void *arg) {
    (void)arg;
    __sync_fetch_and_add(&edge_counter, 1);
}

void test_api_edge_cases(void) {
    printf("\n[TEST] API edge handling...\n");

    edge_counter = 0;

    hthread_issue(N_HARTS, edge_counter_task, NULL);
    hthread_issue(0xFFFFFFFFu, edge_counter_task, NULL);
    hthread_join(N_HARTS);
    hthread_join(0xFFFFFFFFu);
    hthread_dispatch(NULL, NULL);

    hthread_issue(0, edge_counter_task, NULL);
    hthread_join(0);

    if (edge_counter == 1u) {
        printf("[PASS] api_edge_cases: invalid args ignored, valid calls still work.\n");
    } else {
        printf("[FAIL] api_edge_cases: counter=%u expected=1\n", edge_counter);
    }
}
