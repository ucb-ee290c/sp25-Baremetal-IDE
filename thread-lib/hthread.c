/*
 * hthread.c - Work-stealing thread runtime for Bearly25.
 *
 * Implements per-hart deques, atomic steal, and CLINT MSIP wakeups to
 * schedule tasks across cores. Exposes issue/dispatch/join/barrier plus
 * the worker loop for non-hart0 cores.
 */
#include "hthread.h"
#include "chip_config.h"

static wsdeque_t deques[N_HARTS];
static volatile uint32_t pending_tasks[N_HARTS];
static volatile uint8_t hart_busy[N_HARTS];

static volatile uint32_t barrier_count = 0;
static volatile uint32_t barrier_epoch = 0;

// Atomic compare-and-swap operation for lock-free algorithms
static inline int atomic_cas(volatile uint32_t *ptr, uint32_t expected, uint32_t desired) {
    uint32_t old;

    asm volatile (
        "1: lr.w %[old], (%[ptr]);"
        "   bne  %[old], %[exp], 2f;"
        "   sc.w zero, %[des], (%[ptr]);"
        "   bnez zero, 1b;"
        "2:"
        : [old] "=&r" (old)
        : [ptr] "r" (ptr), [exp] "r" (expected), [des] "r" (desired)
        : "memory"
    );
    return old == expected;
}

// Add a task to the bottom of a hart’s deque (owner push)
static inline void ws_push(uint32_t hartid, htask_t task) {
    wsdeque_t *dq = &deques[hartid];

    uint32_t b = dq->bottom;
    dq->tasks[b & (WSQ_SIZE - 1)] = task;
    dq->bottom = b + 1;

    __sync_fetch_and_add(&pending_tasks[hartid], 1);
    hart_busy[hartid] = 1;
}

// Pop a local task from the bottom of the hart’s deque
static inline int ws_pop(uint32_t hartid, htask_t *out) {
    wsdeque_t *dq = &deques[hartid];

    uint32_t b = dq->bottom - 1;
    dq->bottom = b;

    uint32_t t = dq->top;

    if (t <= b) {
        *out = dq->tasks[b & (WSQ_SIZE - 1)];
        __sync_fetch_and_sub(&pending_tasks[hartid], 1);
        return 1; // success
    } else {
        dq->bottom = t;  // undo
        return 0;
    }
}

// Attempt to steal a task from the top of another hart’s queue
static inline int ws_steal(uint32_t victim, htask_t *out) {
    wsdeque_t *dq = &deques[victim];

    uint32_t t = dq->top;
    uint32_t b = dq->bottom;

    if (t >= b) {
        return 0;
    }

    htask_t task = dq->tasks[t & (WSQ_SIZE - 1)];

    if (atomic_cas(&dq->top, t, t + 1)) {
        *out = task;
        __sync_fetch_and_sub(&pending_tasks[victim], 1);
        return 1;
    }

    return 0;
}

// Pushes a task directly onto a specific hart’s queue and wakes that hart
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg) {
    htask_t t = {fn, arg};
    ws_push(hartid, t);
    CLINT->MSIP[hartid] = 1;
}

// Submit a unit of work without deciding which hart executes it.
void hthread_dispatch(void (*fn)(void *), void *arg) {
    htask_t t = {fn, arg};
    ws_push(0, t);

    for (int i = 0; i < N_HARTS; i++)
        CLINT->MSIP[i] = 1;
}

// Block until a specific hart has no more pending tasks
void hthread_join(uint32_t hartid) {
    uint64_t self = READ_CSR("mhartid");
    htask_t task;

    while (pending_tasks[hartid] != 0) {
        // If we're waiting on our own queue, make forward progress locally
        if (self == hartid) {
            if (ws_pop(hartid, &task)) {
                task.fn(task.arg);
                continue;
            }
        }
        asm volatile("nop");
    }
}

// Synchronizes all harts so they reach the same point before proceeding
void hthread_barrier() {
    uint64_t mhartid = READ_CSR("mhartid");
    uint32_t epoch = barrier_epoch;
    uint32_t tmp;

    asm volatile("amoadd.w %0, %1, (%2)" : "=r"(tmp) : "r"(1), "r"(&barrier_count) : "memory");

    if (barrier_count == N_HARTS) {
        barrier_count = 0;
        barrier_epoch++;
    }

    while (barrier_epoch == epoch) {
        asm volatile("nop");
    }
}

// Initializes the entire threading subsystem before any parallel work happens
void hthread_init() {
    for (int i = 0; i < N_HARTS; i++) {
        deques[i].top = 0;
        deques[i].bottom = 0;
        pending_tasks[i] = 0;
        hart_busy[i] = 0;
    }
}

// Define the infinite scheduling loop running on every hart except hart0
void __main(void) {

    uint64_t mhartid = READ_CSR("mhartid");
    CLINT->MSIP[mhartid] = 0;
    htask_t task;

    while (1) {

        // Try local pop
        if (ws_pop(mhartid, &task)) {
            task.fn(task.arg);
            continue;
        }

        // Try stealing
        for (int victim = 0; victim < N_HARTS; victim++) {
            if (victim == mhartid)
                continue;

            if (ws_steal(victim, &task)) {
                task.fn(task.arg);
                continue;
            }
        }

        // If no work, mark idle and sleep
        hart_busy[mhartid] = 0;
        asm volatile("wfi");
        CLINT->MSIP[mhartid] = 0;
    }
}
