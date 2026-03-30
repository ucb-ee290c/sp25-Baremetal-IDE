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
static volatile uint32_t deque_locks[N_HARTS];
static volatile uint32_t pending_tasks[N_HARTS];
static volatile uint8_t hart_busy[N_HARTS];
static volatile uint32_t dispatch_rr = 0;
static volatile uint32_t runtime_ready = 0;

static volatile uint32_t barrier_count = 0;
static volatile uint32_t barrier_epoch = 0;

static inline void lock_deque(uint32_t hartid) {
    while (__sync_lock_test_and_set(&deque_locks[hartid], 1u) != 0u) {
        while (deque_locks[hartid] != 0u) {
            asm volatile("nop");
        }
    }
}

static inline void unlock_deque(uint32_t hartid) {
    __sync_lock_release(&deque_locks[hartid]);
}

static inline void run_task(const htask_t *task) {
    task->fn(task->arg);
    __sync_fetch_and_sub(&pending_tasks[task->owner], 1u);
}

static inline void wake_hart(uint32_t hartid) {
    CLINT->MSIP[hartid] = 1;
}

static inline void wake_other_harts(uint32_t self) {
    for (uint32_t h = 0; h < N_HARTS; ++h) {
        if (h == self) {
            continue;
        }
        wake_hart(h);
    }
}

// Add a task to the bottom of a hart’s deque (owner push)
static inline void ws_push(uint32_t hartid, const htask_t *task) {
    wsdeque_t *dq = &deques[hartid];

    while (1) {
        lock_deque(hartid);

        uint32_t t = dq->top;
        uint32_t b = dq->bottom;
        if ((b - t) < WSQ_SIZE) {
            dq->tasks[b & (WSQ_SIZE - 1)] = *task;
            __sync_fetch_and_add(&pending_tasks[task->owner], 1u);
            dq->bottom = b + 1u;
            unlock_deque(hartid);
            break;
        }

        unlock_deque(hartid);
        asm volatile("nop");
    }

    hart_busy[hartid] = 1u;
}

// Pop a local task from the bottom of the hart’s deque
static inline int ws_pop(uint32_t hartid, htask_t *out) {
    wsdeque_t *dq = &deques[hartid];
    lock_deque(hartid);

    uint32_t t = dq->top;
    uint32_t b = dq->bottom;
    if (t == b) {
        unlock_deque(hartid);
        return 0;
    }

    b -= 1u;
    *out = dq->tasks[b & (WSQ_SIZE - 1u)];
    dq->bottom = b;

    unlock_deque(hartid);
    return 1;
}

// Attempt to steal a task from the top of another hart’s queue
static inline int ws_steal(uint32_t victim, htask_t *out) {
    wsdeque_t *dq = &deques[victim];
    lock_deque(victim);

    uint32_t t = dq->top;
    uint32_t b = dq->bottom;

    if (t == b) {
        unlock_deque(victim);
        return 0;
    }

    htask_t task = dq->tasks[t & (WSQ_SIZE - 1u)];
    if ((task.flags & HTHREAD_TASK_STEALABLE) == 0u) {
        unlock_deque(victim);
        return 0;
    }

    dq->top = t + 1u;
    unlock_deque(victim);

    *out = task;
    return 1;
}

// Pushes a task directly onto a specific hart’s queue and wakes that hart
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg) {
    if (hartid >= N_HARTS || fn == 0) {
        return;
    }

    htask_t t = {
        .fn = fn,
        .arg = arg,
        .owner = hartid,
        .flags = 0u,
    };

    ws_push(hartid, &t);
    wake_hart(hartid);
}

// Submit a unit of work without deciding which hart executes it.
void hthread_dispatch(void (*fn)(void *), void *arg) {
    if (fn == 0) {
        return;
    }

    uint32_t self = (uint32_t)READ_CSR("mhartid");
    uint32_t target = __sync_fetch_and_add(&dispatch_rr, 1u) % N_HARTS;

    // Keep the caller productive while still distributing tasks to other harts.
    if (target == self) {
        fn(arg);
        return;
    }

    htask_t t = {
        .fn = fn,
        .arg = arg,
        .owner = target,
        .flags = HTHREAD_TASK_STEALABLE,
    };

    ws_push(target, &t);
    wake_other_harts(self);
}

// Block until a specific hart has no more pending tasks
void hthread_join(uint32_t hartid) {
    if (hartid >= N_HARTS) {
        return;
    }

    uint32_t self = (uint32_t)READ_CSR("mhartid");
    htask_t task;

    while (pending_tasks[hartid] != 0u) {
        // If we're waiting on our own queue, make forward progress locally
        if (self == hartid) {
            if (ws_pop(hartid, &task)) {
                run_task(&task);
                continue;
            }
        } else {
            // Help remote completion if the victim's front task is stealable.
            if (ws_steal(hartid, &task)) {
                run_task(&task);
                continue;
            }
        }

        wake_hart(hartid);
        asm volatile("nop");
    }
}

// Synchronizes all harts so they reach the same point before proceeding
void hthread_barrier() {
    uint32_t epoch = barrier_epoch;
    uint32_t arrived = __sync_add_and_fetch(&barrier_count, 1u);

    if (arrived == N_HARTS) {
        barrier_count = 0u;
        __sync_synchronize();
        barrier_epoch = epoch + 1u;
    } else {
        while (barrier_epoch == epoch) {
            asm volatile("nop");
        }
    }

    __sync_synchronize();
}

// Initializes the entire threading subsystem before any parallel work happens
void hthread_init() {
    runtime_ready = 0u;
    dispatch_rr = 0u;
    barrier_count = 0u;
    barrier_epoch = 0u;

    for (uint32_t i = 0; i < N_HARTS; i++) {
        deques[i].top = 0;
        deques[i].bottom = 0;
        pending_tasks[i] = 0;
        deque_locks[i] = 0;
        hart_busy[i] = 0u;
        CLINT->MSIP[i] = 0u;
    }

    __sync_synchronize();
    runtime_ready = 1u;
    wake_other_harts((uint32_t)READ_CSR("mhartid"));
}

// Define the infinite scheduling loop running on every hart except hart0
void __main(void) {
    uint32_t mhartid = (uint32_t)READ_CSR("mhartid");
    htask_t task;
    CLINT->MSIP[mhartid] = 0u;

    while (runtime_ready == 0u) {
        asm volatile("wfi");
        CLINT->MSIP[mhartid] = 0u;
    }

    while (1) {
        if (ws_pop(mhartid, &task)) {
            hart_busy[mhartid] = 1u;
            run_task(&task);
            continue;
        }

        int did_work = 0;
        for (uint32_t victim = 0; victim < N_HARTS; victim++) {
            if (victim == mhartid) {
                continue;
            }

            if (ws_steal(victim, &task)) {
                hart_busy[mhartid] = 1u;
                run_task(&task);
                did_work = 1;
                break;
            }
        }

        if (did_work) {
            continue;
        }

        // Avoid missing a wakeup if a producer enqueues right around WFI entry.
        hart_busy[mhartid] = 0u;
        CLINT->MSIP[mhartid] = 0u;
        __sync_synchronize();

        if (ws_pop(mhartid, &task)) {
            hart_busy[mhartid] = 1u;
            run_task(&task);
            continue;
        }

        for (uint32_t victim = 0; victim < N_HARTS; victim++) {
            if (victim == mhartid) {
                continue;
            }
            if (ws_steal(victim, &task)) {
                hart_busy[mhartid] = 1u;
                run_task(&task);
                did_work = 1;
                break;
            }
        }

        if (did_work) {
            continue;
        }

        asm volatile("wfi");
    }
}
