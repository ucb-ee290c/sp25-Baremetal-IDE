/*
 * hthread.h - Public API for the bare-metal work-stealing runtime.
 *
 * Defines task/deque types and exposes init/dispatch/join/barrier APIs used
 * by tests and demos on the multi-hart Bearly25 platform.
 */
#ifndef __HTHREAD_H
#define __HTHREAD_H

#include "clint.h"
#include "riscv.h"
#include <stdint.h>

#define N_HARTS 2
#define WSQ_SIZE 64

typedef struct {
    void (*fn)(void *);
    void *arg;
    /* Runtime-managed metadata for ownership/steal policy. */
    uint32_t owner;
    uint32_t flags;
} htask_t;

typedef struct {
    volatile uint32_t top;
    volatile uint32_t bottom;
    htask_t tasks[WSQ_SIZE];
} wsdeque_t;

#define HTHREAD_TASK_STEALABLE (1u << 0)

void hthread_init();
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_dispatch(void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
void hthread_barrier();

#endif
