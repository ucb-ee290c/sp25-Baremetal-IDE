#ifndef __HTHREAD_H
#define __HTHREAD_H

#include "clint.h"
#include "chip_config.h"
#include "riscv.h"
#include <stdint.h>

#define N_HARTS 2
#define WSQ_SIZE 64

typedef struct {
    void (*fn)(void *);
    void *arg;
} htask_t;

typedef struct {
    volatile uint32_t top;
    volatile uint32_t bottom;
    htask_t tasks[WSQ_SIZE];
} wsdeque_t;


void hthread_init();
void hthread_issue(uint32_t hartid, void (*fn)(void *), void *arg);
void hthread_dispatch(void (*fn)(void *), void *arg);
void hthread_join(uint32_t hartid);
void hthread_barrier();

#endif
