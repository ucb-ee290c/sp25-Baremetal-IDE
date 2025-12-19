/*
 * old_hthread.h - Legacy threading API for early bring-up.
 *
 * Minimal interface for dispatching a single task to a hart and waiting
 * for completion via shared flags.
 */
#ifndef __HTHREAD_H
#define __HTHREAD_H

#include "clint.h"
#include "riscv.h"

#define N_HARTS 2

void hthread_issue(uint32_t hartid, void *(* start_routine)(void *), void *arg);

void hthread_join(uint32_t hartid);

#endif /* __HTHREAD_H */
