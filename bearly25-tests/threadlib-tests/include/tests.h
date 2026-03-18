/*
 * tests.h - Test entry points for the thread-lib work-stealing runtime.
 */
#ifndef __TESTS_H
#define __TESTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void run_all_tests(void);

void test_basic_local_queue(void);
void test_basic_lifo_order(void);
void test_work_stealing_distribution(void);
void test_barrier_correctness(void);

#ifdef __cplusplus
}
#endif

#endif /* __TESTS_H */
