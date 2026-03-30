/*
 * tests.c - Top-level aggregator for thread-lib test cases.
 */
#include "tests.h"
#include <stdio.h>

int g_threadlib_multicore_ok = 0;

void run_all_tests(void) {
    printf("\n=============================\n");
    printf(" Running threading test suite\n");
    printf("=============================\n");

    test_basic_local_queue();
    test_basic_lifo_order();
    test_work_stealing_distribution();
    if (g_threadlib_multicore_ok) {
        test_barrier_correctness();
    } else {
        printf("\n[SKIP] barrier_correctness: multicore work-stealing check did not pass.\n");
    }

    printf("\n[INFO] All tests completed.\n");
}
