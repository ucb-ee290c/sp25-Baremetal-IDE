/*
 * tests.c - Top-level aggregator for thread-lib test cases.
 */
#include "tests.h"
#include <stdio.h>

void run_all_tests(void) {
    printf("\n=============================\n");
    printf(" Running threading test suite\n");
    printf("=============================\n");

    test_basic_local_queue();
    test_basic_lifo_order();
    test_work_stealing_distribution();
    test_barrier_correctness();

    printf("\n[INFO] All tests completed.\n");
}
