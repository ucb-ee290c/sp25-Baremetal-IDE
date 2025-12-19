/*
 * memlat_tests.h - Test entry points for the memory-latency benchmark suite.
 */
#ifndef MEMLAT_TESTS_H
#define MEMLAT_TESTS_H

#include "memlat_core.h"

void memlat_run_l1_hit_test(int core_id);
void memlat_run_dram_cold_miss_test(int core_id);
void memlat_run_scratchpad_hit_test(int core_id);
void memlat_run_tcm_hit_test(int core_id);
void memlat_run_l2_local_hit_test(int core_id);
void memlat_run_l2_remote_hit_test(int core_id);
void memlat_run_scratchpad_under_noc_load(int core_id);

#endif // MEMLAT_TESTS_H
