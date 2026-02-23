/*
 * memlat_tests.h - Test entry points for the memory-latency benchmark suite.
 */

#ifndef MEMLAT_TESTS_H
#define MEMLAT_TESTS_H

void memlat_test_l1_hit(void);
void memlat_test_l2_local_hit(void);
void memlat_test_l2_remote_hit(void);
void memlat_test_dram(void);
void memlat_test_scratchpad(void);
void memlat_test_local_tcm(void);
void memlat_test_remote_tcm(void);

#endif /* MEMLAT_TESTS_H */
