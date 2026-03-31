/*
 * bench_impl.h - RVV matmul benchmark harness.
 */
#ifndef RVV_BENCH_IMPL_H
#define RVV_BENCH_IMPL_H

#include "bench_sizes.h"

void bench_run_case(const RvvMatmulCase *cs);

#endif // RVV_BENCH_IMPL_H
