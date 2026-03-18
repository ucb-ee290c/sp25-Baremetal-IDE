/*
 * bench_sizes.h - Size tables for RVV matmul benchmark cases.
 */
#ifndef RVV_BENCH_SIZES_H
#define RVV_BENCH_SIZES_H

#include "bench_config.h"

extern const RvvMatmulCase RVV_BENCH_SQUARE_CASES[];
extern const int RVV_BENCH_NUM_SQUARE_CASES;

extern const RvvMatmulCase RVV_BENCH_RECT_CASES[];
extern const int RVV_BENCH_NUM_RECT_CASES;

#endif // RVV_BENCH_SIZES_H
