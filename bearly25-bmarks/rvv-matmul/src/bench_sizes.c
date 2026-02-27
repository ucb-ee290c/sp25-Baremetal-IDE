/*
 * bench_sizes.c - Size case tables for RVV matmul benchmarks.
 */
#include "bench_sizes.h"

const RvvMatmulCase RVV_BENCH_SQUARE_CASES[] = {
  {"sq_16", 16, 16, 16},
  {"sq_32", 32, 32, 32},
  {"sq_64", 64, 64, 64},
  {"sq_96", 96, 96, 96},
  {"sq_128", 128, 128, 128},
};

const int RVV_BENCH_NUM_SQUARE_CASES =
    (int)(sizeof(RVV_BENCH_SQUARE_CASES) / sizeof(RVV_BENCH_SQUARE_CASES[0]));

const RvvMatmulCase RVV_BENCH_RECT_CASES[] = {
  {"rect_16x24x32", 16, 24, 32},
  {"rect_24x16x32", 24, 16, 32},
  {"rect_32x48x40", 32, 48, 40},
  {"rect_48x32x40", 48, 32, 40},
  {"rect_64x80x72", 64, 80, 72},
  {"rect_80x64x72", 80, 64, 72},
};

const int RVV_BENCH_NUM_RECT_CASES =
    (int)(sizeof(RVV_BENCH_RECT_CASES) / sizeof(RVV_BENCH_RECT_CASES[0]));
