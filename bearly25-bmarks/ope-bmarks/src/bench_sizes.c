/*
 * bench_sizes.c - Size case tables for OPE benchmarks.
 */
#include "bench_sizes.h"

const OpeSizeCase OPE_BENCH_SQUARE_CASES[] = {
  // {"sq_8", 8, 8, 8},
  // {"sq_16", 16, 16, 16},
  // {"sq_32", 32, 32, 32},
  {"sq_64", 64, 64, 64 },
  // {"sq_128", 128, 128, 128 },
  // {"sq_192", 192, 192, 192 },
  // {"sq_256", 256, 256, 256 },
};
const int OPE_BENCH_NUM_SQUARE_CASES =
  (int)(sizeof(OPE_BENCH_SQUARE_CASES)/sizeof(OPE_BENCH_SQUARE_CASES[0]));

const OpeSizeCase OPE_BENCH_RECT_CASES[] = {
  {"rect_16x24x32", 16, 24, 32},
  {"rect_24x16x32", 24, 16, 32},
  {"rect_30x30x30", 30, 30, 30},
  {"rect_32x48x40", 32, 48, 40},
  {"rect_48x32x40", 48, 32, 40},
  {"rect_64x80x72", 64, 80, 72},
  {"rect_80x64x72", 80, 64, 72},
  {"rect_32x32x40", 32, 32, 40},
  {"rect_32x32x48", 32, 32, 48}, 
  {"rect_32x32x56", 32, 32, 56}, 
};
const int OPE_BENCH_NUM_RECT_CASES =
  (int)(sizeof(OPE_BENCH_RECT_CASES)/sizeof(OPE_BENCH_RECT_CASES[0]));
