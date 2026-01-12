/*
 * bench_sizes.c - Size case tables for core-v-ope benchmarks.
*/

#include "bench_sizes.h"

const OuterSizeCase CORE_V_OPE_CASES[] = {
  {"op_8x8", 8, 8, 1},
  {"op_16x16", 16, 16, 1},
  {"op_32x32", 32, 32, 1},
  {"op_64x64", 64, 64, 1},
  {"op_128x128", 128, 128, 1},
  {"op_256x256", 256, 256, 1},
  {"op_64x128", 64, 128, 1},
  {"op_128x64", 128, 64, 1},
};

const int CORE_V_OPE_NUM_CASES =
  (int)(sizeof(CORE_V_OPE_CASES) / sizeof(CORE_V_OPE_CASES[0]));
