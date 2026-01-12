/*
 * bench_sizes.c - Size case tables for core-v-ope benchmarks.
*/

#include "bench_sizes.h"

const OuterSizeCase CORE_V_OPE_CASES[] = {
  {"sq_16", 16, 16, 16},
  {"sq_32", 32, 32, 32},
  {"sq_64", 64, 64, 64},
  {"sq_128", 128, 128, 128},
  {"sq_256", 256, 256, 256},
};

const int CORE_V_OPE_NUM_CASES =
  (int)(sizeof(CORE_V_OPE_CASES) / sizeof(CORE_V_OPE_CASES[0]));
