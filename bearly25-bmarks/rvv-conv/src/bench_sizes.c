/*
 * bench_sizes.c - Case table for rvv-conv benchmarks.
 */
#include "bench_sizes.h"

const ConvBenchCase RVV_CONV_CASES[] = {
  // {"b1_c28_h128_w128", 1, 28, 128, 128},
  {"b1_c32_h128_w128", 1, 32, 128, 128},
  // {"b1_c16_h64_w64",   1, 16, 64,  64 },
  // {"b2_c28_h128_w128", 2, 28, 128, 128},
};

const int CORE_V_CONV_NUM_CASES =
  (int)(sizeof(RVV_CONV_CASES) / sizeof(RVV_CONV_CASES[0]));
