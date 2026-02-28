/*
 * bench_sizes.c - Case table for acc-conv benchmarks.
 */
#include "bench_sizes.h"

const ConvBenchCase ACC_CONV_CASES[] = {
  {"b1_c32_h128_w128", 1, 2, 128, 128},
};

const int ACC_CONV_NUM_CASES =
  (int)(sizeof(ACC_CONV_CASES) / sizeof(ACC_CONV_CASES[0]));
