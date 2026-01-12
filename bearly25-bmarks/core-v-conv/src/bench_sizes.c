/*
 * bench_sizes.c - Size case tables for core-v-conv benchmarks.
 */
#include "bench_sizes.h"

const ConvSizeCase CORE_V_CONV_CASES[] = {
  {"conv_16", 16, 16},
  {"conv_32", 32, 32},
  {"conv_64", 64, 64},
  {"conv_128", 128, 128},
  {"conv_256", 256, 256},
};

const int CORE_V_CONV_NUM_CASES =
  (int)(sizeof(CORE_V_CONV_CASES) / sizeof(CORE_V_CONV_CASES[0]));
