/*
 * bench_sizes.c - Size case tables for core-v-conv benchmarks.
 */
#include "bench_sizes.h"

#ifndef ASSEMBLY_CONV
const ConvSizeCase CORE_V_CONV_CASES[] = {
  {"conv_16", 16, 16, 3, 1, INT},
  {"conv_32", 32, 32, 3, 1, INT},
  {"conv_64", 64, 64, 3, 1, INT},
  {"conv_128", 128, 128, 3, 1, INT},
  {"conv_256", 256, 256, 3, 1, INT},
};
#else
const ConstSizeCase CORE_V_CONV_CASES[] = {
  {"conv_16_3x3_i8", 16, 16, 3, 1, INT},
  {"conv_16_5x5_i8", 16, 16, 5, 1, INT},
  {"conv_16_3x3_fp32", 16, 16, 3, 4, FLOAT},
  {"conv_16_5x5_fp32", 16, 16, 5, 4, FLOAT},
  {"conv_32_3x3_i8", 32, 32, 3, 1, INT},
  {"conv_32_5x5_i8", 32, 32, 5, 1, INT},
  {"conv_32_3x3_fp32", 32, 32, 3, 4, FLOAT},
  {"conv_32_5x5_fp32", 32, 32, 5, 4, FLOAT},
  {"conv_64_3x3_i8", 64, 64, 3, 1, INT},
  {"conv_64_5x5_i8", 64, 64, 5, 1, INT},
  {"conv_64_3x3_fp32", 64, 64, 3, 4, FLOAT},
  {"conv_64_5x5_fp32", 64, 64, 5, 4, FLOAT},
  {"conv_128_3x3_i8", 128, 128, 3, 1, INT},
  {"conv_128_5x5_i8", 128, 128, 5, 1, INT},
  {"conv_128_3x3_fp32", 128, 128, 3, 4, FLOAT},
  {"conv_128_5x5_fp32", 128, 128, 5, 4, FLOAT}
}
#endif

const int CORE_V_CONV_NUM_CASES =
  (int)(sizeof(CORE_V_CONV_CASES) / sizeof(CORE_V_CONV_CASES[0]));
