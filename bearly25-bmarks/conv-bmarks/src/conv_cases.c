#include "conv_cases.h"

const conv_bench_case_t CONV_BENCH_CASES[] = {
  {"b1_c28_h128_w128", 1, 28, 128, 128},
  {"b1_c28_h128_w128", 1, 32, 128, 128},
};

const uint32_t CONV_BENCH_NUM_CASES =
    (uint32_t)(sizeof(CONV_BENCH_CASES) / sizeof(CONV_BENCH_CASES[0]));
