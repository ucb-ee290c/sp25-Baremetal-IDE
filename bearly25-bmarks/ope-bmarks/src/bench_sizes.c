#include "bench_sizes.h"

const OpeSizeCase OPE_BENCH_SQUARE_CASES[] = {
  {"sq_8", 8, 8, 8},
  {"sq_16", 16, 16, 16},
  {"sq_32", 32, 32, 3},
  {"sq_64", 64, 64, 64 },
  {"sq_128", 128, 128, 128 },
  {"sq_192", 192, 192, 192 },
  {"sq_256", 256, 256, 256 },
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
};
const int OPE_BENCH_NUM_RECT_CASES =
  (int)(sizeof(OPE_BENCH_RECT_CASES)/sizeof(OPE_BENCH_RECT_CASES[0]));

const OpeSizeCase OPE_BENCH_SPECIAL_CASES[] = {
  {"special_8x8", 8, 8, 8},
  {"special_16x16", 16, 16, 16},
  {"special_32x32", 32, 32, 32},
  {"special_64x64", 64, 64, 64},
};
const int OPE_BENCH_NUM_SPECIAL_CASES =
  (int)(sizeof(OPE_BENCH_SPECIAL_CASES)/sizeof(OPE_BENCH_SPECIAL_CASES[0]));
