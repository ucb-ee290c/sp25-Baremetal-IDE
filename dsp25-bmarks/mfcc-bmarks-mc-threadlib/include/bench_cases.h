#ifndef MFCC_BENCH_CASES_H
#define MFCC_BENCH_CASES_H

#include "mfcc_driver.h"

#define MFCC_BENCH_NUM_CASES 8
#define MFCC_BENCH_PRINT_INPUT_N 16

typedef struct {
  const char *name;
  float32_t samples[MFCC_DRIVER_FFT_LEN];
} mfcc_bench_case_t;

void mfcc_bench_prepare_cases(mfcc_bench_case_t *cases, uint32_t num_cases);

#endif
