#ifndef BENCH_SIZES_H
#define BENCH_SIZES_H

#include "bench_config.h"

// Square test sizes (M=N=K)
extern const OpeSizeCase OPE_BENCH_SQUARE_CASES[];
extern const int OPE_BENCH_NUM_SQUARE_CASES;

// Rectangular test sizes (M != N or weird K)
extern const OpeSizeCase OPE_BENCH_RECT_CASES[];
extern const int OPE_BENCH_NUM_RECT_CASES;

// Special tile sizes that line up with the specialized functions
extern const OpeSizeCase OPE_BENCH_SPECIAL_CASES[];
extern const int OPE_BENCH_NUM_SPECIAL_CASES;

#endif // BENCH_SIZES_H
