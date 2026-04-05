#ifndef MFCC_BENCH_CACHE_H
#define MFCC_BENCH_CACHE_H

#include "bench_config.h"

void bench_cache_init(void);
void bench_cache_flush(uint32_t hart_id);

#endif
