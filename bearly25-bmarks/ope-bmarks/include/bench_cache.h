/*
 * bench_cache.h - Cache warm/flush helpers for OPE benchmarks.
 *
 * Used to thrash L2 between cold runs.
 */
#ifndef BENCH_CACHE_H
#define BENCH_CACHE_H

#include "bench_config.h"

// Initialize any cache-related state.
void bench_cache_init(void);

// Thrash L2 by writing to a buffer larger than the cache
void bench_cache_flush(void);

#endif // BENCH_CACHE_H
