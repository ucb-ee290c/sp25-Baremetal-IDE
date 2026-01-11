#ifndef BENCH_IMPL_H
#define BENCH_IMPL_H

#include "bench_config.h"
#include "bench_sizes.h"

// Run benchmarks for a single size case + implementation
// Prints cold/hot stats and checks correctness

// Implementation will:
//   * allocate A/B/C and a CPU reference C_ref
//   * fill inputs
//   * compute scalar reference once
//   * run BENCH_RUNS_COLD times with cache flush
//   * run BENCH_RUNS_HOT times without cache flush
//   * print min/avg cycles for both ope internal acc cycles and total cycles
void bench_run_case(const OpeSizeCase *cs, ope_impl_kind_t impl);

#endif // BENCH_IMPL_H
