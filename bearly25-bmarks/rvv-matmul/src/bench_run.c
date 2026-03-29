/*
 * bench_run.c - Core benchmark loop for RVV matmul cases.
 *
 * Wraps existing RVV kernel implementations with cache-state sweeps and
 * repeated timing statistics.
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_cache.h"
#include "bench_impl.h"
#include "bench_kernels.h"

#if RVV_BENCH_ENABLE_MULTICORE
#include <hthread.h>
#endif

typedef struct {
  uint64_t sum;
  uint64_t best;
  int runs;
} bench_stats_t;

typedef struct {
  size_t M;
  size_t N;
  size_t K;
  size_t a_elems;
  size_t b_elems;
  size_t c_elems;

  float *A_f32;
  float *B_f32;
  float *B_f32_packed;
  float *C_f32;

  int8_t *A_i8;
  int8_t *B_i8;
  int8_t *B_i8_packed_i16;
  int8_t *B_i8_packed_i32;
  int16_t *C_i16;
  int32_t *C_i32;
} rvv_case_ctx_t;

#if RVV_BENCH_SYNC_HARTS >= 2u
static volatile uint32_t g_post_gemm_phase[RVV_BENCH_SYNC_HARTS];
#endif

static inline void rvv_post_gemm_barrier(void) {
#if RVV_BENCH_SYNC_HARTS >= 2u
  const uint32_t hart = rvv_bench_hart_id();
  if (hart >= RVV_BENCH_SYNC_HARTS) {
    return;
  }

  const uint32_t phase = g_post_gemm_phase[hart] + 1u;
  g_post_gemm_phase[hart] = phase;
  asm volatile("fence rw, rw" ::: "memory");

  for (uint32_t h = 0; h < RVV_BENCH_SYNC_HARTS; ++h) {
    if (h == hart) {
      continue;
    }
    while (g_post_gemm_phase[h] < phase) {
      asm volatile("nop");
    }
  }
  asm volatile("fence rw, rw" ::: "memory");
#endif
}

static inline void bench_stats_init(bench_stats_t *stats) {
  stats->sum = 0;
  stats->best = ULLONG_MAX;
  stats->runs = 0;
}

static inline void bench_stats_update(bench_stats_t *stats, uint64_t cycles) {
  stats->sum += cycles;
  if (cycles < stats->best) {
    stats->best = cycles;
  }
  stats->runs += 1;
}

static inline uint64_t bench_stats_avg(const bench_stats_t *stats) {
  if (stats->runs == 0) {
    return 0;
  }
  return stats->sum / (uint64_t)stats->runs;
}

static inline size_t round_up(size_t size, size_t alignment) {
  return ((size + alignment - 1u) / alignment) * alignment;
}

static void *bench_aligned_alloc(size_t alignment, size_t size) {
  if (size == 0) {
    return NULL;
  }
  return aligned_alloc(alignment, round_up(size, alignment));
}

static void rvv_case_ctx_destroy(rvv_case_ctx_t *ctx) {
  if (ctx->A_f32) free(ctx->A_f32);
  if (ctx->B_f32) free(ctx->B_f32);
  if (ctx->B_f32_packed) free(ctx->B_f32_packed);
  if (ctx->C_f32) free(ctx->C_f32);
  if (ctx->A_i8) free(ctx->A_i8);
  if (ctx->B_i8) free(ctx->B_i8);
  if (ctx->B_i8_packed_i16) free(ctx->B_i8_packed_i16);
  if (ctx->B_i8_packed_i32) free(ctx->B_i8_packed_i32);
  if (ctx->C_i16) free(ctx->C_i16);
  if (ctx->C_i32) free(ctx->C_i32);
  memset(ctx, 0, sizeof(*ctx));
}

static int rvv_case_ctx_init(rvv_case_ctx_t *ctx, const RvvMatmulCase *cs) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->M = cs->M;
  ctx->N = cs->N;
  ctx->K = cs->K;
  ctx->a_elems = ctx->M * ctx->K;
  ctx->b_elems = (ctx->K + 1u) * ctx->N;
  ctx->c_elems = ctx->M * ctx->N;

  ctx->A_f32 = (float *)bench_aligned_alloc(8, ctx->a_elems * sizeof(float));
  ctx->B_f32 = (float *)bench_aligned_alloc(8, ctx->b_elems * sizeof(float));
  ctx->B_f32_packed = (float *)bench_aligned_alloc(8, ctx->b_elems * sizeof(float));
  ctx->C_f32 = (float *)bench_aligned_alloc(8, ctx->c_elems * sizeof(float));

  ctx->A_i8 = (int8_t *)bench_aligned_alloc(8, ctx->a_elems * sizeof(int8_t));
  ctx->B_i8 = (int8_t *)bench_aligned_alloc(8, ctx->b_elems * sizeof(int8_t));
  ctx->B_i8_packed_i16 = (int8_t *)bench_aligned_alloc(8, ctx->b_elems * sizeof(int8_t));
  ctx->B_i8_packed_i32 = (int8_t *)bench_aligned_alloc(8, ctx->b_elems * sizeof(int8_t));
  ctx->C_i16 = (int16_t *)bench_aligned_alloc(8, ctx->c_elems * sizeof(int16_t));
  ctx->C_i32 = (int32_t *)bench_aligned_alloc(8, ctx->c_elems * sizeof(int32_t));

  if (!ctx->A_f32 || !ctx->B_f32 || !ctx->B_f32_packed || !ctx->C_f32 ||
      !ctx->A_i8 || !ctx->B_i8 || !ctx->B_i8_packed_i16 || !ctx->B_i8_packed_i32 ||
      !ctx->C_i16 || !ctx->C_i32) {
    if (rvv_bench_is_print_hart()) {
      printf("  ERROR: allocation failed\n");
    }
    rvv_case_ctx_destroy(ctx);
    return -1;
  }

  for (size_t i = 0; i < ctx->a_elems; ++i) {
    int32_t v = (int32_t)((i * 13u + 7u) % 127u) - 63;
    ctx->A_f32[i] = (float)v * 0.125f;
    ctx->A_i8[i] = (int8_t)v;
  }
  for (size_t i = 0; i < ctx->b_elems; ++i) {
    int32_t v = (int32_t)((i * 11u + 3u) % 127u) - 63;
    ctx->B_f32[i] = (float)v * 0.125f;
    ctx->B_i8[i] = (int8_t)v;
  }

  memset(ctx->C_f32, 0, ctx->c_elems * sizeof(float));
  memset(ctx->C_i16, 0, ctx->c_elems * sizeof(int16_t));
  memset(ctx->C_i32, 0, ctx->c_elems * sizeof(int32_t));

  pack_weight_matrix_f32(ctx->K, ctx->N, ctx->B_f32, ctx->B_f32_packed);
  pack_weight_matrix_i8i16(ctx->K, ctx->N, ctx->B_i8, ctx->B_i8_packed_i16);
  pack_weight_matrix_i8i32(ctx->K, ctx->N, ctx->B_i8, ctx->B_i8_packed_i32);

  return 0;
}

static void run_f32_once(const rvv_case_ctx_t *ctx, bool packed) {
  if (packed) {
    f32_gemm_packed(ctx->M, ctx->N, ctx->K,
                    ctx->A_f32, ctx->K,
                    ctx->B_f32_packed,
                    ctx->C_f32, ctx->N, 1);
  } else {
    f32_gemm(ctx->M, ctx->N, ctx->K,
             ctx->A_f32, ctx->K,
             ctx->B_f32,
             ctx->C_f32, ctx->N, 1);
  }
}

static void run_i16_once(const rvv_case_ctx_t *ctx, bool packed) {
  if (packed) {
    int8_int16_gemm_packed(ctx->M, ctx->N, ctx->K,
                           ctx->A_i8, ctx->K,
                           ctx->B_i8_packed_i16,
                           ctx->C_i16, ctx->N, 1);
  } else {
    int8_int16_gemm(ctx->M, ctx->N, ctx->K,
                    ctx->A_i8, ctx->K,
                    ctx->B_i8,
                    ctx->C_i16, ctx->N, 1);
  }
}

static void run_i32_once(const rvv_case_ctx_t *ctx, bool packed) {
  if (packed) {
    int8_gemm_packed(ctx->M, ctx->N, ctx->K,
                     ctx->A_i8, ctx->K,
                     ctx->B_i8_packed_i32,
                     ctx->C_i32, ctx->N, 1);
  } else {
    int8_gemm(ctx->M, ctx->N, ctx->K,
              ctx->A_i8, ctx->K,
              ctx->B_i8,
              ctx->C_i32, ctx->N, 1);
  }
}

#if RVV_BENCH_ENABLE_MULTICORE
typedef struct {
  size_t M;
  size_t N;
  size_t K;
  const int8_t *A;
  size_t a_row_stride;
  const int8_t *B;
  void *C;
  size_t c_row_stride;
  size_t c_col_stride;
  bool packed;
  bool widen_i16;  /* true = i8→i16, false = i8→i32 */
} mc_i8_worker_arg_t;

static void *mc_i8_worker(void *arg_) {
  mc_i8_worker_arg_t *arg = (mc_i8_worker_arg_t *)arg_;
  if (arg->widen_i16) {
    if (arg->packed) {
      int8_int16_gemm_packed(arg->M, arg->N, arg->K,
                              arg->A, arg->a_row_stride,
                              arg->B,
                              (int16_t *)arg->C, arg->c_row_stride,
                              arg->c_col_stride);
    } else {
      int8_int16_gemm(arg->M, arg->N, arg->K,
                      arg->A, arg->a_row_stride,
                      arg->B,
                      (int16_t *)arg->C, arg->c_row_stride,
                      arg->c_col_stride);
    }
  } else {
    if (arg->packed) {
      int8_gemm_packed(arg->M, arg->N, arg->K,
                       arg->A, arg->a_row_stride,
                       arg->B,
                       (int32_t *)arg->C, arg->c_row_stride,
                       arg->c_col_stride);
    } else {
      int8_gemm(arg->M, arg->N, arg->K,
                arg->A, arg->a_row_stride,
                arg->B,
                (int32_t *)arg->C, arg->c_row_stride,
                arg->c_col_stride);
    }
  }
  return NULL;
}

static void run_i16_once_mc(const rvv_case_ctx_t *ctx, bool packed) {
  mc_i8_worker_arg_t args[2];

  const size_t rows_h0 = RVV_BENCH_MC_ROWS_HART0;
  const size_t rows_h1 = RVV_BENCH_MC_ROWS_HART1;
  const int8_t *B = packed ? ctx->B_i8_packed_i16 : ctx->B_i8;

  args[0].M = rows_h0;
  args[0].N = ctx->N;
  args[0].K = ctx->K;
  args[0].A = ctx->A_i8;
  args[0].a_row_stride = ctx->K;
  args[0].B = B;
  args[0].C = ctx->C_i16;
  args[0].c_row_stride = ctx->N;
  args[0].c_col_stride = 1;
  args[0].packed = packed;
  args[0].widen_i16 = true;

  args[1].M = rows_h1;
  args[1].N = ctx->N;
  args[1].K = ctx->K;
  args[1].A = ctx->A_i8 + rows_h0 * ctx->K;
  args[1].a_row_stride = ctx->K;
  args[1].B = B;
  args[1].C = ctx->C_i16 + rows_h0 * ctx->N;
  args[1].c_row_stride = ctx->N;
  args[1].c_col_stride = 1;
  args[1].packed = packed;
  args[1].widen_i16 = true;

  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1, mc_i8_worker, &args[1]);
  (void)mc_i8_worker(&args[0]);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");
}

static void run_i32_once_mc(const rvv_case_ctx_t *ctx, bool packed) {
  mc_i8_worker_arg_t args[2];

  const size_t rows_h0 = RVV_BENCH_MC_ROWS_HART0;
  const size_t rows_h1 = RVV_BENCH_MC_ROWS_HART1;
  const int8_t *B = packed ? ctx->B_i8_packed_i32 : ctx->B_i8;

  args[0].M = rows_h0;
  args[0].N = ctx->N;
  args[0].K = ctx->K;
  args[0].A = ctx->A_i8;
  args[0].a_row_stride = ctx->K;
  args[0].B = B;
  args[0].C = ctx->C_i32;
  args[0].c_row_stride = ctx->N;
  args[0].c_col_stride = 1;
  args[0].packed = packed;
  args[0].widen_i16 = false;

  args[1].M = rows_h1;
  args[1].N = ctx->N;
  args[1].K = ctx->K;
  args[1].A = ctx->A_i8 + rows_h0 * ctx->K;
  args[1].a_row_stride = ctx->K;
  args[1].B = B;
  args[1].C = ctx->C_i32 + rows_h0 * ctx->N;
  args[1].c_row_stride = ctx->N;
  args[1].c_col_stride = 1;
  args[1].packed = packed;
  args[1].widen_i16 = false;

  asm volatile("fence rw, rw" ::: "memory");
  hthread_issue(1, mc_i8_worker, &args[1]);
  (void)mc_i8_worker(&args[0]);
  hthread_join(1);
  asm volatile("fence rw, rw" ::: "memory");
}
#endif /* RVV_BENCH_ENABLE_MULTICORE */

static void print_stats_line(const char *tag,
                             const bench_stats_t *cold,
                             const bench_stats_t *hot) {
  if (!rvv_bench_is_print_hart()) {
    return;
  }

  if (cold->runs > 0 && hot->runs > 0) {
    printf("  %-20s COLD(runs=%d best=%llu avg=%llu) HOT(runs=%d best=%llu avg=%llu)\n",
           tag,
           cold->runs,
           (unsigned long long)cold->best,
           (unsigned long long)bench_stats_avg(cold),
           hot->runs,
           (unsigned long long)hot->best,
           (unsigned long long)bench_stats_avg(hot));
    return;
  }

  if (cold->runs > 0) {
    printf("  %-20s COLD(runs=%d best=%llu avg=%llu) HOT(runs=0)\n",
           tag,
           cold->runs,
           (unsigned long long)cold->best,
           (unsigned long long)bench_stats_avg(cold));
    return;
  }

  if (hot->runs > 0) {
    printf("  %-20s COLD(runs=0) HOT(runs=%d best=%llu avg=%llu)\n",
           tag,
           hot->runs,
           (unsigned long long)hot->best,
           (unsigned long long)bench_stats_avg(hot));
    return;
  }

  printf("  %-20s COLD(runs=0) HOT(runs=0)\n", tag);
}

static void print_disabled_line(const char *tag, const char *reason) {
  if (!rvv_bench_is_print_hart()) {
    return;
  }
  printf("  %-20s DISABLED (%s)\n", tag, reason);
}

static void bench_run_f32_impl(const rvv_case_ctx_t *ctx, const char *tag, bool packed) {
  bench_stats_t cold;
  bench_stats_t hot;
  bench_stats_init(&cold);
  bench_stats_init(&hot);
  memset(ctx->C_f32, 0, ctx->c_elems * sizeof(float));

  for (int r = 0; r < RVV_BENCH_RUNS_COLD; ++r) {
    bench_cache_flush();

    uint64_t t0 = rdcycle64();
    run_f32_once(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&cold, t1 - t0);
  }

  bench_cache_flush();
  run_f32_once(ctx, packed);  // warm-up run
  rvv_post_gemm_barrier();

  for (int r = 0; r < RVV_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    run_f32_once(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&hot, t1 - t0);
  }

  print_stats_line(tag, &cold, &hot);
}

static void bench_run_i16_impl(const rvv_case_ctx_t *ctx, const char *tag, bool packed) {
  bench_stats_t cold;
  bench_stats_t hot;
  bench_stats_init(&cold);
  bench_stats_init(&hot);
  memset(ctx->C_i16, 0, ctx->c_elems * sizeof(int16_t));

  for (int r = 0; r < RVV_BENCH_RUNS_COLD; ++r) {
    bench_cache_flush();

    uint64_t t0 = rdcycle64();
    run_i16_once(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&cold, t1 - t0);
  }

  bench_cache_flush();
  run_i16_once(ctx, packed);  // warm-up run
  rvv_post_gemm_barrier();

  for (int r = 0; r < RVV_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    run_i16_once(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&hot, t1 - t0);
  }

  print_stats_line(tag, &cold, &hot);
}

static void bench_run_i32_impl(const rvv_case_ctx_t *ctx, const char *tag, bool packed) {
  bench_stats_t cold;
  bench_stats_t hot;
  bench_stats_init(&cold);
  bench_stats_init(&hot);
  memset(ctx->C_i32, 0, ctx->c_elems * sizeof(int32_t));

  for (int r = 0; r < RVV_BENCH_RUNS_COLD; ++r) {
    bench_cache_flush();

    uint64_t t0 = rdcycle64();
    run_i32_once(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&cold, t1 - t0);
  }

  bench_cache_flush();
  run_i32_once(ctx, packed);  // warm-up run
  rvv_post_gemm_barrier();

  for (int r = 0; r < RVV_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    run_i32_once(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&hot, t1 - t0);
  }

  print_stats_line(tag, &cold, &hot);
}

#if RVV_BENCH_ENABLE_MULTICORE
static void bench_run_i16_impl_mc(const rvv_case_ctx_t *ctx, const char *tag, bool packed) {
  bench_stats_t cold;
  bench_stats_t hot;
  bench_stats_init(&cold);
  bench_stats_init(&hot);
  memset(ctx->C_i16, 0, ctx->c_elems * sizeof(int16_t));

  for (int r = 0; r < RVV_BENCH_RUNS_COLD; ++r) {
    bench_cache_flush();

    uint64_t t0 = rdcycle64();
    run_i16_once_mc(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&cold, t1 - t0);
  }

  bench_cache_flush();
  run_i16_once_mc(ctx, packed);
  rvv_post_gemm_barrier();

  for (int r = 0; r < RVV_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    run_i16_once_mc(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&hot, t1 - t0);
  }

  print_stats_line(tag, &cold, &hot);
}

static void bench_run_i32_impl_mc(const rvv_case_ctx_t *ctx, const char *tag, bool packed) {
  bench_stats_t cold;
  bench_stats_t hot;
  bench_stats_init(&cold);
  bench_stats_init(&hot);
  memset(ctx->C_i32, 0, ctx->c_elems * sizeof(int32_t));

  for (int r = 0; r < RVV_BENCH_RUNS_COLD; ++r) {
    bench_cache_flush();

    uint64_t t0 = rdcycle64();
    run_i32_once_mc(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&cold, t1 - t0);
  }

  bench_cache_flush();
  run_i32_once_mc(ctx, packed);  // warm-up run
  rvv_post_gemm_barrier();

  for (int r = 0; r < RVV_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    run_i32_once_mc(ctx, packed);
    uint64_t t1 = rdcycle64();
    rvv_post_gemm_barrier();
    bench_stats_update(&hot, t1 - t0);
  }

  print_stats_line(tag, &cold, &hot);
}
#endif /* RVV_BENCH_ENABLE_MULTICORE */

void bench_run_case(const RvvMatmulCase *cs) {
  if (rvv_bench_is_print_hart()) {
    printf("\n=== Case: %s (M=%llu N=%llu K=%llu) ===\n",
           cs->name,
           (unsigned long long)cs->M,
           (unsigned long long)cs->N,
           (unsigned long long)cs->K);
  }

  rvv_case_ctx_t ctx;
  if (rvv_case_ctx_init(&ctx, cs) != 0) {
    if (rvv_bench_is_print_hart()) {
      printf("  ERROR: skipping case due to setup failure\n");
    }
    return;
  }

#if RVV_BENCH_ENABLE_F32
#if RVV_BENCH_ENABLE_UNPACKED
  bench_run_f32_impl(&ctx, "f32_gemm", false);
#else
  print_disabled_line("f32_gemm", "RVV_BENCH_ENABLE_UNPACKED=0");
#endif
#if RVV_BENCH_ENABLE_PACKED
  bench_run_f32_impl(&ctx, "f32_gemm_packed", true);
#else
  print_disabled_line("f32_gemm_packed", "RVV_BENCH_ENABLE_PACKED=0");
#endif
#else
  print_disabled_line("f32_gemm", "RVV_BENCH_ENABLE_F32=0");
  print_disabled_line("f32_gemm_packed", "RVV_BENCH_ENABLE_F32=0");
#endif

#if RVV_BENCH_ENABLE_I8_I16
#if RVV_BENCH_ENABLE_UNPACKED
  bench_run_i16_impl(&ctx, "i8_i16_gemm", false);
#else
  print_disabled_line("i8_i16_gemm", "RVV_BENCH_ENABLE_UNPACKED=0");
#endif
#if RVV_BENCH_ENABLE_PACKED
  bench_run_i16_impl(&ctx, "i8_i16_gemm_packed", true);
#else
  print_disabled_line("i8_i16_gemm_packed", "RVV_BENCH_ENABLE_PACKED=0");
#endif
#else
  print_disabled_line("i8_i16_gemm", "RVV_BENCH_ENABLE_I8_I16=0");
  print_disabled_line("i8_i16_gemm_packed", "RVV_BENCH_ENABLE_I8_I16=0");
#endif

#if RVV_BENCH_ENABLE_MULTICORE && RVV_BENCH_ENABLE_I8_I16
  if (cs->M == (RVV_BENCH_MC_ROWS_HART0 + RVV_BENCH_MC_ROWS_HART1)) {
#if RVV_BENCH_ENABLE_UNPACKED
    bench_run_i16_impl_mc(&ctx, "i8_i16_mc_gemm", false);
#else
    print_disabled_line("i8_i16_mc_gemm", "RVV_BENCH_ENABLE_UNPACKED=0");
#endif
#if RVV_BENCH_ENABLE_PACKED
    bench_run_i16_impl_mc(&ctx, "i8_i16_mc_packed", true);
#else
    print_disabled_line("i8_i16_mc_packed", "RVV_BENCH_ENABLE_PACKED=0");
#endif
  } else {
    if (rvv_bench_is_print_hart()) {
      printf("  i8_i16_mc: SKIPPED (M=%llu != %u+%u)\n",
             (unsigned long long)cs->M,
             RVV_BENCH_MC_ROWS_HART0, RVV_BENCH_MC_ROWS_HART1);
    }
  }
#endif

#if RVV_BENCH_ENABLE_I8_I32
#if RVV_BENCH_ENABLE_UNPACKED
  bench_run_i32_impl(&ctx, "i8_i32_gemm", false);
#else
  print_disabled_line("i8_i32_gemm", "RVV_BENCH_ENABLE_UNPACKED=0");
#endif
#if RVV_BENCH_ENABLE_PACKED
  bench_run_i32_impl(&ctx, "i8_i32_gemm_packed", true);
#else
  print_disabled_line("i8_i32_gemm_packed", "RVV_BENCH_ENABLE_PACKED=0");
#endif
#else
  print_disabled_line("i8_i32_gemm", "RVV_BENCH_ENABLE_I8_I32=0");
  print_disabled_line("i8_i32_gemm_packed", "RVV_BENCH_ENABLE_I8_I32=0");
#endif

#if RVV_BENCH_ENABLE_MULTICORE && RVV_BENCH_ENABLE_I8_I32
  if (cs->M == (RVV_BENCH_MC_ROWS_HART0 + RVV_BENCH_MC_ROWS_HART1)) {
#if RVV_BENCH_ENABLE_UNPACKED
    bench_run_i32_impl_mc(&ctx, "i8_i32_mc_gemm", false);
#else
    print_disabled_line("i8_i32_mc_gemm", "RVV_BENCH_ENABLE_UNPACKED=0");
#endif
#if RVV_BENCH_ENABLE_PACKED
    bench_run_i32_impl_mc(&ctx, "i8_i32_mc_packed", true);
#else
    print_disabled_line("i8_i32_mc_packed", "RVV_BENCH_ENABLE_PACKED=0");
#endif
  } else {
    if (rvv_bench_is_print_hart()) {
      printf("  i8_i32_mc: SKIPPED (M=%llu != %u+%u)\n",
             (unsigned long long)cs->M,
             RVV_BENCH_MC_ROWS_HART0, RVV_BENCH_MC_ROWS_HART1);
    }
  }
#endif

  rvv_case_ctx_destroy(&ctx);
}
