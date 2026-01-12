/*
 * bench_impl.c - Core benchmark loop for core-v-ope outer product cases.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "bench_impl.h"
#include "bench_fill.h"
#include "bench_config.h"
#include "hal_ope.h"

#if BENCH_HAS_VECNN
#include "layers.h"
#endif

#if BENCH_HAS_VECNN
void int8_qgemm(
    size_t M, size_t N, size_t K,
    const int8_t* A, size_t a_row_stride,
    const int8_t* B,
    int8_t* C, size_t c_row_stride,
    size_t c_col_stride,
    requantization_params_t requant_params);
#endif

typedef struct {
  uint64_t sum;
  uint64_t best;
  int runs;
} bench_stats_t;

static void bench_stats_init(bench_stats_t *stats) {
  stats->sum = 0;
  stats->best = UINT64_MAX;
  stats->runs = 0;
}

static void bench_stats_update(bench_stats_t *stats, uint64_t cycles) {
  stats->sum += cycles;
  if (cycles < stats->best) {
    stats->best = cycles;
  }
  stats->runs += 1;
}

static void *bench_aligned_alloc(size_t alignment, size_t size) {
  if (size == 0) {
    return NULL;
  }
  size_t rounded = (size + alignment - 1) / alignment;
  rounded *= alignment;
  return aligned_alloc(alignment, rounded);
}

typedef struct {
  int M;
  int N;
  int K;
#if BENCH_ENABLE_OPE
  ope_mat8_t *ope_A;
  ope_mat8_t *ope_B;
  ope_mat32_t *ope_C;
#endif
  int32_t *C_ref;
#if BENCH_HAS_VECNN
  int8_t *vec_A;
  int8_t *vec_B;        // Original B matrix (K x N)
  int8_t *vec_B_packed; // Packed B for vecnn: (K+1) x N with zero bias row
  int8_t *vec_C;
  int8_t *vec_C_ref;
  float *vec_scale;
#endif
} bench_case_ctx_t;

#ifndef OPE_OUT_FULL_TRANSPOSE
#define OPE_OUT_FULL_TRANSPOSE 0
#endif

#if BENCH_ENABLE_OPE
#if OPE_EXT_FLIP == 1
static void unflip_output(const bench_case_ctx_t *ctx,
                          int32_t *tile_scratch,
                          int32_t *full_scratch __attribute__((unused))) {
  const int rowsU = ctx->ope_C->rowsU;
  const int colsU = ctx->ope_C->colsU;
  for (int tr = 0; tr < rowsU; tr += 8) {
    for (int tc = 0; tc < colsU; tc += 8) {
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
          tile_scratch[c * 8 + r] = ctx->ope_C->data[(tr + r) * colsU + (tc + c)];
        }
      }
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
          ctx->ope_C->data[(tr + r) * colsU + (tc + c)] = tile_scratch[r * 8 + c];
        }
      }
    }
  }
#if OPE_OUT_FULL_TRANSPOSE
  for (int i = 0; i < rowsU; ++i) {
    for (int j = 0; j < colsU; ++j) {
      full_scratch[j * rowsU + i] = ctx->ope_C->data[i * colsU + j];
    }
  }
  memcpy(ctx->ope_C->data, full_scratch,
         (size_t)rowsU * (size_t)colsU * sizeof(int32_t));
#endif
}
#else
static inline void unflip_output(const bench_case_ctx_t *ctx,
                                 int32_t *tile_scratch,
                                 int32_t *full_scratch) {
  (void)ctx;
  (void)tile_scratch;
  (void)full_scratch;
}
#endif
#endif

static void bench_case_ctx_destroy(bench_case_ctx_t *ctx) {
#if BENCH_ENABLE_OPE
  if (ctx->ope_A) ope_mat8_free(ctx->ope_A);
  if (ctx->ope_B) ope_mat8_free(ctx->ope_B);
  if (ctx->ope_C) ope_mat32_free(ctx->ope_C);
#endif
  if (ctx->C_ref) free(ctx->C_ref);
#if BENCH_HAS_VECNN
  if (ctx->vec_A) free(ctx->vec_A);
  if (ctx->vec_B) free(ctx->vec_B);
  if (ctx->vec_B_packed) free(ctx->vec_B_packed);
  if (ctx->vec_C) free(ctx->vec_C);
  if (ctx->vec_C_ref) free(ctx->vec_C_ref);
  if (ctx->vec_scale) free(ctx->vec_scale);
#endif
  memset(ctx, 0, sizeof(*ctx));
}

static int bench_case_ctx_init(bench_case_ctx_t *ctx, const OuterSizeCase *cs) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->M = cs->M;
  ctx->N = cs->N;
  ctx->K = cs->K;

#if BENCH_ENABLE_OPE
  ctx->ope_A = ope_mat8_init(ctx->M, ctx->K, OPE_MAT_ZERO);
  ctx->ope_B = ope_mat8_init(ctx->K, ctx->N, OPE_MAT_ZERO);
  ctx->ope_C = ope_mat32_init(ctx->M, ctx->N, OPE_MAT_ZERO);
  if (!ctx->ope_A || !ctx->ope_B || !ctx->ope_C) {
    printf("ERROR: ope_mat*_init failed\n");
    bench_case_ctx_destroy(ctx);
    return -1;
  }
  bench_fill_int8_small(ctx->ope_A->data, ctx->M, ctx->K, ctx->ope_A->colsU);
  bench_fill_int8_small(ctx->ope_B->data, ctx->K, ctx->N, ctx->ope_B->colsU);
#endif

#if BENCH_HAS_VECNN && BENCH_ENABLE_VEC
  size_t size_A = (size_t)ctx->M * (size_t)ctx->K;
  size_t size_B = (size_t)ctx->K * (size_t)ctx->N;
  size_t size_B_packed = (size_t)(ctx->K + 1) * (size_t)ctx->N;  // Extra row for bias
  size_t size_C = (size_t)ctx->M * (size_t)ctx->N;
  ctx->vec_A = (int8_t *)bench_aligned_alloc(8, size_A * sizeof(int8_t));
  ctx->vec_B = (int8_t *)bench_aligned_alloc(8, size_B * sizeof(int8_t));
  ctx->vec_B_packed = (int8_t *)bench_aligned_alloc(8, size_B_packed * sizeof(int8_t));
  ctx->vec_C = (int8_t *)bench_aligned_alloc(8, size_C * sizeof(int8_t));
  ctx->vec_C_ref = (int8_t *)bench_aligned_alloc(8, size_C * sizeof(int8_t));
  ctx->vec_scale = (float *)bench_aligned_alloc(8, (size_t)ctx->N * sizeof(float));

  if (!ctx->vec_A || !ctx->vec_B || !ctx->vec_B_packed || !ctx->vec_C || !ctx->vec_C_ref || !ctx->vec_scale) {
    printf("ERROR: vec buffer allocation failed\n");
    bench_case_ctx_destroy(ctx);
    return -1;
  }

  bench_fill_int8_small(ctx->vec_A, ctx->M, ctx->K, ctx->K);
  bench_fill_int8_small(ctx->vec_B, ctx->K, ctx->N, ctx->N);
  // Pack B with zero bias row for vecnn int8_qgemm
  bench_pack_B_with_zero_bias(ctx->vec_B, ctx->K, ctx->N, ctx->N, ctx->vec_B_packed);
  bench_fill_int8_zero(ctx->vec_C, ctx->M, ctx->N, ctx->N);
  bench_fill_int8_zero(ctx->vec_C_ref, ctx->M, ctx->N, ctx->N);
  for (int j = 0; j < ctx->N; ++j) {
    ctx->vec_scale[j] = 1.0f;
  }
#endif

  ctx->C_ref = (int32_t *)malloc((size_t)ctx->M * (size_t)ctx->N * sizeof(int32_t));
  if (!ctx->C_ref) {
    printf("ERROR: malloc C_ref failed\n");
    bench_case_ctx_destroy(ctx);
    return -1;
  }

  const int8_t *ref_A = NULL;
  const int8_t *ref_B = NULL;
  int ldA = 0;
  int ldB = 0;

#if BENCH_HAS_VECNN && BENCH_ENABLE_VEC
  if (ctx->vec_A && ctx->vec_B) {
    ref_A = ctx->vec_A;
    ref_B = ctx->vec_B;
    ldA = ctx->K;
    ldB = ctx->N;
  }
#endif

#if BENCH_ENABLE_OPE
  if (!ref_A && ctx->ope_A && ctx->ope_B) {
    ref_A = ctx->ope_A->data;
    ref_B = ctx->ope_B->data;
    ldA = ctx->ope_A->colsU;
    ldB = ctx->ope_B->colsU;
  }
#endif

  if (!ref_A || !ref_B) {
    printf("ERROR: no reference inputs available\n");
    bench_case_ctx_destroy(ctx);
    return -1;
  }

  bench_ref_gemm_i8_i8_i32(ref_A, ref_B, ctx->C_ref,
                           ctx->M, ctx->N, ctx->K,
                           ldA, ldB, ctx->N);

#if BENCH_HAS_VECNN && BENCH_ENABLE_VEC
  bench_ref_quant_i32_to_i8(ctx->C_ref, ctx->vec_C_ref,
                            ctx->M, ctx->N,
                            ctx->N, ctx->N,
                            ctx->vec_scale, 0);
#endif

  return 0;
}

#if BENCH_ENABLE_OPE
static long run_ope_once(bench_case_ctx_t *ctx) {
  return ope_matmul_square(ctx->ope_A, ctx->ope_B, ctx->ope_C);
}
#endif

#if BENCH_HAS_VECNN && BENCH_ENABLE_VEC
static void run_vec_once(bench_case_ctx_t *ctx, requantization_params_t rqp) {
  // Use packed B which has zero bias row prepended
  int8_qgemm((size_t)ctx->M, (size_t)ctx->N, (size_t)ctx->K,
             ctx->vec_A, (size_t)ctx->K,
             ctx->vec_B_packed,
             ctx->vec_C, (size_t)ctx->N,
             (size_t)1,
             rqp);
}
#endif

void bench_run_case(const OuterSizeCase *cs) {
  printf("\n=== Case: %s ===\n", cs->name);
  printf("Dims: A(%dx%d) * B(%dx%d) => C(%dx%d)\n",
         cs->M, cs->K, cs->K, cs->N, cs->M, cs->N);

  bench_case_ctx_t ctx;
  if (bench_case_ctx_init(&ctx, cs) != 0) {
    printf("  ERROR: Failed to init case context\n");
    return;
  }

#if BENCH_ENABLE_OPE
  if (ctx.ope_A && ctx.ope_B && ctx.ope_C) {
    bool ope_can_run = true;
#if BENCH_VERIFY
    printf("  OPE correctness...");
    bench_fill_int32_zero(ctx.ope_C->data, ctx.ope_C->rows, ctx.ope_C->colsU, ctx.ope_C->colsU);
    long sanity_cycles = run_ope_once(&ctx);
    if (sanity_cycles < 0) {
      printf("FAIL\n  ERROR: OPE matmul failed\n");
      ope_can_run = false;
    } else {
      int32_t tile_scratch[64];
      int32_t *unflip_full_buf = NULL;
#if OPE_EXT_FLIP == 1
#if OPE_OUT_FULL_TRANSPOSE
      size_t unflip_elems = (size_t)ctx.ope_C->rowsU * (size_t)ctx.ope_C->colsU;
      unflip_full_buf = (int32_t *)bench_aligned_alloc(8, unflip_elems * sizeof(int32_t));
      if (!unflip_full_buf) {
        printf("FAIL\n  ERROR: Failed to alloc full unflip buffer\n");
        ope_can_run = false;
      }
#endif
#endif
      if (ope_can_run) {
        unflip_output(&ctx, tile_scratch, unflip_full_buf);
        int errs = bench_compare_i32(ctx.ope_C->data, ctx.ope_C->colsU,
                                     ctx.C_ref, ctx.N,
                                     ctx.M, ctx.N, 1);
        if (errs != 0) {
          printf("FAIL\n  Correctness run FAILED; skipping OPE runs.\n");
          ope_can_run = false;
        } else {
          printf("PASS\n");
        }
      }
      if (unflip_full_buf) free(unflip_full_buf);
      if (!ope_can_run) {
        printf("  OPE: skipped due to correctness failure\n");
      }
    }
#endif

    if (ope_can_run) {
      bench_stats_t total_stats;
      bench_stats_t ope_stats;
      bench_stats_init(&total_stats);
      bench_stats_init(&ope_stats);

      for (int r = 0; r < BENCH_RUNS; ++r) {
        bench_fill_int32_zero(ctx.ope_C->data, ctx.ope_C->rows, ctx.ope_C->colsU, ctx.ope_C->colsU);
        uint64_t t0 = rdcycle64();
        long cycles_ope = run_ope_once(&ctx);
        uint64_t t1 = rdcycle64();
        if (cycles_ope < 0) {
          printf("  WARNING: OPE run %d failed; skipping stats\n", r);
          continue;
        }
        bench_stats_update(&total_stats, t1 - t0);
        bench_stats_update(&ope_stats, (uint64_t)cycles_ope);
      }

      if (total_stats.runs > 0) {
        uint64_t avg_total = total_stats.sum / (uint64_t)total_stats.runs;
        uint64_t avg_ope = ope_stats.sum / (uint64_t)ope_stats.runs;
        printf("  OPE: runs=%d, best_total=%llu, avg_total=%llu, best_ope=%llu, avg_ope=%llu\n",
               total_stats.runs,
               (unsigned long long)total_stats.best,
               (unsigned long long)avg_total,
               (unsigned long long)ope_stats.best,
               (unsigned long long)avg_ope);
      } else {
        printf("  OPE: no valid runs\n");
      }
    }
  }
#endif

#if BENCH_ENABLE_VEC
#if BENCH_HAS_VECNN
  if (ctx.vec_A && ctx.vec_B && ctx.vec_C && ctx.vec_C_ref && ctx.vec_scale) {
    requantization_params_t rqp;
    rqp.scale = ctx.vec_scale;
    rqp.zero_point = 0;

    bool vec_can_run = true;
#if BENCH_VERIFY
    printf("  VEC correctness...");
    bench_fill_int8_zero(ctx.vec_C, ctx.M, ctx.N, ctx.N);
    run_vec_once(&ctx, rqp);
    int errs = bench_compare_i8(ctx.vec_C, ctx.N,
                                ctx.vec_C_ref, ctx.N,
                                ctx.M, ctx.N, 1);
    if (errs != 0) {
      printf("FAIL\n  Correctness run FAILED; skipping VEC runs.\n");
      vec_can_run = false;
    } else {
      printf("PASS\n");
    }
#endif

    if (vec_can_run) {
      bench_stats_t vec_stats;
      bench_stats_init(&vec_stats);
      for (int r = 0; r < BENCH_RUNS; ++r) {
        bench_fill_int8_zero(ctx.vec_C, ctx.M, ctx.N, ctx.N);
        uint64_t t0 = rdcycle64();
        run_vec_once(&ctx, rqp);
        uint64_t t1 = rdcycle64();
        bench_stats_update(&vec_stats, t1 - t0);
      }

      if (vec_stats.runs > 0) {
        uint64_t avg_vec = vec_stats.sum / (uint64_t)vec_stats.runs;
        printf("  VEC: runs=%d, best_total=%llu, avg_total=%llu\n",
               vec_stats.runs,
               (unsigned long long)vec_stats.best,
               (unsigned long long)avg_vec);
      } else {
        printf("  VEC: no valid runs\n");
      }
    } else {
      printf("  VEC: skipped due to correctness failure\n");
    }
  }
#else
  printf("  VEC: skipped (vecnn not built)\n");
#endif
#endif

  bench_case_ctx_destroy(&ctx);
}
