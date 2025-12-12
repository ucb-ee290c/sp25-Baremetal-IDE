#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_impl.h"
#include "bench_fill.h"
#include "bench_cache.h"

typedef struct {
  ope_mat8_t *A;
  ope_mat8_t *B;
  ope_mat32_t *C;
  int32_t *C_ref;
} ope_case_ctx_t;

#ifndef OPE_OUT_FULL_TRANSPOSE
#define OPE_OUT_FULL_TRANSPOSE 0
#endif

#if OPE_EXT_FLIP == 1
// Hardware returns tiles transposed; fix tiles and optionally the whole matrix.
static void unflip_output(const ope_case_ctx_t *ctx,
                          int32_t *tile_scratch,
                          int32_t *full_scratch) {
  const int rowsU = ctx->C->rowsU;
  const int colsU = ctx->C->colsU;
  for (int tr = 0; tr < rowsU; tr += 8) {
    for (int tc = 0; tc < colsU; tc += 8) {
      // Transpose one 8x8 tile into scratch
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
          tile_scratch[c * 8 + r] = ctx->C->data[(tr + r) * colsU + (tc + c)];
        }
      }
      // Copy back in original layout
      for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
          ctx->C->data[(tr + r) * colsU + (tc + c)] = tile_scratch[r * 8 + c];
        }
      }
    }
  }
#if OPE_OUT_FULL_TRANSPOSE
  // If tile order was also flipped, transpose the full padded matrix.
  for (int i = 0; i < rowsU; ++i) {
    for (int j = 0; j < colsU; ++j) {
      full_scratch[j * rowsU + i] = ctx->C->data[i * colsU + j];
    }
  }
  memcpy(ctx->C->data, full_scratch, (size_t)rowsU * (size_t)colsU * sizeof(int32_t));
#endif
}
#else
static inline void unflip_output(const ope_case_ctx_t *ctx,
                                 int32_t *tile_scratch,
                                 int32_t *full_scratch) {
  (void)ctx; (void)tile_scratch; (void)full_scratch;
}
#endif

static int ope_case_ctx_init(ope_case_ctx_t *ctx, const OpeSizeCase *cs) {
  const int M = cs->M;
  const int N = cs->N;
  const int K = cs->K;

  ctx->A = ope_mat8_init(M, K, OPE_MAT_ZERO);
  ctx->B = ope_mat8_init(K, N, OPE_MAT_ZERO);
  ctx->C = ope_mat32_init(M, N, OPE_MAT_ZERO);
  if (!ctx->A || !ctx->B || !ctx->C) {
    printf("ERROR: ope_mat*_init failed\n");
    return -1;
  }

  ctx->C_ref = (int32_t *)malloc((size_t)M * (size_t)N * sizeof(int32_t));
  if (!ctx->C_ref) {
    printf("ERROR: malloc C_ref failed\n");
    return -1;
  }

  // Fill A and B with deterministic pattern, zero C and C_ref
  bench_fill_int8_pattern(ctx->A->data, M, K, ctx->A->colsU);
  bench_fill_int8_pattern(ctx->B->data, K, N, ctx->B->colsU);
  bench_fill_int32_zero(ctx->C->data, M, ctx->C->colsU, ctx->C->colsU);
  memset(ctx->C_ref, 0, (size_t)M * (size_t)N * sizeof(int32_t));

  // Scalar CPU reference C_ref (MxN, ldc = N)
  bench_ref_gemm_AT_i8i8_i32(
      ctx->A->data, ctx->B->data, ctx->C_ref,
      M, N, K,
      ctx->A->colsU, ctx->B->colsU, N);

  return 0;
}

static void ope_case_ctx_destroy(ope_case_ctx_t *ctx) {
  if (ctx->A) ope_mat8_free(ctx->A);
  if (ctx->B) ope_mat8_free(ctx->B);
  if (ctx->C) ope_mat32_free(ctx->C);
  if (ctx->C_ref) free(ctx->C_ref);
  memset(ctx, 0, sizeof(*ctx));
}

// Specialized helper for 16/32/64 using remap buffers.
static long run_special_remap_kernel(ope_impl_kind_t impl,
                                     const OpeSizeCase *cs,
                                     const ope_case_ctx_t *ctx) {
  const int M = cs->M;
  const int N = cs->N;
  const int K = cs->K;

  if (!(M == N && N == K)) {
    // Only valid for square; fall back.
    return ope_matmul_arb(ctx->A, ctx->B, ctx->C);
  }

  int tile;
  switch (impl) {
    case OPE_IMPL_SPECIAL_16: tile = 16; break;
    case OPE_IMPL_SPECIAL_32: tile = 32; break;
    case OPE_IMPL_SPECIAL_64: tile = 64; break;
    default: return ope_matmul_arb(ctx->A, ctx->B, ctx->C);
  }

  if (M != tile) {
    return ope_matmul_arb(ctx->A, ctx->B, ctx->C);
  }

  const int kU = ctx->A->colsU;
  const size_t bytes = (size_t)kU * (size_t)kU * sizeof(int8_t);

  int8_t *A_T = (int8_t *)aligned_alloc(8, bytes);
  int8_t *B_remap = (int8_t *)aligned_alloc(8, bytes);
  if (!A_T || !B_remap) {
    printf("WARN: remap alloc failed, falling back to arb\n");
    if (A_T) free(A_T);
    if (B_remap) free(B_remap);
    return ope_matmul_arb(ctx->A, ctx->B, ctx->C);
  }

  ope_remap_matrix_A(ctx->A, A_T);
  ope_remap_matrix_B(ctx->B, B_remap);

  long cycles = 0;
  switch (tile) {
    case 16: cycles = ope_matmul_16x16(A_T, B_remap, ctx->C); break;
    case 32: cycles = ope_matmul_32x32(A_T, B_remap, ctx->C); break;
    case 64: cycles = ope_matmul_64x64(A_T, B_remap, ctx->C); break;
    default: cycles = ope_matmul_arb(ctx->A, ctx->B, ctx->C); break;
  }

  free(A_T);
  free(B_remap);
  return cycles;
}

static long run_impl_once(ope_impl_kind_t impl,
                          const OpeSizeCase *cs,
                          ope_case_ctx_t *ctx) {
  const int M = cs->M;
  const int N = cs->N;
  const int K = cs->K;

  switch (impl) {
    case OPE_IMPL_ARB:
      return ope_matmul_arb(ctx->A, ctx->B, ctx->C);

    case OPE_IMPL_SQUARE:
      if (M == N && N == K) {
        return ope_matmul_square(ctx->A, ctx->B, ctx->C);
      } else {
        return ope_matmul_arb(ctx->A, ctx->B, ctx->C);
      }

    case OPE_IMPL_SPECIAL_8:
      if (M == 8 && N == 8 && K == 8) {
        return ope_matmul_8x8(ctx->A, ctx->B, ctx->C);
      } else {
        return ope_matmul_arb(ctx->A, ctx->B, ctx->C);
      }

    case OPE_IMPL_SPECIAL_16:
    case OPE_IMPL_SPECIAL_32:
    case OPE_IMPL_SPECIAL_64:
      return run_special_remap_kernel(impl, cs, ctx);
    default:
      return ope_matmul_arb(ctx->A, ctx->B, ctx->C);
  }
}

void bench_run_case(const OpeSizeCase *cs, ope_impl_kind_t impl) {
  const int M = cs->M;
  const int N = cs->N;
  const int K = cs->K;

  printf("\n=== Case: %s | Impl: %s ===\n", cs->name, ope_impl_kind_name(impl));
  printf("Dims: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

  ope_case_ctx_t ctx;
  memset(&ctx, 0, sizeof(ctx));
  if (ope_case_ctx_init(&ctx, cs) != 0) {
    printf("  ERROR: Failed to init case context\n");
    return;
  }

  int32_t tile_scratch[64];
  int32_t *unflip_full_buf = NULL;
#if OPE_EXT_FLIP == 1
  size_t unflip_elems = (size_t)ctx.C->rowsU * (size_t)ctx.C->colsU;
#if OPE_OUT_FULL_TRANSPOSE
  unflip_full_buf = (int32_t *)aligned_alloc(8, unflip_elems * sizeof(int32_t));
  if (!unflip_full_buf) {
    printf("  ERROR: Failed to alloc full unflip buffer\n");
    ope_case_ctx_destroy(&ctx);
    return;
  }
#endif
#endif

  // bench_cache_flush();
  bench_fill_int32_zero(ctx.C->data, ctx.C->rows, ctx.C->colsU, ctx.C->colsU);

  long sanity_cycles_ope = 0;
  long sanity_cycles_total = 0;

  {
    uint64_t t0 = rdcycle64();
    sanity_cycles_ope = run_impl_once(impl, cs, &ctx);
    uint64_t t1 = rdcycle64();
    sanity_cycles_total = (long)(t1 - t0);
  }

  unflip_output(&ctx, tile_scratch, unflip_full_buf);
  int errors = bench_compare_results(ctx.C->data, ctx.C->colsU, 
                                     ctx.C_ref, N, M, N, 1);

  if (errors != 0) {
    printf("  Correctness run FAILED; Aborting further runs for this case.\n");
    ope_case_ctx_destroy(&ctx);
    return;
  }

  printf("  Correctness run PASS. OPE cycles=%ld, total cycles=%ld\n", sanity_cycles_ope, sanity_cycles_total);

  // COLD runs: flush cache before each measurement
  long cold_best_ope = LONG_MAX, cold_best_total = LONG_MAX;
  long cold_sum_ope = 0, cold_sum_total = 0;
  int cold_runs = 0;

  for (int r = 0; r < BENCH_RUNS_COLD; ++r) {
    // bench_cache_flush();
    bench_fill_int32_zero(ctx.C->data, ctx.C->rows, ctx.C->colsU, ctx.C->colsU);

    uint64_t t0 = rdcycle64();
    long cycles_ope = run_impl_once(impl, cs, &ctx);
    uint64_t t1 = rdcycle64();
    long cycles_total = (long)(t1 - t0);

    unflip_output(&ctx, tile_scratch, unflip_full_buf);
    int errs = bench_compare_results(ctx.C->data, ctx.C->colsU,
                                     ctx.C_ref, N, M, N, 0);

    if (errs != 0) {
      printf("  WARNING: cold run %d mismatch (%d errors); skipping stats\n", r, errs);
      continue;
    }

    if (cycles_ope   < cold_best_ope)   cold_best_ope   = cycles_ope;
    if (cycles_total < cold_best_total) cold_best_total = cycles_total;
    cold_sum_ope   += cycles_ope;
    cold_sum_total += cycles_total;
    ++cold_runs;
  }

  if (cold_runs > 0) {
    printf("  COLD: runs=%d, best_ope=%ld, best_total=%ld, avg_ope=%ld, avg_total=%ld\n",
           cold_runs,
           cold_best_ope, cold_best_total,
           cold_sum_ope / cold_runs,
           cold_sum_total / cold_runs);
  } else {
    printf("  COLD: no valid runs\n");
  }

  // HOT runs: no cache flush between runs
  long hot_best_ope = LONG_MAX, hot_best_total = LONG_MAX;
  long hot_sum_ope = 0, hot_sum_total = 0;
  int hot_runs = 0;

  for (int r = 0; r < BENCH_RUNS_HOT; ++r) {
    bench_fill_int32_zero(ctx.C->data, ctx.C->rows, ctx.C->colsU, ctx.C->colsU);

    uint64_t t0 = rdcycle64();
    long cycles_ope = run_impl_once(impl, cs, &ctx);
    uint64_t t1 = rdcycle64();
    long cycles_total = (long)(t1 - t0);

    unflip_output(&ctx, tile_scratch, unflip_full_buf);
    int errs = bench_compare_results(ctx.C->data, ctx.C->colsU,
                                    ctx.C_ref, N, M, N, 0);

    if (errs != 0) {
      printf("  WARNING: hot run %d mismatch (%d errors); skipping stats\n", r, errs);
      continue;
    }

    if (cycles_ope < hot_best_ope)   hot_best_ope   = cycles_ope;
    if (cycles_total < hot_best_total) hot_best_total = cycles_total;
    hot_sum_ope += cycles_ope;
    hot_sum_total += cycles_total;
    ++hot_runs;
  }

  if (hot_runs > 0) {
    printf("  HOT:  runs=%d, best_ope=%ld, best_total=%ld, avg_ope=%ld, avg_total=%ld\n",
           hot_runs,
           hot_best_ope, hot_best_total,
           hot_sum_ope / hot_runs,
           hot_sum_total / hot_runs);
  } else {
    printf("  HOT:  no valid runs\n");
  }

  if (unflip_full_buf) free(unflip_full_buf);
  ope_case_ctx_destroy(&ctx);
}
