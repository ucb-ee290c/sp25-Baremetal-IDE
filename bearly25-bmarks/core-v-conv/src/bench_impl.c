/*
 * bench_impl.c - Core benchmark loop for core-v-conv cases.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "bench_impl.h"
#include "bench_fill.h"
#include "bench_config.h"

#if BENCH_HAS_VECNN
#include "layers.h"
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
  int H;
  int W;
  int out_H;
  int out_W;
  int channels;
  size_t input_bytes;
  size_t output_bytes;
  size_t weight_bytes;
  int8_t *input;
  int8_t *output;
  int8_t *output_ref;
  void *weights;
  float *scale;
  int kernel_dim;
  int data_bytes;
  Type data_type; // FLOAT or INT right now
} conv_case_ctx_t;

static void conv_case_ctx_destroy(conv_case_ctx_t *ctx) {
  if (ctx->input) free(ctx->input);
  if (ctx->output) free(ctx->output);
  if (ctx->output_ref) free(ctx->output_ref);
  if (ctx->weights) free(ctx->weights);
  if (ctx->scale) free(ctx->scale);
  memset(ctx, 0, sizeof(*ctx));
}

static int conv_case_ctx_init(conv_case_ctx_t *ctx, const ConvSizeCase *cs) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->H = cs->H;
  ctx->W = cs->W;
  ctx->channels = CONV_CHANNELS;
  ctx->out_H = conv_out_dim(ctx->H, cs->kernel_dim, CONV_STRIDE, CONV_PADDING);
  ctx->out_W = conv_out_dim(ctx->W, cs->kernel_dim, CONV_STRIDE, CONV_PADDING);
  ctx->kernel_dim = cs->kernel_dim;
  ctx->data_type = cs->data_type;
  ctx->data_bytes = cs->data_bytes;

  if (ctx->out_H <= 0 || ctx->out_W <= 0) {
    printf("ERROR: invalid output dims for H=%d W=%d\n", ctx->H, ctx->W);
    return -1;
  }

  ctx->input_bytes = (size_t)ctx->channels * (size_t)ctx->H * (size_t)ctx->W * cs->data_bytes;
  ctx->output_bytes = (size_t)ctx->channels * (size_t)ctx->out_H * (size_t)ctx->out_W * cs->data_bytes;
  ctx->weight_bytes = (size_t)ctx->channels * (sizeof(int32_t) + cs->kernel_dim*cs->kernel_dim * cs->data_bytes);

  ctx->input = (int8_t *)bench_aligned_alloc(8, ctx->input_bytes);
  ctx->output = (int8_t *)bench_aligned_alloc(8, ctx->output_bytes);
  ctx->output_ref = (int8_t *)bench_aligned_alloc(8, ctx->output_bytes);
  ctx->weights = bench_aligned_alloc(8, ctx->weight_bytes);
  ctx->scale = (float *)bench_aligned_alloc(8, (size_t)ctx->channels * sizeof(float));

  if (!ctx->input || !ctx->output || !ctx->output_ref || !ctx->weights || !ctx->scale) {
    printf("ERROR: allocation failed\n");
    conv_case_ctx_destroy(ctx);
    return -1;
  }

  switch(cs->data_type) {
    case INT:
      bench_fill_int8_pattern(ctx->input, ctx->channels, ctx->H, ctx->W);
      bench_fill_int8_pattern(ctx->weights, 1, ctx->kernel_dim, ctx->kernel_dim);
      break;
    case FLOAT:
      bench_fill_float_pattern((void*)ctx->input, ctx->channels, ctx->H, ctx->W, cs->data_bytes);
      bench_fill_float_pattern((void*)ctx->weights, 1, ctx->kernel_dim, ctx->kernel_dim, cs->data_bytes);
      break;
  }
  bench_fill_int8_zero(ctx->output, ctx->output_bytes);
  bench_fill_int8_zero(ctx->output_ref, ctx->output_bytes);
  for (int ch = 0; ch < ctx->channels; ++ch) {
    ctx->scale[ch] = 1.0f;
  }

#if BENCH_VERIFY
  //TODO Float reference calculation (should basically just be changing this function from ints to floats)
  bench_ref_dwconv_i8(ctx->input,
                          ctx->H, ctx->W,
                          ctx->channels,
                          CONV_STRIDE, CONV_PADDING,
                          ctx->weights,
                          ctx->scale,
                          0,
                          ctx->output_ref,
			  ctx->kernel_dim);
#endif

  return 0;
}

void bench_run_case(const ConvSizeCase *cs) {
  printf("\n=== Case: %s ===\n", cs->name);
  printf("Input: %dx%d, C=%d, kernel=%dx%d, stride=%d, pad=%d\n",
         cs->H, cs->W, CONV_CHANNELS,
         cs->kernel_dim, cs->kernel_dim,
         CONV_STRIDE, CONV_PADDING);

#if BENCH_ENABLE_VEC
#if BENCH_HAS_VECNN
  conv_case_ctx_t ctx;
  if (conv_case_ctx_init(&ctx, cs) != 0) {
    printf("  ERROR: Failed to init case context\n");
    return;
  }

  requantization_params_t rqp;
  rqp.scale = ctx.scale;
  rqp.zero_point = 0;

#if BENCH_VERIFY
  //TODO Not modified for FP or variable sizes
  printf("  VEC correctness...");
  bench_fill_int8_zero(ctx.output, ctx.output_bytes);
  dwconv2D_3x3_int8(
      (size_t)ctx.H, (size_t)ctx.W,
      (size_t)ctx.channels,
      (size_t)CONV_STRIDE, (size_t)CONV_PADDING,
      ctx.weights,
      ctx.input,
      ctx.output,
      CONV_RELU,
      rqp);

  // TODO Float comparison
  int errs = bench_compare_i8(ctx.output, ctx.output_ref,
                              ctx.out_H, ctx.out_W,
                              ctx.channels,
                              1);
  if (errs != 0) {
    printf("FAIL\n  Correctness run FAILED; skipping timed runs.\n");
    conv_case_ctx_destroy(&ctx);
    return;
  }
  printf("PASS\n");
#endif

  bench_stats_t stats;
  bench_stats_init(&stats);
  for (int r = 0; r < BENCH_RUNS; ++r) {
    bench_fill_int8_zero(ctx.output, ctx.output_bytes);
    uint64_t t0 = rdcycle64();
    //TODO: Replace this function with the appropriate assembly function
    // https://stackoverflow.com/questions/15132185/mixing-c-and-assembly-sources-and-build-with-cmake
    dwconv2D_3x3_int8(
        (size_t)ctx.H, (size_t)ctx.W,
        (size_t)ctx.channels,
        (size_t)CONV_STRIDE, (size_t)CONV_PADDING,
        ctx.weights,
        ctx.input,
        ctx.output,
        CONV_RELU,
        rqp);
    uint64_t t1 = rdcycle64();
    bench_stats_update(&stats, t1 - t0);
  }

  if (stats.runs > 0) {
    uint64_t avg_total = stats.sum / (uint64_t)stats.runs;
    printf("  VEC: runs=%d, best_total=%llu, avg_total=%llu\n",
           stats.runs,
           (unsigned long long)stats.best,
           (unsigned long long)avg_total);
  } else {
    printf("  VEC: no valid runs\n");
  }

  conv_case_ctx_destroy(&ctx);
#else
  printf("  VEC: skipped (vecnn not built)\n");
#endif
#else
  printf("  VEC: disabled\n");
#endif
}
