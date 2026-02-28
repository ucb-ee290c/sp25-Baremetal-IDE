/*
 * bench_impl.c - Accelerator benchmark loop for acc-conv cases.
 *
 * For each case, runs one correctness check (accelerator vs scalar reference),
 * then cold/hot timing sweeps using the 2D convolution accelerator.
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_config.h"
#include "bench_impl.h"
#include "hal_2d_conv.h"

typedef struct {
  uint64_t sum;
  uint64_t best;
  int runs;
} bench_stats_t;

typedef struct {
  int batch;
  int channels;
  int height;
  int width;

  int out3_h, out3_w;
  int out5_h, out5_w;

  size_t in_elems;
  size_t out3_elems;
  size_t out5_elems;
  size_t out3_stride;  /* 64B-aligned per-call output stride (int16_t elements) */
  size_t out5_stride;

  int8_t  *input_i8;

  int16_t *output_acc_3x3;
  int16_t *output_ref_3x3;

  int16_t *output_acc_5x5;
  int16_t *output_ref_5x5;

  int8_t  *kernel_i8_3x3;
  int8_t  *kernel_i8_5x5;
} conv_case_ctx_t;

typedef void (*conv_kernel_fn_t)(const conv_case_ctx_t *ctx);

static uint8_t g_cache_thrash[CONV_BENCH_CACHE_THRASH_BYTES]
    __attribute__((aligned(CONV_BENCH_CACHE_LINE_BYTES)));

static void spin_cycles(uint64_t cycles) {
  if (cycles == 0u) return;
  uint64_t start = rdcycle64();
  while ((rdcycle64() - start) < cycles)
    asm volatile("" ::: "memory");
}

static void bench_stats_init(bench_stats_t *s) {
  s->sum  = 0;
  s->best = ULLONG_MAX;
  s->runs = 0;
}

static void bench_stats_update(bench_stats_t *s, uint64_t cycles) {
  s->sum += cycles;
  if (cycles < s->best) s->best = cycles;
  s->runs += 1;
}

static uint64_t bench_stats_avg(const bench_stats_t *s) {
  return s->runs ? s->sum / (uint64_t)s->runs : 0;
}

static void print_stats_line(const char *tag,
                             const bench_stats_t *cold,
                             const bench_stats_t *hot) {
  printf("  %-16s COLD(runs=%d best=%llu avg=%llu) HOT(runs=%d best=%llu avg=%llu)\n",
         tag,
         cold->runs, (unsigned long long)cold->best,
         (unsigned long long)bench_stats_avg(cold),
         hot->runs, (unsigned long long)hot->best,
         (unsigned long long)bench_stats_avg(hot));
}

static void *bench_aligned_alloc(size_t size) {
  if (size == 0) return NULL;
  size_t aligned = ((size + 63u) / 64u) * 64u;
  return aligned_alloc(64, aligned);
}

static void bench_cache_init(void) {
  for (size_t i = 0; i < sizeof(g_cache_thrash); ++i)
    g_cache_thrash[i] = (uint8_t)(i ^ 0xA5u);
}

static void bench_cache_flush(void) {
  volatile uint8_t *p = (volatile uint8_t *)g_cache_thrash;
  for (size_t i = 0; i < sizeof(g_cache_thrash); i += CONV_BENCH_CACHE_LINE_BYTES)
    p[i] ^= 0x5Au;
  asm volatile("fence rw, rw" ::: "memory");
}

static void fill_inputs(conv_case_ctx_t *ctx) {
  for (size_t i = 0; i < ctx->in_elems; ++i) {
    int32_t v = (int32_t)((i * 13u + 17u) % 31u) - 15;
    ctx->input_i8[i] = (int8_t)v;
  }
}

static void fill_kernels(conv_case_ctx_t *ctx) {
  static const int8_t k3[9]  = { -1,  0,  1, -2,  0,  2, -1,  0,  1 };
  static const int8_t k5[25] = {
     1,  1,  2,  1,  1,
     1,  2,  3,  2,  1,
     2,  3,  4,  3,  2,
     1,  2,  3,  2,  1,
     1,  1,  2,  1,  1,
  };
  for (int ch = 0; ch < ctx->channels; ++ch) {
    for (int i = 0; i < 9; ++i)
      ctx->kernel_i8_3x3[ch * 9 + i]  = (int8_t)(k3[i] + (int8_t)(ch & 1));
    for (int i = 0; i < 25; ++i)
      ctx->kernel_i8_5x5[ch * 25 + i] = (int8_t)(k5[i] - (int8_t)(ch % 3));
  }
}

static void conv_case_ctx_destroy(conv_case_ctx_t *ctx) {
  free(ctx->input_i8);
  free(ctx->output_acc_3x3);
  free(ctx->output_ref_3x3);
  free(ctx->output_acc_5x5);
  free(ctx->output_ref_5x5);
  free(ctx->kernel_i8_3x3);
  free(ctx->kernel_i8_5x5);
  memset(ctx, 0, sizeof(*ctx));
}

static int conv_case_ctx_init(conv_case_ctx_t *ctx, const ConvBenchCase *cs) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->batch    = cs->batch;
  ctx->channels = cs->channels;
  ctx->height   = cs->height;
  ctx->width    = cs->width;

  if (ctx->batch <= 0 || ctx->channels <= 0 ||
      ctx->height <= 0 || ctx->width <= 0) {
    printf("  ERROR: invalid case dimensions\n");
    return -1;
  }

  ctx->out3_h = conv_valid_out_dim(ctx->height, 3);
  ctx->out3_w = conv_valid_out_dim(ctx->width,  3);
  ctx->out5_h = conv_valid_out_dim(ctx->height, 5);
  ctx->out5_w = conv_valid_out_dim(ctx->width,  5);

  ctx->in_elems = (size_t)ctx->batch * (size_t)ctx->channels *
                  (size_t)ctx->height * (size_t)ctx->width;

  /* Round each per-call output plane up to a 64-byte boundary (32 int16_t elements)
   * so every back-to-back perform_convolution call writes to a distinct 64B-aligned address. */
  ctx->out3_stride = (ctx->out3_h > 0 && ctx->out3_w > 0)
      ? (((size_t)ctx->out3_h * (size_t)ctx->out3_w + 31u) / 32u) * 32u
      : 0u;
  ctx->out5_stride = (ctx->out5_h > 0 && ctx->out5_w > 0)
      ? (((size_t)ctx->out5_h * (size_t)ctx->out5_w + 31u) / 32u) * 32u
      : 0u;

  ctx->out3_elems = (size_t)ctx->batch * (size_t)ctx->channels * ctx->out3_stride;
  ctx->out5_elems = (size_t)ctx->batch * (size_t)ctx->channels * ctx->out5_stride;

  ctx->input_i8      = bench_aligned_alloc(ctx->in_elems * sizeof(int8_t));
  ctx->kernel_i8_3x3 = bench_aligned_alloc((size_t)ctx->channels * 9u);
  ctx->kernel_i8_5x5 = bench_aligned_alloc((size_t)ctx->channels * 25u);

  if (ctx->out3_elems > 0) {
    ctx->output_acc_3x3 = bench_aligned_alloc(ctx->out3_elems * sizeof(int16_t));
    ctx->output_ref_3x3 = bench_aligned_alloc(ctx->out3_elems * sizeof(int16_t));
  }
  if (ctx->out5_elems > 0) {
    ctx->output_acc_5x5 = bench_aligned_alloc(ctx->out5_elems * sizeof(int16_t));
    ctx->output_ref_5x5 = bench_aligned_alloc(ctx->out5_elems * sizeof(int16_t));
  }

  if (!ctx->input_i8 || !ctx->kernel_i8_3x3 || !ctx->kernel_i8_5x5 ||
      (ctx->out3_elems > 0 && (!ctx->output_acc_3x3 || !ctx->output_ref_3x3)) ||
      (ctx->out5_elems > 0 && (!ctx->output_acc_5x5 || !ctx->output_ref_5x5))) {
    printf("  ERROR: allocation failed\n");
    conv_case_ctx_destroy(ctx);
    return -1;
  }

  fill_inputs(ctx);
  fill_kernels(ctx);

  if (ctx->output_acc_3x3) memset(ctx->output_acc_3x3, 0, ctx->out3_elems * sizeof(int16_t));
  if (ctx->output_ref_3x3) memset(ctx->output_ref_3x3, 0, ctx->out3_elems * sizeof(int16_t));
  if (ctx->output_acc_5x5) memset(ctx->output_acc_5x5, 0, ctx->out5_elems * sizeof(int16_t));
  if (ctx->output_ref_5x5) memset(ctx->output_ref_5x5, 0, ctx->out5_elems * sizeof(int16_t));

  return 0;
}

static void run_acc_3x3(const conv_case_ctx_t *ctx) {
  const size_t in_plane  = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = ctx->out3_stride;
  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      const size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      perform_convolution(
          (uint64_t)(ctx->input_i8       + idx * in_plane),
          (uint64_t)(ctx->output_acc_3x3 + idx * out_plane),
          (uint16_t)ctx->height, (uint16_t)ctx->width,
          (uint8_t *)(ctx->kernel_i8_3x3 + (size_t)c * 9u),
          3u, 0u, 1u);
      spin_cycles(CONV_BENCH_INTER_CALL_CYCLES);
    }
  }
}

static void run_acc_5x5(const conv_case_ctx_t *ctx) {
  const size_t in_plane  = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = ctx->out5_stride;
  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      const size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      perform_convolution(
          (uint64_t)(ctx->input_i8       + idx * in_plane),
          (uint64_t)(ctx->output_acc_5x5 + idx * out_plane),
          (uint16_t)ctx->height, (uint16_t)ctx->width,
          (uint8_t *)(ctx->kernel_i8_5x5 + (size_t)c * 25u),
          5u, 0u, 1u);
      spin_cycles(CONV_BENCH_INTER_CALL_CYCLES);
    }
  }
}

static void run_ref_3x3(const conv_case_ctx_t *ctx) {
  const size_t in_plane  = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = ctx->out3_stride;
  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      const size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      convolve(ctx->input_i8       + idx * in_plane,
               (uint16_t)ctx->width, (uint16_t)ctx->height,
               ctx->kernel_i8_3x3 + (size_t)c * 9u,
               3u, 1u, 0u,
               ctx->output_ref_3x3 + idx * out_plane);
    }
  }
}

static void run_ref_5x5(const conv_case_ctx_t *ctx) {
  const size_t in_plane  = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = ctx->out5_stride;
  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      const size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      convolve(ctx->input_i8       + idx * in_plane,
               (uint16_t)ctx->width, (uint16_t)ctx->height,
               ctx->kernel_i8_5x5 + (size_t)c * 25u,
               5u, 1u, 0u,
               ctx->output_ref_5x5 + idx * out_plane);
    }
  }
}

static int verify_outputs(const int16_t *acc, const int16_t *ref,
                          size_t elems, const char *tag) {
  int errors = 0;
  for (size_t i = 0; i < elems; ++i) {
    if (acc[i] != ref[i]) {
      if (++errors <= 4)
        printf("  MISMATCH[%zu]: acc=%d ref=%d\n", i, (int)acc[i], (int)ref[i]);
    }
  }
  if (errors == 0)
    printf("  %-16s PASS (%zu elements)\n", tag, elems);
  else
    printf("  %-16s FAIL (%d/%zu mismatches)\n", tag, errors, elems);
  return errors;
}

static void bench_run_kernel(const conv_case_ctx_t *ctx,
                             const char *tag,
                             conv_kernel_fn_t fn) {
  bench_stats_t cold, hot;
  bench_stats_init(&cold);
  bench_stats_init(&hot);

  for (int r = 0; r < CONV_BENCH_RUNS_COLD; ++r) {
    bench_cache_flush();
    uint64_t t0 = rdcycle64();
    fn(ctx);
    uint64_t t1 = rdcycle64();
    bench_stats_update(&cold, t1 - t0);
  }

  bench_cache_flush();
  fn(ctx); /* warm-up run for hot state */

  for (int r = 0; r < CONV_BENCH_RUNS_HOT; ++r) {
    uint64_t t0 = rdcycle64();
    fn(ctx);
    uint64_t t1 = rdcycle64();
    bench_stats_update(&hot, t1 - t0);
  }

  print_stats_line(tag, &cold, &hot);
}

void bench_run_case(const ConvBenchCase *cs) {
  conv_case_ctx_t ctx;
  static bool cache_initialized = false;

  if (!cache_initialized) {
    bench_cache_init();
    cache_initialized = true;
  }

  printf("\n=== Case: %s ===\n", cs->name);
  printf("  B=%d C=%d H=%d W=%d\n", cs->batch, cs->channels, cs->height, cs->width);

  if (conv_case_ctx_init(&ctx, cs) != 0) {
    printf("  ERROR: failed to initialize case context\n");
    return;
  }

  printf("  out3x3=%dx%d out5x5=%dx%d\n", ctx.out3_h, ctx.out3_w, ctx.out5_h, ctx.out5_w);

  if (ctx.out3_elems > 0) {
    run_acc_3x3(&ctx);
    run_ref_3x3(&ctx);
    verify_outputs(ctx.output_acc_3x3, ctx.output_ref_3x3, ctx.out3_elems, "verify_i8_3x3");
    memset(ctx.output_acc_3x3, 0, ctx.out3_elems * sizeof(int16_t));
  }

  if (ctx.out5_elems > 0) {
    run_acc_5x5(&ctx);
    run_ref_5x5(&ctx);
    verify_outputs(ctx.output_acc_5x5, ctx.output_ref_5x5, ctx.out5_elems, "verify_i8_5x5");
    memset(ctx.output_acc_5x5, 0, ctx.out5_elems * sizeof(int16_t));
  }

  if (ctx.out3_elems > 0)
    bench_run_kernel(&ctx, "acc_i8_3x3", run_acc_3x3);
  if (ctx.out5_elems > 0)
    bench_run_kernel(&ctx, "acc_i8_5x5", run_acc_5x5);

  conv_case_ctx_destroy(&ctx);
}
