/*
 * bench_impl.c - Assembly-kernel benchmark loop for rvv-conv cases.
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_config.h"
#include "bench_impl.h"

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

  int out3_h;
  int out3_w;
  int out5_h;
  int out5_w;

  size_t in_elems;
  size_t out3_elems;
  size_t out5_elems;

  int8_t *input_i8;
  float *input_f32;

  int16_t *output_i16_3x3;
  int16_t *output_i16_5x5;
  float *output_f32_3x3;
  float *output_f32_5x5;

  int8_t *kernel_i8_3x3;
  int8_t *kernel_i8_5x5;
  float *kernel_f32_3x3;
  float *kernel_f32_5x5;
} conv_case_ctx_t;

typedef void (*conv_kernel_fn_t)(const conv_case_ctx_t *ctx);

/* 3x3 float kernel from src/vec-conv.S */
extern void vec_conv_f32_3x3(size_t rows,
                             size_t cols,
                             size_t a_stride,
                             size_t b_stride,
                             const float *k,
                             const float *a,
                             float *b);

/* 3x3 int8->int16 kernel from src/vec-conv-i8.S */
extern void vec_conv_i8_3x3(size_t rows,
                            size_t cols,
                            size_t a_stride,
                            size_t b_stride,
                            const int8_t *k,
                            const int8_t *a,
                            int16_t *b);

/* 5x5 float kernel from src/vec-conv5x5.S */
extern void vec_conv_f32_5x5(size_t rows,
                             size_t cols,
                             size_t a_stride,
                             size_t b_stride,
                             const float *k,
                             const float *a,
                             float *b);

/* 5x5 int8->int16 kernel from src/vec-conv5x5-i8.S */
extern void vec_conv_i8_5x5(size_t rows,
                            size_t cols,
                            size_t a_stride,
                            size_t b_stride,
                            const int8_t *k,
                            const int8_t *a,
                            int16_t *b);

static uint8_t g_cache_thrash[CONV_BENCH_CACHE_THRASH_BYTES]
    __attribute__((aligned(CONV_BENCH_CACHE_LINE_BYTES)));

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

static void print_stats_line(const char *tag,
                             const bench_stats_t *cold,
                             const bench_stats_t *hot) {
  if (cold->runs > 0 && hot->runs > 0) {
    printf("  %-16s COLD(runs=%d best=%llu avg=%llu) HOT(runs=%d best=%llu avg=%llu)\n",
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
    printf("  %-16s COLD(runs=%d best=%llu avg=%llu) HOT(runs=0)\n",
           tag,
           cold->runs,
           (unsigned long long)cold->best,
           (unsigned long long)bench_stats_avg(cold));
    return;
  }

  if (hot->runs > 0) {
    printf("  %-16s COLD(runs=0) HOT(runs=%d best=%llu avg=%llu)\n",
           tag,
           hot->runs,
           (unsigned long long)hot->best,
           (unsigned long long)bench_stats_avg(hot));
    return;
  }

  printf("  %-16s COLD(runs=0) HOT(runs=0)\n", tag);
}

static void print_disabled_line(const char *tag, const char *reason) {
  printf("  %-16s DISABLED (%s)\n", tag, reason);
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

static void bench_cache_init(void) {
  for (size_t i = 0; i < sizeof(g_cache_thrash); ++i) {
    g_cache_thrash[i] = (uint8_t)(i ^ 0xA5u);
  }
}

static void bench_cache_flush(void) {
  volatile uint8_t *p = (volatile uint8_t *)g_cache_thrash;
  for (size_t i = 0; i < sizeof(g_cache_thrash); i += CONV_BENCH_CACHE_LINE_BYTES) {
    p[i] ^= 0x5Au;
  }
  asm volatile("fence rw, rw" ::: "memory");
}

static void fill_inputs(conv_case_ctx_t *ctx) {
  for (size_t i = 0; i < ctx->in_elems; ++i) {
    int32_t v = (int32_t)((i * 13u + 17u) % 31u) - 15;
    ctx->input_i8[i] = (int8_t)v;
    ctx->input_f32[i] = (float)v * 0.125f;
  }
}

static void fill_kernels(conv_case_ctx_t *ctx) {
  static const int8_t k3_base[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1,
  };

  static const int8_t k5_base[25] = {
    1, 1, 2, 1, 1,
    1, 2, 3, 2, 1,
    2, 3, 4, 3, 2,
    1, 2, 3, 2, 1,
    1, 1, 2, 1, 1,
  };

  for (int ch = 0; ch < ctx->channels; ++ch) {
    for (int i = 0; i < 9; ++i) {
      int8_t k = (int8_t)(k3_base[i] + (int8_t)(ch & 1));
      ctx->kernel_i8_3x3[(size_t)ch * 9u + (size_t)i] = k;
      ctx->kernel_f32_3x3[(size_t)ch * 9u + (size_t)i] = (float)k * 0.125f;
    }

    for (int i = 0; i < 25; ++i) {
      int8_t k = (int8_t)(k5_base[i] - (int8_t)(ch % 3));
      ctx->kernel_i8_5x5[(size_t)ch * 25u + (size_t)i] = k;
      ctx->kernel_f32_5x5[(size_t)ch * 25u + (size_t)i] = (float)k * 0.0625f;
    }
  }
}

static void conv_case_ctx_destroy(conv_case_ctx_t *ctx) {
  if (ctx->input_i8) free(ctx->input_i8);
  if (ctx->input_f32) free(ctx->input_f32);

  if (ctx->output_i16_3x3) free(ctx->output_i16_3x3);
  if (ctx->output_i16_5x5) free(ctx->output_i16_5x5);
  if (ctx->output_f32_3x3) free(ctx->output_f32_3x3);
  if (ctx->output_f32_5x5) free(ctx->output_f32_5x5);

  if (ctx->kernel_i8_3x3) free(ctx->kernel_i8_3x3);
  if (ctx->kernel_i8_5x5) free(ctx->kernel_i8_5x5);
  if (ctx->kernel_f32_3x3) free(ctx->kernel_f32_3x3);
  if (ctx->kernel_f32_5x5) free(ctx->kernel_f32_5x5);

  memset(ctx, 0, sizeof(*ctx));
}

static int conv_case_ctx_init(conv_case_ctx_t *ctx, const ConvBenchCase *cs) {
  memset(ctx, 0, sizeof(*ctx));

  ctx->batch = cs->batch;
  ctx->channels = cs->channels;
  ctx->height = cs->height;
  ctx->width = cs->width;

  if (ctx->batch <= 0 || ctx->channels <= 0 || ctx->height <= 0 || ctx->width <= 0) {
    printf("  ERROR: invalid case dimensions\n");
    return -1;
  }

  ctx->out3_h = conv_valid_out_dim(ctx->height, 3);
  ctx->out3_w = conv_valid_out_dim(ctx->width, 3);
  ctx->out5_h = conv_valid_out_dim(ctx->height, 5);
  ctx->out5_w = conv_valid_out_dim(ctx->width, 5);

  ctx->in_elems = (size_t)ctx->batch * (size_t)ctx->channels *
                  (size_t)ctx->height * (size_t)ctx->width;

  ctx->out3_elems = 0;
  if (ctx->out3_h > 0 && ctx->out3_w > 0) {
    ctx->out3_elems = (size_t)ctx->batch * (size_t)ctx->channels *
                      (size_t)ctx->out3_h * (size_t)ctx->out3_w;
  }

  ctx->out5_elems = 0;
  if (ctx->out5_h > 0 && ctx->out5_w > 0) {
    ctx->out5_elems = (size_t)ctx->batch * (size_t)ctx->channels *
                      (size_t)ctx->out5_h * (size_t)ctx->out5_w;
  }

  ctx->input_i8 = (int8_t *)bench_aligned_alloc(64, ctx->in_elems * sizeof(int8_t));
  ctx->input_f32 = (float *)bench_aligned_alloc(64, ctx->in_elems * sizeof(float));

  ctx->kernel_i8_3x3 = (int8_t *)bench_aligned_alloc(64, (size_t)ctx->channels * 9u * sizeof(int8_t));
  ctx->kernel_i8_5x5 = (int8_t *)bench_aligned_alloc(64, (size_t)ctx->channels * 25u * sizeof(int8_t));
  ctx->kernel_f32_3x3 = (float *)bench_aligned_alloc(64, (size_t)ctx->channels * 9u * sizeof(float));
  ctx->kernel_f32_5x5 = (float *)bench_aligned_alloc(64, (size_t)ctx->channels * 25u * sizeof(float));

  if (ctx->out3_elems > 0) {
    ctx->output_i16_3x3 = (int16_t *)bench_aligned_alloc(64, ctx->out3_elems * sizeof(int16_t));
    ctx->output_f32_3x3 = (float *)bench_aligned_alloc(64, ctx->out3_elems * sizeof(float));
  }

  if (ctx->out5_elems > 0) {
    ctx->output_i16_5x5 = (int16_t *)bench_aligned_alloc(64, ctx->out5_elems * sizeof(int16_t));
    ctx->output_f32_5x5 = (float *)bench_aligned_alloc(64, ctx->out5_elems * sizeof(float));
  }

  if (!ctx->input_i8 || !ctx->input_f32 ||
      !ctx->kernel_i8_3x3 || !ctx->kernel_i8_5x5 ||
      !ctx->kernel_f32_3x3 || !ctx->kernel_f32_5x5 ||
      (ctx->out3_elems > 0 && (!ctx->output_i16_3x3 || !ctx->output_f32_3x3)) ||
      (ctx->out5_elems > 0 && (!ctx->output_i16_5x5 || !ctx->output_f32_5x5))) {
    printf("  ERROR: allocation failed\n");
    conv_case_ctx_destroy(ctx);
    return -1;
  }

  fill_inputs(ctx);
  fill_kernels(ctx);

  if (ctx->out3_elems > 0) {
    memset(ctx->output_i16_3x3, 0, ctx->out3_elems * sizeof(int16_t));
    memset(ctx->output_f32_3x3, 0, ctx->out3_elems * sizeof(float));
  }
  if (ctx->out5_elems > 0) {
    memset(ctx->output_i16_5x5, 0, ctx->out5_elems * sizeof(int16_t));
    memset(ctx->output_f32_5x5, 0, ctx->out5_elems * sizeof(float));
  }

  return 0;
}

#if CONV_BENCH_ENABLE_F32_3X3
static void run_f32_3x3(const conv_case_ctx_t *ctx) {
  const size_t in_plane = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = (size_t)ctx->out3_h * (size_t)ctx->out3_w;

  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      const float *in = ctx->input_f32 + idx * in_plane;
      float *out = ctx->output_f32_3x3 + idx * out_plane;
      const float *k = ctx->kernel_f32_3x3 + (size_t)c * 9u;

      vec_conv_f32_3x3((size_t)ctx->height,
                       (size_t)ctx->out3_w,
                       (size_t)ctx->width,
                       (size_t)ctx->out3_w,
                       k,
                       in,
                       out);
    }
  }
}
#endif

#if CONV_BENCH_ENABLE_F32_5X5
static void run_f32_5x5(const conv_case_ctx_t *ctx) {
  const size_t in_plane = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = (size_t)ctx->out5_h * (size_t)ctx->out5_w;

  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      const float *in = ctx->input_f32 + idx * in_plane;
      float *out = ctx->output_f32_5x5 + idx * out_plane;
      const float *k = ctx->kernel_f32_5x5 + (size_t)c * 25u;

      vec_conv_f32_5x5((size_t)ctx->height,
                       (size_t)ctx->out5_w,
                       (size_t)ctx->width,
                       (size_t)ctx->out5_w,
                       k,
                       in,
                       out);
    }
  }
}
#endif

#if CONV_BENCH_ENABLE_I8_3X3
static void run_i8_3x3(const conv_case_ctx_t *ctx) {
  const size_t in_plane = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = (size_t)ctx->out3_h * (size_t)ctx->out3_w;

  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      const int8_t *in = ctx->input_i8 + idx * in_plane;
      int16_t *out = ctx->output_i16_3x3 + idx * out_plane;
      const int8_t *k = ctx->kernel_i8_3x3 + (size_t)c * 9u;

      vec_conv_i8_3x3((size_t)ctx->height,
                      (size_t)ctx->out3_w,
                      (size_t)ctx->width,
                      (size_t)ctx->out3_w,
                      k,
                      in,
                      out);
    }
  }
}
#endif

#if CONV_BENCH_ENABLE_I8_5X5
static void run_i8_5x5(const conv_case_ctx_t *ctx) {
  const size_t in_plane = (size_t)ctx->height * (size_t)ctx->width;
  const size_t out_plane = (size_t)ctx->out5_h * (size_t)ctx->out5_w;

  for (int b = 0; b < ctx->batch; ++b) {
    for (int c = 0; c < ctx->channels; ++c) {
      size_t idx = (size_t)b * (size_t)ctx->channels + (size_t)c;
      const int8_t *in = ctx->input_i8 + idx * in_plane;
      int16_t *out = ctx->output_i16_5x5 + idx * out_plane;
      const int8_t *k = ctx->kernel_i8_5x5 + (size_t)c * 25u;

      vec_conv_i8_5x5((size_t)ctx->height,
                      (size_t)ctx->out5_w,
                      (size_t)ctx->width,
                      (size_t)ctx->out5_w,
                      k,
                      in,
                      out);
    }
  }
}
#endif

static void bench_run_kernel(const conv_case_ctx_t *ctx,
                             const char *tag,
                             conv_kernel_fn_t fn) {
  bench_stats_t cold;
  bench_stats_t hot;
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
  fn(ctx);  // warm-up run for hot state

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

#if CONV_BENCH_ENABLE_F32_3X3
  if (ctx.out3_h > 0 && ctx.out3_w > 0) {
    bench_run_kernel(&ctx, "f32_3x3", run_f32_3x3);
  } else {
    print_disabled_line("f32_3x3", "invalid output dims");
  }
#else
  print_disabled_line("f32_3x3", "CONV_BENCH_ENABLE_F32_3X3=0");
#endif

#if CONV_BENCH_ENABLE_F32_5X5
  if (ctx.out5_h > 0 && ctx.out5_w > 0) {
    bench_run_kernel(&ctx, "f32_5x5", run_f32_5x5);
  } else {
    print_disabled_line("f32_5x5", "invalid output dims");
  }
#else
  print_disabled_line("f32_5x5", "CONV_BENCH_ENABLE_F32_5X5=0");
#endif

#if CONV_BENCH_ENABLE_I8_3X3
  if (ctx.out3_h > 0 && ctx.out3_w > 0) {
    bench_run_kernel(&ctx, "i8_3x3", run_i8_3x3);
  } else {
    print_disabled_line("i8_3x3", "invalid output dims");
  }
#else
  print_disabled_line("i8_3x3", "CONV_BENCH_ENABLE_I8_3X3=0");
#endif

#if CONV_BENCH_ENABLE_I8_5X5
  if (ctx.out5_h > 0 && ctx.out5_w > 0) {
    bench_run_kernel(&ctx, "i8_5x5", run_i8_5x5);
  } else {
    print_disabled_line("i8_5x5", "invalid output dims");
  }
#else
  print_disabled_line("i8_5x5", "CONV_BENCH_ENABLE_I8_5X5=0");
#endif

  conv_case_ctx_destroy(&ctx);
}
