#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "conv_bench.h"
#include "conv_cases.h"
#include "hal_2d_conv.h"

typedef enum {
  CONV_CACHE_COLD = 0,
  CONV_CACHE_WARM_SRC,
  CONV_CACHE_WARM_DST,
  CONV_CACHE_WARM_BOTH,
  CONV_CACHE_HOT_REPEAT,
  CONV_CACHE_STATE_COUNT
} conv_cache_state_t;

typedef struct {
  uint64_t best_cycles;
  uint64_t sum_cycles;
  uint32_t runs;
} conv_stats_t;

typedef struct {
  const conv_bench_case_t *cs;
  uint16_t out_h;
  uint16_t out_w;
  size_t launches_per_run;
  size_t input_plane_bytes;
  size_t output_plane_bytes;
  int8_t *input;
  int16_t *output;
  int16_t *expected_output;
  int8_t kernel[25];
} conv_case_ctx_t;

static const char *k_cache_state_name[CONV_CACHE_STATE_COUNT] = {
    "COLD",
    "WARM_SRC",
    "WARM_DST",
    "WARM_BOTH",
    "HOT_REPEAT",
};

static uint8_t g_cache_evict[CONV_BENCH_CACHE_EVICT_BYTES]
    __attribute__((aligned(CONV_BENCH_CACHE_LINE_BYTES)));

static inline void conv_fence_all(void) {
  asm volatile("fence iorw, iorw" ::: "memory");
}

static inline void conv_mmio_write8(uintptr_t offset, uint8_t value) {
  *(volatile uint8_t *)(MMIO_BASE + offset) = value;
}

static inline uint8_t conv_mmio_read8(uintptr_t offset) {
  return *(volatile uint8_t *)(MMIO_BASE + offset);
}

static inline void conv_mmio_write64(uintptr_t offset, uint64_t value) {
  *(volatile uint64_t *)(MMIO_BASE + offset) = value;
}

static inline void conv_stats_init(conv_stats_t *stats) {
  stats->best_cycles = ULLONG_MAX;
  stats->sum_cycles = 0;
  stats->runs = 0;
}

static inline void conv_stats_update(conv_stats_t *stats, uint64_t cycles) {
  if (cycles < stats->best_cycles) {
    stats->best_cycles = cycles;
  }
  stats->sum_cycles += cycles;
  stats->runs += 1;
}

static inline uint64_t conv_stats_avg(const conv_stats_t *stats) {
  if (stats->runs == 0) {
    return 0;
  }
  return stats->sum_cycles / (uint64_t)stats->runs;
}

static inline size_t round_up(size_t value, size_t align) {
  return ((value + align - 1u) / align) * align;
}

static void *bench_aligned_alloc(size_t alignment, size_t size) {
  if (size == 0u) {
    return NULL;
  }
  return aligned_alloc(alignment, round_up(size, alignment));
}

static inline int conv_out_dim(int input, int kernel, int stride) {
  return (input - kernel) / stride + 1;
}

static void conv_fill_input(int8_t *buf, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i) {
    int v = (int)((i * 17u + 11u) & 0x7Fu);
    buf[i] = (int8_t)(v - 64);
  }
}

static void conv_fill_kernel(int8_t *kernel, uint8_t kernel_size) {
  const size_t elems = (size_t)kernel_size * (size_t)kernel_size;
  for (size_t i = 0; i < elems; ++i) {
    kernel[i] = (int8_t)((int)(i % 7u) - 3);
  }
  for (size_t i = elems; i < 25u; ++i) {
    kernel[i] = 0;
  }
}

static void conv_compute_reference_output(const conv_case_ctx_t *ctx, int16_t *dst) {
  const int in_w = (int)ctx->cs->width;
  const int out_w = (int)ctx->out_w;
  const int kernel = (int)CONV_BENCH_KERNEL_SIZE;
  const int stride = (int)CONV_BENCH_STRIDE;
  const bool use_relu = (CONV_BENCH_USE_RELU != 0u);

  for (int oh = 0; oh < (int)ctx->out_h; ++oh) {
    for (int ow = 0; ow < (int)ctx->out_w; ++ow) {
      int32_t acc = 0;
      const int in_row_base = oh * stride;
      const int in_col_base = ow * stride;

      for (int kh = 0; kh < kernel; ++kh) {
        const int in_row = in_row_base + kh;
        const int k_row = kh * kernel;
        for (int kw = 0; kw < kernel; ++kw) {
          const int in_col = in_col_base + kw;
          int32_t a = (int32_t)ctx->input[in_row * in_w + in_col];
          int32_t b = (int32_t)ctx->kernel[k_row + kw];
          acc += a * b;
        }
      }

      if (use_relu && acc < 0) {
        acc = 0;
      }

      dst[oh * out_w + ow] = (int16_t)acc;
    }
  }
}

static bool conv_verify_output(const conv_case_ctx_t *ctx, size_t launch_idx) {
  const size_t out_elems = ctx->output_plane_bytes / sizeof(int16_t);
  for (size_t i = 0; i < out_elems; ++i) {
    int16_t got = ctx->output[i];
    int16_t exp = ctx->expected_output[i];
    if (got != exp) {
      const size_t oh = i / (size_t)ctx->out_w;
      const size_t ow = i % (size_t)ctx->out_w;
      printf("    output mismatch launch=%llu at (oh=%llu,ow=%llu): got=%d exp=%d\n",
             (unsigned long long)launch_idx,
             (unsigned long long)oh,
             (unsigned long long)ow,
             (int)got,
             (int)exp);
      return false;
    }
  }
  return true;
}

static void conv_case_ctx_destroy(conv_case_ctx_t *ctx) {
  if (ctx->input != NULL) {
    free(ctx->input);
  }
  if (ctx->output != NULL) {
    free(ctx->output);
  }
  if (ctx->expected_output != NULL) {
    free(ctx->expected_output);
  }
  memset(ctx, 0, sizeof(*ctx));
}

static int conv_case_ctx_init(conv_case_ctx_t *ctx, const conv_bench_case_t *cs) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->cs = cs;

  if (CONV_BENCH_KERNEL_SIZE != 3u && CONV_BENCH_KERNEL_SIZE != 5u) {
    printf("ERROR: CONV_BENCH_KERNEL_SIZE must be 3 or 5\n");
    return -1;
  }

  if (CONV_BENCH_STRIDE == 0u) {
    printf("ERROR: CONV_BENCH_STRIDE must be >= 1\n");
    return -1;
  }

  if ((int)cs->height < (int)CONV_BENCH_KERNEL_SIZE ||
      (int)cs->width < (int)CONV_BENCH_KERNEL_SIZE) {
    printf("ERROR: input smaller than kernel for case %s\n", cs->name);
    return -1;
  }

  ctx->out_h = (uint16_t)conv_out_dim((int)cs->height,
                                      (int)CONV_BENCH_KERNEL_SIZE,
                                      (int)CONV_BENCH_STRIDE);

  ctx->out_w = (uint16_t)conv_out_dim((int)cs->width,
                                      (int)CONV_BENCH_KERNEL_SIZE,
                                      (int)CONV_BENCH_STRIDE);

  if (ctx->out_h == 0u || ctx->out_w == 0u) {
    printf("ERROR: non-positive output shape for case %s\n", cs->name);
    return -1;
  }

  ctx->launches_per_run = (size_t)cs->batch_size * (size_t)cs->channels;
  ctx->input_plane_bytes = (size_t)cs->height * (size_t)cs->width * sizeof(int8_t);
  ctx->output_plane_bytes = (size_t)ctx->out_h * (size_t)ctx->out_w * sizeof(int16_t);

  // Reuse one input/output plane for each accelerator launch.
  ctx->input = (int8_t *)bench_aligned_alloc(CONV_BENCH_CACHE_LINE_BYTES, ctx->input_plane_bytes);
  ctx->output =
      (int16_t *)bench_aligned_alloc(CONV_BENCH_CACHE_LINE_BYTES, ctx->output_plane_bytes);
  ctx->expected_output =
      (int16_t *)bench_aligned_alloc(CONV_BENCH_CACHE_LINE_BYTES, ctx->output_plane_bytes);

  if (ctx->input == NULL || ctx->output == NULL || ctx->expected_output == NULL) {
    printf("ERROR: allocation failed for case %s\n", cs->name);
    conv_case_ctx_destroy(ctx);
    return -1;
  }

  conv_fill_input(ctx->input, ctx->input_plane_bytes);
  memset(ctx->output, 0, ctx->output_plane_bytes);
  conv_fill_kernel(ctx->kernel, CONV_BENCH_KERNEL_SIZE);
  conv_compute_reference_output(ctx, ctx->expected_output);
  return 0;
}

static void conv_stream_touch(volatile const uint8_t *buf, size_t bytes) {
  volatile uint8_t sink = 0;
  for (size_t i = 0; i < bytes; i += (size_t)CONV_BENCH_CACHE_LINE_BYTES) {
    sink ^= buf[i];
  }
  asm volatile("" : : "r"(sink) : "memory");
}

static void conv_evict_caches(void) {
  conv_stream_touch((volatile const uint8_t *)g_cache_evict, sizeof(g_cache_evict));
}

static void conv_prepare_cache_state(const conv_case_ctx_t *ctx, conv_cache_state_t state) {
  if (state == CONV_CACHE_HOT_REPEAT) {
    return;
  }

  conv_evict_caches();

  if (state == CONV_CACHE_WARM_SRC || state == CONV_CACHE_WARM_BOTH) {
    conv_stream_touch((volatile const uint8_t *)ctx->input, ctx->input_plane_bytes);
  }
  if (state == CONV_CACHE_WARM_DST || state == CONV_CACHE_WARM_BOTH) {
    conv_stream_touch((volatile const uint8_t *)ctx->output, ctx->output_plane_bytes);
  }

  conv_fence_all();
}

static uint64_t conv_stall_cycles(uint32_t cycles) {
  if (cycles == 0u) {
    return 0u;
  }
  const uint64_t start = conv_bench_rdcycle64();
  while ((conv_bench_rdcycle64() - start) < (uint64_t)cycles) {
    asm volatile("nop");
  }
  return conv_bench_rdcycle64() - start;
}

static void conv_hw_write_kernel(const int8_t *kernel, uint8_t kernel_size) {
  volatile uint8_t *kernel_mmio = (volatile uint8_t *)(MMIO_BASE + CONV2D_KERNEL_REG0_OFFSET);
  const size_t elems = (size_t)kernel_size * (size_t)kernel_size;

  for (size_t i = 0; i < elems; ++i) {
    kernel_mmio[i] = (uint8_t)kernel[i];
  }
  for (size_t i = elems; i < 25u; ++i) {
    kernel_mmio[i] = 0u;
  }
  conv_mmio_write8(CONV2D_KERNEL_SIZE_OFFSET, kernel_size);
}

static bool conv_wait_ready(void) {
  const uint64_t start = conv_bench_rdcycle64();
  while (conv_mmio_read8(CONV2D_READY_REG_OFFSET) == 0u) {
    if ((conv_bench_rdcycle64() - start) > CONV_BENCH_READY_TIMEOUT_CYCLES) {
      return false;
    }
  }
  return true;
}

static bool conv_run_workload(const conv_case_ctx_t *ctx, uint64_t *cycles_out,
                              uint8_t *status_out) {
  uint64_t total_cycles = 0;
  size_t launch_idx = 0;

  conv_mmio_write8(CONV2D_READY_REG_OFFSET, 1u);
  conv_hw_write_kernel(ctx->kernel, CONV_BENCH_KERNEL_SIZE);
  conv_mmio_write8(CONV2D_USE_RELU_OFFSET, (uint8_t)CONV_BENCH_USE_RELU);
  conv_mmio_write8(CONV2D_STRIDE_OFFSET, (uint8_t)CONV_BENCH_STRIDE);

  for (size_t b = 0; b < (size_t)ctx->cs->batch_size; ++b) {
    for (size_t ch = 0; ch < (size_t)ctx->cs->channels; ++ch) {
      const size_t idx = b * (size_t)ctx->cs->channels + ch;
      (void)idx;
      uintptr_t src_addr = (uintptr_t)ctx->input;
      uintptr_t dst_addr = (uintptr_t)ctx->output;

      conv_mmio_write64(CONV2D_SRC_ADDR_OFFSET, (uint64_t)src_addr);
      conv_mmio_write64(CONV2D_DEST_ADDR_OFFSET, (uint64_t)dst_addr);
      conv_mmio_write64(CONV2D_INPUT_HEIGHT_OFFSET, (uint64_t)ctx->cs->height);
      conv_mmio_write64(CONV2D_INPUT_WIDTH_OFFSET, (uint64_t)ctx->cs->width);

      if (launch_idx > 0u) {
        total_cycles += conv_stall_cycles(CONV_BENCH_INTER_RUN_STALL_CYCLES);
      }

      conv_fence_all();
      uint64_t t0 = conv_bench_rdcycle64();
      conv_mmio_write8(CONV2D_READY_REG_OFFSET, 0u);
      bool ready = conv_wait_ready();
      uint64_t t1 = conv_bench_rdcycle64();
      conv_fence_all();

      if (!ready) {
        *status_out = 0xFFu;
        return false;
      }

      total_cycles += (t1 - t0);

      uint8_t status = conv_mmio_read8(CONV2D_STATUS_REG_OFFSET);
      *status_out = status;

#if CONV_BENCH_VERIFY_STATUS
      if (status != 0u) {
        return false;
      }
#endif

#if CONV_BENCH_VERIFY_OUTPUT
      if (!conv_verify_output(ctx, launch_idx)) {
        *status_out = 0xFEu;
        return false;
      }
#endif

      launch_idx += 1;
    }
  }

  *cycles_out = total_cycles;
  return true;
}

static bool conv_run_state(const conv_case_ctx_t *ctx, conv_cache_state_t state, conv_stats_t *stats) {
  conv_stats_init(stats);

  if (state == CONV_CACHE_HOT_REPEAT) {
    conv_prepare_cache_state(ctx, CONV_CACHE_WARM_BOTH);

    uint64_t warm_cycles = 0;
    uint8_t warm_status = 0;
    if (!conv_run_workload(ctx, &warm_cycles, &warm_status)) {
      printf("    warmup failed (status=0x%02x)\n", warm_status);
      return false;
    }
    (void)warm_cycles;
  }

  for (uint32_t run = 0; run < CONV_BENCH_RUNS; ++run) {
    uint64_t run_stall_cycles = 0;
    if (state != CONV_CACHE_HOT_REPEAT) {
      conv_prepare_cache_state(ctx, state);
    } else if (run > 0u) {
      run_stall_cycles += conv_stall_cycles(CONV_BENCH_INTER_RUN_STALL_CYCLES);
    }

    uint64_t cycles = 0;
    uint8_t status = 0;
    if (!conv_run_workload(ctx, &cycles, &status)) {
      printf("    run %u failed (status=0x%02x)\n", run, status);
      return false;
    }

    conv_stats_update(stats, cycles + run_stall_cycles);
  }

  return true;
}

static bool conv_state_enabled(conv_cache_state_t state) {
  switch (state) {
    case CONV_CACHE_COLD:
#if CONV_BENCH_ENABLE_STATE_COLD
      return true;
#else
      return false;
#endif
    case CONV_CACHE_WARM_SRC:
#if CONV_BENCH_ENABLE_STATE_WARM_SRC
      return true;
#else
      return false;
#endif
    case CONV_CACHE_WARM_DST:
#if CONV_BENCH_ENABLE_STATE_WARM_DST
      return true;
#else
      return false;
#endif
    case CONV_CACHE_WARM_BOTH:
#if CONV_BENCH_ENABLE_STATE_WARM_BOTH
      return true;
#else
      return false;
#endif
    case CONV_CACHE_HOT_REPEAT:
#if CONV_BENCH_ENABLE_STATE_HOT_REPEAT
      return true;
#else
      return false;
#endif
    default:
      return false;
  }
}

static bool conv_bench_run_case(const conv_bench_case_t *cs) {
  conv_case_ctx_t ctx;
  if (conv_case_ctx_init(&ctx, cs) != 0) {
    return false;
  }

  printf("\n=== Case: %s ===\n", cs->name);
  printf("  B=%u C=%u H=%u W=%u -> out=%ux%u\n",
         cs->batch_size, cs->channels, cs->height, cs->width, ctx.out_h, ctx.out_w);
  printf("  launches/run=%llu, input_bytes=%llu, output_bytes=%llu\n",
         (unsigned long long)ctx.launches_per_run,
         (unsigned long long)ctx.input_plane_bytes,
         (unsigned long long)ctx.output_plane_bytes);

  bool case_ok = true;
  for (int state = 0; state < (int)CONV_CACHE_STATE_COUNT; ++state) {
    conv_cache_state_t cache_state = (conv_cache_state_t)state;
    if (!conv_state_enabled(cache_state)) {
      continue;
    }

    conv_stats_t stats;
    bool ok = conv_run_state(&ctx, cache_state, &stats);
    case_ok &= ok;

    uint64_t avg_cycles = conv_stats_avg(&stats);
    uint64_t best_per_launch = 0;
    uint64_t avg_per_launch = 0;
    if (ctx.launches_per_run > 0u && stats.runs > 0u) {
      best_per_launch = stats.best_cycles / (uint64_t)ctx.launches_per_run;
      avg_per_launch = avg_cycles / (uint64_t)ctx.launches_per_run;
    }

    printf("  %-10s runs=%u best=%10llu avg=%10llu  best/conv=%8llu avg/conv=%8llu  %s\n",
           k_cache_state_name[state],
           stats.runs,
           (unsigned long long)stats.best_cycles,
           (unsigned long long)avg_cycles,
           (unsigned long long)best_per_launch,
           (unsigned long long)avg_per_launch,
           ok ? "PASS" : "FAIL");
  }

  conv_case_ctx_destroy(&ctx);
  return case_ok;
}

void conv_bench_run_all(void) {
  printf("\n=== Conv Accelerator Benchmark ===\n");
  printf("  runs/state=%u, kernel=%ux%u, stride=%u, relu=%u\n",
         CONV_BENCH_RUNS,
         CONV_BENCH_KERNEL_SIZE,
         CONV_BENCH_KERNEL_SIZE,
         CONV_BENCH_STRIDE,
         CONV_BENCH_USE_RELU);
  printf("  inter_run_stall_cycles=%u (included in timed cycles)\n",
         CONV_BENCH_INTER_RUN_STALL_CYCLES);
  printf("  verify_status=%u verify_output=%u\n",
         CONV_BENCH_VERIFY_STATUS,
         CONV_BENCH_VERIFY_OUTPUT);

  bool all_ok = true;
  for (uint32_t i = 0; i < CONV_BENCH_NUM_CASES; ++i) {
    all_ok &= conv_bench_run_case(&CONV_BENCH_CASES[i]);
  }

  printf("\n=== Conv Accelerator Benchmark Complete: %s ===\n", all_ok ? "PASS" : "FAIL");
}
