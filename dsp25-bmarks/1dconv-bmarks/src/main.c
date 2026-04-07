/*
 * 1D Convolution Accelerator Benchmark
 *
 * Follows the exact MMIO patterns from the verified working tests in
 * generators/dsp-1d-conv-sp25/baremetal_test/ (benchmark_64_wide.c,
 * test-simple.c, test-multiple.c).
 *
 * Key pattern (from benchmark_64_wide.c):
 *   1. Pre-load some input packets to INPUT_ADDR
 *   2. Set LENGTH, DILATION, write kernel to KERNEL_ADDR
 *   3. START = 1
 *   4. Read N/2 output packets, interleaving remaining input writes
 *   5. Between runs: START=0, CLEAR=1 (from test-multiple.c)
 *
 * Reference: "no-offset" convolution producing N outputs:
 *   output[i] = sum(input[i+j] * kernel[j], j=0..K-1)
 *   where input[x >= N] = 0 (zero-extend at end)
 */

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bench_config.h"
#include "chip_config.h"
#include "hal_conv.h"
#include "simple_setup.h"

/* ------------------------------------------------------------------ */
/*  Configuration                                                      */
/* ------------------------------------------------------------------ */

#ifndef BENCH_TARGET_FREQ_HZ
#define BENCH_TARGET_FREQ_HZ 500000000ULL
#endif

#ifndef BENCH_NUM_RUNS
#define BENCH_NUM_RUNS 5u
#endif

#ifndef BENCH_ABS_TOL
#define BENCH_ABS_TOL 0.01f
#endif

#ifndef BENCH_REL_TOL
#define BENCH_REL_TOL 0.01f
#endif

/* Input sizes to sweep (must be even). */
static const uint32_t k_input_sizes[] = { 16, 32, 64, 128, 256, 512, 1024 };

/* Kernel sizes to try.  16 works on logMaxKernelSize=4 chips.
 * Add 8u to also test K=8 mode. */
static const uint32_t k_kernel_sizes[] = { CONV_BENCH_K_LIST };

#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0]))

#define MAX_N 1024u
#define MAX_K 16u
#define MAX_HW_OUT (MAX_N + MAX_K)

/* ------------------------------------------------------------------ */
/*  Buffers                                                            */
/* ------------------------------------------------------------------ */

static uint32_t g_input[MAX_N]       __attribute__((aligned(64)));
static uint32_t g_kernel[MAX_K]      __attribute__((aligned(64)));
static uint32_t g_hw_out[MAX_HW_OUT] __attribute__((aligned(64)));
static float    g_ref_out[MAX_N];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

static inline float bits_to_f32(uint32_t x) {
  union { uint32_t u; float f; } c;
  c.u = x;
  return c.f;
}

static inline uint32_t f32_to_bits(float x) {
  union { float f; uint32_t u; } c;
  c.f = x;
  return c.u;
}

static inline uint64_t read_cycles(void) {
  uint64_t c;
  asm volatile("rdcycle %0" : "=r"(c));
  return c;
}

static inline void fence_rw(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

/* ------------------------------------------------------------------ */
/*  Golden reference — "no-offset" convolution (N outputs)             */
/*  Matches benchmark_64_wide.c / test-multiple.c reference.           */
/* ------------------------------------------------------------------ */

static void ref_convolution(const uint32_t *arr, uint32_t n,
                            const uint32_t *kernel, uint32_t k,
                            float *output) {
  for (uint32_t i = 0; i < n; i++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; j++) {
      uint32_t idx = i + j;
      uint32_t item = (idx < n) ? arr[idx] : 0u;
      acc += bits_to_f32(item) * bits_to_f32(kernel[j]);
    }
    output[i] = acc;
  }
}

/* ------------------------------------------------------------------ */
/*  Raw MMIO convolution — follows working test patterns exactly       */
/* ------------------------------------------------------------------ */

/*
 * Perform one HW convolution via raw MMIO, following the exact sequence
 * from benchmark_64_wide.c:
 *
 *   1. Pre-load ~25% of input packets into INPUT FIFO
 *   2. Write LENGTH, DILATION, KERNEL
 *   3. START=1
 *   4. Read N/2 output packets, interleaving remaining input writes 1:1
 *
 * Reads N output values (N/2 packets) — the valid "no-offset" results.
 * Leaves the trailing K/2 HW packets in the FIFO (cleared after).
 */
static void hw_convolution(const uint32_t *input, uint32_t n,
                           const uint32_t *kernel, uint32_t k,
                           uint32_t *output) {
  uint32_t in_packets  = n / 2;
  uint32_t out_packets = in_packets; /* read N values only */

  /* Pre-load fraction of input before setting params (like working tests).
   * benchmark_64_wide.c pre-loads N/4 elements = N/8 packets for N=256.
   * For small N (≤ 16), pre-load everything. */
  uint32_t pre_load = in_packets;
  if (pre_load > 32u) pre_load = 32u;

  /* 1. Pre-load input */
  for (uint32_t i = 0; i < pre_load; i++) {
    reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR),
                *((uint64_t *)(input + i * 2)));
  }

  /* 2. Set LENGTH, DILATION, KERNEL */
  reg_write32((uintptr_t)(MMIO_BASE + CONV_LENGTH_ADDR), n);
  reg_write16((uintptr_t)(MMIO_BASE + CONV_DILATION_ADDR), 1);

  for (uint32_t i = 0; i < k; i += 2) {
    reg_write64((uintptr_t)(MMIO_BASE + CONV_KERNEL_ADDR),
                *((uint64_t *)(kernel + i)));
  }

  /* 3. START */
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 1);

  /* 4. Read output packets, interleave remaining input 1:1 */
  uint32_t in_pkt = pre_load;
  for (uint32_t i = 0; i < out_packets; i++) {
    uint64_t out64 = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
    uint32_t *up = (uint32_t *)&out64;
    output[i * 2]     = up[0];
    output[i * 2 + 1] = up[1];

    if (in_pkt < in_packets) {
      reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR),
                  *((uint64_t *)(input + in_pkt * 2)));
      in_pkt++;
    }
  }
}

/*
 * Reset between runs — follows test-multiple.c pattern:
 *   START=0, drain leftover output, CLEAR=1
 */
static void hw_reset(void) {
  reg_write8((uintptr_t)(MMIO_BASE + CONV_START_ADDR), 0);
  fence_rw();

  /* Drain any leftover output packets. */
  for (uint32_t i = 0; i < 256u; i++) {
    if (get_register_out_count() == 0u) break;
    (void)reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
  }

  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 1);
  fence_rw();
}

/* ------------------------------------------------------------------ */
/*  Generate simple test data                                          */
/* ------------------------------------------------------------------ */

static void fill_ramp_input(uint32_t *buf, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    /* Small floats: 0.1, 0.2, 0.3, ... */
    buf[i] = f32_to_bits(0.1f * (float)(i + 1));
  }
}

static void fill_kernel_from_generated(uint32_t *dst, uint32_t k,
                                       uint32_t dataset) {
  uint32_t ds = dataset % CONV_BENCH_GENERATED_NUM_DATASETS;
  for (uint32_t i = 0; i < k; i++) {
    dst[i] = g_conv_bench_generated_kernels[ds][i];
  }
}

/* ------------------------------------------------------------------ */
/*  Run one benchmark case                                             */
/* ------------------------------------------------------------------ */

static void run_case(uint32_t n, uint32_t k, uint32_t dataset) {
  if (n > MAX_N || k > MAX_K || (n & 1u) || (k != 8u && k != 16u)) {
    return;
  }

  /* Prepare data */
  fill_ramp_input(g_input, n);
  fill_kernel_from_generated(g_kernel, k, dataset);

  /* Initial reset, then release CLEAR before starting */
  hw_reset();
  fence_rw();

  /* Correctness check (single run) */
  reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 0);
  fence_rw();
  hw_convolution(g_input, n, g_kernel, k, g_hw_out);

  /* Compute reference */
  ref_convolution(g_input, n, g_kernel, k, g_ref_out);

  /* Compare */
  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < n; i++) {
    float hw_f  = bits_to_f32(g_hw_out[i]);
    float ref_f = g_ref_out[i];
    float abs_err = fabsf(hw_f - ref_f);
    float denom = fabsf(ref_f) + 1.0e-9f;
    float rel_err = abs_err / denom;

    if (abs_err > BENCH_ABS_TOL && rel_err > BENCH_REL_TOL) {
      if (mismatches < 8u) {
        printf("  MISMATCH [%u] hw=%.6f ref=%.6f abs_err=%.6e\n",
               i, (double)hw_f, (double)ref_f, (double)abs_err);
      }
      mismatches++;
    }
  }

  if (mismatches > 0) {
    printf("  N=%u K=%u ds=%u: %u/%u mismatches\n", n, k, dataset, mismatches, n);
  }

  /* Performance measurement */
  uint64_t best = UINT64_MAX;
  for (uint32_t run = 0; run < BENCH_NUM_RUNS; run++) {
    hw_reset();
    fence_rw();
    reg_write8((uintptr_t)(MMIO_BASE + CONV_CLEAR_ADDR), 0);
    fence_rw();

    uint64_t t0 = read_cycles();
    hw_convolution(g_input, n, g_kernel, k, g_hw_out);
    uint64_t elapsed = read_cycles() - t0;

    if (elapsed < best) best = elapsed;
  }

  /* Compute naive reference timing */
  uint64_t ref_best = UINT64_MAX;
  for (uint32_t run = 0; run < BENCH_NUM_RUNS; run++) {
    uint64_t t0 = read_cycles();
    ref_convolution(g_input, n, g_kernel, k, g_ref_out);
    uint64_t elapsed = read_cycles() - t0;
    if (elapsed < ref_best) ref_best = elapsed;
  }

  double speedup = (best > 0) ? ((double)ref_best / (double)best) : 0.0;

  printf("BENCH N=%-4u K=%-2u ds=%u  hw_cycles=%-8" PRIu64
         "  cpu_cycles=%-8" PRIu64 "  speedup=%.2fx",
         n, k, dataset, best, ref_best, speedup);
  if (mismatches == 0) {
    printf("  [OK]\n");
  } else {
    printf("  [%u MISMATCHES]\n", mismatches);
  }
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  init_test(BENCH_TARGET_FREQ_HZ);

  printf("\n=== 1D Conv Benchmark ===\n");
  printf("Datasets: %u, Runs: %u, Tolerance: abs=%.4f rel=%.4f\n",
         (unsigned)CONV_BENCH_GENERATED_NUM_DATASETS,
         (unsigned)BENCH_NUM_RUNS,
         (double)BENCH_ABS_TOL,
         (double)BENCH_REL_TOL);

  for (uint32_t ki = 0; ki < ARRAY_LEN(k_kernel_sizes); ki++) {
    uint32_t k = k_kernel_sizes[ki];
    printf("\n--- Kernel size K=%u ---\n", k);

    for (uint32_t ds = 0; ds < CONV_BENCH_GENERATED_NUM_DATASETS; ds++) {
      for (uint32_t ni = 0; ni < ARRAY_LEN(k_input_sizes); ni++) {
        uint32_t n = k_input_sizes[ni];
        if (n > CONV_BENCH_GENERATED_MAX_N) continue;
        run_case(n, k, ds);
      }
    }
  }

  /* Final cleanup */
  hw_reset();

  printf("\n=== Done ===\n");
  return 0;
}
