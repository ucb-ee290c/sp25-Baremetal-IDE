/*
 * 1D Convolution Benchmark — N=16384, K=8
 *
 * Two modes tested:
 *   1. MMIO  — HW accelerator with 1:1 interleaved write-before-read
 *   2. RVV   — pure software convolution using RVV vector intrinsics
 *
 * 10 warmup runs, then 1000 timed runs per mode.
 * Reports: ops, average cycles, best cycles, correctness.
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"
#include "simple_setup.h"

/* ------------------------------------------------------------------ */
/*  Config                                                             */
/* ------------------------------------------------------------------ */

#define INPUT_LEN    16384u
#define KERNEL_LEN   8u
#define VALID_OUT    (INPUT_LEN + (KERNEL_LEN - 1))
#define DRIVER_OUT   (INPUT_LEN + KERNEL_LEN)

#define IN_PACKETS   (INPUT_LEN / 2)
#define K_PACKETS    (KERNEL_LEN / 2)
#define OUT_PACKETS  (IN_PACKETS + K_PACKETS)

#define PRELOAD_PACKETS  8u

#define WARMUP_RUNS  10u
#define TIMED_RUNS   1000u

/* ------------------------------------------------------------------ */
/*  Buffers                                                            */
/* ------------------------------------------------------------------ */

static uint32_t g_input[INPUT_LEN]    __attribute__((aligned(64)));
static uint32_t g_kernel[KERNEL_LEN]  __attribute__((aligned(64)));
static uint32_t g_hw_out[DRIVER_OUT]  __attribute__((aligned(64)));
static float    g_ref_out[VALID_OUT];

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

/* ------------------------------------------------------------------ */
/*  Mode 1: MMIO with 1:1 interleaved write-before-read               */
/* ------------------------------------------------------------------ */

static void mmio_conv(void) {
  conv_init();
  conv_set_params_kernel_only(INPUT_LEN, 1, g_kernel, KERNEL_LEN);

  for (uint32_t i = 0; i < PRELOAD_PACKETS; i++)
    reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR),
                *((uint64_t *)(g_input + i * 2)));

  start_conv();

  uint32_t in_pkt = PRELOAD_PACKETS;
  for (uint32_t i = 0; i < OUT_PACKETS; i++) {
    if (in_pkt < IN_PACKETS) {
      reg_write64((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR),
                  *((uint64_t *)(g_input + in_pkt * 2)));
      in_pkt++;
    }

    uint64_t out64 = reg_read64((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
    uint32_t *up = (uint32_t *)&out64;
    g_hw_out[i * 2]     = up[0];
    g_hw_out[i * 2 + 1] = up[1];
  }
}

/* ------------------------------------------------------------------ */
/*  Mode 2: RVV software convolution                                   */
/*                                                                     */
/*  Pure CPU 1D convolution using RISC-V Vector (RVV) intrinsics.      */
/*  Vectorises over the output dimension with e32m4 (up to 32 FP32).   */
/*  Follows the same HW convention:                                    */
/*    output[i] = sum(input[i+j-(K-1)] * kernel[j], j=0..K-1)         */
/*    with zero-padding for out-of-range indices.                      */
/* ------------------------------------------------------------------ */

#include <riscv_vector.h>

/* Pre-converted kernel floats (set once in main, reused across calls) */
static float g_kf[KERNEL_LEN];

/* ------------------------------------------------------------------ */
/*  RVV conv core — computes outputs [start, end) with boundary handling */
/* ------------------------------------------------------------------ */

static void rvv_conv_range(uint32_t start, uint32_t end) {
  const float *in = (const float *)g_input;
  float *out = (float *)g_hw_out;

  const float k0 = g_kf[0], k1 = g_kf[1], k2 = g_kf[2], k3 = g_kf[3];
  const float k4 = g_kf[4], k5 = g_kf[5], k6 = g_kf[6], k7 = g_kf[7];

  /* Boundary at start: outputs where some taps go out of range */
  uint32_t bulk_start = start;
  if (bulk_start < KERNEL_LEN - 1)
    bulk_start = KERNEL_LEN - 1;

  uint32_t bulk_end = end;
  if (bulk_end > INPUT_LEN)
    bulk_end = INPUT_LEN;

  /* Scalar boundary at start */
  for (uint32_t i = start; i < bulk_start; i++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < KERNEL_LEN; j++) {
      int32_t idx = (int32_t)i + (int32_t)j - (int32_t)(KERNEL_LEN - 1);
      if (idx >= 0 && (uint32_t)idx < INPUT_LEN)
        acc += in[idx] * g_kf[j];
    }
    out[i] = acc;
  }

  /* Vectorised bulk — all loads guaranteed in-range */
  uint32_t i = bulk_start;
  while (i < bulk_end) {
    size_t vl = __riscv_vsetvl_e32m8(bulk_end - i);

    const float *base = in + i - 7;
    vfloat32m8_t vacc = __riscv_vfmul_vf_f32m8(
        __riscv_vle32_v_f32m8(base, vl), k0, vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k1, __riscv_vle32_v_f32m8(base + 1, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k2, __riscv_vle32_v_f32m8(base + 2, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k3, __riscv_vle32_v_f32m8(base + 3, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k4, __riscv_vle32_v_f32m8(base + 4, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k5, __riscv_vle32_v_f32m8(base + 5, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k6, __riscv_vle32_v_f32m8(base + 6, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k7, __riscv_vle32_v_f32m8(base + 7, vl), vl);

    __riscv_vse32_v_f32m8(out + i, vacc, vl);
    i += vl;
  }

  /* Scalar boundary at end */
  for (uint32_t i2 = bulk_end; i2 < end; i2++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < KERNEL_LEN; j++) {
      int32_t idx = (int32_t)i2 + (int32_t)j - (int32_t)(KERNEL_LEN - 1);
      if (idx >= 0 && (uint32_t)idx < INPUT_LEN)
        acc += in[idx] * g_kf[j];
    }
    out[i2] = acc;
  }
}

/* Single-core RVV conv */
static void rvv_conv(void) {
  rvv_conv_range(0, VALID_OUT);
}

/* ------------------------------------------------------------------ */
/*  Dual-core RVV convolution                                          */
/*                                                                     */
/*  Hart 0 computes the first half, hart 1 the second half.            */
/*  Synchronisation via volatile flags + CLINT MSIP wakeup.            */
/* ------------------------------------------------------------------ */

#include "clint.h"

/* Split point: middle of VALID_OUT, aligned to avoid partial vectors */
#define SPLIT_POINT  ((VALID_OUT / 2u) & ~63u)

static volatile uint32_t g_h1_go   __attribute__((aligned(64))) = 0;
static volatile uint32_t g_h1_done __attribute__((aligned(64))) = 0;

static void rvv_conv_dual(void) {
  /* Signal hart 1 to start its half */
  g_h1_done = 0;
  __sync_synchronize();
  g_h1_go = 1;
  __sync_synchronize();
  CLINT->MSIP[1] = 1;  /* wake hart 1 */

  /* Hart 0 does the first half */
  rvv_conv_range(0, SPLIT_POINT);

  /* Wait for hart 1 to finish */
  while (__atomic_load_n(&g_h1_done, __ATOMIC_ACQUIRE) == 0)
    asm volatile("nop");

  /* Clear go flag for next invocation */
  g_h1_go = 0;
  __sync_synchronize();
}

/* ------------------------------------------------------------------ */
/*  Reference convolution (HW convention)                              */
/* ------------------------------------------------------------------ */

static void ref_conv(const uint32_t *input, uint32_t n,
                     const uint32_t *kernel, uint32_t k,
                     float *output, uint32_t out_len) {
  for (uint32_t i = 0; i < out_len; i++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; j++) {
      int arr_index = (int)i + (int)j - (int)(k - 1);
      uint32_t val = 0u;
      if (arr_index >= 0 && (uint32_t)arr_index < n)
        val = input[arr_index];
      acc += bits_to_f32(val) * bits_to_f32(kernel[j]);
    }
    output[i] = acc;
  }
}

/* ------------------------------------------------------------------ */
/*  Benchmark runner                                                   */
/* ------------------------------------------------------------------ */

typedef void (*conv_fn_t)(void);

static void run_bench(const char *name, conv_fn_t fn) {
  uint64_t ops = (uint64_t)VALID_OUT * KERNEL_LEN * 2;

  printf("\n--- %s ---\n", name);

  /* Warmup */
  for (uint32_t r = 0; r < WARMUP_RUNS; r++)
    fn();

  /* Timed */
  uint64_t best = UINT64_MAX;
  uint64_t total = 0;

  for (uint32_t r = 0; r < TIMED_RUNS; r++) {
    uint64_t t0 = read_cycles();
    fn();
    uint64_t elapsed = read_cycles() - t0;

    total += elapsed;
    if (elapsed < best)
      best = elapsed;
  }

  uint64_t avg = total / TIMED_RUNS;

  /* Correctness (on last run's output) */
  ref_conv(g_input, INPUT_LEN, g_kernel, KERNEL_LEN, g_ref_out, VALID_OUT);

  uint32_t mismatches = 0;
  for (uint32_t i = 0; i < VALID_OUT; i++) {
    uint32_t ref_bits;
    memcpy(&ref_bits, &g_ref_out[i], sizeof(ref_bits));
    if (g_hw_out[i] != ref_bits)
      mismatches++;
  }

  printf("avg_cycles: %" PRIu64 "\n", avg);
  printf("best_cycles: %" PRIu64 "\n", best);
  printf("ops: %" PRIu64 "\n", ops);
  printf("correctness: %s\n", mismatches == 0 ? "PASS" : "FAIL");
  if (mismatches > 0)
    printf("  (%u/%u mismatches)\n", mismatches, VALID_OUT);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  init_test(1000000000ULL);

  /* Build kernel */
  memset(g_kernel, 0, sizeof(g_kernel));
  g_kernel[0] = f32_to_bits(1.0f);
  g_kernel[1] = f32_to_bits(0.5f);

  /* Build input */
  for (uint32_t i = 0; i < INPUT_LEN; i++)
    g_input[i] = f32_to_bits(0.1f * (float)((i % 256) + 1));

  /* Pre-convert kernel to float for RVV path */
  for (uint32_t i = 0; i < KERNEL_LEN; i++)
    g_kf[i] = bits_to_f32(g_kernel[i]);

  printf("\n=== 1D Conv Benchmark (N=%u, K=%u, %u warmup, %u timed) ===\n",
         INPUT_LEN, KERNEL_LEN, WARMUP_RUNS, TIMED_RUNS);

  run_bench("MMIO (1:1 interleaved)", mmio_conv);
  run_bench("RVV 1-core", rvv_conv);
  run_bench("RVV 2-core", rvv_conv_dual);

  printf("\n=== Done ===\n");
  return 0;
}

/* Hart 1 entry point — spins waiting for work from hart 0 */
void __attribute__((noreturn)) __main(void) {
  while (1) {
    /* Wait for go signal */
    while (__atomic_load_n(&g_h1_go, __ATOMIC_ACQUIRE) == 0)
      asm volatile("nop");

    /* Compute second half of convolution */
    rvv_conv_range(SPLIT_POINT, VALID_OUT);

    /* Signal done */
    __atomic_store_n(&g_h1_done, 1, __ATOMIC_RELEASE);

    /* Wait until hart 0 clears go (ready for next round) */
    while (__atomic_load_n(&g_h1_go, __ATOMIC_ACQUIRE) != 0)
      asm volatile("nop");
  }
}
