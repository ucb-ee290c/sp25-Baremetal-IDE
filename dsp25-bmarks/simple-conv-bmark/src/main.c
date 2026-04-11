/*
 * 1D Convolution Benchmark (N configurable via bench_mode_config.h, K=8)
 *
 * Modes:
 *   1) mmio_full_correct
 *   2) mmio_steady_state
 *   3) scalar_ref
 *   4) rvv_sc
 *   5) rvv_mc
 *   6) dma_full_correct
 *   7) mmio_rtl_style (START->N-output drain timing)
 *   8) rvv_sc_rtl_style (same transform as rtl_style)
 *
 * Correctness checks are outside the timed region.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <riscv_vector.h>

#include "chip_config.h"
#include "simple_setup.h"
#include "bench_mode_config.h"

#define INPUT_LEN    ((uint32_t)SIMPLE_CONV_INPUT_LENGTH)
#define KERNEL_LEN   8u
#define VALID_OUT    (INPUT_LEN + (KERNEL_LEN - 1u))
#define DRIVER_OUT   (INPUT_LEN + KERNEL_LEN)
#define RTL_OUT      INPUT_LEN

#define WARMUP_RUNS  10u
#define TIMED_RUNS   1000u

#define STEADY_STATE_CHUNKS_PER_TIMED_RUN ((uint32_t)SIMPLE_CONV_STEADY_STATE_CHUNKS_PER_TIMED_RUN)
#define STEADY_STATE_WARMUP_RUNS          ((uint32_t)SIMPLE_CONV_STEADY_STATE_WARMUP_RUNS)
#define STEADY_STATE_TIMED_RUNS           ((uint32_t)SIMPLE_CONV_STEADY_STATE_TIMED_RUNS)
#define RTL_STYLE_PRELOAD_PACKETS_CFG     ((uint32_t)SIMPLE_CONV_RTL_STYLE_PRELOAD_PACKETS)
#define RTL_STYLE_ALLOW_UNSAFE_PRELOAD    ((uint32_t)SIMPLE_CONV_RTL_STYLE_ALLOW_UNSAFE_PRELOAD)

#if SIMPLE_CONV_STEADY_STATE_CHUNKS_PER_TIMED_RUN < 1
#error "SIMPLE_CONV_STEADY_STATE_CHUNKS_PER_TIMED_RUN must be >= 1"
#endif

#if SIMPLE_CONV_STEADY_STATE_TIMED_RUNS < 1
#error "SIMPLE_CONV_STEADY_STATE_TIMED_RUNS must be >= 1"
#endif

#if SIMPLE_CONV_INPUT_LENGTH < 2
#error "SIMPLE_CONV_INPUT_LENGTH must be >= 2"
#endif

#if (SIMPLE_CONV_INPUT_LENGTH % 2) != 0
#error "SIMPLE_CONV_INPUT_LENGTH must be even (2 FP32 per 64b packet)"
#endif

#if SIMPLE_CONV_RTL_STYLE_PRELOAD_PACKETS < 1
#error "SIMPLE_CONV_RTL_STYLE_PRELOAD_PACKETS must be >= 1"
#endif

#define INPUT_PACKETS   (INPUT_LEN / FP32_PER_PACKET)
#define KERNEL_PACKETS  (KERNEL_LEN / FP32_PER_PACKET)
#define OUTPUT_PACKETS  (INPUT_PACKETS + KERNEL_PACKETS)
#define STREAM_OPS      (INPUT_PACKETS + OUTPUT_PACKETS)
#define RTL_OUT_PACKETS (RTL_OUT / FP32_PER_PACKET)

#define OPS_TOTAL ((uint64_t)VALID_OUT * (uint64_t)KERNEL_LEN * 2ULL)
#define OPS_TOTAL_RTL ((uint64_t)RTL_OUT * (uint64_t)KERNEL_LEN * 2ULL)

#define H1_CMD_IDLE 0u
#define H1_CMD_PING 1u
#define H1_CMD_CONV 2u
#define H1_RUNTIME_COOKIE 0x48415254u

#define H1_TIMEOUT_CYCLES 50000000ULL
#define H1_TIMEOUT_POLL_MASK 0x3FFu
#define H1_PING_ATTEMPTS 128u
#define H1_PING_TIMEOUT_CYCLES 2000000ULL
#define H1_RETRY_SPINS 2048u

#define SPLIT_POINT (((VALID_OUT / 2u) < (KERNEL_LEN - 1u)) ? \
                     (KERNEL_LEN - 1u) : (((VALID_OUT / 2u) & ~63u)))

#if KERNEL_LEN != 8
#error "This benchmark's RVV kernel is specialized for KERNEL_LEN=8."
#endif

typedef struct {
  const char *name;
  uint64_t avg_cycles;
  uint64_t best_cycles;
  uint64_t ops_per_work_item;
  uint32_t timed_runs_completed;
  uint32_t work_items_per_timed_run;
  uint32_t checked_outputs;
  uint32_t run_failures;
  uint32_t verify_failures;
  uint32_t mismatches;
  bool show_stream_stats;
} bench_result_t;

typedef bool (*run_once_fn_t)(void);
typedef bool (*run_once_timed_fn_t)(uint64_t *elapsed_cycles);
typedef uint32_t (*mismatch_fn_t)(void);
typedef void (*prepare_fn_t)(void);

static uint32_t g_input_bits[INPUT_LEN]       __attribute__((aligned(64)));
static uint32_t g_kernel_bits[KERNEL_LEN]     __attribute__((aligned(64)));
static uint32_t g_hw_out_bits[DRIVER_OUT]     __attribute__((aligned(64)));

static float g_input_f32[INPUT_LEN]           __attribute__((aligned(64)));
static float g_kernel_f32[KERNEL_LEN]         __attribute__((aligned(64)));
static float g_scalar_ref_out[VALID_OUT]      __attribute__((aligned(64)));
static float g_scalar_ref_rtl_out[RTL_OUT]    __attribute__((aligned(64)));
static float g_scalar_work_out[VALID_OUT]     __attribute__((aligned(64)));
static float g_rvv_sc_out[VALID_OUT]          __attribute__((aligned(64)));
static float g_rvv_mc_out[VALID_OUT]          __attribute__((aligned(64)));
static float g_rvv_sc_rtl_out[RTL_OUT]        __attribute__((aligned(64)));

static volatile uint32_t g_h1_cmd  __attribute__((aligned(64))) = H1_CMD_IDLE;
static volatile uint32_t g_h1_done __attribute__((aligned(64))) = 0u;
static volatile uint32_t g_h1_runtime_cookie __attribute__((aligned(64))) = 0u;
static volatile uint32_t g_h1_seen __attribute__((aligned(64))) = 0u;

static bool g_rvv_mc_ready = false;
static uint32_t g_dma_base_id = 0x1000u;
static bool g_mmio_preconfigured_ready = false;

static inline float bits_to_f32(uint32_t x) {
  union {
    uint32_t u;
    float f;
  } c;
  c.u = x;
  return c.f;
}

static inline uint32_t f32_to_bits(float x) {
  union {
    float f;
    uint32_t u;
  } c;
  c.f = x;
  return c.u;
}

static inline float absf_local(float x) {
  return (x < 0.0f) ? -x : x;
}

static inline uint64_t read_cycles(void) {
  uint64_t c;
  asm volatile("rdcycle %0" : "=r"(c));
  return c;
}

static inline void mmio_write64_local(uintptr_t addr, uint64_t data) {
  *(volatile uint64_t *)addr = data;
}

static inline uint64_t mmio_read64_local(uintptr_t addr) {
  return *(volatile uint64_t *)addr;
}

static uint32_t rtl_style_effective_preload_packets(void) {
  uint32_t preload_packets = RTL_STYLE_PRELOAD_PACKETS_CFG;
  if (preload_packets > INPUT_PACKETS) {
    preload_packets = INPUT_PACKETS;
  }
  if ((RTL_STYLE_ALLOW_UNSAFE_PRELOAD == 0u) &&
      (preload_packets > FIFO_CAPACITY_PACKETS)) {
    preload_packets = FIFO_CAPACITY_PACKETS;
  }
  return preload_packets;
}

static bool status_has_error(uint8_t status) {
  return ((status & (STATUS_ERROR | STATUS_INVALID)) != 0u);
}

static void scalar_conv_f32(const float *input,
                            uint32_t input_len,
                            const float *kernel,
                            uint32_t kernel_len,
                            float *output,
                            uint32_t out_len) {
  for (uint32_t i = 0; i < out_len; i++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < kernel_len; j++) {
      int32_t idx = (int32_t)i + (int32_t)j - (int32_t)(kernel_len - 1u);
      float in = 0.0f;
      if (idx >= 0 && (uint32_t)idx < input_len) {
        in = input[idx];
      }
      acc += in * kernel[j];
    }
    output[i] = acc;
  }
}

/*
 * RTL-style transform used in 1d-conv-rtl benchmark_*_wide.c:
 *   output[i] = sum_j input[i + j] * kernel[j], with right-side zero extension.
 * Output length is exactly INPUT_LEN.
 */
static void scalar_conv_rtl_forward_f32(const float *input,
                                        uint32_t input_len,
                                        const float *kernel,
                                        uint32_t kernel_len,
                                        float *output) {
  for (uint32_t i = 0; i < input_len; i++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < kernel_len; j++) {
      uint32_t idx = i + j;
      float in = (idx < input_len) ? input[idx] : 0.0f;
      acc += in * kernel[j];
    }
    output[i] = acc;
  }
}

static void rvv_conv_range(uint32_t start, uint32_t end, float *output) {
  const float *in = g_input_f32;

  const float k0 = g_kernel_f32[0];
  const float k1 = g_kernel_f32[1];
  const float k2 = g_kernel_f32[2];
  const float k3 = g_kernel_f32[3];
  const float k4 = g_kernel_f32[4];
  const float k5 = g_kernel_f32[5];
  const float k6 = g_kernel_f32[6];
  const float k7 = g_kernel_f32[7];

  uint32_t bulk_start = start;
  if (bulk_start < (KERNEL_LEN - 1u)) {
    bulk_start = (KERNEL_LEN - 1u);
  }

  uint32_t bulk_end = end;
  if (bulk_end > INPUT_LEN) {
    bulk_end = INPUT_LEN;
  }

  for (uint32_t i = start; i < bulk_start; i++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < KERNEL_LEN; j++) {
      int32_t idx = (int32_t)i + (int32_t)j - (int32_t)(KERNEL_LEN - 1u);
      if (idx >= 0 && (uint32_t)idx < INPUT_LEN) {
        acc += in[idx] * g_kernel_f32[j];
      }
    }
    output[i] = acc;
  }

  uint32_t i = bulk_start;
  while (i < bulk_end) {
    size_t vl = __riscv_vsetvl_e32m8(bulk_end - i);
    const float *base = in + i - (KERNEL_LEN - 1u);

    vfloat32m8_t vacc =
        __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(base + 0, vl), k0, vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k1, __riscv_vle32_v_f32m8(base + 1, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k2, __riscv_vle32_v_f32m8(base + 2, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k3, __riscv_vle32_v_f32m8(base + 3, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k4, __riscv_vle32_v_f32m8(base + 4, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k5, __riscv_vle32_v_f32m8(base + 5, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k6, __riscv_vle32_v_f32m8(base + 6, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k7, __riscv_vle32_v_f32m8(base + 7, vl), vl);

    __riscv_vse32_v_f32m8(output + i, vacc, vl);
    i += (uint32_t)vl;
  }

  for (uint32_t i2 = bulk_end; i2 < end; i2++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < KERNEL_LEN; j++) {
      int32_t idx = (int32_t)i2 + (int32_t)j - (int32_t)(KERNEL_LEN - 1u);
      if (idx >= 0 && (uint32_t)idx < INPUT_LEN) {
        acc += in[idx] * g_kernel_f32[j];
      }
    }
    output[i2] = acc;
  }
}

static void rvv_conv_rtl_forward_range(uint32_t start, uint32_t end, float *output) {
  const float *in = g_input_f32;

  const float k0 = g_kernel_f32[0];
  const float k1 = g_kernel_f32[1];
  const float k2 = g_kernel_f32[2];
  const float k3 = g_kernel_f32[3];
  const float k4 = g_kernel_f32[4];
  const float k5 = g_kernel_f32[5];
  const float k6 = g_kernel_f32[6];
  const float k7 = g_kernel_f32[7];

  uint32_t bulk_end = INPUT_LEN;
  if (bulk_end >= (KERNEL_LEN - 1u)) {
    bulk_end -= (KERNEL_LEN - 1u);
  } else {
    bulk_end = 0u;
  }
  if (bulk_end > end) {
    bulk_end = end;
  }

  uint32_t i = start;
  while (i < bulk_end) {
    size_t vl = __riscv_vsetvl_e32m8(bulk_end - i);
    const float *base = in + i;

    vfloat32m8_t vacc =
        __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(base + 0, vl), k0, vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k1, __riscv_vle32_v_f32m8(base + 1, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k2, __riscv_vle32_v_f32m8(base + 2, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k3, __riscv_vle32_v_f32m8(base + 3, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k4, __riscv_vle32_v_f32m8(base + 4, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k5, __riscv_vle32_v_f32m8(base + 5, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k6, __riscv_vle32_v_f32m8(base + 6, vl), vl);
    vacc = __riscv_vfmacc_vf_f32m8(vacc, k7, __riscv_vle32_v_f32m8(base + 7, vl), vl);

    __riscv_vse32_v_f32m8(output + i, vacc, vl);
    i += (uint32_t)vl;
  }

  for (; i < end; i++) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < KERNEL_LEN; j++) {
      uint32_t idx = i + j;
      if (idx < INPUT_LEN) {
        acc += in[idx] * g_kernel_f32[j];
      }
    }
    output[i] = acc;
  }
}

static void h1_start_cmd(uint32_t cmd) {
  __atomic_store_n(&g_h1_done, 0u, __ATOMIC_RELAXED);
  __sync_synchronize();
  __atomic_store_n(&g_h1_cmd, cmd, __ATOMIC_RELEASE);
  __sync_synchronize();
  CLINT->MSIP[1] = 1u;
}

static void h1_finish_cmd(void) {
  __atomic_store_n(&g_h1_cmd, H1_CMD_IDLE, __ATOMIC_RELEASE);
  __sync_synchronize();
  CLINT->MSIP[1] = 1u;
}

static bool h1_wait_done_with_timeout(uint64_t timeout_cycles) {
  uint64_t start = read_cycles();
  uint32_t spin = 0u;

  while (__atomic_load_n(&g_h1_done, __ATOMIC_ACQUIRE) == 0u) {
    if ((spin & H1_TIMEOUT_POLL_MASK) == 0u) {
      if ((read_cycles() - start) > timeout_cycles) {
        return false;
      }
    }
    spin++;
    asm volatile("nop");
  }
  return true;
}

static bool h1_ping(uint64_t timeout_cycles) {
  h1_start_cmd(H1_CMD_PING);
  if (!h1_wait_done_with_timeout(timeout_cycles)) {
    h1_finish_cmd();
    return false;
  }
  h1_finish_cmd();
  return true;
}

static bool ensure_rvv_mc_ready(void) {
  if (g_rvv_mc_ready) {
    return true;
  }

  for (uint32_t attempt = 0u; attempt < H1_PING_ATTEMPTS; attempt++) {
    if (h1_ping(H1_PING_TIMEOUT_CYCLES)) {
      g_rvv_mc_ready = true;
      return true;
    }

    /* Keep sending wake pulses in case hart1 is still in boot/wfi handoff. */
    CLINT->MSIP[1] = 1u;
    for (uint32_t s = 0u; s < H1_RETRY_SPINS; s++) {
      asm volatile("nop");
    }
  }

  g_rvv_mc_ready = false;
  return false;
}

static bool run_mmio_full_correct_once(void) {
  uint8_t status = perform_convolution_1D(
      g_input_bits,
      INPUT_LEN,
      g_kernel_bits,
      KERNEL_LEN,
      g_hw_out_bits,
      1u);
  return !status_has_error(status);
}

static bool run_mmio_rtl_style_once_timed(uint64_t *elapsed_cycles) {
  if ((INPUT_LEN % FP32_PER_PACKET) != 0u) {
    return false;
  }

  const uint32_t input_packets = INPUT_PACKETS;
  const uint32_t output_packets = RTL_OUT_PACKETS;
  const uint32_t preload_packets = rtl_style_effective_preload_packets();

  conv_init();
  if (conv_set_params_kernel_only(INPUT_LEN, 1u, g_kernel_bits, KERNEL_LEN) != 0) {
    return false;
  }

  for (uint32_t i = 0u; i < preload_packets; i++) {
    uint32_t element_idx = i * FP32_PER_PACKET;
    mmio_write64_local((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR),
                       *((uint64_t *)(g_input_bits + element_idx)));
  }

  start_conv();
  uint64_t t0 = read_cycles();

  uint32_t in_packet_idx = preload_packets;
  for (uint32_t out_packet_idx = 0u; out_packet_idx < output_packets; out_packet_idx++) {
    // For this taped-out interface, write-then-read avoids sampling empty output as zeros.
    if (in_packet_idx < input_packets) {
      uint32_t in_element_idx = in_packet_idx * FP32_PER_PACKET;
      mmio_write64_local((uintptr_t)(MMIO_BASE + CONV_INPUT_ADDR),
                         *((uint64_t *)(g_input_bits + in_element_idx)));
      in_packet_idx++;
    }

    uint64_t current_out = mmio_read64_local((uintptr_t)(MMIO_BASE + CONV_OUTPUT_ADDR));
    uint32_t *unpacked = (uint32_t *)&current_out;
    uint32_t out_element_idx = out_packet_idx * FP32_PER_PACKET;
    g_hw_out_bits[out_element_idx] = unpacked[0];
    g_hw_out_bits[out_element_idx + 1u] = unpacked[1];
  }

  uint64_t t1 = read_cycles();
  if (elapsed_cycles != 0) {
    *elapsed_cycles = t1 - t0;
  }
  return !status_has_error(get_register_status());
}

static bool run_mmio_steady_state_once(void) {
  if (!g_mmio_preconfigured_ready) {
    return false;
  }

  for (uint32_t i = 0u; i < STEADY_STATE_CHUNKS_PER_TIMED_RUN; i++) {
    uint8_t status = perform_convolution_1D_preconfigured(
        g_input_bits,
        INPUT_LEN,
        g_kernel_bits,
        KERNEL_LEN,
        g_hw_out_bits);
    if (status_has_error(status)) {
      return false;
    }
  }
  return true;
}

static bool run_scalar_ref_once(void) {
  scalar_conv_f32(g_input_f32, INPUT_LEN, g_kernel_f32, KERNEL_LEN, g_scalar_work_out, VALID_OUT);
  return true;
}

static bool run_rvv_mc_once(void) {
  if (!ensure_rvv_mc_ready()) {
    return false;
  }

  h1_start_cmd(H1_CMD_CONV);
  rvv_conv_range(0u, SPLIT_POINT, g_rvv_mc_out);

  if (!h1_wait_done_with_timeout(H1_TIMEOUT_CYCLES)) {
    h1_finish_cmd();
    g_rvv_mc_ready = false;
    return false;
  }

  h1_finish_cmd();
  return true;
}

static bool run_rvv_sc_once(void) {
  rvv_conv_range(0u, VALID_OUT, g_rvv_sc_out);
  return true;
}

static bool run_rvv_sc_rtl_once(void) {
  rvv_conv_rtl_forward_range(0u, RTL_OUT, g_rvv_sc_rtl_out);
  return true;
}

static void prepare_dma_mode(void) {
  g_dma_base_id = 0x1000u;
}

static void prepare_mmio_preconfigured_mode(void) {
  g_mmio_preconfigured_ready =
      (conv_begin_preconfigured_session(INPUT_LEN, 1u, KERNEL_LEN) == 0);
}

static void prepare_rvv_mc_mode(void) {
  (void)ensure_rvv_mc_ready();
}

static bool run_dma_full_correct_once(void) {
  bool ok = dma_1dConvDriver(
      g_input_bits,
      g_hw_out_bits,
      g_kernel_bits,
      (size_t)INPUT_LEN,
      (size_t)KERNEL_LEN,
      1u,
      g_dma_base_id);
  g_dma_base_id += 4u;

  if (!ok) {
    return false;
  }

  return !status_has_error(get_register_status());
}

static bool float_mismatch(float got, float ref) {
  float diff = absf_local(got - ref);
  float tol = 1e-4f * (absf_local(ref) + 1.0f);
  return diff > tol;
}

static uint32_t count_mismatches_float(const float *got,
                                       const float *ref,
                                       uint32_t count,
                                       const char *tag) {
  uint32_t mismatches = 0u;
  for (uint32_t i = 0; i < count; i++) {
    if (float_mismatch(got[i], ref[i])) {
      if (mismatches < 5u) {
        float diff = absf_local(got[i] - ref[i]);
        float tol = 1e-4f * (absf_local(ref[i]) + 1.0f);
        printf("  mismatch[%u] (%s): got=%f ref=%f diff=%f tol=%f\n",
               i,
               tag,
               got[i],
               ref[i],
               diff,
               tol);
      }
      mismatches++;
    }
  }
  return mismatches;
}

static uint32_t mismatches_mmio_or_dma(void) {
  uint32_t mismatches = 0u;
  for (uint32_t i = 0; i < VALID_OUT; i++) {
    float got = bits_to_f32(g_hw_out_bits[i]);
    float ref = g_scalar_ref_out[i];
    if (float_mismatch(got, ref)) {
      if (mismatches < 5u) {
        float diff = absf_local(got - ref);
        float tol = 1e-4f * (absf_local(ref) + 1.0f);
        printf("  mismatch[%u] (hw): got=0x%08" PRIx32 " (%f) ref=%f diff=%f tol=%f\n",
               i,
               g_hw_out_bits[i],
               got,
               ref,
               diff,
               tol);
      }
      mismatches++;
    }
  }
  return mismatches;
}

static uint32_t mismatches_mmio_rtl_style(void) {
  uint32_t mismatches = 0u;
  for (uint32_t i = 0; i < RTL_OUT; i++) {
    float got = bits_to_f32(g_hw_out_bits[i]);
    /*
     * The accelerator stream in this SoC revision produces full-convolution
     * ordering (VALID_OUT = N + K - 1 payload + one protocol slot in driver),
     * so rtl_style compares against the first N elements of that stream.
     */
    float ref = g_scalar_ref_out[i];
    if (float_mismatch(got, ref)) {
      if (mismatches < 5u) {
        float diff = absf_local(got - ref);
        float tol = 1e-4f * (absf_local(ref) + 1.0f);
        printf("  mismatch[%u] (hw_rtlprefix): got=0x%08" PRIx32 " (%f) ref=%f diff=%f tol=%f\n",
               i,
               g_hw_out_bits[i],
               got,
               ref,
               diff,
               tol);
      }
      mismatches++;
    }
  }
  return mismatches;
}

static uint32_t mismatches_scalar(void) {
  return count_mismatches_float(g_scalar_work_out, g_scalar_ref_out, VALID_OUT, "scalar");
}

static uint32_t mismatches_rvv_mc(void) {
  return count_mismatches_float(g_rvv_mc_out, g_scalar_ref_out, VALID_OUT, "rvv_mc");
}

static uint32_t mismatches_rvv_sc(void) {
  return count_mismatches_float(g_rvv_sc_out, g_scalar_ref_out, VALID_OUT, "rvv_sc");
}

static uint32_t mismatches_rvv_sc_rtl(void) {
  return count_mismatches_float(g_rvv_sc_rtl_out, g_scalar_ref_rtl_out, RTL_OUT, "rvv_sc_rtl");
}

static bench_result_t benchmark_mode(const char *name,
                                     run_once_fn_t run_once,
                                     mismatch_fn_t mismatch_fn,
                                     prepare_fn_t prepare,
                                     bool show_stream_stats,
                                     uint32_t warmup_runs,
                                     uint32_t timed_runs,
                                     uint32_t work_items_per_timed_run) {
  bench_result_t r;
  memset(&r, 0, sizeof(r));
  r.name = name;
  r.best_cycles = UINT64_MAX;
  r.show_stream_stats = show_stream_stats;
  r.work_items_per_timed_run = (work_items_per_timed_run == 0u) ? 1u : work_items_per_timed_run;
  r.checked_outputs = VALID_OUT;
  r.ops_per_work_item = OPS_TOTAL;

  if (prepare != 0) {
    prepare();
  }

  for (uint32_t i = 0; i < warmup_runs; i++) {
    if (!run_once()) {
      r.run_failures++;
      break;
    }
  }

  if (r.run_failures == 0u) {
    uint64_t total = 0u;
    for (uint32_t i = 0; i < timed_runs; i++) {
      uint64_t t0 = read_cycles();
      bool ok = run_once();
      uint64_t elapsed = read_cycles() - t0;

      if (!ok) {
        r.run_failures++;
        break;
      }

      total += elapsed;
      r.timed_runs_completed++;
      if (elapsed < r.best_cycles) {
        r.best_cycles = elapsed;
      }
    }

    if (r.timed_runs_completed > 0u) {
      r.avg_cycles = total / r.timed_runs_completed;
    } else {
      r.best_cycles = 0u;
      r.avg_cycles = 0u;
    }
  } else {
    r.best_cycles = 0u;
    r.avg_cycles = 0u;
  }

  if (r.run_failures == 0u && mismatch_fn != 0) {
    if (!run_once()) {
      r.verify_failures++;
    } else {
      r.mismatches = mismatch_fn();
    }
  }

  return r;
}

static bench_result_t benchmark_mode_inner_timed(const char *name,
                                                 run_once_timed_fn_t run_once_timed,
                                                 mismatch_fn_t mismatch_fn,
                                                 prepare_fn_t prepare,
                                                 bool show_stream_stats,
                                                 uint32_t warmup_runs,
                                                 uint32_t timed_runs,
                                                 uint32_t work_items_per_timed_run) {
  bench_result_t r;
  memset(&r, 0, sizeof(r));
  r.name = name;
  r.best_cycles = UINT64_MAX;
  r.show_stream_stats = show_stream_stats;
  r.work_items_per_timed_run = (work_items_per_timed_run == 0u) ? 1u : work_items_per_timed_run;
  r.checked_outputs = VALID_OUT;
  r.ops_per_work_item = OPS_TOTAL;

  if (prepare != 0) {
    prepare();
  }

  for (uint32_t i = 0; i < warmup_runs; i++) {
    uint64_t warmup_cycles = 0u;
    if (!run_once_timed(&warmup_cycles)) {
      r.run_failures++;
      break;
    }
  }

  if (r.run_failures == 0u) {
    uint64_t total = 0u;
    for (uint32_t i = 0; i < timed_runs; i++) {
      uint64_t elapsed = 0u;
      bool ok = run_once_timed(&elapsed);

      if (!ok) {
        r.run_failures++;
        break;
      }

      total += elapsed;
      r.timed_runs_completed++;
      if (elapsed < r.best_cycles) {
        r.best_cycles = elapsed;
      }
    }

    if (r.timed_runs_completed > 0u) {
      r.avg_cycles = total / r.timed_runs_completed;
    } else {
      r.best_cycles = 0u;
      r.avg_cycles = 0u;
    }
  } else {
    r.best_cycles = 0u;
    r.avg_cycles = 0u;
  }

  if (r.run_failures == 0u && mismatch_fn != 0) {
    uint64_t verify_cycles = 0u;
    if (!run_once_timed(&verify_cycles)) {
      r.verify_failures++;
    } else {
      r.mismatches = mismatch_fn();
    }
  }

  return r;
}

static void print_result(const bench_result_t *r) {
  uint64_t ops_per_work_item = (r->ops_per_work_item == 0u) ? OPS_TOTAL : r->ops_per_work_item;
  uint64_t ops_per_timed_run = ops_per_work_item * (uint64_t)r->work_items_per_timed_run;
  uint32_t checked_outputs = (r->checked_outputs == 0u) ? VALID_OUT : r->checked_outputs;
  printf("\nmode: %s\n", r->name);
  printf("avg_cycles: %" PRIu64 "\n", r->avg_cycles);
  printf("best_cycles: %" PRIu64 "\n", r->best_cycles);
  printf("ops: %" PRIu64 "\n", ops_per_timed_run);
  printf("timed_runs_completed: %u\n", r->timed_runs_completed);

  if (r->work_items_per_timed_run > 1u) {
    uint64_t avg_per_item = r->avg_cycles / (uint64_t)r->work_items_per_timed_run;
    uint64_t best_per_item = r->best_cycles / (uint64_t)r->work_items_per_timed_run;
    uint64_t timed_items = (uint64_t)r->timed_runs_completed * (uint64_t)r->work_items_per_timed_run;
    printf("work_items_per_timed_run: %u\n", r->work_items_per_timed_run);
    printf("timed_work_items_completed: %" PRIu64 "\n", timed_items);
    printf("avg_cycles_per_work_item: %" PRIu64 "\n", avg_per_item);
    printf("best_cycles_per_work_item: %" PRIu64 "\n", best_per_item);
  }

  if (r->show_stream_stats) {
    uint64_t input_packets = (uint64_t)INPUT_PACKETS * (uint64_t)r->work_items_per_timed_run;
    uint64_t output_packets = (uint64_t)OUTPUT_PACKETS * (uint64_t)r->work_items_per_timed_run;
    uint64_t stream_ops = (uint64_t)STREAM_OPS * (uint64_t)r->work_items_per_timed_run;
    printf("input_packets: %" PRIu64 "\n", input_packets);
    printf("output_packets: %" PRIu64 "\n", output_packets);
    printf("stream_ops: %" PRIu64 "\n", stream_ops);
  }

  printf("run_failures: %u\n", r->run_failures);
  printf("verify_failures: %u\n", r->verify_failures);

  if (r->run_failures != 0u || r->verify_failures != 0u || r->timed_runs_completed == 0u) {
    printf("correctness: SKIPPED (run/verify failure)\n");
    return;
  }

  printf("correctness: %s\n", (r->mismatches == 0u) ? "PASS" : "FAIL");
  if (r->mismatches > 0u) {
    printf("  (%u/%u mismatches)\n", r->mismatches, checked_outputs);
  }
}

static void init_data(void) {
  for (uint32_t i = 0; i < INPUT_LEN; i++) {
    g_input_f32[i] = 0.1f * (float)((i % 256u) + 1u);
    g_input_bits[i] = f32_to_bits(g_input_f32[i]);
  }

  for (uint32_t i = 0; i < KERNEL_LEN; i++) {
    g_kernel_f32[i] = 0.0f;
  }
  g_kernel_f32[0] = 1.0f;
  g_kernel_f32[1] = 0.5f;

  for (uint32_t i = 0; i < KERNEL_LEN; i++) {
    g_kernel_bits[i] = f32_to_bits(g_kernel_f32[i]);
  }

  scalar_conv_f32(g_input_f32, INPUT_LEN, g_kernel_f32, KERNEL_LEN, g_scalar_ref_out, VALID_OUT);
  scalar_conv_rtl_forward_f32(g_input_f32, INPUT_LEN, g_kernel_f32, KERNEL_LEN, g_scalar_ref_rtl_out);
}

int main(void) {
  init_test(150000000ULL);
  init_data();
  __sync_synchronize();
  __atomic_store_n(&g_h1_runtime_cookie, H1_RUNTIME_COOKIE, __ATOMIC_RELEASE);
  __sync_synchronize();

  printf("\n=== 1D Conv Benchmark (N=%u, K=%u, %u warmup, %u timed) ===\n",
         INPUT_LEN,
         KERNEL_LEN,
         WARMUP_RUNS,
         TIMED_RUNS);
  printf("timing_scope: run only (correctness compare is outside timed region)\n");
#if SIMPLE_CONV_RUN_ACCELERATOR_MMIO_STEADY_STATE
  printf("steady_state: %u chunks/run, %u warmup runs, %u timed runs\n",
         STEADY_STATE_CHUNKS_PER_TIMED_RUN,
         STEADY_STATE_WARMUP_RUNS,
         STEADY_STATE_TIMED_RUNS);
  printf("steady_state_impl: preconfigured session (no full perform_convolution_1D in hot loop)\n");
#endif
#if SIMPLE_CONV_RUN_ACCELERATOR_MMIO_RTL_STYLE
  printf("rtl_style: START->N-output drain timing (setup+preload excluded)\n");
  printf("rtl_style_compare: first N outputs of accelerator full-convolution stream\n");
  printf("rtl_style_preload_packets_cfg: %u (effective: %u, unsafe_preload=%u)\n",
         RTL_STYLE_PRELOAD_PACKETS_CFG,
         rtl_style_effective_preload_packets(),
         RTL_STYLE_ALLOW_UNSAFE_PRELOAD);
#endif

#if SIMPLE_CONV_RUN_ACCELERATOR_MMIO
  {
    bench_result_t mmio = benchmark_mode(
        "mmio_full_correct",
        run_mmio_full_correct_once,
        mismatches_mmio_or_dma,
        0,
        true,
        WARMUP_RUNS,
        TIMED_RUNS,
        1u);
    print_result(&mmio);
  }
#endif

#if SIMPLE_CONV_RUN_ACCELERATOR_MMIO_STEADY_STATE
  {
    bench_result_t mmio_steady = benchmark_mode(
        "mmio_steady_state",
        run_mmio_steady_state_once,
        mismatches_mmio_or_dma,
        prepare_mmio_preconfigured_mode,
        true,
        STEADY_STATE_WARMUP_RUNS,
        STEADY_STATE_TIMED_RUNS,
        STEADY_STATE_CHUNKS_PER_TIMED_RUN);
    print_result(&mmio_steady);
  }
#endif

#if SIMPLE_CONV_RUN_ACCELERATOR_MMIO_RTL_STYLE
  {
    bench_result_t mmio_rtl = benchmark_mode_inner_timed(
        "mmio_rtl_style",
        run_mmio_rtl_style_once_timed,
        mismatches_mmio_rtl_style,
        0,
        false,
        WARMUP_RUNS,
        TIMED_RUNS,
        1u);
    mmio_rtl.checked_outputs = RTL_OUT;
    mmio_rtl.ops_per_work_item = OPS_TOTAL_RTL;
    print_result(&mmio_rtl);
  }
#endif

#if SIMPLE_CONV_RUN_SCALAR_REFERENCE
  {
    bench_result_t scalar = benchmark_mode(
        "scalar_ref",
        run_scalar_ref_once,
        mismatches_scalar,
        0,
        false,
        WARMUP_RUNS,
        TIMED_RUNS,
        1u);
    print_result(&scalar);
  }
#endif

#if SIMPLE_CONV_RUN_RVV_SINGLE_CORE_REFERENCE
  {
    bench_result_t rvv_sc = benchmark_mode(
        "rvv_sc",
        run_rvv_sc_once,
        mismatches_rvv_sc,
        0,
        false,
        WARMUP_RUNS,
        TIMED_RUNS,
        1u);
    print_result(&rvv_sc);
  }
#endif

#if SIMPLE_CONV_RUN_RVV_SINGLE_CORE_RTL_STYLE
  {
    bench_result_t rvv_sc_rtl = benchmark_mode(
        "rvv_sc_rtl_style",
        run_rvv_sc_rtl_once,
        mismatches_rvv_sc_rtl,
        0,
        false,
        WARMUP_RUNS,
        TIMED_RUNS,
        1u);
    rvv_sc_rtl.checked_outputs = RTL_OUT;
    rvv_sc_rtl.ops_per_work_item = OPS_TOTAL_RTL;
    print_result(&rvv_sc_rtl);
  }
#endif

#if SIMPLE_CONV_RUN_RVV_MULTI_CORE_REFERENCE
  {
    bench_result_t rvv_mc = benchmark_mode(
        "rvv_mc",
        run_rvv_mc_once,
        mismatches_rvv_mc,
        prepare_rvv_mc_mode,
        false,
        WARMUP_RUNS,
        TIMED_RUNS,
        1u);
    print_result(&rvv_mc);
    if (!g_rvv_mc_ready) {
      printf("note: hart1 worker did not respond; rvv_mc was not runnable on this boot.\n");
      printf("debug: h1_seen=%u\n", (unsigned)g_h1_seen);
    }
  }
#endif

#if SIMPLE_CONV_RUN_ACCELERATOR_DMA
  {
    bench_result_t dma = benchmark_mode(
        "dma_full_correct",
        run_dma_full_correct_once,
        mismatches_mmio_or_dma,
        prepare_dma_mode,
        true,
        WARMUP_RUNS,
        TIMED_RUNS,
        1u);
    print_result(&dma);
  }
#endif

#if !(SIMPLE_CONV_RUN_ACCELERATOR_MMIO || SIMPLE_CONV_RUN_ACCELERATOR_MMIO_STEADY_STATE || \
      SIMPLE_CONV_RUN_ACCELERATOR_MMIO_RTL_STYLE || SIMPLE_CONV_RUN_SCALAR_REFERENCE || \
      SIMPLE_CONV_RUN_RVV_SINGLE_CORE_REFERENCE || SIMPLE_CONV_RUN_RVV_SINGLE_CORE_RTL_STYLE || \
      SIMPLE_CONV_RUN_RVV_MULTI_CORE_REFERENCE || SIMPLE_CONV_RUN_ACCELERATOR_DMA)
  printf("note: all benchmark modes are disabled in bench_mode_config.h\n");
#endif

  printf("\n=== Done ===\n");
  return 0;
}

void __attribute__((noreturn)) __main(void) {
  uint64_t mhartid = READ_CSR("mhartid");

  if (mhartid != 1u) {
    while (1) {
      asm volatile("wfi");
    }
  }

  __atomic_store_n(&g_h1_cmd, H1_CMD_IDLE, __ATOMIC_RELEASE);
  __atomic_store_n(&g_h1_done, 0u, __ATOMIC_RELEASE);

  while (__atomic_load_n(&g_h1_runtime_cookie, __ATOMIC_ACQUIRE) != H1_RUNTIME_COOKIE) {
    CLINT->MSIP[1] = 0u;
    asm volatile("wfi");
  }
  __atomic_store_n(&g_h1_seen, 1u, __ATOMIC_RELEASE);

  while (1) {
    CLINT->MSIP[1] = 0u;
    while (__atomic_load_n(&g_h1_cmd, __ATOMIC_ACQUIRE) == H1_CMD_IDLE) {
      asm volatile("wfi");
    }
    CLINT->MSIP[1] = 0u;

    if (__atomic_load_n(&g_h1_cmd, __ATOMIC_ACQUIRE) == H1_CMD_CONV) {
      rvv_conv_range(SPLIT_POINT, VALID_OUT, g_rvv_mc_out);
    }

    __atomic_store_n(&g_h1_done, 1u, __ATOMIC_RELEASE);

    while (__atomic_load_n(&g_h1_cmd, __ATOMIC_ACQUIRE) != H1_CMD_IDLE) {
      asm volatile("wfi");
    }
  }
}
