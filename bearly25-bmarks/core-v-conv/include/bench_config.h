/*
 * bench_config.h - Configuration and shared types for core-v-conv benchmarks.
 */
#ifndef CORE_V_CONV_BENCH_CONFIG_H
#define CORE_V_CONV_BENCH_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef BENCH_RUNS
#define BENCH_RUNS 5
#endif

#ifndef BENCH_VERIFY
#define BENCH_VERIFY 1
#endif

#ifndef BENCH_HAS_VECNN
#define BENCH_HAS_VECNN 0
#endif

#ifndef BENCH_ENABLE_VEC
#define BENCH_ENABLE_VEC 1
#endif

#ifndef CONV_KERNEL_SIZE
#define CONV_KERNEL_SIZE 3
#endif

#ifndef CONV_STRIDE
#define CONV_STRIDE 1
#endif

#ifndef CONV_PADDING
#define CONV_PADDING 0
#endif

#ifndef CONV_CHANNELS
#define CONV_CHANNELS 1
#endif

#ifndef CONV_RELU
#define CONV_RELU 0
#endif

typedef enum {
  INT,
  FLOAT,
} Type;

typedef struct {
  const char *name;
  int H;
  int W;
  int kernel_dim; // Length/height of kernel
  int data_bytes;
  Type data_type;
} ConvSizeCase;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline int conv_out_dim(int input, int kernel, int stride, int padding) {
  return (input + 2 * padding - kernel) / stride + 1;
}

#endif // CORE_V_CONV_BENCH_CONFIG_H
