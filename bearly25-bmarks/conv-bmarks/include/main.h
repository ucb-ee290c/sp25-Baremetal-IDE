#ifndef CONV_BMARKS_MAIN_H
#define CONV_BMARKS_MAIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef CONV_BENCH_TARGET_FREQUENCY_HZ
#define CONV_BENCH_TARGET_FREQUENCY_HZ 800000000ULL
#endif

#ifndef CONV_BENCH_RUNS
#define CONV_BENCH_RUNS 8u
#endif

#ifndef CONV_BENCH_KERNEL_SIZE
#define CONV_BENCH_KERNEL_SIZE 3u
#endif

#ifndef CONV_BENCH_STRIDE
#define CONV_BENCH_STRIDE 1u
#endif

#ifndef CONV_BENCH_USE_RELU
#define CONV_BENCH_USE_RELU 0u
#endif

#ifndef CONV_BENCH_INTER_RUN_STALL_CYCLES
#define CONV_BENCH_INTER_RUN_STALL_CYCLES 128u
#endif

#ifndef CONV_BENCH_READY_TIMEOUT_CYCLES
#define CONV_BENCH_READY_TIMEOUT_CYCLES 200000000ULL
#endif

#ifndef CONV_BENCH_VERIFY_STATUS
#define CONV_BENCH_VERIFY_STATUS 1
#endif

#ifndef CONV_BENCH_ENABLE_STATE_COLD
#define CONV_BENCH_ENABLE_STATE_COLD 1
#endif

#ifndef CONV_BENCH_ENABLE_STATE_WARM_SRC
#define CONV_BENCH_ENABLE_STATE_WARM_SRC 1
#endif

#ifndef CONV_BENCH_ENABLE_STATE_WARM_DST
#define CONV_BENCH_ENABLE_STATE_WARM_DST 1
#endif

#ifndef CONV_BENCH_ENABLE_STATE_WARM_BOTH
#define CONV_BENCH_ENABLE_STATE_WARM_BOTH 1
#endif

#ifndef CONV_BENCH_ENABLE_STATE_HOT_REPEAT
#define CONV_BENCH_ENABLE_STATE_HOT_REPEAT 1
#endif

#ifndef CONV_BENCH_CACHE_LINE_BYTES
#define CONV_BENCH_CACHE_LINE_BYTES 64u
#endif

#ifndef CONV_BENCH_CACHE_EVICT_BYTES
#define CONV_BENCH_CACHE_EVICT_BYTES (512u * 1024u)
#endif

typedef struct {
  const char *name;
  uint16_t batch_size;
  uint16_t channels;
  uint16_t height;
  uint16_t width;
} conv_bench_case_t;

static inline uint64_t conv_bench_rdcycle64(void) {
  uint64_t x;
  asm volatile("rdcycle %0" : "=r"(x));
  return x;
}

void app_init(void);
void app_main(void);

#endif // CONV_BMARKS_MAIN_H
