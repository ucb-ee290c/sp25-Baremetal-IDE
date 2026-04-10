#include "main.h"

_Static_assert((C2C_TRANSFER_DSP_CACHE_LINE_BYTES & (C2C_TRANSFER_DSP_CACHE_LINE_BYTES - 1u)) == 0u,
               "C2C_TRANSFER_DSP_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((C2C_TRANSFER_DSP_CACHE_EVICT_BYTES >= C2C_TRANSFER_DSP_CACHE_LINE_BYTES),
               "C2C_TRANSFER_DSP_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((C2C_TRANSFER_DSP_CACHE_EVICT_BYTES % C2C_TRANSFER_DSP_CACHE_LINE_BYTES) == 0u,
               "C2C_TRANSFER_DSP_CACHE_EVICT_BYTES must be a multiple of cache line size.");

static transfer_cycle_word_t *const g_cycle_word =
    (transfer_cycle_word_t *)(uintptr_t)C2C_TRANSFER_DSP_MAILBOX_ADDR;

static uint8_t g_cache_evict[C2C_TRANSFER_DSP_CACHE_EVICT_BYTES]
    __attribute__((aligned(0x8000)));
static volatile uint8_t g_cache_sink;

uint64_t target_frequency = C2C_TRANSFER_DSP_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t pass = 0; pass < C2C_TRANSFER_DSP_CACHE_EVICT_PASSES; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)C2C_TRANSFER_DSP_CACHE_EVICT_BYTES; i += C2C_TRANSFER_DSP_CACHE_LINE_BYTES) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    transfer_fence_rw();
  }

  g_cache_sink = sink;
  transfer_fence_rw();
}

void app_init(void) {
  init_test(target_frequency);
  g_cache_sink = 0u;

  *g_cycle_word = 0u;
  transfer_fence_rw();
  cache_evict_all();
  transfer_fence_rw();

  C2C_TRANSFER_DSP_LOG("[c2c-transfer-dsp] init cycle_addr=0x%08lx packets=%u interval=%llu startup_delay=%llu\n",
                       (unsigned long)C2C_TRANSFER_DSP_MAILBOX_ADDR,
                       (unsigned)C2C_TRANSFER_DSP_NUM_PACKETS,
                       (unsigned long long)C2C_TRANSFER_DSP_INTERVAL_CYCLES,
                       (unsigned long long)C2C_TRANSFER_DSP_STARTUP_DELAY_CYCLES);
}

void app_main(void) {
  uint64_t next_target = rdcycle64() + C2C_TRANSFER_DSP_STARTUP_DELAY_CYCLES;

  C2C_TRANSFER_DSP_LOG("[c2c-transfer-dsp] begin streaming first_target=%llu\n",
                       (unsigned long long)next_target);

  for (uint32_t seq = 1u; seq <= C2C_TRANSFER_DSP_NUM_PACKETS; ++seq) {
    uint64_t now;
    uint64_t tx_cycle;
    int64_t tx_error;

    while ((now = rdcycle64()) < next_target) {
      (void)now;
      __asm__ volatile("nop");
    }

    tx_cycle = rdcycle64();
    tx_error = (int64_t)tx_cycle - (int64_t)next_target;

    *g_cycle_word = tx_cycle;
    transfer_fence_rw();
    cache_evict_all();
    transfer_fence_rw();

    C2C_TRANSFER_DSP_LOG("[c2c-transfer-dsp] send seq=%u tx_cycle=%llu target=%llu tx_error=%lld\n",
                         (unsigned)seq,
                         (unsigned long long)tx_cycle,
                         (unsigned long long)next_target,
                         (long long)tx_error);

    next_target += C2C_TRANSFER_DSP_INTERVAL_CYCLES;
  }

  C2C_TRANSFER_DSP_LOG("[c2c-transfer-dsp] complete, entering wfi\n");

  while (1) {
    __asm__ volatile("wfi");
  }
}

int main(void) {
  app_init();
  app_main();
  return 0;
}

void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
    __asm__ volatile("wfi");
  }
}
