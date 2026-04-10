#include "main.h"

#define SHM_BASE 0xC0000000UL

static uint8_t g_cache_evict[DSP_SIMPLETEST_CACHE_EVICT_BYTES]
    __attribute__((aligned(DSP_SIMPLETEST_CACHE_LINE_BYTES)));
static volatile uint8_t g_cache_sink;

uint64_t target_frequency = DSP_SIMPLETEST_TARGET_FREQUENCY_HZ;

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t i = 0; i < (uint32_t)DSP_SIMPLETEST_CACHE_EVICT_BYTES; i += DSP_SIMPLETEST_CACHE_LINE_BYTES) {
    sink ^= buf[i];
    buf[i] = (uint8_t)(sink + (uint8_t)i);
  }

  g_cache_sink = sink;
  __asm__ volatile("fence rw, rw" ::: "memory");
}

void app_init(void) {
  init_test(target_frequency);
  g_cache_sink = 0u;
}

void app_main(void) {
  volatile uint32_t *shm0 = (volatile uint32_t *)SHM_BASE;
  volatile uint32_t *shm1 = (volatile uint32_t *)(SHM_BASE + 4);

  /* Write 19 to 0xC0000004 */
  *shm1 = 19u;
  __asm__ volatile("fence rw, rw" ::: "memory");

  /* Signal bearly by writing 0xFFFFFFFF to 0xC0000000 */
  *shm0 = 0xFFFFFFFFu;
  __asm__ volatile("fence rw, rw" ::: "memory");
  cache_evict_all();
  __asm__ volatile("fence rw, rw" ::: "memory");

  printf("[dsp] wrote 19 to 0x%08lx, wrote 0xFFFFFFFF to 0x%08lx\n",
         (unsigned long)(SHM_BASE + 4), (unsigned long)SHM_BASE);
  printf("[dsp] done\n");

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
