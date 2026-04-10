#include "main.h"

#define SHM_BASE 0xC0000000UL

static uint8_t g_cache_evict[BEARLY_SIMPLETEST_CACHE_EVICT_BYTES]
    __attribute__((aligned(BEARLY_SIMPLETEST_CACHE_LINE_BYTES)));
static volatile uint8_t g_cache_sink;

uint64_t target_frequency = BEARLY_SIMPLETEST_TARGET_FREQUENCY_HZ;

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t i = 0; i < (uint32_t)BEARLY_SIMPLETEST_CACHE_EVICT_BYTES; i += BEARLY_SIMPLETEST_CACHE_LINE_BYTES) {
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

  printf("[bearly] hello world from bearly chip\n");
  printf("[bearly] polling 0x%08lx for 0xFFFFFFFF...\n", (unsigned long)SHM_BASE);

  while (1) {
    cache_evict_all();
    uint32_t val = *shm0;
    printf("[bearly] 0x%08lx = 0x%08lx\n", (unsigned long)SHM_BASE, (unsigned long)val);
    if (val == 0xFFFFFFFFu) {
      printf("[bearly] received 0xFFFFFFFF from DSP\n");
      break;
    }
    sleep(5);
  }

  /* Read what DSP wrote at 0xC0000004 (should be 19) */
  cache_evict_all();
  uint32_t val2 = *shm1;
  printf("[bearly] read %lu (0x%08lx) from 0x%08lx\n",
         (unsigned long)val2, (unsigned long)val2, (unsigned long)(SHM_BASE + 4));
  printf("[bearly] done\n");

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
