#include "main.h"

uint64_t target_frequency = HELLO_WFI_TARGET_FREQUENCY_HZ;

/* CLINT MSIP for this hart. Regular MMIO (device memory) — no shared-spad cache dance needed. */
static volatile uint32_t *const g_msip =
    (volatile uint32_t *)(uintptr_t)HELLO_WFI_MSIP_ADDR;

static inline void fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

/* Clear mstatus.MIE (bit 3): a pending software interrupt then WAKES wfi but is NOT taken as a
 * trap (no handler required) — execution simply resumes at the instruction after wfi. */
static inline void disable_global_irqs(void) {
  __asm__ volatile("csrc mstatus, %0" :: "r"(1UL << 3) : "memory");
}

/* Set mie.MSIE (bit 3) so a machine software interrupt (CLINT MSIP) is an enabled wake source. */
static inline void enable_msip_wake(void) {
  __asm__ volatile("csrs mie, %0" :: "r"(1UL << 3) : "memory");
}

void app_init(void) {
  init_test(target_frequency);
  HELLO_WFI_LOG("[hello-wfi] boot: core up (freq=%llu)\n",
                (unsigned long long)target_frequency);
}

void app_main(void) {
  for (uint32_t i = 0; i < (uint32_t)HELLO_WFI_HELLO_COUNT; ++i) {
    HELLO_WFI_LOG("hello world %u/%u\n",
                  (unsigned)(i + 1u), (unsigned)HELLO_WFI_HELLO_COUNT);
  }

  /* Arm the software-interrupt wake path: enabled in mie (wakes wfi), global IRQs off (no trap). */
  disable_global_irqs();
  enable_msip_wake();

  /* Clear any stale MSIP (the bootrom leaves it cleared, but be explicit) so we truly block until
   * a fresh external CLINT write. */
  *g_msip = 0u;
  fence_rw();

  HELLO_WFI_LOG("[hello-wfi] parked in wfi loop; write 1 to CLINT MSIP @0x%08lx to wake\n",
                (unsigned long)HELLO_WFI_MSIP_ADDR);

  /* Block in wfi until MSIP is set by an external write. wfi may wake spuriously, so re-check the
   * pending bit and go back to sleep while it is still clear. */
  while (1) {
    __asm__ volatile("wfi");
    fence_rw();
    if (*g_msip != 0u) {
      break;
    }
  }

  /* Acknowledge the wake by clearing the pending bit. */
  *g_msip = 0u;
  fence_rw();

  HELLO_WFI_LOG("[hello-wfi] woke from wfi -- MSIP observed! goodbye world\n");

  /* Done: park forever. */
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
