#include "main.h"

#include "hello_wfi_link.h"

uint64_t target_frequency = HELLO_WFI_TARGET_FREQUENCY_HZ;

void app_init(void) {
  init_test(target_frequency);
  HELLO_WFI_LOG("[dsp-hello-wfi] boot: core up (freq=%llu)\n",
                (unsigned long long)target_frequency);

  /* Boot init: clear our baton and set our turn register to the peer's value, so we stay asleep
   * until Bearly explicitly hands off to us (also defeats stale spad SRAM from a previous run). */
  hwfi_boot_init();
  HELLO_WFI_LOG("[dsp-hello-wfi] boot init: baton=0, turn=peer(%u) in own spad @0x%09llx\n",
                (unsigned)HELLO_WFI_PEER_TURN,
                (unsigned long long)HELLO_WFI_LOCAL_SPAD_BASE);

  HELLO_WFI_LOG("[dsp-hello-wfi] local_spad=0x%09llx peer_spad=0x%09llx peer_msip=0x%09llx\n",
                (unsigned long long)HELLO_WFI_LOCAL_SPAD_BASE,
                (unsigned long long)HELLO_WFI_PEER_SPAD_BASE,
                (unsigned long long)HELLO_WFI_PEER_MSIP_ADDR);
}

void app_main(void) {
  /* Arm MSIP + timer wake and clear any stale MSIP. */
  hwfi_arm_wake();
  hwfi_clear_own_msip();

  /* Responder: Bearly hands us baton=11, we reply 12, it replies 13, ... forever. Each round we
   * sleep until the turn register in our own spad says it is our turn (re-checked on every MSIP or
   * timer wake), do our (unknown-duration) work, then hand off exactly once. */
  while (1) {
    uint32_t baton;
    uint32_t out;

    HELLO_WFI_LOG("[dsp-hello-wfi] wfi (waiting for our turn; turn-gated + timer safety net)\n");
    hwfi_await_my_turn();

    baton = hwfi_read_baton_local();
    HELLO_WFI_LOG("[dsp-hello-wfi] our turn -> baton=%u received from Bearly\n", (unsigned)baton);

    /* Work of unknown duration (here: `baton` prints, NO spad/MSIP access). We hand off only after
     * it completes — the turn register is flipped as part of the handoff, never on a timer. */
    for (uint32_t k = 1u; k <= baton; ++k) {
      HELLO_WFI_LOG("[dsp-hello-wfi]   work %u/%u\n", (unsigned)k, (unsigned)baton);
    }

    out = baton + 1u;
    HELLO_WFI_LOG("[dsp-hello-wfi] work done -> handing baton=%u to Bearly (peer spad 0x%09llx) and waking it\n",
                  (unsigned)out, (unsigned long long)HELLO_WFI_PEER_SPAD_BASE);
    hwfi_handoff(out);
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
