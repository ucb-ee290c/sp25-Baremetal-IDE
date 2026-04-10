#include "main.h"

_Static_assert((DSP_SIMPLETEST_SHARED_BYTES > sizeof(simpletest_mailbox_t)),
               "DSP_SIMPLETEST_SHARED_BYTES must exceed mailbox size.");
_Static_assert((DSP_SIMPLETEST_REMOTE_RING_BYTES > 0u),
               "DSP_SIMPLETEST_REMOTE_RING_BYTES must be positive.");

static volatile simpletest_mailbox_t *const g_mbox =
    (volatile simpletest_mailbox_t *)(uintptr_t)DSP_SIMPLETEST_REMOTE_MAILBOX_ADDR;
static volatile simpletest_slot_t *const g_ring =
    (volatile simpletest_slot_t *)(uintptr_t)DSP_SIMPLETEST_REMOTE_RING_ADDR;

uint64_t target_frequency = DSP_SIMPLETEST_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static uint32_t ring_slots(void) {
  return DSP_SIMPLETEST_REMOTE_RING_BYTES / (uint32_t)sizeof(simpletest_slot_t);
}

static void init_writer_state(uint32_t slots) {
  g_mbox->magic = SIMPLETEST_MAILBOX_MAGIC;
  g_mbox->version = SIMPLETEST_PROTO_VERSION;
  g_mbox->flags = SIMPLETEST_FLAG_WRITER_READY;
  g_mbox->ring_slots = slots;
  g_mbox->slot_bytes = (uint32_t)sizeof(simpletest_slot_t);
  g_mbox->produced_msgs = 0u;
  g_mbox->last_seq = 0u;
  g_mbox->last_tx_cycle = 0u;

  for (uint32_t i = 0; i < slots; ++i) {
    g_ring[i].commit_seq = 0u;
  }
  simpletest_fence_rw();
}

static void send_message(uint32_t seq, uint32_t msg_id, const char *msg, uint32_t slots) {
  const uint32_t slot_idx = (seq - 1u) % slots;
  volatile simpletest_slot_t *slot = &g_ring[slot_idx];
  uint64_t tx_cycle = rdcycle64();
  uint32_t copy_len = 0u;

  while ((copy_len < (SIMPLETEST_MSG_MAX_BYTES - 1u)) && (msg[copy_len] != '\0')) {
    copy_len++;
  }

  slot->commit_seq = 0u;
  simpletest_fence_rw();

  slot->msg_id = msg_id;
  slot->msg_len = (uint16_t)copy_len;
  slot->reserved = 0u;
  slot->tx_cycle = tx_cycle;
  for (uint32_t i = 0; i < SIMPLETEST_MSG_MAX_BYTES; ++i) {
    slot->msg[i] = '\0';
  }
  for (uint32_t i = 0; i < copy_len; ++i) {
    slot->msg[i] = msg[i];
  }

  simpletest_fence_rw();
  slot->commit_seq = seq;
  simpletest_fence_rw();

  g_mbox->produced_msgs = seq;
  g_mbox->last_seq = seq;
  g_mbox->last_tx_cycle = tx_cycle;

  DSP_SIMPLETEST_LOG("[dsp-simpletest] send seq=%u msg_id=%u cycle=%llu text=\"%s\"\n",
                     (unsigned)seq,
                     (unsigned)msg_id,
                     (unsigned long long)tx_cycle,
                     msg);
}

void app_init(void) {
  init_test(target_frequency);
}

void app_main(void) {
  uint32_t slots = ring_slots();

  if (slots == 0u) {
    DSP_SIMPLETEST_LOG("[dsp-simpletest] invalid ring: slots=0\n");
    while (1) {
      __asm__ volatile("wfi");
    }
  }

  init_writer_state(slots);
  DSP_SIMPLETEST_LOG("[dsp-simpletest] start mailbox=0x%08lx ring=0x%08lx slots=%u sleep_cycles=%llu\n",
                     (unsigned long)DSP_SIMPLETEST_REMOTE_MAILBOX_ADDR,
                     (unsigned long)DSP_SIMPLETEST_REMOTE_RING_ADDR,
                     (unsigned)slots,
                     (unsigned long long)DSP_SIMPLETEST_SLEEP_CYCLES);

  /* Simple mode: send exactly one message. */
  send_message(1u, 0u, "hello world from dsp", slots);
  (void)DSP_SIMPLETEST_NUM_MESSAGES; /* kept as config knob for future extension */

  g_mbox->flags = SIMPLETEST_FLAG_WRITER_READY | SIMPLETEST_FLAG_STREAM_DONE;
  simpletest_fence_rw();
  DSP_SIMPLETEST_LOG("[dsp-simpletest] done sent=1 (entering keepalive loop)\n");

  while (1) {
#if DSP_SIMPLETEST_ACTIVE_KEEPALIVE
    __asm__ volatile("nop");
#else
    __asm__ volatile("wfi");
#endif
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
