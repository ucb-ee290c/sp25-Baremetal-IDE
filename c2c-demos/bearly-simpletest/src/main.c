#include "main.h"

#include <ctype.h>

_Static_assert((BEARLY_SIMPLETEST_SHM_BYTES > sizeof(simpletest_mailbox_t)),
               "BEARLY_SIMPLETEST_SHM_BYTES must exceed mailbox size.");
_Static_assert((BEARLY_SIMPLETEST_CACHE_LINE_BYTES & (BEARLY_SIMPLETEST_CACHE_LINE_BYTES - 1u)) == 0u,
               "BEARLY_SIMPLETEST_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((BEARLY_SIMPLETEST_CACHE_EVICT_BYTES >= BEARLY_SIMPLETEST_CACHE_LINE_BYTES),
               "BEARLY_SIMPLETEST_CACHE_EVICT_BYTES must be at least one cache line.");

typedef struct {
  uint32_t received;
  uint32_t dropped;
  uint32_t seq_errors;
  uint64_t latency_sum;
  uint64_t latency_min;
  uint64_t latency_max;
} rx_stats_t;

static volatile simpletest_mailbox_t *const g_mbox =
    (volatile simpletest_mailbox_t *)(uintptr_t)BEARLY_SIMPLETEST_MAILBOX_ADDR;
static volatile simpletest_slot_t *const g_ring =
    (volatile simpletest_slot_t *)(uintptr_t)BEARLY_SIMPLETEST_RING_ADDR;

static uint8_t g_cache_evict[BEARLY_SIMPLETEST_CACHE_EVICT_BYTES]
    __attribute__((aligned(BEARLY_SIMPLETEST_CACHE_LINE_BYTES)));
static volatile uint8_t g_cache_sink;
static uint32_t g_poll_count;
static uint32_t g_wait_poll_count;
static uint32_t g_consumed;
static rx_stats_t g_stats;

uint64_t target_frequency = BEARLY_SIMPLETEST_TARGET_FREQUENCY_HZ;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t i = 0; i < (uint32_t)BEARLY_SIMPLETEST_CACHE_EVICT_BYTES; i += BEARLY_SIMPLETEST_CACHE_LINE_BYTES) {
    sink ^= buf[i];
    buf[i] = (uint8_t)(sink + (uint8_t)i);
  }

  g_cache_sink = sink;
  simpletest_fence_rw();
}

static inline void refresh_mailbox(void) {
  if ((BEARLY_SIMPLETEST_POLL_EVICT_EVERY <= 1u) ||
      ((g_poll_count++ % BEARLY_SIMPLETEST_POLL_EVICT_EVERY) == 0u)) {
    cache_evict_all();
  }
  simpletest_fence_rw();
}

#if BEARLY_SIMPLETEST_CLEAR_SHM_ON_BOOT
static void clear_shared_scratchpad(void) {
  volatile uint8_t *shm = (volatile uint8_t *)(uintptr_t)BEARLY_SIMPLETEST_SHM_BASE;
  for (uint32_t i = 0; i < (uint32_t)BEARLY_SIMPLETEST_SHM_BYTES; ++i) {
    shm[i] = 0u;
  }
  simpletest_fence_rw();
  cache_evict_all();
  simpletest_fence_rw();
}

static inline uint8_t mailbox_writer_ready_seen(void) {
  return (uint8_t)((g_mbox->magic == SIMPLETEST_MAILBOX_MAGIC) &&
                   (g_mbox->version == SIMPLETEST_PROTO_VERSION) &&
                   ((g_mbox->flags & SIMPLETEST_FLAG_WRITER_READY) != 0u));
}
#endif

static void maybe_clear_shared_scratchpad(void) {
#if BEARLY_SIMPLETEST_CLEAR_SHM_ON_BOOT
  refresh_mailbox();
  if (mailbox_writer_ready_seen()) {
    BEARLY_SIMPLETEST_LOG("[bearly-simpletest] skip clear: writer already active magic=0x%08x flags=0x%08x produced=%u\n",
                          (unsigned)g_mbox->magic,
                          (unsigned)g_mbox->flags,
                          (unsigned)g_mbox->produced_msgs);
    return;
  }
  clear_shared_scratchpad();
  BEARLY_SIMPLETEST_LOG("[bearly-simpletest] cleared shared scratchpad base=0x%08lx bytes=%u\n",
                        (unsigned long)BEARLY_SIMPLETEST_SHM_BASE,
                        (unsigned)BEARLY_SIMPLETEST_SHM_BYTES);
#endif
}

static void reset_state(void) {
  memset(&g_stats, 0, sizeof(g_stats));
  g_stats.latency_min = UINT64_MAX;
  g_cache_sink = 0u;
  g_poll_count = 0u;
  g_wait_poll_count = 0u;
  g_consumed = 0u;
}

static void wait_for_writer_ready(void) {
  const uint32_t max_slots = BEARLY_SIMPLETEST_RING_BYTES / (uint32_t)sizeof(simpletest_slot_t);
  while (1) {
    g_wait_poll_count++;
    refresh_mailbox();
    if ((g_mbox->magic == SIMPLETEST_MAILBOX_MAGIC) &&
        (g_mbox->version == SIMPLETEST_PROTO_VERSION) &&
        ((g_mbox->flags & SIMPLETEST_FLAG_WRITER_READY) != 0u)) {
      if ((g_mbox->ring_slots > 0u) &&
          (g_mbox->ring_slots <= max_slots) &&
          (g_mbox->slot_bytes == (uint32_t)sizeof(simpletest_slot_t))) {
        return;
      }
    }
#if BEARLY_SIMPLETEST_POLL_LOG_EVERY
    if ((g_wait_poll_count % BEARLY_SIMPLETEST_POLL_LOG_EVERY) == 0u) {
      const volatile simpletest_slot_t *slot0 = &g_ring[0];
      const uint8_t b0 = (uint8_t)slot0->msg[0];
      const uint8_t b1 = (uint8_t)slot0->msg[1];
      const uint8_t b2 = (uint8_t)slot0->msg[2];
      const uint8_t b3 = (uint8_t)slot0->msg[3];
      BEARLY_SIMPLETEST_LOG("[bearly-simpletest] polling wait=%u magic=0x%08x ver=%u flags=0x%08x produced=%u ring_slots=%u slot_bytes=%u\n",
                            (unsigned)g_wait_poll_count,
                            (unsigned)g_mbox->magic,
                            (unsigned)g_mbox->version,
                            (unsigned)g_mbox->flags,
                            (unsigned)g_mbox->produced_msgs,
                            (unsigned)g_mbox->ring_slots,
                            (unsigned)g_mbox->slot_bytes);
      BEARLY_SIMPLETEST_LOG("[bearly-simpletest] poll-slot0 commit=%u msg_id=%u msg_len=%u tx_cycle=%llu bytes[0..3]=%02x %02x %02x %02x\n",
                            (unsigned)slot0->commit_seq,
                            (unsigned)slot0->msg_id,
                            (unsigned)slot0->msg_len,
                            (unsigned long long)slot0->tx_cycle,
                            (unsigned)b0,
                            (unsigned)b1,
                            (unsigned)b2,
                            (unsigned)b3);
    }
#endif
    __asm__ volatile("nop");
  }
}

static void handle_one_message(volatile simpletest_slot_t *slot, uint32_t expected_seq) {
  uint64_t rx_cycle;
  uint64_t tx_cycle;
  uint64_t latency;
  uint32_t msg_len;
  char msg_buf[SIMPLETEST_MSG_MAX_BYTES + 1u];
  char ascii_buf[SIMPLETEST_MSG_MAX_BYTES + 1u];

  simpletest_fence_rw();
  rx_cycle = rdcycle64();
  tx_cycle = slot->tx_cycle;
  latency = rx_cycle - tx_cycle;

  msg_len = slot->msg_len;
  if (msg_len > (SIMPLETEST_MSG_MAX_BYTES - 1u)) {
    msg_len = SIMPLETEST_MSG_MAX_BYTES - 1u;
  }
  for (uint32_t i = 0; i < msg_len; ++i) {
    msg_buf[i] = slot->msg[i];
  }
  msg_buf[msg_len] = '\0';
  for (uint32_t i = 0; i < SIMPLETEST_MSG_MAX_BYTES; ++i) {
    unsigned char c = (unsigned char)slot->msg[i];
    ascii_buf[i] = (char)(isprint((int)c) ? c : '.');
  }
  ascii_buf[SIMPLETEST_MSG_MAX_BYTES] = '\0';

  g_stats.received++;
  g_stats.latency_sum += latency;
  if (latency < g_stats.latency_min) {
    g_stats.latency_min = latency;
  }
  if (latency > g_stats.latency_max) {
    g_stats.latency_max = latency;
  }

  BEARLY_SIMPLETEST_LOG("[bearly-simpletest] recv seq=%u msg_id=%u tx_cycle=%llu rx_cycle=%llu link_cycles=%llu text=\"%s\"\n",
                        (unsigned)expected_seq,
                        (unsigned)slot->msg_id,
                        (unsigned long long)tx_cycle,
                        (unsigned long long)rx_cycle,
                        (unsigned long long)latency,
                        msg_buf);
  BEARLY_SIMPLETEST_LOG("[bearly-simpletest] recv raw_ascii=\"%s\"\n", ascii_buf);
  BEARLY_SIMPLETEST_LOG("[bearly-simpletest] recv raw_hex:");
  for (uint32_t i = 0; i < SIMPLETEST_MSG_MAX_BYTES; ++i) {
    BEARLY_SIMPLETEST_LOG("%s%02x", (i == 0u) ? " " : " ", (unsigned)(uint8_t)slot->msg[i]);
  }
  BEARLY_SIMPLETEST_LOG("\n");
}

static void process_messages(void) {
  uint32_t produced;
  uint32_t available;
  uint32_t slots = g_mbox->ring_slots;
  uint32_t max_slots = BEARLY_SIMPLETEST_RING_BYTES / (uint32_t)sizeof(simpletest_slot_t);

  if ((slots == 0u) || (slots > max_slots)) {
    return;
  }

  refresh_mailbox();
  produced = g_mbox->produced_msgs;
  available = produced - g_consumed;
  if (available == 0u) {
    return;
  }

  if (available > slots) {
    uint32_t overflow = available - slots;
    g_stats.dropped += overflow;
    g_stats.seq_errors++;
    g_consumed = produced - slots;
    available = slots;
  }

  cache_evict_all();
  refresh_mailbox();
  produced = g_mbox->produced_msgs;

  while (g_consumed < produced) {
    uint32_t expected_seq = g_consumed + 1u;
    uint32_t slot_idx = g_consumed % slots;
    volatile simpletest_slot_t *slot = &g_ring[slot_idx];
    uint32_t commit = slot->commit_seq;

    if (commit < expected_seq) {
      break;
    }
    if (commit > expected_seq) {
      uint32_t missed = commit - expected_seq;
      g_stats.dropped += missed;
      g_stats.seq_errors++;
      g_consumed += missed;
      continue;
    }

    handle_one_message(slot, expected_seq);
    g_consumed++;
  }
}

void app_init(void) {
  init_test(target_frequency);
  reset_state();
  maybe_clear_shared_scratchpad();
  BEARLY_SIMPLETEST_LOG("[bearly-simpletest] start mailbox=0x%08lx ring=0x%08lx evict_bytes=%u\n",
                        (unsigned long)BEARLY_SIMPLETEST_MAILBOX_ADDR,
                        (unsigned long)BEARLY_SIMPLETEST_RING_ADDR,
                        (unsigned)BEARLY_SIMPLETEST_CACHE_EVICT_BYTES);
}

void app_main(void) {
  wait_for_writer_ready();
  BEARLY_SIMPLETEST_LOG("[bearly-simpletest] writer ready slots=%u slot_bytes=%u\n",
                        (unsigned)g_mbox->ring_slots,
                        (unsigned)g_mbox->slot_bytes);

  while (1) {
    process_messages();

    refresh_mailbox();
    if (((g_mbox->flags & SIMPLETEST_FLAG_STREAM_DONE) != 0u) &&
        (g_consumed == g_mbox->produced_msgs)) {
      uint64_t avg = (g_stats.received == 0u) ? 0u : (g_stats.latency_sum / g_stats.received);
      BEARLY_SIMPLETEST_LOG("[bearly-simpletest] done produced=%u received=%u dropped=%u seq_err=%u link_cycles(best/avg/worst)=%llu/%llu/%llu\n",
                            (unsigned)g_mbox->produced_msgs,
                            (unsigned)g_stats.received,
                            (unsigned)g_stats.dropped,
                            (unsigned)g_stats.seq_errors,
                            (unsigned long long)((g_stats.received == 0u) ? 0u : g_stats.latency_min),
                            (unsigned long long)avg,
                            (unsigned long long)g_stats.latency_max);
      break;
    }

    __asm__ volatile("nop");
  }

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
