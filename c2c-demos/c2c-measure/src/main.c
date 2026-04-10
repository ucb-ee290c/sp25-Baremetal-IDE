#include "main.h"

_Static_assert((C2C_MEASURE_CACHE_LINE_BYTES & (C2C_MEASURE_CACHE_LINE_BYTES - 1u)) == 0u,
               "C2C_MEASURE_CACHE_LINE_BYTES must be a power of two.");
_Static_assert((C2C_MEASURE_CACHE_EVICT_BYTES >= C2C_MEASURE_CACHE_LINE_BYTES),
               "C2C_MEASURE_CACHE_EVICT_BYTES must be at least one cache line.");
_Static_assert((C2C_MEASURE_CACHE_EVICT_BYTES % C2C_MEASURE_CACHE_LINE_BYTES) == 0u,
               "C2C_MEASURE_CACHE_EVICT_BYTES must be a multiple of cache line size.");
_Static_assert((C2C_MEASURE_SHM_BYTES % 8u) == 0u,
               "C2C_MEASURE_SHM_BYTES must be divisible by 8.");
_Static_assert((C2C_MEASURE_PTR_NODES > 1u),
               "C2C_MEASURE_PTR_NODES must be > 1.");
_Static_assert((C2C_MEASURE_PTR_NODES * 8u) <= C2C_MEASURE_SHM_BYTES,
               "Pointer-chase arrays must fit in the shared region.");

static volatile uint32_t *const g_next =
    (volatile uint32_t *)(uintptr_t)C2C_MEASURE_SHM_BASE;
static volatile uint32_t *const g_vals =
    (volatile uint32_t *)(uintptr_t)(C2C_MEASURE_SHM_BASE +
                                     ((uintptr_t)C2C_MEASURE_PTR_NODES * sizeof(uint32_t)));

static uint8_t g_cache_evict[C2C_MEASURE_CACHE_EVICT_BYTES]
    __attribute__((aligned(0x8000)));
static uint32_t g_perm[C2C_MEASURE_PTR_NODES];

static volatile uint8_t g_cache_sink;
static volatile uint32_t g_ptr_sink;
static uint64_t g_measure_overhead_cycles;

uint64_t target_frequency = C2C_MEASURE_TARGET_FREQUENCY_HZ;

typedef struct {
  uint64_t best;
  uint64_t worst;
  uint64_t total;
} cycle_stats_t;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

static inline void fence_rw(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline uint64_t disable_irqs(void) {
  uint64_t old;
  __asm__ volatile("csrrc %0, mstatus, %1" : "=r"(old) : "r"(1ULL << 3) : "memory");
  return old;
}

static inline void restore_irqs(uint64_t old) {
  __asm__ volatile("csrw mstatus, %0" :: "r"(old) : "memory");
}

static inline uint64_t subtract_measurement_overhead(uint64_t cycles) {
  return (cycles > g_measure_overhead_cycles) ? (cycles - g_measure_overhead_cycles) : 0u;
}

static inline uint32_t lcg_next(uint32_t *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

static void stats_reset(cycle_stats_t *stats) {
  stats->best = UINT64_MAX;
  stats->worst = 0u;
  stats->total = 0u;
}

static void stats_add(cycle_stats_t *stats, uint64_t cycles) {
  if (cycles < stats->best) {
    stats->best = cycles;
  }
  if (cycles > stats->worst) {
    stats->worst = cycles;
  }
  stats->total += cycles;
}

static uint64_t stats_avg(const cycle_stats_t *stats, uint32_t trials) {
  return (trials == 0u) ? 0u : (stats->total / (uint64_t)trials);
}

static void print_cycles_per_step(const char *label, uint64_t cycles, uint32_t steps) {
  uint64_t whole;
  uint64_t frac;

  if (steps == 0u) {
    C2C_MEASURE_LOG(" %s=0.00", label);
    return;
  }

  whole = cycles / (uint64_t)steps;
  frac = ((cycles % (uint64_t)steps) * 100u + ((uint64_t)steps / 2u)) / (uint64_t)steps;
  if (frac >= 100u) {
    whole += 1u;
    frac -= 100u;
  }

  C2C_MEASURE_LOG(" %s=%llu.%02llu", label,
                  (unsigned long long)whole,
                  (unsigned long long)frac);
}

static uint64_t measure_overhead_best(uint32_t reps) {
  uint64_t best = UINT64_MAX;

  for (uint32_t i = 0; i < reps; ++i) {
    uint64_t t0;
    uint64_t t1;
    uint64_t d;

    fence_rw();
    t0 = rdcycle64();
    fence_rw();
    t1 = rdcycle64();
    d = t1 - t0;

    if (d < best) {
      best = d;
    }
  }

  return (best == UINT64_MAX) ? 0u : best;
}

static inline void cache_evict_all(void) {
  volatile uint8_t *buf = (volatile uint8_t *)g_cache_evict;
  volatile uint8_t sink = g_cache_sink;

  for (uint32_t pass = 0; pass < C2C_MEASURE_CACHE_EVICT_PASSES; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)C2C_MEASURE_CACHE_EVICT_BYTES; i += C2C_MEASURE_CACHE_LINE_BYTES) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    fence_rw();
  }

  g_cache_sink = sink;
  fence_rw();
}

static void clear_scratch_region(void) {
  volatile uint8_t *shm = (volatile uint8_t *)(uintptr_t)C2C_MEASURE_SHM_BASE;

  for (uint32_t i = 0; i < C2C_MEASURE_SHM_BYTES; ++i) {
    shm[i] = 0u;
  }

  fence_rw();
}

static void build_pointer_ring(void) {
  uint32_t state = C2C_MEASURE_PTR_SHUFFLE_SEED;

  for (uint32_t i = 0; i < C2C_MEASURE_PTR_NODES; ++i) {
    g_perm[i] = i;
  }

  for (uint32_t i = C2C_MEASURE_PTR_NODES - 1u; i > 0u; --i) {
    uint32_t j = lcg_next(&state) % (i + 1u);
    uint32_t tmp = g_perm[i];
    g_perm[i] = g_perm[j];
    g_perm[j] = tmp;
  }

  for (uint32_t i = 0; i < C2C_MEASURE_PTR_NODES; ++i) {
    uint32_t cur = g_perm[i];
    uint32_t nxt = g_perm[(i + 1u) % C2C_MEASURE_PTR_NODES];
    g_next[cur] = nxt;
    g_vals[cur] = cur;
  }

  fence_rw();
}

static uint64_t measure_cache_flush_once(void) {
  uint64_t t0;
  uint64_t t1;

  fence_rw();
  t0 = rdcycle64();
  cache_evict_all();
  fence_rw();
  t1 = rdcycle64();

  return subtract_measurement_overhead(t1 - t0);
}

static uint64_t measure_ptr_read_once(uint32_t steps) {
  volatile uint32_t *next = g_next;
  uint32_t idx = g_ptr_sink % C2C_MEASURE_PTR_NODES;
  uint64_t t0;
  uint64_t t1;

  fence_rw();
  t0 = rdcycle64();

  for (uint32_t i = 0; i < steps; ++i) {
    idx = next[idx];
  }

  fence_rw();
  t1 = rdcycle64();
  g_ptr_sink = idx;

  return subtract_measurement_overhead(t1 - t0);
}

static uint64_t measure_ptr_write_once(uint32_t steps) {
  volatile uint32_t *next = g_next;
  volatile uint32_t *vals = g_vals;
  uint32_t idx = g_ptr_sink % C2C_MEASURE_PTR_NODES;
  uint64_t t0;
  uint64_t t1;

  fence_rw();
  t0 = rdcycle64();

  for (uint32_t i = 0; i < steps; ++i) {
    idx = next[idx];
    vals[idx] = idx ^ i;
  }

  fence_rw();
  t1 = rdcycle64();
  g_ptr_sink = idx;

  return subtract_measurement_overhead(t1 - t0);
}

static void run_cache_flush_bench(void) {
  cycle_stats_t stats;

  stats_reset(&stats);
  for (uint32_t trial = 0; trial < C2C_MEASURE_CACHE_FLUSH_TRIALS; ++trial) {
    uint64_t saved_mstatus = disable_irqs();
    uint64_t cycles = measure_cache_flush_once();
    restore_irqs(saved_mstatus);
    stats_add(&stats, cycles);
  }

  C2C_MEASURE_LOG("[c2c-measure] cache_evict trials=%u best=%llu avg=%llu worst=%llu",
                  (unsigned)C2C_MEASURE_CACHE_FLUSH_TRIALS,
                  (unsigned long long)stats.best,
                  (unsigned long long)stats_avg(&stats, C2C_MEASURE_CACHE_FLUSH_TRIALS),
                  (unsigned long long)stats.worst);
  print_cycles_per_step("best_cycles/line", stats.best,
                        (uint32_t)(C2C_MEASURE_CACHE_EVICT_BYTES / C2C_MEASURE_CACHE_LINE_BYTES));
  print_cycles_per_step("avg_cycles/line", stats_avg(&stats, C2C_MEASURE_CACHE_FLUSH_TRIALS),
                        (uint32_t)(C2C_MEASURE_CACHE_EVICT_BYTES / C2C_MEASURE_CACHE_LINE_BYTES));
  C2C_MEASURE_LOG("\n");
}

static void run_ptr_read_bench(void) {
  cycle_stats_t stats;

  stats_reset(&stats);
  for (uint32_t trial = 0; trial < C2C_MEASURE_PTR_TRIALS; ++trial) {
    uint64_t cycles;
    uint64_t saved_mstatus;

    cache_evict_all();
    saved_mstatus = disable_irqs();
    cycles = measure_ptr_read_once(C2C_MEASURE_PTR_STEPS);
    restore_irqs(saved_mstatus);
    stats_add(&stats, cycles);
  }

  C2C_MEASURE_LOG("[c2c-measure] ptr_read trials=%u steps=%u best=%llu avg=%llu worst=%llu",
                  (unsigned)C2C_MEASURE_PTR_TRIALS,
                  (unsigned)C2C_MEASURE_PTR_STEPS,
                  (unsigned long long)stats.best,
                  (unsigned long long)stats_avg(&stats, C2C_MEASURE_PTR_TRIALS),
                  (unsigned long long)stats.worst);
  print_cycles_per_step("best_cycles/step", stats.best, C2C_MEASURE_PTR_STEPS);
  print_cycles_per_step("avg_cycles/step", stats_avg(&stats, C2C_MEASURE_PTR_TRIALS), C2C_MEASURE_PTR_STEPS);
  C2C_MEASURE_LOG("\n");
}

static void run_ptr_write_bench(void) {
  cycle_stats_t stats;

  stats_reset(&stats);
  for (uint32_t trial = 0; trial < C2C_MEASURE_PTR_TRIALS; ++trial) {
    uint64_t cycles;
    uint64_t saved_mstatus;

    cache_evict_all();
    saved_mstatus = disable_irqs();
    cycles = measure_ptr_write_once(C2C_MEASURE_PTR_STEPS);
    restore_irqs(saved_mstatus);
    stats_add(&stats, cycles);
  }

  C2C_MEASURE_LOG("[c2c-measure] ptr_write trials=%u steps=%u best=%llu avg=%llu worst=%llu",
                  (unsigned)C2C_MEASURE_PTR_TRIALS,
                  (unsigned)C2C_MEASURE_PTR_STEPS,
                  (unsigned long long)stats.best,
                  (unsigned long long)stats_avg(&stats, C2C_MEASURE_PTR_TRIALS),
                  (unsigned long long)stats.worst);
  print_cycles_per_step("best_cycles/step", stats.best, C2C_MEASURE_PTR_STEPS);
  print_cycles_per_step("avg_cycles/step", stats_avg(&stats, C2C_MEASURE_PTR_TRIALS), C2C_MEASURE_PTR_STEPS);
  C2C_MEASURE_LOG("\n");
}

void app_init(void) {
  init_test(target_frequency);
  g_cache_sink = 0u;
  g_ptr_sink = 0u;

  g_measure_overhead_cycles = measure_overhead_best(C2C_MEASURE_OVERHEAD_TRIALS);

  C2C_MEASURE_LOG("[c2c-measure] bearly-only measure init\n");
  C2C_MEASURE_LOG("[c2c-measure] shm_base=0x%08lx shm_bytes=%u ptr_nodes=%u\n",
                  (unsigned long)C2C_MEASURE_SHM_BASE,
                  (unsigned)C2C_MEASURE_SHM_BYTES,
                  (unsigned)C2C_MEASURE_PTR_NODES);
  C2C_MEASURE_LOG("[c2c-measure] cache_evict_bytes=%u line_bytes=%u passes=%u\n",
                  (unsigned)C2C_MEASURE_CACHE_EVICT_BYTES,
                  (unsigned)C2C_MEASURE_CACHE_LINE_BYTES,
                  (unsigned)C2C_MEASURE_CACHE_EVICT_PASSES);
  C2C_MEASURE_LOG("[c2c-measure] timer_overhead(best)=%llu cycles\n",
                  (unsigned long long)g_measure_overhead_cycles);
}

void app_main(void) {
  clear_scratch_region();
  build_pointer_ring();

  run_cache_flush_bench();
  run_ptr_read_bench();
  run_ptr_write_bench();

  C2C_MEASURE_LOG("[c2c-measure] done; entering wfi loop\n");
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
