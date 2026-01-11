#include <stdio.h>
#include <stdint.h>

//I ran this directly in main before, might need some editing now it is its own file

// ---------- RISC-V cycle counter ---------- 
static inline uint64_t rdcycle64(void) {
  uint64_t cycles;
  asm volatile ("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// ---------- Config ----------
#define MAX_BYTES   (1u << 20)   /* 1 MB */
#define MIN_BYTES   (1u << 10)   /* 1 KB */

typedef uint32_t elem_t;

#define MAX_ELEMS   (MAX_BYTES / sizeof(elem_t))
static elem_t buffer[MAX_ELEMS];

//Simple LCG PRNG for deterministic-ish randomness per size 
static uint32_t lcg_state;

static void lcg_seed(uint32_t seed) {
  if (seed == 0) seed = 1;
  lcg_state = seed;
}

static uint32_t lcg_next(void) {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return lcg_state;
}

//Build a RANDOM pointer-chase ring over n_elems entries 
static void build_random_ring(elem_t *buf, uint32_t n_elems) {
  static uint32_t perm[MAX_ELEMS];

  for (uint32_t i = 0; i < n_elems; ++i) perm[i] = i;

  for (uint32_t i = n_elems - 1; i > 0; --i) {
    uint32_t j = lcg_next() % (i + 1);
    uint32_t tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }

  for (uint32_t k = 0; k < n_elems; ++k) {
    uint32_t from = perm[k];
    uint32_t to   = perm[(k + 1) % n_elems];
    buf[from] = to;
  }
}

//Run a pointer chase for `steps` steps starting at index 0 
static uint64_t run_pointer_chase(elem_t *buf, uint64_t steps) {
  uint32_t idx = 0;
  uint64_t start = rdcycle64();
  for (uint64_t i = 0; i < steps; ++i) {
    idx = buf[idx];
  }
  uint64_t end = rdcycle64();

  /* prevent optimization away */
  asm volatile ("" :: "r"(idx));

  return end - start;
}

/* =========================
   Public entry (no params)
   ========================= */
void run_ring_chaser_bench(void) {
  printf("=== CACHE / MEMORY ACCESS TEST (RANDOM pointer chase, short runs) ===\n");

  for (uint32_t bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes <<= 1) {
    uint32_t n_elems = bytes / sizeof(elem_t);
    if (n_elems == 0 || n_elems > MAX_ELEMS) continue;

    /* Seed PRNG based on size so each size is repeatable-ish */
    lcg_seed(bytes ^ 0x12345678u);

    /* Build random pointer-chase ring for this working-set size */
    build_random_ring(buffer, n_elems);

    /* Steps schedule: n_elems*4 clamped to [2000, 20000] */
    uint64_t steps = (uint64_t)n_elems * 4;
    if (steps < 2000ull)  steps = 2000ull;
    if (steps > 20000ull) steps = 20000ull;

    uint64_t cycles = run_pointer_chase(buffer, steps);
    double cycles_per_access = (double)cycles / (double)steps;

    printf("size_bytes=%u  elems=%u  steps=%llu  cycles=%llu  cycles_per_access=%.2f\n",
           bytes,
           n_elems,
           (unsigned long long)steps,
           (unsigned long long)cycles,
           cycles_per_access);
  }

  printf("=== TEST DONE ===\n");
}
