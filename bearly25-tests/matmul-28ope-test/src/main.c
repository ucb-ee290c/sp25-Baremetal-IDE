/*
 * matmul-28ope-test/src/main.c
 *
 * 32x32 int8 matmul benchmark with multiple run modes:
 *
 *   Single-core:
 *     - OPE+RVV interleaved: 8 OPE rows + 24 RVV rows
 *     - Pure RVV:            32 rows via Saturn
 *
 *   Multicore (2 cores):
 *     - OPE+RVV interleaved: each core does 8 OPE + 8 RVV = 16 rows
 *     - Pure RVV:            each core does 16 rows via Saturn
 *
 * Uses cold/hot benchmark runs similar to bearly25-bmarks/rvv-matmul.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "chip_config.h"
#include "bench_cache.h"
#include "bench_config.h"
#include "simple_setup.h"
#include <hthread.h>

#define DIM        32   /* must be multiple of 8 */

/* Single-core split */
#define SC_OPE_ROWS  8
#define SC_RVV_ROWS  24  /* DIM - SC_OPE_ROWS */

/* Multicore split: each core gets 16 rows (8 OPE + 8 RVV) */
#define MC_ROWS_PER_CORE 16
#define MC_OPE_ROWS       8
#define MC_RVV_ROWS       8  /* MC_ROWS_PER_CORE - MC_OPE_ROWS */

#define RUNS_COLD  1
#define RUNS_HOT   0

/* ── forward declarations (opervv.c) ── */
void pack_A_ope(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K);
void pack_B_ope(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void pack_A_rvv(const int8_t *A, int lda, int8_t *A_packed,
                int row_start, int num_rows, int K);
void pack_B_rvv_bias(const int8_t *B, int ldb, int8_t *B_packed, int K, int N);
void gemm_ope_16rows(const int8_t *A_packed, const int8_t *B_packed,
                     int32_t *C, int N, int K, int ldc,
                     const int8_t *A_rvv, const int8_t *B_rvv,
                     int num_rvv_rows, int32_t *C_rvv);
void gemm_rvv_32rows(const int8_t *A_rvv, const int8_t *B_rvv,
                     int32_t *C, int N, int K, int ldc);
void gemm_rvv_rows(const int8_t *A_rvv, const int8_t *B_rvv,
                   int32_t *C, int N, int K, int ldc, int num_rows);

/* ── static buffers (BSS) ── */

static int8_t  A[DIM * DIM];
static int8_t  B[DIM * DIM];
uint64_t target_frequency = 150000000l;


/* OPE packed inputs — aligned for RoCC */
/* Single-core: 1 tile of 8 rows */
static int8_t  A_ope_sc[(SC_OPE_ROWS / 8) * DIM * 8] __attribute__((aligned(64)));
/* Multicore: 1 tile of 8 rows per core */
static int8_t  A_ope_c0[(MC_OPE_ROWS / 8) * DIM * 8] __attribute__((aligned(64)));
static int8_t  A_ope_c1[(MC_OPE_ROWS / 8) * DIM * 8] __attribute__((aligned(64)));

static int8_t  B_ope[(DIM / 8) * DIM * 8] __attribute__((aligned(64)));

/* B_rvv: (DIM+1) x DIM row-major, row 0 = zero bias */
static int8_t  B_rvv[(DIM + 1) * DIM] __attribute__((aligned(64)));

/* Outputs — page-align for deterministic cache set mapping */
static int32_t C_out[DIM * DIM]      __attribute__((aligned(4096)));

/* ── timing / stats ── */
/* rdcycle64() is provided by bench_config.h */

typedef struct {
    uint64_t sum;
    uint64_t best;
    int runs;
} bench_stats_t;

static inline void stats_init(bench_stats_t *s) {
    s->sum = 0;
    s->best = ULLONG_MAX;
    s->runs = 0;
}

static inline void stats_update(bench_stats_t *s, uint64_t cycles) {
    s->sum += cycles;
    if (cycles < s->best) s->best = cycles;
    s->runs++;
}

static inline uint64_t stats_avg(const bench_stats_t *s) {
    return s->runs > 0 ? s->sum / (uint64_t)s->runs : 0;
}

static void stats_print(const char *tag, const bench_stats_t *cold, const bench_stats_t *hot) {
    printf("  %-24s COLD(runs=%d best=%llu avg=%llu) HOT(runs=%d best=%llu avg=%llu)\n",
           tag,
           cold->runs,
           (unsigned long long)cold->best,
           (unsigned long long)stats_avg(cold),
           hot->runs,
           (unsigned long long)hot->best,
           (unsigned long long)stats_avg(hot));
}

/* ── priming helpers ── */

static void prime_inputs_sc(void) {
    volatile int8_t sink;
    volatile int32_t sink32;
    /* OPE output region (8 rows) */
    for (int r = 0; r < SC_OPE_ROWS; r++)
        for (int c = 0; c < DIM; c += 16)
            sink32 = C_out[r * DIM + c];
    /* OPE ACC inputs */
    for (int off = 0; off < (int)sizeof(A_ope_sc); off += 64)
        sink = A_ope_sc[off];
    for (int off = 0; off < (int)sizeof(B_ope); off += 64)
        sink = B_ope[off];
    /* RVV inputs */
    for (int off = 0; off < SC_RVV_ROWS * DIM; off += 64)
        sink = (A + SC_OPE_ROWS * DIM)[off];
    for (int off = 0; off < (DIM + 1) * DIM; off += 64)
        sink = B_rvv[off];
    /* RVV output region */
    for (int r = 0; r < SC_RVV_ROWS; r += 4)
        sink32 = C_out[(SC_OPE_ROWS + r) * DIM];
    asm volatile("fence r, rw" ::: "memory");
}

static void prime_rvv_only(void) {
    volatile int8_t sink;
    volatile int32_t sink32;
    for (int off = 0; off < DIM * DIM; off += 64)
        sink = A[off];
    for (int off = 0; off < (DIM + 1) * DIM; off += 64)
        sink = B_rvv[off];
    for (int r = 0; r < DIM; r += 4)
        sink32 = C_out[r * DIM];
    asm volatile("fence r, rw" ::: "memory");
}

/* ── single-core kernels ── */

static void run_sc_ope_rvv(void) {
    memset(C_out, 0, sizeof(C_out));
    prime_inputs_sc();
    gemm_ope_16rows(A_ope_sc, B_ope, C_out, DIM, DIM, DIM,
                    A + SC_OPE_ROWS * DIM, B_rvv, SC_RVV_ROWS,
                    C_out + SC_OPE_ROWS * DIM);
}

static void run_sc_rvv_only(void) {
    memset(C_out, 0, sizeof(C_out));
    prime_rvv_only();
    gemm_rvv_32rows(A, B_rvv, C_out, DIM, DIM, DIM);
}

/* ── multicore worker ── */

typedef struct {
    const int8_t *A_ope;
    const int8_t *B_ope;
    const int8_t *A_rvv;
    const int8_t *B_rvv;
    int32_t *C;
    int32_t *C_rvv;
    int N;
    int K;
    int ldc;
    int num_rvv_rows;
    int mode;  /* 0 = OPE+RVV, 1 = pure RVV */
    int total_rows;  /* for pure-RVV mode */
} mc_worker_arg_t;

static void *mc_worker(void *arg_) {
    mc_worker_arg_t *arg = (mc_worker_arg_t *)arg_;
    if (arg->mode == 0) {
        /* OPE+RVV interleaved */
        gemm_ope_16rows(arg->A_ope, arg->B_ope,
                        arg->C, arg->N, arg->K, arg->ldc,
                        arg->A_rvv, arg->B_rvv,
                        arg->num_rvv_rows, arg->C_rvv);
    } else {
        /* Pure RVV */
        gemm_rvv_rows(arg->A_rvv, arg->B_rvv,
                      arg->C, arg->N, arg->K, arg->ldc,
                      arg->total_rows);
    }
    return NULL;
}

/* ── multicore OPE+RVV (each core: 8 OPE + 8 RVV = 16 rows) ── */

static void run_mc_ope_rvv(void) {
    memset(C_out, 0, sizeof(C_out));

    mc_worker_arg_t args[2];

    /* Core 0: rows 0-15 (OPE rows 0-7, RVV rows 8-15) */
    args[0].A_ope = A_ope_c0;
    args[0].B_ope = B_ope;
    args[0].A_rvv = A + MC_OPE_ROWS * DIM;  /* row 8 */
    args[0].B_rvv = B_rvv;
    args[0].C = C_out;
    args[0].C_rvv = C_out + MC_OPE_ROWS * DIM;
    args[0].N = DIM;
    args[0].K = DIM;
    args[0].ldc = DIM;
    args[0].num_rvv_rows = MC_RVV_ROWS;
    args[0].mode = 0;

    /* Core 1: rows 16-31 (OPE rows 16-23, RVV rows 24-31) */
    args[1].A_ope = A_ope_c1;
    args[1].B_ope = B_ope;
    args[1].A_rvv = A + (MC_ROWS_PER_CORE + MC_OPE_ROWS) * DIM;  /* row 24 */
    args[1].B_rvv = B_rvv;
    args[1].C = C_out + MC_ROWS_PER_CORE * DIM;
    args[1].C_rvv = C_out + (MC_ROWS_PER_CORE + MC_OPE_ROWS) * DIM;
    args[1].N = DIM;
    args[1].K = DIM;
    args[1].ldc = DIM;
    args[1].num_rvv_rows = MC_RVV_ROWS;
    args[1].mode = 0;

    asm volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, mc_worker, &args[1]);
    (void)mc_worker(&args[0]);
    hthread_join(1);
    asm volatile("fence rw, rw" ::: "memory");
}

/* ── multicore pure-RVV (each core: 16 rows) ── */

static void run_mc_rvv_only(void) {
    memset(C_out, 0, sizeof(C_out));

    mc_worker_arg_t args[2];

    /* Core 0: rows 0-15 */
    args[0].A_rvv = A;
    args[0].B_rvv = B_rvv;
    args[0].C = C_out;
    args[0].N = DIM;
    args[0].K = DIM;
    args[0].ldc = DIM;
    args[0].mode = 1;
    args[0].total_rows = MC_ROWS_PER_CORE;

    /* Core 1: rows 16-31 */
    args[1].A_rvv = A + MC_ROWS_PER_CORE * DIM;
    args[1].B_rvv = B_rvv;
    args[1].C = C_out + MC_ROWS_PER_CORE * DIM;
    args[1].N = DIM;
    args[1].K = DIM;
    args[1].ldc = DIM;
    args[1].mode = 1;
    args[1].total_rows = MC_ROWS_PER_CORE;

    asm volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, mc_worker, &args[1]);
    (void)mc_worker(&args[0]);
    hthread_join(1);
    asm volatile("fence rw, rw" ::: "memory");
}

/* ── bench runner (cold + hot) ── */

typedef void (*run_fn_t)(void);

static void bench_run(const char *tag, run_fn_t fn) {
    bench_stats_t cold, hot;
    stats_init(&cold);
    stats_init(&hot);

    /* Cold runs (cache flush between each) */
    for (int r = 0; r < RUNS_COLD; r++) {
        bench_cache_flush();
        uint64_t t0 = rdcycle64();
        fn();
        uint64_t t1 = rdcycle64();
        asm volatile("fence rw, rw" ::: "memory");
        stats_update(&cold, t1 - t0);
    }

    /* Warm-up */
    bench_cache_flush();
    fn();
    asm volatile("fence rw, rw" ::: "memory");

    /* Hot runs (back to back, cache warm) */
    for (int r = 0; r < RUNS_HOT; r++) {
        uint64_t t0 = rdcycle64();
        fn();
        uint64_t t1 = rdcycle64();
        asm volatile("fence rw, rw" ::: "memory");
        stats_update(&hot, t1 - t0);
    }

    stats_print(tag, &cold, &hot);
}

/* ── application entry ── */

void app_init(void) {
    init_test(target_frequency);
    printf("INit: debug\n");
    bench_cache_init();

    /* Warm-up hart 1 — no-op dispatch to clear any spurious WFI state */
    static mc_worker_arg_t nop_arg = {
        .A_rvv = NULL, .B_rvv = NULL, .C = NULL,
        .N = 0, .K = 0, .ldc = 0, .mode = 1, .total_rows = 0
    };
    hthread_issue(1, mc_worker, &nop_arg);
    hthread_join(1);
}

void app_main(void)
{
    printf("=== matmul-28ope-test: %dx%d int8 benchmark ===\n", DIM, DIM);
    printf("  runs_cold=%d, runs_hot=%d\n", RUNS_COLD, RUNS_HOT);

    /* Fill A and B with a deterministic pattern */
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            A[i * DIM + j] = (int8_t)((i * 3 + j * 7 + 1) % 17 - 8);
            B[i * DIM + j] = (int8_t)((i * 5 + j * 2 + 3) % 13 - 6);
        }

    /* Pack B (shared across all modes) */
    pack_B_ope(B, DIM, B_ope, DIM, DIM);
    pack_B_rvv_bias(B, DIM, B_rvv, DIM, DIM);

    /* Pack A for single-core OPE (rows 0-7) */
    pack_A_ope(A, DIM, A_ope_sc, 0, SC_OPE_ROWS, DIM);

    /* Pack A for multicore OPE: core 0 rows 0-7, core 1 rows 16-23 */
    pack_A_ope(A, DIM, A_ope_c0, 0, MC_OPE_ROWS, DIM);
    pack_A_ope(A, DIM, A_ope_c1, MC_ROWS_PER_CORE, MC_OPE_ROWS, DIM);

    /* ── Single-core benchmarks ── */
    printf("\n--- Single-core ---\n");
    bench_run("sc_ope_rvv(8+24)", run_sc_ope_rvv);
    // bench_run("sc_rvv_only(32)",  run_sc_rvv_only);

    /* ── Multicore benchmarks ── */
    printf("\n--- Multicore (2 cores) ---\n");
    bench_run("mc_ope_rvv(8+8)x2", run_mc_ope_rvv);
    // bench_run("mc_rvv_only(16)x2", run_mc_rvv_only);

    printf("\n=== done ===\n");
}

int main(void)
{
    app_init();
    app_main();
    return 0;
}
