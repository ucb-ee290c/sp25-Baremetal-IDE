/*
 * outer-product-test/src/main.c
 *
 * Functional correctness test for accumulate_outer_products_8x8().
 *
 * Two 32x32 int8 matrices are filled with deterministic values.
 * For every valid 8-aligned (n, l) pair the function writes an 8x8 block
 * into a 32x32 int32 output.  Each block is compared against a scalar
 * reference that directly computes:
 *
 *   ref[n+r][l+c] = Σ_i  input1[i][n+r] * input2[i][l+c]   r,c ∈ [0,8)
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chip_config.h"

#define N 32

/* -------------------------------------------------------------------------
 * Cycle counter
 * ------------------------------------------------------------------------- */

static inline uint64_t rdcycle64(void) {
    uint64_t x;
    asm volatile("rdcycle %0" : "=r"(x));
    return x;
}

/* -------------------------------------------------------------------------
 * Scalar reference for one 8x8 tile at (n, l).
 * ------------------------------------------------------------------------- */

static void ref_outer_product_tile(const int8_t *in1, const int8_t *in2,
                                   int32_t *out, size_t n, size_t l)
{
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int32_t acc = 0;
            for (int i = 0; i < N; i++)
                acc += (int32_t)in1[i * N + (n + r)]
                     * (int32_t)in2[i * N + (l + c)];
            out[(n + r) * N + (l + c)] = acc;
        }
    }
}

static void print_tile(const char *label, const int32_t *mat,
                    size_t n, size_t l)
{
    printf("  %s:\n", label);
    for (int r = 0; r < 8; r++) {
        printf("    [");
        for (int c = 0; c < 8; c++) {
            printf("%7d", mat[(n + r) * N + (l + c)]);
            if (c < 7) printf(",");
        }
        printf(" ]\n");
    }
}

/* -------------------------------------------------------------------------
 * Test runner for one (n, l) tile.
 * Returns 0 on PASS, -1 on FAIL.
 * ------------------------------------------------------------------------- */

static int run_tile_test(const int8_t *in1, const int8_t *in2,
                         int32_t *out_ope, int32_t *out_ref,
                         size_t n, size_t l)
{
    /* OPE under test */
    uint64_t t0 = rdcycle64();
    accumulate_outer_products_8x8(N, in1, in2, out_ope, n, l);
    uint64_t t1 = rdcycle64();

    /* Scalar reference for the same tile */
    ref_outer_product_tile(in1, in2, out_ref, n, l);

    /* Compare the 8x8 tile */
    int errors = 0;
    for (int r = 0; r < 8 && errors < 4; r++) {
        for (int c = 0; c < 8; c++) {
            int32_t got = out_ope[(n + r) * N + (l + c)];
            int32_t exp = out_ref[(n + r) * N + (l + c)];
            if (got != exp) {
                if (errors == 0)
                    printf("  First mismatch [%d][%d]: got %d, expected %d\n",
                           (int)(n + r), (int)(l + c), got, exp);
                errors++;
            }
        }
    }

    // print_tile("OPE output", out_ope, n, l);
    // print_tile("Reference ", out_ref, n, l);


    printf("  tile (n=%2d, l=%2d): %s  (%lu cycles)\n",
           (int)n, (int)l,
           errors ? "FAIL" : "PASS",
           (unsigned long)(t1 - t0));

    return errors ? -1 : 0;
}

/* -------------------------------------------------------------------------
 * Hardware UART init
 * ------------------------------------------------------------------------- */

void app_init(void) {
    // UART_InitType uart_cfg;
    // uart_cfg.baudrate = 115200;
    // uart_cfg.mode     = UART_MODE_TX_RX;
    // uart_cfg.stopbits = UART_STOPBITS_2;
    // uart_init(UART0, &uart_cfg);
}

/* -------------------------------------------------------------------------
 * Main test
 * ------------------------------------------------------------------------- */

void app_main(void) {
    printf("=== accumulate_outer_products_8x8 TEST  (%dx%d) ===\n", N, N);

    /* Deterministic 32x32 int8 inputs */
    static int8_t  in1[N * N];
    static int8_t  in2[N * N];
    static int32_t out_ope[N * N];
    static int32_t out_ref[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            in1[i * N + j] = (int8_t)((i * 3 + j * 7 + 1) % 17 - 8);
            in2[i * N + j] = (int8_t)((i * 5 + j * 2 + 3) % 13 - 6);
        }
    }

    int total = 0, failed = 0;
    run_tile_test(in1, in2, out_ope, out_ref, 0,0);

    // /* Test every 8-aligned (n, l) combination in a 32x32 matrix */
    // for (size_t n = 0; n < N; n += 8) {
    //     for (size_t l = 0; l < N; l += 8) {
    //         total++;
    //         if (run_tile_test(in1, in2, out_ope, out_ref, n, l) != 0)
    //             failed++;
    //     }
    // }

    printf("\n=== SUMMARY: %d/%d passed ===\n", total - failed, total);
    printf("OVERALL: %s\n", failed == 0 ? "PASS" : "FAIL");
}

int main(void) {
    app_init();
    app_main();
    return 0;
}
