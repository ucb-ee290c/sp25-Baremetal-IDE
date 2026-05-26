/*
 * mnist_quant: int8-quantized two-layer FC MNIST network on nn-rvv.
 *
 *   input(f32) → quantize → FC 784→64 (ReLU, int8) → FC 64→10 (int8)
 *               → dequantize → softmax → argmax
 *
 * Uses the legacy `fully_connected_int8` int8/int8 path (separate
 * input/weight/output qparams, no per-channel requantization).
 */
#include "main.h"
#include <nn_rvv/layers.h>
#include <nn_rvv/threading.h>
#include <data/model_params.h>
#include <data/input_data.h>

#ifndef NN_RVV_N_HARTS
#  error "nn-rvv-examples requires nn-rvv: build with -DBUILD_NN_RVV=ON"
#endif

void app_main(void) {
    nn_rvv_threading_init();
    printf("mnist_quant: NN_RVV_N_HARTS=%d\n", (int)NN_RVV_N_HARTS);

    static int8_t input_q   [BATCHES * 784];
    static int8_t dense0_q  [BATCHES *  64];
    static int8_t logits_q  [BATCHES *  10];
    static float  logits_f32[BATCHES *  10];
    static float  probs     [BATCHES *  10];

    unsigned long cyc0, cyc1, ins0, ins1;
    asm volatile ("rdcycle %0"   : "=r"(cyc0));
    asm volatile ("rdinstret %0" : "=r"(ins0));

    quant_f32(BATCHES * 784, input, input_q, qp_input);

    fully_connected_int8(784, 64, BATCHES, input_q, layer0_wb_q, dense0_q,
                         /*relu*/ 1, qp_input, qp_w0, qp_out0);
    fully_connected_int8( 64, 10, BATCHES, dense0_q, layer1_wb_q, logits_q,
                         /*relu*/ 0, qp_out0, qp_w1, qp_out1);

    dequant_f32(BATCHES * 10, logits_q, logits_f32, qp_out1);

    for (size_t b = 0; b < BATCHES; b++) {
        softmax_vec(&logits_f32[b * 10], &probs[b * 10], 10, 1);
    }

    asm volatile ("fence");
    asm volatile ("rdcycle %0"   : "=r"(cyc1));
    asm volatile ("rdinstret %0" : "=r"(ins1));

    printf("  Execution cycles:      %lu\n", cyc1 - cyc0);
    printf("  Instructions executed: %lu\n\n", ins1 - ins0);

    for (size_t b = 0; b < BATCHES; b++) {
        int   pred = 0;
        float mx   = probs[b * 10];
        for (int c = 1; c < 10; c++) {
            if (probs[b * 10 + c] > mx) { mx = probs[b * 10 + c]; pred = c; }
        }
        printf("Input %zu: Predicted digit %d, probs:", b, pred);
        for (int c = 0; c < 10; c++) printf(" %d", (int)(100.0f * probs[b * 10 + c]));
        printf("\n");
    }
}
