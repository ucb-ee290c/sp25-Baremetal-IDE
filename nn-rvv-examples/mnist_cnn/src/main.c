/*
 * mnist_cnn: Conv→Pool→Conv→Pool→FC→FC(+softmax) MNIST network on nn-rvv.
 *
 *   Input               :  1 × 28 × 28
 *   Conv3×3,16,relu     : 16 × 26 × 26
 *   MaxPool3×3,str=3    : 16 ×  8 ×  8
 *   Conv3×3,32,relu     : 32 ×  6 ×  6
 *   MaxPool3×3,str=3    : 32 ×  2 ×  2
 *   Flatten             : 128
 *   FC 128→32,relu
 *   FC  32→10,softmax
 */
#include "main.h"
#include <nn_rvv/layers.h>
#include <nn_rvv/threading.h>
#include <data/model_params.h>
#include <data/input_data.h>

#ifndef NN_RVV_N_HARTS
#  error "nn-rvv-examples requires nn-rvv: build with -DBUILD_NN_RVV=ON"
#endif

/* activation buffers */
static float conv0_out [1 * 26 * 26];
static float pw0_out   [16 * 26 * 26];
static float pool0_out [16 *  8 *  8];
static float conv1_out [16 *  6 *  6];
static float pw1_out   [32 *  6 *  6];
static float pool1_out [32 *  2 *  2];
static float dense0_out[BATCHES * 32];
static float logits    [BATCHES * 10];
static float probs     [BATCHES * 10];

static int argmax10(const float *v) {
    int idx = 0;
    float mx = v[0];
    for (int i = 1; i < 10; ++i) {
        if (v[i] > mx) { mx = v[i]; idx = i; }
    }
    return idx;
}

void app_main(void) {
    nn_rvv_threading_init();
    printf("mnist_cnn: NN_RVV_N_HARTS=%d\n", (int)NN_RVV_N_HARTS);

    for (int i = 0; i < 18; i++) {
        unsigned long cyc0, cyc1, ins0, ins1;
        asm volatile ("rdcycle %0"   : "=r"(cyc0));
        asm volatile ("rdinstret %0" : "=r"(ins0));

        /* Conv-0: 1×28×28 → 16×26×26 (depthwise 3x3 + pointwise 1x1 + ReLU) */
        dw_conv2D_3x3_f32(28, 28, 1, 1, 0, dw0, input + i*784, conv0_out, 0);
        conv2D_1x1_f32   (26, 26, 1, 16, 1, 0, pw0, conv0_out, pw0_out, 1);

        /* MaxPool-0: 3x3 stride 3 → 16×8×8 */
        maxpool_f32(8, 8, 26, 26, 16, 3, pw0_out, pool0_out);

        /* Conv-1: 16×8×8 → 32×6×6 (depthwise 3x3 + pointwise 1x1 + ReLU) */
        dw_conv2D_3x3_f32(8, 8, 16, 1, 0, dw1, pool0_out, conv1_out, 0);
        conv2D_1x1_f32   (6, 6, 16, 32, 1, 0, pw1, conv1_out, pw1_out, 1);

        /* MaxPool-1: 3x3 stride 3 → 32×2×2 = 128 floats flat */
        maxpool_f32(2, 2, 6, 6, 32, 3, pw1_out, pool1_out);

        /* FC-0: 128 → 32 + ReLU */
        fully_connected_f32(128, 32, BATCHES, pool1_out, fc0, dense0_out, 1);
        /* FC-1: 32 → 10 (logits) */
        fully_connected_f32(32,  10, BATCHES, dense0_out, fc1, logits, 0);

        /* Softmax per sample */
        for (size_t b = 0; b < BATCHES; ++b) {
            softmax_vec(&logits[b*10], &probs[b*10], 10, 1);
        }

        asm volatile ("fence");
        asm volatile ("rdcycle %0"   : "=r"(cyc1));
        asm volatile ("rdinstret %0" : "=r"(ins1));

        printf("Execution cycles      : %lu\n", cyc1 - cyc0);
        printf("Instructions executed : %lu\n", ins1 - ins0);

        for (size_t b = 0; b < BATCHES; ++b) {
            int pred = argmax10(&probs[b*10]);
            printf("Input %d -> Predicted digit %d, probs:", i, pred);
            for (int c = 0; c < 10; ++c) printf(" %d", (int)(100*probs[b*10+c]));
            printf("\n");
        }
    }
}
