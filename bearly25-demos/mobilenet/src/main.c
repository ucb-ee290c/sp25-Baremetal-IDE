/*
 * main.c - MobileNetV2 int8 demo inference on Bearly25.
 *
 * Implements the forward pass with depthwise/pointwise conv blocks, ReLU6,
 * global average pooling, and the final FC layer. Uses generated weights
 * and input samples from the include/ directory.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "../include/model_params_self.h"
#include "../include/inputs.h"
#include "layers.h"

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

void print_int8_matrix(int8_t *arr, size_t rows, size_t cols)
{
    printf("matrix: \n");

    for (size_t r = 0; r < rows; r++) {
        printf("  [");
        for (size_t c = 0; c < cols; c++) {
            int idx = r * cols + c;
            printf("%4d", arr[idx]);
            if (c + 1 < cols) printf(", ");
        }
        printf("]\n");
    }
}

#define MAX_INT8_BUFFER (96 * 112 * 112) /* largest feature map */
#define MAX_CHANNELS    1280

static float relu_tmp[MAX_INT8_BUFFER];

/* Dequant int8 -> float with given rq, apply ReLU6, re-quant to int8. 
   NOTE: we *do not* change scale – we reuse the same requantization_params_t. */
static void relu6_apply_int8(
    int8_t *tensor,
    size_t channels,
    size_t rows,
    size_t cols,
    const requantization_params_t *rq_source)
{
    const size_t total = channels * rows * cols;
    if (total > MAX_INT8_BUFFER) return;

    const int32_t zp =
        (rq_source ? rq_source->zero_point : 0);

    size_t idx = 0;
    for (size_t c = 0; c < channels; ++c) {
        const float s =
            (rq_source && rq_source->scale) ? rq_source->scale[c] : 1.0f;
        for (size_t i = 0; i < rows * cols; ++i, ++idx) {
            relu_tmp[idx] = ((float)tensor[idx] - (float)zp) * s;
        }
    }

    /* relu6_int8 is assumed to:
       - clamp relu_tmp in [0, 6]
       - requantize back to int8 using the same rq (per-channel scales). */
    relu6_int8(channels, rows * cols, relu_tmp, tensor,
               rq_source ? *rq_source
                         : (requantization_params_t){ .scale = NULL, .zero_point = 0 });
}

/* Thin wrapper around existing 1x1 conv kernel. */
static void pointwise_conv1x1_int8(
    size_t H, size_t W,
    size_t Cin, size_t Cout,
    const uint8_t *weights,
    int8_t *input,
    int8_t *output,
    const requantization_params_t *rq)
{
    conv_1x1_int8(
        H, W,
        Cin, Cout,
        1, 0,
        input,
        (const void *)weights,
        output,
        /* relu */ 0,
        rq ? *rq : (requantization_params_t){ .scale = NULL, .zero_point = 0 });
}

/* -------------------------------------------------------------------------- */
/* Inverted residual block descriptor                                         */
/* -------------------------------------------------------------------------- */

typedef struct {
    size_t expand;       /* t */
    size_t out_ch;       /* c_out */
    size_t stride;       /* s */
    int    use_residual; /* when stride==1 && Cin==Cout */

    const uint8_t *w_expand; /* may be NULL if expand==1 */
    const uint8_t *w_dw;
    const uint8_t *w_pw;

    const requantization_params_t *rq_expand; /* may be NULL if expand==1 */
    const requantization_params_t *rq_dw;
    const requantization_params_t *rq_pw;     /* also used for residual_add */
} block_desc_t;

/* One block = [optional PW-expand] -> DW -> PW-proj (+ optional residual). */
static void run_block(
    const block_desc_t *cfg,
    size_t in_h, size_t in_w, size_t in_ch,
    int8_t *input,
    int8_t *buf0,
    int8_t *buf1,
    int8_t **out,
    size_t *out_h,
    size_t *out_w,
    size_t *out_ch)
{
    const size_t hidden_ch = (cfg->expand == 0 ? in_ch : in_ch * cfg->expand);
    const int has_expand = (cfg->expand > 1);

    /* 1) Expansion 1x1 (optional), output scale = rq_expand */
    int8_t *after_expand = input;
    if (has_expand) {
        pointwise_conv1x1_int8(
            in_h, in_w,
            in_ch, hidden_ch,
            cfg->w_expand,
            input, buf0,
            cfg->rq_expand);
        relu6_apply_int8(buf0, hidden_ch, in_h, in_w, cfg->rq_expand);
        after_expand = buf0;
    }

    /* 2) Depthwise 3x3 stride {1,2} with SAME padding, output scale = rq_dw */
    const size_t dw_out_h = (in_h + cfg->stride - 1) / cfg->stride;
    const size_t dw_out_w = (in_w + cfg->stride - 1) / cfg->stride;

    dwconv2D_3x3_int8(
        in_h, in_w,
        hidden_ch,
        cfg->stride,
        /* padding = SAME */ 1,
        (const void *)cfg->w_dw,
        after_expand,
        buf1,
        /* relu */ 0,
        cfg->rq_dw ? *cfg->rq_dw
                   : (requantization_params_t){ .scale = NULL, .zero_point = 0 });
    relu6_apply_int8(buf1, hidden_ch, dw_out_h, dw_out_w, cfg->rq_dw);

    /* 3) Pointwise projection 1x1, output scale = rq_pw */
    pointwise_conv1x1_int8(
        dw_out_h, dw_out_w,
        hidden_ch, cfg->out_ch,
        cfg->w_pw,
        buf1, buf0,
        cfg->rq_pw);
    relu6_apply_int8(buf0, cfg->out_ch, dw_out_h, dw_out_w, cfg->rq_pw);

    /* 4) Residual add (if used)
           We keep the SAME activation scale as cfg->rq_pw. */
    if (cfg->use_residual && cfg->stride == 1 && in_ch == cfg->out_ch) {
        residual_add(
            dw_out_h, dw_out_w,
            cfg->out_ch,
            input,      /* skip-connection (already int8 at previous layer's scale) */
            buf0,       /* block output */
            buf0,       /* in-place */
            cfg->rq_pw ? *cfg->rq_pw
                       : (requantization_params_t){ .scale = NULL, .zero_point = 0 });
    }

    *out    = buf0;
    *out_h  = dw_out_h;
    *out_w  = dw_out_w;
    *out_ch = cfg->out_ch;
}

/* -------------------------------------------------------------------------- */
/* Scalar global average pool (H×W → 1) per channel                           */
/* -------------------------------------------------------------------------- */
void avgpool_global_7x7_int8(
    const int8_t *input,
    size_t channels,
    size_t H, size_t W,
    int8_t *output)
{
    const size_t spatial = H * W;
    if (!input || !output || spatial == 0) return;

    for (size_t c = 0; c < channels; ++c) {
        const int8_t *in_ch = input + c * spatial;
        int32_t sum = 0;
        for (size_t i = 0; i < spatial; ++i) {
            sum += (int32_t)in_ch[i];
        }
        int32_t avg = 0;
        if (sum >= 0) {
            avg = (sum + (int32_t)(spatial / 2)) / (int32_t)spatial;
        } else {
            avg = (sum - (int32_t)(spatial / 2)) / (int32_t)spatial;
        }
        if (avg > 127) avg = 127;
        if (avg < -128) avg = -128;
        output[c] = (int8_t)avg;
    }
}

/* -------------------------------------------------------------------------- */
/* Forward pass                                                               */
/* -------------------------------------------------------------------------- */
void mobilenet_forward(const float *input_f32, float *logits_f32)
{
    static int8_t buf0[MAX_INT8_BUFFER];
    static int8_t buf1[MAX_INT8_BUFFER];
    static int8_t pooled[1280];
    static int8_t logits_q[10];

    size_t h = 224, w = 224, ch = 3;

    /* 0) Quantize input using qp_input (per-tensor) */
    quant_f32(
        BATCHES * ch * h * w,
        (float *)input_f32,
        buf0,
        qp_input);

    /* ------------------------------------------------------------------ */
    /* Stem: DW 3x3 s2 SAME (3->3) then PW 1x1 (3->32)                    */
    /* conv names: stem.0.0 (DW), stem.1.0 (PW) → rq_stem_0_0, rq_stem_1_0 */
    /* ------------------------------------------------------------------ */
    const size_t stem_h = (h + 2 - 3) / 2 + 1;
    const size_t stem_w = stem_h; /* input assumed square 224x224 */

    /* DW stem */
    dwconv2D_3x3_int8(
        h, w,
        ch,
        /* stride */ 2,
        /* padding */ 1,
        (const void *)stem_0_0_wb_q,
        buf0,
        buf1,
        /* relu */ 0,
        rq_stem_0_0);
    relu6_apply_int8(buf1, ch, stem_h, stem_w, &rq_stem_0_0);

    /* PW stem */
    pointwise_conv1x1_int8(
        stem_h, stem_w,
        ch, 32,
        stem_1_0_wb_q,
        buf1, buf0,
        &rq_stem_1_0);
    relu6_apply_int8(buf0, 32, stem_h, stem_w, &rq_stem_1_0);

    h = stem_h;
    w = stem_w;
    ch = 32;

    printf("Mobilenet forward checkpoint 1 (after stem): h=%zu w=%zu ch=%zu\n", h, w, ch);

    /* ------------------------------------------------------------------ */
    /* Inverted residual stack                                            */
    /* Each block uses per-layer rq_*: expand, dw, pw                     */
    /* ------------------------------------------------------------------ */
    static const block_desc_t blocks[] = {
        /* t=1, 32->16, stride=1, no residual (first block) */
        {1, 16, 1, 0,
         /* w_expand */ NULL,
         /* w_dw     */ blocks_0_0_0_wb_q,
         /* w_pw     */ blocks_0_1_0_wb_q,
         /* rq_expand */ NULL,
         /* rq_dw     */ &rq_blocks_0_0_0,
         /* rq_pw     */ &rq_blocks_0_1_0},

        /* t=6, 16->24, stride=2, no residual */
        {6, 24, 2, 0,
         blocks_1_0_0_wb_q,
         blocks_1_1_0_wb_q,
         blocks_1_2_0_wb_q,
         &rq_blocks_1_0_0,
         &rq_blocks_1_1_0,
         &rq_blocks_1_2_0},

        /* t=6, 24->24, stride=1, residual */
        {6, 24, 1, 1,
         blocks_2_0_0_wb_q,
         blocks_2_1_0_wb_q,
         blocks_2_2_0_wb_q,
         &rq_blocks_2_0_0,
         &rq_blocks_2_1_0,
         &rq_blocks_2_2_0},

        /* t=6, 24->32, stride=2, no residual */
        {6, 32, 2, 0,
         blocks_3_0_0_wb_q,
         blocks_3_1_0_wb_q,
         blocks_3_2_0_wb_q,
         &rq_blocks_3_0_0,
         &rq_blocks_3_1_0,
         &rq_blocks_3_2_0},

        /* t=6, 32->32, stride=1, residual */
        {6, 32, 1, 1,
         blocks_4_0_0_wb_q,
         blocks_4_1_0_wb_q,
         blocks_4_2_0_wb_q,
         &rq_blocks_4_0_0,
         &rq_blocks_4_1_0,
         &rq_blocks_4_2_0},

        {6, 32, 1, 1,
         blocks_5_0_0_wb_q,
         blocks_5_1_0_wb_q,
         blocks_5_2_0_wb_q,
         &rq_blocks_5_0_0,
         &rq_blocks_5_1_0,
         &rq_blocks_5_2_0},

        {6, 64, 2, 0,
         blocks_6_0_0_wb_q,
         blocks_6_1_0_wb_q,
         blocks_6_2_0_wb_q,
         &rq_blocks_6_0_0,
         &rq_blocks_6_1_0,
         &rq_blocks_6_2_0},

        {6, 64, 1, 1,
         blocks_7_0_0_wb_q,
         blocks_7_1_0_wb_q,
         blocks_7_2_0_wb_q,
         &rq_blocks_7_0_0,
         &rq_blocks_7_1_0,
         &rq_blocks_7_2_0},

        {6, 64, 1, 1,
         blocks_8_0_0_wb_q,
         blocks_8_1_0_wb_q,
         blocks_8_2_0_wb_q,
         &rq_blocks_8_0_0,
         &rq_blocks_8_1_0,
         &rq_blocks_8_2_0},

        {6, 64, 1, 1,
         blocks_9_0_0_wb_q,
         blocks_9_1_0_wb_q,
         blocks_9_2_0_wb_q,
         &rq_blocks_9_0_0,
         &rq_blocks_9_1_0,
         &rq_blocks_9_2_0},

        {6, 96, 1, 0,
         blocks_10_0_0_wb_q,
         blocks_10_1_0_wb_q,
         blocks_10_2_0_wb_q,
         &rq_blocks_10_0_0,
         &rq_blocks_10_1_0,
         &rq_blocks_10_2_0},

        {6, 96, 1, 1,
         blocks_11_0_0_wb_q,
         blocks_11_1_0_wb_q,
         blocks_11_2_0_wb_q,
         &rq_blocks_11_0_0,
         &rq_blocks_11_1_0,
         &rq_blocks_11_2_0},

        {6, 96, 1, 1,
         blocks_12_0_0_wb_q,
         blocks_12_1_0_wb_q,
         blocks_12_2_0_wb_q,
         &rq_blocks_12_0_0,
         &rq_blocks_12_1_0,
         &rq_blocks_12_2_0},

        {6, 160, 2, 0,
         blocks_13_0_0_wb_q,
         blocks_13_1_0_wb_q,
         blocks_13_2_0_wb_q,
         &rq_blocks_13_0_0,
         &rq_blocks_13_1_0,
         &rq_blocks_13_2_0},

        {6, 160, 1, 1,
         blocks_14_0_0_wb_q,
         blocks_14_1_0_wb_q,
         blocks_14_2_0_wb_q,
         &rq_blocks_14_0_0,
         &rq_blocks_14_1_0,
         &rq_blocks_14_2_0},

        {6, 160, 1, 1,
         blocks_15_0_0_wb_q,
         blocks_15_1_0_wb_q,
         blocks_15_2_0_wb_q,
         &rq_blocks_15_0_0,
         &rq_blocks_15_1_0,
         &rq_blocks_15_2_0},

        {6, 320, 1, 0,
         blocks_16_0_0_wb_q,
         blocks_16_1_0_wb_q,
         blocks_16_2_0_wb_q,
         &rq_blocks_16_0_0,
         &rq_blocks_16_1_0,
         &rq_blocks_16_2_0},
    };

    int8_t *cur = buf0;
    for (size_t i = 0; i < sizeof(blocks) / sizeof(blocks[0]); ++i) {
        int8_t *out = NULL;
        run_block(&blocks[i],
                  h, w, ch,
                  cur,
                  buf0,
                  buf1,
                  &out,
                  &h, &w, &ch);
        cur = out;
    }

    printf("Mobilenet forward checkpoint 2 (after blocks): h=%zu w=%zu ch=%zu\n", h, w, ch);

    /* ------------------------------------------------------------------ */
    /* Head: 1x1 320->1280, conv name: head.0.0 → rq_head_0_0             */
    /* ------------------------------------------------------------------ */
    pointwise_conv1x1_int8(
        h, w,
        ch, 1280,
        head_0_wb_q,
        cur, buf0,
        &rq_head_0);
    relu6_apply_int8(buf0, 1280, h, w, &rq_head_0);
    cur = buf0;
    ch  = 1280;

    /* Global average pool 7x7 -> 1x1 (doesn't change scale) */
    avgpool_global_7x7_int8(cur, ch, h, w, pooled);

    /* FC 1280 -> 10, rq_fc is generated for 'fc' layer */
    quant_fully_connected_int8(
        1280,
        10,
        BATCHES,
        pooled,
        (const void *)fc_wb_q,
        logits_q,
        /* relu */ 0,
        /* bias32 */ 0,
        rq_fc);

    /* Dequantize logits for host-side consumption using qp_logits */
    dequant_f32(
        10 * BATCHES,
        logits_q,
        logits_f32,
        qp_logits);
}

int main(void)
{
    static float input_f32[BATCHES * 3 * 224 * 224] = {0};
    static float logits_f32[BATCHES * 10];

    printf("Starting MobileNet\n");

    /* Load first test sample from inputs.h into float buffer */
    const size_t total = INPUT_C * INPUT_H * INPUT_W;
    for (size_t i = 0; i < total; ++i) {
        input_f32[i] = (float)input_samples[0][i];
    }

    printf("Loaded Sample\n");

    mobilenet_forward(input_f32, logits_f32);

    printf("Completed Forward Pass\n");

    /* Compute argmax of logits for quick sanity check */
    int top = 0;
    for (int i = 1; i < 10; ++i) {
        if (logits_f32[i] > logits_f32[top]) {
            top = i;
        }
    }
    printf("Top-1 class: %d  logit: %.4f\n", top, logits_f32[top]);
    return top;
}
