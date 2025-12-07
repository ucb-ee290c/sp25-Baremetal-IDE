#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../include/model_params_self.h"
#include "layers.h"

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */
#define MAX_INT8_BUFFER (96 * 112 * 112) /* covers largest feature map */
#define MAX_CHANNELS    1280

static float relu_tmp[MAX_INT8_BUFFER];
static float rq_tmp_scale[MAX_CHANNELS];

static requantization_params_t make_uniform_rq(
    float scale, int32_t zero_point, size_t channels)
{
    if (scale <= 0.0f) scale = 1.0f;
    if (channels > MAX_CHANNELS) channels = MAX_CHANNELS;
    for (size_t i = 0; i < channels; ++i) {
        rq_tmp_scale[i] = scale;
    }
    requantization_params_t rq = { rq_tmp_scale, zero_point };
    return rq;
}

/* Dequant int8 -> float with a uniform scale, apply ReLU6, re-quant to int8. */
static void relu6_apply_int8(
    int8_t *tensor,
    size_t channels,
    size_t rows,
    size_t cols,
    const requantization_params_t *rq_source)
{
    const size_t total = channels * rows * cols;
    if (total > MAX_INT8_BUFFER) return;

    const float scale_val =
        (rq_source && rq_source->scale) ? rq_source->scale[0] : 1.0f;
    const int32_t zp =
        (rq_source) ? rq_source->zero_point : 0;

    requantization_params_t rq = make_uniform_rq(scale_val, zp, channels);
    size_t idx = 0;
    for (size_t c = 0; c < channels; ++c) {
        const float s = rq.scale[c];
        for (size_t i = 0; i < rows * cols; ++i) {
            relu_tmp[idx] = ((float)tensor[idx] - (float)zp) * s;
            idx++;
        }
    }
    relu6_int8(channels, rows * cols, relu_tmp, tensor, rq);
}

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
        *rq);
}

typedef struct {
    size_t expand;       /* t */
    size_t out_ch;       /* c_out */
    size_t stride;       /* s */
    int use_residual;    /* when stride==1 && Cin==Cout */
    const uint8_t *w_expand;
    const uint8_t *w_dw;
    const uint8_t *w_pw;
    const requantization_params_t *rq; /* per-stage scales (len = out_ch) */
} block_desc_t;

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
    const int has_expand = cfg->expand > 1;
    const float stage_scale = (cfg->rq && cfg->rq->scale) ? cfg->rq->scale[0] : 1.0f;
    requantization_params_t rq_hidden = make_uniform_rq(stage_scale, cfg->rq ? cfg->rq->zero_point : 0, hidden_ch);

    /* 1) Expansion 1x1 (optional) */
    int8_t *after_expand = input;
    if (has_expand) {
        pointwise_conv1x1_int8(
            in_h, in_w,
            in_ch, hidden_ch,
            cfg->w_expand,
            input, buf0,
            &rq_hidden);
        relu6_apply_int8(buf0, hidden_ch, in_h, in_w, &rq_hidden);
        after_expand = buf0;
    }

    /* 2) Depthwise 3x3 stride {1,2} with SAME padding */
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
        rq_hidden);
    relu6_apply_int8(buf1, hidden_ch, dw_out_h, dw_out_w, &rq_hidden);

    /* 3) Pointwise projection 1x1 */
    pointwise_conv1x1_int8(
        dw_out_h, dw_out_w,
        hidden_ch, cfg->out_ch,
        cfg->w_pw,
        buf1, buf0,
        cfg->rq);
    relu6_apply_int8(buf0, cfg->out_ch, dw_out_h, dw_out_w, cfg->rq);

    /* 4) Residual */
    if (cfg->use_residual && cfg->stride == 1 && in_ch == cfg->out_ch) {
        residual_add(
            dw_out_h, dw_out_w,
            cfg->out_ch,
            input,
            buf0,
            buf0,
            *cfg->rq);
    }

    *out = buf0;
    *out_h = dw_out_h;
    *out_w = dw_out_w;
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
        /* round-to-nearest with symmetric offset */
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

    /* 0) Quantize input */
    quant_f32(
        BATCHES * ch * h * w,
        (float *)input_f32,
        buf0,
        qp_input);

    /* Stem: DW 3x3 s2 SAME (3->3) then PW 1x1 (3->32) */
    const size_t stem_h = (h + 2 - 3) / 2 + 1;
    const size_t stem_w = stem_h; /* input assumed square 224x224 */
    dwconv2D_3x3_int8(
        h, w,
        ch,
        /* stride */ 2,
        /* padding */ 1,
        (const void *)stem_0_0_wb_q,
        buf0,
        buf1,
        /* relu */ 0,
        make_uniform_rq(rq_stem.scale[0], rq_stem.zero_point, ch));
    relu6_apply_int8(buf1, ch, stem_h, stem_w, &rq_stem);

    pointwise_conv1x1_int8(
        stem_h, stem_w,
        ch, 32,
        stem_1_0_wb_q,
        buf1, buf0,
        &rq_stem);
    relu6_apply_int8(buf0, 32, stem_h, stem_w, &rq_stem);
    h = stem_h; w = stem_w; ch = 32;

    /* Inverted residual stack */
    static const block_desc_t blocks[] = {
        {1, 16, 1, 0, NULL, blocks_0_net_0_0_wb_q, blocks_0_net_1_0_wb_q, &rq_block0},
        {6, 24, 2, 0, blocks_1_net_0_0_wb_q, blocks_1_net_1_0_wb_q, blocks_1_net_2_0_wb_q, &rq_block1},
        {6, 24, 1, 1, blocks_2_net_0_0_wb_q, blocks_2_net_1_0_wb_q, blocks_2_net_2_0_wb_q, &rq_block2},
        {6, 32, 2, 0, blocks_3_net_0_0_wb_q, blocks_3_net_1_0_wb_q, blocks_3_net_2_0_wb_q, &rq_block3},
        {6, 32, 1, 1, blocks_4_net_0_0_wb_q, blocks_4_net_1_0_wb_q, blocks_4_net_2_0_wb_q, &rq_block4},
        {6, 32, 1, 1, blocks_5_net_0_0_wb_q, blocks_5_net_1_0_wb_q, blocks_5_net_2_0_wb_q, &rq_block5},
        {6, 64, 2, 0, blocks_6_net_0_0_wb_q, blocks_6_net_1_0_wb_q, blocks_6_net_2_0_wb_q, &rq_block6},
        {6, 64, 1, 1, blocks_7_net_0_0_wb_q, blocks_7_net_1_0_wb_q, blocks_7_net_2_0_wb_q, &rq_block7},
        {6, 64, 1, 1, blocks_8_net_0_0_wb_q, blocks_8_net_1_0_wb_q, blocks_8_net_2_0_wb_q, &rq_block8},
        {6, 64, 1, 1, blocks_9_net_0_0_wb_q, blocks_9_net_1_0_wb_q, blocks_9_net_2_0_wb_q, &rq_block9},
        {6, 96, 1, 0, blocks_10_net_0_0_wb_q, blocks_10_net_1_0_wb_q, blocks_10_net_2_0_wb_q, &rq_block10},
        {6, 96, 1, 1, blocks_11_net_0_0_wb_q, blocks_11_net_1_0_wb_q, blocks_11_net_2_0_wb_q, &rq_block11},
        {6, 96, 1, 1, blocks_12_net_0_0_wb_q, blocks_12_net_1_0_wb_q, blocks_12_net_2_0_wb_q, &rq_block12},
        {6, 160, 2, 0, blocks_13_net_0_0_wb_q, blocks_13_net_1_0_wb_q, blocks_13_net_2_0_wb_q, &rq_block13},
        {6, 160, 1, 1, blocks_14_net_0_0_wb_q, blocks_14_net_1_0_wb_q, blocks_14_net_2_0_wb_q, &rq_block14},
        {6, 160, 1, 1, blocks_15_net_0_0_wb_q, blocks_15_net_1_0_wb_q, blocks_15_net_2_0_wb_q, &rq_block15},
        {6, 320, 1, 0, blocks_16_net_0_0_wb_q, blocks_16_net_1_0_wb_q, blocks_16_net_2_0_wb_q, &rq_block16},
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

    /* Head: 1x1 320->1280 */
    pointwise_conv1x1_int8(
        h, w,
        ch, 1280,
        head_0_wb_q,
        cur, buf0,
        &rq_head);
    relu6_apply_int8(buf0, 1280, h, w, &rq_head);
    cur = buf0;
    ch = 1280;

    /* Global average pool 7x7 -> 1x1 */
    avgpool_global_7x7_int8(cur, ch, h, w, pooled);

    /* FC 1280 -> 10 */
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

    /* Dequantize logits for host-side consumption */
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
    mobilenet_forward(input_f32, logits_f32);
    return 0;
}


/*
 * Main function for secondary harts
 * 
 * Multi-threaded programs should provide their own implementation.
 */
void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
   asm volatile ("wfi");
  }
}
