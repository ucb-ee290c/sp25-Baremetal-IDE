#include <stddef.h>
#include <stdint.h>

#include <model_params_self.h>
#include <layers.h>
#include <data/inputs.h>
#include <data/model_params.h>

/* Placeholders for ops that are missing in vec-nn currently */
void conv3x3_int8_same_stride(
    size_t H, size_t W,
    size_t Cin, size_t Cout,
    size_t stride,
    const uint8_t *weights,
    int8_t *input,
    int8_t *output,
    const requantization_params_t rq);

void depthwise_conv3x3_int8_same(
    size_t H, size_t W,
    size_t C,
    size_t stride,
    const uint8_t *weights,
    int8_t *input,
    int8_t *output,
    const requantization_params_t rq,
    int relu6);

void relu6_int8(size_t size, int8_t *data);

void add_residual_int8(
    const int8_t *a,
    const int8_t *b,
    size_t size,
    int8_t *out,
    const requantization_params_t rq);

void avgpool_global_7x7_int8(
    const int8_t *input,
    size_t channels,
    size_t H, size_t W,
    int8_t *output);

/* -------------------------------------------------------------------------- */
/* Helper wrappers for the existing vec-nn kernels.                           */
/* -------------------------------------------------------------------------- */
static void pointwise_conv1x1_int8(
    size_t H, size_t W,
    size_t Cin, size_t Cout,
    const uint8_t *weights,
    int8_t *input,
    int8_t *output,
    const requantization_params_t *rq)
{
    /* Uses existing 1x1 kernel; ReLU6 is applied separately. */
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

/* -------------------------------------------------------------------------- */
/* Inverted residual block runner                                             */
/* -------------------------------------------------------------------------- */
typedef struct {
    size_t expand;       /* t */
    size_t out_ch;       /* c_out */
    size_t stride;       /* s */
    int use_residual;    /* when stride==1 && Cin==Cout */
    const uint8_t *w_expand;
    const uint8_t *w_dw;
    const uint8_t *w_pw;
    const requantization_params_t *rq;
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

    /* 1) Expansion 1x1 (optional) */
    int8_t *after_expand = input;
    if (has_expand) {
        pointwise_conv1x1_int8(
            in_h, in_w,
            in_ch, hidden_ch,
            cfg->w_expand,
            input, buf0,
            cfg->rq);
        relu6_int8(hidden_ch * in_h * in_w, buf0);
        after_expand = buf0;
    }

    /* 2) Depthwise 3x3 stride {1,2} with SAME padding */
    const size_t dw_out_h = (in_h + cfg->stride - 1) / cfg->stride;
    const size_t dw_out_w = (in_w + cfg->stride - 1) / cfg->stride;
    depthwise_conv3x3_int8_same(
        in_h, in_w,
        hidden_ch,
        cfg->stride,
        cfg->w_dw,
        after_expand,
        buf1,
        *cfg->rq,
        /* relu6 */ 1);

    /* 3) Pointwise projection 1x1 */
    pointwise_conv1x1_int8(
        dw_out_h, dw_out_w,
        hidden_ch, cfg->out_ch,
        cfg->w_pw,
        buf1, buf0,
        cfg->rq);
    relu6_int8(cfg->out_ch * dw_out_h * dw_out_w, buf0);

    /* 4) Residual */
    if (cfg->use_residual && cfg->stride == 1 && in_ch == cfg->out_ch) {
        add_residual_int8(
            input,
            buf0,
            cfg->out_ch * dw_out_h * dw_out_w,
            buf0,
            *cfg->rq);
    }

    *out = buf0;
    *out_h = dw_out_h;
    *out_w = dw_out_w;
    *out_ch = cfg->out_ch;
}

/* -------------------------------------------------------------------------- */
/* Forward pass                                                               */
/* -------------------------------------------------------------------------- */
#define MAX_INT8_BUFFER (96 * 112 * 112) /* largest feature map (block1 expand) */

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

    /* Stem: 3x3 stride2 SAME, 3->32 */
    conv3x3_int8_same_stride(
        h, w,
        ch, 32,
        2,
        stem_0_wb_q,
        buf0, buf1,
        rq_stem);
    relu6_int8(32 * 112 * 112, buf1);
    h = 112; w = 112; ch = 32;

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

    int8_t *cur = buf1;
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
    relu6_int8(1280 * h * w, buf0);
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
