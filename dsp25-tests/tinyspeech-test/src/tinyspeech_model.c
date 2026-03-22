#include "tinyspeech_model.h"

#include "misc.h"
#include "modules.h"
#include "weights.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static inline Tensor *W(u_int8_t idx) {
    return model_weights[idx].address;
}

static tinyspeech_debug_trace_t g_last_trace;

static inline float tensor_get_value(const Tensor *t, int32_t idx) {
    if (t->f_data != NULL) {
        return t->f_data[idx];
    }
    return (float)t->data[idx];
}

static void trace_reset(void) {
    memset(&g_last_trace, 0, sizeof(g_last_trace));
}

static void trace_add(const char *name, const Tensor *t) {
    if (g_last_trace.num_stages >= TINYSPEECH_MAX_DEBUG_STAGES) {
        return;
    }

    tinyspeech_stage_checksum_t *stage = &g_last_trace.stages[g_last_trace.num_stages++];
    strncpy(stage->name, name, sizeof(stage->name) - 1);
    stage->name[sizeof(stage->name) - 1] = '\0';
    stage->size = t->size;

    if (t->size <= 0) {
        return;
    }

    float min_v = FLT_MAX;
    float max_v = -FLT_MAX;
    float sum_v = 0.0f;
    float abs_sum_v = 0.0f;

    for (int32_t i = 0; i < t->size; i++) {
        float v = tensor_get_value(t, i);
        if (v < min_v) {
            min_v = v;
        }
        if (v > max_v) {
            max_v = v;
        }
        sum_v += v;
        abs_sum_v += fabsf(v);
    }

    stage->sum = sum_v;
    stage->abs_sum = abs_sum_v;
    stage->min = min_v;
    stage->max = max_v;
}

static void trace_store_logits(const Tensor *t) {
    int32_t n = t->size;
    if (n > TINYSPEECH_NUM_CLASSES) {
        n = TINYSPEECH_NUM_CLASSES;
    }
    g_last_trace.logits_len = n;
    for (int32_t i = 0; i < n; i++) {
        g_last_trace.logits[i] = tensor_get_value(t, i);
    }
}

const tinyspeech_debug_trace_t *tinyspeech_debug_last_trace(void) {
    return &g_last_trace;
}

static Tensor attention_condenser(Tensor *input, u_int8_t *layer_id, int32_t block_idx, int32_t sub_idx) {
    u_int8_t base = *layer_id; /* scale tensor index */
    char label[40];

    Tensor q = maxpool2d(input, 2, 2);
    snprintf(label, sizeof(label), "b%ld.s%ld.pool", (long)block_idx, (long)sub_idx);
    trace_add(label, &q);

    Tensor k = conv2d(&q, W(base + 1), W(base + 2), W(base + 3), 1, 0);
    snprintf(label, sizeof(label), "b%ld.s%ld.gconv", (long)block_idx, (long)sub_idx);
    trace_add(label, &k);

    k = conv2d(&k, W(base + 5), W(base + 6), W(base + 7), 1, 0);
    snprintf(label, sizeof(label), "b%ld.s%ld.pwconv", (long)block_idx, (long)sub_idx);
    trace_add(label, &k);

    k = upsample_nearest(&k, 2);
    snprintf(label, sizeof(label), "b%ld.s%ld.upsample", (long)block_idx, (long)sub_idx);
    trace_add(label, &k);

    k = conv2d(&k, W(base + 9), W(base + 10), W(base + 11), 1, 0);
    snprintf(label, sizeof(label), "b%ld.s%ld.expand", (long)block_idx, (long)sub_idx);
    trace_add(label, &k);

    k = sigmoid(&k);
    snprintf(label, sizeof(label), "b%ld.s%ld.sigmoid", (long)block_idx, (long)sub_idx);
    trace_add(label, &k);

    attention(input, &k, W(base));
    snprintf(label, sizeof(label), "b%ld.s%ld.attn", (long)block_idx, (long)sub_idx);
    trace_add(label, &k);

    *layer_id = (u_int8_t)(base + 13); /* BN gamma index */
    return k;
}

static Tensor attn_bn_block(Tensor *input, u_int8_t *layer_id, int32_t block_idx) {
    char label[40];

    Tensor x = attention_condenser(input, layer_id, block_idx, 1);
    free_tensor(input);

    x = batchnorm2d(&x, W(*layer_id), W(*layer_id + 1), W(*layer_id + 3), W(*layer_id + 4), W(*layer_id + 5));
    snprintf(label, sizeof(label), "b%ld.s1.bn", (long)block_idx);
    trace_add(label, &x);

    *layer_id = (u_int8_t)(*layer_id + 6); /* next attention scale index */

    Tensor y = attention_condenser(&x, layer_id, block_idx, 2);
    free_tensor(&x);

    y = batchnorm2d(&y, W(*layer_id), W(*layer_id + 1), W(*layer_id + 3), W(*layer_id + 4), W(*layer_id + 5));
    snprintf(label, sizeof(label), "b%ld.s2.bn", (long)block_idx);
    trace_add(label, &y);

    *layer_id = (u_int8_t)(*layer_id + 6); /* next block scale index */

    return y;
}

Tensor tinyspeech_run_inference(Tensor *input) {
    trace_reset();
    trace_add("input", input);

    Tensor x = conv2d(input, W(0), W(1), W(2), 1, 1);
    trace_add("stem.conv3x3", &x);
    relu(&x);
    trace_add("stem.relu", &x);

    u_int8_t layer_id = 4; /* first block scale tensor */

    x = attn_bn_block(&x, &layer_id, 1);
    x = attn_bn_block(&x, &layer_id, 2);
    x = attn_bn_block(&x, &layer_id, 3);
    x = attn_bn_block(&x, &layer_id, 4);

    x = conv2d(&x, W(layer_id), W(layer_id + 1), W(layer_id + 2), 1, 1);
    trace_add("head.conv3x3", &x);
    relu(&x);
    trace_add("head.relu", &x);

    Tensor pooled = adaptive_avg_pool2d(&x);
    trace_add("head.gap", &pooled);
    free_tensor(&x);

    Tensor probs = fc_layer(&pooled, W(layer_id + 4));
    trace_store_logits(&probs);
    trace_add("head.fc_logits", &probs);
    free_tensor(&pooled);

    softmax(&probs);
    trace_add("head.softmax", &probs);
    return probs;
}

int32_t tinyspeech_argmax(const Tensor *probs, float *max_prob) {
    int32_t best_idx = 0;
    float best_val = probs->f_data[0];

    for (int32_t i = 1; i < probs->size; i++) {
        if (probs->f_data[i] > best_val) {
            best_val = probs->f_data[i];
            best_idx = i;
        }
    }

    if (max_prob != NULL) {
        *max_prob = best_val;
    }

    return best_idx;
}
