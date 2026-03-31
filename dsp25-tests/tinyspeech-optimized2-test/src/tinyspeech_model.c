#include "tinyspeech_model.h"

#include "misc.h"
#include "modules.h"
#include "weights.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef TINYSPEECH_CONV_FUSE_RELU
#define TINYSPEECH_CONV_FUSE_RELU 0
#endif

#ifndef TINYSPEECH_FUSE_POOL
#define TINYSPEECH_FUSE_POOL 0
#endif

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

static Tensor make_float_input_copy(const Tensor *input) {
    Tensor out = f_create_tensor(input->shape, (int8_t)input->dims);
    for (int32_t i = 0; i < input->size; i++) {
        out.f_data[i] = tensor_get_value(input, i);
    }
    return out;
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

Tensor tinyspeech_run_inference(Tensor *input) {
    trace_reset();
    trace_add("input", input);

    Tensor input_f = make_float_input_copy(input);
#if TINYSPEECH_FUSE_POOL
    Tensor x = conv2d_relu_maxpool2d(&input_f, W(0), W(1), W(2), 1, 1, 2, 2);
    trace_add("conv1", &x);
    trace_add("relu1", &x);
    trace_add("pool1", &x);

    x = conv2d_relu_maxpool2d(&x, W(3), W(4), W(5), 1, 1, 2, 2);
    trace_add("conv2", &x);
    trace_add("relu2", &x);
    trace_add("pool2", &x);
#else
    Tensor x = conv2d(&input_f, W(0), W(1), W(2), 1, 1);
    trace_add("conv1", &x);
#if !TINYSPEECH_CONV_FUSE_RELU
    relu(&x);
#endif
    trace_add("relu1", &x);
    Tensor p1 = maxpool2d(&x, 2, 2);
    free_tensor(&x);
    x = p1;
    trace_add("pool1", &x);

    x = conv2d(&x, W(3), W(4), W(5), 1, 1);
    trace_add("conv2", &x);
#if !TINYSPEECH_CONV_FUSE_RELU
    relu(&x);
#endif
    trace_add("relu2", &x);
    Tensor p2 = maxpool2d(&x, 2, 2);
    free_tensor(&x);
    x = p2;
    trace_add("pool2", &x);
#endif

    x = conv2d(&x, W(6), W(7), W(8), 1, 1);
    trace_add("conv3", &x);
#if !TINYSPEECH_CONV_FUSE_RELU
    relu(&x);
#endif
    trace_add("relu3", &x);

    Tensor pooled = adaptive_avg_pool2d(&x);
    trace_add("gap", &pooled);
    free_tensor(&x);

    Tensor probs = fc_layer(&pooled, W(9));
    trace_store_logits(&probs);
    trace_add("fc_logits", &probs);
    free_tensor(&pooled);

    softmax(&probs);
    trace_add("softmax", &probs);

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
