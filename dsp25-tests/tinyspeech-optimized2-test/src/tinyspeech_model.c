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

#ifndef TINYSPEECH_OUTPUT_SOFTMAX
#define TINYSPEECH_OUTPUT_SOFTMAX 0
#endif

#ifndef TINYSPEECH_ENABLE_TRACE
#define TINYSPEECH_ENABLE_TRACE 0
#endif

static inline uint64_t rdcycle64_model(void) {
    uint64_t x;
    __asm__ volatile("rdcycle %0" : "=r"(x));
    return x;
}

static inline Tensor *W(u_int8_t idx) {
    return model_weights[idx].address;
}

static tinyspeech_debug_trace_t g_last_trace;
static tinyspeech_cycle_profile_t g_last_cycle_profile;

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
#if TINYSPEECH_ENABLE_TRACE
    memset(&g_last_trace, 0, sizeof(g_last_trace));
#else
    g_last_trace.num_stages = 0;
#endif
}

static void trace_add(const char *name, const Tensor *t) {
#if !TINYSPEECH_ENABLE_TRACE
    (void)name;
    (void)t;
    return;
#else
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
#endif
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

#if TINYSPEECH_ENABLE_TRACE
#define TRACE_ADD(name, tensor_ptr) trace_add((name), (tensor_ptr))
#else
#define TRACE_ADD(name, tensor_ptr) do { (void)(name); (void)(tensor_ptr); } while (0)
#endif

const tinyspeech_debug_trace_t *tinyspeech_debug_last_trace(void) {
    return &g_last_trace;
}

const tinyspeech_cycle_profile_t *tinyspeech_last_cycle_profile(void) {
    return &g_last_cycle_profile;
}

Tensor tinyspeech_run_inference(Tensor *input) {
    memset(&g_last_cycle_profile, 0, sizeof(g_last_cycle_profile));
    uint64_t t_total0 = rdcycle64_model();

    tinyspeech_tensor_arena_reset();
    trace_reset();
    TRACE_ADD("input", input);

    uint64_t t0 = rdcycle64_model();
    Tensor input_f = make_float_input_copy(input);
    uint64_t t1 = rdcycle64_model();
    g_last_cycle_profile.input_cast = t1 - t0;
#if TINYSPEECH_FUSE_POOL
    t0 = rdcycle64_model();
    Tensor x = conv2d_relu_maxpool2d(&input_f, W(0), W(1), W(2), 1, 1, 2, 2);
    t1 = rdcycle64_model();
    g_last_cycle_profile.conv1_pool1 = t1 - t0;
    TRACE_ADD("conv1", &x);
    TRACE_ADD("relu1", &x);
    TRACE_ADD("pool1", &x);

    t0 = rdcycle64_model();
    x = conv2d_relu_maxpool2d(&x, W(3), W(4), W(5), 1, 1, 2, 2);
    t1 = rdcycle64_model();
    g_last_cycle_profile.conv2_pool2 = t1 - t0;
    TRACE_ADD("conv2", &x);
    TRACE_ADD("relu2", &x);
    TRACE_ADD("pool2", &x);
#else
    t0 = rdcycle64_model();
    Tensor x = conv2d(&input_f, W(0), W(1), W(2), 1, 1);
    t1 = rdcycle64_model();
    g_last_cycle_profile.conv1_pool1 = t1 - t0;
    TRACE_ADD("conv1", &x);
#if !TINYSPEECH_CONV_FUSE_RELU
    relu(&x);
#endif
    TRACE_ADD("relu1", &x);
    t0 = rdcycle64_model();
    Tensor p1 = maxpool2d(&x, 2, 2);
    t1 = rdcycle64_model();
    g_last_cycle_profile.conv1_pool1 += (t1 - t0);
    free_tensor(&x);
    x = p1;
    TRACE_ADD("pool1", &x);

    t0 = rdcycle64_model();
    x = conv2d(&x, W(3), W(4), W(5), 1, 1);
    t1 = rdcycle64_model();
    g_last_cycle_profile.conv2_pool2 = t1 - t0;
    TRACE_ADD("conv2", &x);
#if !TINYSPEECH_CONV_FUSE_RELU
    relu(&x);
#endif
    TRACE_ADD("relu2", &x);
    t0 = rdcycle64_model();
    Tensor p2 = maxpool2d(&x, 2, 2);
    t1 = rdcycle64_model();
    g_last_cycle_profile.conv2_pool2 += (t1 - t0);
    free_tensor(&x);
    x = p2;
    TRACE_ADD("pool2", &x);
#endif

    t0 = rdcycle64_model();
    Tensor pooled = conv2d_relu_gap(&x, W(6), W(7), W(8), 1, 1);
    t1 = rdcycle64_model();
    g_last_cycle_profile.conv3_gap = t1 - t0;
    TRACE_ADD("conv3", &pooled);
    TRACE_ADD("relu3", &pooled);
    TRACE_ADD("gap", &pooled);

    t0 = rdcycle64_model();
    Tensor probs = fc_layer(&pooled, W(9));
    t1 = rdcycle64_model();
    g_last_cycle_profile.fc_logits = t1 - t0;
    trace_store_logits(&probs);
    TRACE_ADD("fc_logits", &probs);
    free_tensor(&pooled);

#if TINYSPEECH_OUTPUT_SOFTMAX
    t0 = rdcycle64_model();
    softmax(&probs);
    t1 = rdcycle64_model();
    g_last_cycle_profile.softmax = t1 - t0;
    TRACE_ADD("softmax", &probs);
#endif

    g_last_cycle_profile.total = rdcycle64_model() - t_total0;
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
