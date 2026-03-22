#ifndef TINYSPEECH_MODEL_H
#define TINYSPEECH_MODEL_H

#include <stdint.h>
#include "tensor.h"

#define TINYSPEECH_NUM_CLASSES 6
#define TINYSPEECH_MAX_DEBUG_STAGES 80

typedef struct {
    char name[40];
    int32_t size;
    float sum;
    float abs_sum;
    float min;
    float max;
} tinyspeech_stage_checksum_t;

typedef struct {
    int32_t num_stages;
    tinyspeech_stage_checksum_t stages[TINYSPEECH_MAX_DEBUG_STAGES];
    float logits[TINYSPEECH_NUM_CLASSES];
    int32_t logits_len;
} tinyspeech_debug_trace_t;

Tensor tinyspeech_run_inference(Tensor *input);
int32_t tinyspeech_argmax(const Tensor *probs, float *max_prob);
const tinyspeech_debug_trace_t *tinyspeech_debug_last_trace(void);

#endif
