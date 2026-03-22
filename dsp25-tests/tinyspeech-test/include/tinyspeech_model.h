#ifndef TINYSPEECH_MODEL_H
#define TINYSPEECH_MODEL_H

#include <stdint.h>
#include "tensor.h"

#define TINYSPEECH_NUM_CLASSES 6

Tensor tinyspeech_run_inference(Tensor *input);
int32_t tinyspeech_argmax(const Tensor *probs, float *max_prob);

#endif
