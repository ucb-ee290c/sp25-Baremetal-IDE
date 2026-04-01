#ifndef TINYSPEECH_INT8_H
#define TINYSPEECH_INT8_H

#include "tinyspeech_model.h"

int tinyspeech_int8_prepare(const Tensor *conv1_w,
                            const Tensor *conv2_w,
                            const Tensor *conv3_w,
                            const Tensor *fc_w);
int tinyspeech_int8_is_ready(void);

Tensor tinyspeech_run_inference_int8(const Tensor *input,
                                     const Tensor *conv1_bias,
                                     const Tensor *conv2_bias,
                                     const Tensor *conv3_bias,
                                     tinyspeech_cycle_profile_t *profile);

#endif
