#ifndef TINYSPEECH_INT8_H
#define TINYSPEECH_INT8_H

#include "tinyspeech_model.h"

int tinyspeech_int8_prepare(const Tensor *conv1_w,
                            const Tensor *conv2_w,
                            const Tensor *conv3_w,
                            const Tensor *fc_w);
int tinyspeech_int8_is_ready(void);
void tinyspeech_int8_calib_reset(void);
int tinyspeech_int8_calib_finalize(const Tensor *conv1_bias,
                                   const Tensor *conv2_bias,
                                   const Tensor *conv3_bias);
int tinyspeech_int8_fixed_qparams_ready(void);

/* The full data-dependent calibration state is just the three per-conv activation maxima. Read them
 * after calibration to bake in; set them (then call calibration_end/finalize) to reuse a precomputed
 * calibration with no calibration inference pass. */
void tinyspeech_int8_calib_get_max(int32_t *m1, int32_t *m2, int32_t *m3);
void tinyspeech_int8_calib_set_max(int32_t m1, int32_t m2, int32_t m3);

Tensor tinyspeech_run_inference_int8(const Tensor *input,
                                     const Tensor *conv1_bias,
                                     const Tensor *conv2_bias,
                                     const Tensor *conv3_bias,
                                     tinyspeech_cycle_profile_t *profile);

#endif
