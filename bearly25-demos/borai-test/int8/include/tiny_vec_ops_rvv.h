#ifndef BORAI_TEST_TINY_VEC_OPS_RVV_H
#define BORAI_TEST_TINY_VEC_OPS_RVV_H

#include <stdint.h>

/* Returns 1 if handled by a specialized tiny-shape kernel, 0 to fall back. */
int borai_tiny_rmsnorm_f32(float* o, const float* x, const float* weight, int n);

/* Returns 1 if handled by a specialized tiny-shape kernel, 0 to fall back. */
int borai_tiny_quantize_i8(float* out_scale, int8_t* out_q, const float* x, int n);

/* Returns 1 if handled by specialized head_size=8 kernel, 0 to fall back. */
int borai_tiny_dot_qk_head_f32(float* out, const float* q, const float* k, int n);

/* Returns 1 if handled by specialized head_size=8 kernel, 0 to fall back. */
int borai_tiny_axpy_head_f32(float* dst, const float* v, float a, int n);

#endif
