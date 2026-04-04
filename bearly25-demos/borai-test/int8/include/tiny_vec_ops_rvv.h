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

/* Fused tiny attention microkernel for head_size=8 and pos<=2 (len<=3).
 * Inputs are layer-base kv pointers; returns 1 if handled, else 0. */
int borai_tiny_attn_h8_pos012_f32(
    float* xb,
    float* att,
    const float* q,
    const float* key_cache_layer,
    const float* value_cache_layer,
    int pos,
    int kv_dim,
    int kv_head_off,
    float inv_sqrt_head_size);

/* SwiGLU helper: hb[i] = silu(hb[i]) * hb2[i] over [start, end). */
void borai_swiglu_apply_range(float* hb, const float* hb2, int start, int end);

/* SwiGLU helper over [0, n). */
void borai_swiglu_apply(float* hb, const float* hb2, int n);

#endif
