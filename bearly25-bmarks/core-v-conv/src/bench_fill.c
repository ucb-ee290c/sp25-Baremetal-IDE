/*
 * bench_fill.c - Input generation and reference math for core-v-conv.
 */
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"
#include "bench_config.h"

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline int32_t round_float_to_i32(float x) {
  return (int32_t)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

void bench_fill_int8_pattern(int8_t *data, int channels, int rows, int cols) {
  size_t plane = (size_t)rows * (size_t)cols;
  for (int ch = 0; ch < channels; ++ch) {
    int8_t *dst = data + ch * plane;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int v = (ch * 11 + i * 3 + j * 5 + 1) % 13;
        v -= 6;
        dst[i * cols + j] = (int8_t)v;
      }
    }
  }
}
//TODO: HARDCODED TO FP32
void bench_fill_float_pattern(void* data, int channels, int rows, int cols, int data_bytes) {
  float* matrix = (float*) data;
  size_t plane = (size_t) rows * (size_t) cols;
  for (int ch = 0; ch < channels; ++ch) {
    float* dst = matrix + ch*plane;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        // Pseudorandom
        // TODO improve randomness
        float v = (ch*12 / i + (j%3));
        dst[i*cols+j] = v;
      }
    }
  }
}

void bench_fill_int8_zero(int8_t *data, size_t size_bytes) {
  memset(data, 0, size_bytes);
}

void bench_fill_conv_weights(void *weights, int channels) {
  static const int8_t sobel_kernel[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1,
  };

  int32_t *bias = (int32_t *)weights;
  int8_t *kernels = (int8_t *)(bias + channels);
  for (int ch = 0; ch < channels; ++ch) {
    bias[ch] = 0;
    memcpy(kernels + ch * 9, sobel_kernel, sizeof(sobel_kernel));
  }
}

void bench_ref_dwconv_i8(const int8_t *input,
                            int rows, int cols,
                            int channels,
                            int stride, int padding,
                            const void *weights,
                            const float *scale,
                            int32_t zero_point,
                            int8_t *output,
			    int kernel_dim) {
  int out_rows = conv_out_dim(rows, CONV_KERNEL_SIZE, stride, padding);
  int out_cols = conv_out_dim(cols, CONV_KERNEL_SIZE, stride, padding);
  size_t in_plane = (size_t)rows * (size_t)cols;
  size_t out_plane = (size_t)out_rows * (size_t)out_cols;

  const int32_t *bias = (const int32_t *)weights;
  const int8_t *kernels = (const int8_t *)(bias + channels);

  const int32_t min_less_zero = -128 - zero_point;
  const int32_t max_less_zero = 127 - zero_point;

  for (int ch = 0; ch < channels; ++ch) {
    const int8_t *in_ch = input + ch * in_plane;
    int8_t *out_ch = output + ch * out_plane;
    const int8_t *kernel = kernels + ch * 9;

    for (int oy = 0; oy < out_rows; ++oy) {
      for (int ox = 0; ox < out_cols; ++ox) {
        int32_t acc = bias[ch];
        int in_y = oy * stride - padding;
        int in_x = ox * stride - padding;

        for (int ky = 0; ky < kernel_dim; ++ky) {
          int iy = in_y + ky;
          if (iy < 0 || iy >= rows) {
            continue;
          }
          for (int kx = 0; kx < kernel_dim; ++kx) {
            int ix = in_x + kx;
            if (ix < 0 || ix >= cols) {
              continue;
            }
            int8_t a = in_ch[iy * cols + ix];
            int8_t b = kernel[ky * kernel_dim + kx];
            acc += (int32_t)a * (int32_t)b;
          }
        }

        float scaled = (float)acc * scale[ch];
        int32_t q = round_float_to_i32(scaled);
        q = clamp_i32(q, min_less_zero, max_less_zero);
        q += zero_point;
        q = clamp_i32(q, -128, 127);
        out_ch[oy * out_cols + ox] = (int8_t)q;
      }
    }
  }
}

int bench_compare_i8(const int8_t *got,
                     const int8_t *ref,
                     int rows, int cols,
                     int channels,
                     int verbose) {
  int errors = 0;
  size_t plane = (size_t)rows * (size_t)cols;
  for (int ch = 0; ch < channels; ++ch) {
    const int8_t *got_ch = got + ch * plane;
    const int8_t *ref_ch = ref + ch * plane;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int8_t a = got_ch[i * cols + j];
        int8_t b = ref_ch[i * cols + j];
        if (a != b) {
          ++errors;
          if (verbose && errors < 16) {
            printf("  MISMATCH ch=%d (%d,%d): got=%d ref=%d\n",
                   ch, i, j, (int)a, (int)b);
          }
        }
      }
    }
  }

  if (verbose) {
    if (errors == 0) {
      printf("  OUTPUT MATCHES (C=%d, H=%d, W=%d)\n", channels, rows, cols);
    } else {
      printf("  OUTPUT MISMATCH: %d incorrect elements out of %d\n",
             errors, channels * rows * cols);
    }
  }

  return errors;
}
