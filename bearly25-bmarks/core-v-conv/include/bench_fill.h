/*
 * bench_fill.h - Input generation and reference helpers for core-v-conv.
 */
#ifndef CORE_V_CONV_BENCH_FILL_H
#define CORE_V_CONV_BENCH_FILL_H

#include <stdint.h>
#include <stddef.h>

void bench_fill_int8_pattern(int8_t *data, int channels, int rows, int cols);
void bench_fill_float_pattern(void* data, int channels, int rows, int cols, int data_bytes); 
void bench_fill_int8_zero(int8_t *data, size_t size_bytes);

void bench_fill_conv_weights(void *weights, int channels);

void bench_ref_dwconv_i8(const int8_t *input,
                            int rows, int cols,
                            int channels,
                            int stride, int padding,
                            const void *weights,
                            const float *scale,
                            int32_t zero_point,
                            int8_t *output,
			    int kernel_size);

int bench_compare_i8(const int8_t *got,
                     const int8_t *ref,
                     int rows, int cols,
                     int channels,
                     int verbose);

#endif // CORE_V_CONV_BENCH_FILL_H
