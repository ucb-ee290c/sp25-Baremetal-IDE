#include <stdint.h>
#include <stddef.h>

void fully_connected_f32 (
    size_t input_size, 
    size_t output_size, 
    size_t batches, 
    float* input, 
    const float* weights_with_bias,
    float* output, 
    int relu
);

void softmax_vec(
    const float *i, 
    float *o, 
    size_t channels,
    size_t innerSize);