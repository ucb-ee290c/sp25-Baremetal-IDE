#ifndef MODULES_H
#define MODULES_H

#include "tensor.h"

Tensor upsample_nearest(Tensor* input, int8_t scale_factor);
void softmax(Tensor *input) ;
Tensor maxpool2d(Tensor* input, int kernel_size, int stride) ;
Tensor fc_layer(Tensor *input, Tensor *weights) ;
Tensor conv2d(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale, u_int8_t stride, u_int8_t padding);
Tensor conv2d_relu_maxpool2d(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale,
                             u_int8_t stride, u_int8_t padding,
                             int pool_kernel_size, int pool_stride);
Tensor conv2d_relu_gap(Tensor *input, Tensor *weights, Tensor *bias, Tensor *scale,
                       u_int8_t stride, u_int8_t padding);
Tensor batchnorm2d(Tensor* input, Tensor* gamma, Tensor* beta, Tensor* scale, Tensor* mean, Tensor* variance);
Tensor adaptive_avg_pool2d(Tensor *input);
void tinyspeech_prepack_conv_weights(const Tensor *conv1_w,
                                     const Tensor *conv2_w,
                                     const Tensor *conv3_w);

#endif
