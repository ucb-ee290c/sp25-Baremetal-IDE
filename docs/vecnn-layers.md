# vec-nn Layer API

This README is the **high-level API reference** for vec-nn’s public layer interface
(`vec-nn/include/layers.h`).
It is intended for **chip bringup**: you should be able to wire up an application,
generate weights, and run the same layer calls on Spike / RTL / FPGA.

---

## 0) Conventions

### Tensor layout
- Convolution/pooling paths assume **CHW** memory layout:
  - input:  `[C][H][W]` contiguous
  - output: `[C][H_out][W_out]` contiguous

### Quantization types
- `quantization_params_t` (per-tensor)
  - `scale` (float), `zero_point` (int32)
- `requantization_params_t` (per-channel)
  - `scale` is a pointer to an array of scales (typically length = output channels)
  - `zero_point` is shared

### “relu” flags
Many compute ops include `int relu`:
- `0` → no activation
- nonzero → apply ReLU at the end of the op (operator-defined exact behavior)

### Padding encoding
For conv ops:
- `padding = 0` → VALID
- `padding = 1` → SAME
- `padding = 2` → FULL (NOT SUPPORTED YET)

---

## 1) Public API: layers.h

Include:
- `#include <layers.h>` (or `vec-nn/include/layers.h` depending on build)

### 1.1 Quantization helpers

**Quantize float32 → int8**
- `void quant_f32(size_t size, float* input, int8_t* output, quantization_params_t qp);`

**Dequantize int8 → float32**
- `void dequant_f32(size_t size, int8_t* input, float* output, quantization_params_t qp);`

Bringup notes:
- Use these for (a) input quantization, and (b) readable debug outputs (logits → softmax).

---

### 1.2 Transpose

**Transpose (int8)**
- `void transpose_int8(int8_t* input, int8_t* output, size_t rows, size_t cols);`

Bringup notes:
- Useful for weight transforms / layout experimentation and test scaffolding.
- Not used in the baseline CHW conv pipeline unless your exporter requires it.

---

### 1.3 Fully Connected (FC)

**Float FC (optional debug/reference path)**
- `void fully_connected_f32(size_t input_size, size_t output_size, size_t batches, float* input, const float* weights_with_bias, float* output, int relu);`

**Quantized FC (int8 path)**
- `void quant_fully_connected_int8(size_t input_size, size_t output_size, size_t batches, int8_t* input, const void* weights_with_bias, int8_t* output, int relu, int bias32, requantization_params_t requant_params);`

Bringup notes:
- `bias32` selects whether the bias term is stored/consumed as int32 (common in quant flows).
- `weights_with_bias` packing must match your header generator; keep this consistent across platforms.

---

### 1.4 Depthwise Convolution (2D)

**DW Conv 3×3 (int8)**
- `void dwconv2D_3x3_int8(size_t H, size_t W, size_t Cin, size_t stride, size_t padding, const void *dw_weights, int8_t *input, int8_t *output, int relu, requantization_params_t requant_params_dwconv);`

Weight format note from header:
- `dw_weights` length = `Cin * (1 + 9)`  (typically bias + 3×3 per channel)

Bringup notes:
- Input/output are CHW.
- Output shape depends on `stride` and `padding` (VALID vs SAME).
- SAME padding is required for MobileNet-like stacks; VALID is used in the MNIST baseline.

---

### 1.5 Pointwise Convolution (1×1)

**PW Conv 1×1 (int8)**
- `void conv_1x1_int8(size_t rows, size_t cols, size_t channels_in, size_t channels_out, size_t stride, size_t padding, int8_t* input, const void* weights, int8_t* output, int relu, requantization_params_t rqp);`

Bringup notes:
- PW conv is the main channel-mixing op in depthwise-separable networks.
- Per-channel `rqp.scale[c_out]` is typically used for output requantization.

---

### 1.6 Pooling

**MaxPool (int8)**
- `void maxpool_int8(size_t output_rows, size_t output_cols, size_t input_rows, size_t input_cols, size_t channels, size_t stride, int8_t *input, int8_t *output);`

**MaxPool (float32)**
- `void maxpool_f32(size_t output_rows, size_t output_cols, size_t input_rows, size_t input_cols, size_t channels, size_t stride, float *input, float *output);`

Bringup notes:
- Pooling is a good “sanity op” during bringup because it is simple and shape-heavy.
- Use it to validate CHW indexing and output shape computations early.

---

### 1.7 Softmax (float)

**Softmax**
- `void softmax_vec(const float *i, float *o, size_t channels, size_t innerSize);`

Bringup notes:
- Typically used only at the end (e.g., logits → probabilities).
- For int8 pipelines: dequantize logits first, then softmax for human-readable results.

---

### 1.8 Residual Add

**Residual add (int8)**
- `void residual_add(size_t rows, size_t cols, size_t channels, int8_t* a, int8_t* b, int8_t* output, requantization_params_t rqp);`

Bringup notes:
- Residual correctness depends on both inputs being in compatible quantization domains.
- If skip and main path scales differ, you must design for a common scale or add a requant step.

---

### 1.9 Activation: ReLU6

**ReLU6 (int8 output, float input)**
- `void relu6_int8(size_t channels, size_t inner_size, const float *input, int8_t *output, requantization_params_t requant_params);`

Bringup notes:
- Intended usage: dequantize int8 → float, clamp to [0, 6], requantize back to int8.
- This is common in MobileNetV2 stacks (ReLU6 everywhere).

---
