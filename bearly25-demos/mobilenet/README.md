# vec-nn / nn-rvv: C Layer API + PyTorch export

This README is a **bringup-oriented** reference for:
1) the **C layer API** (what apps should call),
2) the **operator implementation surface** under `vec-nn/src/ops/` (what exists conceptually),
3) the **PyTorch → C header export flow** (how to generate `*_wb_q` + quant params for MNIST/MobileNet-style models).

---

## 0) Where things live in the repo

- Public layer API:
  - `vec-nn/include/layers.h`

- Operator implementations:
  - `vec-nn/src/ops/`

- Example applications (bringup workloads):
  - `bearly25-tests/<target>/...`
  - (sometimes also `bearly25-demos/`, `bearly25-bmarks/`)

- Typical generated model headers:
  - `bearly25-tests/<target>/include/data/model_params*.h`
  - `bearly25-tests/<target>/include/data/inputs*.h`

---

## 1) Conventions

### 1.1 Tensor layout
All conv/pool paths are **CHW**:
- input  = `[C][H][W]` contiguous
- output = `[C][H_out][W_out]` contiguous

If you export HWC data from PyTorch and feed it directly, you will get “random accuracy collapse”
without crashing. Fix layout first.

### 1.2 Quantization types
- `quantization_params_t` = per-tensor
  - `scale` (float), `zero_point` (int32)
- `requantization_params_t` = per-channel
  - `scale` = pointer to array of floats (usually length = output channels)
  - `zero_point` (int32)

### 1.3 Padding encoding
- `padding = 0` → VALID
- `padding = 1` → SAME
- `padding = 2` → FULL (NOT SUPPORTED)

---

## 2) C Layer API (layers.h) — app-facing reference

Include:
- `#include <layers.h>`

### 2.1 Quantization helpers

**quant_f32**
- `quant_f32(size, input_f32, output_i8, qp)`
- Purpose: input quantization + debug scaffolding.
- Typical usage: quantize raw input images into int8 activations.

**dequant_f32**
- `dequant_f32(size, input_i8, output_f32, qp)`
- Purpose: convert logits to float for inspection (softmax / printing).

---

### 2.2 Transpose

**transpose_int8**
- `transpose_int8(input, output, rows, cols)`
- Purpose: small layout transforms (typically for weights/debug). Not required for the basic CHW path.

---

### 2.3 Fully Connected layers

**fully_connected_f32**
- `fully_connected_f32(input_size, output_size, batches, input_f32, weights_with_bias_f32, output_f32, relu)`
- Purpose: float reference / debug path.

**quant_fully_connected_int8**
- `quant_fully_connected_int8(input_size, output_size, batches, input_i8, weights_with_bias, output_i8, relu, bias32, rqp)`
- Notes:
  - `bias32` selects int32 bias consumption (common in quant pipelines).
  - `weights_with_bias` packing must match the PyTorch exporter (see Section 4).

---

### 2.4 Depthwise convolution (2D)

**dwconv2D_3x3_int8**
- `dwconv2D_3x3_int8(H, W, Cin, stride, padding, dw_weights, input_chw, output_chw, relu, rqp_dw)`
- Weight format: `dw_weights` length = `Cin * (1 + 9)` (bias + 3×3 per channel)

Bringup note:
- SAME padding is critical for MobileNet-like stacks; MNIST baseline often uses VALID.

(If your branch also provides 5×5 DW conv, it will follow the same conventions.)

---

### 2.5 Pointwise convolution (1×1)

**conv_1x1_int8**
- `conv_1x1_int8(rows, cols, Cin, Cout, stride, padding, input, weights, output, relu, rqp_pw)`
- Purpose: channel mixing (main “compute” in depthwise-separable networks).

---

### 2.6 Pooling

**maxpool_int8**
- `maxpool_int8(out_rows, out_cols, in_rows, in_cols, channels, stride, input, output)`

**maxpool_f32**
- `maxpool_f32(out_rows, out_cols, in_rows, in_cols, channels, stride, input, output)`

Bringup note:
- Pooling is a good early bringup sanity check because it is simple but shape-sensitive.

---

### 2.7 Softmax (float)

**softmax_vec**
- `softmax_vec(i_f32, o_f32, channels, innerSize)`
- Typical usage: dequantize logits → softmax probabilities for printing.

---

### 2.8 Residual add (int8)

**residual_add**
- `residual_add(rows, cols, channels, a_i8, b_i8, output_i8, rqp)`
- Bringup note:
  - Both inputs must be in compatible quantization domains (or you must enforce a shared scale by design).

---

### 2.9 ReLU6 path (float-in → int8-out)

**relu6_int8**
- `relu6_int8(channels, inner_size, input_f32, output_i8, rqp)`
- Intended usage in MobileNet-style stacks:
  - dequant int8 → float
  - clamp to [0, 6]
  - requantize back to int8 (often reusing the same rqp)


## 3) PyTorch → C export (header generation) — bringup “small doc”

### 3.1 What the exporter must produce

For each layer in your model, the C side needs:
1) **weights (+ bias)** in the packing expected by the C kernel
2) quantization parameters:
   - input `quantization_params_t` (per-tensor)
   - per-layer output `requantization_params_t` (per-channel scale array + zero_point)
3) optionally output `quantization_params_t` for final logits (so you can dequantize)

For bringup, keep this mental model:
- PyTorch holds tensors in float32.
- Exporter quantizes weights (and optionally activations calibration gives scales).
- C code consumes:
  - `*_wb_q` arrays (quantized weights + bias)
  - `rq_*` structures (requantization params per op)
  - `qp_input`, `qp_logits`, etc.

### 3.2 Header structure (recommended)

A practical generated header typically defines:
- `#define` shape constants (H/W/C etc.)
- weight arrays:
  - `static const uint8_t <layer>_wb_q[...] = {...};`
- per-channel scales:
  - `static float <layer>_scale[...] = {...};`
- requant params objects:
  - `static const requantization_params_t rq_<layer> = { .scale = <layer>_scale, .zero_point = <zp> };`
- global qp objects:
  - `static const quantization_params_t qp_input = {...};`
  - `static const quantization_params_t qp_logits = {...};`

This layout keeps the C app clean:
- the app only includes `model_params*.h`
- and passes pointers to `layers.h` calls.

### 3.3 Weight packing rules (high level)

Your exporter must match what `layers.h` expects:

**Depthwise 3×3**
- `dw_weights` length = `Cin * (1 + 9)`
- Packing convention (recommended):
  - for each input channel c:
    - 1 bias value (if bias value is 32 bits, then it needs to get packed as 4 uint8 )
    - 9 kernel values (row-major 3×3)

**Pointwise 1×1**
- weights represent a Cout×Cin matrix plus bias.
- Packing is the same as depthwise conv above.
  - Common choice: for each output channel:
    - bias (if bias value is 32 bits, then it needs to get packed as 4 uint8 )
    - Cin weights

**FC**
- weights represent output_size×input_size plus bias.
- If using `bias32=1`, bias should be stored as 4 uint8.

---

## 4) PyTorch front-end workflow (MNIST + MobileNet)

This section is intentionally high-level; the exact exporter script name can vary by branch.

### 4.1 MNIST CNN (baseline)

1) Train float32 model in PyTorch
2) Quantize (symmetric or asymmetric; vec-nn supports both)
3) Export:
   - quantized weights + bias arrays
   - `qp_input`, `qp_logits`
   - `rq_*` for each conv and fc layer
4) Copy generated headers into:
   - `bearly25-tests/mnist_*/include/data/model_params*.h`
5) Build/run in (worth trying Spike for verification).

---
