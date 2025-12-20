# vec-nn

This doc is written for Bearly and DSP SP25 chip bringup: how to build, run, and debug the vectorized neural-network layer library across Spike, RTL sim, and FPGA, and how to structure apps (MNIST CNN + MobileNetV2 WIP).

### Repository Integration Status
As of the current port, the vector NN code is in under `vec-nn` directory in this repo. The vector code has been integrated into CMake build flow. Example targets libe under `bearly25-tests`.

## Getting Started


1. Build an application target: Target are apps that live under `bearly25-<tests/demos/bmarks>/<target>`. To build your application run `make build CHIP=bearly25 TARGET=<target>`.
Common Flags:
- VECNN=1 → include vec-nn library + headers
- RVV=1 → compile with vector ISA enabled
- CHIP=bearly25 → selects the chip/SoC config
- TARGET=<name> → selects the app

Example: 
```bash
make build CHIP=bearly25 TARGET=mnist_cnn_quant_conv2D VECNN=1 RVV=1
```


2. Run on your backend. There are some options that we can use within Chipyard/Baremetal-IDE.

### Spike
This is probably the easiest one. Spike is a functional simulator and it serves as riscv golden model. It is NOT cycle accurate, but it provides ISA level correctness, and it also runs significantly faster
than Saturn on RTL simulation. You can't run accelerator code (eg. ROCC instructions in spike).

Example of Spike after generating your binary: 
```
make build CHIP=bearly25 TARGET=mnist_cnn_quant_conv2D VECNN=1 RVV=1
```

### RTL simulation (SoC-level)
This is where you run cycle accurate simulation of the hardware. If you want to simulate accelerators, you will have to do so here.
Workflow
	1.	Build the same ELF with VECNN=1 RVV=1.
	2.	Generate/load it into your Chipyard/SoC sim flow (verilator/vcs).
	3.	You will need to build a chipyard config for saturn + respective accelerators. I will add a README on how to do this too. Once you have this, go into `CY_ROOT/sims/vcs/` and run `make run-binary CONFIG=<config> LOADMEM=1 BINARY=<path/to/your/generated/elf>`
	4.	Keep in mind that this will use Dram sim 3 to model DRAM, which is significantly faster than our bringup DRAM setup with serial tilelink. If you want to model this accurately in rtl sims too, you will have to instantiate a bringup config in your desired config. 

### FPGA
This is probably the most tedious one as you need to find an FPGA, the shell, and the bitstream. To learn how to do this, watch Ethan's walkthrough. Once that is set up you can upload your generated binary to your FPGA. 
One thing to keep in mind is that if your FPGA bit

### Chip
Directly through Baremetal-IDE generated binary via JTAG or UART-TSI. Find more information for this and for FPGA on [this lab](https://www.google.com/url?q=https://ucb-ee290c.github.io/tutorials/baremetal-ide/Baremetal-IDE-Lab.html&sa=D&source=editors&ust=1766236087680192&usg=AOvVaw1LrLfzfykjtewlf5rH_kCq).

### Summary
| Backend  | What it proves                                             | What it misses                                              | Best use                                  |
|----------|------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------|
| Spike    | ISA-level correctness, RVV codegen sanity                   | Cycle accuracy, real memory system, some device behavior    | Fast iteration + regressions               |
| RTL sim  | Full SoC integration, MMIO/driver correctness, waveform debug | Speed, full-system throughput realism                       | “Will it run on the chip?”                 |
| FPGA    | Realistic timing, bandwidth, long-run stability              | Limited observability, slower iteration                     | Performance + end-to-end demos             |stream was built as a part of the chipyard configs, it won't be able to run binaries that you generate in baremetal-ide unless the linker script, and address map match the desired config.


## High-level Overview
vec-nn is intended to be a lightweight C layer library intended for bare-metal inference on a RISC-V SoC with RVV 1.0 (currently targeting the Saturn Vector Unit inside Chipyard/Bearly25 flows). 
The library provides:
-	Quantization helpers (quant_f32, dequant_f32) with per-tensor scale + zero-point
-	Core kernels (int8-first):
-	dwconv2D_3x3_int8 (depthwise 3×3, CHW layout)
- dwconv2D_5x5_int8 (depthwise 5×5, CHW layout)
-	conv_1x1_int8 (pointwise 1×1)
-	quant_fully_connected_int8 (optionally ReLU and int32 bias path)
-	pooling (maxpool_int8), avgpool, and glue ops (residual_add)
-	relu6_int8 via float-domain clamp + requant
-	Float utilities where useful (fully_connected_f32, maxpool_f32, softmax_vec)


At a high level, vec-nn provides:

### 2.1 Data model
- Explicit tensor shapes and layouts
- **BCHW layout** for all convolutional paths
- **BHWC layout** for linear layers (this is unconvecional, but it makes it easier to write)

### 2.2 Quantization support
- Per-tensor quantization (`scale`, `zero_point`)
- Per-channel requantization for convolution / FC outputs
- Bias handling (int32 where required)
- Clear separation between:
  - **quantize**
  - **compute**
  - **requantize**

### 2.3 Core operators (int8-first)
Conceptually, vec-nn supports the building blocks required for modern CNNs:

- Depthwise convolution (3×3, 5×5)
- Pointwise convolution (1×1)
- Fully connected layers
- Pooling (max / average)
- Elementwise ops (residual add)
- Activations (ReLU, ReLU6)
- Softmax (float domain, for inspection)

Float operators exist primarily for:
- reference
- debug
- final output interpretation

## 3. Example models in this repo

Two models serve as **reference workloads** for bringup:

## 3.1 MNIST CNN (baseline bringup model)

### Purpose
The MNIST CNN is the **minimal realistic CNN** used to:

- Validate the vec-nn API
- Test quantization flows
- Exercise depthwise + pointwise convolutions
- Verify FC layers and softmax output
- Provide fast iteration during early bringup

### Characteristics
- Small input (28×28)
- Shallow network
- Deterministic memory usage
- Easy to debug layer-by-layer

### Why it matters
MNIST is:
- small enough to run everywhere (Spike → FPGA)
- complex enough to catch layout, scaling, and integration bugs
  

## 4.2 MobileNetV2 (system-stress model, WIP)

### Purpose
MobileNetV2 is used as a **scaling and stress test** for vec-nn and the platform.

It exercises:
- Larger feature maps
- Many depthwise + pointwise layers
- Inverted residual blocks
- ReLU6 activations
- Residual connections
- Global average pooling
- Large channel counts (up to 1280)

### Characteristics
- Significantly higher memory pressure
- Many more requantization boundaries
- Sensitive to scaling consistency
- Exposes performance bottlenecks clearly


## 5. How applications are structured (high level)

All vec-nn applications follow the same pattern:

1. **Quantize input**
2. **Run a sequence of layers**
3. **Optionally dequantize outputs**
4. **Inspect or print results**
5. **Measure cycles / instructions**

There is:
- no runtime graph
- no scheduler
- no hidden state

Everything is simple, explicit and visible, which is good for bringup.

## 6. What comes next

This document covers the **C / bringup side**.

A separate document will cover:
- PyTorch models + Quantization strategy + Header generation (`model_params*.h`)
- Weight / bias layout contracts
- Versioning and consistency checks

Together, these form a **PyTorch → C → RVV → chip** flow.

---
  
