#!/usr/bin/env python3
import argparse, os, sys
import numpy as np
import torch

"""
Data generator for:

void dwconv2D_3x3_int8 (
    size_t H, size_t W,
    size_t Cin,
    size_t stride,
    size_t padding,
    const void *dw_weights,  // length = Cin*(4 + 9)
                             // layout (int8):
                             //   first 4*Cin bytes: bias[c] as int32 per channel
                             //                     (b0_c0, b1_c0, b2_c0, b3_c0,
                             //                      b0_c1, ..., b3_c{Cin-1})
                             //   next Cin*9 bytes: weights [Cin][1][3][3] flattened
    int8_t *input,           // CHW: [Cin][H][W]
    int8_t *output,          // CHW: [Cin][H_out][W_out]  (depthwise)
    int relu,
    requantization_params_t requant_params_dwconv
)

Quantization for tests:
    - Per-channel scale[c] chosen to avoid saturation:
        scale[c] = 120.0 / max_abs(acc[c])   (if max_abs > 0, else 1.0)
    - zero_point = 0
    - Quantization: round(acc * scale[c]) then clamp to int8.
"""

# ---------- Reference depthwise conv via PyTorch (int32 accumulator) ----------

def depthwise_conv_int32(
    input_int8: torch.Tensor,   # [Cin, H, W], int8
    weight_int8: torch.Tensor,  # [Cin, 1, 3, 3], int8
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    Run depthwise conv2d and return int32 accumulator:
        output shape: [Cin, H_out, W_out]
    """
    x = input_int8.float().unsqueeze(0)  # [1, Cin, H, W]
    w = weight_int8.float()              # [Cin, 1, 3, 3]

    y = torch.nn.functional.conv2d(
        x,
        w,
        bias=None,
        stride=stride,
        padding=padding,
        groups=input_int8.shape[0],
    )  # [1, Cin, H_out, W_out]

    return y.to(torch.int32).squeeze(0)  # [Cin, H_out, W_out]


def requantize_int32_per_channel(
    acc_int32: torch.Tensor,  # [Cin, H_out, W_out]
    scales: torch.Tensor,     # [Cin] float32
    relu: bool = False,
    relu6: bool = False,
) -> torch.Tensor:
    """
    Quantize int32 accumulator to int8 using per-channel scales:

        q[c, i, j] = round(acc[c, i, j] * scales[c])

    then clamp:
        - no ReLU:   [-128, 127]
        - ReLU:      [0, 127]
        - ReLU6:     [0, floor(6 / scale[c])] (still clamped to int8 range)
    """
    Cin = acc_int32.shape[0]
    acc_f = acc_int32.float()
    s = scales.view(Cin, 1, 1)  # broadcast to spatial dims

    y = torch.round(acc_f * s)
    y = torch.clamp(y, -128, 127)

    if relu6:
        # q6[c] = floor(6 / scale[c]) per channel, capped at int8 max
        q6 = torch.floor(6.0 / scales).clamp(max=127).view(Cin, 1, 1)
        y = torch.clamp(y, min=0)
        y = torch.minimum(y, q6)
    elif relu:
        y = torch.clamp(y, min=0, max=127)

    return y.to(torch.int8)


# ---------- Pack weights to match dw_weights layout ----------

def make_dw_weights(weight_int8: torch.Tensor) -> np.ndarray:
    """
    Pack weights into dw_weights buffer (length = Cin*(4+9)):

    Layout (int8):
        [b0_c0, b1_c0, b2_c0, b3_c0,
         b0_c1, b1_c1, b2_c1, b3_c1,
         ...
         w[0,0,0], ..., w[0,0,8],
         w[1,0,0], ..., w[Cin-1,0,8]]

    - weight_int8 is [Cin, 1, 3, 3].
    - bias[c] is int32, but for testing we emit all-zero bytes.
    """
    Cin = weight_int8.shape[0]

    # Bias per channel: int32 -> 4 int8 bytes; here all zeros for tests.
    bias_bytes = np.zeros((Cin * 4,), dtype=np.int8)   # [4*Cin]

    # Flatten weights channel-major
    w = weight_int8.numpy().reshape(Cin * 9)           # [Cin*9]

    packed = np.concatenate([bias_bytes, w])           # [4*Cin + Cin*9] = Cin*(4+9)
    return packed


# ---------- C header helpers ----------

def write_c_array(f, c_type: str, name: str, data: np.ndarray, per_line: int = 16):
    """Flatten `data` and write as a static C array."""
    flat = data.reshape(-1)
    if c_type == "float":
        fmt = "{:.8f}f"   # e.g., 1.00000000f
    else:
        fmt = "{}"

    f.write(f"static const {c_type} {name}[{flat.size}] = {{\n")
    indent = " " * 4
    for i, val in enumerate(flat):
        if i % per_line == 0:
            f.write(indent)
        s = fmt.format(float(val)) if c_type == "float" else fmt.format(int(val))
        f.write(s)
        if i != flat.size - 1:
            f.write(", ")
        if (i + 1) % per_line == 0:
            f.write("\n")
    if flat.size % per_line != 0:
        f.write("\n")
    f.write("};\n\n")


# ---------- Test-case generation ----------

def generate_case(name, H, W, Cin, stride, padding, seed, out_dir):
    # Allow padding 0 or 1; stride still 1 or 2
    if padding not in (0, 1) or stride not in (1, 2):
        raise ValueError("Only padding=0 or 1 with stride=1 or 2 are supported in this generator.")

    torch.manual_seed(seed)

    # Random int8 data (you can tweak ranges if you want)
    input_int8 = torch.randint(-64, 64, (Cin, H, W), dtype=torch.int8)
    weight_int8 = torch.randint(-8, 8, (Cin, 1, 3, 3), dtype=torch.int8)

    # Int32 accumulator from depthwise conv
    acc_int32 = depthwise_conv_int32(
        input_int8, weight_int8, stride=stride, padding=padding
    )  # [Cin, H_out, W_out]

    Cin_out, H_out, W_out = acc_int32.shape
    assert Cin_out == Cin, f"Cin mismatch: expected {Cin}, got {Cin_out}"

    # ---- Sanity-print pre-quant accumulator ----
    acc_np = acc_int32.numpy()
    print(f"\n=== Pre-quant accumulator for '{name}' ===")
    print(f"  shape: {acc_np.shape}  (Cin={Cin}, H_out={H_out}, W_out={W_out})")
    print(f"  min: {acc_np.min()}, max: {acc_np.max()}")

    # If it's small, dump per-channel matrices for extra sanity
    if acc_np.size <= 128:
        for c in range(Cin):
            print(f"  Channel {c}:")
            for r in range(H_out):
                row = acc_np[c, r, :]
                print("    ", " ".join(f"{v:6d}" for v in row))
    print("=== End pre-quant accumulator ===\n")
    # ---- End sanity-print ----

    # Per-channel scales: choose to keep values mostly in range
    acc_abs = acc_int32.abs().view(Cin, -1)
    max_per_ch = acc_abs.max(dim=1).values  # [Cin]
    scales = torch.ones(Cin, dtype=torch.float32)
    nonzero = max_per_ch > 0
    scales[nonzero] = 120.0 / max_per_ch[nonzero].float()

    zero_point = 0  # still 0 for all channels

    # Quantized outputs
    out_no_relu = requantize_int32_per_channel(acc_int32, scales, relu=False)
    out_relu    = requantize_int32_per_channel(acc_int32, scales, relu=True)
    out_relu6   = requantize_int32_per_channel(acc_int32, scales, relu6=True)

    os.makedirs(out_dir, exist_ok=True)

    # Pack weights to your requested layout
    dw_weights = make_dw_weights(weight_int8)

    # Write header with all data for this case
    upper = name.upper()
    header_path = os.path.join(out_dir, f"dwconv_{name}.h")
    with open(header_path, "w") as f:
        f.write("/* Auto-generated depthwise conv2D 3x3 int8 test case. */\n")
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")

        # Macros for dims + params
        f.write(f"#define DWCONV_{upper}_H {H}\n")
        f.write(f"#define DWCONV_{upper}_W {W}\n")
        f.write(f"#define DWCONV_{upper}_CIN {Cin}\n")
        f.write(f"#define DWCONV_{upper}_STRIDE {stride}\n")
        f.write(f"#define DWCONV_{upper}_PADDING {padding}\n")
        f.write(f"#define DWCONV_{upper}_H_OUT {H_out}\n")
        f.write(f"#define DWCONV_{upper}_W_OUT {W_out}\n\n")
        f.write(f"#define DWCONV_{upper}_HAS_RELU6 1\n\n")

        # Arrays:
        #   input:       [Cin][H][W]          -> flattened CHW
        #   weights:     bias (4 bytes per channel), then [Cin][1][3][3] flattened
        #   ref outputs: [Cin][H_out][W_out]
        #   scales:      [Cin] (per-channel)
        write_c_array(f, "int8_t",  f"dwconv_{name}_input",      input_int8.numpy())
        write_c_array(f, "int8_t",  f"dwconv_{name}_weights",    dw_weights.astype(np.int8))
        write_c_array(f, "int8_t",  f"dwconv_{name}_ref_norelu", out_no_relu.numpy())
        write_c_array(f, "int8_t",  f"dwconv_{name}_ref_relu",   out_relu.numpy())
        write_c_array(f, "int8_t",  f"dwconv_{name}_ref_relu6",  out_relu6.numpy())
        write_c_array(f, "float",   f"dwconv_{name}_scales",     scales.numpy())

    print(f"Generated test case '{name}':")
    print(f"  H={H}, W={W}, Cin={Cin}, stride={stride}, padding={padding}")
    print(f"  H_out={H_out}, W_out={W_out}")
    print(f"  Header: {header_path}")
    print(f"  Note: first {4*Cin} elements of dwconv_{name}_weights are per-channel int32 biases (all 0 bytes).")
    return header_path


# ---------- Generic npy -> C header mode (kept, but doesn't generate data) ----------

def npy_to_c_header(npy_path, var_name, c_type, out_path):
    arr = np.load(npy_path)
    out = sys.stdout if out_path in ("-", None) else open(out_path, "w")
    try:
        out.write("/* Auto-generated from npy_to_c_header */\n")
        out.write("#pragma once\n\n")
        if c_type.startswith("int") or c_type.startswith("uint"):
            out.write("#include <stdint.h>\n\n")
        write_c_array(out, c_type, var_name, arr)
    finally:
        if out is not sys.stdout:
            out.close()


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Generate int8 depthwise conv2d test data and/or convert .npy to C headers."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # gen: create one test case with CLI dims (ONLY header, no npy files)
    pg = subparsers.add_parser("gen", help="Generate random test case and C header.")
    pg.add_argument("--name",    required=True, help="Name/tag for the test case (e.g., p1_s1_small).")
    pg.add_argument("--H",       type=int, required=True, help="Input height.")
    pg.add_argument("--W",       type=int, required=True, help="Input width.")
    pg.add_argument("--Cin",     type=int, required=True, help="Number of input channels.")
    pg.add_argument("--stride",  type=int, choices=[1, 2], required=True, help="Stride (1 or 2).")
    pg.add_argument("--padding", type=int, choices=[0, 1], required=True, help="Padding (0 or 1).")
    pg.add_argument("--seed",    type=int, default=0, help="Random seed for reproducibility.")
    pg.add_argument("--out-dir", default=".", help="Output directory for .h")

    # npy2c: convert any .npy to a single C array header (no new npy files)
    pn = subparsers.add_parser("npy2c", help="Convert a .npy file to a C header with a single array.")
    pn.add_argument("npy_path",          help="Input .npy file.")
    pn.add_argument("--var-name", required=True, help="C variable name for the array.")
    pn.add_argument("--ctype",    required=True, help="C type (e.g., int8_t, uint8_t, float).")
    pn.add_argument("--out",      default="-",   help="Output header path ('-' for stdout).")

    args = parser.parse_args()

    if args.cmd == "gen":
        generate_case(
            name=args.name,
            H=args.H,
            W=args.W,
            Cin=args.Cin,
            stride=args.stride,
            padding=args.padding,
            seed=args.seed,
            out_dir=args.out_dir,
        )
    elif args.cmd == "npy2c":
        npy_to_c_header(args.npy_path, args.var_name, args.ctype, args.out)
    else:
        parser.error("Unknown command")

if __name__ == "__main__":
    main()
