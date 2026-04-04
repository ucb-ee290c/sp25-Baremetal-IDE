#!/usr/bin/env python3

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


TINYSPEECH_STAGE_NAMES = [
    "input",
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2",
    "conv3",
    "relu3",
    "gap",
    "fc_logits",
    "softmax",
]


@dataclass
class InputCase:
    name: str
    expected_label: int
    data: np.ndarray


@dataclass
class RefCase:
    name: str
    expected_label: int
    ref_pred_label: int
    ref_probs: np.ndarray
    ref_logits: np.ndarray
    ref_stage_sums: np.ndarray


def _tokens_from_brace_body(body: str) -> List[str]:
    return [tok.strip() for tok in body.replace("\n", " ").split(",") if tok.strip()]


def _parse_float_token(tok: str) -> float:
    t = tok.strip()
    if t.endswith(("f", "F")):
        t = t[:-1]
    if t.startswith("0x") or t.startswith("0X"):
        bits = int(t, 16) & 0xFFFFFFFF
        return struct.unpack(">f", struct.pack(">I", bits))[0]
    return float(t)


def _parse_float_array(body: str) -> np.ndarray:
    vals = [_parse_float_token(tok) for tok in _tokens_from_brace_body(body)]
    return np.asarray(vals, dtype=np.float32)


def _parse_int_array(body: str) -> np.ndarray:
    vals = [int(tok, 0) for tok in _tokens_from_brace_body(body)]
    return np.asarray(vals, dtype=np.int32)


def parse_define_int(header_text: str, name: str) -> int:
    m = re.search(rf"#define\s+{re.escape(name)}\s+(-?\d+)", header_text)
    if not m:
        raise RuntimeError(f"Missing define: {name}")
    return int(m.group(1))


def parse_define_float(header_text: str, name: str) -> float:
    m = re.search(rf"#define\s+{re.escape(name)}\s+([0-9eE+\-\.]+)f?", header_text)
    if not m:
        raise RuntimeError(f"Missing define: {name}")
    return float(m.group(1))


def load_weights_header(path: Path) -> Dict[str, np.ndarray]:
    text = Path(path).read_text()

    shape_map: Dict[int, List[int]] = {}
    for m in re.finditer(r"static\s+u_int8_t\s+shape_(\d+)\[\]\s*=\s*\{([^}]*)\};", text, re.S):
        sid = int(m.group(1))
        shape_map[sid] = [int(tok, 0) for tok in _tokens_from_brace_body(m.group(2))]

    float_data_map: Dict[int, np.ndarray] = {}
    for m in re.finditer(r"static\s+float\s+data_(\d+)\[\]\s*=\s*\{([^}]*)\};", text, re.S):
        did = int(m.group(1))
        float_data_map[did] = _parse_float_array(m.group(2))

    int8_data_map: Dict[int, np.ndarray] = {}
    for m in re.finditer(r"static\s+int8_t\s+data_(\d+)\[\]\s*=\s*\{([^}]*)\};", text, re.S):
        did = int(m.group(1))
        int8_data_map[did] = _parse_int_array(m.group(2)).astype(np.int8)

    tensors: Dict[str, np.ndarray] = {}
    tensor_re = re.compile(
        r"static\s+Tensor\s+([A-Z0-9_]+)\s*=\s*\{\s*"
        r"\.dims\s*=\s*(\d+)\s*,\s*"
        r"\.size\s*=\s*(\d+)\s*,\s*"
        r"\.shape\s*=\s*shape_(\d+)\s*,\s*"
        r"\.data\s*=\s*(NULL|data_(\d+))\s*,\s*"
        r"\.f_data\s*=\s*(NULL|data_(\d+))\s*"
        r"\};",
        re.S,
    )
    for m in tensor_re.finditer(text):
        name = m.group(1)
        dims = int(m.group(2))
        size = int(m.group(3))
        shape_id = int(m.group(4))
        data_int_id = m.group(6)
        data_float_id = m.group(8)

        if shape_id not in shape_map:
            raise RuntimeError(f"Tensor {name}: missing shape_{shape_id}")
        shape = tuple(shape_map[shape_id][:dims])

        if data_float_id is not None:
            arr = float_data_map[int(data_float_id)]
            tensors[name] = arr.reshape(shape).astype(np.float32, copy=False)
        elif data_int_id is not None:
            arr = int8_data_map[int(data_int_id)]
            tensors[name] = arr.reshape(shape)
        else:
            raise RuntimeError(f"Tensor {name}: neither data nor f_data points to data array.")

        if int(np.prod(shape)) != size:
            raise RuntimeError(f"Tensor {name}: shape product {int(np.prod(shape))} != size {size}")

    required = [
        "CONV1_WEIGHT",
        "CONV1_BIAS",
        "CONV1_ACTIVATION_SCALE",
        "CONV2_WEIGHT",
        "CONV2_BIAS",
        "CONV2_ACTIVATION_SCALE",
        "CONV3_WEIGHT",
        "CONV3_BIAS",
        "CONV3_ACTIVATION_SCALE",
        "FC_WEIGHT",
    ]
    missing = [k for k in required if k not in tensors]
    if missing:
        raise RuntimeError(f"weights.h missing tensors: {missing}")

    return tensors


def load_inputs_header(path: Path) -> tuple[List[InputCase], int, int]:
    text = Path(path).read_text()
    h = parse_define_int(text, "TINYSPEECH_TEST_INPUT_H")
    w = parse_define_int(text, "TINYSPEECH_TEST_INPUT_W")
    input_size = parse_define_int(text, "TINYSPEECH_TEST_INPUT_SIZE")
    ncases = parse_define_int(text, "TINYSPEECH_TEST_NUM_CASES")

    case_re = re.compile(
        r"\{\s*\.name\s*=\s*\"([^\"]+)\"\s*,\s*"
        r"\.expected_label\s*=\s*(-?\d+)\s*,\s*"
        r"\.data\s*=\s*\{([^}]*)\}\s*\},",
        re.S,
    )

    out: List[InputCase] = []
    for m in case_re.finditer(text):
        name = m.group(1)
        expected = int(m.group(2))
        vals = _parse_int_array(m.group(3)).astype(np.int8)
        if vals.size != input_size:
            raise RuntimeError(f"Input case {name}: size {vals.size}, expected {input_size}")
        out.append(InputCase(name=name, expected_label=expected, data=vals))

    if len(out) != ncases:
        raise RuntimeError(f"Parsed {len(out)} input cases, expected {ncases}")

    return out, h, w


def load_reference_header(path: Path) -> Dict[str, RefCase]:
    text = Path(path).read_text()
    nclasses = parse_define_int(text, "TINYSPEECH_REF_NUM_CLASSES")
    nstages = parse_define_int(text, "TINYSPEECH_REF_NUM_STAGES")

    case_re = re.compile(
        r"\{\s*"
        r"\.name\s*=\s*\"([^\"]+)\"\s*,\s*"
        r"\.expected_label\s*=\s*(-?\d+)\s*,\s*"
        r"\.ref_pred_label\s*=\s*(-?\d+)\s*,\s*"
        r"\.ref_probs\s*=\s*\{([^}]*)\}\s*,\s*"
        r"\.ref_logits\s*=\s*\{([^}]*)\}\s*,\s*"
        r"\.ref_stage_sums\s*=\s*\{([^}]*)\}\s*,\s*"
        r"\},",
        re.S,
    )

    refs: Dict[str, RefCase] = {}
    for m in case_re.finditer(text):
        name = m.group(1)
        expected = int(m.group(2))
        ref_pred = int(m.group(3))
        probs = _parse_float_array(m.group(4))
        logits = _parse_float_array(m.group(5))
        stage_sums = _parse_float_array(m.group(6))
        if probs.size != nclasses:
            raise RuntimeError(f"{name}: ref_probs size {probs.size} != {nclasses}")
        if logits.size != nclasses:
            raise RuntimeError(f"{name}: ref_logits size {logits.size} != {nclasses}")
        if stage_sums.size != nstages:
            raise RuntimeError(f"{name}: ref_stage_sums size {stage_sums.size} != {nstages}")
        refs[name] = RefCase(
            name=name,
            expected_label=expected,
            ref_pred_label=ref_pred,
            ref_probs=probs,
            ref_logits=logits,
            ref_stage_sums=stage_sums,
        )

    return refs


def conv2d_nchw(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray,
    out_scale: float = 1.0,
    stride: int = 1,
    padding: int = 1,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)

    n, c, h, width = x.shape
    oc, wc, kh, kw = w.shape
    if wc != c:
        raise RuntimeError(f"conv channel mismatch: input C={c}, weight C={wc}")

    xpad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    win = sliding_window_view(xpad, (kh, kw), axis=(2, 3))  # [N,C,OH,OW,KH,KW]
    win = win[:, :, ::stride, ::stride, :, :]

    out = np.tensordot(win, w, axes=([1, 4, 5], [1, 2, 3]))  # [N,OH,OW,OC]
    out = np.moveaxis(out, -1, 1).astype(np.float32, copy=False)  # [N,OC,OH,OW]
    out += bias.reshape(1, oc, 1, 1)

    scale = np.float32(out_scale if abs(out_scale) > 1e-12 else 1.0)
    out = out / scale
    return out.astype(np.float32, copy=False)


def maxpool2d_nchw(x: np.ndarray, kernel: int = 2, stride: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    win = sliding_window_view(x, (kernel, kernel), axis=(2, 3))  # [N,C,OH,OW,K,K]
    win = win[:, :, ::stride, ::stride, :, :]
    out = np.max(win, axis=(-1, -2))
    return out.astype(np.float32, copy=False)


def run_tinyspeech_simplecnn(
    weights: Dict[str, np.ndarray],
    case_data_int8: np.ndarray,
    input_h: int,
    input_w: int,
) -> Dict[str, np.ndarray]:
    x0 = np.asarray(case_data_int8, dtype=np.int8).reshape(1, 1, input_h, input_w)
    stage: Dict[str, np.ndarray] = {}

    stage["input"] = x0.astype(np.float32)

    x = conv2d_nchw(
        stage["input"],
        weights["CONV1_WEIGHT"],
        weights["CONV1_BIAS"].reshape(-1),
        out_scale=float(weights["CONV1_ACTIVATION_SCALE"].reshape(-1)[0]),
        stride=1,
        padding=1,
    )
    stage["conv1"] = x

    x = np.maximum(x, 0.0).astype(np.float32, copy=False)
    stage["relu1"] = x

    x = maxpool2d_nchw(x, kernel=2, stride=2)
    stage["pool1"] = x

    x = conv2d_nchw(
        x,
        weights["CONV2_WEIGHT"],
        weights["CONV2_BIAS"].reshape(-1),
        out_scale=float(weights["CONV2_ACTIVATION_SCALE"].reshape(-1)[0]),
        stride=1,
        padding=1,
    )
    stage["conv2"] = x

    x = np.maximum(x, 0.0).astype(np.float32, copy=False)
    stage["relu2"] = x

    x = maxpool2d_nchw(x, kernel=2, stride=2)
    stage["pool2"] = x

    x = conv2d_nchw(
        x,
        weights["CONV3_WEIGHT"],
        weights["CONV3_BIAS"].reshape(-1),
        out_scale=float(weights["CONV3_ACTIVATION_SCALE"].reshape(-1)[0]),
        stride=1,
        padding=1,
    )
    stage["conv3"] = x

    x = np.maximum(x, 0.0).astype(np.float32, copy=False)
    stage["relu3"] = x

    x = np.mean(x, axis=(2, 3), keepdims=True).astype(np.float32, copy=False)
    stage["gap"] = x

    flat = x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    logits = flat @ weights["FC_WEIGHT"].astype(np.float32).T
    logits = logits.astype(np.float32, copy=False)
    stage["fc_logits"] = logits

    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shift, dtype=np.float32)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs = probs.astype(np.float32, copy=False)
    stage["softmax"] = probs

    stage_sums = np.asarray(
        [np.sum(stage[name], dtype=np.float64) for name in TINYSPEECH_STAGE_NAMES],
        dtype=np.float64,
    )
    pred = int(np.argmax(probs[0]))

    return {
        "probs": probs[0].astype(np.float64),
        "logits": logits[0].astype(np.float64),
        "stage_sums": stage_sums,
        "pred": np.asarray([pred], dtype=np.int32),
    }


def write_simplecnn_weights_header(weights: Dict[str, np.ndarray], out_path: Path) -> None:
    required = [
        "CONV1_WEIGHT",
        "CONV1_BIAS",
        "CONV1_ACTIVATION_SCALE",
        "CONV2_WEIGHT",
        "CONV2_BIAS",
        "CONV2_ACTIVATION_SCALE",
        "CONV3_WEIGHT",
        "CONV3_BIAS",
        "CONV3_ACTIVATION_SCALE",
        "FC_WEIGHT",
    ]
    missing = [k for k in required if k not in weights]
    if missing:
        raise RuntimeError(f"Cannot write weights.h, missing keys: {missing}")

    def _fmt_vals(vals: Iterable[float], per_line: int = 12) -> str:
        parts = []
        line = []
        for i, v in enumerate(vals):
            line.append(f"{float(v):.8f}")
            if (i + 1) % per_line == 0:
                parts.append("  " + ", ".join(line))
                line = []
        if line:
            parts.append("  " + ", ".join(line))
        return ",\n".join(parts)

    tensor_order = [
        "CONV1_WEIGHT",
        "CONV1_BIAS",
        "CONV1_ACTIVATION_SCALE",
        "CONV2_WEIGHT",
        "CONV2_BIAS",
        "CONV2_ACTIVATION_SCALE",
        "CONV3_WEIGHT",
        "CONV3_BIAS",
        "CONV3_ACTIVATION_SCALE",
        "FC_WEIGHT",
    ]

    shape_id = 0
    data_id = 1
    decls: List[str] = []
    mapping: List[str] = []

    for idx, name in enumerate(tensor_order):
        arr = np.asarray(weights[name], dtype=np.float32)
        shape = list(arr.shape)
        if not shape:
            shape = [1]
            arr = arr.reshape(1)
        dims = len(shape)
        size = int(np.prod(shape))
        shape_str = ", ".join(str(int(x)) for x in shape)
        data_str = _fmt_vals(arr.reshape(-1))

        decls.append(
            "\n".join(
                [
                    f"static u_int8_t shape_{shape_id}[] = {{ {shape_str} }};",
                    f"static float data_{data_id}[] = {{",
                    f"{data_str}",
                    "};",
                    f"static Tensor {name} = {{",
                    f"    .dims = {dims},",
                    f"    .size = {size},",
                    f"    .shape = shape_{shape_id},",
                    "    .data = NULL,",
                    f"    .f_data = data_{data_id}",
                    "};",
                ]
            )
        )
        mapping.append(f"    {{ {idx}, &{name} }},")
        shape_id += 2
        data_id += 2

    out_text = "\n".join(
        [
            "// Automatically generated header file",
            "// Model Type: SimpleCNN",
            "",
            "#ifndef TINYSPEECH_WEIGHTS_H",
            "#define TINYSPEECH_WEIGHTS_H",
            "",
            "#include \"tensor.h\"",
            "#include <stdint.h>",
            "",
            "#define TINYSPEECH_MODEL_SIMPLECNN 1",
            "#define QUANT_MODE_QAT_SQ",
            "#define CONVERT_FLOAT 1",
            "#define CONVERT_INT8 0",
            "",
            "\n\n".join(decls),
            "",
            "typedef struct {",
            "    const u_int8_t id;",
            "    Tensor* address;",
            "} VariableMap;",
            "",
            "VariableMap model_weights[] = {",
            "\n".join(mapping),
            "};",
            "",
            "#endif",
            "",
        ]
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text)
