#!/usr/bin/env python3
"""Generate golden MFCC outputs for dsp25-tests/mfcc-test.

This script mirrors the coefficient/case generation used in src/main.c and
computes a float32 MFCC reference that can be embedded in a C header.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 16000.0
FFT_LEN = 1024
NUM_MEL = 23
NUM_DCT = 12
NUM_CASES = 8
NUM_FFT_BINS = (FFT_LEN // 2) + 1
MAX_FILTER_COEFS = NUM_MEL * NUM_FFT_BINS
REF_TOL_F32 = 0.25

CASE_NAMES = (
    "silence",
    "impulse",
    "alt_sign",
    "sine_440hz",
    "sine_3khz",
    "chirp_100_to_3k",
    "noise",
    "hard_clipped_mix",
)


def clampf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + (hz / 700.0))


def mel_to_hz(mel: float) -> float:
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)


def generate_window() -> np.ndarray:
    w = np.zeros(FFT_LEN, dtype=np.float32)
    for n in range(FFT_LEN):
        w[n] = 0.54 - (0.46 * math.cos((2.0 * math.pi * n) / float(FFT_LEN - 1)))
    return w


def generate_dct() -> np.ndarray:
    dct = np.zeros((NUM_DCT, NUM_MEL), dtype=np.float32)
    m = float(NUM_MEL)
    for k in range(NUM_DCT):
        alpha = math.sqrt(1.0 / m) if k == 0 else math.sqrt(2.0 / m)
        for n in range(NUM_MEL):
            dct[k, n] = alpha * math.cos((math.pi / m) * (n + 0.5) * k)
    return dct


def generate_mel_filterbank():
    f_min_hz = 20.0
    f_max_hz = 4000.0
    mel_min = hz_to_mel(f_min_hz)
    mel_max = hz_to_mel(f_max_hz)

    bins = [0] * (NUM_MEL + 2)
    for i in range(NUM_MEL + 2):
        frac = i / float(NUM_MEL + 1)
        hz = mel_to_hz(mel_min + frac * (mel_max - mel_min))
        bin_f = ((FFT_LEN + 1.0) * hz) / SAMPLE_RATE_HZ
        bin_f = clampf(bin_f, 0.0, float(NUM_FFT_BINS - 1))
        bins[i] = int(bin_f)

    filter_pos = np.zeros(NUM_MEL, dtype=np.int32)
    filter_lengths = np.zeros(NUM_MEL, dtype=np.int32)
    filter_coefs = np.zeros(MAX_FILTER_COEFS, dtype=np.float32)

    coef_count = 0
    for m in range(NUM_MEL):
        left = bins[m]
        center = bins[m + 1]
        right = bins[m + 2]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        if right >= NUM_FFT_BINS:
            right = NUM_FFT_BINS - 1

        start = 0
        count = 0
        for k in range(left, right + 1):
            if k < center:
                v = (k - left) / float(center - left)
            elif k > center:
                v = (right - k) / float(right - center)
            else:
                v = 1.0

            if v > 0.0 and coef_count < MAX_FILTER_COEFS:
                if count == 0:
                    start = k
                filter_coefs[coef_count] = v
                coef_count += 1
                count += 1

        filter_pos[m] = start
        filter_lengths[m] = count

    return filter_pos, filter_lengths, filter_coefs[:coef_count]


def prepare_cases() -> np.ndarray:
    cases = np.zeros((NUM_CASES, FFT_LEN), dtype=np.float32)
    lcg = 0x12345678

    for n in range(FFT_LEN):
        t = n / SAMPLE_RATE_HZ
        frac = n / float(FFT_LEN - 1)

        cases[0, n] = 0.0
        cases[1, n] = 1.0 if n == 0 else 0.0
        cases[2, n] = -0.9 if (n & 1) else 0.9
        cases[3, n] = 0.9 * math.sin(2.0 * math.pi * 440.0 * t)
        cases[4, n] = 0.9 * math.sin(2.0 * math.pi * 3000.0 * t)

        chirp_hz = 100.0 + (2900.0 * frac)
        cases[5, n] = 0.85 * math.sin(2.0 * math.pi * chirp_hz * t)

        lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
        u = ((lcg & 0x00FFFFFF) / 8388607.5) - 1.0
        cases[6, n] = 0.8 * u

        mix = 1.4 * math.sin(2.0 * math.pi * 700.0 * t) + 1.1 * math.sin(2.0 * math.pi * 1200.0 * t)
        cases[7, n] = clampf(mix, -0.7, 0.7)

    return cases


def mfcc_f32_reference(
    src: np.ndarray,
    window: np.ndarray,
    dct: np.ndarray,
    filter_pos: np.ndarray,
    filter_lengths: np.ndarray,
    filter_coefs: np.ndarray,
) -> np.ndarray:
    x = src.astype(np.float32).copy()
    max_value = float(np.max(np.abs(x)))
    if max_value != 0.0:
        x *= 1.0 / max_value

    x *= window

    # Use the first N/2+1 bins, which is what the filterbank uses in this test.
    mag = np.abs(np.fft.rfft(x, n=FFT_LEN)).astype(np.float32)
    if max_value != 0.0:
        mag *= max_value

    mel = np.zeros(NUM_MEL, dtype=np.float32)
    coef_idx = 0
    for i in range(NUM_MEL):
        start = int(filter_pos[i])
        length = int(filter_lengths[i])
        mel[i] = float(np.dot(mag[start : start + length], filter_coefs[coef_idx : coef_idx + length]))
        coef_idx += length

    mel = np.log(mel + 1.0e-6).astype(np.float32)
    return (dct @ mel).astype(np.float32)


def emit_header(out_path: Path, outputs: np.ndarray) -> None:
    lines: list[str] = []
    lines.append("/* Auto-generated by scripts/gen_mfcc_reference.py. */")
    lines.append("/* Do not edit manually. */")
    lines.append("")
    lines.append("#ifndef MFCC_REFERENCE_DATA_H")
    lines.append("#define MFCC_REFERENCE_DATA_H")
    lines.append("")
    lines.append(f"#define MFCC_REF_NUM_CASES {NUM_CASES}")
    lines.append(f"#define MFCC_REF_NUM_DCT {NUM_DCT}")
    lines.append(f"#define MFCC_REF_FFT_LEN {FFT_LEN}")
    lines.append(f"#define MFCC_REF_F32_TOL {REF_TOL_F32:.6f}f")
    lines.append("")
    lines.append("static const char *const g_mfcc_ref_case_names[MFCC_REF_NUM_CASES] = {")
    for name in CASE_NAMES:
        lines.append(f'  "{name}",')
    lines.append("};")
    lines.append("")
    lines.append("static const float g_mfcc_ref_f32[MFCC_REF_NUM_CASES][MFCC_REF_NUM_DCT] = {")
    for ci, row in enumerate(outputs):
        lines.append(f"  /* {CASE_NAMES[ci]} */")
        row_values = ", ".join(f"{float(v):.9f}f" for v in row)
        lines.append(f"  {{ {row_values} }},")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* MFCC_REFERENCE_DATA_H */")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MFCC golden header for mfcc-test.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "include" / "mfcc_reference_data.h",
        help="Output header path",
    )
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print generated MFCC values to stdout",
    )
    args = parser.parse_args()

    window = generate_window()
    dct = generate_dct()
    filter_pos, filter_lengths, filter_coefs = generate_mel_filterbank()
    cases = prepare_cases()

    outputs = np.zeros((NUM_CASES, NUM_DCT), dtype=np.float32)
    for i in range(NUM_CASES):
        outputs[i] = mfcc_f32_reference(cases[i], window, dct, filter_pos, filter_lengths, filter_coefs)

    if args.print_values:
        for i, row in enumerate(outputs):
            pretty = ", ".join(f"{float(v):.9f}" for v in row)
            print(f"{CASE_NAMES[i]}: {pretty}")

    emit_header(args.out, outputs)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
