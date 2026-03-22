#!/usr/bin/env python3
"""Generate TinySpeech test inputs from synthesized keyword waveforms.

This script synthesizes 1-second speech-like waveforms for keyword classes,
computes MFCC(12 x 94) features (n_fft=1024, win=480, hop=160, n_mels=23),
then quantizes to int8 and emits a C header consumed by tinyspeech-test.
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import numpy as np

SR = 16000
DURATION_S = 1.0
N_SAMPLES = int(SR * DURATION_S)

N_FFT = 1024
WIN_LEN = 480
HOP_LEN = 160
N_MELS = 23
N_MFCC = 12

INPUT_H = N_MFCC
INPUT_W = 94
INPUT_SIZE = INPUT_H * INPUT_W

KEYWORDS = ["yes", "no", "on", "off", "stop", "go"]
KEYWORD_TO_LABEL = {k: i for i, k in enumerate(KEYWORDS)}


def _hz_to_mel(f_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + (np.asarray(f_hz) / 700.0))


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)


def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_min = 0.0
    f_max = sr / 2.0
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(np.int32)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_fft // 2 + 1)

        for k in range(left, center):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if 0 <= k < fb.shape[1]:
                fb[m - 1, k] = (right - k) / (right - center)

    return fb


def _build_dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels, dtype=np.float64)
    k = np.arange(n_mfcc, dtype=np.float64)[:, None]
    dct = np.sqrt(2.0 / n_mels) * np.cos((np.pi / n_mels) * (n + 0.5) * k)
    dct[0, :] = np.sqrt(1.0 / n_mels)
    return dct


MEL_FB = _build_mel_filterbank(SR, N_FFT, N_MELS)
DCT_MAT = _build_dct_matrix(N_MFCC, N_MELS)


def _compute_mfcc(wave: np.ndarray) -> np.ndarray:
    if wave.shape[0] != N_SAMPLES:
        raise ValueError(f"Expected waveform length {N_SAMPLES}, got {wave.shape[0]}")

    n_frames = 1 + (N_SAMPLES - N_FFT) // HOP_LEN
    if n_frames != INPUT_W:
        raise ValueError(f"Expected {INPUT_W} frames, got {n_frames}")

    win = np.hamming(WIN_LEN)
    win_pad = np.zeros(N_FFT, dtype=np.float64)
    w0 = (N_FFT - WIN_LEN) // 2
    win_pad[w0 : w0 + WIN_LEN] = win

    frames = np.zeros((n_frames, N_FFT), dtype=np.float64)
    for i in range(n_frames):
        s = i * HOP_LEN
        frames[i, :] = wave[s : s + N_FFT]

    frames *= win_pad[None, :]

    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real**2 + spec.imag**2)

    mel = power @ MEL_FB.T
    mel = np.maximum(mel, 1e-10)
    log_mel = np.log(mel)

    mfcc_t = log_mel @ DCT_MAT.T
    mfcc = mfcc_t.T
    return mfcc


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    env = np.ones_like(t)
    a = max(0.001, attack)
    r = max(0.001, release)
    env *= np.clip(t / a, 0.0, 1.0)
    tail = np.clip((1.0 - t) / r, 0.0, 1.0)
    env *= tail
    return env


def _voiced_segment(
    t: np.ndarray,
    f0_start: float,
    f0_end: float,
    amp: float,
    h_count: int,
    vibrato_hz: float,
    vibrato_depth: float,
) -> np.ndarray:
    f0 = np.linspace(f0_start, f0_end, t.size)
    vib = 1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_hz * t)
    f0 = f0 * vib

    phase = 2.0 * np.pi * np.cumsum(f0) / SR
    y = np.zeros_like(t)
    for h in range(1, h_count + 1):
        y += (1.0 / h) * np.sin(h * phase + (0.07 * h))

    env = _envelope(t.size, attack=0.08, release=0.18)
    return amp * env * y


def _fricative_noise(t: np.ndarray, amp: float, rng: np.random.Generator, hp_alpha: float = 0.95) -> np.ndarray:
    n = rng.normal(0.0, 1.0, size=t.size)
    y = np.zeros_like(n)
    prev_x = 0.0
    prev_y = 0.0
    for i, x in enumerate(n):
        yv = hp_alpha * (prev_y + x - prev_x)
        y[i] = yv
        prev_x = x
        prev_y = yv
    env = _envelope(t.size, attack=0.02, release=0.12)
    return amp * env * y


def _synth_keyword_wave(keyword: str, variant: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, DURATION_S, N_SAMPLES, endpoint=False)
    y = np.zeros_like(t)

    jitter = 1.0 + 0.02 * (variant - 1)

    if keyword == "yes":
        s0 = slice(int(0.05 * SR), int(0.23 * SR))
        s1 = slice(int(0.20 * SR), int(0.62 * SR))
        s2 = slice(int(0.56 * SR), int(0.82 * SR))
        y[s0] += _fricative_noise(t[s0], amp=0.22 * jitter, rng=rng)
        y[s1] += _voiced_segment(t[s1], 220, 170, amp=0.42 * jitter, h_count=9, vibrato_hz=4.5, vibrato_depth=0.02)
        y[s2] += _fricative_noise(t[s2], amp=0.18 * jitter, rng=rng)

    elif keyword == "no":
        s0 = slice(int(0.08 * SR), int(0.30 * SR))
        s1 = slice(int(0.28 * SR), int(0.78 * SR))
        y[s0] += _voiced_segment(t[s0], 180, 165, amp=0.34 * jitter, h_count=8, vibrato_hz=4.0, vibrato_depth=0.015)
        y[s1] += _voiced_segment(t[s1], 165, 125, amp=0.46 * jitter, h_count=10, vibrato_hz=5.0, vibrato_depth=0.02)

    elif keyword == "on":
        s0 = slice(int(0.06 * SR), int(0.32 * SR))
        s1 = slice(int(0.30 * SR), int(0.76 * SR))
        y[s0] += _voiced_segment(t[s0], 200, 175, amp=0.38 * jitter, h_count=8, vibrato_hz=4.0, vibrato_depth=0.018)
        y[s1] += _voiced_segment(t[s1], 175, 135, amp=0.44 * jitter, h_count=9, vibrato_hz=4.6, vibrato_depth=0.02)

    elif keyword == "off":
        s0 = slice(int(0.06 * SR), int(0.34 * SR))
        s1 = slice(int(0.32 * SR), int(0.52 * SR))
        s2 = slice(int(0.50 * SR), int(0.82 * SR))
        y[s0] += _voiced_segment(t[s0], 190, 160, amp=0.36 * jitter, h_count=8, vibrato_hz=4.2, vibrato_depth=0.016)
        y[s1] += _fricative_noise(t[s1], amp=0.16 * jitter, rng=rng)
        y[s2] += _fricative_noise(t[s2], amp=0.24 * jitter, rng=rng)

    elif keyword == "stop":
        b0 = slice(int(0.06 * SR), int(0.14 * SR))
        s1 = slice(int(0.16 * SR), int(0.36 * SR))
        s2 = slice(int(0.34 * SR), int(0.58 * SR))
        s3 = slice(int(0.56 * SR), int(0.86 * SR))
        y[b0] += 0.30 * jitter * np.hanning(max(1, b0.stop - b0.start))
        y[s1] += _fricative_noise(t[s1], amp=0.22 * jitter, rng=rng)
        y[s2] += _voiced_segment(t[s2], 175, 150, amp=0.36 * jitter, h_count=9, vibrato_hz=4.7, vibrato_depth=0.02)
        y[s3] += _fricative_noise(t[s3], amp=0.20 * jitter, rng=rng)

    elif keyword == "go":
        s0 = slice(int(0.05 * SR), int(0.30 * SR))
        s1 = slice(int(0.28 * SR), int(0.78 * SR))
        y[s0] += _voiced_segment(t[s0], 185, 170, amp=0.38 * jitter, h_count=8, vibrato_hz=4.2, vibrato_depth=0.016)
        y[s1] += _voiced_segment(t[s1], 170, 120, amp=0.48 * jitter, h_count=10, vibrato_hz=5.1, vibrato_depth=0.022)

    # low background + slight DC offset variation by variant
    y += rng.normal(0.0, 0.004 + 0.0015 * variant, size=y.size)
    y += (variant - 1) * 0.002

    # normalize to avoid clipping while keeping class differences
    peak = max(1e-8, np.max(np.abs(y)))
    y = 0.92 * y / peak
    return y.astype(np.float64)


def _parse_conv1_activation_scale(weights_path: Path) -> float:
    txt = weights_path.read_text(encoding="utf-8")
    m = re.search(
        r"CONV1_ACTIVATION_SCALE\s*=\s*\{[^\}]*?f_data\s*=\s*data_(\d+)\s*\}\s*;",
        txt,
        flags=re.DOTALL,
    )
    if not m:
        return 1.0

    did = m.group(1)
    m2 = re.search(rf"data_{did}\[\]\s*=\s*\{{\s*(0x[0-9a-fA-F]+)\s*\}}\s*;", txt)
    if not m2:
        return 1.0

    bits = int(m2.group(1), 16)
    return struct.unpack(">f", bits.to_bytes(4, byteorder="big", signed=False))[0]


def _mfcc_to_int8(mfcc: np.ndarray, q_scale: float) -> np.ndarray:
    if q_scale <= 1e-9:
        q_scale = 1.0
    q = np.round(mfcc / q_scale)
    return np.clip(q, -127, 127).astype(np.int8)


def build_cases(weights_path: Path) -> list[tuple[str, int, np.ndarray]]:
    rng = np.random.default_rng(20260322)
    _ = _parse_conv1_activation_scale(weights_path)  # kept for reference/debug only

    raw_cases: list[tuple[str, int, np.ndarray]] = []

    # Silence + noise baseline
    silence = np.zeros(N_SAMPLES, dtype=np.float64)
    silence_mfcc = _compute_mfcc(silence)
    silence_mfcc = silence_mfcc - np.mean(silence_mfcc, axis=1, keepdims=True)
    raw_cases.append(("silence", -1, silence_mfcc))

    noise = rng.normal(0.0, 0.02, size=N_SAMPLES)
    noise_mfcc = _compute_mfcc(noise)
    noise_mfcc = noise_mfcc - np.mean(noise_mfcc, axis=1, keepdims=True)
    raw_cases.append(("noise", -1, noise_mfcc))

    # Keyword-specific synthetic speech-like waveforms
    for kw in KEYWORDS:
        label = KEYWORD_TO_LABEL[kw]
        for var in range(3):
            wav = _synth_keyword_wave(kw, variant=var, rng=rng)
            mfcc = _compute_mfcc(wav)
            mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
            raw_cases.append((f"{kw}_v{var}", label, mfcc))

    abs_vals = np.concatenate([np.abs(mfcc).reshape(-1) for _, _, mfcc in raw_cases])
    # Map the 98th percentile near +-64 to keep dynamic range without heavy saturation.
    q_scale = max(np.percentile(abs_vals, 98.0) / 64.0, 1e-4)

    cases: list[tuple[str, int, np.ndarray]] = []
    for name, label, mfcc in raw_cases:
        cases.append((name, label, _mfcc_to_int8(mfcc, q_scale)))

    return cases


def emit_header(cases: list[tuple[str, int, np.ndarray]], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("#ifndef TINYSPEECH_TEST_INPUTS_H")
    lines.append("#define TINYSPEECH_TEST_INPUTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define TINYSPEECH_TEST_INPUT_H {INPUT_H}")
    lines.append(f"#define TINYSPEECH_TEST_INPUT_W {INPUT_W}")
    lines.append(f"#define TINYSPEECH_TEST_INPUT_SIZE {INPUT_SIZE}")
    lines.append(f"#define TINYSPEECH_TEST_NUM_CASES {len(cases)}")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("  const char *name;")
    lines.append("  int32_t expected_label; /* -1 means unknown/background */")
    lines.append("  int8_t data[TINYSPEECH_TEST_INPUT_SIZE];")
    lines.append("} tinyspeech_test_input_case_t;")
    lines.append("")
    lines.append("static const tinyspeech_test_input_case_t g_tinyspeech_test_inputs[TINYSPEECH_TEST_NUM_CASES] = {")

    for name, label, arr in cases:
        flat = arr.reshape(-1)
        lines.append(f"  {{ .name = \"{name}\", .expected_label = {label}, .data = {{")
        row: list[str] = []
        for i, val in enumerate(flat):
            row.append(str(int(val)))
            if len(row) == 24 or i == len(flat) - 1:
                lines.append("    " + ", ".join(row) + ",")
                row = []
        lines.append("  } },")

    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("dsp25-tests/tinyspeech-test/include/weights.h"),
        help="Path to TinySpeech model weights header",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dsp25-tests/tinyspeech-test/include/tinyspeech_test_inputs.h"),
        help="Output input header path",
    )
    args = parser.parse_args()

    cases = build_cases(args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    emit_header(cases, args.out)
    print(f"Wrote {len(cases)} waveform-derived MFCC cases to {args.out}")


if __name__ == "__main__":
    main()
