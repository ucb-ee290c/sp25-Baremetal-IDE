#!/usr/bin/env python3
"""Generate TinySpeech test inputs using TinySpeech-aligned preprocessing.

The preprocessing basis is aligned with:
- tinyspeech_content/Methods.tex:
  1s audio, band-pass 20Hz..4kHz, MFCC stack with 30ms window and 10ms hop.
- tinyspeech/training/data.py:
  n_fft=1024, win_length=480, hop_length=160, n_mels=23, n_mfcc=12,
  center=False.

For better parity with torchaudio MFCC defaults, this script computes:
1) power mel spectrogram
2) dB conversion with top_db clipping
3) DCT-II (ortho) to MFCC

Finally, the MFCC matrix is quantized to int8 in the expected model input layout
[1, 1, 12, 94].
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

SR = 16000
DURATION_S = 1.0
N_SAMPLES = int(SR * DURATION_S)

N_FFT = 1024
WIN_LEN = 480  # 30 ms @ 16 kHz
HOP_LEN = 160  # 10 ms @ 16 kHz
N_MELS = 23
N_MFCC = 12

BANDPASS_LOW_HZ = 20.0
BANDPASS_HIGH_HZ = 4000.0
TOP_DB = 80.0

INPUT_H = N_MFCC
INPUT_W = 94
INPUT_SIZE = INPUT_H * INPUT_W

KEYWORDS = ["yes", "no", "on", "off", "stop", "go"]
KEYWORD_TO_LABEL = {k: i for i, k in enumerate(KEYWORDS)}


def _hz_to_mel(f_hz: np.ndarray | float) -> np.ndarray | float:
    # HTK mel mapping (matches common KWS pipelines).
    return 2595.0 * np.log10(1.0 + (np.asarray(f_hz) / 700.0))


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10 ** (np.asarray(mel) / 2595.0) - 1.0)


def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
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


def _build_dct_matrix_ortho(n_mfcc: int, n_mels: int) -> np.ndarray:
    # DCT-II orthonormal basis (torchaudio/librosa style for MFCC).
    n = np.arange(n_mels, dtype=np.float64)
    k = np.arange(n_mfcc, dtype=np.float64)[:, None]
    dct = np.sqrt(2.0 / n_mels) * np.cos((np.pi / n_mels) * (n + 0.5) * k)
    dct[0, :] *= 1.0 / np.sqrt(2.0)
    return dct


def _periodic_hann(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * i) / n)


def _bandpass_fft(wave: np.ndarray, sr: int, f_low: float, f_high: float) -> np.ndarray:
    # Lightweight zero-phase spectral mask band-pass (20Hz..4kHz).
    spec = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(wave.size, d=1.0 / sr)
    mask = (freqs >= f_low) & (freqs <= f_high)
    filt = np.fft.irfft(spec * mask, n=wave.size)
    return filt.astype(np.float64)


MEL_FB = _build_mel_filterbank(
    sr=SR,
    n_fft=N_FFT,
    n_mels=N_MELS,
    f_min=BANDPASS_LOW_HZ,
    f_max=BANDPASS_HIGH_HZ,
)
DCT_MAT = _build_dct_matrix_ortho(N_MFCC, N_MELS)


def _compute_mfcc(wave: np.ndarray) -> np.ndarray:
    if wave.shape[0] != N_SAMPLES:
        raise ValueError(f"Expected waveform length {N_SAMPLES}, got {wave.shape[0]}")

    wave = _bandpass_fft(wave.astype(np.float64), SR, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)

    n_frames = 1 + (N_SAMPLES - N_FFT) // HOP_LEN
    if n_frames != INPUT_W:
        raise ValueError(f"Expected {INPUT_W} frames, got {n_frames}")

    # center=False framing
    frames = np.zeros((n_frames, N_FFT), dtype=np.float64)
    for i in range(n_frames):
        s = i * HOP_LEN
        frames[i, :] = wave[s : s + N_FFT]

    # win_length < n_fft uses centered zero-padding.
    win = _periodic_hann(WIN_LEN)
    win_pad = np.zeros(N_FFT, dtype=np.float64)
    start = (N_FFT - WIN_LEN) // 2
    win_pad[start : start + WIN_LEN] = win
    frames *= win_pad[None, :]

    spec = np.fft.rfft(frames, axis=1)
    power = np.real(spec * np.conj(spec))

    mel = power @ MEL_FB.T
    mel = np.maximum(mel, 1e-10)

    # torchaudio-like power->dB with top_db clipping.
    mel_db = 10.0 * np.log10(mel)
    mel_db -= np.max(mel_db)
    mel_db = np.maximum(mel_db, -TOP_DB)

    # [frames, mels] x [mels, mfcc]^T -> [frames, mfcc], then transpose.
    mfcc_t = mel_db @ DCT_MAT.T
    return mfcc_t.T


def _quantize_like_tinyspeech_input(mfcc: np.ndarray) -> np.ndarray:
    """Quantize MFCC to int8 in a way consistent with TinySpeech QAT input style.

    BitConv2d in training normalizes input by RMS over (H, W), then applies
    int8 activation quantization with per-row absmax over the time axis.
    """
    rms = math.sqrt(float(np.mean(mfcc * mfcc)))
    if rms < 1e-12:
        return np.zeros_like(mfcc, dtype=np.int8)

    x_norm = mfcc / rms
    row_absmax = np.max(np.abs(x_norm), axis=1, keepdims=True)
    row_absmax = np.maximum(row_absmax, 1e-8)

    q = np.round((127.0 / row_absmax) * x_norm)
    q = np.clip(q, -127.0, 127.0)
    return q.astype(np.int8)


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
        y += (1.0 / h) * np.sin((h * phase) + (0.07 * h))

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

    y += rng.normal(0.0, 0.004 + 0.0015 * variant, size=y.size)
    y += (variant - 1) * 0.002

    peak = max(1e-8, np.max(np.abs(y)))
    y = 0.92 * y / peak
    return y.astype(np.float64)


def build_cases() -> list[tuple[str, int, np.ndarray]]:
    rng = np.random.default_rng(20260322)
    cases: list[tuple[str, int, np.ndarray]] = []

    silence = np.zeros(N_SAMPLES, dtype=np.float64)
    silence_mfcc = _compute_mfcc(silence)
    cases.append(("silence", -1, _quantize_like_tinyspeech_input(silence_mfcc)))

    noise = rng.normal(0.0, 0.02, size=N_SAMPLES)
    noise_mfcc = _compute_mfcc(noise)
    cases.append(("noise", -1, _quantize_like_tinyspeech_input(noise_mfcc)))

    for kw in KEYWORDS:
        label = KEYWORD_TO_LABEL[kw]
        for var in range(3):
            wav = _synth_keyword_wave(kw, variant=var, rng=rng)
            mfcc = _compute_mfcc(wav)
            cases.append((f"{kw}_v{var}", label, _quantize_like_tinyspeech_input(mfcc)))

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
    lines.append(f"#define TINYSPEECH_TEST_BANDPASS_LOW_HZ {int(BANDPASS_LOW_HZ)}")
    lines.append(f"#define TINYSPEECH_TEST_BANDPASS_HIGH_HZ {int(BANDPASS_HIGH_HZ)}")
    lines.append(f"#define TINYSPEECH_TEST_WINDOW_MS {int((WIN_LEN * 1000) / SR)}")
    lines.append(f"#define TINYSPEECH_TEST_HOP_MS {int((HOP_LEN * 1000) / SR)}")
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
        "--out",
        type=Path,
        default=Path("dsp25-tests/tinyspeech-test/include/tinyspeech_test_inputs.h"),
        help="Output input header path",
    )
    args = parser.parse_args()

    cases = build_cases()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    emit_header(cases, args.out)
    print(f"Wrote {len(cases)} TinySpeech-format MFCC cases to {args.out}")


if __name__ == "__main__":
    main()
