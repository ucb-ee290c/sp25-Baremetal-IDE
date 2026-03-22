#!/usr/bin/env python3
"""Generate deterministic TinySpeech int8 input tensors for baremetal tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

H = 12
W = 94
SIZE = H * W


def _clip_i8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -128, 127).astype(np.int8)


def build_cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(12345)
    t = np.linspace(0.0, 1.0, W, dtype=np.float64)
    m = np.linspace(0.0, 1.0, H, dtype=np.float64)[:, None]

    cases: list[tuple[str, np.ndarray]] = []

    cases.append(("silence", np.zeros((H, W), dtype=np.float64)))

    cases.append(("low_noise", rng.normal(0.0, 7.0, size=(H, W))))

    cases.append(("high_noise", rng.normal(0.0, 20.0, size=(H, W))))

    ramp = (-40.0 + 80.0 * t)[None, :] + (-15.0 + 30.0 * m)
    cases.append(("ramp_tilt", ramp))

    harmonic_a = 28.0 * np.sin(2.0 * np.pi * (1.0 + 3.0 * m) * t[None, :])
    harmonic_a += 9.0 * np.cos(2.0 * np.pi * (0.5 + 2.0 * m) * t[None, :])
    cases.append(("harmonic_a", harmonic_a))

    harmonic_b = 18.0 * np.sin(2.0 * np.pi * (2.0 + 6.0 * m) * t[None, :] + 0.7)
    harmonic_b += 14.0 * np.sin(2.0 * np.pi * (0.8 + 2.5 * m) * t[None, :])
    harmonic_b += 0.7 * ramp
    cases.append(("harmonic_b", harmonic_b))

    checker = np.where(((np.arange(H)[:, None] + np.arange(W)[None, :]) % 2) == 0, 24.0, -24.0)
    checker += 8.0 * np.sin(2.0 * np.pi * t[None, :])
    cases.append(("checker", checker))

    impulse = np.zeros((H, W), dtype=np.float64)
    impulse[:, 18] = 48.0
    impulse[:, 53] = -36.0
    impulse[4:8, 70:73] = 35.0
    impulse += rng.normal(0.0, 3.0, size=(H, W))
    cases.append(("impulse_mix", impulse))

    speech_like = np.zeros((H, W), dtype=np.float64)
    envelope = np.exp(-((t - 0.27) ** 2) / 0.008) + 0.7 * np.exp(-((t - 0.67) ** 2) / 0.015)
    for i in range(H):
        band_freq = 0.8 + 0.55 * i
        speech_like[i, :] = 20.0 * envelope * np.sin(2.0 * np.pi * band_freq * t + 0.13 * i)
    speech_like += rng.normal(0.0, 2.5, size=(H, W))
    cases.append(("speech_like", speech_like))

    return [(name, _clip_i8(arr)) for name, arr in cases]


def emit_header(cases: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("#ifndef TINYSPEECH_TEST_INPUTS_H")
    lines.append("#define TINYSPEECH_TEST_INPUTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define TINYSPEECH_TEST_INPUT_H {H}")
    lines.append(f"#define TINYSPEECH_TEST_INPUT_W {W}")
    lines.append(f"#define TINYSPEECH_TEST_INPUT_SIZE {SIZE}")
    lines.append(f"#define TINYSPEECH_TEST_NUM_CASES {len(cases)}")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("  const char *name;")
    lines.append("  int8_t data[TINYSPEECH_TEST_INPUT_SIZE];")
    lines.append("} tinyspeech_test_input_case_t;")
    lines.append("")
    lines.append("static const tinyspeech_test_input_case_t g_tinyspeech_test_inputs[TINYSPEECH_TEST_NUM_CASES] = {")

    for name, arr in cases:
        flat = arr.reshape(-1)
        lines.append(f"  {{ .name = \"{name}\", .data = {{")
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
        help="Output header path",
    )
    args = parser.parse_args()

    cases = build_cases()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    emit_header(cases, args.out)
    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
