#!/usr/bin/env python3
"""Generate kws_dsp_signals.h: raw float waveforms for the multi-testcase DSP KWS demo.

The DSP producer computes MFCC on-chip, so it needs RAW waveforms (not the precomputed
int8 MFCC maps the BML reference ships). This extracts a small, class-balanced subset of
16000-sample float waveforms straight from the Google Speech Commands archive and emits
them as a single C header in the same float format as the original yes_test_005_signal.h.

Only numpy + the stdlib `wave` module are needed (NO torch/torchaudio) — MFCC happens on
the DSP, not here.

Selection MIRRORS dsp25-tests/tinyspeech-test/scripts/gen_tinyspeech_subset_headers.py
(same seed, same tar order, same per-class shuffle), so the k-th sample of each class here
is the SAME recording as `<class>_test_00k` in the BML reference (tinyspeech_inputs.h).
That makes expected_label and the BML INPUT-CMP diagnostic line up per case.

Usage:
    python gen_dsp_signals.py \
        --archive .../datasets/cache/speech_commands_v0.02.tar.gz \
        --per-class 5 \
        --out .../c2c-demos/dsp-kws-rolling/include/kws_dsp_signals.h
"""

from __future__ import annotations

import argparse
import io
import random
import tarfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Kept identical to the reference generator so selections match.
CLASS_NAMES = ("yes", "no", "on", "off", "stop", "go")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SR = 16000
NUM_SAMPLES = SR  # 1 second, matches KWS_DSP_YES005_NUM_SAMPLES

# The committed BML reference (tinyspeech_inputs.h) holds 100 cases, class-grouped in CLASS_NAMES
# order. Its per-class allocation (allocate_counts(100, 6)) fixes the absolute index of each
# recording; signal (class c, k-th) here is the SAME recording as reference index REF_OFFSETS[c]+k.
REF_NUM_CASES = 100


def _allocate_counts(num_cases: int, nlabels: int) -> List[int]:
    base = num_cases // nlabels
    rem = num_cases % nlabels
    return [base + (1 if i < rem else 0) for i in range(nlabels)]


def _ref_offsets() -> List[int]:
    counts = _allocate_counts(REF_NUM_CASES, len(CLASS_NAMES))
    offs, acc = [], 0
    for c in counts:
        offs.append(acc)
        acc += c
    return offs


def _load_wav_mono_bytes(raw: bytes, target_sr: int = SR) -> np.ndarray:
    """Decode a WAV blob to a mono float32 array of exactly target_sr samples.

    Identical recipe to gen_tinyspeech_subset_headers.py::_load_wav_mono_bytes so the
    on-chip MFCC input matches the reference feature-extraction input bit-for-bit."""
    with wave.open(io.BytesIO(raw), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)

    if sampwidth == 1:
        x = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported sample width: {sampwidth}")

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    if sr != target_sr:
        t_old = np.linspace(0.0, len(x) / sr, num=len(x), endpoint=False)
        n_new = int(round(len(x) * (target_sr / sr)))
        t_new = np.linspace(0.0, len(x) / sr, num=n_new, endpoint=False)
        x = np.interp(t_new, t_old, x).astype(np.float32)

    if len(x) > target_sr:
        x = x[:target_sr]
    elif len(x) < target_sr:
        x = np.pad(x, (0, target_sr - len(x)))
    return x.astype(np.float32)


def _read_split_lists(tf: tarfile.TarFile) -> Tuple[set, set]:
    val_members = [m for m in tf.getmembers() if m.name.endswith("validation_list.txt")]
    test_members = [m for m in tf.getmembers() if m.name.endswith("testing_list.txt")]
    if not val_members or not test_members:
        raise RuntimeError("validation_list.txt/testing_list.txt not found in archive")
    with tf.extractfile(val_members[0]) as f:
        val_list = set(x.strip() for x in f.read().decode("utf-8").splitlines() if x.strip())
    with tf.extractfile(test_members[0]) as f:
        test_list = set(x.strip() for x in f.read().decode("utf-8").splitlines() if x.strip())
    return val_list, test_list


def _collect_members_by_label(archive: Path, split: str) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    with tarfile.open(archive, "r:gz") as tf:
        val_list, test_list = _read_split_lists(tf)
        for m in tf.getmembers():
            if (not m.isfile()) or (not m.name.endswith(".wav")):
                continue
            parts = [p for p in Path(m.name).parts if p not in ("", ".", "speech_commands_v0.02")]
            if len(parts) < 2:
                continue
            label = parts[-2].lower()
            if label not in LABEL_TO_ID:
                continue
            y = LABEL_TO_ID[label]
            rel = "/".join(parts[-2:])
            member_split = "train"
            if rel in test_list:
                member_split = "test"
            elif rel in val_list:
                member_split = "validation"
            if member_split == split:
                out[y].append(m.name)
    return out


def _fmt_farray(vals: np.ndarray, per_line: int = 8) -> List[str]:
    lines: List[str] = []
    flat = vals.reshape(-1)
    for i in range(0, flat.size, per_line):
        chunk = flat[i : i + per_line]
        lines.append("  " + " ".join(f"{float(v):.8f}f," for v in chunk))
    return lines


def _write_header(path: Path, cases: List[dict]) -> None:
    L: List[str] = []
    L.append("#ifndef C2C_KWS_DSP_SIGNALS_H")
    L.append("#define C2C_KWS_DSP_SIGNALS_H")
    L.append("")
    L.append("/* AUTO-GENERATED by c2c-demos/dsp-kws-rolling/scripts/gen_dsp_signals.py. DO NOT EDIT.")
    L.append(" * Raw float waveforms (Google Speech Commands v0.02) for the multi-testcase KWS demo.")
    L.append(" * Selection mirrors the BML reference generator (seed-matched), so signal k of each")
    L.append(" * class is the same recording as <class>_test_00k in tinyspeech_inputs.h. */")
    L.append("")
    L.append("#include <stdint.h>")
    L.append("")
    L.append(f"#define KWS_DSP_NUM_SIGNALS {len(cases)}")
    L.append(f"#define KWS_DSP_SIGNAL_NUM_SAMPLES {NUM_SAMPLES}u")
    L.append("")
    L.append("typedef struct {")
    L.append("  const char *name;")
    L.append("  int32_t expected_label;              /* class id 0..5 */")
    L.append("  int32_t ref_case_index;              /* matching index into g_tinyspeech_test_inputs[] */")
    L.append("  const float *samples;                /* KWS_DSP_SIGNAL_NUM_SAMPLES entries */")
    L.append("} kws_dsp_signal_t;")
    L.append("")
    for i, c in enumerate(cases):
        L.append(f"/* {c['name']} (expected_label={c['expected_label']}, {CLASS_NAMES[c['expected_label']]}) */")
        L.append(f"static const float g_kws_dsp_sig_data_{i}[KWS_DSP_SIGNAL_NUM_SAMPLES] = {{")
        L.extend(_fmt_farray(c["samples"]))
        L.append("};")
        L.append("")
    L.append("static const kws_dsp_signal_t g_kws_dsp_signals[KWS_DSP_NUM_SIGNALS] = {")
    for i, c in enumerate(cases):
        L.append(
            f'  {{ "{c["name"]}", {c["expected_label"]}, {c["ref_case_index"]}, g_kws_dsp_sig_data_{i} }},'
        )
    L.append("};")
    L.append("")
    L.append("#endif /* C2C_KWS_DSP_SIGNALS_H */")
    L.append("")
    path.write_text("\n".join(L))


def main() -> None:
    root = Path(__file__).resolve().parents[3]  # repo root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "dsp25-tests/tinyspeech-test/datasets/cache/speech_commands_v0.02.tar.gz",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--per-class", type=int, default=5, help="waveforms per keyword")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "c2c-demos/dsp-kws-rolling/include/kws_dsp_signals.h",
    )
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(
            f"Archive not found: {args.archive}\n"
            "Download it with:\n"
            "  mkdir -p dsp25-tests/tinyspeech-test/datasets/cache\n"
            "  curl -L -o dsp25-tests/tinyspeech-test/datasets/cache/speech_commands_v0.02.tar.gz \\\n"
            "    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        )

    min_ref_count = min(_allocate_counts(REF_NUM_CASES, len(CLASS_NAMES)))
    if args.per_class > min_ref_count:
        raise SystemExit(
            f"--per-class {args.per_class} exceeds the smallest reference class block "
            f"({min_ref_count}); ref_case_index would cross into another class. Use <= {min_ref_count}."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    members_by_label = _collect_members_by_label(args.archive, split=args.split)

    # Mirror the reference generator: one RNG, shuffle each class in id order, take first k.
    rng = random.Random(args.seed)
    selected: List[Tuple[str, int]] = []
    for y in range(len(CLASS_NAMES)):
        arr = list(members_by_label[y])
        rng.shuffle(arr)
        if len(arr) < args.per_class:
            raise SystemExit(
                f"Not enough '{CLASS_NAMES[y]}' samples: need {args.per_class}, found {len(arr)}"
            )
        selected.extend((m, y) for m in arr[: args.per_class])

    selected_set = {m for m, _ in selected}
    audio_by_member: Dict[str, bytes] = {}
    with tarfile.open(args.archive, "r|gz") as tf:
        for m in tf:
            if (not m.isfile()) or (m.name not in selected_set):
                continue
            f = tf.extractfile(m)
            if f is not None:
                audio_by_member[m.name] = f.read()

    ref_offsets = _ref_offsets()
    per_label_idx: Dict[int, int] = defaultdict(int)
    cases: List[dict] = []
    for member, y in selected:
        raw = audio_by_member.get(member)
        if raw is None:
            raise SystemExit(f"Selected member missing from archive stream: {member}")
        wav = _load_wav_mono_bytes(raw, target_sr=SR)
        k = per_label_idx[y]
        name = f"{CLASS_NAMES[y]}_test_{k:03d}"
        ref_case_index = ref_offsets[y] + k
        per_label_idx[y] += 1
        cases.append(
            {"name": name, "expected_label": y, "ref_case_index": ref_case_index, "samples": wav}
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_header(args.out, cases)

    print(f"Wrote {args.out}")
    print(f"Signals: {len(cases)} ({args.per_class} per class x {len(CLASS_NAMES)} classes)")
    for c in cases:
        print(
            f"  {c['name']:16s} label={c['expected_label']} ({CLASS_NAMES[c['expected_label']]})"
            f"  ref_case_index={c['ref_case_index']}"
        )


if __name__ == "__main__":
    main()
