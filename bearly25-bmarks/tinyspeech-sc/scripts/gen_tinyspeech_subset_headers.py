#!/usr/bin/env python3

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

from tinyspeech_pipeline import TINYSPEECH_STAGE_NAMES, load_weights_header, run_tinyspeech_simplecnn

try:
    import torch
    import torchaudio
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torch and torchaudio are required for gen_tinyspeech_subset_headers.py."
    ) from exc


CLASS_NAMES = ("yes", "no", "on", "off", "stop", "go")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SR = 16000
N_MFCC = 12
N_FFT = 1024
WIN_LEN = 480
HOP_LEN = 160
N_MELS = 23
FRAME_COUNT = 94


def _load_wav_mono_bytes(raw: bytes, target_sr: int = SR) -> np.ndarray:
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

    target_len = target_sr
    if len(x) > target_len:
        x = x[:target_len]
    elif len(x) < target_len:
        x = np.pad(x, (0, target_len - len(x)))
    return x.astype(np.float32)


def _build_mfcc_transform() -> torchaudio.transforms.MFCC:
    return torchaudio.transforms.MFCC(
        sample_rate=SR,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LEN,
            "win_length": WIN_LEN,
            "n_mels": N_MELS,
            "center": False,
        },
    )


def _mfcc_to_runtime_int8(mfcc_transform: torchaudio.transforms.MFCC, waveform: np.ndarray) -> np.ndarray:
    wav = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mfcc = mfcc_transform(wav).squeeze(0).cpu().numpy().astype(np.float32)
    if tuple(mfcc.shape) != (N_MFCC, FRAME_COUNT):
        raise RuntimeError(f"MFCC shape mismatch: got {tuple(mfcc.shape)}, expected {(N_MFCC, FRAME_COUNT)}")

    amax = float(np.max(np.abs(mfcc)))
    if amax < 1e-12:
        q = np.zeros_like(mfcc, dtype=np.int8)
    else:
        q = np.clip(np.round(mfcc * (127.0 / amax)), -127, 127).astype(np.int8)
    return q


def _read_split_lists(tf: tarfile.TarFile) -> Tuple[set[str], set[str]]:
    val_members = [m for m in tf.getmembers() if m.name.endswith("validation_list.txt")]
    test_members = [m for m in tf.getmembers() if m.name.endswith("testing_list.txt")]
    if not val_members or not test_members:
        raise RuntimeError("validation_list.txt/testing_list.txt not found in archive")

    with tf.extractfile(val_members[0]) as f:
        if f is None:
            raise RuntimeError("Failed reading validation_list.txt")
        val_list = set(x.strip() for x in f.read().decode("utf-8").splitlines() if x.strip())
    with tf.extractfile(test_members[0]) as f:
        if f is None:
            raise RuntimeError("Failed reading testing_list.txt")
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


def _allocate_counts(num_cases: int, nlabels: int) -> List[int]:
    base = num_cases // nlabels
    rem = num_cases % nlabels
    return [base + (1 if i < rem else 0) for i in range(nlabels)]


def _write_inputs_header(path: Path, cases: List[dict]) -> None:
    lines: List[str] = []
    lines.append("#ifndef TINYSPEECH_INPUTS_H")
    lines.append("#define TINYSPEECH_INPUTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define TINYSPEECH_TEST_INPUT_H 12")
    lines.append("#define TINYSPEECH_TEST_INPUT_W 94")
    lines.append("#define TINYSPEECH_TEST_INPUT_SIZE 1128")
    lines.append(f"#define TINYSPEECH_TEST_NUM_CASES {len(cases)}")
    lines.append("#define TINYSPEECH_TEST_WINDOW_MS 30")
    lines.append("#define TINYSPEECH_TEST_HOP_MS 10")
    lines.append("#define TINYSPEECH_TEST_BANDPASS_LOW_HZ 0")
    lines.append("#define TINYSPEECH_TEST_BANDPASS_HIGH_HZ 8000")
    lines.append("#define TINYSPEECH_TEST_CONV1_ACT_SCALE 1.00000000f")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("  const char *name;")
    lines.append("  int32_t expected_label; /* -1 means unknown/background */")
    lines.append("  int8_t data[TINYSPEECH_TEST_INPUT_SIZE];")
    lines.append("} tinyspeech_test_input_case_t;")
    lines.append("")
    lines.append("static const tinyspeech_test_input_case_t g_tinyspeech_test_inputs[TINYSPEECH_TEST_NUM_CASES] = {")

    for c in cases:
        lines.append(f"  {{ .name = \"{c['name']}\", .expected_label = {c['expected_label']}, .data = {{")
        flat = c["data"].reshape(-1).astype(np.int32)
        for i in range(0, flat.size, 24):
            chunk = flat[i : i + 24]
            lines.append("    " + ", ".join(str(int(v)) for v in chunk) + ",")
        lines.append("  } },")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")
    path.write_text("\n".join(lines))


def _fmt_farray(vals: np.ndarray) -> str:
    return ", ".join(f"{float(v):.8f}f" for v in vals.reshape(-1))


def _write_reference_header(path: Path, cases: List[dict]) -> None:
    lines: List[str] = []
    lines.append("#ifndef TINYSPEECH_REFERENCE_H")
    lines.append("#define TINYSPEECH_REFERENCE_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define TINYSPEECH_REF_NUM_CASES {len(cases)}")
    lines.append("#define TINYSPEECH_REF_NUM_CLASSES 6")
    lines.append("#define TINYSPEECH_REF_NUM_STAGES 12")
    lines.append("#define TINYSPEECH_REF_PROB_TOL 0.02f")
    lines.append("#define TINYSPEECH_REF_LOGIT_TOL 2.0f")
    lines.append("#define TINYSPEECH_REF_STAGE_SUM_TOL 64.0f")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("  const char *name;")
    lines.append("  int32_t expected_label;")
    lines.append("  int32_t ref_pred_label;")
    lines.append("  float ref_probs[TINYSPEECH_REF_NUM_CLASSES];")
    lines.append("  float ref_logits[TINYSPEECH_REF_NUM_CLASSES];")
    lines.append("  float ref_stage_sums[TINYSPEECH_REF_NUM_STAGES];")
    lines.append("} tinyspeech_ref_case_t;")
    lines.append("")
    lines.append("static const char *g_tinyspeech_ref_stage_names[TINYSPEECH_REF_NUM_STAGES] = {")
    for s in TINYSPEECH_STAGE_NAMES:
        lines.append(f"  \"{s}\",")
    lines.append("};")
    lines.append("")
    lines.append("static const tinyspeech_ref_case_t g_tinyspeech_ref_cases[TINYSPEECH_REF_NUM_CASES] = {")
    for c in cases:
        lines.append("  {")
        lines.append(f"    .name = \"{c['name']}\",")
        lines.append(f"    .expected_label = {c['expected_label']},")
        lines.append(f"    .ref_pred_label = {int(c['ref_pred'])},")
        lines.append(f"    .ref_probs = {{ {_fmt_farray(c['ref_probs'])} }},")
        lines.append(f"    .ref_logits = {{ {_fmt_farray(c['ref_logits'])} }},")
        lines.append(f"    .ref_stage_sums = {{ {_fmt_farray(c['ref_stage_sums'])} }},")
        lines.append("  },")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "datasets/cache/speech_commands_v0.02.tar.gz",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "include/weights.h",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-inputs", type=Path, default=root / "include/tinyspeech_inputs.h")
    parser.add_argument("--out-reference", type=Path, default=root / "include/tinyspeech_reference.h")
    args = parser.parse_args()

    if not args.archive.exists():
        raise RuntimeError(f"Archive not found: {args.archive}")
    if args.num_cases < len(CLASS_NAMES):
        raise RuntimeError("num-cases must be at least number of classes")

    random.seed(args.seed)
    np.random.seed(args.seed)

    members_by_label = _collect_members_by_label(args.archive, split=args.split)
    counts = _allocate_counts(args.num_cases, len(CLASS_NAMES))

    selected_members: List[Tuple[str, int]] = []
    rng = random.Random(args.seed)
    for y, need in enumerate(counts):
        arr = list(members_by_label[y])
        rng.shuffle(arr)
        if len(arr) < need:
            raise RuntimeError(f"Not enough samples for {CLASS_NAMES[y]}: need {need}, found {len(arr)}")
        selected_members.extend((m, y) for m in arr[:need])

    # Preserve class-grouped order in the header.
    selected_set = {m for m, _ in selected_members}
    member_to_label = {m: y for m, y in selected_members}
    audio_by_member: Dict[str, bytes] = {}
    with tarfile.open(args.archive, "r|gz") as tf:
        for m in tf:
            if (not m.isfile()) or (m.name not in selected_set):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            audio_by_member[m.name] = f.read()

    mfcc_transform = _build_mfcc_transform()
    weights = load_weights_header(args.weights)
    per_label_idx = defaultdict(int)
    cases: List[dict] = []

    for member, y in selected_members:
        raw = audio_by_member.get(member)
        if raw is None:
            raise RuntimeError(f"Selected member missing in archive stream: {member}")
        wav = _load_wav_mono_bytes(raw, target_sr=SR)
        mfcc_q = _mfcc_to_runtime_int8(mfcc_transform, wav)
        out = run_tinyspeech_simplecnn(weights, mfcc_q.reshape(-1), input_h=N_MFCC, input_w=FRAME_COUNT)

        case_name = f"{CLASS_NAMES[y]}_test_{per_label_idx[y]:03d}"
        per_label_idx[y] += 1
        cases.append(
            {
                "name": case_name,
                "expected_label": y,
                "data": mfcc_q.reshape(-1).astype(np.int8),
                "ref_pred": int(out["pred"][0]),
                "ref_probs": out["probs"].astype(np.float32),
                "ref_logits": out["logits"].astype(np.float32),
                "ref_stage_sums": out["stage_sums"].astype(np.float32),
            }
        )

    args.out_inputs.parent.mkdir(parents=True, exist_ok=True)
    args.out_reference.parent.mkdir(parents=True, exist_ok=True)
    _write_inputs_header(args.out_inputs, cases)
    _write_reference_header(args.out_reference, cases)

    print(f"Wrote inputs header   : {args.out_inputs}")
    print(f"Wrote reference header: {args.out_reference}")
    print(f"Generated cases       : {len(cases)}")
    for y, n in enumerate(counts):
        print(f"  {CLASS_NAMES[y]}: {n}")


if __name__ == "__main__":
    main()
