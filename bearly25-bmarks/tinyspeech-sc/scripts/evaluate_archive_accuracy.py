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

from tinyspeech_pipeline import load_weights_header, run_tinyspeech_simplecnn

try:
    import torch
    import torchaudio
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torch and torchaudio are required for evaluate_archive_accuracy.py."
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


def _collect_members(
    archive: Path,
    split: str,
    max_per_label: int,
    seed: int,
) -> List[Tuple[str, int]]:
    by_label: Dict[int, List[str]] = defaultdict(list)
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
            if member_split != split:
                continue
            by_label[y].append(m.name)

    rng = random.Random(seed)
    selected: List[Tuple[str, int]] = []
    for y in range(len(CLASS_NAMES)):
        arr = by_label[y]
        rng.shuffle(arr)
        if max_per_label > 0:
            arr = arr[:max_per_label]
        selected.extend((name, y) for name in arr)
    return selected


def _evaluate_members(
    archive: Path,
    members: List[Tuple[str, int]],
    weights_path: Path,
) -> Tuple[int, int, np.ndarray]:
    weights = load_weights_header(weights_path)
    mfcc_transform = _build_mfcc_transform()
    index = {name: y for name, y in members}
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    total = 0
    correct = 0

    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if (not m.isfile()) or (m.name not in index):
                continue
            y = index[m.name]
            f = tf.extractfile(m)
            if f is None:
                continue
            wav = _load_wav_mono_bytes(f.read(), target_sr=SR)
            q = _mfcc_to_runtime_int8(mfcc_transform, wav)
            out = run_tinyspeech_simplecnn(weights, q.reshape(-1), input_h=N_MFCC, input_w=FRAME_COUNT)
            pred = int(out["pred"][0])
            confusion[y, pred] += 1
            total += 1
            if pred == y:
                correct += 1

    return correct, total, confusion


def _format_confusion(conf: np.ndarray) -> str:
    hdr = "            " + " ".join(f"{n:>7s}" for n in CLASS_NAMES)
    lines = [hdr]
    for i, name in enumerate(CLASS_NAMES):
        row = " ".join(f"{int(conf[i, j]):7d}" for j in range(len(CLASS_NAMES)))
        lines.append(f"{name:>10s} {row}")
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to speech_commands_v0.02.tar.gz",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "include" / "weights.h",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=-1,
        help="If >0, limit items per label for faster evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "archive_eval_report.txt",
    )
    args = parser.parse_args()

    if not args.archive.exists():
        raise RuntimeError(f"Archive not found: {args.archive}")

    members = _collect_members(
        archive=args.archive,
        split=args.split,
        max_per_label=args.max_per_label,
        seed=args.seed,
    )
    if not members:
        raise RuntimeError("No members selected; check split/archive.")

    per_label_counts = defaultdict(int)
    for _, y in members:
        per_label_counts[y] += 1

    correct, total, conf = _evaluate_members(args.archive, members, args.weights)
    acc = (100.0 * float(correct) / float(total)) if total else 0.0

    lines: List[str] = []
    lines.append("TinySpeech archive evaluation")
    lines.append(f"archive     : {args.archive}")
    lines.append(f"weights     : {args.weights}")
    lines.append(f"split       : {args.split}")
    lines.append(f"max_per_lbl : {args.max_per_label}")
    lines.append("samples/label:")
    for y, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:>5s}: {per_label_counts[y]}")
    lines.append("")
    lines.append(f"correct/total: {correct}/{total}")
    lines.append(f"accuracy     : {acc:.4f}%")
    lines.append("")
    lines.append("per-class accuracy:")
    for y, name in enumerate(CLASS_NAMES):
        row_total = int(np.sum(conf[y, :]))
        row_acc = (100.0 * float(conf[y, y]) / float(row_total)) if row_total else 0.0
        lines.append(f"  {name:>5s}: {int(conf[y,y])}/{row_total} ({row_acc:.4f}%)")
    lines.append("")
    lines.append("confusion matrix (rows=true, cols=pred):")
    lines.append(_format_confusion(conf))
    lines.append("")

    text = "\n".join(lines)
    print(text)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text + "\n")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
