#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import io
import random
import tarfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tinyspeech_pipeline import write_simplecnn_weights_header

try:
    import torchaudio
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torchaudio is required for rebuild_weights_simplecnn.py. "
        "Install torch+torchaudio in your Python env."
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


class SimpleTinySpeechCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@dataclass
class SplitData:
    x: torch.Tensor
    y: torch.Tensor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _quantize_mfcc_to_int8_like_runtime(mfcc: torch.Tensor) -> torch.Tensor:
    x = mfcc.detach().cpu().numpy().astype(np.float32)
    amax = float(np.max(np.abs(x)))
    if amax < 1e-12:
        q = np.zeros_like(x, dtype=np.int8)
    else:
        scale = 127.0 / amax
        q = np.clip(np.round(x * scale), -127, 127).astype(np.int8)
    return torch.tensor(q.astype(np.float32), dtype=torch.float32)


def _compute_mfcc_quantized(mfcc_transform: torchaudio.transforms.MFCC, waveform: np.ndarray) -> torch.Tensor:
    wav = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mfcc = mfcc_transform(wav).squeeze(0)  # [12,94]
    if tuple(mfcc.shape) != (N_MFCC, FRAME_COUNT):
        raise RuntimeError(f"MFCC shape mismatch: got {tuple(mfcc.shape)}, expected {(N_MFCC, FRAME_COUNT)}")
    return _quantize_mfcc_to_int8_like_runtime(mfcc)


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


def _collect_candidates(archive: Path) -> Dict[str, Dict[int, List[str]]]:
    out: Dict[str, Dict[int, List[str]]] = {
        "train": {i: [] for i in range(len(CLASS_NAMES))},
        "validation": {i: [] for i in range(len(CLASS_NAMES))},
        "test": {i: [] for i in range(len(CLASS_NAMES))},
    }
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
            split = "train"
            if rel in test_list:
                split = "test"
            elif rel in val_list:
                split = "validation"
            out[split][y].append(m.name)
    return out


def _select_members(
    candidates: Dict[str, Dict[int, List[str]]],
    train_per_label: int,
    val_per_label: int,
    test_per_label: int,
    seed: int,
) -> Dict[str, Tuple[str, int]]:
    rng = random.Random(seed)
    selected: Dict[str, Tuple[str, int]] = {}
    split_limits = {"train": train_per_label, "validation": val_per_label, "test": test_per_label}
    for split, by_label in candidates.items():
        limit = split_limits[split]
        for y in range(len(CLASS_NAMES)):
            arr = list(by_label[y])
            rng.shuffle(arr)
            if limit > 0:
                arr = arr[:limit]
            for name in arr:
                selected[name] = (split, y)
    return selected


def _build_feature_splits(archive: Path, selected_by_name: Dict[str, Tuple[str, int]]) -> Dict[str, SplitData]:
    mfcc_transform = _build_mfcc_transform()
    x_by_split: Dict[str, List[torch.Tensor]] = {"train": [], "validation": [], "test": []}
    y_by_split: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}

    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if (not m.isfile()) or (m.name not in selected_by_name):
                continue
            split, y = selected_by_name[m.name]
            f = tf.extractfile(m)
            if f is None:
                continue
            wav = _load_wav_mono_bytes(f.read(), target_sr=SR)
            mfcc_q = _compute_mfcc_quantized(mfcc_transform, wav)  # [12,94] float32 values in int8 range
            x_by_split[split].append(mfcc_q.unsqueeze(0))  # [1,12,94]
            y_by_split[split].append(y)

    out: Dict[str, SplitData] = {}
    for split in ("train", "validation", "test"):
        if not x_by_split[split]:
            out[split] = SplitData(
                x=torch.empty((0, 1, N_MFCC, FRAME_COUNT), dtype=torch.float32),
                y=torch.empty((0,), dtype=torch.long),
            )
            continue
        x = torch.stack(x_by_split[split], dim=0).float()
        y = torch.tensor(y_by_split[split], dtype=torch.long)
        out[split] = SplitData(x=x, y=y)
    return out


def _loader(data: SplitData, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(data.x, data.y), batch_size=batch_size, shuffle=shuffle)


def _accuracy(model: nn.Module, data: SplitData, device: torch.device) -> float:
    if data.x.shape[0] == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device))
        pred = torch.argmax(logits, dim=1).cpu()
        return float((pred == data.y).float().mean().item())


def _model_to_weight_dict(model: SimpleTinySpeechCNN) -> Dict[str, np.ndarray]:
    sd = model.state_dict()
    return {
        "CONV1_WEIGHT": sd["conv1.weight"].detach().cpu().numpy().astype(np.float32),
        "CONV1_BIAS": sd["conv1.bias"].detach().cpu().numpy().astype(np.float32),
        "CONV1_ACTIVATION_SCALE": np.asarray([1.0], dtype=np.float32),
        "CONV2_WEIGHT": sd["conv2.weight"].detach().cpu().numpy().astype(np.float32),
        "CONV2_BIAS": sd["conv2.bias"].detach().cpu().numpy().astype(np.float32),
        "CONV2_ACTIVATION_SCALE": np.asarray([1.0], dtype=np.float32),
        "CONV3_WEIGHT": sd["conv3.weight"].detach().cpu().numpy().astype(np.float32),
        "CONV3_BIAS": sd["conv3.bias"].detach().cpu().numpy().astype(np.float32),
        "CONV3_ACTIVATION_SCALE": np.asarray([1.0], dtype=np.float32),
        "FC_WEIGHT": sd["fc.weight"].detach().cpu().numpy().astype(np.float32),
    }


def main() -> None:
    root = _repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "dsp25-tests/tinyspeech-test/datasets/cache/speech_commands_v0.02.tar.gz",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-label", type=int, default=-1, help="-1 means use all available")
    parser.add_argument("--val-per-label", type=int, default=-1, help="-1 means use all available")
    parser.add_argument("--test-per-label", type=int, default=-1, help="-1 means use all available")
    parser.add_argument("--out-header", type=Path, default=root / "dsp25-tests/tinyspeech-test/include/weights.h")
    parser.add_argument("--report", type=Path, default=root / "dsp25-tests/tinyspeech-test/rebuild_weights_report.txt")
    args = parser.parse_args()

    if not args.archive.exists():
        raise RuntimeError(f"Archive not found: {args.archive}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Collecting candidates...")
    candidates = _collect_candidates(args.archive)
    for split in ("train", "validation", "test"):
        counts = [len(candidates[split][i]) for i in range(len(CLASS_NAMES))]
        print(f"  {split}: " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(CLASS_NAMES))))

    selected = _select_members(
        candidates,
        train_per_label=args.train_per_label,
        val_per_label=args.val_per_label,
        test_per_label=args.test_per_label,
        seed=args.seed,
    )
    print(f"Selected members: {len(selected)}")

    print("Building MFCC tensors...")
    splits = _build_feature_splits(args.archive, selected)
    for split in ("train", "validation", "test"):
        print(f"  {split}: {splits[split].x.shape[0]} samples")
    if splits["train"].x.shape[0] == 0:
        raise RuntimeError("No train samples selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTinySpeechCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    train_loader = _loader(splits["train"], args.batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    report_lines: List[str] = []
    report_lines.append("SimpleCNN rebuild report")
    report_lines.append(f"archive={args.archive}")
    report_lines.append(f"device={device}")
    report_lines.append(
        f"samples train/val/test={splits['train'].x.shape[0]}/{splits['validation'].x.shape[0]}/{splits['test'].x.shape[0]}"
    )
    report_lines.append("")
    report_lines.append("Epoch metrics")
    report_lines.append(
        f"hyperparams: epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
        f"weight_decay={args.weight_decay} label_smoothing={args.label_smoothing}"
    )

    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        n = 0
        correct = 0
        loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
                bs = int(yb.shape[0])
                n += bs
                loss_sum += float(loss.item()) * float(bs)

        train_acc = (float(correct) / float(n)) if n else 0.0
        train_loss = (loss_sum / float(n)) if n else 0.0
        val_acc = _accuracy(model, splits["validation"], device)
        test_acc = _accuracy(model, splits["test"], device)

        print(
            f"  epoch {epoch + 1:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )
        report_lines.append(
            f"epoch {epoch + 1:02d}/{args.epochs}: "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} val_acc={val_acc:.6f} test_acc={test_acc:.6f}"
        )

        score = val_acc if splits["validation"].x.shape[0] > 0 else train_acc
        if score > best_val:
            best_val = score
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_state)
    train_acc = _accuracy(model, splits["train"], device)
    val_acc = _accuracy(model, splits["validation"], device)
    test_acc = _accuracy(model, splits["test"], device)

    print("Best checkpoint accuracy:")
    print(f"  train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}")
    report_lines.append("")
    report_lines.append("Best checkpoint accuracy")
    report_lines.append(f"train={train_acc:.6f}")
    report_lines.append(f"val={val_acc:.6f}")
    report_lines.append(f"test={test_acc:.6f}")

    weights = _model_to_weight_dict(model)
    write_simplecnn_weights_header(weights, args.out_header)
    print(f"Wrote weights.h: {args.out_header}")
    report_lines.append("")
    report_lines.append(f"weights_h={args.out_header}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
