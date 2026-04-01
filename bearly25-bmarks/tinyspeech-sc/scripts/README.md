# TinySpeech Test Scripts

This directory is self-contained for TinySpeech model training/export and validation.

## Scripts

- `evaluate_current_weights.py`
  - Loads `include/weights.h`, `include/tinyspeech_inputs.h`, and `include/tinyspeech_reference.h`.
  - Runs Python inference for the current SimpleCNN runtime path.
  - Compares prediction/probabilities/logits/stage sums against reference.
  - Writes a report file (`weights_eval_report.txt` by default).

- `rebuild_weights_simplecnn.py`
  - Trains a SimpleCNN model for the runtime architecture:
    - `conv1(1->24)` -> ReLU -> pool
    - `conv2(24->48)` -> ReLU -> pool
    - `conv3(48->96)` -> ReLU -> GAP
    - `fc(96->6)`
  - Uses Speech Commands archive + MFCC front-end (`12x94`, `n_fft=1024`, `n_mels=23`).
  - Defaults to using all available train/validation/test samples for the 6 labels.
  - Exports a runtime-compatible `include/weights.h`.
  - Writes a training report (`rebuild_weights_report.txt` by default).

- `evaluate_archive_accuracy.py`
  - Runs large-scale accuracy evaluation directly on Speech Commands archive splits.
  - Evaluates the current `weights.h` with the same preprocessing/runtime path as C.
  - Reports overall accuracy, per-class accuracy, and confusion matrix.

- `tinyspeech_pipeline.py`
  - Shared parsing, runtime forward path, and `weights.h` writer helpers.

- `gen_tinyspeech_subset_headers.py`
  - Builds `tinyspeech_inputs.h` and `tinyspeech_reference.h` directly from Speech Commands archive.
  - Generates a balanced fixed-size labeled subset (default: 100 cases on the chosen split).

## Quick Usage

```bash
python3 dsp25-tests/tinyspeech-test/scripts/evaluate_current_weights.py
```

```bash
python3 dsp25-tests/tinyspeech-test/scripts/rebuild_weights_simplecnn.py \
  --archive /path/to/speech_commands_v0.02.tar.gz
```

```bash
python3 dsp25-tests/tinyspeech-test/scripts/evaluate_archive_accuracy.py \
  --archive /path/to/speech_commands_v0.02.tar.gz \
  --split test
```

```bash
python3 dsp25-tests/tinyspeech-test/scripts/gen_tinyspeech_subset_headers.py \
  --archive /path/to/speech_commands_v0.02.tar.gz \
  --num-cases 100 \
  --split test
```
