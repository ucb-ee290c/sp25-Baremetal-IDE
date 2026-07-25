# 003 — Validate the C2C KWS demo on more sample recordings

**Status:** active (next task, starting 2026-07-22).
**Prereq DONE:** the full demo works end-to-end for ONE sample (`yes_test_005`, embedded in DSP):
DSP MFCC → C2C turn-taking → BML **float** TinySpeech → correct `yes`. See CLAUDE.md "KWS accuracy /
TinySpeech" and plan 002.

## Goal

Prove the demo classifies more than one input correctly — cover other keywords/samples, not just the
single embedded `yes`. Class order: `0=yes 1=no 2=on 3=off 4=stop 5=go`.

## Context / where things stand

- **One input today:** DSP embeds a single waveform, `c2c-demos/dsp-kws/include/yes_test_005_signal.h`
  (`g_kws_dsp_yes005_signal`, 16000 samples), and re-streams it every case. To test other recordings
  end-to-end, DSP needs more embedded waveforms (or live I2S — plan 001 P1).
- **Reference set already on BML:** `bearly25-bmarks/tinyspeech-mc/include/tinyspeech_inputs.h` has
  **100 int8 MFCC maps + expected_label** (`g_tinyspeech_test_inputs[]`), and
  `tinyspeech_reference.h` has expected predictions. BML already includes these (debug harness).
- **Float pipeline is the working path** (`TINYSPEECH_INT8_PIPELINE=0`); int8 conv2 is broken (deferred).
- **MFCC front-end is approximate:** DSP's on-chip MFCC ≠ the reference extractor bit-for-bit
  (`INPUT-CMP max_abs_diff ≈ 72`), but the float model was robust enough for `yes`. Other samples may
  be more sensitive → may need to match `QUANT_SCALE`/`_ZERO` (or the mel/log recipe).

## Two axes of testing (do the cheap one first)

### A. Model-only sweep over the 100 reference inputs (no DSP, fast, high coverage)
Use the existing golden harness on BML: loop `REF_CASE_INDEX` over many/all of
`g_tinyspeech_test_inputs`, run the float model, and compare `argmax` to `expected_label` (and/or
`tinyspeech_reference.h`). Report pass/fail counts. This validates the **on-chip float model** across
all keywords with zero DSP/link involvement. Cheapest confidence that the model itself is correct
on-silicon for every class. (Can run BML standalone — no DSP needed, like the golden mode but cycling
indices; consider a small standalone loop that needs no turn-taking.)

### B. True end-to-end over multiple DSP-streamed recordings
- Embed a handful of additional waveforms on DSP (a few per class: yes/no/on/off/stop/go), following
  the `yes_test_005_signal.h` pattern (a `#define NAME`, `NUM_SAMPLES`, `static const float[]`).
- Have DSP cycle through them (one case per recording), tagging each case with which recording it is
  (e.g. an index/label in the spad control block for BML to log against).
- BML logs predicted vs expected per case. This exercises the real MFCC→C2C→inference path per sample.
- If accuracy is poor for non-`yes` samples, tune the MFCC front-end (scale/zero first; then the
  window/mel/DCT recipe to match the reference extractor). Use `INPUT-CMP` `max_abs_diff` as the metric.

## Steps
1. **A first:** add a BML sweep mode (cycle `REF_CASE_INDEX`, compare argmax vs `expected_label`,
   print a pass/fail tally). Confirm the float model is correct across all 100 reference cases on-chip.
2. **B:** add 2–3 recordings per class as DSP signal headers; DSP round-robins them; BML checks
   predicted vs expected. Tune MFCC scale if needed.
3. Record accuracy results and any MFCC-tuning in CLAUDE.md.

## Deferred (separate work, not blocking 003)
- **int8 conv2 kernel fix** (RVV) to restore the fast path — dump on-chip per-stage sums vs
  `tinyspeech_reference.h::ref_stage_sums` to localize the diverging layer. Then re-enable
  `TINYSPEECH_INT8_PIPELINE=1` and bake the 3 calibration maxima via `tinyspeech_int8_calib_set_max`.
- **Exact MFCC match** to the reference extractor (only if float accuracy on real DSP features is poor).
- **I2S live audio** (plan 001 P1).
