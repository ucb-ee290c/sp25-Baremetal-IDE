#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tinyspeech_pipeline import (
    TINYSPEECH_STAGE_NAMES,
    load_inputs_header,
    load_reference_header,
    load_weights_header,
    parse_define_float,
    run_tinyspeech_simplecnn,
)


def _default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    weights = root / "include" / "weights.h"
    inputs = root / "include" / "tinyspeech_inputs.h"
    reference = root / "include" / "tinyspeech_reference.h"
    report = root / "weights_eval_report.txt"
    return weights, inputs, reference, report


def main() -> None:
    d_weights, d_inputs, d_ref, d_report = _default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=d_weights)
    ap.add_argument("--inputs", type=Path, default=d_inputs)
    ap.add_argument("--reference", type=Path, default=d_ref)
    ap.add_argument("--report", type=Path, default=d_report)
    ap.add_argument("--prob-tol", type=float, default=None)
    ap.add_argument("--logit-tol", type=float, default=None)
    ap.add_argument("--stage-tol", type=float, default=None)
    args = ap.parse_args()

    ref_text = args.reference.read_text()
    prob_tol = args.prob_tol if args.prob_tol is not None else parse_define_float(ref_text, "TINYSPEECH_REF_PROB_TOL")
    logit_tol = args.logit_tol if args.logit_tol is not None else parse_define_float(ref_text, "TINYSPEECH_REF_LOGIT_TOL")
    stage_tol = args.stage_tol if args.stage_tol is not None else parse_define_float(ref_text, "TINYSPEECH_REF_STAGE_SUM_TOL")

    weights = load_weights_header(args.weights)
    cases, input_h, input_w = load_inputs_header(args.inputs)
    refs = load_reference_header(args.reference)

    total = len(cases)
    pred_matches_expected = 0
    labeled_cases = 0
    pred_matches_ref = 0
    full_ref_pass = 0
    prob_fail = 0
    logit_fail = 0
    stage_fail = 0
    missing_ref = 0

    lines: list[str] = []
    lines.append("TinySpeech weights evaluation report")
    lines.append(f"weights   : {args.weights}")
    lines.append(f"inputs    : {args.inputs}")
    lines.append(f"reference : {args.reference}")
    lines.append(f"cases     : {total}")
    lines.append(f"tol(prob/logit/stage): {prob_tol:.6f} / {logit_tol:.6f} / {stage_tol:.6f}")
    lines.append("")

    for idx, case in enumerate(cases):
        out = run_tinyspeech_simplecnn(weights, case.data, input_h, input_w)
        probs = out["probs"]
        logits = out["logits"]
        stage_sums = out["stage_sums"]
        pred = int(out["pred"][0])

        if case.expected_label >= 0:
            labeled_cases += 1
            if pred == case.expected_label:
                pred_matches_expected += 1

        ref = refs.get(case.name)
        if ref is None:
            missing_ref += 1
            lines.append(f"[{idx:02d}] {case.name}: pred={pred} expected={case.expected_label} ref=<missing>")
            continue

        if pred == ref.ref_pred_label:
            pred_matches_ref += 1

        prob_max_err = float(np.max(np.abs(probs - ref.ref_probs.astype(np.float64))))
        logit_max_err = float(np.max(np.abs(logits - ref.ref_logits.astype(np.float64))))
        stage_max_err = float(np.max(np.abs(stage_sums - ref.ref_stage_sums.astype(np.float64))))

        ok_prob = prob_max_err <= prob_tol
        ok_logit = logit_max_err <= logit_tol
        ok_stage = stage_max_err <= stage_tol
        ok_all = ok_prob and ok_logit and ok_stage

        if ok_all:
            full_ref_pass += 1
        if not ok_prob:
            prob_fail += 1
        if not ok_logit:
            logit_fail += 1
        if not ok_stage:
            stage_fail += 1

        status = "PASS" if ok_all else "FAIL"
        lines.append(
            f"[{idx:02d}] {case.name}: {status} pred={pred} expected={case.expected_label} ref_pred={ref.ref_pred_label} "
            f"err(prob/logit/stage)={prob_max_err:.6f}/{logit_max_err:.6f}/{stage_max_err:.6f}"
        )

    lines.append("")
    lines.append("Summary")
    lines.append(f"  pred vs expected labels : {pred_matches_expected}/{total}")
    lines.append(f"  pred vs expected labels (labeled only): {pred_matches_expected}/{labeled_cases}")
    lines.append(f"  pred vs ref_pred_label  : {pred_matches_ref}/{total - missing_ref}")
    lines.append(f"  full reference pass     : {full_ref_pass}/{total - missing_ref}")
    lines.append(f"  prob_fail/logit_fail/stage_fail: {prob_fail}/{logit_fail}/{stage_fail}")
    lines.append(f"  missing reference cases : {missing_ref}")
    lines.append("")
    lines.append("Stage order")
    for i, name in enumerate(TINYSPEECH_STAGE_NAMES):
        lines.append(f"  [{i:02d}] {name}")

    text = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    print(text, end="")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
