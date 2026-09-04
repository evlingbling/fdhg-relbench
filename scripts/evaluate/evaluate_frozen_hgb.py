#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from fdhg.onboarding.auto_relbench import (
    AutoOnboardingOptions,
    _fit_model,
    _predict_model,
    _metric_score,
)


def evaluate_one(matrix_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (matrix_dir / "manifest.json").read_text()
    )

    train = pd.read_parquet(matrix_dir / "train.parquet")
    val = pd.read_parquet(matrix_dir / "val.parquet")

    features = list(manifest["model_feature_columns"])
    label = manifest["label_col"]
    problem_type = manifest["problem_type"]
    metric = manifest["primary_metric"]

    X_train = train[features]
    y_train = train[label]
    X_val = val[features]
    y_val = val[label]

    options = AutoOnboardingOptions()

    model = _fit_model(
        X_train,
        y_train,
        problem_type=problem_type,
        options=options,
    )

    pred = _predict_model(
        model,
        X_val,
        problem_type=problem_type,
    )

    score = _metric_score(
        y_val,
        pred,
        metric=metric,
        problem_type=problem_type,
    )

    result = {
        "dataset": manifest["dataset"],
        "task": manifest["task"],
        "metric": metric,
        "representation": manifest["selected_variant"],
        "selected_variant": manifest["selected_variant"],
        "predictor": "hgb",
        "seed": 0,
        "score": float(score),
        "model_feature_count": len(features),
        "model_feature_columns": features,
        "train_rows": len(train),
        "validation_rows": len(val),
        "official_validation_was_used_for_selection": False,
        "official_validation_decoder_fit_count": 1,
        "decoder_configuration": {
            "decoder": "hist_gradient_boosting",
            "max_iter": 100,
            "min_samples_leaf": 1,
            "random_seed": 0,
        },
        "source_root": str(matrix_dir / "manifest.json"),
        "provenance": (
            "direct frozen selected-representation "
            "official-validation HGB evaluation"
        ),
        "test_split_accessed": False,
    }

    out = output_dir / "seed_0.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    return result


def main():
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--matrix-dir")
    mode.add_argument("--matrix-root")

    ap.add_argument("--output-dir")
    ap.add_argument("--output-root")
    ap.add_argument("--expect-completed", type=int)

    args = ap.parse_args()

    # --------------------------------------------------------
    # Single-task mode
    # --------------------------------------------------------
    if args.matrix_dir:
        if not args.output_dir:
            ap.error("--output-dir is required with --matrix-dir")

        result = evaluate_one(
            Path(args.matrix_dir),
            Path(args.output_dir),
        )

        print("dataset:", result["dataset"])
        print("task:", result["task"])
        print("variant:", result["selected_variant"])
        print("features:", result["model_feature_count"])
        print("metric:", result["metric"])
        print("score:", repr(result["score"]))

        return

    # --------------------------------------------------------
    # Batch mode
    # --------------------------------------------------------
    if not args.output_root:
        ap.error("--output-root is required with --matrix-root")

    matrix_root = Path(args.matrix_root)
    output_root = Path(args.output_root)

    manifests = sorted(matrix_root.glob("*/manifest.json"))

    if args.expect_completed is not None:
        if len(manifests) != args.expect_completed:
            raise RuntimeError(
                f"Expected {args.expect_completed} frozen manifests, "
                f"found {len(manifests)}"
            )

    if output_root.exists():
        shutil.rmtree(output_root)

    completed = 0

    for mp in manifests:
        matrix_dir = mp.parent
        manifest = json.loads(mp.read_text())

        dataset = str(manifest["dataset"])
        task = str(manifest["task"])

        result = evaluate_one(
            matrix_dir,
            output_root / dataset / task,
        )

        if result["test_split_accessed"] is not False:
            raise RuntimeError(
                f"Test split violation: {dataset}/{task}"
            )

        completed += 1

        print(
            f"[{completed:02d}/{len(manifests):02d}] "
            f"{dataset}/{task} "
            f"{result['metric']}={result['score']}"
        )

    print()
    print("HGB completed:", completed)

    if args.expect_completed is not None:
        if completed != args.expect_completed:
            raise RuntimeError(
                f"Expected {args.expect_completed} completed, "
                f"got {completed}"
            )


if __name__ == "__main__":
    main()
