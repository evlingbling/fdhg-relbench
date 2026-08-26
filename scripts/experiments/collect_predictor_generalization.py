#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SEEDS = [41, 42, 43, 44]
PREDICTORS = ["xgboost", "catboost"]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_structural_skips(path: Path):
    rows = load_json(path)

    skips = {}

    for row in rows:
        for seed in row["seeds"]:
            key = (
                str(row["dataset"]),
                str(row["task"]),
                str(row["predictor"]),
                int(seed),
            )

            if key in skips:
                raise RuntimeError(
                    f"duplicate structural skip: {key}"
                )

            skips[key] = str(row["reason"])

    return skips


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "outputs/predictor-generalization/"
            "frozen-gbdt"
        ),
    )

    ap.add_argument(
        "--task-csv",
        type=Path,
        default=Path(
            "configs/benchmark_tasks.csv"
        ),
    )

    ap.add_argument(
        "--structural-skips",
        type=Path,
        default=Path(
            "configs/generalization/"
            "structural_skips.json"
        ),
    )

    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/predictor-generalization/"
            "generalization_runs.csv"
        ),
    )

    args = ap.parse_args()

    with args.task_csv.open() as f:
        task_rows = list(
            csv.DictReader(f)
        )

    tasks = []

    for row in task_rows:
        dataset = str(row["dataset"]).strip()
        task = str(row["task"]).strip()

        if dataset and task:
            tasks.append(
                (dataset, task)
            )

    tasks = list(dict.fromkeys(tasks))

    if len(tasks) != 51:
        raise SystemExit(
            "REFUSING: expected 51 benchmark tasks, "
            f"found {len(tasks)}"
        )

    structural_skips = load_structural_skips(
        args.structural_skips
    )

    expected_keys = {
        (
            dataset,
            task,
            predictor,
            seed,
        )
        for dataset, task in tasks
        for predictor in PREDICTORS
        for seed in SEEDS
    }

    unknown_skips = (
        set(structural_skips)
        - expected_keys
    )

    if unknown_skips:
        raise RuntimeError(
            "structural skip outside expected grid: "
            f"{sorted(unknown_skips)}"
        )

    rows = []
    completed = 0
    skipped = 0
    failed = 0

    for dataset, task in tasks:
        slug = f"{dataset}_{task}"

        for predictor in PREDICTORS:
            for seed in SEEDS:

                key = (
                    dataset,
                    task,
                    predictor,
                    seed,
                )

                metrics_path = (
                    args.result_root
                    / slug
                    / predictor
                    / f"seed_{seed}"
                    / "metrics.json"
                )

                result = None

                if metrics_path.exists():
                    try:
                        result = load_json(
                            metrics_path
                        )
                    except Exception as exc:
                        rows.append({
                            "dataset": dataset,
                            "task": task,
                            "predictor": predictor,
                            "seed": seed,
                            "status": "failed",
                            "reason": (
                                "invalid_metrics_json:"
                                + type(exc).__name__
                            ),
                            "metrics_path":
                                str(metrics_path),
                        })
                        failed += 1
                        continue

                    identity_ok = (
                        result.get("dataset")
                        == dataset
                        and result.get("task")
                        == task
                        and result.get("predictor")
                        == predictor
                        and int(
                            result.get(
                                "seed",
                                -1,
                            )
                        )
                        == seed
                    )

                    if not identity_ok:
                        rows.append({
                            "dataset": dataset,
                            "task": task,
                            "predictor": predictor,
                            "seed": seed,
                            "status": "failed",
                            "reason":
                                "metrics_identity_mismatch",
                            "metrics_path":
                                str(metrics_path),
                        })
                        failed += 1
                        continue

                    status = str(
                        result.get(
                            "status",
                            "completed",
                        )
                    )

                    if status == "completed":
                        if key in structural_skips:
                            rows.append({
                                "dataset": dataset,
                                "task": task,
                                "predictor": predictor,
                                "seed": seed,
                                "status": "failed",
                                "reason": (
                                    "completed_but_declared_"
                                    "structural_skip"
                                ),
                                "metrics_path":
                                    str(metrics_path),
                            })
                            failed += 1
                            continue

                        completed += 1

                        rows.append({
                            "dataset": dataset,
                            "task": task,
                            "predictor": predictor,
                            "seed": seed,
                            "status":
                                "completed",
                            "reason": "",
                            "metrics_path":
                                str(metrics_path),
                        })

                        continue

                    if status == "skipped":
                        reason = str(
                            result.get(
                                "skip_reason",
                                "",
                            )
                        )

                        expected_reason = (
                            structural_skips.get(
                                key
                            )
                        )

                        if (
                            expected_reason is None
                            or reason
                            != expected_reason
                        ):
                            rows.append({
                                "dataset": dataset,
                                "task": task,
                                "predictor": predictor,
                                "seed": seed,
                                "status": "failed",
                                "reason": (
                                    "unexpected_skip:"
                                    + reason
                                ),
                                "metrics_path":
                                    str(metrics_path),
                            })
                            failed += 1
                            continue

                        skipped += 1

                        rows.append({
                            "dataset": dataset,
                            "task": task,
                            "predictor": predictor,
                            "seed": seed,
                            "status": "skipped",
                            "reason": reason,
                            "metrics_path":
                                str(metrics_path),
                        })

                        continue

                    rows.append({
                        "dataset": dataset,
                        "task": task,
                        "predictor": predictor,
                        "seed": seed,
                        "status": "failed",
                        "reason": (
                            "unknown_metrics_status:"
                            + status
                        ),
                        "metrics_path":
                            str(metrics_path),
                    })
                    failed += 1
                    continue

                if key in structural_skips:
                    skipped += 1

                    rows.append({
                        "dataset": dataset,
                        "task": task,
                        "predictor": predictor,
                        "seed": seed,
                        "status":
                            "structural_skip",
                        "reason":
                            structural_skips[key],
                        "metrics_path": "",
                    })

                else:
                    failed += 1

                    rows.append({
                        "dataset": dataset,
                        "task": task,
                        "predictor": predictor,
                        "seed": seed,
                        "status":
                            "missing",
                        "reason":
                            "missing_metrics_json",
                        "metrics_path": "",
                    })

    planned = len(expected_keys)

    print()
    print("PREDICTOR GENERALIZATION COMPLETENESS")
    print("=" * 72)
    print("TASKS =", len(tasks))
    print("PREDICTORS =", len(PREDICTORS))
    print("SEEDS =", len(SEEDS))
    print("PLANNED =", planned)
    print("COMPLETED =", completed)
    print("SKIPPED =", skipped)
    print("FAILED =", failed)

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dataset",
        "task",
        "predictor",
        "seed",
        "status",
        "reason",
        "metrics_path",
    ]

    with args.output_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("WROTE =", args.output_csv)

    if planned != 408:
        raise SystemExit(
            f"REFUSING: expected 408 planned runs, "
            f"got {planned}"
        )

    if completed + skipped != planned:
        raise SystemExit(
            "COMPLETENESS FAIL: "
            f"{completed}+{skipped}!={planned}"
        )

    if failed:
        raise SystemExit(
            f"COMPLETENESS FAIL: {failed} failures"
        )

    print()
    print(
        "COMPLETENESS PASS: "
        f"{completed} completed + "
        f"{skipped} structural skips "
        f"= {planned}/{planned}"
    )


if __name__ == "__main__":
    main()
