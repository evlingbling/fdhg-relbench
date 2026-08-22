#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def valid_existing(
    root: Path,
    dataset: str,
    task: str,
    selected_variant: str,
) -> bool:
    slug = f"{dataset}_{task}"
    p = root / slug

    manifest = p / "manifest.json"
    train = p / "train.parquet"
    val = p / "val.parquet"

    if not (
        manifest.exists()
        and train.exists()
        and val.exists()
    ):
        return False

    try:
        m = json.loads(manifest.read_text())
    except Exception:
        return False

    return (
        m.get("dataset") == dataset
        and m.get("task") == task
        and m.get("selected_variant") == selected_variant
        and m.get("test_split_accessed") is False
        and (
            m.get(
                "official_validation_used_for_selection"
            )
            is False
        )
        and (
            m.get(
                "same_candidate_pool_verified"
            )
            is True
        )
        and int(
            m.get(
                "model_feature_count",
                0,
            )
        ) > 0
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "outputs/predictor-generalization/"
            "frozen-36-task-snapshot.csv"
        ),
    )

    ap.add_argument(
        "--export-root",
        type=Path,
        default=Path(
            "outputs/predictor-generalization/"
            "frozen-matrices"
        ),
    )

    ap.add_argument(
        "--write",
        action="store_true",
    )

    args = ap.parse_args()

    with args.snapshot.open() as f:
        rows = list(csv.DictReader(f))

    print("SNAPSHOT_ROWS =", len(rows))

    if len(rows) != 36:
        raise SystemExit(
            f"REFUSING: expected 36 rows, "
            f"found {len(rows)}"
        )

    todo = []

    for row in rows:
        dataset = row["dataset"]
        task = row["task"]

        if valid_existing(
            args.export_root,
            dataset,
            task,
            row["selected_variant"],
        ):
            print(
                "READY_EXISTING",
                f"{dataset}/{task}",
            )
        else:
            todo.append(
                (dataset, task)
            )
            print(
                "NEEDS_EXPORT",
                f"{dataset}/{task}",
            )

    print()
    print("EXISTING_READY =", len(rows) - len(todo))
    print("NEEDS_EXPORT =", len(todo))

    if not args.write:
        print(
            "WRITE_ENABLED=0 "
            "(batch preflight only)"
        )
        return

    failures = []

    for index, (dataset, task) in enumerate(
        todo,
        start=1,
    ):
        print()
        print("=" * 100)
        print(
            f"[{index}/{len(todo)}] "
            f"{dataset}/{task}"
        )

        cmd = [
            sys.executable,
            "scripts/experiments/"
            "export_frozen_selected_matrices.py",
            "--dataset",
            dataset,
            "--task",
            task,
            "--export-root",
            str(args.export_root),
            "--write",
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            failures.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "returncode":
                        result.returncode,
                }
            )

            print(
                "FAILED",
                dataset,
                task,
                result.returncode,
            )

    print()
    print("=" * 100)
    print("BATCH FINISHED")
    print("FAILURES =", len(failures))

    if failures:
        for item in failures:
            print(item)

        raise SystemExit(1)


if __name__ == "__main__":
    main()
