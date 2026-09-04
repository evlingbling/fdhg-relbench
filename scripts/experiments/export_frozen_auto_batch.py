#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FINAL_GATE_ROOT = Path(
    "outputs/final-gate-51task-v2"
)

DEFAULT_SELECTED_ROOT = Path(
    "outputs/predictor-generalization/frozen-matrices"
)

DEFAULT_AUTO_ROOT = Path(
    "outputs/predictor-generalization/auto-frozen-matrices"
)

EXPORTER = (
    ROOT
    / "scripts"
    / "experiments"
    / "export_frozen_auto_matrices.py"
)

AUTO_VARIANTS = {
    "auto",
    "auto_only",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def counterfactual_tasks(selected_root: Path):
    manifests = sorted(
        selected_root.glob("*/manifest.json")
    )

    if len(manifests) != 51:
        raise RuntimeError(
            f"Expected 51 selected frozen manifests, "
            f"found {len(manifests)}"
        )

    rows = []

    for mp in manifests:
        m = load_json(mp)

        variant = str(
            m["selected_variant"]
        )

        if variant.lower() in AUTO_VARIANTS:
            continue

        rows.append({
            "dataset": str(m["dataset"]),
            "task": str(m["task"]),
            "selected_variant": variant,
            "slug": mp.parent.name,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--final-gate-root",
        type=Path,
        default=DEFAULT_FINAL_GATE_ROOT,
    )

    ap.add_argument(
        "--selected-root",
        type=Path,
        default=DEFAULT_SELECTED_ROOT,
    )

    ap.add_argument(
        "--auto-root",
        type=Path,
        default=DEFAULT_AUTO_ROOT,
    )

    ap.add_argument(
        "--expect-counterfactuals",
        type=int,
        default=18,
    )

    ap.add_argument(
        "--write",
        action="store_true",
    )

    args = ap.parse_args()

    tasks = counterfactual_tasks(
        args.selected_root
    )

    print("=" * 100)
    print("AUTO COUNTERFACTUAL FROZEN MATRIX EXPORT")
    print("=" * 100)
    print("SELECTED_ROOT:", args.selected_root)
    print("AUTO_ROOT:", args.auto_root)
    print("TASKS:", len(tasks))
    print()

    if (
        len(tasks)
        != args.expect_counterfactuals
    ):
        raise RuntimeError(
            f"Expected {args.expect_counterfactuals} "
            f"non-Auto tasks, found {len(tasks)}"
        )

    for i, row in enumerate(tasks, 1):
        print(
            f"[{i:02d}/{len(tasks):02d}] "
            f"{row['dataset']}/{row['task']} "
            f"source={row['selected_variant']}"
        )

    if not args.write:
        print()
        print(
            "WRITE_ENABLED=0 "
            "(inventory validated; no matrices exported)"
        )
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(
        ROOT / "src"
    )

    completed = 0

    for i, row in enumerate(tasks, 1):
        print()
        print("=" * 100)
        print(
            f"EXPORT [{i:02d}/{len(tasks):02d}] "
            f"{row['dataset']}/{row['task']}"
        )
        print("=" * 100)

        cmd = [
            sys.executable,
            str(EXPORTER),
            "--output-root",
            str(args.final_gate_root),
            "--export-root",
            str(args.auto_root),
            "--dataset",
            row["dataset"],
            "--task",
            row["task"],
            "--write",
        ]

        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "Auto counterfactual export failed: "
                f"{row['dataset']}/{row['task']}"
            )

        manifest_path = (
            args.auto_root
            / row["slug"]
            / "manifest.json"
        )

        if not manifest_path.exists():
            raise RuntimeError(
                f"Missing exported manifest: {manifest_path}"
            )

        m = load_json(manifest_path)

        if str(
            m.get("selected_variant", "")
        ) != "auto_only":
            raise RuntimeError(
                "Counterfactual manifest is not auto_only: "
                f"{manifest_path}"
            )

        if (
            m.get("test_split_accessed")
            is not False
        ):
            raise RuntimeError(
                "Test split violation: "
                f"{manifest_path}"
            )

        completed += 1

    print()
    print("=" * 100)
    print("AUTO COUNTERFACTUAL EXPORT COMPLETE")
    print("COMPLETED:", completed)
    print("=" * 100)


if __name__ == "__main__":
    main()
