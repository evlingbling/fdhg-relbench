#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MANIFEST = (
    ROOT
    / "outputs"
    / "predictor-generalization"
    / "evaluation_manifest.csv"
)

DEFAULT_BUNDLE = (
    ROOT
    / "outputs"
    / "predictor-generalization"
    / "local-bundle"
)


def copy_checked(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(src)

    if "test" in src.name.lower():
        raise RuntimeError(
            f"REFUSING TEST ARTIFACT: {src}"
        )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    ap.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE,
    )

    ap.add_argument(
        "--expect-ready",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    ready = [
        row
        for row in rows
        if row["preflight_status"] == "READY"
    ]

    print()
    print("LOCAL PREDICTOR-EVAL BUNDLE")
    print("=" * 100)
    print(f"MANIFEST_ROWS={len(rows)}")
    print(f"READY_ROWS={len(ready)}")

    if (
        args.expect_ready is not None
        and len(ready) != args.expect_ready
    ):
        raise SystemExit(
            "REFUSING EXPORT: expected "
            f"{args.expect_ready} READY tasks, "
            f"found {len(ready)}"
        )

    bundle = args.bundle_root
    bundle.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_rows = []

    for row in ready:
        dataset = row["dataset"]
        task = row["task"]

        slug = f"{dataset}__{task}"

        dst_root = (
            bundle
            / "tasks"
            / slug
        )

        train_src = Path(
            row["train_parquet"]
        )

        val_src = Path(
            row["val_parquet"]
        )

        joint_src = Path(
            row["joint_selection"]
        )

        metadata_src = (
            Path(row["metadata_path"])
            if row["metadata_path"]
            else None
        )

        train_dst = (
            dst_root / "train.parquet"
        )

        val_dst = (
            dst_root / "val.parquet"
        )

        joint_dst = (
            dst_root
            / "joint_selection.json"
        )

        metadata_dst = (
            dst_root
            / "resolved_task_metadata.yaml"
        )

        copy_checked(
            train_src,
            train_dst,
        )

        copy_checked(
            val_src,
            val_dst,
        )

        copy_checked(
            joint_src,
            joint_dst,
        )

        metadata_rel = ""

        if metadata_src is not None:
            copy_checked(
                metadata_src,
                metadata_dst,
            )

            metadata_rel = str(
                metadata_dst.relative_to(
                    bundle
                )
            )

        out_rows.append({
            "dataset": dataset,
            "task": task,

            "selected_variant":
                row["selected_variant"],

            "metric":
                row["metric"],

            "problem_type":
                row["problem_type"],

            "label_col":
                row["label_col"],

            "train_parquet":
                str(
                    train_dst.relative_to(
                        bundle
                    )
                ),

            "val_parquet":
                str(
                    val_dst.relative_to(
                        bundle
                    )
                ),

            "joint_selection":
                str(
                    joint_dst.relative_to(
                        bundle
                    )
                ),

            "metadata_path":
                metadata_rel,
        })

        print(
            f"EXPORTED {dataset}/{task}"
        )

    manifest_out = (
        bundle / "bundle_manifest.csv"
    )

    fieldnames = [
        "dataset",
        "task",
        "selected_variant",
        "metric",
        "problem_type",
        "label_col",
        "train_parquet",
        "val_parquet",
        "joint_selection",
        "metadata_path",
    ]

    with manifest_out.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(out_rows)

    print()
    print(f"BUNDLE_ROOT={bundle}")
    print(f"BUNDLE_TASKS={len(out_rows)}")
    print(f"BUNDLE_MANIFEST={manifest_out}")


if __name__ == "__main__":
    main()
