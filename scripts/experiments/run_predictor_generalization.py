#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB = ROOT / ".fdhg" / "experiments.db"

DEFAULT_SEARCH_ROOTS = [
    ROOT / "outputs" / "final-gate-51task-v2",
]

DEFAULT_OUT_ROOT = (
    ROOT
    / "outputs"
    / "predictor-generalization"
)

SEEDS = [41, 42, 43, 44]
PREDICTORS = ["xgboost", "catboost"]


def load_completed(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """
        SELECT
            dataset,
            task,
            selected_variant,
            metric,
            output_dir,
            summary_json
        FROM runs
        WHERE status = 'completed'
          AND experiment_group = 'final-gate-51-task-v2'
        ORDER BY dataset, task
        """
    ).fetchall()

    con.close()

    # Defensive deduplication by dataset/task.
    unique = {}

    for row in rows:
        key = (
            row["dataset"],
            row["task"],
        )
        unique[key] = dict(row)

    return list(unique.values())


def find_joint(dataset, task, search_roots):
    slug = f"{dataset}_{task}"

    candidates = []

    for root in search_roots:
        path = (
            root
            / slug
            / "strategies"
            / "joint"
            / slug
            / "joint_selection.json"
        )

        if path.exists():
            candidates.append(
                path.resolve()
            )

    # Deduplicate worker/master symlink views.
    seen = set()
    unique = []

    for path in candidates:
        s = str(path)

        if s not in seen:
            seen.add(s)
            unique.append(path)

    if not unique:
        return None

    unique.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return unique[0]


def verify_joint(path):
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return None, (
            "invalid_joint_json:"
            + type(e).__name__
        )

    if (
        d.get("test_split_accessed")
        is not False
    ):
        return None, (
            "test_split_accessed_not_false"
        )

    if (
        d.get(
            "official_validation_was_used_for_selection"
        )
        is not False
    ):
        return None, (
            "official_validation_used_for_selection_not_false"
        )

    if (
        d.get("same_candidate_pool_verified")
        is not True
    ):
        return None, (
            "same_candidate_pool_verified_not_true"
        )

    if not d.get("selected_variant"):
        return None, "missing_selected_variant"

    return d, None


def task_root_from_joint(path):
    # .../<task>/strategies/joint/<slug>/joint_selection.json
    return path.parents[3]


def selected_root(task_root):
    return (
        task_root
        / "strategies"
        / "selected"
    )


def parquet_inventory(root):
    if not root.exists():
        return []

    files = []

    for p in root.rglob("*.parquet"):

        lower = p.name.lower()

        # Critical: downstream preparation must never
        # accidentally pick up a test artifact.
        if "test" in lower:
            continue

        files.append(p)

    return sorted(files)


def classify_parquet(path):
    name = path.name.lower()

    if "train" in name:
        return "train"

    if (
        "val" in name
        or "valid" in name
        or "validation" in name
    ):
        return "val"

    return "other"


def candidate_pairs(files):
    trains = [
        p for p in files
        if classify_parquet(p) == "train"
    ]

    vals = [
        p for p in files
        if classify_parquet(p) == "val"
    ]

    pairs = []

    for train in trains:
        for val in vals:

            # Prefer same parent directory.
            if train.parent == val.parent:
                pairs.append(
                    (train, val, "same_parent")
                )

    # If no exact same-parent pair exists,
    # keep inventory visible but refuse to guess.
    return pairs


def find_metadata(task_root):
    candidates = [
        task_root
        / "pipeline"
        / "resolved_task_metadata.yaml",

        task_root
        / "pipeline"
        / "resolved_task_metadata.yml",

        task_root
        / "resolved_task_metadata.yaml",

        task_root
        / "resolved_task_metadata.yml",
    ]

    for p in candidates:
        if p.exists():
            return p

    found = list(
        task_root.rglob(
            "resolved_task_metadata.yaml"
        )
    )

    return (
        found[0]
        if len(found) == 1
        else None
    )


def inspect_metadata(path):
    if path is None:
        return {
            "label_col": "",
            "problem_type": "",
            "primary_metric": "",
        }

    try:
        import yaml

        d = yaml.safe_load(
            path.read_text()
        ) or {}

    except Exception:
        return {
            "label_col": "",
            "problem_type": "",
            "primary_metric": "",
        }

    return {
        "label_col": str(
            d.get("label_col")
            or d.get("target_col")
            or ""
        ),

        "problem_type": str(
            d.get("problem_type")
            or d.get("task_type")
            or ""
        ),

        "primary_metric": str(
            d.get("primary_metric")
            or ""
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--expect-completed",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--write-manifest",
        action="store_true",
    )

    ap.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=(
            "Experiment registry SQLite database "
            "(default: .fdhg/experiments.db)"
        ),
    )

    ap.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Final-gate artifact root containing per-task outputs. "
            "May be specified multiple times."
        ),
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=(
            "Directory for predictor-generalization artifacts."
        ),
    )

    args = ap.parse_args()

    search_roots = (
        args.search_root
        if args.search_root
        else DEFAULT_SEARCH_ROOTS
    )

    rows = load_completed(args.db)

    print()
    print(
        "PREDICTOR GENERALIZATION "
        "PRE-FLIGHT"
    )
    print("=" * 128)

    print(
        f"SQLITE_COMPLETED_TASKS={len(rows)}"
    )

    if (
        args.expect_completed is not None
        and len(rows)
        != args.expect_completed
    ):
        raise SystemExit(
            "REFUSING: expected "
            f"{args.expect_completed} "
            "completed tasks, found "
            f"{len(rows)}"
        )

    manifest_rows = []
    problems = []

    for row in rows:

        dataset = row["dataset"]
        task = row["task"]

        joint_path = find_joint(
            dataset,
            task,
            search_roots,
        )

        if joint_path is None:
            problems.append(
                (
                    dataset,
                    task,
                    "joint_selection_not_found",
                )
            )
            continue

        joint, error = verify_joint(
            joint_path
        )

        if error:
            problems.append(
                (
                    dataset,
                    task,
                    error,
                )
            )
            continue

        task_root = task_root_from_joint(
            joint_path
        )

        sel_root = selected_root(
            task_root
        )

        files = parquet_inventory(
            sel_root
        )

        pairs = candidate_pairs(
            files
        )

        metadata_path = find_metadata(
            task_root
        )

        meta = inspect_metadata(
            metadata_path
        )

        selected_variant = joint.get(
            "selected_variant"
        )

        problem = ""

        if not sel_root.exists():
            problem = (
                "selected_root_missing"
            )

        elif len(pairs) == 0:
            problem = (
                "no_unambiguous_train_val_pair"
            )

        elif len(pairs) > 1:
            problem = (
                "multiple_train_val_pairs"
            )

        elif not meta["label_col"]:
            problem = (
                "label_col_unresolved"
            )

        elif not meta["problem_type"]:
            problem = (
                "problem_type_unresolved"
            )

        if problem:
            problems.append(
                (
                    dataset,
                    task,
                    problem,
                )
            )

        train_path = ""
        val_path = ""

        if len(pairs) == 1:
            train_path = str(
                pairs[0][0]
            )

            val_path = str(
                pairs[0][1]
            )

        manifest_rows.append({
            "dataset":
                dataset,

            "task":
                task,

            "selected_variant":
                selected_variant,

            "selected_source_strategy":
                joint.get(
                    "selected_source_strategy",
                    "",
                ),

            "metric":
                joint.get(
                    "primary_metric",
                    row.get("metric") or "",
                ),

            "problem_type":
                meta["problem_type"],

            "label_col":
                meta["label_col"],

            "joint_selection":
                str(joint_path),

            "task_root":
                str(task_root),

            "selected_root":
                str(sel_root),

            "train_parquet":
                train_path,

            "val_parquet":
                val_path,

            "parquet_count":
                len(files),

            "pair_count":
                len(pairs),

            "metadata_path":
                (
                    str(metadata_path)
                    if metadata_path
                    else ""
                ),

            "preflight_status":
                (
                    "READY"
                    if not problem
                    else problem
                ),
        })

    print()
    print(
        f"{'DATASET/TASK':47} "
        f"{'SELECTED VARIANT':30} "
        f"{'TYPE':14} "
        f"STATUS"
    )
    print("-" * 128)

    for r in manifest_rows:

        target = (
            f"{r['dataset']}/"
            f"{r['task']}"
        )

        print(
            f"{target[:47]:47} "
            f"{str(r['selected_variant'])[:30]:30} "
            f"{str(r['problem_type'])[:14]:14} "
            f"{r['preflight_status']}"
        )

    ready = [
        r
        for r in manifest_rows
        if r["preflight_status"]
        == "READY"
    ]

    print()
    print("=" * 128)

    print(
        f"TOTAL={len(rows)} "
        f"READY={len(ready)} "
        f"PROBLEMS={len(problems)}"
    )

    if problems:
        print()
        print("PROBLEMS")
        print("-" * 128)

        for dataset, task, reason in problems:
            print(
                f"{dataset}/{task}: "
                f"{reason}"
            )

    if args.write_manifest:

        args.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            args.output_root
            / "evaluation_manifest.csv"
        )

        fieldnames = [
            "dataset",
            "task",
            "selected_variant",
            "selected_source_strategy",
            "metric",
            "problem_type",
            "label_col",
            "joint_selection",
            "task_root",
            "selected_root",
            "train_parquet",
            "val_parquet",
            "parquet_count",
            "pair_count",
            "metadata_path",
            "preflight_status",
        ]

        with manifest_path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                manifest_rows
            )

        print()
        print(
            f"WROTE_MANIFEST={manifest_path}"
        )

    # Important:
    # This V1 intentionally has NO execute mode.
    # It cannot launch XGBoost/CatBoost.
    print()
    print(
        "EXECUTION_ENABLED=0 "
        "(pre-flight only)"
    )


if __name__ == "__main__":
    main()
