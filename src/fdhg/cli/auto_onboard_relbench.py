from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.onboarding.auto_relbench import (
    AutoOnboardingOptions,
    auto_onboard_relbench,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fully automatic leakage-safe RelBench onboarding."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-metadata-config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    download = parser.add_mutually_exclusive_group()
    download.add_argument("--download", action="store_true")
    download.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "Run train-only feature selection and write its artifacts "
            "without evaluating the official validation split."
        ),
    )
    parser.add_argument("--selection-folds", type=int, default=1)
    parser.add_argument("--feature-budget", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--selection-decoder",
        choices=("hist_gradient_boosting",),
        default="hist_gradient_boosting",
    )
    parser.add_argument("--max-relations", type=int, default=3)
    parser.add_argument("--max-numeric-columns", type=int, default=4)
    parser.add_argument("--max-categorical-columns", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        report = auto_onboard_relbench(
            dataset_name=args.dataset,
            task_name=args.task,
            output_root=Path(args.output_root),
            write=bool(args.write),
            overwrite=bool(args.overwrite),
            download=bool(args.download),
            task_metadata_config=(
                None
                if args.task_metadata_config is None
                else Path(args.task_metadata_config)
            ),
            options=AutoOnboardingOptions(
                selection_folds=args.selection_folds,
                feature_budget=args.feature_budget,
                min_delta=args.min_delta,
                selection_decoder=args.selection_decoder,
                max_relations=args.max_relations,
                max_numeric_columns=args.max_numeric_columns,
                max_categorical_columns=args.max_categorical_columns,
            ),
            evaluate_official_validation=not args.selection_only,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("DATASET", report.dataset)
    print("TASK", report.task)
    print("STATUS", report.status)
    print("TASK_TYPE", report.task_type or "")
    print("METRIC", report.metric or "")
    print("RELATION_CANDIDATES", report.relation_candidates)
    print(
        "SELECTED_RELATIONS",
        ",".join(report.selected_relations) if report.selected_relations else "",
    )
    print("CANDIDATE_FEATURES", report.candidate_features)
    print("SELECTED_FEATURES", report.selected_features)
    if report.inner_selection_score is not None:
        print(f"INNER_SELECTION_{str(report.metric or '').upper()}", report.inner_selection_score)
    if report.official_validation_score is not None:
        print(
            f"OFFICIAL_VALIDATION_{str(report.metric or '').upper()}",
            report.official_validation_score,
        )
    print("FALLBACK", report.fallback)
    workload = report.workload or {}
    if workload:
        print(
            "EXPECTED_CHILD_RELATION_SCANS",
            workload.get("child_relation_scan_count", ""),
        )
        print(
            "CANDIDATE_MATRIX_MATERIALIZATIONS",
            workload.get("candidate_matrix_materialization_count", ""),
        )
    print("TEST_SPLIT_ACCESSED", report.test_split_accessed)
    print("OUTPUT_DIR", report.output_dir)
    print("BLOCKERS", "|".join(report.blockers))
    return 0 if report.status in {"completed", "reused", "dry_run_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
