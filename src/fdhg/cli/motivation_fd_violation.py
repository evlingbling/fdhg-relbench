from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from fdhg.onboarding.motivation_fd_violation import (
    FdViolationOptions,
    motivation_fd_violation,
)


def _parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Controlled FD violation experiment for motivation analysis."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--edge", required=True, help="Edge spec like badges:Class->TagBased")
    parser.add_argument("--output-root", default="results/motivation_fd_violation")
    parser.add_argument("--auto-output-root", default="outputs/auto-onboarding-3fold")
    parser.add_argument("--dfs-feature-config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    download = parser.add_mutually_exclusive_group()
    download.add_argument("--download", action="store_true")
    download.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument("--corruption-levels", default="0.0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--corruption-seeds", default="41,42,43,44")
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
        report = motivation_fd_violation(
            dataset_name=args.dataset,
            task_name=args.task,
            edge_spec=args.edge,
            output_root=Path(args.output_root),
            write=bool(args.write),
            overwrite=bool(args.overwrite),
            download=bool(args.download),
            auto_output_root=Path(args.auto_output_root),
            dfs_source_root=Path("."),
            dfs_feature_config=None if args.dfs_feature_config is None else Path(args.dfs_feature_config),
            options=FdViolationOptions(
                selection_folds=args.selection_folds,
                feature_budget=args.feature_budget,
                min_delta=args.min_delta,
                selection_decoder=args.selection_decoder,
                max_relations=args.max_relations,
                max_numeric_columns=args.max_numeric_columns,
                max_categorical_columns=args.max_categorical_columns,
                corruption_levels=_parse_float_list(args.corruption_levels),
                corruption_seeds=_parse_int_list(args.corruption_seeds),
            ),
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("DATASET", report.dataset)
    print("TASK", report.task)
    print("STATUS", report.status)
    print("FOLD_ROWS", report.fold_rows)
    print("AGGREGATE_ROWS", report.aggregate_rows)
    print("FOLD_CSV", report.fold_csv or "")
    print("AGGREGATE_CSV", report.aggregate_csv or "")
    print("FIGURE_AGGREGATE_CSV", report.figure_aggregate_csv or "")
    print("TEST_SPLIT_ACCESSED", report.test_split_accessed)
    print("OUTPUT_DIR", report.output_dir)
    print("BLOCKERS", "|".join(report.blockers))
    return 0 if report.status in {"completed", "dry_run_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
