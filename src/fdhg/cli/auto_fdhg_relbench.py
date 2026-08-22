from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from fdhg.onboarding.auto_fdhg import AutoFdhgOptions, auto_fdhg_relbench


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validation-gated Auto+FDHG RelBench compiler."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--auto-output-root", default="outputs/auto-onboarding-3fold")
    parser.add_argument("--dfs-feature-config")
    parser.add_argument("--canonical-onboarding-root", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    download = parser.add_mutually_exclusive_group()
    download.add_argument("--download", action="store_true")
    download.add_argument("--no-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument("--feature-budget", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--selection-decoder",
        choices=("hist_gradient_boosting",),
        default="hist_gradient_boosting",
    )
    parser.add_argument("--max-fdhg-edges", type=int, default=4)
    parser.add_argument("--max-selected-fdhg-edges", type=int)
    parser.add_argument("--max-relations", type=int, default=3)
    parser.add_argument("--max-numeric-columns", type=int, default=4)
    parser.add_argument("--max-categorical-columns", type=int, default=4)
    screening = parser.add_mutually_exclusive_group()
    screening.add_argument("--enable-edge-screening", dest="enable_edge_screening", action="store_true", default=True)
    screening.add_argument("--disable-edge-screening", dest="enable_edge_screening", action="store_false")
    parser.add_argument("--edge-screening-min-delta", type=float, default=0.0)
    parser.add_argument("--edge-screening-min-positive-folds", type=int)
    parser.add_argument(
        "--edge-screening-rule",
        choices=("fixed_count", "positive_fraction", "pooled_oof"),
        default="fixed_count",
    )
    parser.add_argument("--edge-screening-min-positive-fraction", type=float, default=2 / 3)
    parser.add_argument("--edge-screening-max-relative-fold-degradation", type=float)
    parser.add_argument(
        "--edge-selection-strategy",
        choices=("independent", "greedy", "greedy_backward"),
        default="independent",
    )
    parser.add_argument(
        "--continuous-fdhg-mode",
        choices=("exclude", "quantile"),
        default="exclude",
    )
    parser.add_argument("--continuous-fdhg-bins", type=int, default=8)
    parser.add_argument("--continuous-fdhg-min-effective-bins", type=int, default=2)
    parser.add_argument("--fdhg-candidate-edges-file", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if args.edge_screening_min_positive_folds is not None:
        if args.edge_screening_min_positive_folds < 1:
            parser.error("--edge-screening-min-positive-folds must be at least 1")
        if args.edge_screening_min_positive_folds > args.selection_folds:
            parser.error("--edge-screening-min-positive-folds cannot exceed --selection-folds")
    if args.max_selected_fdhg_edges is not None and args.max_selected_fdhg_edges < 0:
        parser.error("--max-selected-fdhg-edges must be non-negative")
    if not (0.0 < args.edge_screening_min_positive_fraction <= 1.0):
        parser.error("--edge-screening-min-positive-fraction must be in (0, 1]")
    if (
        args.edge_screening_max_relative_fold_degradation is not None
        and args.edge_screening_max_relative_fold_degradation < 0
    ):
        parser.error("--edge-screening-max-relative-fold-degradation must be non-negative")
    if args.continuous_fdhg_bins < 1:
        parser.error("--continuous-fdhg-bins must be positive")
    if args.continuous_fdhg_min_effective_bins < 1:
        parser.error("--continuous-fdhg-min-effective-bins must be positive")
    try:
        report = auto_fdhg_relbench(
            dataset_name=args.dataset,
            task_name=args.task,
            output_root=Path(args.output_root),
            write=bool(args.write),
            overwrite=bool(args.overwrite),
            download=bool(args.download),
            auto_output_root=Path(args.auto_output_root),
            dfs_source_root=(
                Path(args.canonical_onboarding_root)
                if args.canonical_onboarding_root
                else Path(".")
            ),
            dfs_feature_config=(
                None if args.dfs_feature_config is None else Path(args.dfs_feature_config)
            ),
            options=AutoFdhgOptions(
                selection_folds=args.selection_folds,
                feature_budget=args.feature_budget,
                min_delta=args.min_delta,
                selection_decoder=args.selection_decoder,
                max_fdhg_edges=args.max_fdhg_edges,
                max_selected_fdhg_edges=args.max_selected_fdhg_edges,
                max_relations=args.max_relations,
                max_numeric_columns=args.max_numeric_columns,
                max_categorical_columns=args.max_categorical_columns,
                enable_edge_screening=args.enable_edge_screening,
                edge_screening_min_delta=args.edge_screening_min_delta,
                edge_screening_min_positive_folds=args.edge_screening_min_positive_folds,
                edge_screening_rule=args.edge_screening_rule,
                edge_screening_min_positive_fraction=args.edge_screening_min_positive_fraction,
                edge_screening_max_relative_fold_degradation=args.edge_screening_max_relative_fold_degradation,
                edge_selection_strategy=args.edge_selection_strategy,
                continuous_fdhg_mode=args.continuous_fdhg_mode,
                continuous_fdhg_bins=args.continuous_fdhg_bins,
                continuous_fdhg_min_effective_bins=args.continuous_fdhg_min_effective_bins,
                fdhg_candidate_edges_file=args.fdhg_candidate_edges_file,
            ),
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        if args.debug:
            traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("DATASET", report.dataset)
    print("TASK", report.task)
    print("STATUS", report.status)
    print("SELECTED_VARIANT", report.selected_variant or "")
    print("METRIC", report.metric or "")
    print("METRIC_DIRECTION", report.metric_direction or "")
    for variant, score in (report.mean_scores or {}).items():
        print(f"INNER_MEAN_{variant.upper()}", score)
    if report.official_validation_score is not None:
        print("OFFICIAL_VALIDATION_SCORE", report.official_validation_score)
    print("DFS_FEATURES", report.dfs_features)
    print("DFS_DECLARATIONS", report.dfs_declarations)
    print("DFS_MODEL_COLUMNS", report.dfs_model_columns)
    print("AUTO_FEATURES", report.auto_features)
    print("FDHG_RESIDUAL_FEATURES", report.fdhg_features)
    print("FDHG_DECLARED_RESIDUAL_FEATURES", report.fdhg_declared_residual_features)
    print(
        "FDHG_USABLE_RESIDUAL_FEATURES_BY_FOLD",
        report.fdhg_usable_residual_features_by_fold or {},
    )
    print("FDHG_FINAL_REFIT_USABLE_FEATURES", report.fdhg_final_refit_usable_features)
    print("CANDIDATE_FDHG_EDGES", report.candidate_edges)
    print("SCREENED_IN_FDHG_EDGES", report.screened_in_edges)
    print("SCREENED_OUT_FDHG_EDGES", report.screened_out_edges)
    print("ACCEPTED_FDHG_EDGES", report.accepted_edges)
    print("EXPECTED_SCANS", report.expected_scans)
    print("EXPECTED_MATERIALIZATIONS", report.expected_materializations)
    print("TEST_SPLIT_ACCESSED", report.test_split_accessed)
    print("OUTPUT_DIR", report.output_dir)
    print("BLOCKERS", "|".join(report.blockers))
    return 0 if report.status in {"completed", "reused", "dry_run_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
