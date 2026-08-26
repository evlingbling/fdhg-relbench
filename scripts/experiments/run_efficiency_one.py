from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fdhg.onboarding.auto_fdhg import (
    AutoFdhgOptions,
    auto_fdhg_relbench,
)


METHODS = {
    "dfs",
    "auto",
    "all",
    "independent",
    "greedy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--auto-root", required=True, type=Path)
    p.add_argument("--canonical-root", required=True, type=Path)
    p.add_argument("--candidate-file", required=True, type=Path)
    p.add_argument(
        "--feature-budget",
        type=int,
        default=32,
        help="Auto feature budget. Defaults to 32 for efficiency benchmarking.",
    )
    p.add_argument(
        "--positive-folds",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Minimum number of positive inner folds required for FDHG screening.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    common = dict(
        selection_folds=3,
        feature_budget=args.feature_budget,
        min_delta=0.0,
        selection_decoder="hist_gradient_boosting",
        max_relations=3,
        max_numeric_columns=4,
        max_categorical_columns=6,
        edge_screening_min_delta=0.0,
        edge_screening_min_positive_folds=args.positive_folds,
        edge_screening_rule="fixed_count",
        continuous_fdhg_mode="exclude",
    )

    if args.method == "dfs":
        options = AutoFdhgOptions(
            **common,
            max_fdhg_edges=0,
            max_selected_fdhg_edges=0,
            discover_fdhg_edges=False,
            enable_edge_screening=False,
            edge_selection_strategy="independent",
            fdhg_candidate_edges_file=None,
            force_final_variant="dfs_fallback",
        )

    elif args.method == "auto":
        options = AutoFdhgOptions(
            **common,
            max_fdhg_edges=0,
            max_selected_fdhg_edges=0,
            discover_fdhg_edges=False,
            enable_edge_screening=False,
            edge_selection_strategy="independent",
            fdhg_candidate_edges_file=None,
            force_final_variant="auto_only",
        )

    elif args.method == "all":
        options = AutoFdhgOptions(
            **common,
            max_fdhg_edges=32,
            max_selected_fdhg_edges=32,
            discover_fdhg_edges=False,
            enable_edge_screening=False,
            edge_selection_strategy="independent",
            fdhg_candidate_edges_file=args.candidate_file,
            force_final_variant="auto_plus_fdhg",
        )

    elif args.method == "independent":
        options = AutoFdhgOptions(
            **common,
            max_fdhg_edges=32,
            max_selected_fdhg_edges=32,
            discover_fdhg_edges=False,
            enable_edge_screening=True,
            edge_selection_strategy="independent",
            fdhg_candidate_edges_file=args.candidate_file,
            force_final_variant="auto_plus_fdhg",
        )

    elif args.method == "greedy":
        options = AutoFdhgOptions(
            **common,
            max_fdhg_edges=32,
            max_selected_fdhg_edges=32,
            discover_fdhg_edges=False,
            enable_edge_screening=True,
            edge_selection_strategy="greedy",
            fdhg_candidate_edges_file=args.candidate_file,
            force_final_variant="auto_plus_fdhg",
        )

    else:
        raise AssertionError(args.method)

    report = auto_fdhg_relbench(
        dataset_name=args.dataset,
        task_name=args.task,
        output_root=args.output_root,
        auto_output_root=args.auto_root,
        dfs_source_root=args.canonical_root,
        write=True,
        overwrite=True,
        download=False,
        options=options,
    )

    payload = {
        "dataset": args.dataset,
        "task": args.task,
        "method": args.method,
        "status": report.status,
        "selected_variant": report.selected_variant,
        "metric": report.metric,
        "metric_direction": report.metric_direction,
        "official_validation_score": report.official_validation_score,
        "dfs_features": report.dfs_features,
        "dfs_model_columns": report.dfs_model_columns,
        "auto_features": report.auto_features,
        "candidate_edges": report.candidate_edges,
        "accepted_edges": report.accepted_edges,
        "screened_in_edges": report.screened_in_edges,
        "fdhg_final_refit_usable_features": (
            report.fdhg_final_refit_usable_features
        ),
        "test_split_accessed": report.test_split_accessed,
        "output_dir": str(report.output_dir),
        "blockers": list(report.blockers),
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if report.status not in {"completed", "reused"}:
        return 2
    if report.test_split_accessed:
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
