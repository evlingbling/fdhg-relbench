from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/ablations/cross-fold-consistency"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/ablations/cross-fold-consistency/"
            "cross_fold_consistency_summary.csv"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    rows = []

    for p in sorted(args.input_root.rglob("manifest.json")):
        match = re.search(r"positive_folds_(\d+)", str(p))
        if match is None:
            continue

        positive_folds = int(match.group(1))
        m = json.loads(p.read_text())

        scores = m.get("mean_scores", {})

        rows.append(
            {
                "dataset": m.get("dataset"),
                "task": m.get("task"),
                "metric": m.get("metric"),
                "metric_direction": m.get("metric_direction"),
                "positive_folds": positive_folds,
                "selection_folds": 3,
                "inner_mean_dfs": scores.get("dfs_fallback"),
                "inner_mean_auto": scores.get("auto_only"),
                "inner_mean_auto_plus_fdhg": scores.get(
                    "auto_plus_fdhg"
                ),
                "selected_variant": m.get("selected_variant"),
                "candidate_edges": m.get(
                    "candidate_fdhg_edge_count"
                ),
                "selected_edges": m.get(
                    "strategy_selected_edge_count",
                    m.get("selected_screened_edge_count"),
                ),
                "screened_out_edges": m.get(
                    "screened_out_fdhg_edge_count"
                ),
                "official_validation_score": m.get(
                    "official_validation_score"
                ),
                "test_split_accessed": bool(
                    m.get("test_split_accessed", False)
                ),
                "official_validation_used_for_selection": bool(
                    m.get(
                        "official_validation_was_used_for_selection",
                        False,
                    )
                ),
            }
        )

    if not rows:
        raise SystemExit("no cross-fold manifests found")

    df = pd.DataFrame(rows).sort_values(
        ["dataset", "task", "positive_folds"]
    )

    if len(df) != 6:
        raise SystemExit(
            f"expected 6 runs, found {len(df)}"
        )

    expected = {
        ("rel-trial", "studies-enrollment", 1),
        ("rel-trial", "studies-enrollment", 2),
        ("rel-trial", "studies-enrollment", 3),
        ("rel-trial", "study-outcome", 1),
        ("rel-trial", "study-outcome", 2),
        ("rel-trial", "study-outcome", 3),
    }
    observed = set(
        df[["dataset", "task", "positive_folds"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected:
        raise SystemExit(
            "cross-fold run set mismatch:\n"
            f"missing={sorted(expected - observed)}\n"
            f"unexpected={sorted(observed - expected)}"
        )

    if df["test_split_accessed"].eq(True).any():
        raise SystemExit("test split access detected")

    if df[
        "official_validation_used_for_selection"
    ].eq(True).any():
        raise SystemExit(
            "official validation used for selection"
        )

    required = [
        "candidate_edges",
        "selected_edges",
        "screened_out_edges",
    ]
    if df[required].isna().any().any():
        bad = df[df[required].isna().any(axis=1)]
        raise SystemExit(
            "missing edge-count fields:\n"
            + bad.to_string(index=False)
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print()
    print("rows =", len(df))
    print(
        "all test_split_accessed=False =",
        not df["test_split_accessed"].eq(True).any(),
    )
    print(
        "all official_validation_used_for_selection=False =",
        not df[
            "official_validation_used_for_selection"
        ].eq(True).any(),
    )
    print("wrote:", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
