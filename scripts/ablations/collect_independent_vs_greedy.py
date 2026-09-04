#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def strategy_score(data, strategy):
    scores = data.get("mean_scores", {})
    selected = str(data.get("selected_variant", "")).lower()

    if strategy == "greedy":
        for key in ("auto_plus_fdhg", "auto_plus_fdhg_greedy"):
            if scores.get(key) is not None:
                return float(scores[key])

    if strategy == "independent":
        for key in ("auto_plus_fdhg", "auto_plus_fdhg_independent"):
            if scores.get(key) is not None:
                return float(scores[key])

    if "auto_only" in selected:
        value = scores.get("auto_only")
        if value is not None:
            return float(value)

    if data.get("selected_score") is not None:
        return float(data["selected_score"])

    return None


def selected_edge_ids(data):
    ids = data.get("strategy_selected_edge_ids", [])
    if not ids:
        ids = data.get("screened_in_edge_ids", [])
    return [str(x) for x in ids]


def classify(independent, greedy, direction, relative_tolerance, exact_tolerance):
    diff = greedy - independent

    if abs(diff) <= exact_tolerance:
        return "exact_tie"

    scale = max(abs(independent), abs(greedy), 1e-12)
    tolerance = relative_tolerance * scale

    if abs(diff) <= tolerance:
        return "within_tolerance"

    lower = direction in {"lower", "lower_is_better"}

    if lower:
        return "greedy_better" if greedy < independent else "independent_better"

    return "greedy_better" if greedy > independent else "independent_better"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--exact-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/ablations/independent-vs-greedy/summary.csv"
        ),
    )
    args = parser.parse_args()

    final_root = (
        args.artifact_root.resolve()
        / "outputs"
        / "final-gate-51task-v2"
    )

    if not final_root.exists():
        raise FileNotFoundError(final_root)

    rows = []

    for task_root in sorted(final_root.iterdir()):
        if not task_root.is_dir() or task_root.name.startswith("_"):
            continue

        slug = task_root.name

        independent_file = (
            task_root
            / "strategies"
            / "independent"
            / slug
            / "selected_variant.json"
        )

        greedy_file = (
            task_root
            / "strategies"
            / "greedy"
            / slug
            / "selected_variant.json"
        )

        independent = load_json(independent_file)
        greedy = load_json(greedy_file)

        if independent is None or greedy is None:
            continue

        independent_score = strategy_score(independent, "independent")
        greedy_score = strategy_score(greedy, "greedy")

        if independent_score is None or greedy_score is None:
            continue

        direction = (
            greedy.get("metric_direction")
            or independent.get("metric_direction")
        )

        outcome = classify(
            independent_score,
            greedy_score,
            direction,
            args.relative_tolerance,
            args.exact_tolerance,
        )

        independent_edges = selected_edge_ids(independent)
        greedy_edges = selected_edge_ids(greedy)

        edge_reduction = len(independent_edges) - len(greedy_edges)

        if edge_reduction > 0:
            sparsity = "greedy_sparser"
        elif edge_reduction < 0:
            sparsity = "independent_sparser"
        else:
            sparsity = "same_sparsity"

        rows.append({
            "task_key": slug,
            "metric": greedy.get("primary_metric") or independent.get("primary_metric"),
            "metric_direction": direction,
            "independent_score": independent_score,
            "greedy_score": greedy_score,
            "score_difference_greedy_minus_independent":
                greedy_score - independent_score,
            "outcome": outcome,
            "independent_variant": independent.get("selected_variant"),
            "greedy_variant": greedy.get("selected_variant"),
            "independent_edges": len(independent_edges),
            "greedy_edges": len(greedy_edges),
            "edge_reduction_independent_minus_greedy": edge_reduction,
            "sparsity_outcome": sparsity,
            "test_split_accessed_independent":
                independent.get("test_split_accessed"),
            "test_split_accessed_greedy":
                greedy.get("test_split_accessed"),
        })

    if not rows:
        raise RuntimeError(
            "No common Independent/Greedy strategy artifacts found."
        )

    df = pd.DataFrame(rows)

    # Table 9 common support:
    # both strategies must produce a non-empty FDHG augmentation.
    df = df[
        (df["independent_edges"] > 0)
        & (df["greedy_edges"] > 0)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No common-support tasks with non-empty "
            "Independent and Greedy FDHG selections."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    counts = df["outcome"].value_counts().to_dict()
    sparsity = df["sparsity_outcome"].value_counts().to_dict()

    reductions = df[
        "edge_reduction_independent_minus_greedy"
    ].to_numpy(dtype=float)

    print()
    print("===== INDEPENDENT VS GREEDY =====")
    print("COMMON SUPPORT:", len(df))
    print("GREEDY BETTER:", counts.get("greedy_better", 0))
    print("WITHIN TOLERANCE:", counts.get("within_tolerance", 0))
    print("EXACT TIE:", counts.get("exact_tie", 0))
    print("INDEPENDENT BETTER:", counts.get("independent_better", 0))

    print()
    print("GREEDY SPARSER:", sparsity.get("greedy_sparser", 0))
    print("SAME SPARSITY:", sparsity.get("same_sparsity", 0))
    print(
        "INDEPENDENT SPARSER:",
        sparsity.get("independent_sparser", 0),
    )
    print("MEDIAN EDGE REDUCTION:", float(np.median(reductions)))

    bad_test = df[
        (df["test_split_accessed_independent"] == True)
        | (df["test_split_accessed_greedy"] == True)
    ]

    print("TEST-SPLIT VIOLATIONS:", len(bad_test))

    print()
    print(
        df[
            [
                "task_key",
                "independent_score",
                "greedy_score",
                "outcome",
                "independent_edges",
                "greedy_edges",
                "sparsity_outcome",
            ]
        ].to_string(index=False)
    )

    print()
    print("WROTE:", args.output)


if __name__ == "__main__":
    main()
