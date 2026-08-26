from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASKS = [
    ("rel-event", "user-attendance"),
    ("rel-f1", "driver-position"),
    ("rel-trial", "studies-enrollment"),
    ("rel-trial", "study-outcome"),
]

BUDGETS = [4, 8, 12, 16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/ablations/auto-budget"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/ablations/auto-budget/"
            "auto_budget_summary.csv"
        ),
    )
    return p.parse_args()


def direction_aware_gain(
    auto_score: float,
    greedy_score: float,
    direction: str,
) -> float:
    if direction == "higher":
        return greedy_score - auto_score
    if direction == "lower":
        return auto_score - greedy_score
    raise ValueError(f"unsupported metric direction: {direction}")


def load_manifest(
    root: Path,
    *,
    budget: int,
    dataset: str,
    task: str,
) -> dict:
    slug = f"{dataset}_{task}"
    p = (
        root
        / f"budget_{budget}"
        / slug
        / slug
        / "manifest.json"
    )
    if not p.exists():
        return {}

    return json.loads(p.read_text())


def main() -> int:
    args = parse_args()
    rows = []

    for budget in BUDGETS:
        for dataset, task in TASKS:
            m = load_manifest(
                args.input_root,
                budget=budget,
                dataset=dataset,
                task=task,
            )

            if not m:
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "budget": budget,
                        "status": "missing",
                        "blocker": "manifest_missing",
                    }
                )
                continue

            scores = m.get("mean_scores", {})
            auto_score = scores.get("auto_only")
            greedy_score = scores.get("auto_plus_fdhg")

            direction = m.get("metric_direction")
            selected_edges = int(
                m.get(
                    "strategy_selected_edge_count",
                    m.get(
                        "selected_screened_edge_count",
                        0,
                    ),
                )
                or 0
            )

            blockers = m.get("blockers", [])
            if isinstance(blockers, list):
                blocker_text = " | ".join(str(x) for x in blockers)
            else:
                blocker_text = str(blockers or "")

            test_accessed = bool(
                m.get("test_split_accessed", False)
            )

            auto_features = None

            auto_candidates = []

            auto_source = m.get("auto_selected_feature_source")
            if auto_source:
                auto_candidates.append(Path(auto_source))

            auto_candidates.append(
                args.input_root
                / "_auto_roots"
                / f"budget_{budget}"
                / f"{dataset}_{task}"
                / "selected_features.json"
            )

            for auto_path in auto_candidates:
                if auto_path.exists():
                    data = json.loads(auto_path.read_text())
                    auto_features = len(
                        data.get("selected_features", [])
                    )
                    break

            gain = None
            if (
                auto_score is not None
                and greedy_score is not None
                and direction is not None
            ):
                gain = direction_aware_gain(
                    float(auto_score),
                    float(greedy_score),
                    str(direction),
                )

            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "budget": budget,
                    "metric": m.get("metric"),
                    "metric_direction": direction,
                    "auto_feature_count": auto_features,
                    "auto_score": auto_score,
                    "selected_edges": selected_edges,
                    "greedy_score": greedy_score,
                    "greedy_gain_vs_auto": gain,
                    "pairwise_rescue_used": bool(
                        m.get("pairwise_rescue_used", False)
                    ),
                    "pairwise_rescue_reason": m.get(
                        "pairwise_rescue_reason"
                    ),
                    "selected_variant": m.get(
                        "selected_variant"
                    ),
                    "gate_selected_variant": m.get(
                        "gate_selected_variant"
                    ),
                    "final_evaluated_variant": m.get(
                        "final_evaluated_variant"
                    ),
                    "official_validation": m.get(
                        "official_validation_score"
                    ),
                    "official_validation_used_for_selection":
                        bool(
                            m.get(
                                "official_validation_was_used_for_selection",
                                False,
                            )
                        ),
                    "candidate_edges": m.get(
                        "candidate_fdhg_edge_count"
                    ),
                    "screened_in_edges": m.get(
                        "screened_in_fdhg_edge_count"
                    ),
                    "screened_out_edges": m.get(
                        "screened_out_fdhg_edge_count"
                    ),
                    "test_split_accessed": test_accessed,
                    "status": (
                        "completed"
                        if not blocker_text
                        else "blocked"
                    ),
                    "blocker": blocker_text,
                }
            )

    df = pd.DataFrame(rows)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(args.output, index=False)

    print(df.to_string(index=False))
    print()

    completed = df[df["status"] == "completed"]

    print("ROWS", len(df))
    print("COMPLETED", len(completed))
    print(
        "TASKS",
        completed[["dataset", "task"]]
        .drop_duplicates()
        .shape[0],
    )
    print(
        "BUDGETS",
        sorted(
            completed["budget"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    )
    test_accessed_series = (
        completed["test_split_accessed"]
        .eq(True)
    )

    print(
        "TEST_ACCESSED",
        int(test_accessed_series.sum()),
    )
    validation_used_series = (
        completed[
            "official_validation_used_for_selection"
        ]
        .eq(True)
    )

    print(
        "VALIDATION_USED_FOR_SELECTION",
        int(validation_used_series.sum()),
    )

    if test_accessed_series.any():
        raise SystemExit("test split access detected")

    if validation_used_series.any():
        raise SystemExit(
            "official validation used for selection"
        )

    augmented = completed[
        completed["selected_edges"].fillna(0) > 0
    ]

    if not augmented.empty:
        improved = int(
            (
                augmented["greedy_gain_vs_auto"]
                .fillna(float("-inf"))
                > 0
            ).sum()
        )
        print(
            "GREEDY_IMPROVES_WHEN_AUGMENTED",
            improved,
            "/",
            len(augmented),
        )

    if len(completed) != 16:
        raise SystemExit(
            f"COMPLETENESS FAIL: expected 16 completed runs, "
            f"found {len(completed)}"
        )

    assert set(completed["budget"]) == {4, 8, 12, 16}
    assert (
        completed[["dataset", "task"]]
        .drop_duplicates()
        .shape[0]
        == 4
    )
    assert not completed["test_split_accessed"].any()
    assert not completed[
        "official_validation_used_for_selection"
    ].any()
    assert completed["auto_score"].notna().all()

    print("COMPLETENESS PASS: 16/16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
