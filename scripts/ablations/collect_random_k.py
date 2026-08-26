from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TASKS = [
    ("rel-event", "user-attendance", 12, 2),
    ("rel-f1", "driver-position", 8, 3),
    ("rel-trial", "studies-enrollment", 4, 3),
    ("rel-trial", "study-outcome", 12, 6),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/ablations/random-k"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/ablations/random-k/random_k_summary.csv"
        ),
    )
    return p.parse_args()


def find_one(root: Path, name: str) -> Path:
    xs = list(root.rglob(name))
    if len(xs) != 1:
        raise RuntimeError(
            f"expected exactly one {name} below {root}, got {xs}"
        )
    return xs[0]


def direction_gain(
    score: float,
    baseline: float,
    direction: str,
) -> float:
    if direction in {"higher", "higher_is_better"}:
        return score - baseline
    if direction in {"lower", "lower_is_better"}:
        return baseline - score
    raise ValueError(f"unknown metric direction: {direction}")


def main() -> int:
    args = parse_args()
    rows = []

    for dataset, task, feature_budget, k in TASKS:
        slug = f"{dataset}_{task}"

        for seed in range(20):
            trial_root = (
                args.input_root
                / slug
                / f"seed_{seed:02d}"
            )

            sampled_path = (
                trial_root / "sampled_candidate_edges.json"
            )
            if not sampled_path.exists():
                continue

            sampled = json.loads(sampled_path.read_text())
            sampled_ids = [
                str(x.get("edge_id", ""))
                for x in sampled
            ]

            result_root = trial_root / "result"
            manifest_path = find_one(
                result_root,
                "manifest.json",
            )
            m = json.loads(manifest_path.read_text())

            scores = m.get("mean_scores", {})
            auto_score = scores.get("auto_only")
            random_score = scores.get("auto_plus_fdhg")

            if auto_score is None or random_score is None:
                raise RuntimeError(
                    f"{slug} seed={seed}: missing paired scores"
                )

            direction = str(
                m.get("metric_direction")
            )

            test_accessed = bool(
                m.get("test_split_accessed", False)
            )
            validation_used = bool(
                m.get(
                    "official_validation_was_used_for_selection",
                    False,
                )
            )

            if test_accessed:
                raise RuntimeError(
                    f"TEST SPLIT ACCESSED: {slug} seed={seed}"
                )

            if validation_used:
                raise RuntimeError(
                    "OFFICIAL VALIDATION USED FOR SELECTION: "
                    f"{slug} seed={seed}"
                )

            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "metric": m.get("metric"),
                    "metric_direction": direction,
                    "auto_feature_budget": feature_budget,
                    "candidate_count": 32,
                    "k": k,
                    "random_seed": seed,
                    "sampled_edge_ids": " | ".join(sampled_ids),
                    "sampled_edge_count": len(sampled_ids),
                    "inner_mean_auto": auto_score,
                    "inner_mean_random_k": random_score,
                    "direction_aware_gain_vs_auto":
                        direction_gain(
                            random_score,
                            auto_score,
                            direction,
                        ),
                    "test_split_accessed": test_accessed,
                    "official_validation_used_for_selection":
                        validation_used,
                    "status": "completed",
                    "blocker": "",
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
    print("ROWS", len(df))

    if not df.empty:
        print(
            "TASKS",
            df[["dataset", "task"]]
            .drop_duplicates()
            .shape[0],
        )
        print(
            "TEST_ACCESSED",
            int(df["test_split_accessed"].sum()),
        )

    if len(df) != 80:
        raise SystemExit(
            f"COMPLETENESS FAIL: expected 80 runs, found {len(df)}"
        )

    assert (
        df.groupby(["dataset", "task"]).size() == 20
    ).all()
    assert (df["sampled_edge_count"] == df["k"]).all()
    assert not df["test_split_accessed"].any()
    assert not df[
        "official_validation_used_for_selection"
    ].any()

    print("COMPLETENESS PASS: 80/80")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
