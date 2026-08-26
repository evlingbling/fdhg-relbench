from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


TASKS = [
    {
        "dataset": "rel-event",
        "task": "user-attendance",
        "feature_budget": 12,
        "k": 2,
    },
    {
        "dataset": "rel-f1",
        "task": "driver-position",
        "feature_budget": 8,
        "k": 3,
    },
    {
        "dataset": "rel-trial",
        "task": "studies-enrollment",
        "feature_budget": 4,
        "k": 3,
    },
    {
        "dataset": "rel-trial",
        "task": "study-outcome",
        "feature_budget": 12,
        "k": 6,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("outputs/final-gate-51task-v2"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/random-k"),
    )
    p.add_argument(
        "--canonical-onboarding-root",
        type=Path,
        default=None,
        help=(
            "Optional canonical onboarding root. Defaults to "
            "<canonical-root>/_canonical_onboarding."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(20)),
    )
    p.add_argument(
        "--task",
        choices=[f"{x['dataset']}/{x['task']}" for x in TASKS],
        default=None,
        help="Optional single representative task to run.",
    )
    return p.parse_args()


def task_dir_name(dataset: str, task: str) -> str:
    return f"{dataset}_{task}"


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    task_specs = TASKS
    if args.task is not None:
        task_specs = [
            x
            for x in TASKS
            if f"{x['dataset']}/{x['task']}" == args.task
        ]

    for spec in task_specs:
        dataset = spec["dataset"]
        task = spec["task"]
        feature_budget = spec["feature_budget"]
        k = spec["k"]

        name = task_dir_name(dataset, task)
        canonical_task_root = args.canonical_root / name

        candidate_file = (
            canonical_task_root
            / "candidates"
            / "fixed_candidate_edges.json"
        )
        auto_root = canonical_task_root / "auto"

        canonical_onboarding_root = (
            args.canonical_onboarding_root
            if args.canonical_onboarding_root is not None
            else args.canonical_root / "_canonical_onboarding"
        )

        if not candidate_file.exists():
            raise FileNotFoundError(candidate_file)

        edges = json.loads(candidate_file.read_text())

        if not isinstance(edges, list):
            raise TypeError(
                f"candidate file must contain a JSON list: {candidate_file}"
            )

        edges = sorted(
            edges,
            key=lambda edge: str(edge.get("edge_id", "")),
        )

        if len(edges) < k:
            raise ValueError(
                f"{name}: candidate_count={len(edges)} < k={k}"
            )

        for seed in args.seeds:
            rng = random.Random(seed)

            sampled_indices = rng.sample(range(len(edges)), k=k)
            sampled_edges = [edges[i] for i in sampled_indices]

            trial_root = (
                args.output_root
                / name
                / f"seed_{seed:02d}"
            )
            trial_root.mkdir(parents=True, exist_ok=True)

            sampled_file = trial_root / "sampled_candidate_edges.json"
            sampled_file.write_text(
                json.dumps(
                    sampled_edges,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )

            result_root = trial_root / "result"

            cmd = [
                sys.executable,
                "scripts/experiments/run_efficiency_one.py",
                "--dataset",
                dataset,
                "--task",
                task,
                "--method",
                "all",
                "--output-root",
                str(result_root),
                "--auto-root",
                str(auto_root),
                "--canonical-root",
                str(canonical_onboarding_root),
                "--candidate-file",
                str(sampled_file),
                "--feature-budget",
                str(feature_budget),
            ]

            print()
            print("=" * 88)
            print(
                f"{dataset}/{task} "
                f"seed={seed} "
                f"K={k} "
                f"budget={feature_budget}"
            )
            print(
                "edges:",
                " | ".join(
                    str(edge.get("edge_id", ""))
                    for edge in sampled_edges
                ),
            )
            print("=" * 88)

            subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
