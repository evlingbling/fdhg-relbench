from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASKS = {
    "rel-trial/studies-enrollment": {
        "dataset": "rel-trial",
        "task": "studies-enrollment",
        "candidate_budget": 8,
    },
    "rel-trial/study-outcome": {
        "dataset": "rel-trial",
        "task": "study-outcome",
        "candidate_budget": 32,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("."),
        help=(
            "Root containing historical Auto and candidate artifacts. "
            "For archived paper artifacts this is the old repository root."
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/cross-fold-consistency"),
    )
    p.add_argument(
        "--positive-folds",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[1, 2, 3],
    )
    p.add_argument(
        "--task",
        choices=sorted(TASKS),
        default=None,
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    artifact_root = args.artifact_root.resolve()

    auto_root = (
        artifact_root
        / "outputs"
        / "historical-auto-onboarding-3fold"
    )

    candidate_root = (
        artifact_root
        / "configs"
        / "historical_candidates"
    )

    specs = TASKS.items()
    if args.task is not None:
        specs = [(args.task, TASKS[args.task])]

    for task_key, spec in specs:
        dataset = spec["dataset"]
        task = spec["task"]
        budget = spec["candidate_budget"]

        candidate_file = (
            candidate_root
            / f"rel-trial_{task}_budget{budget}.json"
        )

        if not auto_root.exists():
            raise FileNotFoundError(auto_root)

        if not candidate_file.exists():
            raise FileNotFoundError(candidate_file)

        for positive_folds in args.positive_folds:
            result_root = (
                args.output_root
                / f"{dataset}_{task}"
                / f"positive_folds_{positive_folds}"
            )

            print()
            print("=" * 88)
            print(
                f"{task_key} "
                f"positive_folds={positive_folds}/3 "
                f"candidate_budget={budget}"
            )
            print("=" * 88)

            cmd = [
                sys.executable,
                "-m",
                "fdhg.cli.auto_fdhg_relbench",
                "--dataset",
                dataset,
                "--task",
                task,
                "--output-root",
                str(result_root),
                "--auto-output-root",
                str(auto_root),
                "--fdhg-candidate-edges-file",
                str(candidate_file),
                "--selection-folds",
                "3",
                "--feature-budget",
                "32",
                "--max-fdhg-edges",
                str(budget),
                "--max-selected-fdhg-edges",
                str(budget),
                "--edge-selection-strategy",
                "greedy",
                "--edge-screening-rule",
                "fixed_count",
                "--edge-screening-min-delta",
                "0",
                "--edge-screening-min-positive-folds",
                str(positive_folds),
                "--continuous-fdhg-mode",
                "exclude",
                "--no-download",
                "--write",
                "--overwrite",
            ]

            subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
