from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


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
        "--artifact-root",
        type=Path,
        default=Path("."),
        help="Root containing historical final-gate-51task-v2 artifacts.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/auto-budget"),
    )
    p.add_argument(
        "--canonical-onboarding-root",
        type=Path,
        default=None,
        help=(
            "Optional complete canonical onboarding root. "
            "When omitted, use the canonical root under the "
            "historical final-gate artifact tree."
        ),
    )
    p.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task filter: dataset/task",
    )
    p.add_argument(
        "--budget",
        type=int,
        choices=BUDGETS,
        default=None,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    artifact_root = args.artifact_root.resolve()
    output_root = args.output_root.resolve()

    final_gate_root = (
        artifact_root
        / "outputs"
        / "final-gate-51task-v2"
    )

    for dataset, task in TASKS:
        task_key = f"{dataset}/{task}"
        task_slug = f"{dataset}_{task}"

        if args.task is not None and args.task != task_key:
            continue

        budgets = [args.budget] if args.budget is not None else BUDGETS

        for budget in budgets:
            canonical_task_root = final_gate_root / task_slug

            auto_trial_root = (
                canonical_task_root
                / "auto_budget_trials"
                / f"budget_{budget}"
            )

            auto_selected = (
                auto_trial_root
                / task_slug
                / "selected_features.json"
            )

            candidate_file = (
                canonical_task_root
                / "candidates"
                / "fixed_candidate_edges.json"
            )

            canonical_base = (
                args.canonical_onboarding_root.resolve()
                if args.canonical_onboarding_root is not None
                else (
                    canonical_task_root.parent
                    / "_canonical_onboarding"
                )
            )

            canonical_onboarding = (
                canonical_base
                / f"relbench-v1-{dataset}_{task}"
            )

            if not auto_selected.exists():
                raise FileNotFoundError(
                    f"missing budget-specific Auto artifact: {auto_selected}"
                )

            if not candidate_file.exists():
                raise FileNotFoundError(
                    f"missing fixed candidate pool: {candidate_file}"
                )

            # Stage the budget-specific Auto representation in the
            # directory layout expected by auto_fdhg_relbench.
            staged_root = (
                output_root
                / "_auto_roots"
                / f"budget_{budget}"
            )
            staged_auto_root = (
                staged_root
                / task_slug
            )

            if staged_auto_root.exists():
                shutil.rmtree(staged_auto_root)

            staged_auto_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                auto_selected,
                staged_auto_root / "selected_features.json",
            )

            auto_manifest = (
                auto_trial_root
                / task_slug
                / "auto_onboarding_manifest.json"
            )
            if auto_manifest.exists():
                shutil.copy2(
                    auto_manifest,
                    staged_auto_root / "auto_onboarding_manifest.json",
                )

            run_root = (
                output_root
                / f"budget_{budget}"
                / task_slug
            )

            completed_manifests = list(
                run_root.rglob("manifest.json")
            )

            completed = False

            for manifest_path in completed_manifests:
                try:
                    import json
                    manifest = json.loads(
                        manifest_path.read_text()
                    )
                except Exception:
                    continue

                if (
                    manifest.get("dataset") == dataset
                    and manifest.get("task") == task
                    and manifest.get("status") == "completed"
                    and manifest.get("test_split_accessed") is False
                    and manifest.get(
                        "official_validation_was_used_for_selection"
                    ) is False
                ):
                    completed = True
                    break

            if completed:
                print(
                    "READY_EXISTING",
                    task_key,
                    f"budget={budget}",
                )
                continue

            print()
            print("=" * 100)
            print(
                f"{task_key} Auto budget={budget}"
            )
            print("=" * 100)

            cmd = [
                sys.executable,
                "-m",
                "fdhg.cli.auto_fdhg_relbench",
                "--dataset",
                dataset,
                "--task",
                task,
                "--output-root",
                str(run_root),
                "--auto-output-root",
                str(
                    output_root
                    / "_auto_roots"
                    / f"budget_{budget}"
                ),
                "--fdhg-candidate-edges-file",
                str(candidate_file),
                "--canonical-onboarding-root",
                str(canonical_onboarding),
                "--selection-folds",
                "3",
                "--feature-budget",
                str(budget),
                "--max-fdhg-edges",
                "32",
                "--max-selected-fdhg-edges",
                "32",
                "--edge-selection-strategy",
                "greedy",
                "--edge-screening-rule",
                "fixed_count",
                "--edge-screening-min-delta",
                "0.0",
                "--edge-screening-min-positive-folds",
                "2",
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
