from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DATASET = "rel-f1"
TASK = "driver-position"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the pairwise-initialization ablation."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root containing preserved paper artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations/pairwise-initialization"),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    final_gate_root = artifact_root / "outputs" / "final-gate-51task-v2"
    canonical_onboarding_root = final_gate_root / "_canonical_onboarding"

    task_artifact_root = final_gate_root / f"{DATASET}_{TASK}"
    auto_root = task_artifact_root / "auto"
    candidate_file = task_artifact_root / "candidates" / "fixed_candidate_edges.json"

    selected_features = auto_root / f"{DATASET}_{TASK}" / "selected_features.json"

    if not selected_features.exists():
        raise FileNotFoundError(
            f"Missing frozen Auto representation: {selected_features}"
        )

    if not candidate_file.exists():
        raise FileNotFoundError(
            f"Missing frozen FDHG candidate pool: {candidate_file}"
        )

    settings = {
        "enabled": [],
        "disabled": ["--disable-pairwise-rescue"],
    }

    for name, extra in settings.items():
        run_root = output_root / name

        cmd = [
            args.python,
            "-m",
            "fdhg.cli.auto_fdhg_relbench",
            "--dataset",
            DATASET,
            "--task",
            TASK,
            "--output-root",
            str(run_root),
            "--auto-output-root",
            str(auto_root),
            "--canonical-onboarding-root",
            str(canonical_onboarding_root),
            "--selection-folds",
            "3",
            "--feature-budget",
            "32",
            "--max-fdhg-edges",
            "32",
            "--fdhg-candidate-edges-file",
            str(candidate_file),
            "--max-selected-fdhg-edges",
            "32",
            "--edge-selection-strategy",
            "greedy",
            "--edge-screening-rule",
            "fixed_count",
            "--edge-screening-min-delta",
            "0",
            "--edge-screening-min-positive-folds",
            "2",
            "--continuous-fdhg-mode",
            "exclude",
            "--no-download",
            "--write",
            "--overwrite",
            *extra,
        ]
        run(cmd)

    rows = []
    for name in ("enabled", "disabled"):
        manifest = load_manifest(
            output_root / name / f"{DATASET}_{TASK}" / "manifest.json"
        )

        mean_scores = manifest.get("mean_scores", {})
        greedy_score = mean_scores.get("auto_plus_fdhg")
        auto_score = mean_scores.get("auto_only")

        rows.append(
            {
                "setting": name,
                "selected_variant": manifest.get("selected_variant"),
                "accepted_edges": manifest.get("accepted_edges"),
                "screened_in_edges": manifest.get("screened_in_edges"),
                "pairwise_rescue_used": manifest.get("pairwise_rescue_used"),
                "pairwise_rescue_reason": manifest.get("pairwise_rescue_reason"),
                "auto_inner_score": auto_score,
                "greedy_inner_score": greedy_score,
                "test_split_accessed": manifest.get("test_split_accessed"),
            }
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    print("\n===== PAIRWISE INITIALIZATION SUMMARY =====")
    for row in rows:
        print(row)

    print("\nWROTE", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
