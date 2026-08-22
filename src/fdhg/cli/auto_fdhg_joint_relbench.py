from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping


HIGHER_IS_BETTER = {
    "roc_auc",
    "ap",
    "average_precision",
    "accuracy",
    "f1",
    "macro_f1",
    "weighted_f1",
    "mrr",
}

LOWER_IS_BETTER = {
    "rmse",
    "mae",
    "log_loss",
    "logloss",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_direction(metric: str, direction: str | None) -> str:
    value = str(direction or "").strip().lower()

    if value in {"higher", "higher_is_better", "maximize", "max"}:
        return "higher"

    if value in {"lower", "lower_is_better", "minimize", "min"}:
        return "lower"

    metric = metric.strip().lower()

    if metric in HIGHER_IS_BETTER:
        return "higher"

    if metric in LOWER_IS_BETTER:
        return "lower"

    raise ValueError(
        f"cannot_infer_metric_direction:"
        f"metric={metric}:direction={direction}"
    )


def utility(score: float, direction: str) -> float:
    return score if direction == "higher" else -score


def improvement(
    candidate: float,
    reference: float,
    direction: str,
) -> float:
    if direction == "higher":
        return candidate - reference
    return reference - candidate


def resolve_tolerance(
    *,
    metric: str,
    auto_score: float,
    classification_epsilon: float,
    regression_relative_epsilon: float,
) -> float:
    if metric.strip().lower() in LOWER_IS_BETTER:
        return abs(auto_score) * regression_relative_epsilon
    return classification_epsilon


def selected_edge_ids(manifest: Mapping[str, Any]) -> list[str]:
    values = manifest.get("strategy_selected_edge_ids", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def selected_edge_count(manifest: Mapping[str, Any]) -> int:
    value = manifest.get("strategy_selected_edge_count")
    try:
        return int(value)
    except (TypeError, ValueError):
        return len(selected_edge_ids(manifest))


def run_dir(root: Path, dataset: str, task: str) -> Path:
    return root / f"{dataset}_{task}"


def build_base_command(
    args: argparse.Namespace,
    *,
    strategy: str,
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "fdhg.cli.auto_fdhg_relbench",
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--output-root",
        str(output_root),
        "--auto-output-root",
        str(args.auto_output_root),
        "--fdhg-candidate-edges-file",
        str(args.fdhg_candidate_edges_file),
        "--selection-folds",
        str(args.selection_folds),
        "--feature-budget",
        str(args.feature_budget),
        "--max-fdhg-edges",
        str(args.max_fdhg_edges),
        "--max-selected-fdhg-edges",
        str(args.max_selected_fdhg_edges),
        "--edge-selection-strategy",
        strategy,
        "--edge-screening-rule",
        args.edge_screening_rule,
        "--edge-screening-min-delta",
        str(args.edge_screening_min_delta),
        "--edge-screening-min-positive-folds",
        str(args.edge_screening_min_positive_folds),
        "--continuous-fdhg-mode",
        args.continuous_fdhg_mode,
    ]

    if args.canonical_onboarding_root is not None:
        command.extend([
            "--canonical-onboarding-root",
            str(args.canonical_onboarding_root),
        ])

    if args.download:
        command.append("--download")
    else:
        command.append("--no-download")

    command.append("--write")

    if args.overwrite:
        command.append("--overwrite")

    return command


def execute_run(
    command: list[str],
    *,
    log_path: Path,
    env: Mapping[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("RUN:", " ".join(command))
    print("LOG:", log_path)
    print("=" * 100)

    with log_path.open("w") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(env),
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"subrun_failed:return_code={return_code}:"
            f"log={log_path}"
        )


def choose_baseline_strategy(
    *,
    auto_score: float,
    dfs_score: float | None,
    direction: str,
    tolerance: float,
) -> tuple[str, float, str]:
    if dfs_score is None:
        return (
            "auto_only",
            auto_score,
            "dfs_score_unavailable_selected_auto",
        )

    dfs_minus_auto = (
        utility(dfs_score, direction)
        - utility(auto_score, direction)
    )

    if dfs_minus_auto > tolerance:
        return (
            "dfs_fallback",
            dfs_score,
            "dfs_better_than_auto_beyond_tolerance",
        )

    return (
        "auto_only",
        auto_score,
        (
            "auto_selected_over_dfs_within_tolerance"
            if abs(dfs_minus_auto) <= tolerance
            else "auto_better_than_dfs"
        ),
    )


def choose_joint_strategy(
    *,
    auto_score: float,
    dfs_score: float | None,
    independent_score: float | None,
    greedy_score: float | None,
    independent_count: int,
    greedy_count: int,
    direction: str,
    tolerance: float,
    exact_tie_tolerance: float,
) -> dict[str, Any]:
    baseline_variant, baseline_score, baseline_reason = (
        choose_baseline_strategy(
            auto_score=auto_score,
            dfs_score=dfs_score,
            direction=direction,
            tolerance=tolerance,
        )
    )

    admissible: dict[str, dict[str, Any]] = {}
    gains_over_baseline: dict[str, float | None] = {
        "auto_plus_fdhg_independent": None,
        "auto_plus_fdhg_greedy": None,
    }

    if independent_score is not None:
        independent_gain = improvement(
            independent_score,
            baseline_score,
            direction,
        )
        gains_over_baseline[
            "auto_plus_fdhg_independent"
        ] = independent_gain

        if independent_gain > tolerance:
            admissible["auto_plus_fdhg_independent"] = {
                "score": independent_score,
                "edge_count": independent_count,
            }

    if greedy_score is not None:
        greedy_gain = improvement(
            greedy_score,
            baseline_score,
            direction,
        )
        gains_over_baseline[
            "auto_plus_fdhg_greedy"
        ] = greedy_gain

        if greedy_gain > tolerance:
            admissible["auto_plus_fdhg_greedy"] = {
                "score": greedy_score,
                "edge_count": greedy_count,
            }

    if not admissible:
        return {
            "selected_variant": baseline_variant,
            "selected_score": baseline_score,
            "baseline_variant": baseline_variant,
            "baseline_score": baseline_score,
            "baseline_reason": baseline_reason,
            "admissible_fdhg_variants": [],
            "fdhg_gains_over_baseline": gains_over_baseline,
            "selection_reason": (
                "no_fdhg_variant_improved_over_baseline_"
                "beyond_tolerance"
            ),
        }

    if len(admissible) == 1:
        selected_variant = next(iter(admissible))
        selected = admissible[selected_variant]

        return {
            "selected_variant": selected_variant,
            "selected_score": float(selected["score"]),
            "baseline_variant": baseline_variant,
            "baseline_score": baseline_score,
            "baseline_reason": baseline_reason,
            "admissible_fdhg_variants": list(admissible),
            "fdhg_gains_over_baseline": gains_over_baseline,
            "selection_reason": (
                f"only_{selected_variant}_improved_over_"
                "baseline_beyond_tolerance"
            ),
        }

    independent = admissible[
        "auto_plus_fdhg_independent"
    ]
    greedy = admissible[
        "auto_plus_fdhg_greedy"
    ]

    greedy_minus_independent = (
        utility(float(greedy["score"]), direction)
        - utility(float(independent["score"]), direction)
    )

    if abs(greedy_minus_independent) > tolerance:
        if greedy_minus_independent > 0:
            selected_variant = "auto_plus_fdhg_greedy"
            reason = (
                "greedy_better_than_independent_"
                "beyond_tolerance"
            )
        else:
            selected_variant = (
                "auto_plus_fdhg_independent"
            )
            reason = (
                "independent_better_than_greedy_"
                "beyond_tolerance"
            )
    else:
        independent_edges = int(
            independent["edge_count"]
        )
        greedy_edges = int(greedy["edge_count"])

        if greedy_edges < independent_edges:
            selected_variant = "auto_plus_fdhg_greedy"
            reason = (
                "fdhg_performance_within_tolerance_"
                "selected_sparser_greedy"
            )
        elif independent_edges < greedy_edges:
            selected_variant = (
                "auto_plus_fdhg_independent"
            )
            reason = (
                "fdhg_performance_within_tolerance_"
                "selected_sparser_independent"
            )
        else:
            selected_variant = (
                "auto_plus_fdhg_independent"
            )
            reason = (
                "fdhg_performance_and_edge_count_tied_"
                "selected_independent"
            )

    selected = admissible[selected_variant]

    return {
        "selected_variant": selected_variant,
        "selected_score": float(selected["score"]),
        "baseline_variant": baseline_variant,
        "baseline_score": baseline_score,
        "baseline_reason": baseline_reason,
        "admissible_fdhg_variants": list(admissible),
        "fdhg_gains_over_baseline": gains_over_baseline,
        "fdhg_score_difference": abs(
            greedy_minus_independent
        ),
        "exact_numerical_tie": (
            abs(greedy_minus_independent)
            <= exact_tie_tolerance
        ),
        "selection_reason": reason,
    }


def copy_selected_artifact(
    *,
    source_dir: Path,
    destination_dir: Path,
    overwrite: bool,
) -> None:
    if destination_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"selected_output_exists:{destination_dir}"
            )
        shutil.rmtree(destination_dir)

    shutil.copytree(source_dir, destination_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent and greedy FDHG selection under identical "
            "train-only folds, then apply a conservative joint gate."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--auto-output-root", required=True, type=Path)
    parser.add_argument(
        "--canonical-onboarding-root",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--fdhg-candidate-edges-file",
        required=True,
        type=Path,
    )

    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument("--feature-budget", type=int, default=32)
    parser.add_argument("--max-fdhg-edges", type=int, default=32)
    parser.add_argument(
        "--max-selected-fdhg-edges",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--edge-screening-rule",
        default="fixed_count",
    )
    parser.add_argument(
        "--edge-screening-min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--edge-screening-min-positive-folds",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--continuous-fdhg-mode",
        default="exclude",
    )

    parser.add_argument(
        "--classification-epsilon",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--regression-relative-epsilon",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--exact-tie-tolerance",
        type=float,
        default=1e-12,
    )

    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    args.auto_output_root = args.auto_output_root.resolve()
    if args.canonical_onboarding_root is not None:
        args.canonical_onboarding_root = (
            args.canonical_onboarding_root.resolve()
        )
    args.fdhg_candidate_edges_file = (
        args.fdhg_candidate_edges_file.resolve()
    )

    if not args.fdhg_candidate_edges_file.exists():
        raise FileNotFoundError(args.fdhg_candidate_edges_file)

    independent_root = args.output_root / "independent"
    greedy_root = args.output_root / "greedy"
    selected_root = args.output_root / "selected"
    joint_root = args.output_root / "joint"
    log_root = args.output_root / "logs"

    for path in [
        independent_root,
        greedy_root,
        selected_root,
        joint_root,
        log_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)

    independent_command = build_base_command(
        args,
        strategy="independent",
        output_root=independent_root,
    )
    greedy_command = build_base_command(
        args,
        strategy="greedy",
        output_root=greedy_root,
    )

    try:
        execute_run(
            independent_command,
            log_path=log_root / "independent.log",
            env=env,
        )
        execute_run(
            greedy_command,
            log_path=log_root / "greedy.log",
            env=env,
        )
    except Exception:
        if args.debug:
            traceback.print_exc()
        raise

    independent_dir = run_dir(
        independent_root,
        args.dataset,
        args.task,
    )
    greedy_dir = run_dir(
        greedy_root,
        args.dataset,
        args.task,
    )

    independent_selected = read_json(
        independent_dir / "selected_variant.json"
    )
    greedy_selected = read_json(
        greedy_dir / "selected_variant.json"
    )
    independent_manifest = read_json(
        independent_dir / "manifest.json"
    )
    greedy_manifest = read_json(
        greedy_dir / "manifest.json"
    )
    candidate_sha = sha256_file(args.fdhg_candidate_edges_file)
    independent_candidate_sha = str(
        independent_manifest.get("candidate_edges_file_sha256") or ""
    )
    greedy_candidate_sha = str(
        greedy_manifest.get("candidate_edges_file_sha256") or ""
    )
    same_candidate_pool_verified = (
        independent_candidate_sha == candidate_sha
        and greedy_candidate_sha == candidate_sha
    )
    if not same_candidate_pool_verified:
        raise ValueError("candidate_pool_sha_mismatch")

    independent_scores = independent_selected.get("mean_scores", {})
    greedy_scores = greedy_selected.get("mean_scores", {})

    metric = str(
        independent_selected.get("primary_metric")
        or independent_manifest.get("primary_metric")
        or greedy_selected.get("primary_metric")
        or greedy_manifest.get("primary_metric")
    )

    direction = normalize_direction(
        metric,
        independent_selected.get("metric_direction")
        or independent_manifest.get("metric_direction")
        or greedy_selected.get("metric_direction")
        or greedy_manifest.get("metric_direction"),
    )

    auto_score = finite_float(independent_scores.get("auto_only"))
    dfs_score = finite_float(independent_scores.get("dfs_fallback"))
    independent_score = finite_float(
        independent_scores.get("auto_plus_fdhg")
    )
    greedy_score = finite_float(
        greedy_scores.get("auto_plus_fdhg")
    )

    if auto_score is None:
        raise ValueError("missing_auto_only_inner_score")

    independent_ids = selected_edge_ids(independent_manifest)
    greedy_ids = selected_edge_ids(greedy_manifest)

    independent_count = selected_edge_count(independent_manifest)
    greedy_count = selected_edge_count(greedy_manifest)

    tolerance = resolve_tolerance(
        metric=metric,
        auto_score=auto_score,
        classification_epsilon=args.classification_epsilon,
        regression_relative_epsilon=(
            args.regression_relative_epsilon
        ),
    )

    gate_result = choose_joint_strategy(
        auto_score=auto_score,
        dfs_score=dfs_score,
        independent_score=independent_score,
        greedy_score=greedy_score,
        independent_count=independent_count,
        greedy_count=greedy_count,
        direction=direction,
        tolerance=tolerance,
        exact_tie_tolerance=args.exact_tie_tolerance,
    )

    selected_variant = str(
        gate_result["selected_variant"]
    )
    selected_score = float(
        gate_result["selected_score"]
    )
    baseline_variant = str(
        gate_result["baseline_variant"]
    )
    baseline_score = float(
        gate_result["baseline_score"]
    )
    baseline_reason = str(
        gate_result["baseline_reason"]
    )
    gate_reason = str(
        gate_result["selection_reason"]
    )

    if selected_variant == "auto_plus_fdhg_greedy":
        source_strategy = "greedy"
        source_dir = greedy_dir
        final_ids = greedy_ids
        final_count = greedy_count
    elif selected_variant == "auto_plus_fdhg_independent":
        source_strategy = "independent"
        source_dir = independent_dir
        final_ids = independent_ids
        final_count = independent_count
    else:
        # Auto and DFS artifacts are equivalent across the two strategy runs.
        source_strategy = "independent"
        source_dir = independent_dir
        final_ids = []
        final_count = 0

    selected_dir = run_dir(
        selected_root,
        args.dataset,
        args.task,
    )

    copy_selected_artifact(
        source_dir=source_dir,
        destination_dir=selected_dir,
        overwrite=args.overwrite,
    )

    official_validation_was_used = False
    test_split_accessed = bool(
        independent_manifest.get("test_split_accessed", False)
        or greedy_manifest.get("test_split_accessed", False)
    )

    joint_mean_scores = {
        "auto_only": auto_score,
        "dfs_fallback": dfs_score,
        "auto_plus_fdhg_independent": independent_score,
        "auto_plus_fdhg_greedy": greedy_score,
    }

    strategy_prefix = {
        "auto_only": "auto_selected",
        "dfs_fallback": "dfs_selected",
        "auto_plus_fdhg_independent": (
            "independent_fdhg_selected"
        ),
        "auto_plus_fdhg_greedy": (
            "greedy_fdhg_selected"
        ),
    }[selected_variant]

    final_reason = (
        f"{strategy_prefix}_after_joint_train_only_comparison;"
        f"baseline={baseline_reason};"
        f"gate={gate_reason}"
    )

    joint_selection = {
        "dataset": args.dataset,
        "task": args.task,
        "primary_metric": metric,
        "metric_direction": direction,
        "mean_scores": joint_mean_scores,
        "baseline_variant": baseline_variant,
        "baseline_score": baseline_score,
        "baseline_selection_reason": baseline_reason,
        "admissible_fdhg_variants": gate_result[
            "admissible_fdhg_variants"
        ],
        "fdhg_gains_over_baseline": gate_result[
            "fdhg_gains_over_baseline"
        ],
        "gate_selection_reason": gate_reason,
        "selection_tolerance": tolerance,
        "classification_epsilon": args.classification_epsilon,
        "regression_relative_epsilon": (
            args.regression_relative_epsilon
        ),
        "exact_tie_tolerance": args.exact_tie_tolerance,
        "independent_selected_edge_ids": independent_ids,
        "independent_selected_edge_count": independent_count,
        "greedy_selected_edge_ids": greedy_ids,
        "greedy_selected_edge_count": greedy_count,
        "selected_variant": selected_variant,
        "selected_score": selected_score,
        "selected_source_strategy": source_strategy,
        "selected_edge_ids": final_ids,
        "selected_edge_count": final_count,
        "selection_reason": final_reason,
        "official_validation_was_used_for_selection": (
            official_validation_was_used
        ),
        "test_split_accessed": test_split_accessed,
        "candidate_file": str(args.fdhg_candidate_edges_file),
        "candidate_file_sha256": candidate_sha,
        "independent_candidate_file_sha256": independent_candidate_sha,
        "greedy_candidate_file_sha256": greedy_candidate_sha,
        "same_candidate_pool_verified": same_candidate_pool_verified,
        "independent_output_dir": str(independent_dir),
        "greedy_output_dir": str(greedy_dir),
        "selected_output_dir": str(selected_dir),
    }

    joint_task_dir = run_dir(
        joint_root,
        args.dataset,
        args.task,
    )
    joint_task_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        joint_task_dir / "joint_selection.json",
        joint_selection,
    )

    selected_variant_path = selected_dir / "selected_variant.json"
    selected_variant_payload = read_json(selected_variant_path)
    selected_variant_payload.update({
        "selected_variant": selected_variant,
        "selected_score": selected_score,
        "selected_source_strategy": source_strategy,
        "selection_reason": final_reason,
        "selection_tolerance": tolerance,
        "mean_scores": joint_mean_scores,
        "baseline_variant": baseline_variant,
        "baseline_score": baseline_score,
        "baseline_selection_reason": baseline_reason,
        "admissible_fdhg_variants": gate_result[
            "admissible_fdhg_variants"
        ],
        "fdhg_gains_over_baseline": gate_result[
            "fdhg_gains_over_baseline"
        ],
        "gate_selection_reason": gate_reason,
        "official_validation_was_used_for_selection": False,
        "test_split_accessed": test_split_accessed,
        "candidate_file": str(args.fdhg_candidate_edges_file),
        "candidate_file_sha256": candidate_sha,
        "same_candidate_pool_verified": same_candidate_pool_verified,
    })
    write_json(
        selected_variant_path,
        selected_variant_payload,
    )

    manifest_path = selected_dir / "manifest.json"
    manifest_payload = read_json(manifest_path)
    manifest_payload.update({
        "joint_strategy_gate": True,
        "joint_selected_variant": selected_variant,
        "joint_selected_source_strategy": source_strategy,
        "joint_selection_reason": final_reason,
        "joint_selection_tolerance": tolerance,
        "joint_mean_scores": joint_mean_scores,
        "independent_selected_edge_ids": independent_ids,
        "independent_selected_edge_count": independent_count,
        "greedy_selected_edge_ids": greedy_ids,
        "greedy_selected_edge_count": greedy_count,
        "strategy_selected_edge_ids": final_ids,
        "strategy_selected_edge_count": final_count,
        "official_validation_was_used_for_selection": False,
        "test_split_accessed": test_split_accessed,
        "candidate_file": str(args.fdhg_candidate_edges_file),
        "candidate_file_sha256": candidate_sha,
        "independent_candidate_file_sha256": independent_candidate_sha,
        "greedy_candidate_file_sha256": greedy_candidate_sha,
        "same_candidate_pool_verified": same_candidate_pool_verified,
    })
    write_json(
        manifest_path,
        manifest_payload,
    )

    print("\n" + "=" * 100)
    print("JOINT FDHG SELECTION")
    print("=" * 100)
    print("DATASET", args.dataset)
    print("TASK", args.task)
    print("METRIC", metric)
    print("METRIC_DIRECTION", direction)
    print("AUTO_ONLY", auto_score)
    print("DFS_FALLBACK", dfs_score)
    print("INDEPENDENT", independent_score)
    print("GREEDY", greedy_score)
    print("INDEPENDENT_EDGES", independent_count)
    print("GREEDY_EDGES", greedy_count)
    print("SELECTION_TOLERANCE", tolerance)
    print("SELECTED_VARIANT", selected_variant)
    print("SELECTED_SOURCE_STRATEGY", source_strategy)
    print("SELECTED_EDGE_COUNT", final_count)
    print("SELECTION_REASON", final_reason)
    print(
        "OFFICIAL_VALIDATION_USED_FOR_SELECTION",
        official_validation_was_used,
    )
    print("TEST_SPLIT_ACCESSED", test_split_accessed)
    print("JOINT_SELECTION_FILE", joint_task_dir / "joint_selection.json")
    print("SELECTED_OUTPUT_DIR", selected_dir)


if __name__ == "__main__":
    main()
