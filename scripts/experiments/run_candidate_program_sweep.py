from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fdhg.compiler.validation_export import (
    build_validation_export_records,
    inspect_candidate_safety_evidence,
    write_validation_export_csv,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    printable = shlex.join(command)
    print(f"[RUN] {printable}", flush=True)
    print(f"[LOG] {log_path}", flush=True)

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"COMMAND: {printable}\n")
        log.write(
            f"STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log.write("=" * 100 + "\n")
        log.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        return_code = process.wait()

        log.write(f"\nRETURN_CODE: {return_code}\n")
        log.write(
            f"FINISHED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    return return_code


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def standard_artifact_exists(
    root: Path,
) -> bool:
    return all(
        (
            root
            / f"target_with_dfs_agg_{split}.parquet"
        ).exists()
        for split in ["train", "val"]
    )



def resolve_label_col(
    *,
    cfg: dict[str, Any],
    task_spec: dict[str, Any],
    dataset: str,
    task: str,
) -> str:
    """
    Resolve the target column without task-specific Python
    registration.

    Resolution order:
      1. experiment task specification
      2. reproduction task configuration
      3. common nested configuration sections
      4. inference from the prepared DFS parquet
    """
    direct_candidates = [
        task_spec.get("label_col"),
        cfg.get("label_col"),
        cfg.get("target_col"),
    ]

    for candidate in direct_candidates:
        if isinstance(candidate, str) and candidate:
            return candidate

    for section_name in [
        "target",
        "evaluation",
        "pairwise",
        "task",
    ]:
        section = cfg.get(section_name)

        if not isinstance(section, dict):
            continue

        for key in [
            "label_col",
            "target_col",
            "label",
        ]:
            candidate = section.get(key)

            if isinstance(candidate, str) and candidate:
                return candidate

    dfs_path = (
        Path("outputs/e2e")
        / f"{dataset}_{task}"
        / "dfs"
        / "target_with_dfs_agg_train.parquet"
    )

    if not dfs_path.exists():
        raise FileNotFoundError(
            "Cannot infer label column because the DFS "
            f"artifact is missing: {dfs_path}"
        )

    frame = pd.read_parquet(dfs_path)

    drop_cols = set(
        cfg.get("evaluation", {}).get(
            "drop_cols",
            [],
        )
    )

    feature_columns = {
        column
        for column in frame.columns
        if column.startswith(
            ("f_", "dfs::", "fdhg::")
        )
    }

    candidates = [
        column
        for column in frame.columns
        if column not in feature_columns
        and column not in drop_cols
        and column not in {
            "__fdhg_row_id",
            "__row_id",
        }
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely infer the label column for "
            f"{dataset}/{task}. Candidate non-feature columns: "
            f"{candidates}; drop_cols={sorted(drop_cols)}; "
            f"all_columns={list(frame.columns)}"
        )

    label_col = candidates[0]

    print(
        f"[LABEL] {dataset}/{task} -> "
        f"{label_col} (inferred from DFS artifact)",
        flush=True,
    )

    return label_col


def evaluator_path(
    *,
    problem_type: str,
    backend: str,
) -> str:
    if problem_type == "binary":
        return (
            "scripts/evaluate/"
            "evaluate_binary_tabpfn.py"
        )

    if problem_type == "regression":
        return (
            "scripts/evaluate/"
            "evaluate_regression_tabpfn.py"
        )

    if problem_type == "multiclass":
        if backend == "catboost":
            return (
                "scripts/evaluate/"
                "evaluate_multiclass_catboost.py"
            )

        return (
            "scripts/evaluate/"
            "evaluate_multiclass_tabpfn.py"
        )

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )


def candidate_artifact_dir(
    *,
    dataset: str,
    task: str,
    candidate: str,
) -> Path:
    task_root = (
        Path("outputs/e2e")
        / f"{dataset}_{task}"
    )

    aliases = {
        "dfs": task_root / "dfs",
        "structural_compact": task_root / "fdhg",
    }

    if candidate in aliases:
        return aliases[candidate]

    return (
        task_root
        / "candidates"
        / candidate
    )


def load_task_config(
    config_path: Path,
    dataset: str,
    task: str,
) -> dict[str, Any]:
    config = load_yaml(config_path)

    def normalize(value: str) -> str:
        return (
            value.lower()
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("__", "_")
        )

    def iter_task_entries(
        value: Any,
        prefix: str = "",
    ):
        if not isinstance(value, dict):
            return

        looks_like_task = (
            "problem_type" in value
            and "label_col" in value
        )

        if looks_like_task:
            yield prefix, value
            return

        for key, child in value.items():
            child_prefix = (
                str(key)
                if not prefix
                else f"{prefix}/{key}"
            )

            if isinstance(child, dict):
                yield from iter_task_entries(
                    child,
                    child_prefix,
                )

    canonical = f"{dataset}/{task}"

    explicit_aliases = {
        (
            "rel-ratebeer",
            "user-place-liked_pairwise",
        ): {
            "rel-ratebeer/user-place-liked_pairwise",
            "rel-ratebeer/user-place-pairwise",
            "ratebeer_user_place_pairwise",
            "user_place_liked_pairwise",
            "user_place_pairwise",
        },
    }

    target_forms = {
        normalize(canonical),
        normalize(f"{dataset}_{task}"),
        normalize(task),
    }

    target_forms.update(
        normalize(alias)
        for alias in explicit_aliases.get(
            (dataset, task),
            set(),
        )
    )

    matches = []

    for path, value in iter_task_entries(config):
        candidate_forms = {
            normalize(path),
        }

        for field in [
            "dataset",
            "task",
            "name",
            "task_key",
            "feature_builder",
            "builder",
        ]:
            field_value = value.get(field)

            if field_value is not None:
                candidate_forms.add(
                    normalize(str(field_value))
                )

        config_dataset = value.get("dataset")
        config_task = value.get("task")

        if config_dataset and config_task:
            candidate_forms.add(
                normalize(
                    f"{config_dataset}/{config_task}"
                )
            )

        exact_overlap = (
            target_forms
            & candidate_forms
        )

        fuzzy_match = any(
            target in candidate
            or candidate in target
            for target in target_forms
            for candidate in candidate_forms
            if len(target) >= 8
            and len(candidate) >= 8
        )

        if exact_overlap or fuzzy_match:
            matches.append((path, value))

    if not matches:
        available = [
            path
            for path, _ in iter_task_entries(config)
        ]

        raise KeyError(
            f"Task config missing for {canonical}. "
            f"Discovered task entries: {available[:50]}"
        )

    if len(matches) > 1:
        exact = [
            item
            for item in matches
            if normalize(item[0]) in target_forms
        ]

        if len(exact) == 1:
            matches = exact
        else:
            raise ValueError(
                f"Ambiguous task config for {canonical}: "
                f"{[path for path, _ in matches]}"
            )

    path, value = matches[0]

    print(
        f"[CONFIG] {canonical} -> {path}",
        flush=True,
    )

    return value


def read_metric_row(
    path: Path,
) -> dict[str, Any]:
    frame = pd.read_csv(path)

    if len(frame) != 1:
        raise ValueError(
            f"Expected one metric row in {path}"
        )

    return frame.iloc[0].to_dict()


def select_best_candidate(
    *,
    result_root: Path,
    candidates: list[str],
    seeds: list[int],
    primary_metric: str,
    secondary_metric: str | None,
    metric_direction: str,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    seed_result_rows = []

    for candidate in candidates:
        run_rows = []

        for seed in seeds:
            path = (
                result_root
                / candidate
                / f"seed{seed}"
                / "metrics.csv"
            )

            if not path.exists():
                continue

            metric_row = read_metric_row(path)
            run_rows.append(metric_row)

            seed_result_rows.append({
                **metric_row,
                "candidate": candidate,
                "seed": seed,
            })

        if len(run_rows) != len(seeds):
            continue

        runs = pd.DataFrame(run_rows)

        if primary_metric not in runs.columns:
            continue

        primary_values = pd.to_numeric(
            runs[primary_metric],
            errors="raise",
        )

        secondary_mean = None

        if (
            secondary_metric is not None
            and secondary_metric in runs.columns
        ):
            secondary_mean = float(
                pd.to_numeric(
                    runs[secondary_metric],
                    errors="raise",
                ).mean()
            )

        n_features = float(
            pd.to_numeric(
                runs["n_features"],
                errors="raise",
            ).mean()
        )

        rows.append({
            "candidate": candidate,
            "n_runs": len(runs),
            "primary_metric": primary_metric,
            "primary_mean": float(
                primary_values.mean()
            ),
            "primary_std": float(
                primary_values.std(ddof=1)
            ),
            "secondary_metric": secondary_metric,
            "secondary_mean": secondary_mean,
            "n_features_mean": n_features,
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise ValueError(
            f"No complete candidates under {result_root}"
        )

    seed_results = pd.DataFrame(
        seed_result_rows
    )

    selected_candidate, gate_info = apply_stability_gate(
        candidate_summary=summary,
        seed_results=seed_results,
        primary_metric=primary_metric,
    )

    selected_rows = summary.loc[
        summary["candidate"] == selected_candidate
    ]

    if selected_rows.empty:
        raise RuntimeError(
            "Stability gate selected an unavailable candidate: "
            f"{selected_candidate}"
        )

    best = selected_rows.iloc[0].to_dict()

    best.update({
        "selection_reason": gate_info["reason"],
        "stability_gate": gate_info,
    })

    summary["selected"] = (
        summary["candidate"]
        == selected_candidate
    )

    summary["selection_reason"] = ""

    summary.loc[
        summary["selected"],
        "selection_reason",
    ] = gate_info["reason"]

    return summary, best


def load_seed_metric_rows(
    *,
    result_root: Path,
    candidates: list[str],
    seeds: list[int],
    primary_metric: str,
) -> pd.DataFrame:
    rows = []

    for candidate in candidates:
        for seed in seeds:
            path = (
                result_root
                / candidate
                / f"seed{seed}"
                / "metrics.csv"
            )
            if not path.exists():
                continue

            metric_row = read_metric_row(path)

            if primary_metric not in metric_row:
                continue

            rows.append({
                **metric_row,
                "candidate": candidate,
                "seed": seed,
                "evidence_location": str(path),
            })

    return pd.DataFrame(rows)



def discover_materialized_candidates(
    *,
    task_output_root: Path,
    configured_candidates: list[str],
) -> list[str]:
    """Discover compiler-materialized candidate programs."""

    discovered = list(configured_candidates)

    candidates_root = task_output_root / "candidates"

    if not candidates_root.exists():
        return discovered

    for candidate_dir in sorted(
        path
        for path in candidates_root.iterdir()
        if path.is_dir()
    ):
        candidate = candidate_dir.name

        # Private/intermediate compiler artifacts are not
        # selectable candidate programs.
        if candidate.startswith("_"):
            continue

        has_train = (
            candidate_dir
            / "target_with_dfs_agg_train.parquet"
        ).exists()

        has_val = (
            candidate_dir
            / "target_with_dfs_agg_val.parquet"
        ).exists()

        if (
            has_train
            and has_val
            and candidate not in discovered
        ):
            discovered.append(candidate)

    return discovered

def apply_stability_gate(
    *,
    candidate_summary: pd.DataFrame,
    seed_results: pd.DataFrame,
    primary_metric: str,
) -> tuple[str, dict]:
    """Select a candidate only when its paired multi-seed gain is stable."""

    baseline_name = "dfs"

    baseline_summary = candidate_summary.loc[
        candidate_summary["candidate"] == baseline_name
    ]

    if baseline_summary.empty:
        raise ValueError(
            "DFS baseline is missing from candidate summary."
        )

    baseline_mean = float(
        baseline_summary.iloc[0]["primary_mean"]
    )

    lower_is_better = primary_metric in {
        "rmse",
        "mae",
        "mse",
        "log_loss",
    }

    minimum_mean_gain_by_metric = {
        "accuracy": 0.001,
        "roc_auc": 0.0,
        "average_precision": 0.0,
        "macro_f1": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "mse": 0.0,
        "log_loss": 0.0,
    }

    minimum_mean_gain = (
        minimum_mean_gain_by_metric.get(
            primary_metric,
            0.0,
        )
    )

    accepted = []

    baseline_rows = seed_results.loc[
        seed_results["candidate"] == baseline_name,
        ["seed", primary_metric],
    ].rename(
        columns={
            primary_metric: "baseline_score",
        }
    )

    for _, row in candidate_summary.iterrows():
        candidate = str(row["candidate"])

        if candidate == baseline_name:
            continue

        candidate_rows = seed_results.loc[
            seed_results["candidate"] == candidate,
            ["seed", primary_metric],
        ].rename(
            columns={
                primary_metric: "candidate_score",
            }
        )

        paired = baseline_rows.merge(
            candidate_rows,
            on="seed",
            how="inner",
            validate="one_to_one",
        )

        if len(paired) < 4:
            continue

        if lower_is_better:
            paired["gain"] = (
                paired["baseline_score"]
                - paired["candidate_score"]
            )

            mean_gain = (
                baseline_mean
                - float(row["primary_mean"])
            )
        else:
            paired["gain"] = (
                paired["candidate_score"]
                - paired["baseline_score"]
            )

            mean_gain = (
                float(row["primary_mean"])
                - baseline_mean
            )

        wins = int(
            (paired["gain"] > 0).sum()
        )

        minimum_delta = float(
            paired["gain"].min()
        )

        passes = (
            mean_gain > minimum_mean_gain
            and wins >= 3
        )

        if not passes:
            continue

        accepted.append({
            "candidate": candidate,
            "mean_gain": mean_gain,
            "wins": wins,
            "minimum_delta": minimum_delta,
            "primary_mean": float(
                row["primary_mean"]
            ),
            "n_features_mean": float(
                row["n_features_mean"]
            ),
        })

    if not accepted:
        return baseline_name, {
            "reason": (
                "no_candidate_passed_stability_gate"
            ),
            "baseline_mean": baseline_mean,
            "primary_metric": primary_metric,
            "minimum_mean_gain": minimum_mean_gain,
            "minimum_required_wins": 3,
        }

    accepted.sort(
        key=lambda result: (
            -result["mean_gain"],
            -result["wins"],
            result["n_features_mean"],
        )
    )

    best = accepted[0]

    return best["candidate"], {
        "reason": "passed_stability_gate",
        "primary_metric": primary_metric,
        "minimum_mean_gain": minimum_mean_gain,
        "minimum_required_wins": 3,
        **best,
    }

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-config",
        default=(
            "configs/experiments/"
            "performance_expansion_tasks.yaml"
        ),
    )
    parser.add_argument(
        "--task-config",
        default="configs/reproduction/tasks.yaml",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--run-root",
        default="outputs/candidate_program_sweep",
    )
    parser.add_argument(
        "--result-root",
        default="results/candidate_program_sweep",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
    )
    parser.add_argument(
        "--canonical-validation-output",
        help=(
            "Optional path for selector-ready canonical "
            "validation/safety CSV. No file is written unless this "
            "flag is supplied."
        ),
    )
    parser.add_argument(
        "--overwrite-validation-output",
        action="store_true",
    )

    args = parser.parse_args()

    experiment_config = load_yaml(
        Path(args.experiment_config)
    )

    task_config_path = Path(args.task_config)
    run_root = Path(args.run_root)
    global_result_root = Path(args.result_root)

    tasks = experiment_config["tasks"]
    candidates = experiment_config[
        "candidate_programs"
    ]
    default_seeds = experiment_config.get(
        "seeds",
        [41, 42, 43, 44],
    )

    env = dict(os.environ)

    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        "src"
        if not current_pythonpath
        else f"src:{current_pythonpath}"
    )

    successes = []
    failures = []
    unavailable = []
    canonical_validation_records = []

    if args.canonical_validation_output is not None:
        _validate_validation_output_path(
            Path(args.canonical_validation_output),
            overwrite=args.overwrite_validation_output,
        )

    for task_index, task_spec in enumerate(
        tasks,
        start=1,
    ):
        dataset = task_spec["dataset"]
        task = task_spec["task"]
        task_key = f"{dataset}/{task}"
        task_slug = f"{dataset}_{task}"

        cfg = load_task_config(
            task_config_path,
            dataset,
            task,
        )

        problem_type = task_spec.get(
            "problem_type",
            cfg["problem_type"],
        )
        backend = cfg.get(
            "evaluation",
            {},
        ).get(
            "backend",
            "tabpfn",
        )

        primary_metric = task_spec[
            "primary_metric"
        ]
        secondary_metric = task_spec.get(
            "secondary_metric"
        )
        metric_direction = task_spec.get(
            "metric_direction",
            (
                "lower"
                if primary_metric in {
                    "rmse",
                    "mae",
                    "mse",
                    "log_loss",
                }
                else "higher"
            ),
        )

        seeds = task_spec.get(
            "seeds",
            default_seeds,
        )

        label_col = resolve_label_col(
            cfg=cfg,
            task_spec=task_spec,
            dataset=dataset,
            task=task,
        )

        drop_cols = ",".join(
            cfg["evaluation"]["drop_cols"]
        )

        evaluator = evaluator_path(
            problem_type=problem_type,
            backend=backend,
        )

        task_result_root = (
            global_result_root
            / task_slug
        )

        print(
            "\n"
            + "#" * 100
            + f"\nTASK {task_index}/{len(tasks)}: {task_key}"
            + "\n"
            + "#" * 100,
            flush=True,
        )

        available_candidates = []

        task_output_root = (
            Path("outputs/e2e")
            / task_slug
        )

        task_candidates = (
            discover_materialized_candidates(
                task_output_root=task_output_root,
                configured_candidates=candidates,
            )
        )

        print(
            "[DISCOVERED CANDIDATES]",
            task_key,
            task_candidates,
            flush=True,
        )

        for candidate in task_candidates:
            feature_root = candidate_artifact_dir(
                dataset=dataset,
                task=task,
                candidate=candidate,
            )

            if not standard_artifact_exists(
                feature_root
            ):
                unavailable.append({
                    "dataset": dataset,
                    "task": task,
                    "candidate": candidate,
                    "artifact_dir": str(feature_root),
                    "reason": "artifact_missing",
                })

                print(
                    f"[UNAVAILABLE] {task_key} "
                    f"candidate={candidate}: "
                    f"{feature_root}",
                    flush=True,
                )
                continue

            available_candidates.append(candidate)

            for seed in seeds:
                out_dir = (
                    task_result_root
                    / candidate
                    / f"seed{seed}"
                )

                metrics_path = (
                    out_dir / "metrics.csv"
                )

                marker = (
                    run_root
                    / "markers"
                    / task_slug
                    / candidate
                    / f"seed{seed}.success.json"
                )

                if (
                    metrics_path.exists()
                    and marker.exists()
                    and not args.rerun
                ):
                    print(
                        f"[SKIP] {task_key} "
                        f"{candidate} seed={seed}",
                        flush=True,
                    )
                    continue

                evaluator_variant = candidate

                command = [
                    sys.executable,
                    "-u",
                    evaluator,
                    "--train-parquet",
                    str(
                        feature_root
                        / "target_with_dfs_agg_train.parquet"
                    ),
                    "--val-parquet",
                    str(
                        feature_root
                        / "target_with_dfs_agg_val.parquet"
                    ),
                    "--output-dir",
                    str(out_dir),
                    "--dataset",
                    dataset,
                    "--task",
                    task,
                    "--variant",
                    evaluator_variant,
                    "--label-col",
                    label_col,
                    "--drop-cols",
                    drop_cols,
                    "--seed",
                    str(seed),
                ]

                if backend != "catboost":
                    command.extend([
                        "--device",
                        args.device,
                    ])

                log_path = (
                    run_root
                    / "logs"
                    / task_slug
                    / candidate
                    / f"seed{seed}.log"
                )

                started = time.time()

                return_code = run_command(
                    command,
                    log_path=log_path,
                    env=env,
                )

                elapsed = time.time() - started

                record = {
                    "dataset": dataset,
                    "task": task,
                    "candidate": candidate,
                    "seed": seed,
                    "return_code": return_code,
                    "elapsed_seconds": elapsed,
                    "artifact_dir": str(feature_root),
                    "result_dir": str(out_dir),
                    "log_path": str(log_path),
                }

                if return_code == 0:
                    write_json(marker, record)
                    successes.append(record)

                    print(
                        f"[PASS] {task_key} "
                        f"{candidate} seed={seed}",
                        flush=True,
                    )
                else:
                    failures.append(record)

                    failure_marker = (
                        run_root
                        / "markers"
                        / task_slug
                        / candidate
                        / f"seed{seed}.failure.json"
                    )

                    write_json(
                        failure_marker,
                        record,
                    )

                    print(
                        f"[FAIL] {task_key} "
                        f"{candidate} seed={seed}",
                        flush=True,
                    )

                    if args.stop_on_error:
                        raise SystemExit(return_code)

        complete_candidates = []

        for candidate in available_candidates:
            complete = all(
                (
                    task_result_root
                    / candidate
                    / f"seed{seed}"
                    / "metrics.csv"
                ).exists()
                for seed in seeds
            )

            if complete:
                complete_candidates.append(
                    candidate
                )

        if complete_candidates:
            summary, best = select_best_candidate(
                result_root=task_result_root,
                candidates=complete_candidates,
                seeds=seeds,
                primary_metric=primary_metric,
                secondary_metric=secondary_metric,
                metric_direction=metric_direction,
            )
            seed_results = load_seed_metric_rows(
                result_root=task_result_root,
                candidates=task_candidates,
                seeds=seeds,
                primary_metric=primary_metric,
            )

            summary_dir = (
                Path("results/compiler")
                / "candidate_program_sweep"
                / task_slug
            )
            summary_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            summary.to_csv(
                summary_dir
                / "candidate_summary.csv",
                index=False,
            )

            write_json(
                summary_dir
                / "selected_candidate.json",
                best,
            )

            print(
                "\n[SELECTED]",
                task_key,
                "->",
                best["candidate"],
                primary_metric,
                best["primary_mean"],
                flush=True,
            )

            if args.canonical_validation_output is not None:
                safety = {}
                for candidate in task_candidates:
                    safety[candidate] = (
                        inspect_candidate_safety_evidence(
                            dataset=dataset,
                            task=task,
                            program_id=candidate,
                            artifact_dir=candidate_artifact_dir(
                                dataset=dataset,
                                task=task,
                                candidate=candidate,
                            ),
                            baseline_program_id="dfs",
                        )
                    )

                export_report = build_validation_export_records(
                    dataset=dataset,
                    task=task,
                    split="validation",
                    primary_metric=primary_metric,
                    metric_direction=metric_direction,
                    candidate_program_ids=task_candidates,
                    expected_seeds=seeds,
                    aggregate_rows=(
                        summary.to_dict("records")
                    ),
                    seed_rows=seed_results.to_dict("records"),
                    selected_program_id=str(best["candidate"]),
                    baseline_program_id="dfs",
                    safety_evidence_by_program=safety,
                )
                canonical_validation_records.extend(
                    export_report.aggregate_records
                )

    report_dir = run_root / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(successes).to_csv(
        report_dir / "successful_runs.csv",
        index=False,
    )

    pd.DataFrame(failures).to_csv(
        report_dir / "failed_runs.csv",
        index=False,
    )

    pd.DataFrame(unavailable).to_csv(
        report_dir / "unavailable_candidates.csv",
        index=False,
    )

    if args.canonical_validation_output is not None:
        output_path = Path(args.canonical_validation_output)
        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            write_validation_export_csv(
                canonical_validation_records,
                handle,
            )
        print("[WROTE]", output_path)

    print("\n" + "=" * 100)
    print("CANDIDATE PROGRAM SWEEP COMPLETE")
    print("=" * 100)
    print("successful runs:", len(successes))
    print("failed runs:", len(failures))
    print(
        "unavailable candidates:",
        len(unavailable),
    )


def _validate_validation_output_path(
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    paper_tables = Path("results/paper_tables").resolve()
    resolved = output_path.resolve()
    if (
        resolved == paper_tables
        or paper_tables in resolved.parents
    ):
        raise ValueError(
            "refusing to write canonical validation output under "
            "results/paper_tables"
        )

    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    if (
        output_path.parent.exists()
        and not output_path.parent.is_dir()
    ):
        raise NotADirectoryError(output_path.parent)


if __name__ == "__main__":
    main()
