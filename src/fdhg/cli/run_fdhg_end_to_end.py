from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fdhg.onboarding.relbench_v1 import (
    _candidate_relation_fingerprint,
    _load_relbench_objects,
    _schema_fingerprint,
    _task_metadata_from_config,
    _table_df,
    _verified_one_hop_relations,
    resolve_relbench_task_metadata,
    resolved_metadata_reusable,
)


PIPELINE_VERSION = "fdhg-end-to-end-v1"
RAW_PROFILE_SCHEMA_PATH = Path("raw_profile") / "schema_profile.json"


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def task_slug(dataset: str, task: str) -> str:
    return f"{dataset}_{task}"


def run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    env: Mapping[str, str],
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    printable = " ".join(str(value) for value in command)

    print()
    print("=" * 100)
    print("COMMAND:", printable)
    print("LOG:", log_path)
    print("=" * 100)

    if dry_run:
        return

    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(env),
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"stage_failed:return_code={return_code}:log={log_path}"
        )


def completed_auto_artifact(
    auto_root: Path,
    dataset: str,
    task: str,
) -> bool:
    directory = auto_root / task_slug(dataset, task)

    return (
        (directory / "selected_features.json").exists()
        and (directory / "auto_onboarding_manifest.json").exists()
    )


def completed_discovery_artifact(
    discovery_root: Path,
    dataset: str,
    task: str,
) -> bool:
    directory = discovery_root / task_slug(dataset, task)

    return (
        (directory / "candidate_discovery.json").exists()
        and (directory / "manifest.json").exists()
    )


def completed_joint_artifact(
    joint_output_root: Path,
    dataset: str,
    task: str,
) -> bool:
    joint_file = (
        joint_output_root
        / "joint"
        / task_slug(dataset, task)
        / "joint_selection.json"
    )

    selected_dir = (
        joint_output_root
        / "selected"
        / task_slug(dataset, task)
    )

    return (
        joint_file.exists()
        and (selected_dir / "manifest.json").exists()
        and (selected_dir / "selected_variant.json").exists()
    )


def require_files(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "missing_required_artifacts:\n" + "\n".join(missing)
        )


def canonical_dataset_name(dataset: str) -> str:
    return (
        dataset
        if dataset.startswith("relbench-v1-")
        else f"relbench-v1-{dataset}"
    )


def canonical_task_dir(root: Path, dataset: str, task: str) -> Path:
    return root / f"{canonical_dataset_name(dataset)}_{task}"


def validate_task_metadata_config(
    path: Path | None,
    *,
    dataset: str,
    task: str,
) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided", "task_key": f"{dataset}/{task}"}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("task_metadata_config_must_be_mapping")
    tasks = raw.get("tasks", {})
    if not isinstance(tasks, Mapping):
        raise ValueError("task_metadata_config_tasks_must_be_mapping")
    key = f"{dataset}/{task}"
    legacy_key = f"relbench-v1-{dataset}/{task}"
    if legacy_key in tasks and key not in tasks:
        raise ValueError(
            "invalid_task_metadata_key:"
            f"{legacy_key}:expected:{key}"
        )
    if key not in tasks:
        raise ValueError(f"missing_task_metadata_key:{key}")
    row = dict(tasks[key] or {})
    if "problem_type" in row:
        raw_problem = str(row["problem_type"]).strip().lower()
        row["problem_type"] = {
            "binary": "binary_classification",
            "multiclass": "multiclass_classification",
            "regression": "regression",
        }.get(raw_problem, raw_problem)
    return {
        "status": "validated",
        "path": str(path),
        "task_key": key,
        "normalized_task_metadata": row,
    }


def write_internal_task_metadata_config(
    *,
    path: Path,
    dataset: str,
    task: str,
    resolved_metadata: Mapping[str, Any],
) -> None:
    row = {
        "entity_key": resolved_metadata["entity_key"],
        "target_time_col": resolved_metadata["target_time_col"],
        "label_col": resolved_metadata["label_col"],
        "problem_type": resolved_metadata["problem_type"],
        "primary_metric": resolved_metadata["primary_metric"],
        "metric_direction": resolved_metadata["metric_direction"],
        "target_table": resolved_metadata["entity_table"],
        "child_table": resolved_metadata["child_table"],
        "child_fk": resolved_metadata["child_fk"],
        "child_event_time_col": resolved_metadata["child_event_time_col"],
        "target_lookup_column": resolved_metadata.get(
            "relation_entity_key",
            resolved_metadata["entity_key"],
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"tasks": {f"{dataset}/{task}": row}}, sort_keys=True),
        encoding="utf-8",
    )


def resolve_pipeline_task_metadata(
    *,
    dataset: str,
    task: str,
    download: bool,
    explicit_config: Path | None,
    selection_folds: int,
    pipeline_root: Path,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    resolved_path = pipeline_root / "resolved_task_metadata.json"
    explicit = _task_metadata_from_config(
        explicit_config,
        dataset_name=dataset,
        task_name=task,
    )
    dataset_obj, task_obj, _ = _load_relbench_objects(dataset, task, download)
    database = dataset_obj.get_db()
    table_dict = getattr(database, "table_dict", None)
    if not isinstance(table_dict, Mapping) or not table_dict:
        raise ValueError("missing_database_tables")
    train_df = _table_df(task_obj.get_table("train"))
    schema_fp = _schema_fingerprint(table_dict)
    train_fp = hashlib.sha256(
        json.dumps(
            {
                "columns": [str(col) for col in train_df.columns],
                "dtypes": {
                    str(col): str(train_df[col].dtype)
                    for col in train_df.columns
                },
                "rows": train_df.astype("string").fillna("<NA>").to_dict("records"),
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    preliminary = {
        "entity_key": explicit.get("entity_key")
        or getattr(task_obj, "entity_col", None),
        "target_table": explicit.get("target_table")
        or getattr(task_obj, "entity_table", None),
        "relation_threshold": explicit.get("relation_threshold", 0.98),
    }
    if preliminary["entity_key"] is None:
        preliminary["entity_key"] = getattr(task_obj.__class__, "entity_col", None)
    try:
        candidates = _verified_one_hop_relations(
            table_dict,
            entity_key=str(preliminary["entity_key"]),
            target_table=(
                None
                if preliminary.get("target_table") is None
                else str(preliminary["target_table"])
            ),
            threshold=float(preliminary["relation_threshold"]),
        )
    except ValueError as exc:
        # Preliminary relation discovery is used only for cache fingerprinting.
        # The authoritative resolver below handles event-row relation fallback.
        if str(exc) != "relation_verification_blocker":
            raise
        candidates = []

    relation_fp = _candidate_relation_fingerprint(candidates)
    if (
        resume
        and not overwrite
        and resolved_path.exists()
    ):
        cached = read_json(resolved_path)
        if resolved_metadata_reusable(
            cached,
            dataset_name=dataset,
            task_name=task,
            schema_fingerprint=schema_fp,
            train_split_fingerprint=train_fp,
            selection_folds=selection_folds,
            candidate_relation_fingerprint=relation_fp,
        ):
            return {
                "status": "reused",
                "resolved_task_metadata_file": str(resolved_path),
                "resolved_task_metadata": cached,
            }
    resolved = resolve_relbench_task_metadata(
        dataset_name=dataset,
        task_name=task,
        dataset=dataset_obj,
        task=task_obj,
        database=database,
        explicit_metadata=explicit,
        selection_folds=selection_folds,
        output_dir=pipeline_root,
        train_df=train_df,
    )
    payload = resolved.to_dict()
    return {
        "status": "completed",
        "resolved_task_metadata_file": str(resolved_path),
        "resolved_task_metadata": payload,
    }


def path_exists_status(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def resolve_generated_config_input_path(
    value: str,
    *,
    config_parent: Path,
) -> tuple[Path, str]:
    path = Path(value)
    if path.is_absolute():
        return path, "absolute"

    cwd_candidate = (Path.cwd() / path).resolve()
    config_candidate = (config_parent / path).resolve()
    parts = path.parts
    repo_root_relative = len(parts) >= 2 and parts[0] == "outputs"
    if repo_root_relative:
        candidates = [
            (cwd_candidate, "cwd_repo_root_relative"),
            (config_candidate, "config_relative_fallback"),
        ]
    else:
        candidates = [
            (config_candidate, "config_relative"),
            (cwd_candidate, "cwd_fallback"),
        ]
    for candidate, policy in candidates:
        if candidate.exists():
            return candidate, policy
    return candidates[0]


def normalize_generated_config(
    *,
    source_path: Path,
    destination_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("generated_config_must_be_mapping")
    normalized = dict(config)
    converted: list[dict[str, str]] = []
    missing: list[str] = []
    base = source_path.parent

    def convert(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        path = Path(value)
        absolute, policy = resolve_generated_config_input_path(
            value,
            config_parent=base,
        )
        converted.append({
            "from": value,
            "to": str(absolute),
            "exists": path_exists_status(absolute),
            "resolution_policy": policy,
        })
        if not absolute.exists():
            missing.append(str(absolute))
        return str(absolute)

    split = dict(normalized.get("split") or {})
    for key in ("train_target_path", "validation_target_path"):
        if key in split:
            split[key] = convert(split[key])
    normalized["split"] = split

    tables = {
        str(name): dict(raw or {})
        for name, raw in (normalized.get("tables") or {}).items()
    }
    for raw in tables.values():
        if "path" in raw:
            raw["path"] = convert(raw["path"])
    normalized["tables"] = tables

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                key: (
                    convert(value)
                    if str(key).endswith("_path")
                    and isinstance(value, str)
                    and key
                    not in {"train_target_path", "validation_target_path"}
                    else walk(value)
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [walk(value) for value in node]
        return node

    normalized = walk(normalized)
    if missing:
        raise FileNotFoundError(
            "missing_generated_config_input_paths:\n"
            + "\n".join(sorted(set(missing)))
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        yaml.safe_dump(normalized, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "source_path": str(source_path),
        "normalized_path": str(destination_path),
        "converted_paths": converted,
    }


def completed_relbench_export(export_dir: Path, config_path: Path) -> bool:
    manifest = export_dir / "export_manifest.json"
    if not manifest.exists() or not config_path.exists():
        return False
    try:
        payload = read_json(manifest)
    except (json.JSONDecodeError, OSError):
        return False
    if payload.get("status") not in {"completed", "reused"}:
        return False
    if payload.get("dataset_name") != export_dir.parent.name:
        return False
    if payload.get("task_name") != export_dir.name:
        return False
    exported = payload.get("exported_file_hashes")
    if not isinstance(exported, Mapping):
        return False
    required = {
        "target_train.parquet",
        "target_validation.parquet",
    }
    required.update(
        str(name)
        for name in exported
        if str(name).endswith(".parquet")
    )
    return all((export_dir / name).exists() for name in required)


def normalized_config_is_valid(
    path: Path,
    *,
    dataset: str,
    task: str,
) -> bool:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(raw, Mapping):
        return False
    if raw.get("dataset") != canonical_dataset_name(dataset):
        return False
    task_row = raw.get("task")
    if not isinstance(task_row, Mapping):
        return False
    if task_row.get("task_id") != task:
        return False
    paths: list[Path] = []
    split = raw.get("split") or {}
    if isinstance(split, Mapping):
        for key in ("train_target_path", "validation_target_path"):
            if key in split:
                paths.append(Path(str(split[key])))
    tables = raw.get("tables") or {}
    if isinstance(tables, Mapping):
        for table in tables.values():
            if isinstance(table, Mapping) and "path" in table:
                paths.append(Path(str(table["path"])))
    return bool(paths) and all(path.exists() for path in paths)


def completed_canonical_onboarding(
    directory: Path,
    *,
    dataset: str | None = None,
    task: str | None = None,
) -> bool:
    required = [
        directory / "onboarding_manifest.json",
        directory / "baseline_feature_config.json",
        directory / "baseline_feature_manifest.csv",
        directory / "target_with_dfs_agg_train.parquet",
        directory / "target_with_dfs_agg_val.parquet",
        directory / "temporal_safety_audit.csv",
        directory / "leakage_safety_audit.csv",
    ]
    if any(not path.exists() for path in required):
        return False
    manifest = read_json(directory / "onboarding_manifest.json")
    strategy = (
        manifest.get("materialization_strategy")
        or (manifest.get("baseline_feature_workload") or {}).get(
            "materialization_strategy"
        )
    )
    dataset_ok = (
        True
        if dataset is None
        else manifest.get("dataset") == canonical_dataset_name(dataset)
    )
    task_ok = True if task is None else manifest.get("task") == task
    return (
        manifest.get("status") in {"completed", "reused"}
        and dataset_ok
        and task_ok
        and strategy == "grouped_temporal_sweep"
    )


def nested_columns_excluded(profile_path: Path) -> list[dict[str, str]]:
    if not profile_path.exists():
        return []
    profile = read_json(profile_path)
    rows: list[dict[str, str]] = []
    for table, payload in sorted(profile.items()):
        for column in payload.get("columns", []):
            if column.get("profiling_exclusion_reason") == (
                "nested_or_unhashable_object"
            ):
                rows.append({
                    "table": str(table),
                    "column": str(column.get("column")),
                    "reason": "nested_or_unhashable_object",
                })
    return rows



def resolve_budget_tiers(
    *,
    auto_candidate_proxy: int,
    fdhg_directed_pair_proxy: int,
) -> dict[str, int]:
    """Resolve bounded task-adaptive feature budgets.

    The inputs are label-free structural proxies derived only from verified
    one-hop source relations and scalar source columns.
    """
    if auto_candidate_proxy <= 32:
        feature_budget = 8
    elif auto_candidate_proxy <= 128:
        feature_budget = 12
    else:
        feature_budget = 16

    if fdhg_directed_pair_proxy <= 16:
        max_fdhg_edges = 8
    elif fdhg_directed_pair_proxy <= 64:
        max_fdhg_edges = 16
    elif fdhg_directed_pair_proxy <= 256:
        max_fdhg_edges = 32
    else:
        max_fdhg_edges = 64

    return {
        "feature_budget": feature_budget,
        "max_fdhg_edges": max_fdhg_edges,
        "max_selected_fdhg_edges": min(max_fdhg_edges, 32),
    }


def _column_contains_nested_values(series: Any) -> bool:
    """Return True when a source column contains list/dict/set-like values."""
    try:
        values = series.dropna().head(128).tolist()
    except Exception:
        return True

    return any(
        isinstance(value, (list, tuple, dict, set))
        for value in values
    )


def _budget_eligible_columns(
    frame: Any,
    *,
    excluded_columns: set[str],
) -> list[str]:
    """Return scalar, non-key columns used only for budget estimation."""
    eligible: list[str] = []

    for raw_column in frame.columns:
        column = str(raw_column)
        lower = column.lower()

        if column in excluded_columns:
            continue

        if (
            lower == "id"
            or lower.endswith("_id")
            or lower.endswith("_key")
            or lower.endswith("timestamp")
            or lower.endswith("_time")
            or lower.endswith("_date")
        ):
            continue

        try:
            series = frame[raw_column]
        except Exception:
            continue

        if _column_contains_nested_values(series):
            continue

        try:
            non_null_count = int(series.notna().sum())
        except Exception:
            continue

        if non_null_count == 0:
            continue

        eligible.append(column)

    return eligible


def resolve_automatic_pipeline_budgets(
    *,
    dataset: str,
    task: str,
    download: bool,
    resolved_metadata: Mapping[str, Any],
    max_numeric_columns: int,
    max_categorical_columns: int,
) -> dict[str, Any]:
    """Resolve label-free task budgets from verified relational structure."""
    dataset_obj, _, _ = _load_relbench_objects(
        dataset,
        task,
        download,
    )
    database = dataset_obj.get_db()
    table_dict = getattr(database, "table_dict", None)

    if not isinstance(table_dict, Mapping) or not table_dict:
        raise ValueError("missing_database_tables_for_budget_resolution")

    entity_key = str(resolved_metadata["entity_key"])
    entity_table = str(
        resolved_metadata.get("entity_table")
        or resolved_metadata.get("target_table")
    )

    relations = _verified_one_hop_relations(
        table_dict,
        entity_key=entity_key,
        target_table=entity_table,
        threshold=float(
            resolved_metadata.get("relation_threshold", 0.98)
        ),
    )

    relation_tables: list[str] = []
    for relation in relations:
        table = (
            relation.get("child_table")
            or relation.get("source_table")
            or relation.get("table")
        )
        if table is None:
            continue

        table = str(table)
        if table in table_dict and table not in relation_tables:
            relation_tables.append(table)

    # Fall back to the resolved primary child table when relation rows use
    # an unexpected representation.
    primary_child = resolved_metadata.get("child_table")
    if (
        not relation_tables
        and primary_child is not None
        and str(primary_child) in table_dict
    ):
        relation_tables.append(str(primary_child))

    excluded_columns = {
        str(value)
        for value in (
            resolved_metadata.get("entity_key"),
            resolved_metadata.get("target_time_col"),
            resolved_metadata.get("label_col"),
            resolved_metadata.get("child_fk"),
            resolved_metadata.get("child_event_time_col"),
        )
        if value is not None
    }

    eligible_columns_by_table: dict[str, list[str]] = {}

    for table in relation_tables:
        frame = _table_df(table_dict[table])
        eligible_columns_by_table[table] = _budget_eligible_columns(
            frame,
            excluded_columns=excluded_columns,
        )

    per_relation_column_cap = (
        int(max_numeric_columns)
        + int(max_categorical_columns)
    )

    # Auto generates multiple aggregations per source column. Five is the
    # current structural proxy for count/mean-or-mode/min/max/last-style
    # candidates; it does not inspect labels or validation performance.
    auto_candidate_proxy = sum(
        min(len(columns), per_relation_column_cap) * 5
        for columns in eligible_columns_by_table.values()
    )

    fdhg_directed_pair_proxy = sum(
        len(columns) * max(len(columns) - 1, 0)
        for columns in eligible_columns_by_table.values()
    )

    resolved = resolve_budget_tiers(
        auto_candidate_proxy=auto_candidate_proxy,
        fdhg_directed_pair_proxy=fdhg_directed_pair_proxy,
    )

    return {
        "budget_policy": "auto",
        "budget_policy_version": "schema_scaled_v1",
        "verified_relation_count": len(relation_tables),
        "verified_relation_tables": relation_tables,
        "eligible_columns_by_table": eligible_columns_by_table,
        "eligible_column_count_by_table": {
            table: len(columns)
            for table, columns in eligible_columns_by_table.items()
        },
        "auto_candidate_proxy": auto_candidate_proxy,
        "fdhg_directed_pair_proxy": fdhg_directed_pair_proxy,
        "resolved_feature_budget": resolved["feature_budget"],
        "resolved_max_fdhg_edges": resolved["max_fdhg_edges"],
        "resolved_max_selected_fdhg_edges": (
            resolved["max_selected_fdhg_edges"]
        ),
        "used_label_values": False,
        "used_official_validation": False,
        "used_test_split": False,
    }


def auto_trial_task_dir(
    trial_root: Path,
    *,
    dataset: str,
    task: str,
) -> Path:
    return trial_root / task_slug(dataset, task)


def load_auto_trial_result(
    trial_root: Path,
    *,
    dataset: str,
    task: str,
    budget: int,
) -> dict[str, Any]:
    directory = auto_trial_task_dir(
        trial_root,
        dataset=dataset,
        task=task,
    )
    manifest_path = directory / "auto_onboarding_manifest.json"
    selected_path = directory / "selected_features.json"

    require_files([manifest_path, selected_path])

    manifest = read_json(manifest_path)
    score = manifest.get("inner_selection_score")
    if score is None:
        raise ValueError(
            f"missing_inner_selection_score_for_budget:{budget}"
        )

    if manifest.get("official_validation_evaluated") is not False:
        raise ValueError(
            "auto_budget_trial_evaluated_official_validation:"
            f"budget={budget}"
        )

    return {
        "budget": int(budget),
        "inner_selection_score": float(score),
        "selected_feature_count": len(
            manifest.get("selected_features") or []
        ),
        "candidate_feature_count": len(
            manifest.get("candidate_features") or []
        ),
        "fallback": bool(manifest.get("fallback", False)),
        "stopping_reason": str(
            manifest.get("stopping_reason", "")
        ),
        "output_dir": str(directory),
        "official_validation_evaluated": False,
        "official_validation_used_for_selection": False,
        "test_split_accessed": bool(
            manifest.get("test_split_accessed", False)
        ),
    }


def select_train_only_auto_budget(
    *,
    trials: Sequence[Mapping[str, Any]],
    metric_direction: str,
    classification_epsilon: float,
    regression_relative_epsilon: float,
) -> dict[str, Any]:
    if not trials:
        raise ValueError("missing_auto_budget_trials")

    normalized = [dict(row) for row in trials]
    direction = str(metric_direction).strip().lower()

    if direction in {"higher", "higher_is_better", "maximize"}:
        best_score = max(
            float(row["inner_selection_score"])
            for row in normalized
        )
        tolerance = float(classification_epsilon)
        eligible = [
            row
            for row in normalized
            if (
                best_score
                - float(row["inner_selection_score"])
            ) <= tolerance
        ]
    elif direction in {"lower", "lower_is_better", "minimize"}:
        best_score = min(
            float(row["inner_selection_score"])
            for row in normalized
        )
        tolerance = (
            abs(best_score) * float(regression_relative_epsilon)
        )
        eligible = [
            row
            for row in normalized
            if (
                float(row["inner_selection_score"])
                - best_score
            ) <= tolerance
        ]
    else:
        raise ValueError(
            f"unsupported_metric_direction:{metric_direction}"
        )

    selected = min(
        eligible,
        key=lambda row: (
            int(row["budget"]),
            float(row["inner_selection_score"]),
        ),
    )

    return {
        "selected_budget": int(selected["budget"]),
        "selected_inner_score": float(
            selected["inner_selection_score"]
        ),
        "best_observed_inner_score": float(best_score),
        "selection_tolerance": float(tolerance),
        "selection_reason": (
            "smallest_budget_within_train_only_tolerance_of_best"
        ),
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete leakage-safe FDHG RelBench pipeline: "
            "download, Auto onboarding, train-only candidate discovery, "
            "fixed-pool Independent/Greedy selection, and conservative "
            "joint model selection."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True, type=Path)

    parser.add_argument(
        "--task-metadata-config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--canonical-export-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--canonical-onboarding-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--generated-config-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--skip-canonical-dfs",
        action="store_true",
    )
    parser.add_argument(
        "--canonical-dfs-output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dfs-feature-config",
        type=Path,
        default=None,
    )

    parser.add_argument("--selection-folds", type=int, default=3)
    parser.add_argument(
        "--budget-policy",
        choices=[
            "fixed",
            "auto",
            "schema_scaled",
            "train_only_grid",
        ],
        default="fixed",
        help=(
            "Use explicit CLI budgets (fixed) or resolve task-adaptive "
            "label-free budgets from verified relational structure (auto)."
        ),
    )
    parser.add_argument("--feature-budget", type=int, default=32)
    parser.add_argument(
        "--auto-budget-grid",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16],
        help=(
            "Candidate Auto feature budgets evaluated using train-only "
            "inner CV when --budget-policy=train_only_grid."
        ),
    )
    parser.add_argument("--min-delta", type=float, default=0.0)

    parser.add_argument(
        "--selection-decoder",
        choices=["hist_gradient_boosting"],
        default="hist_gradient_boosting",
    )

    parser.add_argument("--max-fdhg-edges", type=int, default=32)
    parser.add_argument(
        "--max-selected-fdhg-edges",
        type=int,
        default=32,
    )
    parser.add_argument("--max-relations", type=int, default=8)
    parser.add_argument("--max-numeric-columns", type=int, default=8)
    parser.add_argument(
        "--max-categorical-columns",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--edge-screening-rule",
        choices=["fixed_count", "positive_fraction", "pooled_oof"],
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
        "--edge-screening-min-positive-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--edge-screening-max-relative-fold-degradation",
        type=float,
        default=float("inf"),
    )

    parser.add_argument(
        "--continuous-fdhg-mode",
        choices=["exclude", "quantile"],
        default="exclude",
    )
    parser.add_argument(
        "--continuous-fdhg-bins",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--continuous-fdhg-min-effective-bins",
        type=int,
        default=2,
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
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument("--debug", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.output_root = args.output_root.resolve()

    if args.task_metadata_config is not None:
        args.task_metadata_config = (
            args.task_metadata_config.resolve()
        )
    args.canonical_export_root = (
        args.canonical_export_root
        or (args.output_root / "_canonical_exports")
    ).resolve()
    args.canonical_onboarding_root = (
        args.canonical_onboarding_root
        or (args.output_root / "_canonical_onboarding")
    ).resolve()
    args.generated_config_root = (
        args.generated_config_root
        or (args.output_root / "_generated_configs")
    ).resolve()
    if args.canonical_dfs_output_dir is not None:
        args.canonical_dfs_output_dir = (
            args.canonical_dfs_output_dir.resolve()
        )

    if args.dfs_feature_config is not None:
        args.dfs_feature_config = args.dfs_feature_config.resolve()

    if args.selection_folds < 2:
        raise ValueError("selection_folds_must_be_at_least_2")

    if args.feature_budget < 1:
        raise ValueError("feature_budget_must_be_positive")

    if not args.auto_budget_grid:
        raise ValueError("auto_budget_grid_must_not_be_empty")

    if any(int(value) < 1 for value in args.auto_budget_grid):
        raise ValueError(
            "auto_budget_grid_values_must_be_positive"
        )

    args.auto_budget_grid = sorted({
        int(value) for value in args.auto_budget_grid
    })

    if args.max_fdhg_edges < 1:
        raise ValueError("max_fdhg_edges_must_be_positive")

    if args.max_selected_fdhg_edges < 1:
        raise ValueError(
            "max_selected_fdhg_edges_must_be_positive"
        )

    reuse_existing_canonical_dfs = (
        args.skip_canonical_dfs
        or args.canonical_dfs_output_dir is not None
    )

    run_root = (
        args.output_root
        / task_slug(args.dataset, args.task)
    )

    auto_root = run_root / "auto"
    auto_budget_trials_root = run_root / "auto_budget_trials"
    discovery_root = run_root / "discovery"
    candidate_root = run_root / "candidates"
    joint_output_root = run_root / "strategies"
    log_root = run_root / "logs"
    pipeline_root = run_root / "pipeline"

    candidate_file = (
        candidate_root / "fixed_candidate_edges.json"
    )

    selected_dir = (
        joint_output_root
        / "selected"
        / task_slug(args.dataset, args.task)
    )

    joint_selection_file = (
        joint_output_root
        / "joint"
        / task_slug(args.dataset, args.task)
        / "joint_selection.json"
    )

    pipeline_manifest_path = (
        pipeline_root / "pipeline_manifest.json"
    )
    resolved_metadata_path = pipeline_root / "resolved_task_metadata.json"
    budget_resolution_path = pipeline_root / "budget_resolution.json"
    auto_budget_selection_path = (
        pipeline_root / "auto_budget_selection.json"
    )
    internal_metadata_config_path = (
        pipeline_root / "resolved_task_metadata.yaml"
    )
    relation_screening_path = pipeline_root / "relation_screening.csv"
    generated_config_path = (
        args.generated_config_root
        / f"{task_slug(args.dataset, args.task)}_onboarding.yaml"
    )
    normalized_config_path = (
        args.generated_config_root
        / f"{task_slug(args.dataset, args.task)}_onboarding.normalized.yaml"
    )
    canonical_export_dir = (
        args.canonical_export_root / args.dataset / args.task
    )
    canonical_onboarding_dir = (
        args.canonical_dfs_output_dir
        or canonical_task_dir(
            args.canonical_onboarding_root,
            args.dataset,
            args.task,
        )
    )

    for directory in [
        run_root,
        auto_root,
        auto_budget_trials_root,
        discovery_root,
        candidate_root,
        joint_output_root,
        log_root,
        pipeline_root,
        args.canonical_export_root,
        args.canonical_onboarding_root,
        args.generated_config_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)

    stage_status: dict[str, str] = {}
    stage_commands: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Stage 00: task metadata resolution.
    # ------------------------------------------------------------------

    metadata_resolution = resolve_pipeline_task_metadata(
        dataset=args.dataset,
        task=args.task,
        download=args.download,
        explicit_config=args.task_metadata_config,
        selection_folds=args.selection_folds,
        pipeline_root=pipeline_root,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    write_internal_task_metadata_config(
        path=internal_metadata_config_path,
        dataset=args.dataset,
        task=args.task,
        resolved_metadata=metadata_resolution["resolved_task_metadata"],
    )
    explicit_metadata_validation = validate_task_metadata_config(
        args.task_metadata_config,
        dataset=args.dataset,
        task=args.task,
    )
    metadata_resolution["explicit_metadata"] = explicit_metadata_validation
    write_json(log_root / "00_metadata_resolution.json", metadata_resolution)
    stage_status["task_metadata_resolution"] = metadata_resolution["status"]

    normalized_budget_policy = (
        "schema_scaled"
        if args.budget_policy == "auto"
        else args.budget_policy
    )

    if normalized_budget_policy == "schema_scaled":
        budget_resolution = resolve_automatic_pipeline_budgets(
            dataset=args.dataset,
            task=args.task,
            download=args.download,
            resolved_metadata=metadata_resolution[
                "resolved_task_metadata"
            ],
            max_numeric_columns=args.max_numeric_columns,
            max_categorical_columns=args.max_categorical_columns,
        )
        resolved_feature_budget = int(
            budget_resolution["resolved_feature_budget"]
        )
        resolved_max_fdhg_edges = int(
            budget_resolution["resolved_max_fdhg_edges"]
        )
        resolved_max_selected_fdhg_edges = int(
            budget_resolution[
                "resolved_max_selected_fdhg_edges"
            ]
        )
    elif normalized_budget_policy == "train_only_grid":
        resolved_feature_budget = min(args.auto_budget_grid)
        resolved_max_fdhg_edges = 32
        resolved_max_selected_fdhg_edges = 32
        budget_resolution = {
            "budget_policy": "train_only_grid",
            "budget_policy_version": "train_only_grid_v1",
            "candidate_auto_feature_budgets": list(
                args.auto_budget_grid
            ),
            "resolved_feature_budget": None,
            "resolved_max_fdhg_edges": 32,
            "resolved_max_selected_fdhg_edges": 32,
            "used_label_values": True,
            "label_usage_scope": "train_only_inner_folds",
            "used_official_validation": False,
            "used_test_split": False,
        }
    else:
        resolved_feature_budget = int(args.feature_budget)
        resolved_max_fdhg_edges = int(args.max_fdhg_edges)
        resolved_max_selected_fdhg_edges = int(
            args.max_selected_fdhg_edges
        )
        budget_resolution = {
            "budget_policy": "fixed",
            "budget_policy_version": "explicit_cli_v1",
            "resolved_feature_budget": resolved_feature_budget,
            "resolved_max_fdhg_edges": resolved_max_fdhg_edges,
            "resolved_max_selected_fdhg_edges": (
                resolved_max_selected_fdhg_edges
            ),
            "used_label_values": False,
            "used_official_validation": False,
            "used_test_split": False,
        }

    write_json(budget_resolution_path, budget_resolution)
    write_json(
        log_root / "00_budget_resolution.json",
        budget_resolution,
    )
    stage_status["budget_resolution"] = "completed"

    print("BUDGET_POLICY", budget_resolution["budget_policy"])
    print(
        "RESOLVED_FEATURE_BUDGET",
        (
            "pending_train_only_grid"
            if normalized_budget_policy == "train_only_grid"
            else resolved_feature_budget
        ),
    )
    print("RESOLVED_MAX_FDHG_EDGES", resolved_max_fdhg_edges)
    print(
        "RESOLVED_MAX_SELECTED_FDHG_EDGES",
        resolved_max_selected_fdhg_edges,
    )

    # ------------------------------------------------------------------
    # Stage 01: RelBench canonical export.
    # ------------------------------------------------------------------

    export_command = [
        sys.executable,
        "-m",
        "fdhg.cli.export_relbench_v1",
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--output-root",
        str(args.canonical_export_root),
        "--config-output",
        str(generated_config_path),
        "--write",
        "--download" if args.download else "--no-download",
    ]

    export_command.extend([
        "--task-metadata-config",
        str(internal_metadata_config_path),
    ])

    if args.overwrite:
        export_command.append("--overwrite")

    stage_commands["relbench_export"] = export_command

    if reuse_existing_canonical_dfs:
        stage_status["relbench_export"] = "skipped"
    elif (
        args.resume
        and completed_relbench_export(
            canonical_export_dir,
            generated_config_path,
        )
        and not args.overwrite
    ):
        print("SKIP relbench_export: existing completed artifact")
        stage_status["relbench_export"] = "reused"
    else:
        run_command(
            export_command,
            log_path=log_root / "01_relbench_export.log",
            env=env,
            dry_run=args.dry_run,
        )
        stage_status["relbench_export"] = (
            "planned" if args.dry_run else "completed"
        )

    if not reuse_existing_canonical_dfs and not args.dry_run:
        require_files([
            canonical_export_dir / "export_manifest.json",
            generated_config_path,
        ])

    # ------------------------------------------------------------------
    # Stage 02: generated config path normalization.
    # ------------------------------------------------------------------

    stage_commands["canonical_config_normalization"] = [
        "normalize_generated_config",
        str(generated_config_path),
        str(normalized_config_path),
    ]

    if reuse_existing_canonical_dfs:
        stage_status["canonical_config_normalization"] = "skipped"
    elif (
        args.resume
        and normalized_config_is_valid(
            normalized_config_path,
            dataset=args.dataset,
            task=args.task,
        )
        and not args.overwrite
    ):
        print(
            "SKIP config_normalization: existing normalized config"
        )
        stage_status["canonical_config_normalization"] = "reused"
    else:
        if not args.dry_run:
            normalization_report = normalize_generated_config(
                source_path=generated_config_path,
                destination_path=normalized_config_path,
            )
            write_json(
                log_root / "02_config_normalization.json",
                normalization_report,
            )
        stage_status["canonical_config_normalization"] = (
            "planned" if args.dry_run else "completed"
        )

    # ------------------------------------------------------------------
    # Stage 03: canonical DFS onboarding.
    # ------------------------------------------------------------------

    canonical_command = [
        sys.executable,
        "-m",
        "fdhg.cli.onboard_dataset",
        "--config",
        str(normalized_config_path),
        "--output-root",
        str(args.canonical_onboarding_root),
        "--write",
    ]

    if args.overwrite:
        canonical_command.append("--overwrite")

    stage_commands["canonical_dfs_onboarding"] = canonical_command

    if reuse_existing_canonical_dfs:
        if not completed_canonical_onboarding(
            canonical_onboarding_dir,
            dataset=args.dataset,
            task=args.task,
        ):
            raise FileNotFoundError(
                "missing_canonical_dfs_output_dir:"
                f"{canonical_onboarding_dir}"
            )
        stage_status["canonical_dfs_onboarding"] = "skipped"
    elif (
        args.resume
        and completed_canonical_onboarding(
            canonical_onboarding_dir,
            dataset=args.dataset,
            task=args.task,
        )
        and not args.overwrite
    ):
        print(
            "SKIP canonical_dfs_onboarding: "
            "existing completed artifact"
        )
        stage_status["canonical_dfs_onboarding"] = "reused"
    else:
        run_command(
            canonical_command,
            log_path=log_root / "03_canonical_dfs_onboarding.log",
            env=env,
            dry_run=args.dry_run,
        )
        stage_status["canonical_dfs_onboarding"] = (
            "planned" if args.dry_run else "completed"
        )

    if not args.dry_run:
        if not completed_canonical_onboarding(
            canonical_onboarding_dir,
            dataset=args.dataset,
            task=args.task,
        ):
            raise RuntimeError(
                "stage_failed:canonical_dfs_onboarding:"
                f"{canonical_onboarding_dir}"
            )

    # ------------------------------------------------------------------
    # Stage 04: Auto onboarding and RelBench download/load.
    # ------------------------------------------------------------------

    def build_auto_command(
        *,
        output_root: Path,
        feature_budget: int,
        selection_only: bool,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "fdhg.cli.auto_onboard_relbench",
            "--dataset",
            args.dataset,
            "--task",
            args.task,
            "--output-root",
            str(output_root),
            "--selection-folds",
            str(args.selection_folds),
            "--feature-budget",
            str(feature_budget),
            "--min-delta",
            str(args.min_delta),
            "--selection-decoder",
            args.selection_decoder,
            "--max-relations",
            str(args.max_relations),
            "--max-numeric-columns",
            str(args.max_numeric_columns),
            "--max-categorical-columns",
            str(args.max_categorical_columns),
            "--write",
            "--task-metadata-config",
            str(internal_metadata_config_path),
            "--download" if args.download else "--no-download",
        ]

        if selection_only:
            command.append("--selection-only")

        if args.overwrite:
            command.append("--overwrite")

        return command

    if normalized_budget_policy == "train_only_grid":
        auto_budget_trials: list[dict[str, Any]] = []

        for budget in args.auto_budget_grid:
            trial_root = (
                auto_budget_trials_root / f"budget_{budget}"
            )
            trial_command = build_auto_command(
                output_root=trial_root,
                feature_budget=budget,
                selection_only=True,
            )
            stage_commands[
                f"auto_budget_trial_{budget}"
            ] = trial_command

            if (
                args.resume
                and completed_auto_artifact(
                    trial_root,
                    args.dataset,
                    args.task,
                )
                and not args.overwrite
            ):
                print(
                    "SKIP auto_budget_trial:",
                    budget,
                    "existing completed artifact",
                )
                stage_status[
                    f"auto_budget_trial_{budget}"
                ] = "reused"
            else:
                run_command(
                    trial_command,
                    log_path=(
                        log_root
                        / f"04_auto_budget_trial_{budget}.log"
                    ),
                    env=env,
                    dry_run=args.dry_run,
                )
                stage_status[
                    f"auto_budget_trial_{budget}"
                ] = (
                    "planned"
                    if args.dry_run
                    else "completed"
                )

            if not args.dry_run:
                auto_budget_trials.append(
                    load_auto_trial_result(
                        trial_root,
                        dataset=args.dataset,
                        task=args.task,
                        budget=budget,
                    )
                )

        if args.dry_run:
            resolved_feature_budget = min(
                args.auto_budget_grid
            )
            auto_budget_selection = {
                "policy": "train_only_grid",
                "status": "planned",
                "candidate_budgets": list(
                    args.auto_budget_grid
                ),
                "selected_budget": None,
                "official_validation_used_for_selection": False,
                "official_validation_evaluated_during_trials": False,
                "test_split_accessed": False,
            }
        else:
            resolved_metadata = metadata_resolution[
                "resolved_task_metadata"
            ]
            auto_budget_choice = (
                select_train_only_auto_budget(
                    trials=auto_budget_trials,
                    metric_direction=resolved_metadata[
                        "metric_direction"
                    ],
                    classification_epsilon=(
                        args.classification_epsilon
                    ),
                    regression_relative_epsilon=(
                        args.regression_relative_epsilon
                    ),
                )
            )
            resolved_feature_budget = int(
                auto_budget_choice["selected_budget"]
            )

            auto_budget_selection = {
                "policy": "train_only_grid",
                "policy_version": "train_only_grid_v1",
                "candidate_budgets": list(
                    args.auto_budget_grid
                ),
                "metric": resolved_metadata[
                    "primary_metric"
                ],
                "metric_direction": resolved_metadata[
                    "metric_direction"
                ],
                "trials": auto_budget_trials,
                **auto_budget_choice,
                "official_validation_used_for_selection": False,
                "official_validation_evaluated_during_trials": False,
                "test_split_accessed": any(
                    bool(row["test_split_accessed"])
                    for row in auto_budget_trials
                ),
            }

        write_json(
            auto_budget_selection_path,
            auto_budget_selection,
        )

        budget_resolution[
            "resolved_feature_budget"
        ] = (
            None
            if args.dry_run
            else resolved_feature_budget
        )
        budget_resolution[
            "auto_budget_selection_file"
        ] = str(auto_budget_selection_path)
        write_json(
            budget_resolution_path,
            budget_resolution,
        )
        write_json(
            log_root / "00_budget_resolution.json",
            budget_resolution,
        )

        print(
            "TRAIN_ONLY_SELECTED_FEATURE_BUDGET",
            (
                "pending"
                if args.dry_run
                else resolved_feature_budget
            ),
        )

    auto_command = build_auto_command(
        output_root=auto_root,
        feature_budget=resolved_feature_budget,
        selection_only=False,
    )

    stage_commands["auto_onboarding"] = auto_command

    if (
        args.resume
        and completed_auto_artifact(
            auto_root,
            args.dataset,
            args.task,
        )
        and not args.overwrite
    ):
        print("SKIP auto_onboarding: existing completed artifact")
        stage_status["auto_onboarding"] = "reused"
    else:
        run_command(
            auto_command,
            log_path=log_root / "04_auto_onboarding.log",
            env=env,
            dry_run=args.dry_run,
        )
        stage_status["auto_onboarding"] = (
            "planned" if args.dry_run else "completed"
        )

    if not args.dry_run:
        auto_task_dir = (
            auto_root / task_slug(args.dataset, args.task)
        )

        require_files([
            auto_task_dir / "selected_features.json",
            auto_task_dir / "auto_onboarding_manifest.json",
            auto_task_dir / "official_validation_metrics.json",
            auto_task_dir
            / "official_validation_predictions.parquet",
        ])

        final_auto_manifest = read_json(
            auto_task_dir / "auto_onboarding_manifest.json"
        )
        if (
            final_auto_manifest.get(
                "official_validation_evaluated"
            )
            is not True
        ):
            raise RuntimeError(
                "final_auto_did_not_evaluate_official_validation"
            )

    # ------------------------------------------------------------------
    # Stage 2: Candidate discovery bootstrap.
    #
    # No candidate replay file is supplied here. auto_fdhg_relbench
    # therefore discovers candidates from the earliest inner-train fold.
    # ------------------------------------------------------------------

    discovery_command = [
        sys.executable,
        "-m",
        "fdhg.cli.auto_fdhg_relbench",
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--output-root",
        str(discovery_root),
        "--auto-output-root",
        str(auto_root),
        "--canonical-onboarding-root",
        str(
            canonical_onboarding_dir
            if args.canonical_dfs_output_dir is not None
            else args.canonical_onboarding_root
        ),
        "--selection-folds",
        str(args.selection_folds),
        "--feature-budget",
        str(resolved_feature_budget),
        "--min-delta",
        str(args.min_delta),
        "--selection-decoder",
        args.selection_decoder,
        "--max-fdhg-edges",
        str(resolved_max_fdhg_edges),
        "--max-selected-fdhg-edges",
        str(resolved_max_selected_fdhg_edges),
        "--max-relations",
        str(args.max_relations),
        "--max-numeric-columns",
        str(args.max_numeric_columns),
        "--max-categorical-columns",
        str(args.max_categorical_columns),
        "--enable-edge-screening",
        "--edge-screening-rule",
        args.edge_screening_rule,
        "--edge-screening-min-delta",
        str(args.edge_screening_min_delta),
        "--edge-screening-min-positive-folds",
        str(args.edge_screening_min_positive_folds),
        "--edge-screening-min-positive-fraction",
        str(args.edge_screening_min_positive_fraction),
        "--edge-screening-max-relative-fold-degradation",
        str(args.edge_screening_max_relative_fold_degradation),
        "--edge-selection-strategy",
        "independent",
        "--continuous-fdhg-mode",
        args.continuous_fdhg_mode,
        "--continuous-fdhg-bins",
        str(args.continuous_fdhg_bins),
        "--continuous-fdhg-min-effective-bins",
        str(args.continuous_fdhg_min_effective_bins),
        "--no-download",
        "--write",
    ]

    if args.dfs_feature_config is not None:
        discovery_command.extend([
            "--dfs-feature-config",
            str(args.dfs_feature_config),
        ])
    if args.debug:
        discovery_command.append("--debug")

    if args.overwrite:
        discovery_command.append("--overwrite")

    stage_commands["candidate_discovery"] = discovery_command

    if (
        args.resume
        and completed_discovery_artifact(
            discovery_root,
            args.dataset,
            args.task,
        )
        and not args.overwrite
    ):
        print("SKIP candidate_discovery: existing completed artifact")
        stage_status["candidate_discovery"] = "reused"
    else:
        run_command(
            discovery_command,
            log_path=log_root / "05_candidate_discovery.log",
            env=env,
            dry_run=args.dry_run,
        )
        stage_status["candidate_discovery"] = (
            "planned" if args.dry_run else "completed"
        )

    discovery_task_dir = (
        discovery_root
        / task_slug(args.dataset, args.task)
    )

    if not args.dry_run:
        require_files([
            discovery_task_dir / "candidate_discovery.json",
            discovery_task_dir / "manifest.json",
        ])

    # ------------------------------------------------------------------
    # Stage 3: Export and freeze candidate pool.
    # ------------------------------------------------------------------

    export_command = [
        sys.executable,
        "-m",
        "fdhg.cli.export_fdhg_candidate_edges",
        "--input-output-dir",
        str(discovery_task_dir),
        "--output-file",
        str(candidate_file),
    ]

    stage_commands["candidate_export"] = export_command

    if (
        args.resume
        and candidate_file.exists()
        and not args.overwrite
    ):
        print("SKIP candidate_export: existing candidate file")
        stage_status["candidate_export"] = "reused"
    else:
        run_command(
            export_command,
            log_path=log_root / "06_candidate_export.log",
            env=env,
            dry_run=args.dry_run,
        )
        stage_status["candidate_export"] = (
            "planned" if args.dry_run else "completed"
        )

    if not args.dry_run:
        require_files([candidate_file])

        candidate_payload = read_json(candidate_file)
        if isinstance(candidate_payload, list):
            candidate_count = len(candidate_payload)
        else:
            edges = (
                candidate_payload.get("edges")
                or candidate_payload.get("candidate_edges")
                or candidate_payload.get("accepted_edges")
                or []
            )
            candidate_count = len(edges)

        if candidate_count == 0:
            print(
                "WARNING: candidate discovery produced zero accepted edges. "
                "The downstream joint gate may select Auto or DFS."
            )
    else:
        candidate_count = None

    # ------------------------------------------------------------------
    # Stage 4: Fixed-pool Independent + Greedy + conservative joint gate.
    # ------------------------------------------------------------------

    joint_command = [
        sys.executable,
        "-m",
        "fdhg.cli.auto_fdhg_joint_relbench",
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--output-root",
        str(joint_output_root),
        "--auto-output-root",
        str(auto_root),
        "--canonical-onboarding-root",
        str(
            canonical_onboarding_dir
            if args.canonical_dfs_output_dir is not None
            else args.canonical_onboarding_root
        ),
        "--fdhg-candidate-edges-file",
        str(candidate_file),
        "--selection-folds",
        str(args.selection_folds),
        "--feature-budget",
        str(resolved_feature_budget),
        "--max-fdhg-edges",
        str(resolved_max_fdhg_edges),
        "--max-selected-fdhg-edges",
        str(resolved_max_selected_fdhg_edges),
        "--edge-screening-rule",
        args.edge_screening_rule,
        "--edge-screening-min-delta",
        str(args.edge_screening_min_delta),
        "--edge-screening-min-positive-folds",
        str(args.edge_screening_min_positive_folds),
        "--continuous-fdhg-mode",
        args.continuous_fdhg_mode,
        "--classification-epsilon",
        str(args.classification_epsilon),
        "--regression-relative-epsilon",
        str(args.regression_relative_epsilon),
        "--exact-tie-tolerance",
        str(args.exact_tie_tolerance),
        "--no-download",
    ]

    if args.overwrite:
        joint_command.append("--overwrite")
    if args.debug:
        joint_command.append("--debug")

    stage_commands["joint_selection"] = joint_command

    if (
        args.resume
        and completed_joint_artifact(
            joint_output_root,
            args.dataset,
            args.task,
        )
        and not args.overwrite
    ):
        print("SKIP joint_selection: existing completed artifact")
        stage_status["joint_selection"] = "reused"
    else:
        run_command(
            joint_command,
            log_path=log_root / "07_joint_selection.log",
            env=env,
            dry_run=args.dry_run,
        )
        stage_status["joint_selection"] = (
            "planned" if args.dry_run else "completed"
        )

    # ------------------------------------------------------------------
    # Stage 5: Final safety and provenance manifest.
    # ------------------------------------------------------------------

    if args.dry_run:
        resolved_metadata = metadata_resolution["resolved_task_metadata"]
        pipeline_manifest = {
            "pipeline_version": PIPELINE_VERSION,
        "budget_policy": budget_resolution["budget_policy"],
        "budget_resolution_file": str(budget_resolution_path),
        "auto_budget_selection_file": (
            str(auto_budget_selection_path)
            if normalized_budget_policy == "train_only_grid"
            else None
        ),
        "resolved_feature_budget": resolved_feature_budget,
        "resolved_max_fdhg_edges": resolved_max_fdhg_edges,
        "resolved_max_selected_fdhg_edges": (
            resolved_max_selected_fdhg_edges
        ),
            "status": "dry_run",
            "dataset": args.dataset,
            "task": args.task,
            "output_root": str(run_root),
            "stage_status": stage_status,
            "stage_commands": stage_commands,
            "download_requested": args.download,
            "selection_folds": args.selection_folds,
            "feature_budget": args.feature_budget,
            "max_fdhg_edges": args.max_fdhg_edges,
            "max_selected_fdhg_edges": (
                args.max_selected_fdhg_edges
            ),
            "official_validation_was_used_for_selection": False,
            "official_validation_used_for_selection": False,
            "test_split_accessed": False,
            "canonical_export_dir": str(canonical_export_dir),
            "canonical_config_path": str(generated_config_path),
            "normalized_config_path": str(normalized_config_path),
            "canonical_onboarding_dir": str(canonical_onboarding_dir),
            "task_metadata_resolution_status": metadata_resolution["status"],
            "resolved_task_metadata_file": str(resolved_metadata_path),
            "task_metadata_sha256": sha256_file(resolved_metadata_path),
            "task_metadata_source_by_field": resolved_metadata.get(
                "source_by_field",
                {},
            ),
            "relation_candidate_count": len(
                resolved_metadata.get("candidate_relations_considered", [])
            ),
            "relation_selection_method": resolved_metadata.get(
                "relation_selection_method"
            ),
            "selected_relation": {
                "child_table": resolved_metadata.get("child_table"),
                "child_fk": resolved_metadata.get("child_fk"),
                "child_event_time_col": resolved_metadata.get(
                    "child_event_time_col"
                ),
            },
            "relation_screening_file": str(relation_screening_path),
            "official_validation_used_for_resolution": False,
            "test_split_accessed_during_resolution": False,
        }

        write_json(
            pipeline_manifest_path,
            pipeline_manifest,
        )

        print()
        print("DRY RUN COMPLETE")
        print("PIPELINE MANIFEST", pipeline_manifest_path)
        return

    require_files([
        joint_selection_file,
        selected_dir / "manifest.json",
        selected_dir / "selected_variant.json",
    ])
    require_files([resolved_metadata_path])

    joint_selection = read_json(joint_selection_file)
    selected_manifest = read_json(
        selected_dir / "manifest.json"
    )
    selected_variant = read_json(
        selected_dir / "selected_variant.json"
    )
    resolved_metadata = read_json(resolved_metadata_path)

    test_split_accessed = bool(
        joint_selection.get("test_split_accessed", False)
        or selected_manifest.get("test_split_accessed", False)
        or selected_variant.get("test_split_accessed", False)
    )

    official_validation_used = bool(
        joint_selection.get(
            "official_validation_was_used_for_selection",
            False,
        )
        or selected_manifest.get(
            "official_validation_was_used_for_selection",
            False,
        )
        or selected_variant.get(
            "official_validation_was_used_for_selection",
            False,
        )
    )

    if test_split_accessed:
        raise RuntimeError(
            "safety_check_failed:test_split_accessed"
        )

    if official_validation_used:
        raise RuntimeError(
            "safety_check_failed:"
            "official_validation_used_for_selection"
        )

    auto_task_dir = (
        auto_root / task_slug(args.dataset, args.task)
    )
    canonical_onboarding_manifest = (
        canonical_onboarding_dir / "onboarding_manifest.json"
    )
    independent_score = (
        joint_selection.get("mean_scores", {})
        .get("auto_plus_fdhg_independent")
    )
    greedy_score = (
        joint_selection.get("mean_scores", {})
        .get("auto_plus_fdhg_greedy")
    )

    pipeline_manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "budget_policy": budget_resolution["budget_policy"],
        "budget_resolution_file": str(budget_resolution_path),
        "auto_budget_selection_file": (
            str(auto_budget_selection_path)
            if normalized_budget_policy == "train_only_grid"
            else None
        ),
        "resolved_feature_budget": resolved_feature_budget,
        "resolved_max_fdhg_edges": resolved_max_fdhg_edges,
        "resolved_max_selected_fdhg_edges": (
            resolved_max_selected_fdhg_edges
        ),
        "status": "completed",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": args.dataset,
        "task": args.task,
        "output_root": str(run_root),
        "stages": stage_status,
        "stage_status": stage_status,
        "stage_commands": stage_commands,
        "download_requested": args.download,
        "selection_folds": args.selection_folds,
        "feature_budget": args.feature_budget,
        "max_fdhg_edges": args.max_fdhg_edges,
        "max_selected_fdhg_edges": (
            args.max_selected_fdhg_edges
        ),
        "edge_screening_rule": args.edge_screening_rule,
        "edge_screening_min_delta": (
            args.edge_screening_min_delta
        ),
        "edge_screening_min_positive_folds": (
            args.edge_screening_min_positive_folds
        ),
        "classification_epsilon": (
            args.classification_epsilon
        ),
        "regression_relative_epsilon": (
            args.regression_relative_epsilon
        ),
        "canonical_export_dir": str(canonical_export_dir),
        "canonical_config_path": str(generated_config_path),
        "normalized_config_path": str(normalized_config_path),
        "canonical_onboarding_dir": str(canonical_onboarding_dir),
        "task_metadata_resolution_status": metadata_resolution["status"],
        "resolved_task_metadata_file": str(resolved_metadata_path),
        "task_metadata_sha256": sha256_file(resolved_metadata_path),
        "task_metadata_source_by_field": resolved_metadata.get(
            "source_by_field",
            {},
        ),
        "relation_candidate_count": len(
            resolved_metadata.get("candidate_relations_considered", [])
        ),
        "relation_selection_method": resolved_metadata.get(
            "relation_selection_method"
        ),
        "selected_relation": {
            "child_table": resolved_metadata.get("child_table"),
            "child_fk": resolved_metadata.get("child_fk"),
            "child_event_time_col": resolved_metadata.get(
                "child_event_time_col"
            ),
        },
        "relation_screening_file": str(relation_screening_path),
        "official_validation_used_for_resolution": False,
        "test_split_accessed_during_resolution": False,
        "canonical_onboarding_manifest": str(
            canonical_onboarding_manifest
        ),
        "canonical_onboarding_manifest_sha256": sha256_file(
            canonical_onboarding_manifest
        ),
        "auto_output_dir": str(auto_task_dir),
        "auto_selected_features_file": str(
            auto_task_dir / "selected_features.json"
        ),
        "auto_selected_features_sha256": sha256_file(
            auto_task_dir / "selected_features.json"
        ),
        "candidate_discovery_output_dir": str(
            discovery_task_dir
        ),
        "candidate_discovery_scope": (
            "earliest_inner_train_fold"
        ),
        "candidate_file": str(candidate_file),
        "candidate_file_sha256": sha256_file(candidate_file),
        "candidate_edge_count": candidate_count,
        "joint_selection_file": str(joint_selection_file),
        "independent_score": independent_score,
        "independent_inner_score": independent_score,
        "independent_edge_count": joint_selection.get(
            "independent_selected_edge_count"
        ),
        "greedy_score": greedy_score,
        "greedy_inner_score": greedy_score,
        "greedy_edge_count": joint_selection.get(
            "greedy_selected_edge_count"
        ),
        "selection_tolerance": joint_selection.get(
            "selection_tolerance"
        ),
        "selected_output_dir": str(selected_dir),
        "selected_variant": joint_selection.get(
            "selected_variant"
        ),
        "selected_source_strategy": joint_selection.get(
            "selected_source_strategy"
        ),
        "selected_edge_count": joint_selection.get(
            "selected_edge_count"
        ),
        "selected_edge_ids": joint_selection.get(
            "selected_edge_ids",
            [],
        ),
        "selection_reason": joint_selection.get(
            "selection_reason"
        ),
        "independent_selected_edge_count": (
            joint_selection.get(
                "independent_selected_edge_count"
            )
        ),
        "greedy_selected_edge_count": (
            joint_selection.get(
                "greedy_selected_edge_count"
            )
        ),
        "official_validation_was_used_for_selection": False,
        "official_validation_used_for_selection": False,
        "test_split_accessed": False,
        "independent_candidate_file_sha256": joint_selection.get(
            "independent_candidate_file_sha256"
        ),
        "greedy_candidate_file_sha256": joint_selection.get(
            "greedy_candidate_file_sha256"
        ),
        "same_candidate_pool_verified": bool(
            joint_selection.get("same_candidate_pool_verified")
        ),
        "nested_columns_excluded": nested_columns_excluded(
            canonical_onboarding_dir / RAW_PROFILE_SCHEMA_PATH
        ),
        "safety_status": "pass",
    }

    write_json(
        pipeline_manifest_path,
        pipeline_manifest,
    )

    print()
    print("=" * 100)
    print("FDHG END-TO-END PIPELINE COMPLETE")
    print("=" * 100)
    print("DATASET", args.dataset)
    print("TASK", args.task)
    print(
        "SELECTED_VARIANT",
        pipeline_manifest["selected_variant"],
    )
    print(
        "SELECTED_SOURCE_STRATEGY",
        pipeline_manifest["selected_source_strategy"],
    )
    print(
        "SELECTED_EDGE_COUNT",
        pipeline_manifest["selected_edge_count"],
    )
    print(
        "CANDIDATE_EDGE_COUNT",
        pipeline_manifest["candidate_edge_count"],
    )
    print(
        "OFFICIAL_VALIDATION_USED_FOR_SELECTION",
        False,
    )
    print("TEST_SPLIT_ACCESSED", False)
    print("SELECTED_OUTPUT_DIR", selected_dir)
    print("PIPELINE_MANIFEST", pipeline_manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--debug" in sys.argv:
            traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
