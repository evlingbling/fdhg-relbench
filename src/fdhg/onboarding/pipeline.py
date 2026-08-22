from __future__ import annotations

import csv
import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml
import numpy as np

from fdhg.compiler.candidate_safety import AUDIT_COLUMNS


ONBOARDING_VERSION = "onboarding-v1"
BASELINE_AUTO = "baseline_auto"


@dataclass(frozen=True)
class OnboardingReport:
    dataset: str
    task: str
    status: str
    output_dir: Path
    blockers: tuple[str, ...]
    reused: bool = False
    dry_run: bool = False
    planned_feature_columns: tuple[str, ...] = ()
    workload: Mapping[str, Any] | None = None


def onboard_dataset(
    *,
    config_path: Path,
    output_root: Path,
    write: bool = False,
    overwrite: bool = False,
) -> OnboardingReport:
    config = _load_yaml(config_path)
    dataset = str(config["dataset"])
    task = dict(config["task"])
    task_id = str(task["task_id"])
    output_dir = output_root / f"{dataset}_{task_id}"
    blockers: list[str] = []
    try:
        tables = _load_tables(config_path, config)
        profiles = profile_tables(tables)

        task_cfg = dict(config.get("task") or {})

        split_strategy = str(
            (config.get("split") or {}).get("strategy", "")
        )
        official_split = (
            load_official_pre_split_targets(config_path, config)
            if split_strategy == "official_pre_split"
            else None
        )

        explicit_relation = all(
            task_cfg.get(name) is not None
            for name in (
                "child_table",
                "child_fk",
                "child_event_time_col",
            )
        )

        is_dbinfer_retailrocket_cvr = (
            dataset == "dbinfer-retailrocket"
            and task_id == "cvr"
        )

        if is_dbinfer_retailrocket_cvr:
            if official_split is None:
                raise ValueError(
                    "dbinfer_requires_official_pre_split"
                )

            from .dbinfer_v1 import (
                RETAILROCKET_CVR_RELATION_SPECS,
                discover_dbinfer_event_relations,
            )

            dbinfer_candidates = discover_dbinfer_event_relations(
                table_dict={
                    name: table.frame
                    for name, table in tables.items()
                },
                train_targets=official_split["train"],
                target_time_col=str(
                    task_cfg["target_time_col"]
                ),
                relation_specs=RETAILROCKET_CVR_RELATION_SPECS,
            )

            accepted_candidates = [
                row
                for row in dbinfer_candidates
                if row["status"] == "accepted"
            ]

            if not accepted_candidates:
                raise ValueError(
                    "no_dbinfer_event_relation_candidates"
                )

            # Preserve the explicitly configured baseline relation while
            # recording the complete audited DBInfer relation candidate set.
            baseline_child_table = str(
                task_cfg["child_table"]
            )
            baseline_child_fk = str(
                task_cfg["child_fk"]
            )
            baseline_lookup = str(
                task_cfg.get(
                    "relation_entity_key",
                    task_cfg["entity_key"],
                )
            )

            baseline_matches = [
                row
                for row in accepted_candidates
                if (
                    str(row["child_table"])
                    == baseline_child_table
                    and str(row["child_fk"])
                    == baseline_child_fk
                    and str(row["target_lookup_column"])
                    == baseline_lookup
                )
            ]

            if len(baseline_matches) != 1:
                raise ValueError(
                    "ambiguous_dbinfer_baseline_relation"
                )

            relations = {
                "candidates": dbinfer_candidates,
                "accepted": baseline_matches[0],
            }

        elif explicit_relation:
            # An explicitly configured relation is validated by resolve_task().
            # Do not require unrelated configured FKs in the dataset schema
            # to pass automatic relation discovery.
            relations = {
                "candidates": [],
                "accepted": None,
            }

        else:
            relations = discover_relations(
                tables=tables,
                profiles=profiles,
                configured=config.get("tables", {}),
                threshold=float(config.get("relation_threshold", 0.98)),
            )
        resolved = resolve_task(
            config,
            tables=tables,
            relations=relations,
            official_targets=official_split,
        )
        split = (
            official_split
            if official_split is not None
            else split_targets(
                target=tables[resolved["target_table"]].frame,
                target_time_col=resolved["target_time_col"],
                train_end=resolved["train_end"],
                validation_end=resolved["validation_end"],
            )
        )
        feature_result = plan_baseline_features(
            dataset=dataset,
            task=task_id,
            target_table=resolved["target_table"],
            child_table=resolved["child_table"],
            child=tables[resolved["child_table"]].frame,
            entity_key=resolved["entity_key"],
            relation_entity_key=resolved.get(
                "relation_entity_key",
                resolved["entity_key"],
            ),
            child_fk=resolved["child_fk"],
            child_time_col=resolved["child_time_col"],
            target_time_col=resolved["target_time_col"],
            label_col=resolved["label_col"],
            strict_before=bool(
                resolved.get(
                    "strict_before",
                    False,
                )
            ),
            numeric_col=resolved["numeric_col"],
            child_primary_key=tables[resolved["child_table"]].primary_key,
            train_row_count=len(split["train"]),
            validation_row_count=len(split["validation"]),
        )
        if not feature_result["features"]:
            blockers.append("no_supported_baseline_primitives")
    except Exception as exc:
        blockers.append(str(exc))
        return OnboardingReport(
            dataset=dataset,
            task=task_id,
            status="blocked",
            output_dir=output_dir,
            blockers=tuple(blockers),
            dry_run=not write,
        )
    if blockers:
        return OnboardingReport(
            dataset=dataset,
            task=task_id,
            status="blocked",
            output_dir=output_dir,
            blockers=tuple(blockers),
            dry_run=not write,
        )
    manifest_identity = _manifest_identity(
        config_path=config_path,
        config=config,
        tables=tables,
        feature_result=feature_result,
    )
    if output_dir.exists() and not overwrite:
        manifest_path = output_dir / "onboarding_manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("reuse_identity") == manifest_identity:
                return OnboardingReport(
                    dataset=dataset,
                    task=task_id,
                    status="reused",
                    output_dir=output_dir,
                    blockers=(),
                    reused=True,
                    dry_run=not write,
                    planned_feature_columns=tuple(
                        feature_result["feature_columns"]
                    ),
                    workload=existing.get(
                        "baseline_feature_workload",
                        feature_result["workload"],
                    ),
                )
        if write:
            raise FileExistsError(output_dir)
    if not write:
        return OnboardingReport(
            dataset=dataset,
            task=task_id,
            status="dry_run_ready",
            output_dir=output_dir,
            blockers=(),
            dry_run=True,
            planned_feature_columns=tuple(feature_result["feature_columns"]),
            workload=feature_result["workload"],
        )
    feature_result = build_baseline_features(
        dataset=dataset,
        task=task_id,
        target_table=resolved["target_table"],
        target_train=split["train"],
        target_val=split["validation"],
        child_table=resolved["child_table"],
        child=tables[resolved["child_table"]].frame,
        entity_key=resolved["entity_key"],
        relation_entity_key=resolved.get(
            "relation_entity_key",
            resolved["entity_key"],
        ),
        child_fk=resolved["child_fk"],
        child_time_col=resolved["child_time_col"],
        target_time_col=resolved["target_time_col"],
        label_col=resolved["label_col"],
        strict_before=bool(
            resolved.get(
                "strict_before",
                False,
            )
        ),
        numeric_col=resolved["numeric_col"],
        child_primary_key=tables[resolved["child_table"]].primary_key,
    )
    staging = output_dir.parent / f"_{output_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_outputs(
            staging=staging,
            config_path=config_path,
            config=config,
            tables=tables,
            profiles=profiles,
            relations=relations,
            resolved=resolved,
            split=split,
            feature_result=feature_result,
            reuse_identity=manifest_identity,
        )
        _validate_publication(staging)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return OnboardingReport(
        dataset=dataset,
        task=task_id,
        status="completed",
        output_dir=output_dir,
        blockers=(),
        planned_feature_columns=tuple(feature_result["feature_columns"]),
        workload=feature_result["workload"],
    )


@dataclass(frozen=True)
class RawTable:
    name: str
    path: Path
    primary_key: str | None
    frame: pd.DataFrame
    file_sha256: str
    foreign_keys: tuple[Mapping[str, Any], ...]
    event_time_col: str | None = None



def is_hashable_scalar(value: Any) -> bool:
    if isinstance(value, (np.ndarray, list, dict, set)):
        return False

    try:
        hash(value)
    except TypeError:
        return False

    return True


def contains_nested_or_unhashable_values(series: pd.Series) -> bool:
    """Return whether sampled non-null values are nested or unhashable."""
    non_null = series.dropna()

    for value in non_null.head(32):
        if not is_hashable_scalar(value):
            return True

    return False


def safe_unique_count(series: pd.Series) -> int | None:
    """Compute cardinality only for scalar, hashable columns."""
    non_null = series.dropna()

    if non_null.empty:
        return 0

    if contains_nested_or_unhashable_values(series):
        return None

    try:
        return int(non_null.nunique(dropna=True))
    except TypeError:
        return None


def _safe_unique_count(series: pd.Series) -> int | None:
    return safe_unique_count(series)


def _contains_unhashable_values(series: pd.Series) -> bool:
    return contains_nested_or_unhashable_values(series)


def _safe_is_unique_non_null(series: pd.Series) -> bool:
    """Evaluate uniqueness only for scalar, hashable columns."""
    if not series.notna().all():
        return False

    if _contains_unhashable_values(series):
        return False

    try:
        return bool(series.nunique(dropna=True) == len(series))
    except TypeError:
        return False


def _safe_is_constant(series: pd.Series) -> bool:
    """Evaluate constancy only for scalar, hashable columns."""
    if _contains_unhashable_values(series):
        return False

    try:
        return bool(series.nunique(dropna=False) <= 1)
    except TypeError:
        return False



def _safe_duplicate_row_count(frame: pd.DataFrame) -> int:
    """Count duplicate rows using only scalar, hashable columns."""
    scalar_columns: list[str] = []

    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()

        if non_null.empty:
            scalar_columns.append(column)
            continue

        sample = non_null.iloc[0]

        if isinstance(sample, (np.ndarray, list, dict, set)):
            continue

        try:
            hash(sample)
        except TypeError:
            continue

        scalar_columns.append(column)

    if not scalar_columns:
        return 0

    try:
        return int(frame[scalar_columns].duplicated().sum())
    except TypeError:
        # A later row may contain an unhashable value even if the first sample did not.
        safe_columns: list[str] = []

        for column in scalar_columns:
            if not _contains_unhashable_values(frame[column]):
                safe_columns.append(column)

        if not safe_columns:
            return 0

        return int(frame[safe_columns].duplicated().sum())


def profile_tables(tables: Mapping[str, RawTable]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, table in sorted(tables.items()):
        frame = table.frame
        columns = []
        for column in frame.columns:
            series = frame[column]
            parse_rate = _timestamp_parse_rate(str(column), series)
            is_nested = contains_nested_or_unhashable_values(series)
            columns.append({
                "table": name,
                "column": str(column),
                "dtype": str(series.dtype),
                "null_fraction": float(series.isna().mean()) if len(series) else 0.0,
                "unique_count": _safe_unique_count(series),
                "is_unique_non_null": _safe_is_unique_non_null(series),
                "timestamp_parse_success_rate": parse_rate,
                "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "is_constant": _safe_is_constant(series),
                "is_nested_object": bool(is_nested),
                "hashability": (
                    "nested_unhashable"
                    if is_nested
                    else "scalar_hashable"
                ),
                "profiling_exclusion_reason": (
                    "nested_or_unhashable_object" if is_nested else ""
                ),
            })
        out[name] = {
            "path": str(table.path),
            "sha256": table.file_sha256,
            "row_count": int(len(frame)),
            "duplicate_rows": _safe_duplicate_row_count(frame),
            "columns": columns,
            "candidate_primary_keys": [
                row["column"]
                for row in columns
                if row["is_unique_non_null"]
            ],
            "candidate_event_time_columns": [
                row["column"]
                for row in columns
                if row["timestamp_parse_success_rate"] >= 0.98
                and not row["is_constant"]
            ],
        }
    return out


def discover_relations(
    *,
    tables: Mapping[str, RawTable],
    profiles: Mapping[str, Mapping[str, Any]],
    configured,
    threshold: float,
) -> dict[str, Any]:
    candidates = []
    accepted = []
    for child_name, raw in sorted((configured or {}).items()):
        for fk in raw.get("foreign_keys", ()) or ():
            parent_name = str(fk["references"]["table"])
            child_col = str(fk["column"])
            parent_col = str(fk["references"]["column"])
            child = tables[child_name].frame[child_col]
            parent = tables[parent_name].frame[parent_col]
            key_hashable = not (
                contains_nested_or_unhashable_values(child)
                or contains_nested_or_unhashable_values(parent)
            )
            parent_is_pk = (
                tables[parent_name].primary_key == parent_col
                and parent.notna().all()
                and key_hashable
                and safe_unique_count(parent) == len(parent)
            )
            dtype_compatible = str(child.dtype) == str(parent.dtype)
            non_null = child.dropna()
            if key_hashable and len(non_null):
                parent_values = set(parent.dropna())
                coverage = float(non_null.isin(parent_values).mean())
                orphan_count = int((~non_null.isin(parent_values)).sum())
            else:
                coverage = 0.0
                orphan_count = int(len(non_null))
            row = {
                "child_table": child_name,
                "child_column": child_col,
                "parent_table": parent_name,
                "parent_column": parent_col,
                "non_null_coverage": float(child.notna().mean()) if len(child) else 0.0,
                "referential_coverage": coverage,
                "orphan_count": orphan_count,
                "dtype_compatible": dtype_compatible,
                "hashability": (
                    "scalar_hashable"
                    if key_hashable
                    else "nested_unhashable"
                ),
                "parent_primary_key_proven": parent_is_pk,
                "accepted": (
                    parent_is_pk
                    and dtype_compatible
                    and coverage >= threshold
                    and orphan_count == 0
                ),
            }
            candidates.append(row)
            if row["accepted"]:
                accepted.append(row)
    if not accepted:
        if any(row["orphan_count"] for row in candidates):
            raise ValueError("referential_integrity_failure")
        raise ValueError("ambiguous_foreign_key")
    if len(accepted) > 1:
        raise ValueError("ambiguous_foreign_key")
    return {"candidates": candidates, "accepted": accepted[0]}


def resolve_task(
    config: Mapping[str, Any],
    *,
    tables: Mapping[str, RawTable],
    relations: Mapping[str, Any],
    official_targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = dict(config.get("task") or {})
    required = (
        "task_id",
        "target_table",
        "entity_key",
        "target_time_col",
        "label_col",
        "problem_type",
        "primary_metric",
        "metric_direction",
    )
    missing = [name for name in required if not task.get(name)]
    if missing:
        if "label_col" in missing:
            raise ValueError("missing_label")
        raise ValueError("missing_task_fields:" + ",".join(missing))
    split = dict(config.get("split") or {})
    if split.get("strategy") not in {"temporal", "official_pre_split"}:
        raise ValueError("unsupported_split_strategy")
    if split.get("strategy") == "temporal" and (
        not split.get("train_end") or not split.get("validation_end")
    ):
        raise ValueError("invalid_temporal_split")
    if (
        split.get("strategy") == "temporal"
        and pd.Timestamp(split["train_end"]) >= pd.Timestamp(split["validation_end"])
    ):
        raise ValueError("invalid_temporal_split")
    target_table = str(task["target_table"])
    target = tables[target_table].frame
    target_frames = (
        (official_targets["train"], official_targets["validation"])
        if official_targets is not None
        else (target,)
    )
    for column in (task["entity_key"], task["target_time_col"], task["label_col"]):
        for frame in target_frames:
            if column not in frame.columns:
                if column == task["label_col"]:
                    raise ValueError("missing_label")
                if column == task["target_time_col"]:
                    raise ValueError("missing_target_time")
                raise ValueError(f"missing_target_column:{column}")
    relation_entity_key = str(
        task.get(
            "relation_entity_key",
            task["entity_key"],
        )
    )

    for frame in target_frames:
        if relation_entity_key not in frame.columns:
            raise ValueError(
                "missing_target_relation_lookup_column:"
                + relation_entity_key
            )

    if official_targets is not None:
        _validate_official_target_identity(
            official_targets["train"],
            official_targets["validation"],
            entity_key=str(task["entity_key"]),
            target_time_col=str(task["target_time_col"]),
        )

    accepted = relations["accepted"]

    explicit_relation = all(
        task.get(name) is not None
        for name in (
            "child_table",
            "child_fk",
            "child_event_time_col",
        )
    )

    if explicit_relation:
        child_table = str(
            task["child_table"]
        )
        child_fk = str(
            task["child_fk"]
        )
        child_time = str(
            task["child_event_time_col"]
        )

        if child_table not in tables:
            raise ValueError(
                "missing_explicit_child_table:"
                + child_table
            )

        child_frame = tables[
            child_table
        ].frame

        if child_fk not in child_frame.columns:
            raise ValueError(
                "missing_explicit_child_fk:"
                + child_fk
            )

        if child_time not in child_frame.columns:
            raise ValueError(
                "missing_explicit_child_event_time:"
                + child_time
            )

    else:
        if accepted["parent_table"] != target_table:
            raise ValueError(
                "unsupported_relation_shape"
            )

        child_table = str(
            accepted["child_table"]
        )
        child_fk = str(
            accepted["child_column"]
        )
        child_time = tables[
            child_table
        ].event_time_col

        if not child_time:
            raise ValueError(
                "ambiguous_event_time"
            )

        child_time = str(child_time)

    numeric_col = _select_numeric_source(
        child=tables[child_table],
        child_fk=child_fk,
        child_time_col=child_time,
    )

    return {
        "dataset": config["dataset"],
        "task_id": task["task_id"],
        "target_table": target_table,

        # Prediction-row identity.
        "entity_key": task["entity_key"],

        # Separate target-side relational lookup key.
        "relation_entity_key":
            relation_entity_key,

        "target_time_col": task["target_time_col"],
        "label_col": task["label_col"],
        "problem_type": task["problem_type"],
        "primary_metric": task["primary_metric"],
        "metric_direction": task["metric_direction"],
        "train_end": split.get("train_end"),
        "validation_end": split.get("validation_end"),
        "child_table": child_table,
        "child_fk": child_fk,
        "child_time_col": child_time,
        "numeric_col": numeric_col,

        # Legacy relations retain <=.  Event-row self-history can
        # explicitly request strict-before semantics.
        "strict_before": bool(
            task.get(
                "strict_before",
                False,
            )
        ),

        "relation_orientation": str(
            task.get(
                "relation_orientation",
                "incoming_fk",
            )
        ),
    }


def load_official_pre_split_targets(
    config_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    split = dict(config.get("split") or {})
    if split.get("source") == "test" or "test_target_path" in split:
        raise ValueError("attempted_test_access")
    train_path = _resolve_config_path(config_path, split.get("train_target_path"))
    validation_path = _resolve_config_path(
        config_path,
        split.get("validation_target_path"),
    )
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    if train.empty or validation.empty:
        raise ValueError("missing_task_split")
    if list(train.columns) != list(validation.columns):
        raise ValueError("incompatible_target_schemas")
    if _row_fingerprints(train).intersection(_row_fingerprints(validation)):
        raise ValueError("train_validation_overlap")
    return {
        "strategy": "official_pre_split",
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "train_target_path": str(train_path),
        "validation_target_path": str(validation_path),
        "train_target_hash": _file_sha256(train_path),
        "validation_target_hash": _file_sha256(validation_path),
        "source": str(split.get("source", "")),
        "train_split_name": str(split.get("train_split_name", "train")),
        "validation_split_name": str(
            split.get("validation_split_name", "val")
        ),
        "test_split_accessed": False,
    }


def _validate_official_target_identity(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    entity_key: str,
    target_time_col: str,
) -> None:
    key_cols = [entity_key, target_time_col]
    for frame in (train, validation):
        if frame.duplicated(subset=key_cols).any():
            raise ValueError("duplicate_target_identity")


def _row_fingerprints(frame: pd.DataFrame) -> set[str]:
    rows = frame.astype("string").fillna("<NA>").to_dict("records")
    return {
        hashlib.sha256(
            json.dumps(row, sort_keys=True, default=str).encode()
        ).hexdigest()
        for row in rows
    }


def _looks_like_identifier(column: str) -> bool:
    normalized = str(column).replace("-", "_").lower()

    return (
        normalized in {
            "id",
            "primary_key",
            "number",
        }
        or normalized.endswith("_id")
        or normalized.endswith("id")
        or normalized.endswith("_key")
        or normalized.endswith("key")
    )


def _select_numeric_source(
    *,
    child: RawTable,
    child_fk: str,
    child_time_col: str,
) -> str | None:
    excluded = {child_fk, child_time_col}

    if child.primary_key is not None:
        excluded.add(child.primary_key)

    for foreign_key in child.foreign_keys:
        column = foreign_key.get("column")
        if column:
            excluded.add(str(column))

    numeric = [
        col
        for col in child.frame.columns
        if col not in excluded
        and not _looks_like_identifier(col)
        and pd.api.types.is_numeric_dtype(child.frame[col])
        and not pd.api.types.is_bool_dtype(child.frame[col])
        and child.frame[col].nunique(dropna=True) > 1
    ]

    preferred = (
        "positionOrder",
        "grid",
        "points",
        "laps",
        "milliseconds",
        "fastestLap",
        "rank",
        "position",
    )

    for column in preferred:
        if column in numeric:
            return column

    return sorted(numeric)[0] if numeric else None


def split_targets(
    target: pd.DataFrame,
    *,
    target_time_col: str,
    train_end: str,
    validation_end: str,
) -> dict[str, pd.DataFrame]:
    times = pd.to_datetime(target[target_time_col], errors="coerce")
    if times.isna().any():
        raise ValueError("missing_target_time")
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    if train_end_ts >= validation_end_ts:
        raise ValueError("invalid_temporal_split")
    train = target.loc[times <= train_end_ts].copy()
    validation = target.loc[(times > train_end_ts) & (times <= validation_end_ts)].copy()
    if train.empty or validation.empty:
        raise ValueError("invalid_temporal_split")
    overlap = set(train.index).intersection(set(validation.index))
    if overlap:
        raise ValueError("train_validation_overlap")
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True)}


def plan_baseline_features(
    *,
    dataset: str,
    task: str,
    target_table: str,
    child_table: str,
    child: pd.DataFrame,
    entity_key: str,
    child_fk: str,
    child_time_col: str,
    target_time_col: str,
    label_col: str,
    relation_entity_key: str | None = None,
    strict_before: bool = False,
    numeric_col: str | None = None,
    child_primary_key: str | None = None,
    train_row_count: int = 0,
    validation_row_count: int = 0,
) -> dict[str, Any]:
    relation_entity_key = str(
        relation_entity_key or entity_key
    )

    features = _declared_baseline_features(
        child_table=child_table,
        child=child,
        child_fk=child_fk,
        child_time_col=child_time_col,
        numeric_col=numeric_col,
        child_primary_key=child_primary_key,
        excluded_source_columns={
            str(label_col),
        },
    )
    feature_columns = []
    manifests = []
    for primitive, source_col, agg in features:
        out = _feature_name(child_table, source_col, agg)
        feature_columns.append(out)
        aux = f"{out}__is_missing"
        manifests.append({
            "dataset": dataset,
            "task": task,
            "program_id": BASELINE_AUTO,
            "primitive_id": primitive,
            "source_table": child_table,
            "source_column": source_col or "",
            "output_column": out,
            "auxiliary_output_columns": aux,
            "join_key": child_fk,
            "target_entity_key": entity_key,
            "target_relation_key":
                relation_entity_key,
            "child_event_time_col": child_time_col,
            "target_time_col": target_time_col,
            "temporal_predicate": (
                f"{child_table}.{child_time_col} "
                f"{'<' if strict_before else '<='} "
                f"{target_table}.{target_time_col}"
            ),
            "aggregation": agg,
            "train_validation_applicability": "train|validation",
            "leakage_safe": "true",
            "temporal_safe": "true",
            "provenance_complete": "true",
            "implementation_version": ONBOARDING_VERSION,
            "materialization_strategy": "grouped_temporal_sweep",
        })
    evidence = [
        {
            "dataset": dataset,
            "task": task,
            "program_id": BASELINE_AUTO,
            "primitive_id": row["primitive_id"],
            "source_table": row["source_table"],
            "source_column": row["output_column"],
            "output_column": row["output_column"],
            "status": "proven",
            "notes": ["onboarding_manifest"],
        }
        for row in manifests
    ]
    return {
        "features": manifests,
        "feature_columns": tuple(feature_columns),
        "lowering_evidence": evidence,
        "candidate_programs": [{
            "program_id": BASELINE_AUTO,
            "primitive_ids": [row["primitive_id"] for row in manifests],
            "description": "Automatically onboarded leakage-safe relational baseline.",
        }],
        "workload": {
            "materialization_strategy": "grouped_temporal_sweep",
            "target_train_rows": int(train_row_count),
            "target_validation_rows": int(validation_row_count),
            "child_rows": int(len(child)),
            "materialization_executed": False,
            "implementation_version": ONBOARDING_VERSION,
            "implementation_strategy": "grouped_temporal_sweep",
            "child_row_count": int(len(child)),
            "train_target_row_count": int(train_row_count),
            "validation_target_row_count": int(validation_row_count),
            "planned_feature_count": len(manifests),
            "planned_feature_columns": tuple(feature_columns),
        },
    }


def build_baseline_features(
    *,
    dataset: str,
    task: str,
    target_table: str,
    target_train: pd.DataFrame,
    target_val: pd.DataFrame,
    child_table: str,
    child: pd.DataFrame,
    entity_key: str,
    child_fk: str,
    child_time_col: str,
    target_time_col: str,
    label_col: str,
    relation_entity_key: str | None = None,
    strict_before: bool = False,
    numeric_col: str | None = None,
    child_primary_key: str | None = None,
) -> dict[str, Any]:
    plan = plan_baseline_features(
        dataset=dataset,
        task=task,
        target_table=target_table,
        child_table=child_table,
        child=child,
        entity_key=entity_key,
        relation_entity_key=relation_entity_key,
        child_fk=child_fk,
        child_time_col=child_time_col,
        target_time_col=target_time_col,
        label_col=label_col,
        strict_before=strict_before,
        numeric_col=numeric_col,
        child_primary_key=child_primary_key,
        train_row_count=len(target_train),
        validation_row_count=len(target_val),
    )
    features = [
        (
            row["primitive_id"],
            (row["source_column"] or None),
            row["aggregation"],
        )
        for row in plan["features"]
    ]
    train = _apply_features(
        target_train,
        child,
        features,
        child_table,
        child_fk,
        child_time_col,
        entity_key,
        target_time_col,
        relation_entity_key=relation_entity_key,
        strict_before=strict_before,
        child_primary_key=child_primary_key,
    )
    val = _apply_features(
        target_val,
        child,
        features,
        child_table,
        child_fk,
        child_time_col,
        entity_key,
        target_time_col,
        relation_entity_key=relation_entity_key,
        strict_before=strict_before,
        child_primary_key=child_primary_key,
    )
    if list(train.columns) != list(val.columns):
        raise ValueError("train/validation feature schema mismatch")
    plan["workload"] = {
        **plan["workload"],
        "materialization_executed": True,
    }
    return {
        **plan,
        "train": train,
        "validation": val,
    }


def _declared_baseline_features(
    *,
    child_table: str,
    child: pd.DataFrame,
    child_fk: str,
    child_time_col: str,
    numeric_col: str | None,
    child_primary_key: str | None,
    excluded_source_columns: Sequence[str] = (),
) -> list[tuple[str, str | None, str]]:
    excluded = {
        child_fk,
        child_time_col,
        child_primary_key,
        *map(str, excluded_source_columns),
    }

    numeric_cols = [
        col
        for col in child.columns
        if col not in excluded
        and not _looks_like_identifier(col)
        and pd.api.types.is_numeric_dtype(child[col])
        and not pd.api.types.is_bool_dtype(child[col])
        and child[col].nunique(dropna=True) > 1
    ]
    categorical_cols = [
        col
        for col in child.columns
        if col not in excluded
        and not _looks_like_identifier(col)
        and not pd.api.types.is_numeric_dtype(child[col])
        and not contains_nested_or_unhashable_values(child[col])
    ]
    features = [("baseline::count", None, "count")]
    if numeric_col is not None and numeric_col in numeric_cols:
        col = numeric_col
        for primitive, agg in (
            ("baseline::numeric_mean", "mean"),
            ("baseline::numeric_std", "std"),
            ("baseline::numeric_min", "min"),
            ("baseline::numeric_max", "max"),
        ):
            features.append((primitive, col, agg))
    features.append(("baseline::days_since_last", None, "days_since_last"))
    if categorical_cols:
        features.append((
            "baseline::history::past_unique_values",
            sorted(categorical_cols)[0],
            "nunique",
        ))
    return features


def _apply_features(
    target,
    child,
    features,
    child_table,
    child_fk,
    child_time_col,
    entity_key,
    target_time_col,
    *,
    relation_entity_key: str | None = None,
    strict_before: bool = False,
    child_primary_key: str | None = None,
):
    return _apply_features_optimized(
        target,
        child,
        features,
        child_table,
        child_fk,
        child_time_col,
        entity_key,
        target_time_col,
        relation_entity_key=relation_entity_key,
        strict_before=strict_before,
        child_primary_key=child_primary_key,
    )


def _apply_features_optimized(
    target,
    child,
    features,
    child_table,
    child_fk,
    child_time_col,
    entity_key,
    target_time_col,
    *,
    relation_entity_key: str | None = None,
    strict_before: bool = False,
    child_primary_key: str | None = None,
):
    out = target.copy().reset_index(drop=True)

    relation_entity_key = str(
        relation_entity_key or entity_key
    )

    if relation_entity_key not in out.columns:
        raise ValueError(
            "missing_target_relation_lookup_column:"
            + relation_entity_key
        )
    if out.empty:
        for _, source_col, agg in features:
            col = _feature_name(child_table, source_col, agg)
            out[col] = []
            out[f"{col}__is_missing"] = []
        return out
    state = _child_temporal_state(
        child=child,
        child_table=child_table,
        features=features,
        child_fk=child_fk,
        child_time_col=child_time_col,
        child_primary_key=child_primary_key,
    )
    target_columns = list(
        dict.fromkeys(
            [
                entity_key,
                relation_entity_key,
                target_time_col,
            ]
        )
    )

    target_work = out[
        target_columns
    ].copy()
    target_work["_target_pos"] = range(len(target_work))
    target_work[target_time_col] = pd.to_datetime(
        target_work[target_time_col],
        errors="coerce",
    )
    result = pd.DataFrame({"_target_pos": range(len(out))})
    for _, source_col, agg in features:
        col = _feature_name(child_table, source_col, agg)
        result[col] = np.nan
    state_by_entity = {
        entity: group
        for entity, group in state.groupby(child_fk, sort=False, dropna=False)
    }
    valid_target_work = target_work[target_work[target_time_col].notna()]
    for entity, target_group in valid_target_work.groupby(
        relation_entity_key,
        sort=False,
        dropna=False,
    ):
        target_sorted = target_group.sort_values(
            [target_time_col, "_target_pos"],
            kind="mergesort",
        )
        child_group = state_by_entity.get(entity)
        if child_group is None or child_group.empty:
            continue
        merged = pd.merge_asof(
            target_sorted,
            child_group,
            left_on=target_time_col,
            right_on=child_time_col,
            direction="backward",
            allow_exact_matches=not strict_before,
        )
        positions = merged["_target_pos"].to_numpy()
        for _, source_col, agg in features:
            col = _feature_name(child_table, source_col, agg)
            state_col = _state_column(child_table, source_col, agg)
            if state_col in merged:
                if agg == "days_since_last":
                    delta = (
                        merged[target_time_col] - merged[state_col]
                    ).dt.total_seconds() / 86400.0
                    result.loc[positions, col] = delta.to_numpy()
                else:
                    result.loc[positions, col] = merged[state_col].to_numpy()
    for _, source_col, agg in features:
        col = _feature_name(child_table, source_col, agg)
        series = result[col]
        if agg in {"count", "nunique"}:
            series = series.fillna(0.0)
        vals = series.to_numpy()
        out[col] = vals
        out[f"{col}__is_missing"] = pd.Series(vals).isna().to_numpy()
    return out


def _child_temporal_state(
    *,
    child: pd.DataFrame,
    child_table: str,
    features,
    child_fk: str,
    child_time_col: str,
    child_primary_key: str | None,
) -> pd.DataFrame:
    work = child.copy()
    work["_child_pos"] = range(len(work))
    work[child_time_col] = pd.to_datetime(work[child_time_col], errors="coerce")
    work = work[work[child_time_col].notna()].copy()
    sort_cols = [child_fk, child_time_col]
    if child_primary_key is not None and child_primary_key in work.columns:
        sort_cols.append(child_primary_key)
    sort_cols.append("_child_pos")
    work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    grouped = work.groupby(child_fk, sort=False, dropna=False)
    for _, source_col, agg in features:
        state_col = _state_column(child_table, source_col, agg)
        if agg == "count":
            work[state_col] = grouped.cumcount().astype(float) + 1.0
        elif agg == "days_since_last":
            work[state_col] = work[child_time_col]
        elif agg == "nunique":
            work[state_col] = _cumulative_nunique(
                work,
                group_col=child_fk,
                value_col=source_col,
            )
        else:
            numeric = pd.to_numeric(work[source_col], errors="coerce")
            non_null = numeric.notna().astype(float)
            work["_value"] = numeric.fillna(0.0)
            work["_nonnull"] = non_null
            nonnull_count = work.groupby(child_fk, sort=False)["_nonnull"].cumsum()
            value_sum = work.groupby(child_fk, sort=False)["_value"].cumsum()
            if agg == "mean":
                work[state_col] = value_sum / nonnull_count.replace(0.0, np.nan)
            elif agg == "std":
                square_sum = work.assign(
                    _square=work["_value"] * work["_value"]
                ).groupby(child_fk, sort=False)["_square"].cumsum()
                mean = value_sum / nonnull_count.replace(0.0, np.nan)
                variance = (square_sum / nonnull_count.replace(0.0, np.nan)) - (mean * mean)
                work[state_col] = np.sqrt(np.maximum(variance, 0.0))
            elif agg == "min":
                masked = numeric.where(numeric.notna(), np.nan)
                work[state_col] = (
                    masked.groupby(work[child_fk], sort=False)
                    .cummin()
                    .groupby(work[child_fk], sort=False)
                    .ffill()
                )
            elif agg == "max":
                masked = numeric.where(numeric.notna(), np.nan)
                work[state_col] = (
                    masked.groupby(work[child_fk], sort=False)
                    .cummax()
                    .groupby(work[child_fk], sort=False)
                    .ffill()
                )
            else:
                raise ValueError(f"unsupported aggregation:{agg}")
    keep = [child_fk, child_time_col]
    for _, source_col, agg in features:
        keep.append(_state_column(child_table, source_col, agg))
    return work[keep]


def _cumulative_nunique(
    frame: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
) -> pd.Series:
    counts = []
    seen_by_group: dict[object, set[object]] = {}
    for group, value in zip(frame[group_col], frame[value_col]):
        seen = seen_by_group.setdefault(group, set())
        if not pd.isna(value):
            seen.add(value)
        counts.append(float(len(seen)))
    return pd.Series(counts, index=frame.index, dtype="float64")


def _state_column(child_table: str, source_col: str | None, agg: str) -> str:
    return f"__state__{_feature_name(child_table or 'child', source_col, agg)}"


def _feature_name(child_table: str, source_col: str | None, agg: str) -> str:
    if agg == "count":
        return f"f_{child_table}_count"
    if agg == "days_since_last":
        return f"f_{child_table}_days_since_last"
    if agg == "nunique":
        return f"f_{child_table}_{source_col}_nunique"
    return f"f_{child_table}_{source_col}_{agg}"


def _load_tables(config_path: Path, config: Mapping[str, Any]) -> dict[str, RawTable]:
    base = config_path.parent
    out = {}
    for name, raw in sorted((config.get("tables") or {}).items()):
        path = Path(str(raw["path"]))
        if not path.is_absolute():
            path = (base / path).resolve()
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"unsupported_table_format:{path}")
        out[name] = RawTable(
            name=name,
            path=path,
            primary_key=raw.get("primary_key"),
            frame=frame,
            file_sha256=_file_sha256(path),
            foreign_keys=tuple(raw.get("foreign_keys") or ()),
            event_time_col=raw.get("event_time_col"),
        )
    return out


def _write_outputs(**kwargs) -> None:
    staging = kwargs["staging"]
    config = kwargs["config"]
    resolved = kwargs["resolved"]
    profiles = kwargs["profiles"]
    relations = kwargs["relations"]
    feature = kwargs["feature_result"]
    raw_profile = staging / "raw_profile"
    raw_profile.mkdir(parents=True)
    _write_json(raw_profile / "schema_profile.json", profiles)
    _write_schema_csv(raw_profile / "schema_profile.csv", profiles)
    _write_csv(staging / "relation_candidates.csv", relations["candidates"])
    _write_json(staging / "relation_graph.json", relations["accepted"])
    _write_csv(staging / "relation_audit.csv", relations["candidates"])
    _write_csv(staging / "time_column_audit.csv", _time_rows(kwargs["tables"]))
    _write_yaml(staging / "resolved_task_spec.yaml", _resolved_task_yaml(config, resolved, feature))
    _write_json(staging / "split_manifest.json", _split_manifest(kwargs["split"]))
    _write_csv(staging / "split_audit.csv", [_split_manifest(kwargs["split"])])
    feature["train"].to_parquet(staging / "target_with_dfs_agg_train.parquet", index=False)
    feature["validation"].to_parquet(staging / "target_with_dfs_agg_val.parquet", index=False)
    _write_json(staging / "baseline_feature_config.json", {
        "features": feature["features"],
        "implementation_version": ONBOARDING_VERSION,
        "workload": feature.get("workload", {}),
    })
    _write_csv(staging / "baseline_feature_manifest.csv", feature["features"])
    _write_csv(staging / "lowering_provenance.csv", feature["lowering_evidence"])
    _write_audit(staging / "temporal_safety_audit.csv", config["dataset"], resolved["task_id"], feature["features"], "temporal_safety")
    _write_audit(staging / "leakage_safety_audit.csv", config["dataset"], resolved["task_id"], feature["features"], "leakage_safety")
    manifest = _onboarding_manifest(staging, kwargs)
    _write_json(staging / "onboarding_manifest.json", manifest)


def _resolved_task_yaml(config, resolved, feature):
    key = f"{config['dataset']}/{resolved['task_id']}"
    drop_cols, drop_reasons = _evaluation_drop_columns(resolved, feature)
    return {"tasks": {key: {
        "problem_type": resolved["problem_type"],
        "label_col": resolved["label_col"],
        "primary_metric": resolved["primary_metric"],
        "metric_direction": resolved["metric_direction"],
        "target": {"entity_key": resolved["entity_key"], "time_col": resolved["target_time_col"]},
        "dfs": {
            "child_table": resolved["child_table"],
            "child_time_col": resolved["child_time_col"],
            "numeric_col": resolved.get("numeric_col"),
            "baseline_operations": [
                "count",
                "numeric_mean",
                "numeric_std",
                "numeric_min",
                "numeric_max",
                "days_since_last",
            ],
        },
        "candidate_programs": feature["candidate_programs"],
        "evaluation": {
            "drop_cols": drop_cols,
            "drop_reasons": drop_reasons,
        },
        "prepared_artifacts": {
            "provider": "onboarding",
            "onboarding_manifest": {"path": "onboarding_manifest.json"},
        },
    }}}


def _evaluation_drop_columns(resolved, feature) -> tuple[list[str], dict[str, str]]:
    label_col = resolved["label_col"]
    entity_key = resolved["entity_key"]
    target_time_col = resolved["target_time_col"]
    train = feature["train"]
    reasons: dict[str, str] = {}
    for column in train.columns:
        if column == label_col or str(column).startswith("f_"):
            continue
        reason = None
        if column == entity_key:
            reason = "target_entity_key"
        elif column == target_time_col:
            reason = "target_prediction_time"
        elif str(column).lower() in {"id", "primary_key"}:
            reason = "identifier_metadata"
        elif str(column).lower().endswith("_id"):
            reason = "identifier_metadata"
        elif pd.api.types.is_datetime64_any_dtype(train[column]):
            reason = "datetime_metadata"
        if reason is not None:
            reasons[str(column)] = reason
    return sorted(reasons), {key: reasons[key] for key in sorted(reasons)}


def _onboarding_manifest(staging: Path, kwargs) -> dict[str, Any]:
    config = kwargs["config"]
    resolved = kwargs["resolved"]
    feature = kwargs["feature_result"]
    files = {
        name: _file_sha256(staging / name)
        for name in (
            "target_with_dfs_agg_train.parquet",
            "target_with_dfs_agg_val.parquet",
            "baseline_feature_manifest.csv",
            "lowering_provenance.csv",
            "temporal_safety_audit.csv",
            "leakage_safety_audit.csv",
        )
    }
    return {
        "dataset": config["dataset"],
        "task": resolved["task_id"],
        "status": "completed",
        "onboarding_version": ONBOARDING_VERSION,
        "reuse_identity": kwargs["reuse_identity"],
        "train_target": {
            "dataset": config["dataset"], "task": resolved["task_id"], "split": "train", "role": "target",
            "table": "target_with_dfs_agg", "path": "target_with_dfs_agg_train.parquet",
        },
        "validation_target": {
            "dataset": config["dataset"], "task": resolved["task_id"], "split": "validation", "role": "target",
            "table": "target_with_dfs_agg", "path": "target_with_dfs_agg_val.parquet",
        },
        "candidate_programs": feature["candidate_programs"],
        "lowering_evidence": feature["lowering_evidence"],
        "baseline_feature_workload": feature.get("workload", {}),
        "file_hashes": files,
        "task_config": _resolved_task_yaml(config, resolved, feature),
    }


def _manifest_identity(**kwargs) -> str:
    payload = {
        "version": ONBOARDING_VERSION,
        "config_sha256": _file_sha256(kwargs["config_path"]),
        "tables": {name: table.file_sha256 for name, table in sorted(kwargs["tables"].items())},
        "features": kwargs["feature_result"]["features"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _validate_publication(staging: Path) -> None:
    required = (
        "raw_profile/schema_profile.json", "raw_profile/schema_profile.csv",
        "relation_candidates.csv", "relation_graph.json", "relation_audit.csv", "time_column_audit.csv",
        "resolved_task_spec.yaml", "split_manifest.json", "split_audit.csv",
        "target_with_dfs_agg_train.parquet", "target_with_dfs_agg_val.parquet",
        "baseline_feature_config.json", "baseline_feature_manifest.csv", "lowering_provenance.csv",
        "temporal_safety_audit.csv", "leakage_safety_audit.csv", "onboarding_manifest.json",
    )
    missing = [name for name in required if not (staging / name).exists()]
    if missing:
        raise ValueError("provenance_incomplete:" + ",".join(missing))


def _write_audit(path: Path, dataset: str, task: str, features, audit_type: str) -> None:
    rows = []
    for row in features:
        rows.append({
            "dataset": dataset, "task": task, "program_id": BASELINE_AUTO,
            "audit_type": audit_type, "primitive_id": row["primitive_id"],
            "status": "passed", "passed": "true", "source_table": row["source_table"],
            "source_column": row["source_column"], "output_column": row["output_column"],
            "rejection_reason": "", "evidence_location": "onboarding_manifest",
            "notes": row["temporal_predicate"],
        })
    _write_csv(path, rows, fieldnames=AUDIT_COLUMNS)


def _split_manifest(split) -> dict[str, Any]:
    out = {
        "strategy": split.get("strategy", "temporal"),
        "train_rows": len(split["train"]),
        "validation_rows": len(split["validation"]),
    }
    for key in (
        "train_target_hash",
        "validation_target_hash",
        "source",
        "train_split_name",
        "validation_split_name",
        "test_split_accessed",
    ):
        if key in split:
            out[key] = split[key]
    return out


def _time_rows(tables):
    rows = []
    for table in tables.values():
        for col in table.frame.columns:
            rate = _timestamp_parse_rate(str(col), table.frame[col])
            rows.append({"table": table.name, "column": col, "parse_success_rate": rate, "accepted": str(col == table.event_time_col).lower()})
    return rows


def _timestamp_parse_rate(column: str, series: pd.Series) -> float:
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0

    name = str(column).lower()
    time_like_name = any(
        token in name
        for token in (
            "timestamp",
            "created_at",
            "updated_at",
            "event_time",
            "date",
            "time",
        )
    )

    if not time_like_name:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce")

    return float(parsed.notna().mean()) if len(series) else 0.0


def _parse_timestamps(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce")


def _write_schema_csv(path: Path, profiles) -> None:
    rows = []
    for profile in profiles.values():
        rows.extend(profile["columns"])
    _write_csv(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames=None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("onboarding config must be a mapping")
    return data


def _resolve_config_path(config_path: Path, raw_path: Any) -> Path:
    if raw_path is None:
        raise ValueError("missing_task_split")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
