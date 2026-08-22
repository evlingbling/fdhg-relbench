from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from fdhg.compiler.ambiguity import normalize_join_key_pair

from .relbench_v1 import _class_name, _load_relbench_objects, _table_df


AUTO_ONBOARDING_VERSION = "auto-relbench-onboarding-v1"
MATERIALIZATION_STRATEGY = "grouped_temporal_sweep"


@dataclass(frozen=True)
class AutoRelBenchOnboardingReport:
    dataset: str
    task: str
    status: str
    output_dir: Path
    blockers: tuple[str, ...]
    dry_run: bool
    task_type: str | None = None
    metric: str | None = None
    metric_direction: str | None = None
    relation_candidates: int = 0
    selected_relations: tuple[str, ...] = ()
    candidate_features: int = 0
    selected_features: int = 0
    inner_selection_score: float | None = None
    official_validation_score: float | None = None
    fallback: bool = False
    test_split_accessed: bool = False
    reused: bool = False
    workload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AutoOnboardingOptions:
    selection_folds: int = 1
    feature_budget: int = 8
    min_delta: float = 0.0
    selection_decoder: str = "hist_gradient_boosting"
    max_relations: int = 3
    max_numeric_columns: int = 4
    max_categorical_columns: int = 4
    relation_threshold: float = 0.98
    max_categorical_cardinality: int = 64
    max_text_cardinality: int = 256
    max_mean_text_length: float = 80.0


def auto_onboard_relbench(
    *,
    dataset_name: str,
    task_name: str,
    output_root: Path,
    write: bool = False,
    overwrite: bool = False,
    download: bool = False,
    task_metadata_config: Path | None = None,
    options: AutoOnboardingOptions | None = None,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
    evaluate_official_validation: bool = True,
) -> AutoRelBenchOnboardingReport:
    options = options or AutoOnboardingOptions()
    output_dir = output_root / f"{dataset_name}_{task_name}"
    try:
        prepared = prepare_auto_onboarding(
            dataset_name=dataset_name,
            task_name=task_name,
            output_root=output_root,
            download=download,
            task_metadata_config=task_metadata_config,
            options=options,
            object_loader=object_loader,
            include_selection=write,
            evaluate_official_validation=evaluate_official_validation,
        )
    except Exception as exc:
        return AutoRelBenchOnboardingReport(
            dataset=dataset_name,
            task=task_name,
            status="blocked",
            output_dir=output_dir,
            blockers=(str(exc),),
            dry_run=not write,
        )
    blockers = tuple(prepared.get("blockers", ()))
    if blockers:
        return _report("blocked", prepared, dry_run=not write, blockers=blockers)
    if not write:
        return _report("dry_run_ready", prepared, dry_run=True)

    manifest_path = output_dir / "auto_onboarding_manifest.json"
    identity = prepared["identity_hash"]
    if output_dir.exists() and not overwrite:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("reuse_identity") == identity:
                return _report("reused", prepared, dry_run=False, reused=True)
        raise FileExistsError(output_dir)

    staging = output_dir.parent / f"_{output_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_outputs(staging, prepared)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    prepared["output_dir"] = output_dir
    return _report("completed", prepared, dry_run=False)


def prepare_auto_onboarding(
    *,
    dataset_name: str,
    task_name: str,
    output_root: Path,
    download: bool,
    task_metadata_config: Path | None,
    options: AutoOnboardingOptions,
    include_selection: bool,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
    evaluate_official_validation: bool = True,
) -> dict[str, Any]:
    if object_loader is not None:
        dataset, task, relbench_version = object_loader(
            dataset_name,
            task_name,
            download,
        )
    elif (
        (dataset_name, task_name)
        in {
            ("dbinfer-retailrocket", "cvr"),
            ("dbinfer-diginetica", "ctr"),
            ("dbinfer-avs", "repeater"),
        }
    ):
        from fdhg.onboarding.dbinfer_v1 import (
            load_dbinfer_relbench_like_objects,
        )

        dataset, task, relbench_version = (
            load_dbinfer_relbench_like_objects(
                dataset_name,
                task_name,
                download,
            )
        )
    else:
        dataset, task, relbench_version = (
            _load_relbench_objects(
                dataset_name,
                task_name,
                download,
            )
        )
    database = dataset.get_db()
    table_dict = getattr(database, "table_dict", None)
    if not isinstance(table_dict, Mapping) or not table_dict:
        raise ValueError("missing_database_tables")
    train_df = _table_df(task.get_table("train")).copy()
    validation_df = _table_df(task.get_table("val")).copy()
    validation_schema = validation_schema_only(validation_df)
    if train_df.empty:
        raise ValueError("missing_task_train_split")
    if validation_df.empty:
        raise ValueError("missing_task_validation_split")
    metadata = resolve_task_metadata(
        dataset_name=dataset_name,
        task_name=task_name,
        task=task,
        train_df=train_df,
        validation_df=validation_df,
        task_metadata_config=task_metadata_config,
    )
    _validate_target_frames(train_df, validation_schema, metadata)
    if (
        (dataset_name, task_name)
        in {
            ("dbinfer-retailrocket", "cvr"),
            ("dbinfer-diginetica", "ctr"),
            ("dbinfer-avs", "repeater"),
        }
    ):
        from fdhg.onboarding.dbinfer_v1 import (
            AVS_REPEATER_RELATION_SPECS,
            DIGINETICA_CTR_RELATION_SPECS,
            RETAILROCKET_CVR_RELATION_SPECS,
            discover_dbinfer_event_relations,
        )

        if (
            dataset_name == "dbinfer-retailrocket"
            and task_name == "cvr"
        ):
            relation_specs = (
                RETAILROCKET_CVR_RELATION_SPECS
            )

            required_lookup_columns = [
                "itemid",
                "visitorid",
            ]

            dbinfer_metadata = {
                "entity_key": "__row_id",
                "relation_entity_key": "itemid",
                "child_table": "ItemProperty",
                "child_fk": "itemid",
                "child_event_time_col": "timestamp",
                "strict_before": False,
                "target_table": "View",
                "event_row_original_entity_key":
                    "__row_id",
            }

        elif (
            dataset_name == "dbinfer-diginetica"
            and task_name == "ctr"
        ):
            relation_specs = (
                DIGINETICA_CTR_RELATION_SPECS
            )

            required_lookup_columns = [
                "queryId",
                "itemId",
            ]

            dbinfer_metadata = {
                "entity_key": "__row_id",
                "relation_entity_key": "queryId",
                "child_table": "Query",
                "child_fk": "queryId",
                "child_event_time_col": "timestamp",

                # Provisional until the Query/target timestamp
                # semantics audit below is completed.
                "strict_before": False,

                "target_table": "QueryResult",
                "event_row_original_entity_key":
                    "__row_id",
            }

        else:
            relation_specs = (
                AVS_REPEATER_RELATION_SPECS
            )

            required_lookup_columns = [
                "id",
            ]

            dbinfer_metadata = {
                "entity_key": "id",
                "relation_entity_key": "id",
                "child_table": "Transaction",
                "child_fk": "id",
                "child_event_time_col": "date",
                "strict_before": False,
                "target_table": "History",
                "event_row_original_entity_key": "id",
            }

        target_lookup_value_mapping = None

        if (
            dataset_name == "dbinfer-avs"
            and task_name == "repeater"
        ):
            task_adapter = getattr(
                task,
                "_task_adapter",
                None,
            )

            entity_mapping = getattr(
                task_adapter,
                "entity_mapping",
                None,
            )

            if not isinstance(
                entity_mapping,
                Mapping,
            ) or not entity_mapping:
                raise ValueError(
                    "blocked_missing_dbinfer_inverse_entity_mapping"
                )

            target_lookup_value_mapping = {
                mapped: raw
                for raw, mapped
                in entity_mapping.items()
            }

            if (
                len(target_lookup_value_mapping)
                != len(entity_mapping)
            ):
                raise ValueError(
                    "non_bijective_dbinfer_entity_mapping"
                )

        missing_lookup_columns = [
            col
            for col in required_lookup_columns
            if col not in train_df.columns
            or col not in validation_df.columns
        ]

        if missing_lookup_columns:
            raise ValueError(
                "missing_dbinfer_target_lookup_columns:"
                + ",".join(
                    sorted(missing_lookup_columns)
                )
            )

        relations = discover_dbinfer_event_relations(
            table_dict={
                name: _table_df(table)
                for name, table in table_dict.items()
            },
            train_targets=train_df,
            target_time_col=str(
                metadata["target_time_col"]
            ),
            relation_specs=relation_specs,
            target_lookup_value_mapping=(
                target_lookup_value_mapping
            ),
        )

        if (
            dataset_name == "dbinfer-avs"
            and task_name == "repeater"
        ):
            raw_lookup_column = "__dbinfer_raw_entity_id"

            for split_name, frame in (
                ("train", train_df),
                ("validation", validation_df),
            ):
                if raw_lookup_column not in frame.columns:
                    raise ValueError(
                        "missing_dbinfer_raw_entity_lookup:"
                        + split_name
                    )

            relations = [
                {
                    **row,
                    "target_lookup_column": raw_lookup_column,
                    "target_lookup_value_transform":
                        "dbinfer_inverse_entity_mapping",
                }
                if row.get("status") == "accepted"
                else row
                for row in relations
            ]

            required_lookup_columns = list(
                dict.fromkeys(
                    [
                        *required_lookup_columns,
                        raw_lookup_column,
                    ]
                )
            )

        metadata = {
            **metadata,
            **dbinfer_metadata,
            "event_row_lookup_columns":
                required_lookup_columns,
            "event_row_relation_fallback": True,
            "relation_source":
                "dbinfer_logical_relation_specs",
        }

    else:
        relations = discover_relation_candidates(
            table_dict=table_dict,
            train_targets=train_df,
            metadata=metadata,
            threshold=options.relation_threshold,
        )

    accepted = [
        row
        for row in relations
        if row["status"] == "accepted"
    ]

    is_explicit_dbinfer_task = (
        (dataset_name, task_name)
        in {
            ("dbinfer-retailrocket", "cvr"),
            ("dbinfer-diginetica", "ctr"),
            ("dbinfer-avs", "repeater"),
        }
    )

    if not accepted and not is_explicit_dbinfer_task:
        (
            event_relations,
            enriched_train_df,
            enriched_validation_df,
            event_metadata,
        ) = _discover_event_row_relation_candidates(
            dataset_name=dataset_name,
            task_name=task_name,
            dataset=dataset,
            task=task,
            database=database,
            table_dict=table_dict,
            train_targets=train_df,
            validation_targets=validation_df,
            selection_folds=options.selection_folds,
        )

        if event_relations:
            from fdhg.onboarding.relbench_v1 import (
                _enrich_target_relation_key,
            )

            relations = event_relations

            row_entity_key = str(
                metadata["entity_key"]
            )
            target_table = str(
                metadata["target_table"]
            )
            target_time_col = str(
                metadata["target_time_col"]
            )

            # Auto evaluates multiple relations jointly.  Therefore
            # the target frame must carry every target-side lookup
            # column required by the accepted candidate relations,
            # not only the globally selected resolver relation key.
            required_lookup_columns = sorted({
                str(
                    row.get(
                        "target_lookup_column",
                        row_entity_key,
                    )
                )
                for row in event_relations
                if row.get("status") == "accepted"
            })

            for lookup_col in required_lookup_columns:
                if lookup_col == row_entity_key:
                    continue

                if lookup_col not in enriched_train_df.columns:
                    enriched_train_df = (
                        _enrich_target_relation_key(
                            enriched_train_df,
                            table_dict=table_dict,
                            target_table=target_table,
                            row_entity_key=row_entity_key,
                            relation_entity_key=lookup_col,
                            target_time_col=target_time_col,
                        )
                    )

                if lookup_col not in enriched_validation_df.columns:
                    enriched_validation_df = (
                        _enrich_target_relation_key(
                            enriched_validation_df,
                            table_dict=table_dict,
                            target_table=target_table,
                            row_entity_key=row_entity_key,
                            relation_entity_key=lookup_col,
                            target_time_col=target_time_col,
                        )
                    )

            train_df = enriched_train_df
            validation_df = (
                enriched_validation_df
            )

            metadata = {
                **metadata,
                **event_metadata,
                "event_row_lookup_columns":
                    required_lookup_columns,
            }

            accepted = [
                row
                for row in relations
                if row["status"]
                == "accepted"
            ]

    accepted = accepted[
        : options.max_relations
    ]
    validation_schema = validation_schema_only(
        validation_df
    )

    column_audit = classify_source_columns(
        table_dict=table_dict,
        relations=accepted,
        metadata=metadata,
        options=options,
    )
    candidates = generate_candidate_features(
        relations=accepted,
        column_audit=column_audit,
        options=options,
    )
    split_plan = make_inner_temporal_splits(
        train_df,
        time_col=metadata["target_time_col"],
        requested_folds=options.selection_folds,
    )
    selection: dict[str, Any]
    final: dict[str, Any]
    if include_selection:
        selected = select_features(
            train_targets=train_df,
            table_dict=table_dict,
            metadata=metadata,
            candidates=candidates,
            split_plan=split_plan,
            options=options,
        )
        if evaluate_official_validation:
            final = final_evaluate(
                train_targets=train_df,
                validation_targets=validation_df,
                table_dict=table_dict,
                metadata=metadata,
                selected_features=selected["selected_features"],
                options=options,
            )
            final["official_validation_evaluated"] = True
        else:
            final = {
                "official_validation_score": None,
                "official_validation_metric": {},
                "official_validation_predictions": pd.DataFrame(),
                "official_validation_evaluated": False,
            }
        selection = selected
    else:
        selection = {
            "selected_features": [],
            "dropped_features": [],
            "selection_trials": [],
            "inner_selection_score": None,
            "fallback": False,
            "fallback_reason": "",
            "fallback_level": "",
            "stopping_reason": "dry_run",
            "workload": _estimate_selection_workload(
                split_plan=split_plan,
                candidates=candidates,
            ),
        }
        final = {
            "official_validation_score": None,
            "official_validation_metric": {},
            "official_validation_predictions": pd.DataFrame(),
            "official_validation_evaluated": False,
        }
    output_dir = output_root / f"{dataset_name}_{task_name}"
    manifest = _manifest(
        dataset_name=dataset_name,
        task_name=task_name,
        output_dir=output_dir,
        relbench_version=relbench_version,
        dataset=dataset,
        task=task,
        database=database,
        metadata=metadata,
        relations=relations,
        column_audit=column_audit,
        candidates=candidates,
        split_plan=split_plan,
        selection=selection,
        final=final,
        options=options,
    )
    identity_hash = _text_sha256(json.dumps({
        "version": AUTO_ONBOARDING_VERSION,
        "dataset": dataset_name,
        "task": task_name,
        "relbench_version": relbench_version,
        "metadata": metadata,
        "relations": relations,
        "columns": column_audit,
        "candidates": candidates,
        "options": options.__dict__,
    }, sort_keys=True, default=str))
    manifest["reuse_identity"] = identity_hash
    return {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "output_dir": output_dir,
        "relbench_version": relbench_version,
        "dataset": dataset,
        "task": task,
        "database": database,
        "table_dict": table_dict,
        "train_df": train_df,
        "validation_df": validation_df,
        "validation_schema": validation_schema,
        "metadata": metadata,
        "relations": relations,
        "accepted_relations": accepted,
        "column_audit": column_audit,
        "candidate_features": candidates,
        "split_plan": split_plan,
        "selection": selection,
        "final": final,
        "options": options,
        "manifest": manifest,
        "identity_hash": identity_hash,
        "blockers": (),
    }


def resolve_task_metadata(
    *,
    dataset_name: str,
    task_name: str,
    task: Any,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    task_metadata_config: Path | None = None,
) -> dict[str, Any]:
    config = _task_metadata_from_config(
        task_metadata_config,
        dataset_name=dataset_name,
        task_name=task_name,
    )
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    fields = {
        "entity_key": ("entity_col", "entity_key", "entity_col_name"),
        "target_time_col": ("time_col", "target_time_col"),
        "label_col": ("target_col", "label_col"),
    }
    for field, attrs in fields.items():
        value, source = _resolve_field(task, attrs, config, field, task_metadata_config)
        if value is not None:
            values[field] = str(value)
            sources[f"{field}_source"] = source
    problem_raw, problem_source = _resolve_field(
        task,
        ("task_type", "problem_type"),
        config,
        "problem_type",
        task_metadata_config,
    )
    problem_type = _normalize_problem_type(problem_raw)
    if not problem_type:
        problem_type = _infer_problem_type(train_df[values.get("label_col", "")]) if values.get("label_col") in train_df else None
        problem_source = "train_label_dtype"
    if problem_type is None:
        raise ValueError("missing_task_metadata:problem_type")
    values["problem_type"] = problem_type
    sources["problem_type_source"] = problem_source
    metric_raw, metric_source = _resolve_field(
        task,
        ("primary_metric", "metric", "metrics", "eval_metrics"),
        config,
        "primary_metric",
        task_metadata_config,
    )
    primary_metric = _choose_metric(problem_type, metric_raw)
    if primary_metric is None:
        raise ValueError("missing_task_metadata:primary_metric")
    values["primary_metric"] = primary_metric
    sources["primary_metric_source"] = metric_source if metric_raw is not None else "task_type_inference"
    direction_raw, direction_source = _resolve_field(
        task,
        ("metric_direction",),
        config,
        "metric_direction",
        task_metadata_config,
    )
    metric_direction = _metric_direction(primary_metric, direction_raw)
    if metric_direction is None:
        raise ValueError("missing_task_metadata:metric_direction")
    values["metric_direction"] = metric_direction
    sources["metric_direction_source"] = direction_source if direction_raw is not None else "metric_inference"
    missing = [
        name for name in ("entity_key", "target_time_col", "label_col")
        if not values.get(name)
    ]
    if missing:
        raise ValueError("missing_task_metadata:" + ",".join(sorted(missing)))
    for col, code in (
        (values["entity_key"], "missing_entity_key"),
        (values["target_time_col"], "missing_target_timestamp"),
        (values["label_col"], "missing_label"),
    ):
        if col not in train_df.columns or col not in validation_df.columns:
            raise ValueError(code)
    target_table = config.get("target_table") or getattr(
        task,
        "entity_table",
        None,
    )
    if target_table:
        values["target_table"] = str(target_table)
        sources["target_table_source"] = (
            f"metadata_config:{task_metadata_config}:"
            f"tasks.{dataset_name}/{task_name}.target_table"
            if config.get("target_table")
            else "task_attr:entity_table"
        )

    for key in (
        "child_table",
        "child_fk",
        "child_event_time_col",
        "relation_threshold",
    ):
        if config.get(key) is not None:
            values[key] = config[key]
            sources[f"{key}_source"] = (
                f"metadata_config:{task_metadata_config}:"
                f"tasks.{dataset_name}/{task_name}.{key}"
            )

    return {**values, **sources}


def discover_relation_candidates(
    *,
    table_dict: Mapping[str, Any],
    train_targets: pd.DataFrame,
    metadata: Mapping[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    entity_key = metadata["entity_key"]
    target_time_col = metadata["target_time_col"]
    target_table = metadata.get("target_table")

    if target_table:
        if target_table not in table_dict:
            return [{
                "status": "rejected",
                "rejection_reasons": "missing_target_table",
            }]
        parent_tables = [target_table]
    else:
        parent_tables = [
            name for name, table in sorted(table_dict.items())
            if getattr(table, "pkey_col", None) == entity_key
            and entity_key in _table_df(table).columns
        ]

    rows: list[dict[str, Any]] = []
    if len(parent_tables) != 1:
        return [{
            "status": "rejected",
            "rejection_reasons": "parent_primary_key_not_unique",
        }]

    parent = parent_tables[0]
    parent_df = _table_df(table_dict[parent])
    parent_key = getattr(table_dict[parent], "pkey_col", None)

    if (
        parent_key is None
        or parent_key not in parent_df.columns
        or not parent_df[parent_key].is_unique
    ):
        return [{
            "status": "rejected",
            "rejection_reasons": "parent_primary_key_not_unique",
        }]

    parent_ids = set(parent_df[parent_key].dropna())
    for child_table, table in sorted(table_dict.items()):
        fkeys = getattr(table, "fkey_col_to_pkey_table", {}) or {}
        for child_fk, parent_table in sorted(fkeys.items()):
            child = _table_df(table)
            time_col = getattr(table, "time_col", None)
            reasons = []
            if parent_table != parent:
                reasons.append("foreign_key_not_task_entity")
            if child_table == parent:
                reasons.append("same_table_relation")
            if child_fk not in child.columns:
                reasons.append("missing_child_fk")
            if time_col is None or str(time_col) not in child.columns:
                reasons.append("missing_child_event_time")
            dtype_compatible = (
                child_fk in child.columns
                and str(child[child_fk].dtype) == str(parent_df[parent_key].dtype)
            )
            if not dtype_compatible:
                reasons.append("fk_dtype_incompatible")
            non_null = child[child_fk].dropna() if child_fk in child.columns else pd.Series(dtype="object")
            referential_coverage = float(non_null.isin(parent_ids).mean()) if len(non_null) else 0.0
            if referential_coverage < threshold:
                reasons.append("referential_coverage_below_threshold")
            history_coverage, cold_start_rate = (0.0, 1.0)
            source_before_train = False
            if time_col is not None and str(time_col) in child.columns and child_fk in child.columns:
                child_times = pd.to_datetime(child[str(time_col)], errors="coerce")
                target_times = pd.to_datetime(train_targets[target_time_col], errors="coerce")
                source_before_train = bool(
                    child_times.notna().any()
                    and target_times.notna().any()
                    and child_times.min() <= target_times.max()
                )
                history_coverage = _history_coverage(
                    train_targets=train_targets,
                    child=child.assign(**{str(time_col): child_times}),
                    entity_key=entity_key,
                    child_fk=child_fk,
                    child_time_col=str(time_col),
                    target_time_col=target_time_col,
                )
                cold_start_rate = 1.0 - history_coverage
            if not source_before_train:
                reasons.append("no_source_data_before_training_targets")
            rows.append({
                "child_table": child_table,
                "child_fk": str(child_fk),
                "parent_table": parent,
                "parent_key": entity_key,
                "child_event_time_col": None if time_col is None else str(time_col),
                "referential_coverage": referential_coverage,
                "child_rows": int(len(child)),
                "training_target_history_coverage": history_coverage,
                "cold_start_rate": cold_start_rate,
                "target_named_column_present": bool(metadata["label_col"] in child.columns),
                "status": "rejected" if reasons else "accepted",
                "rejection_reasons": "|".join(reasons),
            })
    rows.sort(
        key=lambda row: (
            row["status"] != "accepted",
            -float(row.get("training_target_history_coverage", 0.0)),
            -float(row.get("referential_coverage", 0.0)),
            -int(row.get("child_rows", 0)),
            str(row.get("child_table", "")),
        )
    )
    relation_rank = 1
    for row in rows:
        if row["status"] == "accepted":
            row["relation_rank"] = relation_rank
            relation_rank += 1
        else:
            row["relation_rank"] = ""
    return rows



def _discover_event_row_relation_candidates(
    *,
    dataset_name: str,
    task_name: str,
    dataset: Any,
    task: Any,
    database: Any,
    table_dict: Mapping[str, Any],
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    selection_folds: int,
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Fallback relation discovery for prediction-row/event-table tasks.

    The original prediction-row entity key is preserved.  A selected
    outgoing FK from the target event table is attached as a separate
    target-side relational lookup key.
    """

    from fdhg.onboarding.relbench_v1 import (
        _enrich_target_relation_key,
        resolve_relbench_task_metadata,
    )

    try:
        resolved = resolve_relbench_task_metadata(
            dataset_name=dataset_name,
            task_name=task_name,
            dataset=dataset,
            task=task,
            database=database,
            explicit_metadata=None,
            selection_folds=selection_folds,
            train_df=train_targets,
            validation_df=validation_targets,
        )
    except ValueError as exc:
        if str(exc) != "relation_verification_blocker":
            raise
        # Event-row relation discovery is an optional fallback path.
        # If no verified relational path exists, preserve the original
        # targets and allow ordinary Auto/static-entity fallback.
        return [], train_targets, validation_targets, {}

    row_entity_key = str(resolved.entity_key)
    relation_entity_key = str(
        resolved.relation_entity_key
    )
    target_time_col = str(
        resolved.target_time_col
    )

    enriched_train = train_targets
    enriched_validation = validation_targets

    if relation_entity_key != row_entity_key:
        enriched_train = _enrich_target_relation_key(
            train_targets,
            table_dict=table_dict,
            target_table=str(
                resolved.entity_table
            ),
            row_entity_key=row_entity_key,
            relation_entity_key=relation_entity_key,
            target_time_col=target_time_col,
        )

        enriched_validation = (
            _enrich_target_relation_key(
                validation_targets,
                table_dict=table_dict,
                target_table=str(
                    resolved.entity_table
                ),
                row_entity_key=row_entity_key,
                relation_entity_key=relation_entity_key,
                target_time_col=target_time_col,
            )
        )

    relations: list[dict[str, Any]] = []

    verified_rows = [
        dict(row)
        for row in resolved.candidate_relations_considered
        if bool(row.get("verified"))
    ]

    # Preserve the canonical train-only screening order.
    #
    # relation_screening is stored in candidate iteration order, not
    # performance order, so reconstruct the ranking from the recorded
    # train-only metric.  The canonical selected relation is forced to
    # rank first; remaining relations follow metric direction with
    # deterministic schema tie-breaking.
    screening_rows = [
        dict(row)
        for row in resolved.relation_screening
    ]

    metric_direction = str(
        resolved.metric_direction
    )

    def screening_sort_key(
        row: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        raw_score = row.get(
            "mean_inner_fold_score"
        )

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = float("nan")

        if math.isfinite(score):
            score_key = (
                -score
                if metric_direction == "higher"
                else score
            )
        else:
            score_key = float("inf")

        return (
            0 if bool(row.get("selected")) else 1,
            score_key,
            str(row.get("child_table", "")),
            str(row.get("child_column", "")),
            str(
                row.get(
                    "child_event_time_col",
                    "",
                )
            ),
            str(
                row.get(
                    "target_lookup_column",
                    "",
                )
            ),
        )

    screening_rows.sort(
        key=screening_sort_key
    )

    screening_order: dict[
        tuple[str, str, str, str],
        int,
    ] = {}

    for rank, row in enumerate(
        screening_rows,
        start=1,
    ):
        key = (
            str(row.get("child_table", "")),
            str(row.get("child_column", "")),
            str(
                row.get(
                    "child_event_time_col",
                    "",
                )
            ),
            str(
                row.get(
                    "target_lookup_column",
                    "",
                )
            ),
        )

        screening_order[key] = rank

    for row in verified_rows:
        child_table = str(
            row["child_table"]
        )
        child_fk = str(
            row["child_column"]
        )
        child_time_col = str(
            row["child_event_time_col"]
        )
        lookup_col = str(
            row.get(
                "target_lookup_column",
                relation_entity_key,
            )
        )

        child = _table_df(
            table_dict[child_table]
        )

        scoring_targets = enriched_train

        # A candidate may use a different outgoing FK than the
        # globally selected relation. Enrich candidate-specific
        # lookup keys when needed.
        if lookup_col not in scoring_targets.columns:
            from fdhg.onboarding.relbench_v1 import (
                _enrich_target_relation_key,
            )

            scoring_targets = (
                _enrich_target_relation_key(
                    train_targets,
                    table_dict=table_dict,
                    target_table=str(
                        resolved.entity_table
                    ),
                    row_entity_key=row_entity_key,
                    relation_entity_key=lookup_col,
                    target_time_col=target_time_col,
                )
            )

        history_coverage = _history_coverage(
            train_targets=scoring_targets,
            child=child,
            entity_key=lookup_col,
            child_fk=child_fk,
            child_time_col=child_time_col,
            target_time_col=target_time_col,
        )

        key = (
            child_table,
            child_fk,
            child_time_col,
            lookup_col,
        )

        relations.append({
            "child_table": child_table,
            "child_fk": child_fk,
            "parent_table": str(
                row["parent_table"]
            ),
            "parent_key": str(
                row["parent_column"]
            ),
            "child_event_time_col":
                child_time_col,
            "target_lookup_column":
                lookup_col,
            "target_lookup_value_transform": str(
                row.get(
                    "target_lookup_value_transform",
                    "",
                )
            ),
            "strict_before": bool(
                row.get(
                    "strict_before",
                    False,
                )
            ),
            "allow_exact_matches":
                not bool(
                    row.get(
                        "strict_before",
                        False,
                    )
                ),
            "relation_orientation": str(
                row.get(
                    "relation_orientation",
                    "target_outgoing_fk_to_parent_history",
                )
            ),
            "referential_coverage": float(
                row.get(
                    "referential_coverage",
                    0.0,
                )
            ),
            "child_rows": int(
                len(child)
            ),
            "training_target_history_coverage":
                float(history_coverage),
            "cold_start_rate":
                float(1.0 - history_coverage),
            "target_named_column_present":
                bool(
                    resolved.label_col
                    in child.columns
                ),
            "status": "accepted",
            "rejection_reasons": "",
            "relation_rank":
                screening_order.get(
                    key,
                    10**9,
                ),
        })

    relations.sort(
        key=lambda row: (
            int(
                row.get(
                    "relation_rank",
                    10**9,
                )
            ),
            -float(
                row.get(
                    "training_target_history_coverage",
                    0.0,
                )
            ),
            str(
                row.get(
                    "child_table",
                    "",
                )
            ),
            str(
                row.get(
                    "child_fk",
                    "",
                )
            ),
            str(
                row.get(
                    "target_lookup_column",
                    "",
                )
            ),
        )
    )

    for rank, row in enumerate(
        relations,
        start=1,
    ):
        row["relation_rank"] = rank

    metadata_updates = {
        "relation_entity_key":
            relation_entity_key,
        "relation_selection_method":
            resolved.relation_selection_method,
        "relation_selection_reason":
            resolved.relation_selection_reason,
        "relation_source":
            resolved.provenance.get(
                "relation",
                "relbench_v1:event_row_fallback",
            ),
        "event_row_relation_fallback":
            True,
        "event_row_original_entity_key":
            row_entity_key,
    }

    return (
        relations,
        enriched_train,
        enriched_validation,
        metadata_updates,
    )



def classify_source_columns(
    *,
    table_dict: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> list[dict[str, Any]]:
    rows = []
    all_fks_by_table = {
        name: set(
            (
                getattr(
                    table,
                    "fkey_col_to_pkey_table",
                    {},
                )
                or {}
            ).keys()
        )
        for name, table in table_dict.items()
    }

    # Multiple relational paths may originate from the same child table
    # (e.g. View.itemid and View.visitorid).  Column semantics are a
    # property of the source table, not of each relation path, so audit
    # each child table exactly once.
    #
    # The current downstream representation indexes column audit rows by
    # child_table, so different event-time columns for the same child
    # table are intentionally rejected rather than silently conflated.
    table_time_cols: dict[str, str | None] = {}

    # A child-side relation key identifies which source rows belong to a
    # prediction entity.  Even when physically encoded as an integer, it is
    # an identifier rather than a quantitative source attribute and must not
    # receive arithmetic aggregations such as mean/std/min/max.
    relation_keys_by_table: dict[str, set[str]] = {}
    for relation in relations:
        if relation.get("status") != "accepted":
            continue
        table_name = str(relation["child_table"])
        child_fk = relation.get("child_fk")
        if child_fk is not None:
            relation_keys_by_table.setdefault(
                table_name,
                set(),
            ).add(str(child_fk))

    for relation in relations:
        table_name = str(relation["child_table"])
        raw_time_col = relation.get("child_event_time_col")
        time_col = (
            None
            if raw_time_col is None
            else str(raw_time_col)
        )

        if table_name in table_time_cols:
            if table_time_cols[table_name] != time_col:
                raise ValueError(
                    "multiple_event_time_columns_for_child_table:"
                    + table_name
                )
            continue

        table_time_cols[table_name] = time_col

        table = table_dict[table_name]
        frame = _table_df(table)
        pk = getattr(table, "pkey_col", None)
        fks = all_fks_by_table.get(
            table_name,
            set(),
        )

        for column in frame.columns:
            column_name = str(column)

            if column_name in relation_keys_by_table.get(
                table_name,
                set(),
            ):
                semantic = "identifier"
                accepted = False
                reason = "relation_child_fk"
            else:
                semantic, accepted, reason = _classify_column(
                    column=column_name,
                    series=frame[column],
                    primary_key=pk,
                    foreign_keys=fks,
                    time_col=time_col,
                    metadata=metadata,
                    options=options,
                )

            rows.append({
                "child_table": table_name,
                "column": str(column),
                "semantic_type": semantic,
                "accepted": accepted,
                "reason": (
                    f"{reason}|historical_source_safe"
                    if accepted
                    else reason
                ),
                "schema_rank_score": 0.0,
                "ranking_components": "",
                "rank_within_relation": "",
                "selected_for_candidate_generation": False,
            })

    return _rank_semantic_rows(
        rows,
        table_dict=table_dict,
        options=options,
    )



def generate_candidate_features(
    *,
    relations: Sequence[Mapping[str, Any]],
    column_audit: Sequence[Mapping[str, Any]],
    options: AutoOnboardingOptions,
) -> list[dict[str, Any]]:
    by_table: dict[str, list[Mapping[str, Any]]] = {}
    for row in column_audit:
        by_table.setdefault(str(row["child_table"]), []).append(row)
    candidates: list[dict[str, Any]] = []
    feature_index = 0
    for relation in relations:
        if relation["status"] != "accepted":
            continue
        table = str(relation["child_table"])
        for agg in ("count", "days_since_last", "active_span_days", "event_frequency"):
            candidates.append(_candidate(feature_index, relation, None, agg, "relation"))
            feature_index += 1
        numeric = sorted([
            row for row in by_table.get(table, [])
            if row["accepted"] and row["semantic_type"] in {"continuous_numeric", "ordinal_numeric"}
        ], key=lambda row: (int(row["rank_within_relation"]), str(row["column"])))[: options.max_numeric_columns]
        for row in numeric:
            row["selected_for_candidate_generation"] = True
        for row in numeric:
            for agg in ("mean", "std", "min", "max", "last"):
                candidates.append(_candidate(feature_index, relation, row["column"], agg, "numeric"))
                candidates[-1].update({
                    "column_semantic_type": row["semantic_type"],
                    "schema_rank_score": row["schema_rank_score"],
                    "ranking_components": row["ranking_components"],
                    "rank_within_relation": row["rank_within_relation"],
                    "relation_rank": relation.get("relation_rank", ""),
                })
                feature_index += 1
        categoricals = sorted([
            row for row in by_table.get(table, [])
            if row["accepted"] and row["semantic_type"] == "low_cardinality_categorical"
        ], key=lambda row: (int(row["rank_within_relation"]), str(row["column"])))[: options.max_categorical_columns]
        for row in categoricals:
            row["selected_for_candidate_generation"] = True
        for row in categoricals:
            candidates.append(_candidate(feature_index, relation, row["column"], "nunique", "categorical"))
            candidates[-1].update({
                "column_semantic_type": row["semantic_type"],
                "schema_rank_score": row["schema_rank_score"],
                "ranking_components": row["ranking_components"],
                "rank_within_relation": row["rank_within_relation"],
                "relation_rank": relation.get("relation_rank", ""),
            })
            feature_index += 1
    return candidates[: max(options.feature_budget * 4, options.feature_budget)]


def make_inner_temporal_splits(
    targets: pd.DataFrame,
    *,
    time_col: str,
    requested_folds: int = 1,
) -> dict[str, Any]:
    times = pd.to_datetime(targets[time_col], errors="coerce")
    unique_times = pd.Series(times.dropna().unique()).sort_values().reset_index(drop=True)
    if len(unique_times) < 2:
        cutoff_pos = max(1, int(len(targets) * 0.8))
        order = targets.assign(_time=times).sort_values(["_time"], kind="mergesort").index
        train_idx = list(order[:cutoff_pos])
        val_idx = list(order[cutoff_pos:])
        if not val_idx and train_idx:
            val_idx = [train_idx.pop()]
        return {
            "protocol": "row_order_fallback",
            "folds": [{
                "fold": 0,
                "train_indices": train_idx,
                "validation_indices": val_idx,
                "cutoff_timestamp": None,
                "train_start": None,
                "train_end": None,
                "validation_start": None,
                "validation_end": None,
                "unique_train_timestamps": 0,
                "unique_validation_timestamps": 0,
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
            }],
        }
    max_folds = max(1, min(int(requested_folds), 3, len(unique_times) - 1))
    folds = []
    if max_folds == 1:
        split_at = min(
            len(unique_times) - 1,
            max(1, int(math.ceil(len(unique_times) * 0.8))),
        )
        windows = [(split_at, len(unique_times))]
    else:
        tail_start = max(1, int(math.floor(len(unique_times) * 0.6)))
        tail_positions = list(range(tail_start, len(unique_times)))
        chunks = np.array_split(tail_positions, max_folds)
        windows = [
            (int(chunk[0]), int(chunk[-1]) + 1)
            for chunk in chunks
            if len(chunk)
        ]
    for fold_id, (start_pos, end_pos) in enumerate(windows):
        if start_pos <= 0 or start_pos >= len(unique_times):
            continue
        train_times = unique_times.iloc[:start_pos]
        val_times = unique_times.iloc[start_pos:end_pos]
        if val_times.empty:
            continue
        train_end = train_times.iloc[-1]
        val_start = val_times.iloc[0]
        val_end = val_times.iloc[-1]
        train_idx = targets.index[times <= train_end].tolist()
        val_idx = targets.index[(times >= val_start) & (times <= val_end)].tolist()
        if train_idx and val_idx:
            folds.append({
                "fold": fold_id,
                "train_indices": train_idx,
                "validation_indices": val_idx,
                "cutoff_timestamp": str(train_end),
                "validation_end_timestamp": str(val_end),
                "train_start": str(train_times.iloc[0]),
                "train_end": str(train_end),
                "validation_start": str(val_start),
                "validation_end": str(val_end),
                "unique_train_timestamps": int(len(train_times)),
                "unique_validation_timestamps": int(len(val_times)),
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
            })
    if not folds:
        return make_inner_temporal_splits(targets, time_col=time_col, requested_folds=1)
    return {"protocol": "expanding_window" if len(folds) > 1 else "single_holdout", "folds": folds}


def select_features(
    *,
    train_targets: pd.DataFrame,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    split_plan: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    selected: list[Mapping[str, Any]] = []
    remaining = list(candidates)
    cache = build_candidate_matrix_cache(
        train_targets=train_targets,
        table_dict=table_dict,
        metadata=metadata,
        candidates=candidates,
        split_plan=split_plan,
    )
    baseline_score = _score_feature_set_from_cache(
        cache=cache,
        metadata=metadata,
        features=[],
        options=options,
    )
    best_score = baseline_score["score"]
    trials.append(_trial_row(
        phase="baseline",
        fold="all",
        candidate="",
        before=[],
        after=[],
        metric=metadata["primary_metric"],
        score=baseline_score["score"],
        mean_score=baseline_score["score"],
        stability=baseline_score["stability"],
        improvement=0.0,
        accepted=True,
        reason="dummy_baseline",
    ))
    stopping_reason = "no_candidate_improves"
    while remaining and len(selected) < options.feature_budget:
        scored = []
        for candidate in remaining:
            trial_features = [*selected, candidate]
            score = _score_feature_set_from_cache(
                cache=cache,
                metadata=metadata,
                features=trial_features,
                options=options,
            )
            scored.append((candidate, score))
            improvement = _improvement(best_score, score["score"], metadata["metric_direction"])
            trials.append(_trial_row(
                phase="forward",
                fold="all",
                candidate=str(candidate["feature_id"]),
                before=selected,
                after=trial_features,
                metric=metadata["primary_metric"],
                score=score["score"],
                mean_score=score["score"],
                stability=score["stability"],
                improvement=improvement,
                accepted=False,
                reason="evaluated",
            ))
        scored.sort(
            key=lambda item: (
                -_improvement(best_score, item[1]["score"], metadata["metric_direction"]),
                item[1]["stability"],
                str(item[0]["feature_id"]),
            )
        )
        candidate, score = scored[0]
        improvement = _improvement(best_score, score["score"], metadata["metric_direction"])
        if improvement <= options.min_delta:
            _mark_trial_decision(
                trials,
                phase="forward",
                candidate_id=str(candidate["feature_id"]),
                accepted=False,
                reason="min_delta",
            )
            stopping_reason = "min_delta"
            break
        selected.append(candidate)
        selected[-1] = {
            **dict(selected[-1]),
            "selection_origin": "forward",
            "inner_fold_scores": score["fold_scores"],
            "backward_cleanup_result": "kept",
        }
        best_score = score["score"]
        _mark_trial_decision(
            trials,
            phase="forward",
            candidate_id=str(candidate["feature_id"]),
            accepted=True,
            reason="best_improvement",
        )
        remaining = [row for row in remaining if row["feature_id"] != candidate["feature_id"]]
        stopping_reason = "feature_budget" if len(selected) >= options.feature_budget else "no_candidate_improves"
    changed = True
    dropped = []
    while changed and selected:
        changed = False
        for candidate in list(selected):
            reduced = [row for row in selected if row["feature_id"] != candidate["feature_id"]]
            score = _score_feature_set_from_cache(
                cache=cache,
                metadata=metadata,
                features=reduced,
                options=options,
            )
            improvement = _improvement(best_score, score["score"], metadata["metric_direction"])
            remove = improvement >= -abs(options.min_delta)
            trials.append(_trial_row(
                phase="backward",
                fold="all",
                candidate=str(candidate["feature_id"]),
                before=selected,
                after=reduced,
                metric=metadata["primary_metric"],
                score=score["score"],
                mean_score=score["score"],
                stability=score["stability"],
                improvement=improvement,
                accepted=remove,
                reason="removed_negligible_or_improved" if remove else "kept",
            ))
            if remove:
                selected = reduced
                best_score = score["score"]
                dropped.append({**dict(candidate), "drop_reason": "backward_cleanup"})
                changed = True
                break
    fallback = False
    fallback_reason = ""
    fallback_level = "selected_relational_subset"
    if not selected:
        fallback = True
        fallback_reason = "no_inner_validation_improvement"
        selected, fallback_level = _fallback_features(
            candidates,
            table_dict=table_dict,
            metadata=metadata,
            options=options,
        )
        if selected:
            if any(row.get("kind") == "static_entity" for row in selected):
                cache = build_candidate_matrix_cache(
                    train_targets=train_targets,
                    table_dict=table_dict,
                    metadata=metadata,
                    candidates=selected,
                    split_plan=split_plan,
                )
            score = _score_feature_set_from_cache(
                cache=cache,
                metadata=metadata,
                features=selected,
                options=options,
            )
            best_score = score["score"]
            selected = [
                {
                    **dict(row),
                    "selection_origin": fallback_level,
                    "inner_fold_scores": score["fold_scores"],
                    "backward_cleanup_result": "fallback",
                }
                for row in selected
            ]
            trials.append(_trial_row(
                phase="fallback",
                fold="all",
                candidate="|".join(str(row["feature_id"]) for row in selected),
                before=[],
                after=selected,
                metric=metadata["primary_metric"],
                score=score["score"],
                mean_score=score["score"],
                stability=score["stability"],
                improvement=0.0,
                accepted=True,
                reason=fallback_level,
            ))
        else:
            fallback_level = "dummy_baseline"
    return {
        "selected_features": [dict(row) for row in selected],
        "dropped_features": dropped,
        "selection_trials": trials,
        "inner_selection_score": best_score,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "fallback_level": fallback_level,
        "stopping_reason": stopping_reason,
        "workload": {
            **cache["workload"],
            "model_trial_count": len(trials),
        },
    }


def final_evaluate(
    *,
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    selected_features: Sequence[Mapping[str, Any]],
    options: AutoOnboardingOptions,
) -> dict[str, Any]:
    train_x = materialize_feature_frame(
        train_targets,
        table_dict=table_dict,
        features=selected_features,
        entity_key=metadata["entity_key"],
        target_time_col=metadata["target_time_col"],
    )
    val_x = materialize_feature_frame(
        validation_targets,
        table_dict=table_dict,
        features=selected_features,
        entity_key=metadata["entity_key"],
        target_time_col=metadata["target_time_col"],
    )
    feature_cols = [row["output_column"] for row in selected_features]
    model = _fit_model(
        train_x[feature_cols] if feature_cols else pd.DataFrame(index=train_x.index),
        train_targets[metadata["label_col"]],
        problem_type=metadata["problem_type"],
        options=options,
    )
    pred = _predict_model(
        model,
        val_x[feature_cols] if feature_cols else pd.DataFrame(index=val_x.index),
        problem_type=metadata["problem_type"],
    )
    score = _metric_score(
        validation_targets[metadata["label_col"]],
        pred,
        metric=metadata["primary_metric"],
        problem_type=metadata["problem_type"],
    )
    predictions = validation_targets[[metadata["entity_key"], metadata["target_time_col"]]].copy()
    predictions["prediction"] = pred
    predictions["label"] = validation_targets[metadata["label_col"]].to_numpy()
    return {
        "official_validation_score": score,
        "official_validation_metric": {
            "split": "validation",
            "primary_metric": metadata["primary_metric"],
            "metric_direction": metadata["metric_direction"],
            metadata["primary_metric"]: score,
            "n_features": len(feature_cols),
        },
        "official_validation_predictions": predictions,
    }


def materialize_feature_frame(
    targets: pd.DataFrame,
    *,
    table_dict: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    entity_key: str,
    target_time_col: str,
) -> pd.DataFrame:
    out = targets.reset_index(drop=True).copy()
    static_features = [
        feature for feature in features
        if feature.get("kind") == "static_entity"
    ]
    if static_features:
        static_frame = _materialize_static_entity_features(
            targets=out,
            table_dict=table_dict,
            features=static_features,
            entity_key=entity_key,
        )
        for col in static_frame.columns:
            out[col] = static_frame[col].to_numpy()
    relational_features = [
        feature for feature in features
        if feature.get("kind") != "static_entity"
    ]
    for relation_key, rel_features in _features_by_relation(relational_features).items():
        relation = rel_features[0]
        child = _table_df(
            table_dict[relation["child_table"]]
        )

        target_lookup_column = str(
            relation.get(
                "target_lookup_column",
                entity_key,
            )
        )

        required_target_columns = list(
            dict.fromkeys(
                [
                    entity_key,
                    target_lookup_column,
                    target_time_col,
                ]
            )
        )

        missing_target_columns = [
            column
            for column in required_target_columns
            if column not in out.columns
        ]

        if missing_target_columns:
            raise ValueError(
                "missing_target_relation_lookup_columns:"
                + ",".join(sorted(missing_target_columns))
            )

        relation_frame = _materialize_relation_features(
            targets=out[required_target_columns],
            child=child,
            relation=relation,
            features=rel_features,
            entity_key=entity_key,
            target_time_col=target_time_col,
        )

        for col in relation_frame.columns:
            out[col] = relation_frame[col].to_numpy()
    return out


def build_candidate_matrix_cache(
    *,
    train_targets: pd.DataFrame,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    split_plan: Mapping[str, Any],
) -> dict[str, Any]:
    folds = []
    materialization_count = 0
    relation_scan_count = 0
    peak_bytes = 0
    for fold in split_plan["folds"]:
        inner_train = train_targets.loc[fold["train_indices"]].reset_index(drop=True)
        inner_val = train_targets.loc[fold["validation_indices"]].reset_index(drop=True)
        train_x = materialize_feature_frame(
            inner_train,
            table_dict=table_dict,
            features=candidates,
            entity_key=metadata["entity_key"],
            target_time_col=metadata["target_time_col"],
        )
        val_x = materialize_feature_frame(
            inner_val,
            table_dict=table_dict,
            features=candidates,
            entity_key=metadata["entity_key"],
            target_time_col=metadata["target_time_col"],
        )
        feature_cols = [row["output_column"] for row in candidates]
        materialization_count += 2
        relation_scan_count += 2 * len(_features_by_relation([
            row for row in candidates if row.get("kind") != "static_entity"
        ]))
        peak_bytes = max(
            peak_bytes,
            int(train_x[feature_cols].memory_usage(deep=True).sum()) if feature_cols else 0,
            int(val_x[feature_cols].memory_usage(deep=True).sum()) if feature_cols else 0,
        )
        folds.append({
            "fold": fold,
            "train_x": train_x,
            "validation_x": val_x,
            "train_y": inner_train[metadata["label_col"]].reset_index(drop=True),
            "validation_y": inner_val[metadata["label_col"]].reset_index(drop=True),
        })
    return {
        "folds": folds,
        "feature_columns": [row["output_column"] for row in candidates],
        "workload": {
            "candidate_matrix_materialization_count": materialization_count,
            "child_relation_scan_count": relation_scan_count,
            "cached_fold_count": len(folds),
            "candidate_column_count": len(candidates),
            "estimated_peak_matrix_bytes": peak_bytes,
            "selection_materialization_strategy": "fold_candidate_matrix_cache",
        },
    }


def _score_feature_set_from_cache(
    *,
    cache: Mapping[str, Any],
    metadata: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    options: AutoOnboardingOptions,
) -> dict[str, Any]:
    scores = []
    fold_scores = []
    feature_cols = [row["output_column"] for row in features]
    for fold_cache in cache["folds"]:
        train_x = fold_cache["train_x"]
        val_x = fold_cache["validation_x"]
        model = _fit_model(
            train_x[feature_cols] if feature_cols else pd.DataFrame(index=train_x.index),
            fold_cache["train_y"],
            problem_type=metadata["problem_type"],
            options=options,
        )
        pred = _predict_model(
            model,
            val_x[feature_cols] if feature_cols else pd.DataFrame(index=val_x.index),
            problem_type=metadata["problem_type"],
        )
        score = _metric_score(
            fold_cache["validation_y"],
            pred,
            metric=metadata["primary_metric"],
            problem_type=metadata["problem_type"],
        )
        scores.append(score)
        fold_scores.append({
            "fold": fold_cache["fold"]["fold"],
            "score": score,
        })
    return {
        "score": float(np.mean(scores)) if scores else math.nan,
        "stability": float(np.std(scores)) if len(scores) > 1 else 0.0,
        "fold_scores": fold_scores,
    }


def _trial_row(
    *,
    phase: str,
    fold: str,
    candidate: str,
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    metric: str,
    score: float,
    mean_score: float,
    stability: float,
    improvement: float,
    accepted: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "fold": fold,
        "candidate_added_or_removed": candidate,
        "selected_feature_ids_before_trial": "|".join(str(row["feature_id"]) for row in before),
        "selected_feature_ids_after_trial": "|".join(str(row["feature_id"]) for row in after),
        "metric": metric,
        "score": score,
        "mean_score": mean_score,
        "stability": stability,
        "improvement": improvement,
        "accepted_decision": accepted,
        "decision_reason": reason,
    }


def _mark_trial_decision(
    trials: list[dict[str, Any]],
    *,
    phase: str,
    candidate_id: str,
    accepted: bool,
    reason: str,
) -> None:
    for row in reversed(trials):
        if row["phase"] == phase and row["candidate_added_or_removed"] == candidate_id:
            row["accepted_decision"] = accepted
            row["decision_reason"] = reason
            return


def _materialize_relation_features(
    *,
    targets: pd.DataFrame,
    child: pd.DataFrame,
    relation: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    entity_key: str,
    target_time_col: str,
) -> pd.DataFrame:
    child_fk = relation["child_fk"]
    child_time_col = relation["child_event_time_col"]

    target_lookup_column = str(
        relation.get(
            "target_lookup_column",
            entity_key,
        )
    )

    if target_lookup_column not in targets.columns:
        raise ValueError(
            "missing_target_relation_lookup_columns:"
            + target_lookup_column
        )

    source_cols = [
        str(feature["source_column"])
        for feature in features
        if feature.get("source_column")
    ]
    required_cols = list(dict.fromkeys([child_fk, child_time_col, *source_cols]))
    work = child[required_cols].copy()
    work["_child_pos"] = range(len(work))
    work[child_time_col] = pd.to_datetime(work[child_time_col], errors="coerce")
    work = work[work[child_time_col].notna()].sort_values(
        [child_fk, child_time_col, "_child_pos"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = work.groupby(child_fk, sort=False, dropna=False)
    state_cols: list[str] = []
    if work.empty:
        result = pd.DataFrame(index=range(len(targets)))
        for feature in features:
            result[feature["output_column"]] = 0.0 if feature["aggregation"] in {"count", "nunique"} else np.nan
        return result
    work["__count"] = grouped.cumcount().astype(float) + 1.0
    work["__latest_time"] = work[child_time_col]
    work["__first_time"] = grouped[child_time_col].transform("first")
    work["__active_span_days"] = (
        work["__latest_time"] - work["__first_time"]
    ).dt.total_seconds() / 86400.0
    work["__event_frequency"] = work["__count"] / work["__active_span_days"].replace(0.0, np.nan)
    base_map = {
        "count": "__count",
        "days_since_last": "__latest_time",
        "active_span_days": "__active_span_days",
        "event_frequency": "__event_frequency",
    }
    for feature in features:
        agg = feature["aggregation"]
        source = feature.get("source_column")
        state_col = f"__state__{feature['feature_id']}"
        if agg in base_map:
            state_cols.append(base_map[agg])
            continue
        if agg == "nunique":
            work[state_col] = _cumulative_nunique(work, group_col=child_fk, value_col=source)
        else:
            numeric = pd.to_numeric(work[source], errors="coerce")
            valid = numeric.notna().astype(float)
            work["__value"] = numeric.fillna(0.0)
            count = valid.groupby(work[child_fk], sort=False).cumsum()
            total = work["__value"].groupby(work[child_fk], sort=False).cumsum()
            if agg == "mean":
                work[state_col] = total / count.replace(0.0, np.nan)
            elif agg == "std":
                sq = (work["__value"] * work["__value"]).groupby(work[child_fk], sort=False).cumsum()
                mean = total / count.replace(0.0, np.nan)
                work[state_col] = np.sqrt(np.maximum((sq / count.replace(0.0, np.nan)) - (mean * mean), 0.0))
            elif agg == "min":
                work[state_col] = numeric.groupby(work[child_fk], sort=False).cummin().groupby(work[child_fk], sort=False).ffill()
            elif agg == "max":
                work[state_col] = numeric.groupby(work[child_fk], sort=False).cummax().groupby(work[child_fk], sort=False).ffill()
            elif agg == "last":
                work[state_col] = numeric.groupby(work[child_fk], sort=False).ffill()
            else:
                raise ValueError(f"unsupported_feature_aggregation:{agg}")
        state_cols.append(state_col)
    keep = [child_fk, child_time_col, *sorted(set(state_cols))]
    state = work[keep]
    target_work = targets.copy().reset_index(drop=True)
    target_work["_target_pos"] = range(len(target_work))
    target_work[target_time_col] = pd.to_datetime(target_work[target_time_col], errors="coerce")
    result_arrays = {
        str(feature["output_column"]): np.full(len(target_work), np.nan, dtype="float64")
        for feature in features
    }
    target_valid = target_work[target_work[target_time_col].notna()].copy()
    if not target_valid.empty and not state.empty:
        state_lookup = state.copy()
        target_valid["__lookup_entity"], state_lookup["__lookup_entity"] = normalize_join_key_pair(
            target_valid[target_lookup_column],
            state_lookup[child_fk],
        )
        target_sorted = target_valid.sort_values(
            [target_time_col, "__lookup_entity", "_target_pos"],
            kind="mergesort",
        )
        state_sorted = state_lookup.sort_values(
            [child_time_col, "__lookup_entity"],
            kind="mergesort",
        )
        merged = pd.merge_asof(
            target_sorted,
            state_sorted,
            left_on=target_time_col,
            right_on=child_time_col,
            by="__lookup_entity",
            direction="backward",
            allow_exact_matches=bool(
                relation.get(
                    "allow_exact_matches",
                    True,
                )
            ),
        )
        positions = merged["_target_pos"].to_numpy()
        for feature in features:
            agg = feature["aggregation"]
            if agg == "count":
                values = merged["__count"]
            elif agg == "days_since_last":
                values = (
                    merged[target_time_col] - merged["__latest_time"]
                ).dt.total_seconds() / 86400.0
            elif agg == "active_span_days":
                values = merged["__active_span_days"]
            elif agg == "event_frequency":
                values = merged["__event_frequency"]
            else:
                values = merged[f"__state__{feature['feature_id']}"]
            result_arrays[str(feature["output_column"])][positions] = values.to_numpy()
    result = pd.DataFrame(result_arrays, index=range(len(target_work)))
    for feature in features:
        if feature["aggregation"] in {"count", "nunique"}:
            result[feature["output_column"]] = result[feature["output_column"]].fillna(0.0)
    return result


def _materialize_static_entity_features(
    *,
    targets: pd.DataFrame,
    table_dict: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    entity_key: str,
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=range(len(targets)))
    table_name = str(features[0]["child_table"])
    entity = _table_df(table_dict[table_name]).copy()
    entity = entity.drop_duplicates(subset=[entity_key], keep="first")
    result = pd.DataFrame(index=range(len(targets)))
    joined = targets[[entity_key]].merge(entity, on=entity_key, how="left", sort=False)
    for feature in features:
        source = str(feature["source_column"])
        out_col = str(feature["output_column"])
        if feature.get("column_semantic_type") == "low_cardinality_categorical":
            values = entity[source].dropna().astype(str)
            categories = {value: idx for idx, value in enumerate(sorted(values.unique()), start=1)}
            result[out_col] = joined[source].astype("string").map(categories).astype("float64")
        else:
            result[out_col] = pd.to_numeric(joined[source], errors="coerce")
    return result


def _score_feature_set(
    *,
    train_targets: pd.DataFrame,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    split_plan: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> dict[str, Any]:
    scores = []
    trials = []
    for fold in split_plan["folds"]:
        inner_train = train_targets.loc[fold["train_indices"]].reset_index(drop=True)
        inner_val = train_targets.loc[fold["validation_indices"]].reset_index(drop=True)
        train_x = materialize_feature_frame(
            inner_train,
            table_dict=table_dict,
            features=features,
            entity_key=metadata["entity_key"],
            target_time_col=metadata["target_time_col"],
        )
        val_x = materialize_feature_frame(
            inner_val,
            table_dict=table_dict,
            features=features,
            entity_key=metadata["entity_key"],
            target_time_col=metadata["target_time_col"],
        )
        feature_cols = [row["output_column"] for row in features]
        model = _fit_model(
            train_x[feature_cols] if feature_cols else pd.DataFrame(index=train_x.index),
            inner_train[metadata["label_col"]],
            problem_type=metadata["problem_type"],
            options=options,
        )
        pred = _predict_model(
            model,
            val_x[feature_cols] if feature_cols else pd.DataFrame(index=val_x.index),
            problem_type=metadata["problem_type"],
        )
        score = _metric_score(
            inner_val[metadata["label_col"]],
            pred,
            metric=metadata["primary_metric"],
            problem_type=metadata["problem_type"],
        )
        scores.append(score)
        trials.append({
            "fold": fold["fold"],
            "feature_ids": "|".join(str(row["feature_id"]) for row in features),
            "feature_count": len(features),
            "score": score,
            "metric": metadata["primary_metric"],
        })
    return {
        "score": float(np.mean(scores)) if scores else math.nan,
        "stability": float(np.std(scores)) if len(scores) > 1 else 0.0,
        "trials": trials,
    }


class _DummyModel:
    def __init__(self, value: Any, problem_type: str):
        self.value = value
        self.problem_type = problem_type

    def predict(self, x):
        return np.repeat(self.value, len(x))


def _fit_model(x: pd.DataFrame, y: pd.Series, *, problem_type: str, options: AutoOnboardingOptions):
    if x.shape[1] == 0 or y.nunique(dropna=True) <= 1:
        if problem_type == "regression":
            value = float(y.mean())
        elif problem_type in {"binary", "binary_classification"}:
            non_null = y.dropna()
            classes = np.unique(non_null.to_numpy())
            if len(classes) >= 2:
                positive_class = classes[1]
                value = float((non_null == positive_class).mean())
            else:
                value = 0.0
        else:
            value = y.mode(dropna=True).iloc[0]
        return _DummyModel(value, problem_type)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if problem_type == "regression":
        model = HistGradientBoostingRegressor(
            random_state=0,
            max_iter=100,
            min_samples_leaf=1,
        )
    else:
        # HistGradientBoostingClassifier defaults to early_stopping="auto".
        # For large classification datasets this creates an internal
        # stratified validation split. Such a split is mathematically
        # infeasible when any class occurs fewer than two times.
        #
        # The surrounding Auto pipeline already evaluates candidates using
        # explicit train-only temporal folds, so in this rare-class case we
        # disable only the decoder's internal early-stopping split rather
        # than altering the task labels or the temporal selection protocol.
        non_null_y = y.dropna()
        class_counts = non_null_y.value_counts()
        hgb_early_stopping = (
            False
            if int(class_counts.min()) < 2
            else "auto"
        )

        model = HistGradientBoostingClassifier(
            random_state=0,
            max_iter=100,
            min_samples_leaf=1,
            early_stopping=hgb_early_stopping,
        )
    pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), model)
    return pipe.fit(x, y)


def _predict_model(model, x: pd.DataFrame, *, problem_type: str) -> np.ndarray:
    if isinstance(model, _DummyModel):
        return model.predict(x)
    if problem_type in {"binary", "binary_classification"} and hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    return model.predict(x)


def _metric_score(y_true, y_pred, *, metric: str, problem_type: str) -> float:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_squared_error, roc_auc_score

    metric = metric.lower()
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric in {"mse", "mean_squared_error"}:
        return float(mean_squared_error(y_true, y_pred))
    if metric in {"auroc", "roc_auc"}:
        return float(roc_auc_score(y_true, y_pred))
    if metric in {"ap", "average_precision"}:
        return float(average_precision_score(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric in {"macro_f1", "f1_macro"}:
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"unsupported_metric:{metric}")


def _write_outputs(staging: Path, prepared: Mapping[str, Any]) -> None:
    _write_json(staging / "auto_onboarding_manifest.json", prepared["manifest"])
    _write_csv(staging / "relation_candidates.csv", prepared["relations"])
    _write_csv(staging / "column_semantic_audit.csv", prepared["column_audit"])
    _write_csv(staging / "candidate_features.csv", prepared["candidate_features"])
    _write_json(staging / "inner_temporal_splits.json", _public_split_plan(prepared["split_plan"]))
    _write_csv(staging / "selection_trials.csv", prepared["selection"]["selection_trials"])
    _write_json(staging / "selected_features.json", {
        "selected_features": prepared["selection"]["selected_features"],
        "dropped_features": prepared["selection"]["dropped_features"],
        "fallback": prepared["selection"]["fallback"],
        "fallback_reason": prepared["selection"]["fallback_reason"],
        "fallback_level": prepared["selection"].get("fallback_level", ""),
        "stopping_reason": prepared["selection"]["stopping_reason"],
        "workload": prepared["selection"].get("workload", {}),
    })
    if prepared["final"].get("official_validation_evaluated", False):
        _write_json(
            staging / "official_validation_metrics.json",
            prepared["final"]["official_validation_metric"],
        )
        prepared["final"]["official_validation_predictions"].to_parquet(
            staging / "official_validation_predictions.parquet",
            index=False,
        )
    manifest = json.loads((staging / "auto_onboarding_manifest.json").read_text(encoding="utf-8"))
    manifest["file_hashes"] = {
        path.name: _file_sha256(path)
        for path in sorted(staging.iterdir())
        if path.name != "auto_onboarding_manifest.json" and path.is_file()
    }
    _write_json(staging / "auto_onboarding_manifest.json", manifest)


def _manifest(
    *,
    dataset_name: str,
    task_name: str,
    output_dir: Path,
    relbench_version: str,
    dataset: Any,
    task: Any,
    database: Any,
    metadata: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
    column_audit: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    split_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    final: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "task": task_name,
        "status": "completed" if final["official_validation_score"] is not None else "dry_run_ready",
        "relbench_version": relbench_version,
        "implementation_version": AUTO_ONBOARDING_VERSION,
        "dataset_class": _class_name(dataset),
        "task_class": _class_name(task),
        "database_class": _class_name(database),
        "task_metadata": dict(metadata),
        "task_metadata_sources": {
            key: value for key, value in metadata.items() if key.endswith("_source")
        },
        "test_split_accessed": False,
        "official_validation_evaluated": bool(
            final.get("official_validation_evaluated", False)
        ),
        "official_validation_used_for_selection": False,
        "candidate_relations": list(relations),
        "selected_relations": sorted({
            row["child_table"] for row in selection["selected_features"]
        }),
        "rejected_relations": [
            row for row in relations if row.get("status") == "rejected"
        ],
        "column_semantic_audit": list(column_audit),
        "candidate_features": list(candidates),
        "selected_features": selection["selected_features"],
        "dropped_features": selection["dropped_features"],
        "inner_temporal_splits": _public_split_plan(split_plan),
        "selection_metric": metadata["primary_metric"],
        "selection_decoder": options.selection_decoder,
        "feature_budget": options.feature_budget,
        "stopping_reason": selection["stopping_reason"],
        "fallback": selection["fallback"],
        "fallback_reason": selection["fallback_reason"],
        "fallback_level": selection.get("fallback_level", ""),
        "inner_selection_score": selection["inner_selection_score"],
        "official_validation_metric": final["official_validation_metric"],
        "materialization_strategy": MATERIALIZATION_STRATEGY,
        "workload": selection.get("workload", {}),
        "file_hashes": {},
        "output_dir": str(output_dir),
    }


def _classify_column(
    *,
    column: str,
    series: pd.Series,
    primary_key: str | None,
    foreign_keys: set[str],
    time_col: str | None,
    metadata: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> tuple[str, bool, str]:
    lower = column.lower()
    if column == primary_key:
        return "identifier", False, "primary_key"
    if column in foreign_keys:
        return "foreign_key", False, "foreign_key"
    if column == metadata["entity_key"]:
        return "unsafe_or_unknown", False, "task_entity_key"
    if column == metadata["label_col"]:
        return (
            "unsafe_or_unknown",
            False,
            "target_name_collision_excluded|prediction_window_overlap",
        )
    if column == time_col or pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp", False, "timestamp"
    if re.search(r"(^id$|_id$|(^|_)key$|_key$|uuid|hash|url|email|phone|postal|zip)", lower):
        return "identifier", False, "id_like_name"
    if lower == "number" and series.nunique(dropna=True) / max(len(series), 1) > 0.8:
        return "identifier", False, "number_identifier_like"
    if pd.api.types.is_bool_dtype(series):
        return "boolean", False, "boolean_excluded_from_numeric"
    if pd.api.types.is_numeric_dtype(series):
        unique = int(series.nunique(dropna=True))
        semantic = "ordinal_numeric" if unique <= 16 else "continuous_numeric"
        return semantic, True, f"numeric_unique_count={unique}"
    non_null = series.dropna().astype(str)
    mean_len = float(non_null.str.len().mean()) if len(non_null) else 0.0
    cardinality = int(non_null.nunique(dropna=True))
    if mean_len > options.max_mean_text_length or cardinality > options.max_text_cardinality:
        return "free_text", False, f"text_or_high_cardinality cardinality={cardinality} mean_len={mean_len:.2f}"
    if cardinality <= options.max_categorical_cardinality:
        return "low_cardinality_categorical", True, f"categorical_cardinality={cardinality}"
    return "high_cardinality_categorical", False, f"categorical_cardinality={cardinality}"


def _rank_semantic_rows(
    rows: list[dict[str, Any]],
    *,
    table_dict: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> list[dict[str, Any]]:
    by_table: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_table.setdefault(str(row["child_table"]), []).append(row)
    for table_name, table_rows in by_table.items():
        frame = _table_df(table_dict[table_name])
        scored = []
        for row in table_rows:
            if not row["accepted"]:
                continue
            score, components = _schema_rank_score(
                frame[row["column"]],
                semantic_type=row["semantic_type"],
                options=options,
            )
            row["schema_rank_score"] = score
            row["ranking_components"] = json.dumps(components, sort_keys=True)
            scored.append(row)
        scored.sort(
            key=lambda row: (
                -float(row["schema_rank_score"]),
                str(row["semantic_type"]),
                str(row["column"]),
            )
        )
        for rank, row in enumerate(scored, start=1):
            row["rank_within_relation"] = rank
    return rows


def _schema_rank_score(
    series: pd.Series,
    *,
    semantic_type: str,
    options: AutoOnboardingOptions,
) -> tuple[float, dict[str, float]]:
    non_null_rate = float(series.notna().mean()) if len(series) else 0.0
    unique = int(series.nunique(dropna=True))
    unique_ratio = float(unique / max(series.notna().sum(), 1))
    components = {
        "non_null_rate": non_null_rate,
        "unique_ratio": unique_ratio,
        "non_constant": 1.0 if unique > 1 else 0.0,
    }
    if semantic_type in {"continuous_numeric", "ordinal_numeric"}:
        numeric = pd.to_numeric(series, errors="coerce")
        finite = (
            float(np.isfinite(numeric.dropna()).mean())
            if numeric.notna().any()
            else 0.0
        )
        components["finite_value_rate"] = finite
        components["reasonable_unique_ratio"] = 1.0 if 0.0 < unique_ratio < 0.95 else 0.0
        score = (
            0.35 * components["non_null_rate"]
            + 0.25 * components["non_constant"]
            + 0.25 * components["finite_value_rate"]
            + 0.15 * components["reasonable_unique_ratio"]
        )
    elif semantic_type == "low_cardinality_categorical":
        values = series.dropna().astype(str)
        probs = values.value_counts(normalize=True)
        entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
        max_entropy = math.log2(max(len(probs), 1)) if len(probs) else 1.0
        components["normalized_entropy"] = entropy / max(max_entropy, 1e-12)
        components["bounded_cardinality"] = 1.0 if unique <= options.max_categorical_cardinality else 0.0
        score = (
            0.40 * components["non_null_rate"]
            + 0.20 * components["non_constant"]
            + 0.20 * components["bounded_cardinality"]
            + 0.20 * components["normalized_entropy"]
        )
    else:
        score = 0.0
    return float(score), components


def _candidate(index: int, relation: Mapping[str, Any], source: str | None, agg: str, kind: str) -> dict[str, Any]:
    child = str(relation["child_table"])
    source_part = "" if source is None else f"_{source}"

    namespace = str(
        relation.get("feature_namespace", "")
    ).strip()

    namespace_part = (
        ""
        if not namespace
        else f"_{namespace}"
    )

    output = (
        f"f_{child}"
        f"{namespace_part}"
        f"{source_part}"
        f"_{agg}"
    )
    return {
        "feature_id": f"auto_{index:04d}",
        "kind": kind,
        "child_table": child,
        "child_fk": str(relation["child_fk"]),
        "child_event_time_col": str(relation["child_event_time_col"]),
        "target_lookup_column": str(
            relation.get(
                "target_lookup_column",
                relation.get(
                    "parent_key",
                    relation["child_fk"],
                ),
            )
        ),
        "feature_namespace": namespace,
        "allow_exact_matches": bool(
            relation.get("allow_exact_matches", True)
        ),
        "strict_before": bool(
            relation.get("strict_before", False)
        ),
        "source_column": source,
        "aggregation": agg,
        "output_column": output,
        "temporal_predicate": (
            f"{child}.{relation['child_event_time_col']} "
            f"{'<' if bool(relation.get('strict_before', False)) else '<='} "
            f"target.{relation.get('target_time_col', 'timestamp')}"
        ),
        "materialization_strategy": MATERIALIZATION_STRATEGY,
        "relation_rank": relation.get("relation_rank", ""),
    }


def _static_entity_features(
    *,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> list[dict[str, Any]]:
    entity_key = metadata["entity_key"]
    parent_tables = [
        name for name, table in sorted(table_dict.items())
        if getattr(table, "pkey_col", None) == entity_key
        and entity_key in _table_df(table).columns
    ]
    if len(parent_tables) != 1:
        return []
    table_name = parent_tables[0]
    table = table_dict[table_name]
    frame = _table_df(table)
    rows = []
    for column in frame.columns:
        semantic, accepted, reason = _classify_column(
            column=str(column),
            series=frame[column],
            primary_key=getattr(table, "pkey_col", None),
            foreign_keys=set((getattr(table, "fkey_col_to_pkey_table", {}) or {}).keys()),
            time_col=None,
            metadata=metadata,
            options=options,
        )
        if accepted and semantic in {
            "continuous_numeric",
            "ordinal_numeric",
            "low_cardinality_categorical",
        }:
            score, components = _schema_rank_score(
                frame[column],
                semantic_type=semantic,
                options=options,
            )
            rows.append({
                "column": str(column),
                "semantic_type": semantic,
                "schema_rank_score": score,
                "ranking_components": json.dumps(components, sort_keys=True),
                "reason": reason,
            })
    rows.sort(key=lambda row: (-float(row["schema_rank_score"]), row["column"]))
    features = []
    for idx, row in enumerate(rows[: options.feature_budget]):
        features.append({
            "feature_id": f"static_{idx:04d}",
            "kind": "static_entity",
            "child_table": table_name,
            "source_column": row["column"],
            "aggregation": "static",
            "output_column": f"f_{table_name}_{row['column']}_static",
            "column_semantic_type": row["semantic_type"],
            "schema_rank_score": row["schema_rank_score"],
            "ranking_components": row["ranking_components"],
            "rank_within_relation": idx + 1,
            "relation_rank": "",
            "selection_origin": "static_entity_fallback",
            "materialization_strategy": "entity_static_join",
        })
    return features


def _history_coverage(
    *,
    train_targets: pd.DataFrame,
    child: pd.DataFrame,
    entity_key: str,
    child_fk: str,
    child_time_col: str,
    target_time_col: str,
) -> float:
    target_times = pd.to_datetime(train_targets[target_time_col], errors="coerce")
    child_times = pd.to_datetime(child[child_time_col], errors="coerce")
    if len(train_targets) == 0:
        return 0.0
    child_work = child[[child_fk]].copy()
    child_work["_time"] = child_times
    first_source_time = child_work.dropna(subset=["_time"]).groupby(
        child_fk,
        sort=False,
        dropna=False,
    )["_time"].min()
    mapped_first_time = train_targets[entity_key].map(first_source_time)
    hits = mapped_first_time.notna() & target_times.notna() & (mapped_first_time <= target_times)
    return float(hits.mean())


def _features_by_relation(
    features: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    out: dict[
        tuple[str, str, str],
        list[Mapping[str, Any]],
    ] = {}

    for feature in features:
        target_lookup_column = str(
            feature.get(
                "target_lookup_column",
                feature["child_fk"],
            )
        )

        key = (
            str(feature["child_table"]),
            str(feature["child_fk"]),
            target_lookup_column,
        )

        out.setdefault(key, []).append(feature)

    return out


def _cumulative_nunique(frame: pd.DataFrame, *, group_col: str, value_col: str) -> pd.Series:
    counts = []
    seen_by_group: dict[object, set[object]] = {}
    for group, value in zip(frame[group_col], frame[value_col]):
        seen = seen_by_group.setdefault(group, set())
        if not pd.isna(value):
            seen.add(value)
        counts.append(float(len(seen)))
    return pd.Series(counts, index=frame.index, dtype="float64")


def _fallback_features(
    candidates: Sequence[Mapping[str, Any]],
    *,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    options: AutoOnboardingOptions,
) -> tuple[list[Mapping[str, Any]], str]:
    count_recency = [
        row for row in candidates
        if row["aggregation"] in {"count", "days_since_last"}
    ]
    if count_recency:
        first_table = count_recency[0]["child_table"]
        return (
            [row for row in count_recency if row["child_table"] == first_table][:2],
            "count_plus_recency_relation",
        )
    count_only = [row for row in candidates if row["aggregation"] == "count"]
    if count_only:
        return count_only[:1], "count_only_relation"
    static = _static_entity_features(
        table_dict=table_dict,
        metadata=metadata,
        options=options,
    )
    if static:
        return static, "static_entity_features"
    return [], "dummy_baseline"


def _estimate_selection_workload(
    *,
    split_plan: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    folds = len(split_plan.get("folds", ()))
    relations = len(_features_by_relation([
        row for row in candidates if row.get("kind") != "static_entity"
    ]))
    rows = sum(
        int(fold.get("train_rows", 0)) + int(fold.get("validation_rows", 0))
        for fold in split_plan.get("folds", ())
    )
    return {
        "candidate_matrix_materialization_count": folds * 2,
        "child_relation_scan_count": folds * 2 * relations,
        "model_trial_count": 0,
        "cached_fold_count": folds,
        "candidate_column_count": len(candidates),
        "estimated_peak_matrix_bytes": int(rows * max(len(candidates), 1) * 8),
        "selection_materialization_strategy": "fold_candidate_matrix_cache",
    }


def _improvement(old: float, new: float, direction: str) -> float:
    if math.isnan(old):
        return math.inf
    return (old - new) if direction == "lower" else (new - old)


def _normalize_problem_type(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(getattr(value, "value", getattr(value, "name", value))).lower()
    raw = raw.replace("tasktype.", "").replace("-", "_")
    if "regression" in raw:
        return "regression"
    if raw in {"binary", "binary_classification"} or "binary" in raw:
        return "binary"
    if raw in {"multiclass", "multi_class", "multiclass_classification"} or "multi" in raw:
        return "multiclass"
    return raw


def _infer_problem_type(label: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(label) and label.nunique(dropna=True) > 16:
        return "regression"
    if label.nunique(dropna=True) <= 2:
        return "binary"
    return "multiclass"


def _choose_metric(problem_type: str, raw: Any) -> str | None:
    metrics = _metric_names(raw)
    if problem_type == "regression":
        return _first_available(metrics, ("rmse", "mse", "mae")) or "rmse"
    if problem_type == "binary":
        return _first_available(metrics, ("roc_auc", "auroc", "ap", "average_precision")) or "roc_auc"
    if problem_type == "multiclass":
        return _first_available(metrics, ("accuracy", "macro_f1", "f1_macro")) or "accuracy"
    return None


def _metric_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [_normalize_metric(raw)]
    if isinstance(raw, Mapping):
        return [_normalize_metric(key) for key in raw.keys()]
    try:
        return [_normalize_metric(item) for item in raw]
    except TypeError:
        return [_normalize_metric(raw)]


def _normalize_metric(value: Any) -> str:
    raw_value = getattr(value, "__name__", None)
    if raw_value is None:
        raw_value = getattr(value, "value", getattr(value, "name", value))
    raw = str(raw_value).lower()
    raw = raw.replace(" ", "_").replace("-", "_")
    if raw == "auroc":
        return "roc_auc"
    if raw == "average_precision":
        return "ap"
    if raw == "f1_macro":
        return "macro_f1"
    return raw


def _first_available(metrics: Sequence[str], preferred: Sequence[str]) -> str | None:
    normalized = [_normalize_metric(item) for item in metrics]
    for metric in preferred:
        if metric in normalized:
            return metric
    return None


def _metric_direction(metric: str, raw: Any) -> str | None:
    if raw is not None:
        value = str(getattr(raw, "value", getattr(raw, "name", raw))).lower()
        if value in {"lower", "min", "minimize", "smaller"}:
            return "lower"
        if value in {"higher", "max", "maximize", "larger"}:
            return "higher"
    return "lower" if metric in {"rmse", "mse", "mae", "log_loss"} else "higher"


def _resolve_field(task: Any, attrs: Sequence[str], config: Mapping[str, Any], field: str, config_path: Path | None) -> tuple[Any, str]:
    if field in config:
        return config[field], f"metadata_config:{config_path}:{field}"
    for attr in attrs:
        value = getattr(task, attr, None)
        if value is not None:
            return value, f"task_attr:{attr}"
    for attr in attrs:
        value = getattr(task.__class__, attr, None)
        if value is not None:
            return value, f"task_class_attr:{attr}"
    return None, "missing"


def _task_metadata_from_config(path: Path | None, *, dataset_name: str, task_name: str) -> dict[str, Any]:
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("task_metadata_config must be a mapping")
    tasks = raw.get("tasks", {})
    if not isinstance(tasks, Mapping):
        raise ValueError("task_metadata_config tasks must be a mapping")
    row = tasks.get(f"{dataset_name}/{task_name}", {})
    if not isinstance(row, Mapping):
        raise ValueError("task metadata row must be a mapping")
    return dict(row)


def validation_schema_only(validation: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [str(col) for col in validation.columns],
        "dtypes": {str(col): str(dtype) for col, dtype in validation.dtypes.items()},
        "row_count": int(len(validation)),
    }


def _validate_target_frames(train: pd.DataFrame, validation_schema: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    validation_columns = list(validation_schema["columns"])
    if list(train.columns) != validation_columns:
        raise ValueError("incompatible_target_schemas")
    if train.empty or int(validation_schema["row_count"]) <= 0:
        raise ValueError("missing_official_split")
    for col in (metadata["entity_key"], metadata["target_time_col"], metadata["label_col"]):
        if col not in train.columns or col not in validation_columns:
            raise ValueError(f"missing_target_column:{col}")


def _public_split_plan(split_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": split_plan["protocol"],
        "folds": [
            {
                key: value for key, value in fold.items()
                if key not in {"train_indices", "validation_indices"}
            }
            for fold in split_plan["folds"]
        ],
    }


def _report(status: str, prepared: Mapping[str, Any], *, dry_run: bool, blockers: tuple[str, ...] = (), reused: bool = False) -> AutoRelBenchOnboardingReport:
    metadata = prepared["metadata"]
    selected = prepared["selection"]["selected_features"]
    return AutoRelBenchOnboardingReport(
        dataset=prepared["dataset_name"],
        task=prepared["task_name"],
        status=status,
        output_dir=prepared["output_dir"],
        blockers=blockers,
        dry_run=dry_run,
        task_type=metadata["problem_type"],
        metric=metadata["primary_metric"],
        metric_direction=metadata["metric_direction"],
        relation_candidates=len(prepared["relations"]),
        selected_relations=tuple(sorted({row["child_table"] for row in selected})),
        candidate_features=len(prepared["candidate_features"]),
        selected_features=len(selected),
        inner_selection_score=prepared["selection"]["inner_selection_score"],
        official_validation_score=prepared["final"]["official_validation_score"],
        fallback=bool(prepared["selection"]["fallback"]),
        test_split_accessed=False,
        reused=reused,
        workload=prepared["selection"].get("workload", {}),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
