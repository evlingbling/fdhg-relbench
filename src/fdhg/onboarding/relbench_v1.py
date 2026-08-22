from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import yaml


REL_BENCH_EXPORT_VERSION = "relbench-v1-export-v1"
LOADER_API_PATH = "relbench.datasets.get_dataset|relbench.tasks.get_task"
METRIC_POLICY_VERSION = "relbench-task-metadata-metric-policy-v1"
RELATION_SCREENING_TIE_TOLERANCE = 1e-12

HIGHER_IS_BETTER_METRICS = {
    "roc_auc",
    "auroc",
    "average_precision",
    "ap",
    "accuracy",
    "f1",
    "macro_f1",
    "f1_macro",
    "weighted_f1",
    "r2",
    "mrr",
}
LOWER_IS_BETTER_METRICS = {
    "rmse",
    "mae",
    "mse",
    "mean_squared_error",
    "log_loss",
    "cross_entropy",
}
METRIC_PRIORITY_BY_PROBLEM_TYPE = {
    "binary_classification": (
        "roc_auc",
        "average_precision",
        "accuracy",
        "log_loss",
    ),
    "multiclass_classification": (
        "accuracy",
        "macro_f1",
        "log_loss",
    ),
    "regression": (
        "rmse",
        "mae",
        "r2",
    ),
}


@dataclass(frozen=True)
class RelBenchV1ExportReport:
    dataset: str
    task: str
    status: str
    output_dir: Path
    config_path: Path
    blockers: tuple[str, ...]
    dry_run: bool
    reused: bool
    relation_count: int
    table_count: int
    train_rows: int
    validation_rows: int
    relbench_version: str
    dataset_class: str
    task_class: str
    table_names: tuple[str, ...]
    entity_key: str | None
    target_time_col: str | None
    label_col: str | None
    child_relation: str | None
    child_event_time_col: str | None


@dataclass(frozen=True)
class ResolvedTaskMetadata:
    dataset: str
    task: str
    entity_table: str
    entity_key: str
    target_time_col: str
    label_col: str
    problem_type: str
    primary_metric: str
    metric_direction: str
    child_table: str
    child_fk: str
    child_event_time_col: str
    relation_entity_key: str
    provenance: Mapping[str, str]
    candidate_relations_considered: tuple[Mapping[str, Any], ...]
    relation_selection_reason: str
    relation_selection_method: str
    official_validation_used_for_resolution: bool
    official_validation_used_for_selection: bool
    test_split_accessed: bool
    schema_fingerprint: str
    train_split_fingerprint: str
    selection_folds: int
    metric_policy_version: str
    candidate_relation_fingerprint: str
    relation_screening: tuple[Mapping[str, Any], ...] = ()
    relation_screening_folds: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "entity_table": self.entity_table,
            "entity_key": self.entity_key,
            "target_time_col": self.target_time_col,
            "label_col": self.label_col,
            "problem_type": self.problem_type,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "child_table": self.child_table,
            "child_fk": self.child_fk,
            "child_event_time_col": self.child_event_time_col,
            "relation_entity_key": self.relation_entity_key,
            "provenance": dict(self.provenance),
            "source_by_field": dict(self.provenance),
            "candidate_relations_considered": [
                dict(row) for row in self.candidate_relations_considered
            ],
            "relation_selection_reason": self.relation_selection_reason,
            "relation_selection_method": self.relation_selection_method,
            "official_validation_used_for_resolution": (
                self.official_validation_used_for_resolution
            ),
            "official_validation_used_for_selection": (
                self.official_validation_used_for_selection
            ),
            "test_split_accessed": self.test_split_accessed,
            "schema_fingerprint": self.schema_fingerprint,
            "train_split_fingerprint": self.train_split_fingerprint,
            "selection_folds": self.selection_folds,
            "metric_policy_version": self.metric_policy_version,
            "candidate_relation_fingerprint": (
                self.candidate_relation_fingerprint
            ),
            "relation_screening": [
                dict(row) for row in self.relation_screening
            ],
            "relation_screening_folds": [
                dict(row) for row in self.relation_screening_folds
            ],
        }


def resolve_relbench_task_metadata(
    *,
    dataset_name: str,
    task_name: str,
    dataset: Any,
    task: Any,
    database: Any,
    explicit_metadata: Mapping[str, Any] | None,
    selection_folds: int,
    output_dir: Path | None = None,
    train_df: pd.DataFrame | None = None,
    validation_df: pd.DataFrame | None = None,
) -> ResolvedTaskMetadata:
    table_dict = getattr(database, "table_dict", None)
    if not isinstance(table_dict, Mapping) or not table_dict:
        raise ValueError("missing_database_tables")
    if train_df is None:
        train_df = _table_df(task.get_table("train"))
    if validation_df is None:
        validation_df = _table_df(task.get_table("val"))
    explicit = dict(explicit_metadata or {})
    metadata = _resolve_semantic_metadata(
        dataset_name=dataset_name,
        task_name=task_name,
        task=task,
        train_df=train_df,
        validation_df=validation_df,
        explicit_metadata=explicit,
    )
    test_timestamp = getattr(dataset, "test_timestamp", None)
    if dataset_name.startswith("dbinfer-"):
        test_timestamp = None

    _validate_targets(
        train_df,
        validation_df,
        entity_key=metadata["entity_key"],
        target_time_col=metadata["target_time_col"],
        label_col=metadata["label_col"],
        test_timestamp=test_timestamp,
    )
    relation_threshold = float(
        metadata.get(
            "relation_threshold",
            0.98,
        )
    )

    try:
        relation_candidates = _verified_one_hop_relations(
            table_dict,
            entity_key=metadata["entity_key"],
            target_table=metadata.get("target_table"),
            threshold=relation_threshold,
        )
    except ValueError as exc:
        if (
            str(exc)
            != "relation_verification_blocker"
        ):
            raise
        relation_candidates = []

    if not any(
        bool(row.get("verified"))
        for row in relation_candidates
    ):
        target_table = metadata.get(
            "target_table"
        )

        if target_table:
            relation_candidates = (
                _verified_event_row_relations(
                    table_dict,
                    target_table=str(
                        target_table
                    ),
                    row_entity_key=str(
                        metadata[
                            "entity_key"
                        ]
                    ),
                    target_time_col=str(
                        metadata[
                            "target_time_col"
                        ]
                    ),
                    threshold=relation_threshold,
                )
            )

    if (
        not any(
            bool(row.get("verified"))
            for row in relation_candidates
        )
        and dataset_name.startswith("dbinfer-")
    ):
        relation_candidates = _verified_dbinfer_shared_fk_relations(
            task=task,
            table_dict=table_dict,
            train_df=train_df,
            target_lookup_column=str(metadata["entity_key"]),
            threshold=relation_threshold,
        )

    relation, selection = _select_relation(
        relation_candidates,
        metadata=metadata,
        table_dict=table_dict,
        train_df=train_df,
        selection_folds=selection_folds,
    )
    provenance = {
        key.removesuffix("_source"): str(value)
        for key, value in metadata.items()
        if key.endswith("_source")
    }
    for field, source in selection.get("provenance", {}).items():
        provenance[field] = str(source)
    resolved = ResolvedTaskMetadata(
        dataset=dataset_name,
        task=task_name,
        entity_table=str(metadata.get("target_table") or relation["parent_table"]),
        entity_key=str(metadata["entity_key"]),
        target_time_col=str(metadata["target_time_col"]),
        label_col=str(metadata["label_col"]),
        problem_type=str(metadata["problem_type"]),
        primary_metric=str(metadata["primary_metric"]),
        metric_direction=str(metadata["metric_direction"]),
        child_table=str(relation["child_table"]),
        child_fk=str(relation["child_column"]),
        child_event_time_col=str(relation["child_event_time_col"]),
        relation_entity_key=str(
            relation.get(
                "target_lookup_column",
                metadata["entity_key"],
            )
        ),
        provenance=provenance,
        candidate_relations_considered=tuple(dict(row) for row in relation_candidates),
        relation_selection_reason=str(selection["reason"]),
        relation_selection_method=str(selection["method"]),
        official_validation_used_for_resolution=False,
        official_validation_used_for_selection=False,
        test_split_accessed=False,
        schema_fingerprint=_schema_fingerprint(table_dict),
        train_split_fingerprint=_dataframe_identity_hash(train_df),
        selection_folds=int(selection_folds),
        metric_policy_version=METRIC_POLICY_VERSION,
        candidate_relation_fingerprint=_candidate_relation_fingerprint(
            relation_candidates
        ),
        relation_screening=tuple(selection.get("screening", ())),
        relation_screening_folds=tuple(selection.get("screening_folds", ())),
    )
    if output_dir is not None:
        _write_resolved_metadata_outputs(output_dir, resolved)
    return resolved


def resolved_metadata_reusable(
    payload: Mapping[str, Any],
    *,
    dataset_name: str,
    task_name: str,
    schema_fingerprint: str,
    train_split_fingerprint: str,
    selection_folds: int,
    candidate_relation_fingerprint: str,
) -> bool:
    return (
        payload.get("dataset") == dataset_name
        and payload.get("task") == task_name
        and payload.get("schema_fingerprint") == schema_fingerprint
        and payload.get("train_split_fingerprint") == train_split_fingerprint
        and int(payload.get("selection_folds", -1)) == int(selection_folds)
        and payload.get("metric_policy_version") == METRIC_POLICY_VERSION
        and payload.get("candidate_relation_fingerprint")
        == candidate_relation_fingerprint
    )


def export_relbench_v1(
    *,
    dataset_name: str,
    task_name: str,
    output_root: Path,
    config_output: Path,
    download: bool,
    write: bool = False,
    overwrite: bool = False,
    task_metadata_config: Path | None = None,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
) -> RelBenchV1ExportReport:
    try:
        dataset, task, relbench_version = (
            object_loader(dataset_name, task_name, download)
            if object_loader is not None
            else _load_relbench_objects(dataset_name, task_name, download)
        )
        prepared = _prepare_export(
            dataset_name=dataset_name,
            task_name=task_name,
            dataset=dataset,
            task=task,
            relbench_version=relbench_version,
            output_root=output_root,
            config_output=config_output,
            task_metadata_config=task_metadata_config,
        )
    except Exception as exc:
        return _blocked_report(
            dataset_name,
            task_name,
            output_root,
            config_output,
            blockers=(str(exc),),
            dry_run=not write,
        )

    if prepared["blockers"]:
        return _blocked_report(
            dataset_name,
            task_name,
            output_root,
            config_output,
            blockers=tuple(prepared["blockers"]),
            dry_run=not write,
            prepared=prepared,
        )

    output_dir = prepared["output_dir"]
    manifest_path = output_dir / "export_manifest.json"
    if output_dir.exists() and not overwrite:
        if manifest_path.exists() and config_output.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source_identity_hash")
                == prepared["manifest"]["source_identity_hash"]
                and manifest.get("generated_onboarding_config_hash")
                == _text_sha256(config_output.read_text(encoding="utf-8"))
            ):
                return _report("reused", prepared, dry_run=not write, reused=True)
        if write:
            return _report(
                "blocked",
                prepared,
                dry_run=False,
                blockers=("conflicting_existing_output_identity",),
            )

    if not write:
        return _report("dry_run_ready", prepared, dry_run=True)
    if (
        config_output.exists()
        and not overwrite
        and config_output.read_text(encoding="utf-8") != prepared["config_text"]
    ):
        return _report(
            "blocked",
            prepared,
            dry_run=False,
            blockers=("conflicting_existing_output_identity",),
        )

    staging = output_dir.parent / f"_{output_dir.name}.staging"
    if staging.exists():
        return _report(
            "blocked",
            prepared,
            dry_run=False,
            blockers=("partial_staging_output",),
        )
    staging.mkdir(parents=True)
    try:
        _write_export(staging, prepared)
        _validate_export(staging)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(output_dir)
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        _write_config(config_output, prepared["config_text"], overwrite=overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _report("completed", prepared, dry_run=False)


def _load_relbench_objects(
    dataset_name: str,
    task_name: str,
    download: bool,
) -> tuple[Any, Any, str]:
    from importlib.metadata import version

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset = get_dataset(dataset_name, download=download)
    task = get_task(dataset_name, task_name, download=download)
    return dataset, task, version("relbench")


def _prepare_export(
    *,
    dataset_name: str,
    task_name: str,
    dataset: Any,
    task: Any,
    relbench_version: str,
    output_root: Path,
    config_output: Path,
    task_metadata_config: Path | None,
) -> dict[str, Any]:
    database = dataset.get_db()
    table_dict = getattr(database, "table_dict", None)
    if not isinstance(table_dict, Mapping) or not table_dict:
        raise ValueError("missing_database_tables")
    train_table = task.get_table("train")
    validation_table = task.get_table("val")
    train_df = _table_df(train_table)
    validation_df = _table_df(validation_table)
    if train_df.empty:
        raise ValueError("missing_task_train_split")
    if validation_df.empty:
        raise ValueError("missing_task_validation_split")
    explicit_metadata = _task_metadata_from_config(
        task_metadata_config,
        dataset_name=dataset_name,
        task_name=task_name,
    )
    resolved_metadata = resolve_relbench_task_metadata(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata=explicit_metadata,
        selection_folds=1,
        output_dir=None,
        train_df=train_df,
        validation_df=validation_df,
    )
    metadata = _metadata_mapping_from_resolved(
        resolved_metadata
    )

    if (
        resolved_metadata.relation_entity_key
        != resolved_metadata.entity_key
    ):
        train_df = _enrich_target_relation_key(
            train_df,
            table_dict=table_dict,
            target_table=(
                resolved_metadata.entity_table
            ),
            row_entity_key=(
                resolved_metadata.entity_key
            ),
            relation_entity_key=(
                resolved_metadata.relation_entity_key
            ),
            target_time_col=(
                resolved_metadata.target_time_col
            ),
        )

        validation_df = (
            _enrich_target_relation_key(
                validation_df,
                table_dict=table_dict,
                target_table=(
                    resolved_metadata.entity_table
                ),
                row_entity_key=(
                    resolved_metadata.entity_key
                ),
                relation_entity_key=(
                    resolved_metadata.relation_entity_key
                ),
                target_time_col=(
                    resolved_metadata.target_time_col
                ),
            )
        )

    if list(train_df.columns) != list(validation_df.columns):
        raise ValueError("incompatible_target_schemas")
    relation_candidates = [
        dict(row)
        for row
        in resolved_metadata.candidate_relations_considered
    ]

    selected_relation_matches = [
        row
        for row in relation_candidates
        if (
            str(row.get("child_table"))
            == str(resolved_metadata.child_table)
            and str(row.get("child_column"))
            == str(resolved_metadata.child_fk)
            and str(
                row.get(
                    "child_event_time_col"
                )
            )
            == str(
                resolved_metadata.child_event_time_col
            )
            and str(
                row.get(
                    "target_lookup_column",
                    resolved_metadata.entity_key,
                )
            )
            == str(
                resolved_metadata.relation_entity_key
            )
        )
    ]

    if len(selected_relation_matches) != 1:
        raise ValueError(
            "selected_relation_bridge_resolution_failed:"
            f"{len(selected_relation_matches)}"
        )

    relation = dict(
        selected_relation_matches[0]
    )
    output_dir = output_root / dataset_name / task_name
    config_text = _onboarding_config_text(
        dataset_name=dataset_name,
        task_name=task_name,
        output_dir=output_dir,
        config_output=config_output,
        table_dict=table_dict,
        relation=relation,
        metadata=metadata,
    )
    manifest = _manifest(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset=dataset,
        task=task,
        database=database,
        table_dict=table_dict,
        train_df=train_df,
        validation_df=validation_df,
        relation=relation,
        relation_candidates=relation_candidates,
        metadata=metadata,
        relbench_version=relbench_version,
        config_text=config_text,
    )
    return {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "output_dir": output_dir,
        "config_output": config_output,
        "dataset": dataset,
        "task": task,
        "table_dict": table_dict,
        "train_df": train_df,
        "validation_df": validation_df,
        "relation": relation,
        "relation_candidates": relation_candidates,
        "metadata": metadata,
        "resolved_metadata": resolved_metadata,
        "manifest": manifest,
        "config_text": config_text,
        "blockers": (),
        "relbench_version": relbench_version,
    }


def _resolve_semantic_metadata(
    *,
    dataset_name: str,
    task_name: str,
    task: Any,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    explicit_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(explicit_metadata)
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    fields = {
        "entity_key": ("entity_col", "entity_key", "entity_col_name"),
        "target_time_col": ("time_col", "target_time_col"),
        "label_col": ("target_col", "label_col"),
        "problem_type": ("task_type", "problem_type"),
    }
    for field, attrs in fields.items():
        value, source = _resolve_metadata_field(
            task,
            field=field,
            attrs=attrs,
            config=config,
            config_path=None,
            dataset_name=dataset_name,
            task_name=task_name,
        )
        if value is not None:
            values[field] = _normalize_metadata_value(
                field,
                value,
            )
            sources[f"{field}_source"] = source
    missing = [field for field in fields if not values.get(field)]
    if missing:
        raise ValueError("missing_task_metadata:" + ",".join(sorted(missing)))
    primary_metric, metric_source = _resolve_primary_metric(
        task=task,
        problem_type=str(values["problem_type"]),
        explicit_metadata=config,
    )
    values["primary_metric"] = primary_metric
    sources["primary_metric_source"] = metric_source
    metric_direction, direction_source = _resolve_metric_direction(
        primary_metric,
        explicit_metadata=config,
    )
    values["metric_direction"] = metric_direction
    sources["metric_direction_source"] = direction_source

    # The task-frame entity column may differ from the entity-table
    # primary key, e.g. task.UserId -> users.Id in rel-stack.
    target_table = config.get("target_table") or getattr(
        task,
        "entity_table",
        None,
    )
    if target_table:
        values["target_table"] = str(target_table)
        sources["target_table_source"] = (
            "explicit_metadata:"
            f"tasks.{dataset_name}/{task_name}.target_table"
            if config.get("target_table")
            else "task_attr:entity_table"
        )
    entity_key = values["entity_key"]
    target_time_col = values["target_time_col"]
    label_col = values["label_col"]
    if (
        not entity_key
        or entity_key not in train_df.columns
        or entity_key not in validation_df.columns
    ):
        raise ValueError("missing_entity_key")
    if (
        not target_time_col
        or target_time_col not in train_df.columns
        or target_time_col not in validation_df.columns
    ):
        raise ValueError("missing_target_timestamp")
    if (
        not label_col
        or label_col not in train_df.columns
        or label_col not in validation_df.columns
    ):
        raise ValueError("missing_label")
    supported_problem_types = {
        "regression",
        "binary_classification",
        "multiclass_classification",
    }
    if values["problem_type"] not in supported_problem_types:
        raise ValueError(
            "unsupported_task_type:"
            f"{values['problem_type']}"
        )
    if not values["primary_metric"] or not values["metric_direction"]:
        raise ValueError("unsupported_task_type")
    for optional in (
        "child_table",
        "child_fk",
        "child_event_time_col",
        "target_lookup_column",
        "relation_threshold",
    ):
        if optional in config:
            values[optional] = config[optional]
            sources[f"{optional}_source"] = (
                "explicit_metadata:"
                f"tasks.{dataset_name}/{task_name}.{optional}"
            )
    if any(key in values for key in ("child_table", "child_fk", "child_event_time_col")):
        required_relation = ("child_table", "child_fk", "child_event_time_col")
        missing_relation = [
            key for key in required_relation if not values.get(key)
        ]
        if missing_relation:
            raise ValueError(
                "missing_relation_metadata:"
                + ",".join(sorted(missing_relation))
            )
        sources["relation_source"] = (
            "explicit_metadata:"
            f"tasks.{dataset_name}/{task_name}"
        )
    else:
        sources["relation_source"] = "relbench_schema:auto_verified"
    return {
        **values,
        **sources,
    }


def _task_metadata(
    *,
    dataset_name: str,
    task_name: str,
    task: Any,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    task_metadata_config: Path | None,
) -> dict[str, Any]:
    return _resolve_semantic_metadata(
        dataset_name=dataset_name,
        task_name=task_name,
        task=task,
        train_df=train_df,
        validation_df=validation_df,
        explicit_metadata=_task_metadata_from_config(
            task_metadata_config,
            dataset_name=dataset_name,
            task_name=task_name,
        ),
    )



def _enrich_target_relation_key(
    target_df: pd.DataFrame,
    *,
    table_dict: Mapping[str, Any],
    target_table: str,
    row_entity_key: str,
    relation_entity_key: str,
    target_time_col: str,
) -> pd.DataFrame:
    """Attach a relational lookup key while preserving prediction-row identity."""

    if relation_entity_key == row_entity_key:
        return target_df.reset_index(drop=True).copy()

    if target_table not in table_dict:
        raise ValueError(
            f"missing_target_table_for_relation_enrichment:{target_table}"
        )

    source = _table_df(
        table_dict[target_table]
    ).reset_index(drop=True)

    out = target_df.reset_index(drop=True).copy()

    required_source = [
        row_entity_key,
        relation_entity_key,
        target_time_col,
    ]

    missing_source = [
        col
        for col in required_source
        if col not in source.columns
    ]

    if missing_source:
        raise ValueError(
            "missing_source_columns_for_relation_enrichment:"
            + ",".join(sorted(missing_source))
        )

    if row_entity_key not in out.columns:
        raise ValueError(
            f"missing_target_row_identity:{row_entity_key}"
        )

    if source[row_entity_key].duplicated().any():
        raise ValueError(
            "non_unique_source_row_identity_for_relation_enrichment"
        )

    lookup = source[
        required_source
    ].copy().rename(
        columns={
            target_time_col: "__source_target_time",
        }
    )

    enriched = out.merge(
        lookup,
        on=row_entity_key,
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if len(enriched) != len(out):
        raise AssertionError(
            "relation_enrichment_changed_target_row_count"
        )

    target_time = pd.to_datetime(
        enriched[target_time_col],
        errors="coerce",
    )

    source_time = pd.to_datetime(
        enriched["__source_target_time"],
        errors="coerce",
    )

    mismatch = ~(
        (target_time == source_time)
        | (
            target_time.isna()
            & source_time.isna()
        )
    )

    if bool(mismatch.any()):
        raise ValueError(
            "relation_enrichment_timestamp_verification_failed"
        )

    enriched = enriched.drop(
        columns=["__source_target_time"]
    )

    if enriched[
        relation_entity_key
    ].isna().all():
        raise ValueError(
            "relation_enrichment_lookup_key_all_missing"
        )

    return enriched


def _verified_event_row_relations(
    table_dict: Mapping[str, Any],
    *,
    target_table: str,
    row_entity_key: str,
    target_time_col: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Discover target-row FK -> parent -> temporal-child history relations."""

    if target_table not in table_dict:
        return []

    target_obj = table_dict[target_table]
    target_df = _table_df(target_obj)

    if (
        getattr(
            target_obj,
            "pkey_col",
            None,
        )
        != row_entity_key
        or row_entity_key
        not in target_df.columns
    ):
        return []

    target_fks = (
        getattr(
            target_obj,
            "fkey_col_to_pkey_table",
            {},
        )
        or {}
    )

    candidates: list[dict[str, Any]] = []

    for (
        target_lookup_col,
        parent_table,
    ) in sorted(target_fks.items()):

        if (
            target_lookup_col
            not in target_df.columns
            or parent_table
            not in table_dict
        ):
            continue

        parent_obj = table_dict[
            parent_table
        ]
        parent_df = _table_df(parent_obj)
        parent_key = getattr(
            parent_obj,
            "pkey_col",
            None,
        )

        if (
            parent_key is None
            or parent_key
            not in parent_df.columns
        ):
            continue

        lookup_non_null = target_df[
            target_lookup_col
        ].dropna()

        lookup_coverage = (
            float(
                lookup_non_null.isin(
                    set(
                        parent_df[
                            parent_key
                        ].dropna()
                    )
                ).mean()
            )
            if len(lookup_non_null)
            else 0.0
        )

        lookup_dtype_compatible = (
            str(
                target_df[
                    target_lookup_col
                ].dtype
            )
            == str(
                parent_df[
                    parent_key
                ].dtype
            )
        )

        if (
            not lookup_dtype_compatible
            or lookup_coverage
            < threshold
        ):
            continue

        for (
            child_name,
            child_obj,
        ) in sorted(table_dict.items()):

            child_fks = (
                getattr(
                    child_obj,
                    "fkey_col_to_pkey_table",
                    {},
                )
                or {}
            )

            child_time_col = getattr(
                child_obj,
                "time_col",
                None,
            )

            child_df = _table_df(
                child_obj
            )

            if (
                child_time_col is None
                or str(
                    child_time_col
                )
                not in child_df.columns
            ):
                continue

            for (
                child_fk,
                child_parent,
            ) in sorted(child_fks.items()):

                if (
                    child_parent
                    != parent_table
                ):
                    continue

                if (
                    child_fk
                    not in child_df.columns
                ):
                    continue

                child_non_null = child_df[
                    child_fk
                ].dropna()

                child_coverage = (
                    float(
                        child_non_null.isin(
                            set(
                                parent_df[
                                    parent_key
                                ].dropna()
                            )
                        ).mean()
                    )
                    if len(child_non_null)
                    else 0.0
                )

                dtype_compatible = (
                    str(
                        child_df[
                            child_fk
                        ].dtype
                    )
                    == str(
                        parent_df[
                            parent_key
                        ].dtype
                    )
                )

                row = {
                    "parent_table":
                        str(parent_table),
                    "parent_column":
                        str(parent_key),
                    "child_table":
                        str(child_name),
                    "child_column":
                        str(child_fk),
                    "child_event_time_col":
                        str(child_time_col),
                    "target_lookup_column":
                        str(target_lookup_col),
                    "relation_orientation":
                        "target_outgoing_fk_to_parent_history",
                    "strict_before": bool(
                        str(child_name) == str(target_table)
                        and str(child_time_col) == str(target_time_col)
                    ),
                    "parent_primary_key_proven":
                        True,
                    "child_fk_present":
                        True,
                    "dtype_compatible":
                        dtype_compatible,
                    "referential_coverage":
                        child_coverage,
                    "target_lookup_coverage":
                        lookup_coverage,
                    "target_lookup_dtype_compatible":
                        lookup_dtype_compatible,
                    "child_time_column_present":
                        True,
                }

                row["verified"] = all((
                    row[
                        "parent_primary_key_proven"
                    ],
                    row[
                        "child_fk_present"
                    ],
                    row[
                        "dtype_compatible"
                    ],
                    row[
                        "referential_coverage"
                    ] >= threshold,
                    row[
                        "target_lookup_coverage"
                    ] >= threshold,
                    row[
                        "target_lookup_dtype_compatible"
                    ],
                    row[
                        "child_time_column_present"
                    ],
                ))

                candidates.append(row)

    return candidates


def _verified_dbinfer_shared_fk_relations(
    *,
    task: Any,
    table_dict: Mapping[str, Any],
    train_df: pd.DataFrame,
    target_lookup_column: str,
    threshold: float = 0.98,
) -> list[dict[str, Any]]:
    """Recover DBInfer shared-FK relations without scanning source values.

    DBInfer task rows may use dense mapped entity IDs while materialized
    source tables retain raw foreign-key IDs.  The adapter metadata
    preserves the logical FK declarations and the task adapter exposes
    the raw-to-mapped entity mapping.

    Verification here is therefore structural:
      - task lookup column declares a logical parent FK,
      - source column declares that same FK,
      - source time column exists,
      - target mapped IDs can be inverted into the raw key space.

    Source tables are never mutated and source FK values are not scanned.
    """

    task_adapter = getattr(
        task,
        "_task_adapter",
        None,
    )
    dbinfer_task = getattr(
        task_adapter,
        "dbinfer_task",
        None,
    )
    task_meta = getattr(
        dbinfer_task,
        "metadata",
        None,
    )
    entity_mapping = getattr(
        task_adapter,
        "entity_mapping",
        None,
    )

    dataset_adapter = getattr(
        task,
        "_dataset_adapter",
        None,
    )
    dbinfer_dataset = getattr(
        dataset_adapter,
        "dbinfer_dataset",
        None,
    )
    dataset_meta = getattr(
        dbinfer_dataset,
        "metadata",
        None,
    )

    if (
        task_meta is None
        or dataset_meta is None
        or not isinstance(entity_mapping, Mapping)
        or not entity_mapping
        or target_lookup_column not in train_df.columns
    ):
        return []

    task_lookup_schema = next(
        (
            column
            for column in (
                getattr(task_meta, "columns", None)
                or ()
            )
            if (
                str(getattr(column, "name", ""))
                == str(target_lookup_column)
                and str(getattr(column, "dtype", ""))
                == "foreign_key"
                and getattr(column, "link_to", None)
            )
        ),
        None,
    )

    if task_lookup_schema is None:
        return []

    shared_parent = str(
        task_lookup_schema.link_to
    )
    if "." not in shared_parent:
        return []

    (
        logical_parent_table,
        logical_parent_key,
    ) = shared_parent.rsplit(".", 1)

    inverse_mapping = {
        mapped: raw
        for raw, mapped in entity_mapping.items()
    }

    target_values = train_df[
        target_lookup_column
    ]

    raw_target_values = target_values.map(
        inverse_mapping
    )

    target_mapping_coverage = (
        float(raw_target_values.notna().mean())
        if len(raw_target_values)
        else 0.0
    )

    raw_target_non_null = (
        raw_target_values.dropna()
    )

    # Do not admit the target/entity table itself as a historical source.
    target_entity_table = str(
        getattr(task, "entity_table", "")
        or ""
    )

    candidates: list[dict[str, Any]] = []

    for table_schema in (
        getattr(dataset_meta, "tables", None)
        or ()
    ):
        child_table = str(
            getattr(table_schema, "name", "")
        )
        child_time_col = getattr(
            table_schema,
            "time_column",
            None,
        )

        if (
            not child_table
            or child_table == target_entity_table
            or child_table not in table_dict
            or child_time_col is None
        ):
            continue

        child_schema = next(
            (
                column
                for column in (
                    getattr(
                        table_schema,
                        "columns",
                        None,
                    )
                    or ()
                )
                if (
                    str(getattr(column, "dtype", ""))
                    == "foreign_key"
                    and str(
                        getattr(
                            column,
                            "link_to",
                            "",
                        )
                    )
                    == shared_parent
                )
            ),
            None,
        )

        if child_schema is None:
            continue

        child_column = str(
            getattr(child_schema, "name", "")
        )

        child_df = _table_df(
            table_dict[child_table]
        )

        if (
            child_column not in child_df.columns
            or str(child_time_col)
            not in child_df.columns
        ):
            continue

        # dtype inspection is metadata-level / Series dtype only:
        # no source-value scan is performed.
        dtype_compatible = (
            not raw_target_non_null.empty
            and str(raw_target_non_null.dtype)
            == str(child_df[child_column].dtype)
        )

        row = {
            "parent_table":
                logical_parent_table,
            "parent_column":
                logical_parent_key,
            "child_table":
                child_table,
            "child_column":
                child_column,
            "child_event_time_col":
                str(child_time_col),
            "target_lookup_column":
                str(target_lookup_column),
            "relation_orientation":
                "dbinfer_shared_declared_fk",
            "shared_declared_parent":
                shared_parent,
            "strict_before":
                False,

            # Customer may be logical-only and absent from the
            # materialized RelBench DB, so do not claim its physical
            # PK was inspected.
            "parent_primary_key_proven":
                False,
            "logical_parent_declared":
                True,
            "child_fk_present":
                True,
            "dtype_compatible":
                dtype_compatible,
            "child_time_column_present":
                True,

            "target_mapping_coverage":
                target_mapping_coverage,
            "dbinfer_declared_fk_provenance":
                True,
            "target_lookup_value_transform":
                "dbinfer_inverse_entity_mapping",
            "verification_basis":
                "declared_shared_fk_plus_target_inverse_mapping",
        }

        row["verified"] = all((
            row[
                "logical_parent_declared"
            ],
            row[
                "child_fk_present"
            ],
            row[
                "dtype_compatible"
            ],
            row[
                "child_time_column_present"
            ],
            row[
                "dbinfer_declared_fk_provenance"
            ],
            row[
                "target_mapping_coverage"
            ] >= threshold,
        ))

        candidates.append(row)

    return candidates


def _verified_one_hop_relations(
    table_dict: Mapping[str, Any],
    *,
    entity_key: str,
    target_table: str | None = None,
    parent_key: str | None = None,
    threshold: float = 0.98,
) -> list[dict[str, Any]]:
    resolved_parent_key = parent_key

    if target_table:
        if target_table not in table_dict:
            raise ValueError(
                f"relation_metadata_inconsistency:"
                f"missing_target_table:{target_table}"
            )

        target = table_dict[target_table]
        inferred_parent_key = getattr(target, "pkey_col", None)

        if resolved_parent_key is None:
            resolved_parent_key = inferred_parent_key

        parent_candidates = (
            [target_table]
            if resolved_parent_key
            and resolved_parent_key in _table_df(target).columns
            else []
        )
    else:
        resolved_parent_key = resolved_parent_key or entity_key
        parent_candidates = [
            name for name, table in sorted(table_dict.items())
            if getattr(table, "pkey_col", None) == resolved_parent_key
            and resolved_parent_key in _table_df(table).columns
        ]
    if len(parent_candidates) != 1:
        raise ValueError("relation_metadata_inconsistency")
    parent = parent_candidates[0]
    candidates = []
    for child_name, table in sorted(table_dict.items()):
        fkeys = getattr(table, "fkey_col_to_pkey_table", {}) or {}
        for child_col, parent_table in sorted(fkeys.items()):
            if parent_table != parent:
                continue
            child = _table_df(table)
            parent_df = _table_df(table_dict[parent])
            parent_is_pk = (
                getattr(table_dict[parent], "pkey_col", None)
                == resolved_parent_key
            )
            child_fk_present = child_col in child.columns
            dtype_compatible = (
                child_fk_present
                and str(child[child_col].dtype)
                == str(parent_df[resolved_parent_key].dtype)
            )
            non_null = (
                child[child_col].dropna()
                if child_fk_present
                else pd.Series(dtype="object")
            )
            referential_coverage = (
                float(
                    non_null.isin(
                        set(parent_df[resolved_parent_key].dropna())
                    ).mean()
                )
                if len(non_null)
                else 0.0
            )
            child_event_time_col = getattr(table, "time_col", None)
            row = {
                "parent_table": parent,
                "parent_column": resolved_parent_key,
                "child_table": child_name,
                "child_column": str(child_col),
                "child_event_time_col": (
                    None if child_event_time_col is None else str(child_event_time_col)
                ),
                "target_lookup_column": str(entity_key),
                "relation_orientation": "incoming_fk",
                "strict_before": False,
                "parent_primary_key_proven": parent_is_pk,
                "child_fk_present": child_fk_present,
                "dtype_compatible": dtype_compatible,
                "referential_coverage": referential_coverage,
                "child_time_column_present": (
                    child_event_time_col is not None
                    and str(child_event_time_col) in child.columns
                ),
            }
            row["verified"] = all((
                row["parent_primary_key_proven"],
                row["child_fk_present"],
                row["dtype_compatible"],
                row["referential_coverage"] >= threshold,
                row["child_time_column_present"],
            ))
            candidates.append(row)
    if not candidates:
        raise ValueError("relation_verification_blocker")
    return candidates


def _select_relation(
    candidates: list[dict[str, Any]],
    *,
    metadata: Mapping[str, Any],
    table_dict: Mapping[str, Any] | None = None,
    train_df: pd.DataFrame | None = None,
    selection_folds: int = 1,
) -> tuple[dict[str, str], dict[str, Any]]:
    verified = [row for row in candidates if row["verified"]]
    explicit = {
        "child_table": metadata.get("child_table"),
        "child_column": metadata.get("child_fk"),
        "child_event_time_col": metadata.get("child_event_time_col"),
        "target_lookup_column": metadata.get("target_lookup_column"),
    }
    has_explicit = all(
        explicit[key]
        for key in (
            "child_table",
            "child_column",
            "child_event_time_col",
        )
    )
    if has_explicit:
        matches = [
            row for row in verified
            if row["child_table"] == explicit["child_table"]
            and row["child_column"] == explicit["child_column"]
            and row["child_event_time_col"] == explicit["child_event_time_col"]
            and (
                explicit["target_lookup_column"] is None
                or row.get("target_lookup_column")
                == explicit["target_lookup_column"]
            )
        ]
        if len(matches) != 1:
            raise ValueError("invalid_explicit_relation")
        return {
            "parent_table": str(matches[0]["parent_table"]),
            "parent_column": str(matches[0]["parent_column"]),
            "child_table": str(matches[0]["child_table"]),
            "child_column": str(matches[0]["child_column"]),
            "child_event_time_col": str(matches[0]["child_event_time_col"]),
            "target_lookup_column": str(
                matches[0].get(
                    "target_lookup_column",
                    metadata["entity_key"],
                )
            ),
        }, {
            "method": "explicit_metadata",
            "reason": "explicit_verified_relation",
            "provenance": {
                "child_table": metadata.get("child_table_source", "explicit_metadata"),
                "child_fk": metadata.get("child_fk_source", "explicit_metadata"),
                "child_event_time_col": metadata.get(
                    "child_event_time_col_source",
                    "explicit_metadata",
                ),
                "relation": metadata.get("relation_source", "explicit_metadata"),
            },
        }
    if len(verified) == 1:
        return {
            "parent_table": str(verified[0]["parent_table"]),
            "parent_column": str(verified[0]["parent_column"]),
            "child_table": str(verified[0]["child_table"]),
            "child_column": str(verified[0]["child_column"]),
            "child_event_time_col": str(verified[0]["child_event_time_col"]),
            "target_lookup_column": str(
                verified[0].get(
                    "target_lookup_column",
                    metadata["entity_key"],
                )
            ),
        }, {
            "method": "single_verified_relation",
            "reason": "only_verified_relation",
            "provenance": {
                "child_table": "relbench_schema:auto_verified",
                "child_fk": "relbench_schema:auto_verified",
                "child_event_time_col": "relbench_schema:auto_verified",
                "relation": "relbench_schema:auto_verified",
            },
        }
    if not verified:
        raise ValueError("relation_verification_blocker")
    if table_dict is None or train_df is None:
        raise ValueError("multiple_verified_relations_require_screening_context")
    selected, screening, fold_rows = _screen_verified_relations(
        verified,
        table_dict=table_dict,
        train_df=train_df,
        metadata=metadata,
        selection_folds=selection_folds,
    )
    return {
        "parent_table": str(selected["parent_table"]),
        "parent_column": str(selected["parent_column"]),
        "child_table": str(selected["child_table"]),
        "child_column": str(selected["child_column"]),
        "child_event_time_col": str(selected["child_event_time_col"]),
        "target_lookup_column": str(
            selected.get(
                "target_lookup_column",
                metadata["entity_key"],
            )
        ),
    }, {
        "method": "train_inner_fold_screening",
        "reason": "best_train_only_canonical_dfs_relation",
        "screening": screening,
        "screening_folds": fold_rows,
        "provenance": {
            "child_table": "train_inner_fold_screening",
            "child_fk": "train_inner_fold_screening",
            "child_event_time_col": "train_inner_fold_screening",
            "relation": "train_inner_fold_screening",
        },
    }


def _relation_for_auto(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "child_table": str(row["child_table"]),
        "child_fk": str(row["child_column"]),
        "parent_table": str(row["parent_table"]),
        "parent_key": str(row["parent_column"]),
        "child_event_time_col": str(row["child_event_time_col"]),
        "target_lookup_column": str(
            row.get(
                "target_lookup_column",
                row["parent_column"],
            )
        ),
        "strict_before": bool(
            row.get("strict_before", False)
        ),
        "allow_exact_matches": not bool(
            row.get("strict_before", False)
        ),
        "status": "accepted",
        "rejection_reasons": "",
        "relation_rank": 1,
    }


def _canonical_relation_features(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    from fdhg.onboarding.auto_relbench import _candidate

    relation = _relation_for_auto(row)
    return [
        _candidate(0, relation, None, "count", "relation"),
        _candidate(1, relation, None, "days_since_last", "relation"),
    ]


def _screen_verified_relations(
    verified: Sequence[Mapping[str, Any]],
    *,
    table_dict: Mapping[str, Any],
    train_df: pd.DataFrame,
    metadata: Mapping[str, Any],
    selection_folds: int,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    from fdhg.onboarding.auto_relbench import (
        AutoOnboardingOptions,
        _score_feature_set,
        make_inner_temporal_splits,
    )

    options = AutoOnboardingOptions(
        selection_folds=selection_folds,
        feature_budget=2,
    )
    split_plan = make_inner_temporal_splits(
        train_df,
        time_col=str(metadata["target_time_col"]),
        requested_folds=selection_folds,
    )
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    scored: list[tuple[Mapping[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for row in sorted(
        verified,
        key=lambda item: (
            str(item.get("child_table", "")),
            str(item.get("child_column", "")),
            str(item.get("child_event_time_col", "")),
        ),
    ):
        features = _canonical_relation_features(
            row
        )

        lookup_col = str(
            row.get(
                "target_lookup_column",
                metadata["entity_key"],
            )
        )

        scoring_targets = train_df
        scoring_metadata = dict(metadata)

        if (
            lookup_col
            != metadata["entity_key"]
        ):
            scoring_targets = (
                _enrich_target_relation_key(
                    train_df,
                    table_dict=table_dict,
                    target_table=str(
                        metadata[
                            "target_table"
                        ]
                    ),
                    row_entity_key=str(
                        metadata[
                            "entity_key"
                        ]
                    ),
                    relation_entity_key=lookup_col,
                    target_time_col=str(
                        metadata[
                            "target_time_col"
                        ]
                    ),
                )
            )

            scoring_metadata[
                "entity_key"
            ] = lookup_col

        scoring_split_plan = (
            make_inner_temporal_splits(
                scoring_targets,
                time_col=str(
                    metadata[
                        "target_time_col"
                    ]
                ),
                requested_folds=selection_folds,
            )
        )

        score = _score_feature_set(
            train_targets=scoring_targets,
            table_dict=table_dict,
            metadata=scoring_metadata,
            features=features,
            split_plan=scoring_split_plan,
            options=options,
        )
        trials = [dict(trial) for trial in score.get("trials", ())]
        for trial in trials:
            fold_rows.append({
                "child_table": row["child_table"],
                "child_column": row["child_column"],
                "child_event_time_col": row["child_event_time_col"],
                "target_lookup_column": row.get(
                    "target_lookup_column",
                    metadata["entity_key"],
                ),
                "strict_before": bool(
                    row.get("strict_before", False)
                ),
                "temporal_predicate": (
                    "<"
                    if bool(row.get("strict_before", False))
                    else "<="
                ),
                "fold": trial["fold"],
                "primary_metric": metadata["primary_metric"],
                "metric_direction": metadata["metric_direction"],
                "score": trial["score"],
                "official_validation_used_for_resolution": False,
                "test_split_accessed": False,
            })
        summary = {
            "child_table": row["child_table"],
            "child_column": row["child_column"],
            "child_event_time_col": row["child_event_time_col"],
            "parent_table": row["parent_table"],
            "parent_column": row["parent_column"],
            "target_lookup_column": row.get(
                "target_lookup_column",
                metadata["entity_key"],
            ),
            "strict_before": bool(
                row.get("strict_before", False)
            ),
            "temporal_predicate": (
                "<"
                if bool(row.get("strict_before", False))
                else "<="
            ),
            "primary_metric": metadata["primary_metric"],
            "metric_direction": metadata["metric_direction"],
            "mean_inner_fold_score": score["score"],
            "inner_fold_score_std": score["stability"],
            "materialized_model_columns": len(features),
            "expected_relation_scans": len(split_plan["folds"]) * 2,
            "official_validation_used_for_resolution": False,
            "test_split_accessed": False,
        }
        summary_rows.append(summary)
        scored.append((row, summary, trials))
    direction = str(metadata["metric_direction"])

    def better_key(item: tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]) -> tuple[Any, ...]:
        row, summary, _ = item
        score = float(summary["mean_inner_fold_score"])
        score_key = -score if direction == "higher" else score
        return (
            _score_bucket(score_key),
            int(summary["materialized_model_columns"]),
            int(summary["expected_relation_scans"]),
            str(row["child_table"]),
            str(row["child_column"]),
            str(row["child_event_time_col"]),
        )

    scored.sort(key=better_key)
    selected = scored[0][0]
    for summary in summary_rows:
        summary["selected"] = (
            summary["child_table"] == selected["child_table"]
            and summary["child_column"] == selected["child_column"]
            and summary["child_event_time_col"] == selected["child_event_time_col"]
        )
    return selected, tuple(summary_rows), tuple(fold_rows)


def _score_bucket(score_key: float) -> int:
    if not math.isfinite(score_key):
        return 10**18
    return int(round(score_key / RELATION_SCREENING_TIE_TOLERANCE))


def _metadata_mapping_from_resolved(resolved: ResolvedTaskMetadata) -> dict[str, Any]:
    values = {
        "entity_key": resolved.entity_key,
        "target_time_col": resolved.target_time_col,
        "label_col": resolved.label_col,
        "problem_type": resolved.problem_type,
        "primary_metric": resolved.primary_metric,
        "metric_direction": resolved.metric_direction,
        "target_table": resolved.entity_table,
        "child_table": resolved.child_table,
        "child_fk": resolved.child_fk,
        "child_event_time_col": resolved.child_event_time_col,
        "relation_entity_key": resolved.relation_entity_key,
        "relation_source": resolved.provenance.get("relation", ""),
        "relation_selection_method": resolved.relation_selection_method,
        "relation_selection_reason": resolved.relation_selection_reason,
        "parent_column": next(
            (
                str(row.get("parent_column"))
                for row in resolved.candidate_relations_considered
                if row.get("child_table") == resolved.child_table
                and row.get("child_column") == resolved.child_fk
                and row.get("child_event_time_col")
                == resolved.child_event_time_col
            ),
            resolved.entity_key,
        ),
    }
    for field, source in resolved.provenance.items():
        values[f"{field}_source"] = source
    return values


def _resolve_primary_metric(
    *,
    task: Any,
    problem_type: str,
    explicit_metadata: Mapping[str, Any],
) -> tuple[str, str]:
    if explicit_metadata.get("primary_metric") is not None:
        return (
            _normalize_metric_name(explicit_metadata["primary_metric"]),
            "explicit_metadata:primary_metric",
        )
    discovered = _discover_task_metrics(task)
    if discovered:
        supported = [_normalize_metric_name(metric) for metric in discovered]
        for metric in METRIC_PRIORITY_BY_PROBLEM_TYPE.get(problem_type, ()):
            if metric in supported:
                return metric, "official_metric_declaration"
        if len(supported) == 1 and _metric_direction_from_name(supported[0]):
            return supported[0], "official_metric_declaration"
        raise ValueError(
            "unknown_primary_metric:"
            f"task_type={problem_type}:"
            f"discovered_evaluator_metrics={','.join(supported)}:"
            f"inspected_task_attributes={','.join(_metric_attribute_names(task))}"
        )
    priority = METRIC_PRIORITY_BY_PROBLEM_TYPE.get(problem_type, ())
    if not priority:
        raise ValueError(
            "unknown_primary_metric:"
            f"task_type={problem_type}:"
            "discovered_evaluator_metrics=:"
            f"inspected_task_attributes={','.join(_metric_attribute_names(task))}"
        )
    return priority[0], "task_type_metric_policy"


def _resolve_metric_direction(
    primary_metric: str,
    *,
    explicit_metadata: Mapping[str, Any],
) -> tuple[str, str]:
    if explicit_metadata.get("metric_direction") is not None:
        direction = str(explicit_metadata["metric_direction"]).strip().lower()
        if direction not in {"higher", "lower"}:
            raise ValueError(f"unknown_metric_direction:{direction}")
        return direction, "explicit_metadata:metric_direction"
    direction = _metric_direction_from_name(primary_metric)
    if direction is None:
        raise ValueError(f"missing_task_metadata:metric_direction:{primary_metric}")
    return direction, "metric_direction_policy"


def _metric_direction_from_name(metric: str) -> str | None:
    normalized = _normalize_metric_name(metric)
    if normalized in HIGHER_IS_BETTER_METRICS:
        return "higher"
    if normalized in LOWER_IS_BETTER_METRICS:
        return "lower"
    return None


def _normalize_metric_name(value: Any) -> str:
    raw = value.value if hasattr(value, "value") else value
    if callable(raw) and getattr(raw, "__name__", None):
        raw = raw.__name__

    normalized = str(raw).strip().lower()
    aliases = {
        "roc_auc_score": "roc_auc",
        "accuracy_score": "accuracy",
    }
    return aliases.get(normalized, normalized)


def _discover_task_metrics(task: Any) -> tuple[str, ...]:
    metrics: list[str] = []
    for attr in (
        "primary_metric",
        "metric",
        "metrics",
        "eval_metrics",
        "evaluator_metrics",
    ):
        value = getattr(task, attr, None)
        metrics.extend(_flatten_metric_value(value))
    evaluator = getattr(task, "evaluator", None) or getattr(task, "eval", None)
    if callable(evaluator):
        try:
            evaluator = evaluator()
        except TypeError:
            evaluator = None
    if evaluator is not None:
        for attr in ("primary_metric", "metric", "metrics", "eval_metrics"):
            metrics.extend(_flatten_metric_value(getattr(evaluator, attr, None)))
    return tuple(dict.fromkeys(_normalize_metric_name(metric) for metric in metrics))


def _flatten_metric_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or hasattr(value, "value"):
        return [_normalize_metric_name(value)]
    if isinstance(value, Mapping):
        return [_normalize_metric_name(key) for key in value]
    if isinstance(value, Sequence):
        return [_normalize_metric_name(item) for item in value]
    return []


def _metric_attribute_names(task: Any) -> tuple[str, ...]:
    names = []
    for attr in (
        "primary_metric",
        "metric",
        "metrics",
        "eval_metrics",
        "evaluator_metrics",
        "evaluator",
        "eval",
    ):
        if getattr(task, attr, None) is not None:
            names.append(attr)
    return tuple(names)


def _validate_targets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    entity_key: str,
    target_time_col: str,
    label_col: str,
    test_timestamp: Any,
) -> None:
    for col, code in (
        (entity_key, "missing_entity_key"),
        (target_time_col, "missing_target_timestamp"),
        (label_col, "missing_label"),
    ):
        if col not in train.columns or col not in validation.columns:
            raise ValueError(code)
    if _row_fingerprints(train).intersection(_row_fingerprints(validation)):
        raise ValueError("train_validation_overlap")
    for frame in (train, validation):
        if frame.duplicated(subset=[entity_key, target_time_col]).any():
            raise ValueError("duplicate_target_identity")
    if test_timestamp is not None:
        val_times = pd.to_datetime(validation[target_time_col], errors="coerce")
        if val_times.isna().any() or (val_times >= pd.Timestamp(test_timestamp)).any():
            raise ValueError("test_timestamp_boundary_violation")


def _task_metadata_from_config(
    path: Path | None,
    *,
    dataset_name: str,
    task_name: str,
) -> dict[str, Any]:
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("task_metadata_config must be a mapping")
    tasks = raw.get("tasks", {})
    if not isinstance(tasks, Mapping):
        raise ValueError("task_metadata_config tasks must be a mapping")
    key = f"{dataset_name}/{task_name}"
    legacy_key = f"relbench-v1-{dataset_name}/{task_name}"
    if legacy_key in tasks and key not in tasks:
        raise ValueError(
            "invalid_task_metadata_key:"
            f"{legacy_key}:expected:{key}"
        )
    row = tasks.get(key, {})
    if not isinstance(row, Mapping):
        raise ValueError("task metadata row must be a mapping")
    return dict(row)



def _normalize_metadata_value(field: str, value: Any) -> str:
    if value is None:
        return ""

    if field == "problem_type":
        raw = value.value if hasattr(value, "value") else value
        normalized = str(raw).strip().lower()
        return {
            "binary": "binary_classification",
            "multiclass": "multiclass_classification",
            "regression": "regression",
        }.get(normalized, normalized)

    if field in {"primary_metric", "metric_direction"}:
        return str(value).strip().lower()

    return str(value).strip()


def _resolve_metadata_field(
    task: Any,
    *,
    field: str,
    attrs: tuple[str, ...],
    config: Mapping[str, Any],
    config_path: Path | None,
    dataset_name: str,
    task_name: str,
) -> tuple[Any, str]:
    if field in config and config[field] is not None:
        return (
            config[field],
            f"explicit_metadata:tasks.{dataset_name}/{task_name}.{field}",
        )
    for attr in attrs:
        value = getattr(task, attr, None)
        if value is not None:
            return value, f"task_attr:{attr}"
    for attr in attrs:
        value = getattr(task.__class__, attr, None)
        if value is not None:
            return value, f"task_class_attr:{attr}"
    return None, "missing"


def _onboarding_config_text(
    *,
    dataset_name: str,
    task_name: str,
    output_dir: Path,
    config_output: Path,
    table_dict: Mapping[str, Any],
    relation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    dataset_id = f"relbench-v1-{dataset_name}"
    tables: dict[str, Any] = {}
    for table_name, table in sorted(table_dict.items()):
        row: dict[str, Any] = {
            "path": _relative_or_absolute(
                output_dir / "tables" / f"{table_name}.parquet",
                config_output.parent,
            ),
        }
        if getattr(table, "pkey_col", None):
            row["primary_key"] = str(getattr(table, "pkey_col"))
        if table_name == relation["child_table"]:
            row["event_time_col"] = relation["child_event_time_col"]
            row["foreign_keys"] = [{
                "column": relation["child_column"],
                "references": {
                    "table": relation["parent_table"],
                    "column": relation["parent_column"],
                },
            }]
        tables[table_name] = row
    config = {
        "dataset": dataset_id,
        "tables": tables,
        "task": {
            "task_id": task_name,

            # Prediction-row table identity is independent of the
            # parent table used by the selected relation.
            "target_table": metadata.get(
                "target_table",
                metadata.get(
                    "entity_table",
                    relation["parent_table"],
                ),
            ),

            # Prediction-row identity.
            "entity_key": metadata["entity_key"],

            # Target-side key used only for relational lookup.
            "relation_entity_key": metadata.get(
                "relation_entity_key",
                metadata["entity_key"],
            ),

            # Explicit selected relation.  Pipeline onboarding should
            # consume this rather than re-inferring an incoming-only
            # relation shape.
            "child_table": relation["child_table"],
            "child_fk": relation["child_column"],
            "child_event_time_col":
                relation["child_event_time_col"],

            "relation_parent_table":
                relation["parent_table"],
            "relation_parent_column":
                relation["parent_column"],

            "relation_orientation": relation.get(
                "relation_orientation",
                "incoming_fk",
            ),

            "strict_before": bool(
                relation.get(
                    "strict_before",
                    False,
                )
            ),

            "target_time_col": metadata["target_time_col"],
            "label_col": metadata["label_col"],
            "problem_type": metadata["problem_type"],
            "primary_metric": metadata["primary_metric"],
            "metric_direction": metadata["metric_direction"],
        },
        "split": {
            "strategy": "official_pre_split",
            "source": "relbench",
            "train_split_name": "train",
            "validation_split_name": "val",
            "train_target_path": _relative_or_absolute(
                output_dir / "target_train.parquet",
                config_output.parent,
            ),
            "validation_target_path": _relative_or_absolute(
                output_dir / "target_validation.parquet",
                config_output.parent,
            ),
            "test_split_accessed": False,
        },
    }
    return yaml.safe_dump(config, sort_keys=True)


def _manifest(
    *,
    dataset_name: str,
    task_name: str,
    dataset: Any,
    task: Any,
    database: Any,
    table_dict: Mapping[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    relation: Mapping[str, str],
    relation_candidates: list[Mapping[str, Any]],
    metadata: Mapping[str, str],
    relbench_version: str,
    config_text: str,
) -> dict[str, Any]:
    table_meta = {}
    for table_name, table in sorted(table_dict.items()):
        frame = _table_df(table)
        table_meta[table_name] = {
            "row_count": int(len(frame)),
            "columns": [str(col) for col in frame.columns],
            "dtypes": {str(col): str(frame[col].dtype) for col in frame.columns},
            "primary_key": getattr(table, "pkey_col", None),
            "foreign_keys": dict(getattr(table, "fkey_col_to_pkey_table", {}) or {}),
            "time_column": getattr(table, "time_col", None),
        }
    payload = {
        "exporter_version": REL_BENCH_EXPORT_VERSION,
        "relbench_version": relbench_version,
        "loader_api_path": LOADER_API_PATH,
        "dataset": dataset_name,
        "task": task_name,
        "dataset_class": _class_name(dataset),
        "task_class": _class_name(task),
        "database_class": _class_name(database),
        "database_table_names": sorted(table_dict),
        "database_table_metadata": table_meta,
        "database_table_content_hashes": {
            table_name: _dataframe_identity_hash(_table_df(table))
            for table_name, table in sorted(table_dict.items())
        },
        "train_target_content_hash": _dataframe_identity_hash(train_df),
        "validation_target_content_hash": _dataframe_identity_hash(validation_df),
        "train_row_count": int(len(train_df)),
        "validation_row_count": int(len(validation_df)),
        "entity_key": metadata["entity_key"],
        "target_timestamp_column": metadata["target_time_col"],
        "label_column": metadata["label_col"],
        "task_type": metadata["problem_type"],
        "metric": metadata["primary_metric"],
        "metric_direction": metadata["metric_direction"],
        "entity_key_source": metadata["entity_key_source"],
        "target_time_col_source": metadata["target_time_col_source"],
        "label_col_source": metadata["label_col_source"],
        "problem_type_source": metadata["problem_type_source"],
        "primary_metric_source": metadata["primary_metric_source"],
        "metric_direction_source": metadata["metric_direction_source"],
        "relation_source": metadata["relation_source"],
        "task_metadata_resolution_status": "completed",
        "task_metadata_source_by_field": {
            key.removesuffix("_source"): value
            for key, value in metadata.items()
            if key.endswith("_source")
        },
        "relation_candidate_count": len(relation_candidates),
        "relation_selection_method": metadata.get(
            "relation_selection_method",
            "",
        ),
        "selected_relation": dict(relation),
        "official_validation_used_for_resolution": False,
        "test_split_accessed_during_resolution": False,
        "val_timestamp": str(getattr(dataset, "val_timestamp", "")),
        "test_timestamp": str(getattr(dataset, "test_timestamp", "")),
        "train_split_name": "train",
        "validation_split_name": "val",
        "test_split_accessed": False,
        "verified_relation": dict(relation),
        "relation_candidates": list(relation_candidates),
        "generated_onboarding_config_hash": _text_sha256(config_text),
    }
    payload["source_identity_hash"] = _text_sha256(
        json.dumps(payload, sort_keys=True, default=str)
    )
    return payload


def _write_export(staging: Path, prepared: Mapping[str, Any]) -> None:
    tables_dir = staging / "tables"
    tables_dir.mkdir(parents=True)
    exported_hashes = {}
    for table_name, table in sorted(prepared["table_dict"].items()):
        path = tables_dir / f"{table_name}.parquet"
        _table_df(table).to_parquet(path, index=False)
        exported_hashes[f"tables/{table_name}.parquet"] = _file_sha256(path)
    prepared["train_df"].to_parquet(staging / "target_train.parquet", index=False)
    prepared["validation_df"].to_parquet(
        staging / "target_validation.parquet",
        index=False,
    )
    manifest = dict(prepared["manifest"])
    manifest["status"] = "completed"
    exported_hashes["target_train.parquet"] = _file_sha256(
        staging / "target_train.parquet"
    )
    exported_hashes["target_validation.parquet"] = _file_sha256(
        staging / "target_validation.parquet"
    )
    manifest["exported_file_hashes"] = exported_hashes
    (staging / "export_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _validate_export(staging: Path) -> None:
    required = (
        "target_train.parquet",
        "target_validation.parquet",
        "export_manifest.json",
    )
    missing = [name for name in required if not (staging / name).exists()]
    if missing:
        raise ValueError("partial_staging_output")


def _write_config(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_text(encoding="utf-8") == text:
            return
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _report(
    status: str,
    prepared: Mapping[str, Any],
    *,
    dry_run: bool,
    reused: bool = False,
    blockers: tuple[str, ...] = (),
) -> RelBenchV1ExportReport:
    relation = prepared["relation"]
    metadata = prepared["metadata"]
    return RelBenchV1ExportReport(
        dataset=prepared["dataset_name"],
        task=prepared["task_name"],
        status=status,
        output_dir=prepared["output_dir"],
        config_path=prepared["config_output"],
        blockers=blockers,
        dry_run=dry_run,
        reused=reused,
        relation_count=1,
        table_count=len(prepared["table_dict"]),
        train_rows=len(prepared["train_df"]),
        validation_rows=len(prepared["validation_df"]),
        relbench_version=prepared["relbench_version"],
        dataset_class=_class_name(prepared["dataset"]),
        task_class=_class_name(prepared["task"]),
        table_names=tuple(sorted(prepared["table_dict"])),
        entity_key=metadata["entity_key"],
        target_time_col=metadata["target_time_col"],
        label_col=metadata["label_col"],
        child_relation=(
            f"{relation['child_table']}.{relation['child_column']}->"
            f"{relation['parent_table']}.{relation['parent_column']}"
        ),
        child_event_time_col=relation["child_event_time_col"],
    )


def _blocked_report(
    dataset: str,
    task: str,
    output_root: Path,
    config_output: Path,
    *,
    blockers: tuple[str, ...],
    dry_run: bool,
    prepared: Mapping[str, Any] | None = None,
) -> RelBenchV1ExportReport:
    if prepared is not None:
        return _report(
            "blocked",
            prepared,
            dry_run=dry_run,
            blockers=blockers,
        )
    return RelBenchV1ExportReport(
        dataset=dataset,
        task=task,
        status="blocked",
        output_dir=output_root / dataset / task,
        config_path=config_output,
        blockers=blockers,
        dry_run=dry_run,
        reused=False,
        relation_count=0,
        table_count=0,
        train_rows=0,
        validation_rows=0,
        relbench_version="",
        dataset_class="",
        task_class="",
        table_names=(),
        entity_key=None,
        target_time_col=None,
        label_col=None,
        child_relation=None,
        child_event_time_col=None,
    )


def _table_df(table: Any) -> pd.DataFrame:
    frame = getattr(table, "df", None)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("missing_database_tables")
    return frame


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _class_name(obj: Any) -> str:
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"


def _row_fingerprints(frame: pd.DataFrame) -> set[str]:
    rows = frame.astype("string").fillna("<NA>").to_dict("records")
    return {
        hashlib.sha256(
            json.dumps(row, sort_keys=True, default=str).encode()
        ).hexdigest()
        for row in rows
    }


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dataframe_identity_hash(frame: pd.DataFrame) -> str:
    payload = {
        "columns": [str(col) for col in frame.columns],
        "dtypes": {str(col): str(frame[col].dtype) for col in frame.columns},
        "rows": frame.astype("string").fillna("<NA>").to_dict("records"),
    }
    return _text_sha256(json.dumps(payload, sort_keys=True, default=str))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_fingerprint(table_dict: Mapping[str, Any]) -> str:
    payload = {}
    for table_name, table in sorted(table_dict.items()):
        frame = _table_df(table)
        payload[str(table_name)] = {
            "columns": [str(col) for col in frame.columns],
            "dtypes": {str(col): str(frame[col].dtype) for col in frame.columns},
            "primary_key": getattr(table, "pkey_col", None),
            "foreign_keys": dict(getattr(table, "fkey_col_to_pkey_table", {}) or {}),
            "time_column": getattr(table, "time_col", None),
        }
    return _text_sha256(json.dumps(payload, sort_keys=True, default=str))


def _candidate_relation_fingerprint(
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    rows = [
        {
            "parent_table": row.get("parent_table"),
            "parent_column": row.get("parent_column"),
            "child_table": row.get("child_table"),
            "child_column": row.get("child_column"),
            "child_event_time_col": row.get("child_event_time_col"),
            "verified": bool(row.get("verified")),
        }
        for row in candidates
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("parent_table", "")),
            str(row.get("parent_column", "")),
            str(row.get("child_table", "")),
            str(row.get("child_column", "")),
            str(row.get("child_event_time_col", "")),
        )
    )
    return _text_sha256(json.dumps(rows, sort_keys=True, default=str))


def _write_resolved_metadata_outputs(
    output_dir: Path,
    resolved: ResolvedTaskMetadata,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = resolved.to_dict()
    (output_dir / "resolved_task_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_rows_csv(output_dir / "relation_screening.csv", resolved.relation_screening)
    _write_rows_csv(
        output_dir / "relation_screening_folds.csv",
        resolved.relation_screening_folds,
    )


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "not_required"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
