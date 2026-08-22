from __future__ import annotations

import csv
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fdhg.compiler.ambiguity import (
    edge_from_mapping,
    fit_ambiguity_map,
    normalize_lhs_frame,
    normalize_series,
)
from fdhg.compiler.edge_reliability import compute_edge_reliability
from fdhg.compiler.fold_safe_fdhg import materialize_ambiguity_features
from fdhg.onboarding.auto_fdhg import (
    AutoFdhgOptions,
    _write_json,
    align_feature_blocks,
    audit_residual_columns,
    feature_columns,
    materialize_declared_feature_frame_pair,
    prepare_auto_fdhg,
    score_matrix,
)
from fdhg.onboarding.motivation_reliability_utility import (
    _duplicate_feature_count,
    _FilteredTable,
    _mean,
    _ordered_row,
    _std,
    fit_transform_static_single_edge_fold,
    fold_train_source_view_for_edge,
    motivation_candidate_relations,
)

MOTIVATION_FD_VIOLATION_VERSION = "motivation-fd-violation-v1"
DEFAULT_CORRUPTION_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_CORRUPTION_SEEDS = (41, 42, 43, 44)

FOLD_LEVEL_COLUMNS = (
    "dataset",
    "task",
    "edge_id",
    "source_table",
    "determinant",
    "dependent",
    "fold",
    "corruption_seed",
    "requested_corruption_rate",
    "effective_changed_row_rate",
    "source_row_count",
    "eligible_row_count",
    "requested_sampled_row_count",
    "actual_changed_row_count",
    "induced_fd_violation_row_count",
    "effective_changed_rate_among_eligible",
    "effective_changed_rate_among_all_source_rows",
    "achieved_violation_rate",
    "maximum_feasible_changed_row_count",
    "maximum_feasible_marginal_preserving_changed_row_rate",
    "corruption_saturated",
    "dependent_min_class_frequency",
    "dependent_max_class_frequency",
    "dependent_minority_fraction",
    "determinant_group_support",
    "baseline_violation_rate",
    "determinant_cardinality",
    "dependent_cardinality",
    "reliability_raw",
    "reliability_non_singleton",
    "reliability_loo",
    "reliability_entropy",
    "conditional_entropy_normalized",
    "violation_rate",
    "non_singleton_coverage",
    "residual_feature_count",
    "residual_nonzero_rate",
    "residual_variance",
    "residual_unique_value_count",
    "constant_generated_feature_count",
    "duplicate_generated_feature_count",
    "base_score",
    "corrupted_edge_score",
    "delta_over_base",
    "uncorrupted_edge_score",
    "delta_relative_to_uncorrupted",
    "metric",
    "metric_direction",
    "edge_status",
    "failure_reason",
    "reliability_fit_scope",
    "reliability_fit_horizon",
    "future_row_violation_count",
    "inner_validation_row_usage_count",
    "official_validation_row_usage_count",
    "test_row_usage_count",
)

AGGREGATE_COLUMNS = (
    "dataset",
    "task",
    "edge_id",
    "source_table",
    "determinant",
    "dependent",
    "requested_corruption_rate",
    "mean_effective_changed_row_rate",
    "std_effective_changed_row_rate",
    "mean_effective_changed_rate_among_eligible",
    "mean_achieved_violation_rate",
    "mean_induced_fd_violation_row_count",
    "mean_reliability_loo",
    "std_reliability_loo",
    "mean_reliability_raw",
    "mean_reliability_entropy",
    "mean_conditional_entropy_normalized",
    "mean_violation_rate",
    "mean_residual_nonzero_rate",
    "mean_residual_variance",
    "mean_delta_over_base",
    "std_delta_over_base",
    "mean_delta_relative_to_uncorrupted",
    "std_delta_relative_to_uncorrupted",
    "valid_fold_count",
    "corruption_seed_count",
)

FIGURE_AGGREGATE_COLUMNS = (
    "dataset",
    "task",
    "edge_id",
    "source_table",
    "determinant",
    "dependent",
    "requested_corruption_rate",
    "mean_effective_changed_row_rate",
    "mean_effective_changed_rate_among_eligible",
    "mean_achieved_violation_rate",
    "mean_reliability_loo",
    "sem_reliability_loo",
    "mean_delta_relative_to_uncorrupted",
    "sem_delta_relative_to_uncorrupted",
    "mean_residual_nonzero_rate",
    "sem_residual_nonzero_rate",
    "mean_residual_variance",
    "sem_residual_variance",
    "valid_fold_count",
)


@dataclass(frozen=True)
class FdViolationOptions:
    selection_folds: int = 3
    feature_budget: int = 8
    min_delta: float = 0.0
    selection_decoder: str = "hist_gradient_boosting"
    max_relations: int = 3
    max_numeric_columns: int = 4
    max_categorical_columns: int = 4
    corruption_levels: tuple[float, ...] = DEFAULT_CORRUPTION_LEVELS
    corruption_seeds: tuple[int, ...] = DEFAULT_CORRUPTION_SEEDS


@dataclass(frozen=True)
class FdViolationReport:
    dataset: str
    task: str
    status: str
    output_dir: Path
    blockers: tuple[str, ...]
    dry_run: bool
    fold_csv: Path | None = None
    aggregate_csv: Path | None = None
    figure_aggregate_csv: Path | None = None
    fold_rows: int = 0
    aggregate_rows: int = 0
    test_split_accessed: bool = False


def motivation_fd_violation(
    *,
    dataset_name: str,
    task_name: str,
    edge_spec: str,
    output_root: Path,
    write: bool = False,
    overwrite: bool = False,
    download: bool = False,
    auto_output_root: Path = Path("outputs/auto-onboarding-3fold"),
    dfs_source_root: Path = Path("."),
    dfs_feature_config: Path | None = None,
    options: FdViolationOptions | None = None,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
) -> FdViolationReport:
    options = options or FdViolationOptions()
    output_dir = output_root / dataset_name / task_name / _edge_path_token(edge_spec)
    try:
        fold_rows, aggregate_rows, figure_rows, blockers, manifest = prepare_fd_violation_experiment(
            dataset_name=dataset_name,
            task_name=task_name,
            edge_spec=edge_spec,
            output_root=output_root,
            download=download,
            auto_output_root=auto_output_root,
            dfs_source_root=dfs_source_root,
            dfs_feature_config=dfs_feature_config,
            options=options,
            object_loader=object_loader,
        )
    except AssertionError as exc:
        return FdViolationReport(
            dataset=dataset_name,
            task=task_name,
            status="failed",
            output_dir=output_dir,
            blockers=(str(exc),),
            dry_run=not write,
        )
    except Exception as exc:  # noqa: BLE001 - setup failures are reported as blockers.
        return FdViolationReport(
            dataset=dataset_name,
            task=task_name,
            status="blocked",
            output_dir=output_dir,
            blockers=(str(exc),),
            dry_run=not write,
        )
    if blockers:
        status = "no_candidates" if blockers == ["requested_edge_not_resolved"] else "blocked"
        if write:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output_dir / "manifest.json", {**manifest, "status": status})
        return FdViolationReport(
            dataset=dataset_name,
            task=task_name,
            status=status,
            output_dir=output_dir,
            blockers=tuple(blockers),
            dry_run=not write,
        )
    valid_rows = [row for row in fold_rows if row.get("edge_status") == "ok"]
    status = "completed" if valid_rows else "no_candidates"
    if write:
        if output_dir.exists() and not overwrite:
            raise FileExistsError(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_csv = output_dir / "fold_level.csv"
        aggregate_csv = output_dir / "edge_corruption_aggregate.csv"
        figure_csv = output_dir / "figure_aggregate.csv"
        _write_ordered_csv(fold_csv, fold_rows, FOLD_LEVEL_COLUMNS)
        _write_ordered_csv(aggregate_csv, aggregate_rows, AGGREGATE_COLUMNS)
        _write_ordered_csv(figure_csv, figure_rows, FIGURE_AGGREGATE_COLUMNS)
        if figure_rows:
            from fdhg.analysis.plot_fd_violation import plot_fd_violation

            plot_fd_violation(figure_csv, output_dir / "figures")
        _write_json(output_dir / "manifest.json", {**manifest, "status": status})
    else:
        fold_csv = aggregate_csv = figure_csv = None
    return FdViolationReport(
        dataset=dataset_name,
        task=task_name,
        status="dry_run_ready" if not write and status == "completed" else status,
        output_dir=output_dir,
        blockers=(),
        dry_run=not write,
        fold_csv=fold_csv,
        aggregate_csv=aggregate_csv,
        figure_aggregate_csv=figure_csv,
        fold_rows=len(fold_rows),
        aggregate_rows=len(aggregate_rows),
        test_split_accessed=False,
    )


def prepare_fd_violation_experiment(
    *,
    dataset_name: str,
    task_name: str,
    edge_spec: str,
    output_root: Path,
    download: bool,
    auto_output_root: Path,
    dfs_source_root: Path,
    dfs_feature_config: Path | None,
    options: FdViolationOptions,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    auto_options = AutoFdhgOptions(
        selection_folds=options.selection_folds,
        feature_budget=options.feature_budget,
        min_delta=options.min_delta,
        selection_decoder=options.selection_decoder,
        max_relations=options.max_relations,
        max_numeric_columns=options.max_numeric_columns,
        max_categorical_columns=options.max_categorical_columns,
        random_seed=0,
        enable_edge_screening=False,
        discover_fdhg_edges=False,
    )
    prepared = prepare_auto_fdhg(
        dataset_name=dataset_name,
        task_name=task_name,
        output_root=output_root,
        download=download,
        auto_output_root=auto_output_root,
        dfs_source_root=dfs_source_root,
        dfs_feature_config=dfs_feature_config,
        options=auto_options,
        object_loader=object_loader,
        include_gate=False,
    )
    blockers = [str(item) for item in prepared.get("blockers", ())]
    edge = resolve_requested_edge(edge_spec, prepared)
    if edge is None:
        blockers.append("requested_edge_not_resolved")
    manifest = {
        "implementation_version": MOTIVATION_FD_VIOLATION_VERSION,
        "dataset": dataset_name,
        "task": task_name,
        "edge_spec": edge_spec,
        "selection_folds": options.selection_folds,
        "corruption_levels": [float(q) for q in options.corruption_levels],
        "corruption_seeds": [int(seed) for seed in options.corruption_seeds],
        "blockers": blockers,
        "test_split_accessed": False,
    }
    if blockers or edge is None:
        return [], [], [], blockers, manifest
    prepared = {
        **prepared,
        "manifest": {**prepared.get("manifest", {}), "dataset": dataset_name, "task": task_name},
        "motivation_relations": motivation_candidate_relations(prepared),
    }
    fold_rows = evaluate_fd_violation(prepared=prepared, edge=edge, options=options)
    aggregate_rows = aggregate_fd_violation_rows(fold_rows)
    figure_rows = figure_fd_violation_rows(fold_rows)
    manifest.update({
        "fold_observation_count": len(fold_rows),
        "aggregate_observation_count": len(aggregate_rows),
        "figure_observation_count": len(figure_rows),
    })
    return fold_rows, aggregate_rows, figure_rows, [], manifest


def resolve_requested_edge(edge_spec: str, prepared: Mapping[str, Any]) -> dict[str, Any] | None:
    table_name, lhs_columns, rhs_column = parse_edge_spec(edge_spec)
    table = prepared.get("table_dict", {}).get(table_name)
    if table is None:
        return None
    df = getattr(table, "df", None)
    if df is None:
        from fdhg.onboarding.relbench_v1 import _table_df

        df = _table_df(table)
    missing = [col for col in [*lhs_columns, rhs_column] if col not in df.columns]
    if missing:
        return None
    relation = _relation_for_table(prepared, table_name)
    edge = {
        "edge_id": f"{table_name}:{'|'.join(lhs_columns)}->{rhs_column}",
        "source_table": table_name,
        "lhs_columns": tuple(lhs_columns),
        "rhs_column": rhs_column,
        "edge_rank": 1,
        "selection_status": "accepted",
        "rejection_reason": "",
    }
    if relation:
        edge["source_entity_column"] = relation.get("child_fk", "")
        edge["source_entity_column_resolution"] = "accepted_relation_child_fk"
        edge["source_relation_id"] = (
            f"{relation.get('child_table', '')}:{relation.get('child_fk', '')}"
            f"->{relation.get('parent_table', '')}:{relation.get('parent_key', '')}"
        )
    return edge


def parse_edge_spec(edge_spec: str) -> tuple[str, tuple[str, ...], str]:
    if ":" not in edge_spec or "->" not in edge_spec:
        raise ValueError("edge_spec_must_look_like_table:X->A")
    table, rest = edge_spec.split(":", 1)
    lhs, rhs = rest.split("->", 1)
    lhs_columns = tuple(part.strip() for part in lhs.split("|") if part.strip())
    rhs_column = rhs.strip()
    if not table.strip() or not lhs_columns or not rhs_column:
        raise ValueError("edge_spec_must_look_like_table:X->A")
    return table.strip(), lhs_columns, rhs_column


def evaluate_fd_violation(
    *,
    prepared: Mapping[str, Any],
    edge: Mapping[str, Any],
    options: FdViolationOptions,
) -> list[dict[str, Any]]:
    metadata = prepared["metadata"]
    table_dict = prepared["table_dict"]
    split_plan = prepared["split_plan"]
    join_keys = prepared["join_keys"]
    auto_features = prepared["auto_features"]
    rows: list[dict[str, Any]] = []
    contexts: dict[int, dict[str, Any]] = {}
    auto_feature_cols_by_fold: dict[int, list[str]] = {}
    for fold in split_plan["folds"]:
        fold_id = int(fold["fold"])
        train_targets = prepared["train_df"].loc[fold["train_indices"]].reset_index(drop=True)
        validation_targets = prepared["train_df"].loc[fold["validation_indices"]].reset_index(drop=True)
        auto_train, auto_val = materialize_declared_feature_frame_pair(
            train_targets,
            validation_targets,
            table_dict=table_dict,
            features=auto_features,
            entity_key=metadata["entity_key"],
            target_time_col=metadata["target_time_col"],
        )
        auto_cols = feature_columns(auto_train, join_keys, metadata)
        base_score = score_matrix(
            train_x=auto_train,
            val_x=auto_val,
            train_y=train_targets[metadata["label_col"]],
            val_y=validation_targets[metadata["label_col"]],
            feature_cols=auto_cols,
            metadata=metadata,
            options=_auto_options(options, random_seed=0),
        )
        auto_feature_cols_by_fold[fold_id] = auto_cols
        contexts[fold_id] = {
            "inner_train": train_targets,
            "inner_val": validation_targets,
            "auto_train": auto_train,
            "auto_val": auto_val,
            "base_score": base_score,
        }
    uncorrupted_delta_by_fold_seed: dict[tuple[int, int], float] = {}
    uncorrupted_score_by_fold_seed: dict[tuple[int, int], float] = {}
    levels = sorted(float(q) for q in options.corruption_levels)
    for seed in options.corruption_seeds:
        for q in levels:
            for fold in split_plan["folds"]:
                fold_id = int(fold["fold"])
                context = contexts[fold_id]
                row = _base_violation_row(
                    prepared=prepared,
                    edge=edge,
                    fold=fold_id,
                    corruption_seed=int(seed),
                    requested_corruption_rate=q,
                    base_score=float(context["base_score"]),
                )
                try:
                    evaluated = evaluate_corrupted_fold(
                        prepared=prepared,
                        edge=edge,
                        context=context,
                        fold_id=fold_id,
                        corruption_rate=q,
                        corruption_seed=int(seed),
                        auto_feature_cols=auto_feature_cols_by_fold[fold_id],
                        options=options,
                    )
                    row.update(evaluated)
                    if q == 0.0 and row["edge_status"] == "ok":
                        uncorrupted_delta_by_fold_seed[(fold_id, int(seed))] = float(row["delta_over_base"])
                        uncorrupted_score_by_fold_seed[(fold_id, int(seed))] = float(row["corrupted_edge_score"])
                    uncorrupted_delta = uncorrupted_delta_by_fold_seed.get((fold_id, int(seed)), math.nan)
                    uncorrupted_score = uncorrupted_score_by_fold_seed.get((fold_id, int(seed)), math.nan)
                    row["uncorrupted_edge_score"] = uncorrupted_score
                    row["delta_relative_to_uncorrupted"] = (
                        float(row["delta_over_base"]) - uncorrupted_delta
                        if np.isfinite(float(row.get("delta_over_base", math.nan))) and np.isfinite(uncorrupted_delta)
                        else math.nan
                    )
                except Exception as exc:  # noqa: BLE001 - per-row failures are data.
                    row.update({"edge_status": "failed", "failure_reason": str(exc)})
                rows.append(_ordered_row(row, FOLD_LEVEL_COLUMNS))
    _assert_leakage_counters_zero(rows)
    return rows


def evaluate_corrupted_fold(
    *,
    prepared: Mapping[str, Any],
    edge: Mapping[str, Any],
    context: Mapping[str, Any],
    fold_id: int,
    corruption_rate: float,
    corruption_seed: int,
    auto_feature_cols: Sequence[str],
    options: FdViolationOptions,
) -> dict[str, Any]:
    metadata = prepared["metadata"]
    table_dict = prepared["table_dict"]
    fit_view = fold_train_source_view_for_edge(
        table_dict=table_dict,
        metadata=metadata,
        train_targets=context["inner_train"],
        validation_targets=context["inner_val"],
        edge=edge,
    )
    row: dict[str, Any] = dict(fit_view["audit"])
    row["inner_validation_row_usage_count"] = 0
    if fit_view["blocked_reason"]:
        row.update({"edge_status": "blocked", "failure_reason": fit_view["blocked_reason"]})
        return row
    fit_rows = fit_view["fit_rows"]
    row.update(edge_suitability_audit(
        fit_rows,
        lhs_columns=edge["lhs_columns"],
        rhs_column=str(edge["rhs_column"]),
    ))
    corrupted_fit_rows, corruption_audit = corrupt_dependent_by_fd_aware_swaps(
        fit_rows,
        lhs_columns=edge["lhs_columns"],
        rhs_column=str(edge["rhs_column"]),
        rate=corruption_rate,
        seed=corruption_seed,
    )
    _assert_corruption_invariants(
        original=fit_rows,
        corrupted=corrupted_fit_rows,
        edge=edge,
        audit=corruption_audit,
    )
    row.update(corruption_audit)
    row["source_row_count"] = len(corrupted_fit_rows)
    reliability = compute_edge_reliability(
        corrupted_fit_rows,
        lhs_columns=edge["lhs_columns"],
        rhs_column=str(edge["rhs_column"]),
        edge_rank=1,
    )
    row.update(reliability)
    row["violation_rate"] = (
        1.0 - float(reliability["reliability_raw"])
        if np.isfinite(float(reliability["reliability_raw"]))
        else math.nan
    )
    corrupted_table_dict = {
        **table_dict,
        str(edge["source_table"]): _FilteredTable(table_dict[str(edge["source_table"])], corrupted_fit_rows),
    }
    if row["reliability_fit_scope"] == "fold_train_static_entity_snapshot":
        fdhg = fit_transform_static_single_edge_fold(
            inner_train_rows=context["inner_train"],
            inner_validation_rows=context["inner_val"],
            source_table=corrupted_table_dict[str(edge["source_table"])],
            task_metadata=metadata,
            edge=edge,
            fit_rows=corrupted_fit_rows,
            fold=fold_id,
        )
    else:
        fdhg = fit_transform_corrupted_temporal_single_edge(
            inner_train_rows=context["inner_train"],
            inner_validation_rows=context["inner_val"],
            source_tables=corrupted_table_dict,
            task_metadata=metadata,
            edge=edge,
            fit_rows=corrupted_fit_rows,
            fold=fold_id,
            fit_horizon=row["reliability_fit_horizon"],
        )
    join_keys = prepared["join_keys"]
    fdhg_cols = feature_columns(fdhg["train_x"], join_keys, metadata)
    audit_rows, usable_cols = audit_residual_columns(
        frame=fdhg["train_x"],
        feature_cols=fdhg_cols,
        fold=fold_id,
        provenance=fdhg["feature_provenance"],
    )
    residual_stats = residual_feature_stats(fdhg["train_x"], fdhg_cols)
    row.update(residual_stats)
    row["residual_feature_count"] = len(fdhg_cols)
    row["duplicate_generated_feature_count"] = _duplicate_feature_count(fdhg["train_x"], fdhg_cols)
    row["constant_generated_feature_count"] = int(sum(not audit["usable"] for audit in audit_rows))
    future_count = sum(int(audit.get("future_lookup_violation_count") or 0) for audit in fdhg.get("target_lookup_audit", []))
    row["future_row_violation_count"] = int(row.get("future_row_violation_count") or 0) + future_count
    if future_count:
        row.update({"edge_status": "failed", "failure_reason": "future_lookup_violation"})
        return row
    if not usable_cols:
        row.update({"edge_status": "failed", "failure_reason": "no_usable_features"})
        return row
    single_train = align_feature_blocks(
        target_rows=context["inner_train"],
        blocks=[("auto", context["auto_train"]), ("fdhg", fdhg["train_x"])],
        join_keys=join_keys,
        metadata=metadata,
    )
    single_val = align_feature_blocks(
        target_rows=context["inner_val"],
        blocks=[("auto", context["auto_val"]), ("fdhg", fdhg["validation_x"])],
        join_keys=join_keys,
        metadata=metadata,
    )
    edge_score = score_matrix(
        train_x=single_train,
        val_x=single_val,
        train_y=context["inner_train"][metadata["label_col"]],
        val_y=context["inner_val"][metadata["label_col"]],
        feature_cols=[*auto_feature_cols, *usable_cols],
        metadata=metadata,
        options=_auto_options(options, random_seed=corruption_seed),
    )
    row["corrupted_edge_score"] = edge_score
    row["delta_over_base"] = oriented_delta(
        base_score=float(context["base_score"]),
        edge_score=float(edge_score),
        direction=str(metadata["metric_direction"]),
    )
    row.update({"edge_status": "ok", "failure_reason": ""})
    return row


def fit_transform_corrupted_temporal_single_edge(
    *,
    inner_train_rows: pd.DataFrame,
    inner_validation_rows: pd.DataFrame,
    source_tables: Mapping[str, Any],
    task_metadata: Mapping[str, Any],
    edge: Mapping[str, Any],
    fit_rows: pd.DataFrame,
    fold: int,
    fit_horizon: Any,
) -> dict[str, Any]:
    table = source_tables[str(edge["source_table"])]
    mapping = fit_ambiguity_map(
        fit_rows,
        lhs_columns=edge["lhs_columns"],
        rhs_column=str(edge["rhs_column"]),
    )
    fitted = edge_from_mapping(
        edge_id=str(edge["edge_id"]),
        source_table=str(edge["source_table"]),
        lhs_columns=edge["lhs_columns"],
        rhs_column=str(edge["rhs_column"]),
        mapping=mapping,
        fit_df=fit_rows,
        time_col=getattr(table, "time_col", None),
        fit_horizon=fit_horizon,
        fold=fold,
    )
    train_x, train_prov, train_lookup_audit = materialize_ambiguity_features(
        fitted_edges=[fitted],
        target_rows=inner_train_rows,
        source_tables=source_tables,
        task_metadata=task_metadata,
        source_entity_columns_by_edge={
            str(edge["edge_id"]): str(edge.get("source_entity_column", ""))
        },
    )
    val_x, val_prov, val_lookup_audit = materialize_ambiguity_features(
        fitted_edges=[fitted],
        target_rows=inner_validation_rows,
        source_tables=source_tables,
        task_metadata=task_metadata,
        source_entity_columns_by_edge={
            str(edge["edge_id"]): str(edge.get("source_entity_column", ""))
        },
    )
    return {
        "fitted_edges": [fitted],
        "train_x": train_x,
        "validation_x": val_x,
        "feature_provenance": train_prov + val_prov,
        "target_lookup_audit": train_lookup_audit + val_lookup_audit,
    }


def corrupt_dependent_by_permutation(
    rows: pd.DataFrame,
    *,
    lhs_columns: Sequence[str] | None = None,
    rhs_column: str,
    rate: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return corrupt_dependent_by_fd_aware_swaps(
        rows,
        lhs_columns=() if lhs_columns is None else lhs_columns,
        rhs_column=rhs_column,
        rate=rate,
        seed=seed,
    )


def corrupt_dependent_by_fd_aware_swaps(
    rows: pd.DataFrame,
    *,
    lhs_columns: Sequence[str],
    rhs_column: str,
    rate: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rhs_column not in rows.columns:
        raise ValueError(f"missing_corruption_column:{rhs_column}")
    missing_lhs = [col for col in lhs_columns if col not in rows.columns]
    if missing_lhs:
        raise ValueError(f"missing_corruption_lhs_column:{','.join(sorted(missing_lhs))}")
    if rate < 0.0 or rate > 1.0:
        raise ValueError("corruption_rate_must_be_between_0_and_1")
    out = rows.copy()
    plan = _fd_aware_swap_plan(rows, lhs_columns=lhs_columns, rhs_column=rhs_column, seed=seed)
    max_changed = int(plan["maximum_feasible_changed_row_count"])
    requested_changed = min(max_changed, int(round(float(rate) * max_changed)))
    requested_changed -= requested_changed % 2
    rng = np.random.default_rng(int(seed))
    pair_scores = pd.Series(rng.random(len(plan["pairs"])), index=range(len(plan["pairs"])))
    pair_order = pair_scores.sort_values(kind="mergesort").index.tolist()
    ordered_pairs = [plan["pairs"][idx] for idx in pair_order]
    selected_pairs = list(ordered_pairs[: requested_changed // 2])
    original_rhs = rows[rhs_column].copy()
    for left, right in selected_pairs:
        left_value = out.at[left, rhs_column]
        out.at[left, rhs_column] = out.at[right, rhs_column]
        out.at[right, rhs_column] = left_value
    actual_changed = int(original_rhs.ne(out[rhs_column]).sum())
    baseline_violations = _fd_violation_row_count(rows, lhs_columns=lhs_columns, rhs_column=rhs_column)
    achieved_violations = _fd_violation_row_count(out, lhs_columns=lhs_columns, rhs_column=rhs_column)
    eligible_count = int(plan["eligible_row_count"])
    source_count = len(rows)
    original_counts = rows[rhs_column].value_counts(dropna=False).sort_index()
    corrupted_counts = out[rhs_column].value_counts(dropna=False).sort_index()
    preserved = original_counts.equals(corrupted_counts)
    return out, {
        "source_row_count": source_count,
        "eligible_row_count": eligible_count,
        "requested_sampled_row_count": requested_changed,
        "actual_changed_row_count": actual_changed,
        "induced_fd_violation_row_count": max(0, achieved_violations - baseline_violations),
        "effective_changed_row_rate": float(actual_changed / max(1, eligible_count)),
        "effective_changed_rate_among_eligible": float(actual_changed / max(1, eligible_count)),
        "effective_changed_rate_among_all_source_rows": float(actual_changed / max(1, source_count)),
        "achieved_violation_rate": float(achieved_violations / max(1, eligible_count)),
        "maximum_feasible_changed_row_count": max_changed,
        "maximum_feasible_marginal_preserving_changed_row_rate": float(max_changed / max(1, eligible_count)),
        "corruption_saturated": bool(float(rate) >= 1.0 and max_changed > 0),
        "selected_swap_pair_count": len(selected_pairs),
        "unchanged_assignment_count": 0,
        "corruption_selected_row_count": requested_changed,
        "corruption_eligible_row_count": eligible_count,
        "dependent_marginal_counts_preserved": bool(preserved),
    }


def _fd_aware_swap_plan(
    rows: pd.DataFrame,
    *,
    lhs_columns: Sequence[str],
    rhs_column: str,
    seed: int,
) -> dict[str, Any]:
    rhs_norm = normalize_series(rows[rhs_column])
    lhs_norm = (
        normalize_lhs_frame(rows, lhs_columns)
        if lhs_columns
        else pd.Series(["__ALL__"] * len(rows), index=rows.index, dtype="string")
    )
    eligible_mask = (rhs_norm != "__NULL__") & (lhs_norm != "__NULL__")
    eligible = rows.index[eligible_mask.to_numpy()].tolist()
    if not eligible:
        return {"eligible_row_count": 0, "maximum_feasible_changed_row_count": 0, "pairs": []}
    dominant_by_group = _dominant_dependent_by_group(lhs_norm.loc[eligible], rhs_norm.loc[eligible])
    remaining_slots = _non_dominant_receive_slots(lhs_norm.loc[eligible], rhs_norm.loc[eligible], dominant_by_group)
    del seed
    candidates: list[tuple[int, str, str, Any, Any]] = []
    for left_pos, left_idx in enumerate(eligible):
        left_value = rhs_norm.at[left_idx]
        left_group = lhs_norm.at[left_idx]
        for right_idx in eligible[left_pos + 1:]:
            right_value = rhs_norm.at[right_idx]
            if left_value == right_value:
                continue
            right_group = lhs_norm.at[right_idx]
            if right_value == dominant_by_group.get(left_group):
                continue
            if left_value == dominant_by_group.get(right_group):
                continue
            same_group_penalty = int(left_group == right_group)
            candidates.append((same_group_penalty, str(left_idx), str(right_idx), left_idx, right_idx))
    used: set[Any] = set()
    pairs: list[tuple[Any, Any]] = []
    used_slots = dict(remaining_slots)
    for _penalty, _left_text, _right_text, left_idx, right_idx in sorted(candidates):
        if left_idx in used or right_idx in used:
            continue
        left_group = str(lhs_norm.at[left_idx])
        right_group = str(lhs_norm.at[right_idx])
        if used_slots.get(left_group, 0) <= 0 or used_slots.get(right_group, 0) <= 0:
            continue
        used.add(left_idx)
        used.add(right_idx)
        used_slots[left_group] -= 1
        used_slots[right_group] -= 1
        pairs.append((left_idx, right_idx))
    return {
        "eligible_row_count": len(eligible),
        "maximum_feasible_changed_row_count": 2 * len(pairs),
        "pairs": pairs,
    }


def _dominant_dependent_by_group(lhs_norm: pd.Series, rhs_norm: pd.Series) -> dict[str, str]:
    tmp = pd.DataFrame({"lhs": lhs_norm.astype(str), "rhs": rhs_norm.astype(str)})
    dominant: dict[str, str] = {}
    for lhs, group in tmp.groupby("lhs", sort=True):
        counts = group["rhs"].value_counts(sort=False)
        dominant[str(lhs)] = str(sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0])
    return dominant


def _non_dominant_receive_slots(
    lhs_norm: pd.Series,
    rhs_norm: pd.Series,
    dominant_by_group: Mapping[str, str],
) -> dict[str, int]:
    slots: dict[str, int] = {}
    tmp = pd.DataFrame({"lhs": lhs_norm.astype(str), "rhs": rhs_norm.astype(str)})
    for lhs, group in tmp.groupby("lhs", sort=True):
        dominant = dominant_by_group[str(lhs)]
        current_non_dominant = int((group["rhs"] != dominant).sum())
        max_disagreement_rows = len(group) // 2
        slots[str(lhs)] = max(0, max_disagreement_rows - current_non_dominant)
    return slots


def _fd_violation_row_count(
    rows: pd.DataFrame,
    *,
    lhs_columns: Sequence[str],
    rhs_column: str,
) -> int:
    lhs_norm = (
        normalize_lhs_frame(rows, lhs_columns)
        if lhs_columns
        else pd.Series(["__ALL__"] * len(rows), index=rows.index, dtype="string")
    )
    rhs_norm = normalize_series(rows[rhs_column])
    tmp = pd.DataFrame({"lhs": lhs_norm, "rhs": rhs_norm})
    tmp = tmp[(tmp["lhs"] != "__NULL__") & (tmp["rhs"] != "__NULL__")]
    count = 0
    for _, group in tmp.groupby("lhs", sort=True):
        counts = group["rhs"].value_counts(sort=False)
        if not counts.empty:
            count += int(len(group) - counts.max())
    return count


def edge_suitability_audit(
    rows: pd.DataFrame,
    *,
    lhs_columns: Sequence[str],
    rhs_column: str,
) -> dict[str, Any]:
    rhs_norm = normalize_series(rows[rhs_column]) if rhs_column in rows else pd.Series(dtype="string")
    lhs_norm = (
        normalize_lhs_frame(rows, lhs_columns)
        if lhs_columns
        else pd.Series(["__ALL__"] * len(rows), index=rows.index, dtype="string")
    )
    eligible_mask = (rhs_norm != "__NULL__") & (lhs_norm != "__NULL__")
    eligible_rhs = rhs_norm.loc[eligible_mask]
    counts = eligible_rhs.value_counts(sort=False)
    eligible_count = int(eligible_mask.sum())
    plan = _fd_aware_swap_plan(rows, lhs_columns=lhs_columns, rhs_column=rhs_column, seed=0)
    baseline_violations = _fd_violation_row_count(rows, lhs_columns=lhs_columns, rhs_column=rhs_column)
    if eligible_count <= 0 or counts.empty:
        min_freq = 0
        max_freq = 0
        dependent_cardinality = 0
        minority_fraction = math.nan
    else:
        min_freq = int(counts.min())
        max_freq = int(counts.max())
        dependent_cardinality = int(len(counts))
        minority_fraction = float(min_freq / eligible_count)
    group_support = pd.DataFrame({"lhs": lhs_norm.loc[eligible_mask]}).groupby("lhs", sort=True).size()
    return {
        "dependent_cardinality": dependent_cardinality,
        "dependent_min_class_frequency": min_freq,
        "dependent_max_class_frequency": max_freq,
        "dependent_minority_fraction": minority_fraction,
        "maximum_feasible_changed_row_count": int(plan["maximum_feasible_changed_row_count"]),
        "maximum_feasible_marginal_preserving_changed_row_rate": float(
            int(plan["maximum_feasible_changed_row_count"]) / max(1, eligible_count)
        ),
        "determinant_group_support": int((group_support > 1).sum()) if not group_support.empty else 0,
        "baseline_violation_rate": float(baseline_violations / max(1, eligible_count)),
    }


def residual_feature_stats(frame: pd.DataFrame, feature_cols: Sequence[str]) -> dict[str, Any]:
    if not feature_cols:
        return {
            "residual_nonzero_rate": math.nan,
            "residual_variance": math.nan,
            "residual_unique_value_count": 0,
        }
    numeric = frame[list(feature_cols)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "residual_nonzero_rate": float((finite != 0.0).mean()) if len(finite) else math.nan,
        "residual_variance": float(np.var(finite)) if len(finite) else math.nan,
        "residual_unique_value_count": int(pd.Series(finite).nunique(dropna=True)) if len(finite) else 0,
    }


def aggregate_fd_violation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    valid = df[df["edge_status"].eq("ok")].copy()
    if valid.empty:
        return []
    out: list[dict[str, Any]] = []
    group_cols = ["dataset", "task", "edge_id", "requested_corruption_rate"]
    for keys, group in valid.groupby(group_cols, sort=True):
        first = group.iloc[0]
        row = {
            "dataset": keys[0],
            "task": keys[1],
            "edge_id": keys[2],
            "source_table": first["source_table"],
            "determinant": first["determinant"],
            "dependent": first["dependent"],
            "requested_corruption_rate": float(keys[3]),
            "mean_effective_changed_row_rate": _mean(group["effective_changed_row_rate"]),
            "std_effective_changed_row_rate": _std(group["effective_changed_row_rate"]),
            "mean_effective_changed_rate_among_eligible": _mean(group["effective_changed_rate_among_eligible"]),
            "mean_achieved_violation_rate": _mean(group["achieved_violation_rate"]),
            "mean_induced_fd_violation_row_count": _mean(group["induced_fd_violation_row_count"]),
            "mean_reliability_loo": _mean(group["reliability_loo"]),
            "std_reliability_loo": _std(group["reliability_loo"]),
            "mean_reliability_raw": _mean(group["reliability_raw"]),
            "mean_reliability_entropy": _mean(group["reliability_entropy"]),
            "mean_conditional_entropy_normalized": _mean(group["conditional_entropy_normalized"]),
            "mean_violation_rate": _mean(group["violation_rate"]),
            "mean_residual_nonzero_rate": _mean(group["residual_nonzero_rate"]),
            "mean_residual_variance": _mean(group["residual_variance"]),
            "mean_delta_over_base": _mean(group["delta_over_base"]),
            "std_delta_over_base": _std(group["delta_over_base"]),
            "mean_delta_relative_to_uncorrupted": _mean(group["delta_relative_to_uncorrupted"]),
            "std_delta_relative_to_uncorrupted": _std(group["delta_relative_to_uncorrupted"]),
            "valid_fold_count": len(group),
            "corruption_seed_count": int(group["corruption_seed"].nunique()),
        }
        out.append(_ordered_row(row, AGGREGATE_COLUMNS))
    return out


def figure_fd_violation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    valid = df[df["edge_status"].eq("ok")].copy()
    out: list[dict[str, Any]] = []
    for keys, group in valid.groupby(["dataset", "task", "edge_id", "requested_corruption_rate"], sort=True):
        first = group.iloc[0]
        n = max(1, len(group))
        row = {
            "dataset": keys[0],
            "task": keys[1],
            "edge_id": keys[2],
            "source_table": first["source_table"],
            "determinant": first["determinant"],
            "dependent": first["dependent"],
            "requested_corruption_rate": float(keys[3]),
            "mean_effective_changed_row_rate": _mean(group["effective_changed_row_rate"]),
            "mean_effective_changed_rate_among_eligible": _mean(group["effective_changed_rate_among_eligible"]),
            "mean_achieved_violation_rate": _mean(group["achieved_violation_rate"]),
            "mean_reliability_loo": _mean(group["reliability_loo"]),
            "sem_reliability_loo": _std(group["reliability_loo"]) / math.sqrt(n),
            "mean_delta_relative_to_uncorrupted": _mean(group["delta_relative_to_uncorrupted"]),
            "sem_delta_relative_to_uncorrupted": _std(group["delta_relative_to_uncorrupted"]) / math.sqrt(n),
            "mean_residual_nonzero_rate": _mean(group["residual_nonzero_rate"]),
            "sem_residual_nonzero_rate": _std(group["residual_nonzero_rate"]) / math.sqrt(n),
            "mean_residual_variance": _mean(group["residual_variance"]),
            "sem_residual_variance": _std(group["residual_variance"]) / math.sqrt(n),
            "valid_fold_count": len(group),
        }
        out.append(_ordered_row(row, FIGURE_AGGREGATE_COLUMNS))
    return out


def oriented_delta(*, base_score: float, edge_score: float, direction: str) -> float:
    return float(edge_score - base_score) if direction == "higher" else float(base_score - edge_score)


def _base_violation_row(
    *,
    prepared: Mapping[str, Any],
    edge: Mapping[str, Any],
    fold: int,
    corruption_seed: int,
    requested_corruption_rate: float,
    base_score: float,
) -> dict[str, Any]:
    metadata = prepared["metadata"]
    return {
        "dataset": prepared["manifest"]["dataset"],
        "task": prepared["manifest"]["task"],
        "edge_id": edge["edge_id"],
        "source_table": edge["source_table"],
        "determinant": "|".join(edge["lhs_columns"]),
        "dependent": edge["rhs_column"],
        "fold": int(fold),
        "corruption_seed": int(corruption_seed),
        "requested_corruption_rate": float(requested_corruption_rate),
        "effective_changed_row_rate": math.nan,
        "base_score": base_score,
        "corrupted_edge_score": math.nan,
        "delta_over_base": math.nan,
        "uncorrupted_edge_score": math.nan,
        "delta_relative_to_uncorrupted": math.nan,
        "metric": metadata["primary_metric"],
        "metric_direction": metadata["metric_direction"],
        "edge_status": "pending",
        "failure_reason": "",
        "future_row_violation_count": 0,
        "inner_validation_row_usage_count": 0,
        "official_validation_row_usage_count": 0,
        "test_row_usage_count": 0,
    }


def _auto_options(options: FdViolationOptions, *, random_seed: int) -> AutoFdhgOptions:
    return AutoFdhgOptions(
        selection_folds=options.selection_folds,
        feature_budget=options.feature_budget,
        min_delta=options.min_delta,
        selection_decoder=options.selection_decoder,
        max_relations=options.max_relations,
        max_numeric_columns=options.max_numeric_columns,
        max_categorical_columns=options.max_categorical_columns,
        random_seed=random_seed,
    )


def _assert_corruption_invariants(
    *,
    original: pd.DataFrame,
    corrupted: pd.DataFrame,
    edge: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    if len(original) != len(corrupted):
        raise AssertionError("corruption_changed_row_count")
    table_columns = [col for col in original.columns if col != str(edge["rhs_column"])]
    for col in table_columns:
        if not original[col].reset_index(drop=True).equals(corrupted[col].reset_index(drop=True)):
            raise AssertionError(f"corruption_changed_non_dependent_column:{col}")
    if not audit.get("dependent_marginal_counts_preserved"):
        raise AssertionError("corruption_changed_dependent_marginal_counts")


def _assert_leakage_counters_zero(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        for col in (
            "future_row_violation_count",
            "inner_validation_row_usage_count",
            "official_validation_row_usage_count",
            "test_row_usage_count",
        ):
            value = row.get(col, 0)
            if value not in ("", None) and int(float(value)) != 0:
                raise AssertionError(f"leakage_counter_nonzero:{col}")


def _relation_for_table(prepared: Mapping[str, Any], table_name: str) -> Mapping[str, Any] | None:
    for relation in motivation_candidate_relations(prepared):
        if str(relation.get("child_table", "")) == table_name:
            return relation
    return None


def _edge_path_token(edge_spec: str) -> str:
    return edge_spec.replace(":", "_").replace("->", "_to_").replace("|", "_")


def _write_ordered_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(_ordered_row(row, columns))
