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
    fitted_edge_to_audit_row,
    materialize_ambiguity_from_map,
)
from fdhg.compiler.edge_reliability import (
    compute_edge_reliability,
)
from fdhg.compiler.fold_safe_fdhg import (
    discover_earliest_fold_candidate_edges,
    train_source_view,
)
from fdhg.onboarding.auto_fdhg import (
    AutoFdhgOptions,
    _edge_identity_row,
    _write_json,
    align_feature_blocks,
    audit_residual_columns,
    edge_fold_gain,
    edge_screening_delta_passes,
    edge_screening_positive_fold_passes,
    feature_columns,
    fit_transform_single_edge_fdhg_fold_cached,
    materialize_declared_feature_frame_pair,
    prepare_auto_fdhg,
    resolve_edge_screening_min_positive_folds,
    score_matrix,
)
from fdhg.onboarding.auto_relbench import make_inner_temporal_splits

MOTIVATION_VERSION = "motivation-reliability-utility-v1"
FOLD_LEVEL_COLUMNS = (
    "dataset",
    "task",
    "seed",
    "fold",
    "edge_id",
    "edge_rank",
    "determinant",
    "dependent",
    "source_table",
    "relational_path",
    "reliability_raw",
    "reliability_non_singleton",
    "reliability_loo",
    "reliability_entropy",
    "conditional_entropy_normalized",
    "non_singleton_coverage",
    "total_support",
    "determinant_group_count",
    "determinant_cardinality",
    "dependent_cardinality",
    "singleton_group_count",
    "singleton_group_ratio",
    "singleton_row_ratio",
    "mean_group_size",
    "median_group_size",
    "max_group_size",
    "non_singleton_row_count",
    "generated_feature_count",
    "generated_feature_names",
    "duplicate_generated_feature_count",
    "constant_generated_feature_count",
    "base_score",
    "edge_score",
    "delta",
    "metric",
    "metric_direction",
    "edge_status",
    "failure_reason",
    "official_validation_was_used",
    "test_split_accessed",
    "reliability_fit_scope",
    "reliability_fit_horizon",
    "target_entity_key",
    "source_entity_column",
    "source_entity_column_resolution",
    "source_relation_id",
    "source_row_count_before_filtering",
    "source_row_count_after_filtering",
    "validation_target_entity_overlap_in_reliability_rows",
    "future_row_violation_count",
    "official_validation_row_usage_count",
    "test_row_usage_count",
)
AGGREGATE_COLUMNS = (
    "dataset",
    "task",
    "seed",
    "edge_id",
    "edge_rank",
    "determinant",
    "dependent",
    "source_table",
    "relational_path",
    "mean_reliability_loo",
    "std_reliability_loo",
    "mean_reliability_raw",
    "std_reliability_raw",
    "mean_reliability_entropy",
    "std_reliability_entropy",
    "mean_delta",
    "std_delta",
    "positive_fold_count",
    "passing_fold_count",
    "screening_min_delta",
    "screening_min_positive_folds",
    "selected_by_existing_screening_rule",
    "selected_by_two_of_three_rule",
    "mean_coverage",
    "valid_fold_count",
)
FIGURE_AGGREGATE_COLUMNS = (
    "dataset",
    "task",
    "edge_id",
    "edge_rank",
    "determinant",
    "dependent",
    "source_table",
    "relational_path",
    "mean_reliability_loo",
    "seed_std_reliability_loo",
    "fold_std_reliability_loo",
    "mean_reliability_raw",
    "mean_reliability_entropy",
    "mean_delta",
    "seed_std_delta",
    "fold_std_delta",
    "positive_fold_count",
    "passing_fold_count",
    "screening_min_delta",
    "screening_min_positive_folds",
    "selected_by_existing_screening_rule",
    "mean_coverage",
    "seed_count",
    "valid_fold_count",
)


@dataclass(frozen=True)
class MotivationOptions:
    selection_folds: int = 3
    feature_budget: int = 8
    min_delta: float = 0.0
    selection_decoder: str = "hist_gradient_boosting"
    motivation_edge_budget: int = 32
    max_relations: int = 3
    max_numeric_columns: int = 4
    max_categorical_columns: int = 4
    random_seed: int = 0
    edge_screening_min_delta: float = 0.0
    edge_screening_min_positive_folds: int | None = None
    smoke_max_train_rows: int | None = None


@dataclass(frozen=True)
class MotivationReport:
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
    candidate_edges: int = 0
    test_split_accessed: bool = False


def motivation_reliability_utility(
    *,
    dataset_name: str,
    task_name: str,
    output_root: Path,
    write: bool = False,
    overwrite: bool = False,
    download: bool = False,
    auto_output_root: Path = Path("outputs/auto-onboarding-3fold"),
    dfs_source_root: Path = Path("."),
    dfs_feature_config: Path | None = None,
    seeds: Sequence[int] = (0,),
    options: MotivationOptions | None = None,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
) -> MotivationReport:
    options = options or MotivationOptions()
    output_dir = output_root / dataset_name / task_name
    try:
        fold_rows, aggregate_rows, figure_rows, blockers, manifest = prepare_motivation_experiment(
            dataset_name=dataset_name,
            task_name=task_name,
            output_root=output_root,
            download=download,
            auto_output_root=auto_output_root,
            dfs_source_root=dfs_source_root,
            dfs_feature_config=dfs_feature_config,
            seeds=seeds,
            options=options,
            object_loader=object_loader,
        )
    except Exception as exc:  # noqa: BLE001 - report setup blockers through the API.
        return MotivationReport(
            dataset=dataset_name,
            task=task_name,
            status="blocked",
            output_dir=output_dir,
            blockers=(str(exc),),
            dry_run=not write,
        )
    if blockers:
        return MotivationReport(
            dataset=dataset_name,
            task=task_name,
            status="blocked",
            output_dir=output_dir,
            blockers=tuple(blockers),
            dry_run=not write,
            fold_rows=len(fold_rows),
            aggregate_rows=len(aggregate_rows),
            candidate_edges=int(manifest.get("candidate_edge_count", 0)),
        )
    if int(manifest.get("candidate_edge_count", 0)) <= 0:
        if write:
            if output_dir.exists() and not overwrite:
                raise FileExistsError(output_dir)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output_dir / "manifest.json", manifest)
        return MotivationReport(
            dataset=dataset_name,
            task=task_name,
            status="no_candidates",
            output_dir=output_dir,
            blockers=(),
            dry_run=not write,
            fold_rows=len(fold_rows),
            aggregate_rows=len(aggregate_rows),
            candidate_edges=0,
        )
    if not write:
        return MotivationReport(
            dataset=dataset_name,
            task=task_name,
            status="dry_run_ready",
            output_dir=output_dir,
            blockers=(),
            dry_run=True,
            fold_rows=len(fold_rows),
            aggregate_rows=len(aggregate_rows),
            candidate_edges=int(manifest.get("candidate_edge_count", 0)),
        )
    if output_dir.exists() and not overwrite:
        raise FileExistsError(output_dir)
    staging = output_dir.parent / f"_{output_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_json(staging / "manifest.json", manifest)
        _write_ordered_csv(staging / "fold_level.csv", fold_rows, FOLD_LEVEL_COLUMNS)
        _write_ordered_csv(staging / "edge_aggregate.csv", aggregate_rows, AGGREGATE_COLUMNS)
        _write_ordered_csv(
            staging / "figure_edge_aggregate.csv",
            figure_rows,
            FIGURE_AGGREGATE_COLUMNS,
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return MotivationReport(
        dataset=dataset_name,
        task=task_name,
        status="completed",
        output_dir=output_dir,
        blockers=(),
        dry_run=False,
        fold_csv=output_dir / "fold_level.csv",
        aggregate_csv=output_dir / "edge_aggregate.csv",
        figure_aggregate_csv=output_dir / "figure_edge_aggregate.csv",
        fold_rows=len(fold_rows),
        aggregate_rows=len(aggregate_rows),
        candidate_edges=int(manifest.get("candidate_edge_count", 0)),
    )


def prepare_motivation_experiment(
    *,
    dataset_name: str,
    task_name: str,
    output_root: Path,
    download: bool,
    auto_output_root: Path,
    dfs_source_root: Path,
    dfs_feature_config: Path | None,
    seeds: Sequence[int],
    options: MotivationOptions,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    prepared_by_seed: dict[int, Mapping[str, Any]] = {}
    for seed in seeds:
        auto_options = AutoFdhgOptions(
            selection_folds=options.selection_folds,
            feature_budget=options.feature_budget,
            min_delta=options.min_delta,
            selection_decoder=options.selection_decoder,
            max_fdhg_edges=options.motivation_edge_budget,
            max_relations=options.max_relations,
            max_numeric_columns=options.max_numeric_columns,
            max_categorical_columns=options.max_categorical_columns,
            random_seed=int(seed),
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
        if options.smoke_max_train_rows is not None:
            prepared = _apply_smoke_train_limit(prepared, options.smoke_max_train_rows)
        prepared = {
            **prepared,
            "motivation_relations": motivation_candidate_relations(prepared),
        }
        discovery = discover_earliest_fold_candidate_edges(
            prepared=prepared,
            edge_budget=options.motivation_edge_budget,
        )
        prepared = {
            **prepared,
            "accepted_fdhg_edges": discovery["accepted_edges"],
            "rejected_fdhg_edges": discovery["rejected_edges"],
            "candidate_discovery": discovery["provenance"],
        }
        prepared_by_seed[int(seed)] = prepared
        blockers.extend(str(item) for item in prepared.get("blockers", ()))
    first = next(iter(prepared_by_seed.values())) if prepared_by_seed else {}
    if blockers:
        manifest = _manifest(dataset_name, task_name, seeds, options, first, blockers)
        return [], [], [], blockers, manifest
    for seed, prepared in prepared_by_seed.items():
        fold_rows.extend(_evaluate_seed(seed=seed, prepared=prepared, options=options))
    aggregate_rows = aggregate_fold_rows(fold_rows, options=options)
    figure_rows = figure_aggregate_rows(fold_rows, options=options)
    manifest = _manifest(dataset_name, task_name, seeds, options, first, blockers)
    manifest.update({
        "fold_observation_count": len(fold_rows),
        "aggregate_observation_count": len(aggregate_rows),
        "figure_edge_observation_count": len(figure_rows),
    })
    return fold_rows, aggregate_rows, figure_rows, blockers, manifest


def _evaluate_seed(
    *,
    seed: int,
    prepared: Mapping[str, Any],
    options: MotivationOptions,
) -> list[dict[str, Any]]:
    metadata = prepared["metadata"]
    table_dict = prepared["table_dict"]
    split_plan = prepared["split_plan"]
    join_keys = prepared["join_keys"]
    auto_features = prepared["auto_features"]
    edges = [dict(edge) for edge in prepared["accepted_fdhg_edges"][: options.motivation_edge_budget]]
    for idx, edge in enumerate(edges, start=1):
        edge["edge_rank"] = idx

    rows: list[dict[str, Any]] = []
    auto_feature_cols_by_fold: dict[int, list[str]] = {}
    auto_context_by_fold: dict[int, dict[str, Any]] = {}
    lookup_source_columns_by_table: dict[str, list[str]] = {}
    for edge in edges:
        table = str(edge.get("source_table", ""))
        for lhs in edge.get("lhs_columns", ()):
            lookup_source_columns_by_table.setdefault(table, [])
            if str(lhs) not in lookup_source_columns_by_table[table]:
                lookup_source_columns_by_table[table].append(str(lhs))

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
            options=AutoFdhgOptions(
                selection_folds=options.selection_folds,
                feature_budget=options.feature_budget,
                min_delta=options.min_delta,
                selection_decoder=options.selection_decoder,
                max_fdhg_edges=options.motivation_edge_budget,
                max_relations=options.max_relations,
                max_numeric_columns=options.max_numeric_columns,
                max_categorical_columns=options.max_categorical_columns,
                random_seed=seed,
            ),
        )
        auto_feature_cols_by_fold[fold_id] = auto_cols
        auto_context_by_fold[fold_id] = {
            "inner_train": train_targets,
            "inner_val": validation_targets,
            "auto_train": auto_train,
            "auto_val": auto_val,
            "base_score": base_score,
        }

    pit_lookup_cache: dict[tuple[int, str, str], tuple[pd.DataFrame, dict[str, Any]]] = {}
    for edge in edges:
        for fold in split_plan["folds"]:
            fold_id = int(fold["fold"])
            context = auto_context_by_fold[fold_id]
            row = _base_fold_row(
                prepared=prepared,
                seed=seed,
                fold=fold_id,
                edge=edge,
                base_score=float(context["base_score"]),
            )
            try:
                fit_view = fold_train_source_view_for_edge(
                    table_dict=table_dict,
                    metadata=metadata,
                    train_targets=context["inner_train"],
                    validation_targets=context["inner_val"],
                    edge=edge,
                )
                row.update(fit_view["audit"])
                if fit_view["blocked_reason"]:
                    row.update({
                        "edge_status": "blocked",
                        "failure_reason": fit_view["blocked_reason"],
                    })
                    rows.append(_ordered_row(row, FOLD_LEVEL_COLUMNS))
                    continue
                row.update(compute_edge_reliability(
                    fit_view["fit_rows"],
                    lhs_columns=edge["lhs_columns"],
                    rhs_column=str(edge["rhs_column"]),
                    edge_rank=int(edge["edge_rank"]),
                ))
                if fit_view["audit"]["reliability_fit_scope"] == "fold_train_static_entity_snapshot":
                    fdhg = fit_transform_static_single_edge_fold(
                        inner_train_rows=context["inner_train"],
                        inner_validation_rows=context["inner_val"],
                        source_table=fit_view["table_dict"][str(edge["source_table"])],
                        task_metadata=metadata,
                        edge=edge,
                        fit_rows=fit_view["fit_rows"],
                        fold=fold_id,
                    )
                else:
                    fdhg = fit_transform_single_edge_fdhg_fold_cached(
                        inner_train_rows=context["inner_train"],
                        inner_validation_rows=context["inner_val"],
                        source_tables=fit_view["table_dict"],
                        task_metadata=metadata,
                        edge=edge,
                        lookup_source_columns_by_table=lookup_source_columns_by_table,
                        pit_lookup_cache=pit_lookup_cache,
                        fold=fold_id,
                    )
                fdhg_cols = feature_columns(fdhg["train_x"], join_keys, metadata)
                audit_rows, usable_cols = audit_residual_columns(
                    frame=fdhg["train_x"],
                    feature_cols=fdhg_cols,
                    fold=fold_id,
                    provenance=fdhg["feature_provenance"],
                )
                duplicate_count = _duplicate_feature_count(fdhg["train_x"], fdhg_cols)
                constant_count = int(sum(not audit["usable"] for audit in audit_rows))
                row.update({
                    "generated_feature_count": len(fdhg_cols),
                    "generated_feature_names": "|".join(fdhg_cols),
                    "duplicate_generated_feature_count": duplicate_count,
                    "constant_generated_feature_count": constant_count,
                })
                future_count = sum(
                    int(audit.get("future_lookup_violation_count") or 0)
                    for audit in fdhg.get("target_lookup_audit", [])
                )
                if future_count:
                    row.update({
                        "edge_status": "leakage_violation",
                        "failure_reason": "future_lookup_violation",
                    })
                elif not usable_cols:
                    row.update({
                        "edge_status": "no_usable_features",
                        "failure_reason": "no_usable_features",
                    })
                else:
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
                        feature_cols=[*auto_feature_cols_by_fold[fold_id], *usable_cols],
                        metadata=metadata,
                        options=AutoFdhgOptions(
                            selection_folds=options.selection_folds,
                            feature_budget=options.feature_budget,
                            min_delta=options.min_delta,
                            selection_decoder=options.selection_decoder,
                            max_fdhg_edges=options.motivation_edge_budget,
                            max_relations=options.max_relations,
                            max_numeric_columns=options.max_numeric_columns,
                            max_categorical_columns=options.max_categorical_columns,
                            random_seed=seed,
                        ),
                    )
                    row.update({
                        "edge_score": edge_score,
                        "delta": edge_fold_gain(
                            auto_score=float(context["base_score"]),
                            auto_plus_single_edge_score=edge_score,
                            direction=metadata["metric_direction"],
                        ),
                        "edge_status": "ok",
                        "failure_reason": "",
                    })
            except Exception as exc:  # noqa: BLE001 - edge failures must be recorded, not dropped.
                row.update({
                    "edge_status": "failed",
                    "failure_reason": str(exc),
                })
            rows.append(_ordered_row(row, FOLD_LEVEL_COLUMNS))
    return rows


def aggregate_fold_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    options: MotivationOptions,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for keys, group in df.groupby(["dataset", "task", "seed", "edge_id"], sort=True):
        valid = group[group["edge_status"].eq("ok") & pd.to_numeric(group["delta"], errors="coerce").notna()]
        first = group.iloc[0]
        deltas = pd.to_numeric(valid["delta"], errors="coerce")
        positive = int(
            sum(edge_screening_positive_fold_passes(delta=value) for value in deltas)
        ) if not valid.empty else 0
        min_delta = float(options.edge_screening_min_delta)
        min_positive = _resolved_screening_min_positive_folds(options)
        passing = int(
            sum(edge_screening_delta_passes(delta=value, min_delta=min_delta) for value in deltas)
        ) if not valid.empty else 0
        selected = passing >= min_positive
        is_two_of_three = int(options.selection_folds) == 3 and min_positive == 2
        row = {
            "dataset": keys[0],
            "task": keys[1],
            "seed": keys[2],
            "edge_id": keys[3],
            "edge_rank": first.get("edge_rank", ""),
            "determinant": first.get("determinant", ""),
            "dependent": first.get("dependent", ""),
            "source_table": first.get("source_table", ""),
            "relational_path": first.get("relational_path", ""),
            "mean_reliability_loo": _mean(group["reliability_loo"]),
            "std_reliability_loo": _std(group["reliability_loo"]),
            "mean_reliability_raw": _mean(group["reliability_raw"]),
            "std_reliability_raw": _std(group["reliability_raw"]),
            "mean_reliability_entropy": _mean(group["reliability_entropy"]),
            "std_reliability_entropy": _std(group["reliability_entropy"]),
            "mean_delta": _mean(valid["delta"]) if not valid.empty else math.nan,
            "std_delta": _std(valid["delta"]) if not valid.empty else math.nan,
            "positive_fold_count": positive,
            "passing_fold_count": passing,
            "screening_min_delta": min_delta,
            "screening_min_positive_folds": min_positive,
            "selected_by_existing_screening_rule": selected,
            "selected_by_two_of_three_rule": selected if is_two_of_three else "",
            "mean_coverage": _mean(group["non_singleton_coverage"]),
            "valid_fold_count": len(valid),
        }
        out.append(_ordered_row(row, AGGREGATE_COLUMNS))
    return out


def figure_aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    options: MotivationOptions,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    seed_agg = pd.DataFrame(aggregate_fold_rows(rows, options=options))
    if seed_agg.empty:
        return []
    out: list[dict[str, Any]] = []
    for keys, group in seed_agg.groupby(["dataset", "task", "edge_id"], sort=True):
        first = group.iloc[0]
        row = {
            "dataset": keys[0],
            "task": keys[1],
            "edge_id": keys[2],
            "edge_rank": first.get("edge_rank", ""),
            "determinant": first.get("determinant", ""),
            "dependent": first.get("dependent", ""),
            "source_table": first.get("source_table", ""),
            "relational_path": first.get("relational_path", ""),
            "mean_reliability_loo": _mean(group["mean_reliability_loo"]),
            "seed_std_reliability_loo": _std(group["mean_reliability_loo"]),
            "fold_std_reliability_loo": _mean(group["std_reliability_loo"]),
            "mean_reliability_raw": _mean(group["mean_reliability_raw"]),
            "mean_reliability_entropy": _mean(group["mean_reliability_entropy"]),
            "mean_delta": _mean(group["mean_delta"]),
            "seed_std_delta": _std(group["mean_delta"]),
            "fold_std_delta": _mean(group["std_delta"]),
            "positive_fold_count": int(pd.to_numeric(group["positive_fold_count"], errors="coerce").fillna(0).sum()),
            "passing_fold_count": int(pd.to_numeric(group["passing_fold_count"], errors="coerce").fillna(0).sum()),
            "screening_min_delta": float(options.edge_screening_min_delta),
            "screening_min_positive_folds": _resolved_screening_min_positive_folds(options),
            "selected_by_existing_screening_rule": bool(group["selected_by_existing_screening_rule"].eq(True).any()),
            "mean_coverage": _mean(group["mean_coverage"]),
            "seed_count": int(group["seed"].nunique()),
            "valid_fold_count": int(pd.to_numeric(group["valid_fold_count"], errors="coerce").fillna(0).sum()),
        }
        out.append(_ordered_row(row, FIGURE_AGGREGATE_COLUMNS))
    return out


def _base_fold_row(
    *,
    prepared: Mapping[str, Any],
    seed: int,
    fold: int,
    edge: Mapping[str, Any],
    base_score: float,
) -> dict[str, Any]:
    metadata = prepared["metadata"]
    identity = _edge_identity_row(edge)
    return {
        "dataset": prepared["manifest"]["dataset"],
        "task": prepared["manifest"]["task"],
        "seed": int(seed),
        "fold": int(fold),
        "edge_id": identity["edge_id"],
        "edge_rank": int(edge.get("edge_rank", 0)),
        "determinant": identity["lhs_columns"],
        "dependent": identity["rhs_column"],
        "source_table": identity["source_table"],
        "relational_path": f"{identity['source_table']}:{identity['lhs_columns']}->{identity['rhs_column']}",
        "base_score": base_score,
        "edge_score": math.nan,
        "delta": math.nan,
        "metric": metadata["primary_metric"],
        "metric_direction": metadata["metric_direction"],
        "edge_status": "pending",
        "failure_reason": "",
        "official_validation_was_used": False,
        "test_split_accessed": False,
        "reliability_fit_scope": "",
        "reliability_fit_horizon": "",
        "source_row_count_before_filtering": "",
        "source_row_count_after_filtering": "",
        "validation_target_entity_overlap_in_reliability_rows": "",
        "future_row_violation_count": "",
        "official_validation_row_usage_count": 0,
        "test_row_usage_count": 0,
    }


def _duplicate_feature_count(frame: pd.DataFrame, feature_cols: Sequence[str]) -> int:
    seen: dict[int, str] = {}
    duplicates = 0
    for col in feature_cols:
        hashed = int(pd.util.hash_pandas_object(frame[col], index=False).sum())
        if hashed in seen and frame[col].equals(frame[seen[hashed]]):
            duplicates += 1
        else:
            seen[hashed] = col
    return duplicates


class _FilteredTable:
    def __init__(self, original: Any, df: pd.DataFrame) -> None:
        self.df = df
        self.pkey_col = getattr(original, "pkey_col", None)
        self.fkey_col_to_pkey_table = getattr(original, "fkey_col_to_pkey_table", {}) or {}
        self.time_col = getattr(original, "time_col", None)


def fold_train_source_view_for_edge(
    *,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    edge: Mapping[str, Any],
) -> dict[str, Any]:
    table_name = str(edge["source_table"])
    view = train_source_view(
        table=table_dict[table_name],
        metadata=metadata,
        train_targets=train_targets,
        validation_targets=validation_targets,
        edge=edge,
    )
    if view["blocked_reason"]:
        return {**view, "table_dict": table_dict}
    filtered = _FilteredTable(table_dict[table_name], view["fit_rows"])
    return {**view, "table_dict": {**table_dict, table_name: filtered}}


def _apply_smoke_train_limit(prepared: Mapping[str, Any], max_train_rows: int) -> dict[str, Any]:
    if max_train_rows < 1:
        raise ValueError("smoke_max_train_rows_must_be_at_least_1")
    metadata = prepared["metadata"]
    train_df = prepared["train_df"].copy()
    times = pd.to_datetime(train_df[metadata["target_time_col"]], errors="coerce")
    limited = (
        train_df.assign(__smoke_time=times)
        .sort_values(["__smoke_time"], kind="mergesort")
        .head(max_train_rows)
        .drop(columns=["__smoke_time"])
        .reset_index(drop=True)
    )
    split_plan = make_inner_temporal_splits(
        limited,
        time_col=metadata["target_time_col"],
        requested_folds=prepared["options"].selection_folds,
    )
    manifest = {
        **prepared["manifest"],
        "smoke_only": True,
        "smoke_max_train_rows": int(max_train_rows),
        "smoke_original_train_rows": len(train_df),
        "smoke_limited_train_rows": len(limited),
    }
    return {**prepared, "train_df": limited, "split_plan": split_plan, "manifest": manifest}


def motivation_candidate_relations(prepared: Mapping[str, Any]) -> list[dict[str, Any]]:
    accepted = [dict(row) for row in prepared.get("accepted_relations", [])]
    seen = {
        (str(row.get("child_table", "")), str(row.get("child_fk", "")))
        for row in accepted
    }
    for row in prepared.get("relations", []):
        key = (str(row.get("child_table", "")), str(row.get("child_fk", "")))
        if key in seen:
            continue
        reasons = {
            reason
            for reason in str(row.get("rejection_reasons", "")).split("|")
            if reason
        }
        if reasons and reasons.issubset({"missing_child_event_time", "no_source_data_before_training_targets"}):
            accepted.append({**dict(row), "status": "accepted", "motivation_static_relation": True})
            seen.add(key)
    return accepted


def fit_transform_static_single_edge_fold(
    *,
    inner_train_rows: pd.DataFrame,
    inner_validation_rows: pd.DataFrame,
    source_table: Any,
    task_metadata: Mapping[str, Any],
    edge: Mapping[str, Any],
    fit_rows: pd.DataFrame,
    fold: int,
) -> dict[str, Any]:
    del source_table
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
        time_col=None,
        fit_horizon=None,
        fold=fold,
    )

    def materialize(target_rows: pd.DataFrame, split: str) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        entity_key = str(task_metadata["entity_key"])
        source_entity = str(edge.get("source_entity_column") or entity_key)
        if source_entity not in fit_rows.columns:
            raise ValueError("missing_static_source_entity_column")
        if fit_rows[source_entity].duplicated().any():
            raise ValueError("static_source_entity_not_unique")
        lookup = target_rows[[entity_key, task_metadata["target_time_col"]]].reset_index(drop=True).copy()
        source_cols = [source_entity, *edge["lhs_columns"]]
        merged = lookup.merge(
            fit_rows[source_cols].drop_duplicates(source_entity),
            left_on=entity_key,
            right_on=source_entity,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        source_view = merged[list(edge["lhs_columns"])].copy()
        frame, provenance = materialize_ambiguity_from_map(source_view, fitted_edge=fitted)
        result = lookup.copy()
        for col in frame.columns:
            result[col] = frame[col].to_numpy()
        audit = {
            "fold": fold,
            "edge_id": edge["edge_id"],
            "split": split,
            "target_row_count": len(target_rows),
            "matched_target_rows": int(merged[source_entity].notna().sum()) if source_entity in merged else 0,
            "unmatched_target_rows": int(merged[source_entity].isna().sum()) if source_entity in merged else len(target_rows),
            "target_lookup_coverage": float(merged[source_entity].notna().mean()) if source_entity in merged and len(merged) else 0.0,
            "maximum_lookup_source_time": None,
            "maximum_target_time": str(pd.to_datetime(target_rows[task_metadata["target_time_col"]], errors="coerce").max()),
            "future_lookup_violation_count": 0,
            "mapping_fit_horizon": None,
            "maximum_mapping_source_time": None,
            "rejection_reason": "",
        }
        return result, provenance, audit

    train_x, train_prov, train_audit = materialize(inner_train_rows, "train")
    val_x, val_prov, val_audit = materialize(inner_validation_rows, "validation")
    return {
        "fitted_edges": [fitted],
        "edge_audit": [fitted_edge_to_audit_row(fitted)],
        "train_x": train_x,
        "validation_x": val_x,
        "feature_provenance": train_prov + val_prov,
        "target_lookup_audit": [train_audit, val_audit],
    }


def _resolved_screening_min_positive_folds(options: MotivationOptions) -> int:
    return resolve_edge_screening_min_positive_folds(
        AutoFdhgOptions(
            selection_folds=options.selection_folds,
            edge_screening_min_positive_folds=options.edge_screening_min_positive_folds,
        )
    )


def _mean(values: Any) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else math.nan


def _std(values: Any) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.std(ddof=0)) if len(numeric) else math.nan


def _ordered_row(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    return {col: row.get(col, "") for col in columns}


def _write_ordered_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(_ordered_row(row, columns))


def _manifest(
    dataset_name: str,
    task_name: str,
    seeds: Sequence[int],
    options: MotivationOptions,
    prepared: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    metadata = prepared.get("metadata", {})
    return {
        "implementation_version": MOTIVATION_VERSION,
        "dataset": dataset_name,
        "task": task_name,
        "seeds": [int(seed) for seed in seeds],
        "selection_folds": options.selection_folds,
        "motivation_edge_budget": options.motivation_edge_budget,
        "candidate_edge_count": len(prepared.get("accepted_fdhg_edges", [])),
        "candidate_discovery_protocol": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_protocol",
            "",
        ),
        "candidate_discovery_fold": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_fold",
            "",
        ),
        "candidate_discovery_target_row_count": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_target_row_count",
            "",
        ),
        "candidate_discovery_fit_horizon": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_fit_horizon",
            "",
        ),
        "candidate_discovery_entity_count": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_entity_count",
            "",
        ),
        "candidate_discovery_scope": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_scope",
            "",
        ),
        "candidate_discovery_max_timestamp": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_max_timestamp",
            "",
        ),
        "candidate_discovery_source_row_counts": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_source_row_counts",
            {},
        ),
        "candidate_column_audit": prepared.get("candidate_discovery", {}).get(
            "candidate_column_audit",
            [],
        ),
        "candidate_pair_count_before_edge_validation": prepared.get(
            "candidate_discovery",
            {},
        ).get("candidate_pair_count_before_edge_validation", 0),
        "accepted_candidate_edge_count": prepared.get("candidate_discovery", {}).get(
            "accepted_candidate_edge_count",
            0,
        ),
        "rejected_candidate_edge_count": prepared.get("candidate_discovery", {}).get(
            "rejected_candidate_edge_count",
            0,
        ),
        "rejection_reason_counts": prepared.get("candidate_discovery", {}).get(
            "rejection_reason_counts",
            {},
        ),
        "official_validation_rows_used_for_candidate_discovery": prepared.get(
            "candidate_discovery",
            {},
        ).get("official_validation_rows_used_for_candidate_discovery", 0),
        "inner_validation_rows_used_for_candidate_discovery": prepared.get(
            "candidate_discovery",
            {},
        ).get("inner_validation_rows_used_for_candidate_discovery", 0),
        "test_rows_used_for_candidate_discovery": prepared.get("candidate_discovery", {}).get(
            "test_rows_used_for_candidate_discovery",
            0,
        ),
        "metric": metadata.get("primary_metric", ""),
        "metric_direction": metadata.get("metric_direction", ""),
        "smoke_only": bool(options.smoke_max_train_rows is not None),
        "smoke_max_train_rows": options.smoke_max_train_rows or "",
        "edge_screening_min_delta": options.edge_screening_min_delta,
        "edge_screening_min_positive_folds": _resolved_screening_min_positive_folds(options),
        "official_validation_was_used": False,
        "official_validation_was_used_for_candidate_selection": False,
        "official_validation_was_used_for_reliability": False,
        "official_validation_was_used_for_edge_evaluation": False,
        "test_split_accessed": False,
        "blockers": list(blockers),
    }
