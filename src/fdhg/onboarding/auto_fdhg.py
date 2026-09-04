from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import shutil
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from fdhg.compiler.fold_safe_fdhg import (
    CONTINUOUS_DISCRETIZATION_AUDIT_COLUMNS,
    discover_earliest_fold_candidate_edges,
    fit_afd_edges,
    fit_transform_fdhg_fold,
    materialize_ambiguity_features,
    point_in_time_asof_join,
    resolve_source_lookup_entity_key,
    _apply_fitted_edge_discretization_to_lookup as apply_fitted_edge_discretization_to_lookup,
    _edge_lhs_lookup_columns as auto_fdhg_edge_lhs_lookup_columns,
)
from fdhg.compiler.ambiguity import fitted_edge_to_audit_row, materialize_ambiguity_from_map
from fdhg.onboarding.relbench_v1 import _table_df
from fdhg.onboarding.auto_relbench import (
    AutoOnboardingOptions,
    _fit_model,
    _metric_score,
    _predict_model,
    materialize_feature_frame,
    prepare_auto_onboarding,
)


AUTO_FDHG_VERSION = "auto-fdhg-relbench-v1"
VARIANTS = ("dfs_fallback", "auto_only", "auto_plus_fdhg")
BASE_VARIANTS = ("dfs_fallback", "auto_only")


@dataclass(frozen=True)
class AutoFdhgOptions:
    selection_folds: int = 3
    feature_budget: int = 8
    min_delta: float = 0.0
    selection_decoder: str = "hist_gradient_boosting"
    max_fdhg_edges: int = 4
    max_selected_fdhg_edges: int | None = None
    max_relations: int = 3
    max_numeric_columns: int = 4
    max_categorical_columns: int = 4
    random_seed: int = 0
    enable_edge_screening: bool = True
    edge_screening_min_delta: float = 0.0
    edge_screening_min_positive_folds: int | None = None
    edge_screening_rule: str = "fixed_count"
    edge_screening_min_positive_fraction: float = 2 / 3
    edge_screening_max_relative_fold_degradation: float | None = None
    discover_fdhg_edges: bool = True
    edge_selection_strategy: str = "independent"
    enable_pairwise_rescue: bool = True
    continuous_fdhg_mode: str = "exclude"
    continuous_fdhg_bins: int = 8
    continuous_fdhg_min_effective_bins: int = 2
    fdhg_candidate_edges_file: Path | None = None
    force_final_variant: str | None = None


@dataclass(frozen=True)
class AutoFdhgReport:
    dataset: str
    task: str
    status: str
    output_dir: Path
    blockers: tuple[str, ...]
    dry_run: bool
    selected_variant: str | None = None
    metric: str | None = None
    metric_direction: str | None = None
    mean_scores: Mapping[str, float] | None = None
    official_validation_score: float | None = None
    dfs_features: int = 0
    dfs_declarations: int = 0
    dfs_model_columns: int = 0
    auto_features: int = 0
    fdhg_features: int = 0
    fdhg_declared_residual_features: int = 0
    fdhg_usable_residual_features_by_fold: Mapping[str, int] | None = None
    fdhg_final_refit_usable_features: int = 0
    accepted_edges: int = 0
    candidate_edges: int = 0
    screened_in_edges: int = 0
    screened_out_edges: int = 0
    expected_scans: int = 0
    expected_materializations: int = 0
    test_split_accessed: bool = False


def auto_fdhg_relbench(
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
    options: AutoFdhgOptions | None = None,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
) -> AutoFdhgReport:
    options = options or AutoFdhgOptions()
    output_dir = output_root / f"{dataset_name}_{task_name}"
    try:
        prepared = prepare_auto_fdhg(
            dataset_name=dataset_name,
            task_name=task_name,
            output_root=output_root,
            download=download,
            auto_output_root=auto_output_root,
            dfs_source_root=dfs_source_root,
            dfs_feature_config=dfs_feature_config,
            options=options,
            object_loader=object_loader,
            include_gate=write,
        )
    except Exception as exc:
        return AutoFdhgReport(
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
    manifest_path = output_dir / "manifest.json"
    identity = prepared["identity_hash"]
    if output_dir.exists() and not overwrite:
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("reuse_identity") == identity:
                return _report("reused", prepared, dry_run=False)
        raise FileExistsError(output_dir)
    staging = output_dir.parent / f"_{output_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _write_outputs(staging, prepared)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.rename(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _report("completed", prepared, dry_run=False)


def prepare_auto_fdhg(
    *,
    dataset_name: str,
    task_name: str,
    output_root: Path,
    download: bool,
    auto_output_root: Path,
    dfs_source_root: Path,
    dfs_feature_config: Path | None = None,
    options: AutoFdhgOptions,
    object_loader: Callable[[str, str, bool], tuple[Any, Any, str]] | None = None,
    include_gate: bool = False,
) -> dict[str, Any]:
    auto_options = AutoOnboardingOptions(
        selection_folds=options.selection_folds,
        feature_budget=options.feature_budget,
        min_delta=options.min_delta,
        selection_decoder=options.selection_decoder,
        max_relations=options.max_relations,
        max_numeric_columns=options.max_numeric_columns,
        max_categorical_columns=options.max_categorical_columns,
    )
    base = prepare_auto_onboarding(
        dataset_name=dataset_name,
        task_name=task_name,
        output_root=output_root,
        download=download,
        task_metadata_config=None,
        options=auto_options,
        object_loader=object_loader,
        include_selection=False,
    )
    metadata = dict(base["metadata"])
    table_dict = base["table_dict"]
    train_df = base["train_df"]
    val_df = base["validation_df"]
    split_plan = base["split_plan"]
    join_keys = resolve_join_keys(dataset_name, task_name, metadata)
    target_lookup_value_mapping = None

    if dataset_name.startswith("dbinfer-"):
        task_object = base.get("task")
        task_adapter = getattr(
            task_object,
            "_task_adapter",
            None,
        )
        entity_mapping = getattr(
            task_adapter,
            "entity_mapping",
            None,
        )

        if (
            isinstance(entity_mapping, Mapping)
            and entity_mapping
        ):
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

    auto_source = auto_output_root / f"{dataset_name}_{task_name}" / "selected_features.json"
    dfs_resolution = resolve_canonical_dfs_features(
        dataset_name=dataset_name,
        task_name=task_name,
        root=dfs_source_root,
        metadata=metadata,
        explicit_config=dfs_feature_config,
    )
    blockers: list[str] = []
    if not auto_source.exists():
        blockers.append(f"missing_auto_selected_features:{auto_source}")
        auto_features: list[dict[str, Any]] = []
        auto_hash = None
    else:
        auto_payload = json.loads(auto_source.read_text(encoding="utf-8"))
        auto_features = _extract_feature_declarations(auto_payload)
        auto_hash = _file_sha256(auto_source)
    external_auto_reference_only = (
        str(dataset_name).startswith("dbinfer-")
        and not dfs_resolution["features"]
    )

    if (
        not dfs_resolution["features"]
        and not external_auto_reference_only
    ):
        blockers.append(str(dfs_resolution["blocker"]))

    dfs_features = list(dfs_resolution["features"])
    if options.edge_selection_strategy not in {"independent", "greedy", "greedy_backward"}:
        raise ValueError(f"unsupported_edge_selection_strategy:{options.edge_selection_strategy}")
    if options.force_final_variant not in {None, "dfs_fallback", "auto_only", "auto_plus_fdhg"}:
        raise ValueError(
            f"unsupported_force_final_variant:{options.force_final_variant}"
        )
    if options.edge_screening_rule not in {"fixed_count", "positive_fraction", "pooled_oof"}:
        raise ValueError(f"unsupported_edge_screening_rule:{options.edge_screening_rule}")
    if not (0.0 < float(options.edge_screening_min_positive_fraction) <= 1.0):
        raise ValueError("edge_screening_min_positive_fraction_must_be_in_0_1")
    if (
        options.edge_screening_max_relative_fold_degradation is not None
        and float(options.edge_screening_max_relative_fold_degradation) < 0.0
    ):
        raise ValueError("edge_screening_max_relative_fold_degradation_must_be_non_negative")
    if options.continuous_fdhg_mode not in {"exclude", "quantile"}:
        raise ValueError(f"unsupported_continuous_fdhg_mode:{options.continuous_fdhg_mode}")
    if options.continuous_fdhg_bins < 1:
        raise ValueError("continuous_fdhg_bins_must_be_positive")
    if options.continuous_fdhg_min_effective_bins < 1:
        raise ValueError("continuous_fdhg_min_effective_bins_must_be_positive")
    if options.fdhg_candidate_edges_file is not None:
        replay = load_historical_candidate_edges(
            path=Path(options.fdhg_candidate_edges_file),
            table_dict=table_dict,
            max_edges=options.max_fdhg_edges,
        )
        accepted_edges = replay["accepted_edges"]
        rejected_edges = []
        candidate_discovery = replay["provenance"]
    elif options.discover_fdhg_edges:
        discovery = discover_earliest_fold_candidate_edges(
            prepared=base,
            edge_budget=options.max_fdhg_edges,
            continuous_fdhg_mode=options.continuous_fdhg_mode,
            continuous_fdhg_bins=options.continuous_fdhg_bins,
            continuous_fdhg_min_effective_bins=options.continuous_fdhg_min_effective_bins,
            target_lookup_value_mapping=(
                target_lookup_value_mapping
            ),
        )
        accepted_edges = discovery["accepted_edges"]
        rejected_edges = discovery["rejected_edges"]
        candidate_discovery = dict(discovery["provenance"])
    else:
        accepted_edges, rejected_edges = [], []
        candidate_discovery = {
            "candidate_discovery_protocol": "disabled",
            "strategy_name": options.edge_selection_strategy,
            "candidate_count_before_budget": 0,
            "candidate_count_after_budget": 0,
            "candidate_rediscovery_performed": False,
            "candidate_column_audit": [],
            "rejection_reason_counts": {},
            "ordered_candidate_edge_ids": [],
            "inner_validation_rows_used_for_candidate_discovery": 0,
            "official_validation_rows_used_for_candidate_discovery": 0,
            "test_rows_used_for_candidate_discovery": 0,
        }
    candidate_discovery["edge_selection_strategy"] = options.edge_selection_strategy
    candidate_discovery["strategy_name"] = options.edge_selection_strategy
    fold_metadata = selection_fold_metadata(
        split_plan=split_plan,
        requested_selection_folds=options.selection_folds,
    )
    workload = {
        "expected_child_relation_scans": int(
            len(split_plan["folds"]) * (
                _relation_count(dfs_features)
                + _relation_count(auto_features)
                + len(accepted_edges)
            )
        ),
        "expected_matrix_materializations": int(len(split_plan["folds"]) * 3),
    }
    gate: dict[str, Any]
    final: dict[str, Any]
    if include_gate and not blockers:
        gate = evaluate_joint_gate(
            dataset_name=dataset_name,
            task_name=task_name,
            train_targets=train_df,
            table_dict=table_dict,
            metadata=metadata,
            split_plan=split_plan,
            dfs_features=dfs_features,
            auto_features=auto_features,
            fdhg_edges=accepted_edges,
            join_keys=join_keys,
            options=options,
            fold_metadata=fold_metadata,
            target_lookup_value_mapping=(
                target_lookup_value_mapping
            ),
        )
        final_selected_variant = (
            options.force_final_variant
            if options.force_final_variant is not None
            else gate["selected_variant"]
        )
        final = final_refit_and_evaluate(
            dataset_name=dataset_name,
            task_name=task_name,
            selected_variant=final_selected_variant,
            train_targets=train_df,
            validation_targets=val_df,
            table_dict=table_dict,
            metadata=metadata,
            dfs_features=dfs_features,
            auto_features=auto_features,
            fdhg_edges=gate.get("screened_fdhg_edges", accepted_edges),
            join_keys=join_keys,
            options=options,
            target_lookup_value_mapping=(
                target_lookup_value_mapping
            ),
        )
    else:
        gate = {
            "fold_metrics": [],
            "gate_trials": [],
            "selected_variant": None,
            "mean_scores": {},
            "selected_fdhg_edges": [],
            "rejected_fdhg_edges": [],
            "fdhg_feature_provenance": [],
            "fdhg_fold_feature_audit": [],
            "fdhg_target_lookup_audit": [],
            "fdhg_usable_residual_features_by_fold": {},
            "fdhg_declared_residual_features": 0,
            "screened_fdhg_edges": [],
            "edge_screening": [],
            "edge_screening_fold_metrics": [],
            "edge_screening_enabled": options.enable_edge_screening,
            "edge_screening_min_delta": options.edge_screening_min_delta,
            "edge_screening_rule": options.edge_screening_rule,
            "edge_screening_min_positive_fraction": options.edge_screening_min_positive_fraction,
            "edge_screening_min_positive_folds": resolve_edge_screening_min_positive_folds(
                options,
                effective_selection_folds=fold_metadata["effective_selection_folds"],
            ),
            "edge_screening_max_relative_fold_degradation": options.edge_screening_max_relative_fold_degradation,
            "edge_selection_strategy": options.edge_selection_strategy,
            "max_selected_fdhg_edges": options.max_selected_fdhg_edges,
            "ordered_candidate_edge_ids": [],
            "independent_screened_in_edge_ids": [],
            "strategy_selected_edge_ids": [],
            "final_combined_block_edge_ids": [],
            "edge_selection_trace": [],
            "edge_selection_step_count": 0,
            "edge_selection_stop_reason": "not_evaluated_without_gate",
            "candidate_discovery": candidate_discovery,
            "pair_screening": [],
            "pair_screening_fold_metrics": [],
            "continuous_discretization_audit": [],
            "continuous_discretization_boundaries": {},
            "pairwise_rescue_used": False,
            "selected_initial_pair": "",
            "pairwise_rescue_reason": "not_evaluated_without_gate",
            "fdhg_screening_fallback": "",
        }
        final = {
            "official_validation_score": None,
            "official_validation_metrics": {},
            "official_validation_predictions": pd.DataFrame(),
            "fdhg_final_refit_usable_features": 0,
            "fdhg_final_refit_skipped_reason": "",
            "fdhg_final_feature_audit": [],
            "continuous_discretization_audit": [],
            "continuous_discretization_boundaries": {},
        }
    identity_payload = {
        "version": AUTO_FDHG_VERSION,
        "dataset": dataset_name,
        "task": task_name,
        "selection_folds": options.selection_folds,
        "requested_selection_folds": options.selection_folds,
        "effective_selection_folds": fold_metadata["effective_selection_folds"],
        "effective_fold_ids": fold_metadata["effective_fold_ids"],
        "feature_budget": options.feature_budget,
        "min_delta": options.min_delta,
        "max_fdhg_edges": options.max_fdhg_edges,
        "max_selected_fdhg_edges": options.max_selected_fdhg_edges,
        "enable_edge_screening": options.enable_edge_screening,
        "edge_screening_min_delta": options.edge_screening_min_delta,
        "edge_screening_min_positive_folds": options.edge_screening_min_positive_folds,
        "edge_screening_rule": options.edge_screening_rule,
        "edge_screening_min_positive_fraction": options.edge_screening_min_positive_fraction,
        "edge_screening_max_relative_fold_degradation": options.edge_screening_max_relative_fold_degradation,
        "edge_selection_strategy": options.edge_selection_strategy,
        "continuous_fdhg_mode": options.continuous_fdhg_mode,
        "continuous_fdhg_bins": options.continuous_fdhg_bins,
        "continuous_fdhg_min_effective_bins": options.continuous_fdhg_min_effective_bins,
        "fdhg_candidate_edges_file": str(options.fdhg_candidate_edges_file or ""),
        "fdhg_candidate_edges_file_sha256": (
            _file_sha256(Path(options.fdhg_candidate_edges_file))
            if options.fdhg_candidate_edges_file is not None
            else ""
        ),
        "dfs_source": dfs_resolution["provenance"],
        "auto_source": str(auto_source),
        "auto_hash": auto_hash,
        "decoder": decoder_config(options),
    }
    identity_hash = _text_sha256(json.dumps(identity_payload, sort_keys=True, default=str))
    run_counters = dict(gate.get("run_counters", {}))
    for key, value in final.get("run_counters", {}).items():
        run_counters[key] = int(run_counters.get(key, 0)) + int(value)
    if "official_validation_decoder_fit_count" in final.get("run_counters", {}):
        run_counters["decoder_fit_count"] = int(run_counters.get("decoder_fit_count", 0)) + int(
            final["run_counters"].get("official_validation_decoder_fit_count", 0)
        )
        run_counters["decoder_prediction_count"] = int(run_counters.get("decoder_prediction_count", 0)) + int(
            final["run_counters"].get("official_validation_prediction_count", 0)
        )
    manifest = {
        "dataset": dataset_name,
        "task": task_name,
        "relbench_version": base["relbench_version"],
        "implementation_version": AUTO_FDHG_VERSION,
        "reuse_identity": identity_hash,
        "task_metadata": metadata,
        "test_split_accessed": False,
        "official_validation_was_used_for_selection": False,
        "join_keys": join_keys,
        "metric": metadata["primary_metric"],
        "metric_direction": metadata["metric_direction"],
        "selection_folds": options.selection_folds,
        "selection_folds_semantics": "requested maximum inner temporal folds; observed folds are effective_selection_folds",
        "requested_selection_folds": options.selection_folds,
        "effective_selection_folds": fold_metadata["effective_selection_folds"],
        "effective_fold_ids": fold_metadata["effective_fold_ids"],
        "temporal_fold_quality": fold_metadata["fold_quality"],
        "manifest_warnings": fold_metadata["warnings"],
        "min_delta": options.min_delta,
        "decoder_configuration": decoder_config(options),
        "canonical_dfs_source": dfs_resolution["provenance"],
        "dfs_declaration_count": dfs_resolution["provenance"].get("declaration_count", len(dfs_features)),
        "dfs_model_column_count": dfs_resolution["provenance"].get("model_column_count", len(_declared_model_columns(dfs_features))),
        "auto_selected_feature_source": str(auto_source),
        "accepted_fdhg_edges": accepted_edges,
        "rejected_fdhg_edges": rejected_edges,
        "edge_screening_enabled": gate.get("edge_screening_enabled", options.enable_edge_screening),
        "edge_screening_min_delta": gate.get("edge_screening_min_delta", options.edge_screening_min_delta),
        "edge_screening_rule": gate.get("edge_screening_rule", options.edge_screening_rule),
        "edge_screening_min_positive_fraction": gate.get(
            "edge_screening_min_positive_fraction",
            options.edge_screening_min_positive_fraction,
        ),
        "edge_screening_max_relative_fold_degradation": gate.get(
            "edge_screening_max_relative_fold_degradation",
            options.edge_screening_max_relative_fold_degradation,
        ),
        "edge_screening_min_positive_folds": gate.get(
            "edge_screening_min_positive_folds",
            resolve_edge_screening_min_positive_folds(
                options,
                effective_selection_folds=fold_metadata["effective_selection_folds"],
            ),
        ),
        "edge_selection_strategy": gate.get("edge_selection_strategy", options.edge_selection_strategy),
        "max_selected_fdhg_edges": gate.get("max_selected_fdhg_edges", options.max_selected_fdhg_edges),
        "continuous_fdhg_mode": options.continuous_fdhg_mode,
        "continuous_fdhg_bins": options.continuous_fdhg_bins,
        "continuous_fdhg_min_effective_bins": options.continuous_fdhg_min_effective_bins,
        "candidate_discovery": candidate_discovery,
        "candidate_discovery_protocol": candidate_discovery.get("candidate_discovery_protocol", ""),
        "candidate_edges_file": candidate_discovery.get("candidate_edges_file", ""),
        "candidate_edges_file_sha256": candidate_discovery.get("candidate_edges_file_sha256", ""),
        "loaded_candidate_edge_count": candidate_discovery.get("loaded_candidate_edge_count", ""),
        "candidate_rediscovery_performed": candidate_discovery.get(
            "candidate_rediscovery_performed",
            bool(options.discover_fdhg_edges and options.fdhg_candidate_edges_file is None),
        ),
        "candidate_discovery_fold": candidate_discovery.get("candidate_discovery_fold", ""),
        "candidate_discovery_fit_horizon": candidate_discovery.get("candidate_discovery_fit_horizon", ""),
        "candidate_count_before_budget": candidate_discovery.get("candidate_count_before_budget", 0),
        "candidate_count_after_budget": candidate_discovery.get("candidate_count_after_budget", len(accepted_edges)),
        "candidate_count_after_candidate_budget": candidate_discovery.get("candidate_count_after_budget", len(accepted_edges)),
        "candidate_column_audit": candidate_discovery.get("candidate_column_audit", []),
        "continuous_discretization_audit_count": len(gate.get("continuous_discretization_audit", []))
        + len(final.get("continuous_discretization_audit", [])),
        "candidate_rejection_reason_counts": candidate_discovery.get("rejection_reason_counts", {}),
        "ordered_candidate_edge_ids": candidate_discovery.get("ordered_candidate_edge_ids", []),
        "candidate_fdhg_edge_count": len(accepted_edges),
        "fdhg_edge_fold_instance_count": len(gate.get("edge_screening_fold_metrics", [])),
        "screened_in_fdhg_edge_count": len(gate.get("screened_fdhg_edges", [])),
        "screened_out_fdhg_edge_count": max(0, len(accepted_edges) - len(gate.get("screened_fdhg_edges", []))),
        "independent_screened_in_edge_ids": gate.get("independent_screened_in_edge_ids", []),
        "strategy_selected_edge_ids": gate.get("strategy_selected_edge_ids", []),
        "final_combined_block_edge_ids": gate.get("final_combined_block_edge_ids", []),
        "screened_in_edge_ids": gate.get("strategy_selected_edge_ids", [
            str(row.get("edge_id", "")) for row in gate.get("screened_fdhg_edges", [])
        ]),
        "selected_screened_edge_count": len(gate.get("screened_fdhg_edges", [])),
        "strategy_selected_edge_count": len(gate.get("screened_fdhg_edges", [])),
        "edge_selection_step_count": gate.get("edge_selection_step_count", 0),
        "edge_selection_stop_reason": gate.get("edge_selection_stop_reason", ""),
        "fdhg_screening_fallback": gate.get("fdhg_screening_fallback", ""),
        "pairwise_rescue_used": gate.get("pairwise_rescue_used", False),
        "selected_initial_pair": gate.get("selected_initial_pair", ""),
        "pairwise_rescue_reason": gate.get("pairwise_rescue_reason", ""),
        "fold_accepted_fdhg_edges": gate.get("selected_fdhg_edges", []),
        "fold_rejected_fdhg_edges": gate.get("rejected_fdhg_edges", []),
        "fold_boundaries": _public_folds(split_plan),
        "workload": workload,
        "selected_variant": gate["selected_variant"],
        "gate_selected_variant": gate["selected_variant"],
        "final_evaluated_variant": (
            options.force_final_variant
            if options.force_final_variant is not None
            else gate["selected_variant"]
        ),
        "forced_final_evaluation": bool(
            options.force_final_variant is not None
        ),
        "mean_scores": gate["mean_scores"],
        "fdhg_declared_residual_features": gate.get("fdhg_declared_residual_features", 0),
        "fdhg_usable_residual_features_by_fold": gate.get("fdhg_usable_residual_features_by_fold", {}),
        "fdhg_usable_residual_feature_count_max": max(
            [int(v) for v in gate.get("fdhg_usable_residual_features_by_fold", {}).values()] or [0]
        ),
        "fdhg_final_refit_usable_features": final.get("fdhg_final_refit_usable_features", 0),
        "fdhg_final_refit_skipped_reason": final.get("fdhg_final_refit_skipped_reason", ""),
        "run_counters": run_counters,
        **run_counters,
        "official_validation_score": final["official_validation_score"],
        "blockers": blockers,
    }
    return {
        **base,
        "output_dir": output_root / f"{dataset_name}_{task_name}",
        "options": options,
        "join_keys": join_keys,
        "dfs_features": dfs_features,
        "dfs_provenance": dfs_resolution["provenance"],
        "dfs_declaration_count": dfs_resolution["provenance"].get("declaration_count", len(dfs_features)),
        "dfs_model_column_count": dfs_resolution["provenance"].get("model_column_count", len(_declared_model_columns(dfs_features))),
        "auto_features": auto_features,
        "auto_source": auto_source,
        "accepted_fdhg_edges": accepted_edges,
        "rejected_fdhg_edges": rejected_edges,
        "candidate_discovery": candidate_discovery,
        "gate": gate,
        "final": final,
        "manifest": manifest,
        "identity_hash": identity_hash,
        "blockers": tuple(blockers),
        "workload": workload,
    }



def _select_best_passing_greedy_trial(
    trials,
    *,
    trial_forward_passes,
    trial_selection_gain,
):
    """Return the highest-gain gate-passing Greedy trial.

    Candidate order is used as the deterministic tie-break. Returning None
    means that no remaining candidate satisfies the forward screening gate.
    """
    passing_trials = [
        item
        for item in trials
        if trial_forward_passes(item[2])
    ]
    if not passing_trials:
        return None

    return min(
        passing_trials,
        key=lambda item: (
            -trial_selection_gain(item[2])
            if np.isfinite(trial_selection_gain(item[2]))
            else math.inf,
            item[0],
        ),
    )

def evaluate_joint_gate(
    *,
    dataset_name: str,
    task_name: str,
    train_targets: pd.DataFrame,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    split_plan: Mapping[str, Any],
    dfs_features: Sequence[Mapping[str, Any]],
    auto_features: Sequence[Mapping[str, Any]],
    fdhg_edges: Sequence[Mapping[str, Any]],
    join_keys: Sequence[str],
    options: AutoFdhgOptions,
    fold_metadata: Mapping[str, Any] | None = None,
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    fold_metadata = fold_metadata or selection_fold_metadata(
        split_plan=split_plan,
        requested_selection_folds=options.selection_folds,
    )
    requested_selection_folds = int(fold_metadata["requested_selection_folds"])
    effective_selection_folds = int(fold_metadata["effective_selection_folds"])
    effective_fold_ids = list(fold_metadata["effective_fold_ids"])

    # External DBInfer generalization evaluates FDHG as an augmentation
    # to the Auto baseline. These tasks do not have a canonical DFS
    # artifact, so keep the same Auto/FDHG screening logic while
    # omitting DFS-only materialization and final four-way gate logic.
    external_auto_reference_only = (
        str(dataset_name).startswith("dbinfer-")
        and not dfs_features
    )
    run_counters: dict[str, Any] = {
        "decoder_fit_count": 0,
        "dfs_decoder_fit_count": 0,
        "auto_decoder_fit_count": 0,
        "single_edge_decoder_fit_count": 0,
        "combined_decoder_fit_count": 0,
        "official_validation_decoder_fit_count": 0,
        "decoder_prediction_count": 0,
        "final_gate_reused_scores": True,
        "fdhg_mapping_fit_count": 0,
        "fdhg_materialization_count": 0,
        "source_scan_count": 0,
        "feature_materialization_count": 0,
        "target_lookup_count": 0,
    }
    timing_rows: list[dict[str, Any]] = []

    def timed(stage: str, fn, **extra):
        started = time.perf_counter()
        try:
            return fn()
        finally:
            timing_rows.append({
                "stage": stage,
                "elapsed_seconds": time.perf_counter() - started,
                **extra,
            })

    fold_metrics: list[dict[str, Any]] = []
    edge_screening_rows: list[dict[str, Any]] = []
    edge_screening_fold_rows: list[dict[str, Any]] = []
    pair_screening_rows: list[dict[str, Any]] = []
    pair_screening_fold_rows: list[dict[str, Any]] = []
    edge_selection_trace_rows: list[dict[str, Any]] = []
    fdhg_feature_provenance: list[dict[str, Any]] = []
    accepted_edge_audit: list[dict[str, Any]] = []
    rejected_edge_audit: list[dict[str, Any]] = []
    fdhg_fold_feature_audit: list[dict[str, Any]] = []
    fdhg_target_lookup_audit: list[dict[str, Any]] = []
    continuous_discretization_audit: list[dict[str, Any]] = []
    continuous_discretization_boundaries: dict[str, Any] = {}
    usable_by_fold: dict[str, int] = {}
    declared_residual_features: set[str] = set()
    scores_by_variant: dict[str, list[float]] = {
        variant: []
        for variant in (
            ("auto_only",)
            if external_auto_reference_only
            else BASE_VARIANTS
        )
    }
    fold_scores: dict[tuple[int, str], float] = {}
    fold_contexts: list[dict[str, Any]] = []
    auto_feature_cols_by_fold: dict[int, list[str]] = {}
    single_edge_cache: dict[tuple[str, int], dict[str, Any]] = {}
    lookup_source_columns_by_table: dict[str, list[str]] = {}
    for edge in fdhg_edges:
        table = str(edge.get("source_table", ""))
        for lhs in auto_fdhg_edge_lhs_lookup_columns(edge):
            lookup_source_columns_by_table.setdefault(table, [])
            if str(lhs) not in lookup_source_columns_by_table[table]:
                lookup_source_columns_by_table[table].append(str(lhs))
    pit_lookup_cache: dict[tuple[int, str, str], tuple[pd.DataFrame, dict[str, Any]]] = {}

    def evaluate_candidate_block(
        *,
        stage: str,
        candidate_edges: Sequence[Mapping[str, Any]],
        baseline_scores_by_fold: Mapping[int, float],
        baseline_edges: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        block_scores: dict[int, float] = {}
        fold_gains: dict[int, float] = {}
        usable_counts: dict[int, int] = {}
        future_violations = 0
        materialization_failed = False
        error_reason = ""
        edge_ids = tuple(str(edge.get("edge_id", "")) for edge in candidate_edges)
        baseline_edge_ids = tuple(
            str(edge.get("edge_id", "")) for edge in (baseline_edges or [])
        )
        pooled_labels: list[Any] = []
        pooled_baseline_predictions: list[Any] = []
        pooled_candidate_predictions: list[Any] = []
        fold_baseline_scores_for_trial: list[float] = []
        fold_ids_for_trial: list[int] = []

        def score_edge_block(
            *,
            context: Mapping[str, Any],
            fold_id: int,
            edges: Sequence[Mapping[str, Any]],
            edge_ids_for_block: tuple[str, ...],
            stage_name: str,
            need_predictions: bool,
        ) -> dict[str, Any]:
            if not edges:
                run_counters["decoder_fit_count"] += 1
                run_counters["auto_decoder_fit_count"] += 1
                run_counters["decoder_prediction_count"] += 1
                scorer = score_matrix_with_predictions if need_predictions else score_matrix
                result = timed(
                    f"score_{stage_name}_auto_baseline",
                    lambda context=context, fold_id=fold_id, scorer=scorer: scorer(
                        train_x=context["auto_train"],
                        val_x=context["auto_val"],
                        train_y=context["inner_train"][metadata["label_col"]],
                        val_y=context["inner_val"][metadata["label_col"]],
                        feature_cols=auto_feature_cols_by_fold[fold_id],
                        metadata=metadata,
                        options=options,
                    ),
                    fold=fold_id,
                    edge_count=0,
                    edge_ids="",
                )
                if need_predictions:
                    return {
                        "score": float(result["score"]),
                        "prediction": result["prediction"],
                        "usable_count": len(auto_feature_cols_by_fold[fold_id]),
                        "future_lookup_violation_count": 0,
                    }
                return {
                    "score": float(result),
                    "prediction": None,
                    "usable_count": len(auto_feature_cols_by_fold[fold_id]),
                    "future_lookup_violation_count": 0,
                }

            cached_single = (
                single_edge_cache.get((edge_ids_for_block[0], fold_id))
                if len(edge_ids_for_block) == 1
                else None
            )
            cached_block = [
                single_edge_cache.get((edge_id, fold_id))
                for edge_id in edge_ids_for_block
            ]
            if cached_single is not None:
                fdhg = cached_single["fdhg"]
            elif cached_block and all(item is not None for item in cached_block):
                fdhg = compose_cached_single_edge_block(
                    cached_items=[item for item in cached_block if item is not None],
                    join_keys=join_keys,
                    metadata=metadata,
                )
            else:
                run_counters["fdhg_mapping_fit_count"] += 1
                run_counters["fdhg_materialization_count"] += 2
                run_counters["target_lookup_count"] += 2
                fdhg = timed(
                    stage_name,
                    lambda context=context, fold_id=fold_id, edges=edges: fit_transform_fdhg_fold(
                        inner_train_rows=context["inner_train"],
                        inner_validation_rows=context["inner_val"],
                        source_tables=table_dict,
                        schema=None,
                        task_metadata=metadata,
                        candidate_edges=edges,
                        max_edges=len(edges),
                        fold=fold_id,
                        continuous_fdhg_mode=options.continuous_fdhg_mode,
                        continuous_fdhg_bins=options.continuous_fdhg_bins,
                        continuous_fdhg_min_effective_bins=options.continuous_fdhg_min_effective_bins,
                        dataset=dataset_name,
                        task=task_name,
                        fit_split="fold_train",
                        **(
                            {
                                "target_lookup_value_mapping":
                                    target_lookup_value_mapping
                            }
                            if target_lookup_value_mapping
                            is not None
                            else {}
                        ),
                    ),
                    fold=fold_id,
                    edge_count=len(edges),
                    edge_ids="|".join(edge_ids_for_block),
                )
            fdhg_cols = feature_columns(fdhg["train_x"], join_keys, metadata)
            _collect_continuous_discretization(
                audit_rows=continuous_discretization_audit,
                boundaries=continuous_discretization_boundaries,
                materialization=fdhg,
                fold=fold_id,
                stage=stage_name,
            )
            _audit_rows, usable_cols = audit_residual_columns(
                frame=fdhg["train_x"],
                feature_cols=fdhg_cols,
                fold=fold_id,
                provenance=fdhg["feature_provenance"],
            )
            usable_count = len(usable_cols)
            future_count = sum(
                int(row.get("future_lookup_violation_count") or 0)
                for row in fdhg.get("target_lookup_audit", [])
            )
            if usable_count <= 0 or future_count != 0:
                return {
                    "score": math.nan,
                    "prediction": None,
                    "usable_count": usable_count,
                    "future_lookup_violation_count": future_count,
                }
            block_train = align_feature_blocks(
                target_rows=context["inner_train"],
                blocks=[("auto", context["auto_train"]), ("fdhg", fdhg["train_x"])],
                join_keys=join_keys,
                metadata=metadata,
            )
            block_val = align_feature_blocks(
                target_rows=context["inner_val"],
                blocks=[("auto", context["auto_val"]), ("fdhg", fdhg["validation_x"])],
                join_keys=join_keys,
                metadata=metadata,
            )
            run_counters["decoder_fit_count"] += 1
            run_counters["combined_decoder_fit_count"] += 1
            run_counters["decoder_prediction_count"] += 1
            scorer = score_matrix_with_predictions if need_predictions else score_matrix
            result = timed(
                f"score_{stage_name}",
                lambda block_train=block_train, block_val=block_val, usable_cols=usable_cols, context=context, fold_id=fold_id, scorer=scorer: scorer(
                    train_x=block_train,
                    val_x=block_val,
                    train_y=context["inner_train"][metadata["label_col"]],
                    val_y=context["inner_val"][metadata["label_col"]],
                    feature_cols=[*auto_feature_cols_by_fold[fold_id], *usable_cols],
                    metadata=metadata,
                    options=options,
                ),
                fold=fold_id,
                edge_count=len(edges),
                edge_ids="|".join(edge_ids_for_block),
            )
            if need_predictions:
                return {
                    "score": float(result["score"]),
                    "prediction": result["prediction"],
                    "usable_count": usable_count,
                    "future_lookup_violation_count": future_count,
                }
            return {
                "score": float(result),
                "prediction": None,
                "usable_count": usable_count,
                "future_lookup_violation_count": future_count,
            }

        for context in fold_contexts:
            fold_id = int(context["fold"])
            try:
                if options.edge_screening_rule == "pooled_oof":
                    baseline_result = score_edge_block(
                        context=context,
                        fold_id=fold_id,
                        edges=baseline_edges or [],
                        edge_ids_for_block=baseline_edge_ids,
                        stage_name=f"{stage}_baseline",
                        need_predictions=True,
                    )
                    candidate_result = score_edge_block(
                        context=context,
                        fold_id=fold_id,
                        edges=candidate_edges,
                        edge_ids_for_block=edge_ids,
                        stage_name=stage,
                        need_predictions=True,
                    )
                    baseline_score = float(baseline_result["score"])
                    score = float(candidate_result["score"])
                    usable_count = int(candidate_result["usable_count"])
                    future_count = int(baseline_result["future_lookup_violation_count"]) + int(
                        candidate_result["future_lookup_violation_count"]
                    )
                    if (
                        np.isfinite(baseline_score)
                        and np.isfinite(score)
                        and baseline_result["prediction"] is not None
                        and candidate_result["prediction"] is not None
                    ):
                        pooled_labels.append(context["inner_val"][metadata["label_col"]].reset_index(drop=True))
                        pooled_baseline_predictions.append(baseline_result["prediction"])
                        pooled_candidate_predictions.append(candidate_result["prediction"])
                    gain = metric_improvement(
                        candidate=score,
                        reference=baseline_score,
                        direction=metadata["metric_direction"],
                    ) if np.isfinite(score) and np.isfinite(baseline_score) else math.nan
                else:
                    cached_single = (
                        single_edge_cache.get((edge_ids[0], fold_id))
                        if len(edge_ids) == 1
                        else None
                    )
                    if cached_single is not None:
                        score = float(cached_single["score"])
                        usable_count = int(cached_single["usable_count"])
                        future_count = int(cached_single["future_lookup_violation_count"])
                    else:
                        block_result = score_edge_block(
                            context=context,
                            fold_id=fold_id,
                            edges=candidate_edges,
                            edge_ids_for_block=edge_ids,
                            stage_name=stage,
                            need_predictions=False,
                        )
                        score = float(block_result["score"])
                        usable_count = int(block_result["usable_count"])
                        future_count = int(block_result["future_lookup_violation_count"])
                    gain = metric_improvement(
                        candidate=score,
                        reference=float(baseline_scores_by_fold[fold_id]),
                        direction=metadata["metric_direction"],
                    ) if np.isfinite(score) else math.nan
                future_violations += int(future_count)
                usable_counts[fold_id] = int(usable_count)
                block_scores[fold_id] = float(score)
                fold_gains[fold_id] = float(gain)
                fold_baseline_scores_for_trial.append(
                    float(baseline_scores_by_fold[fold_id])
                    if options.edge_screening_rule != "pooled_oof"
                    else float(baseline_score)
                )
                fold_ids_for_trial.append(fold_id)
            except Exception as exc:
                materialization_failed = True
                error_reason = str(exc)
                usable_counts[fold_id] = 0
                block_scores[fold_id] = math.nan
                fold_gains[fold_id] = math.nan
                fold_baseline_scores_for_trial.append(float(baseline_scores_by_fold[fold_id]))
                fold_ids_for_trial.append(fold_id)
        finite_gains = [float(value) for value in fold_gains.values() if np.isfinite(value)]
        aggregate_baseline_score = None
        aggregate_candidate_score = None
        aggregate_gain = None
        if options.edge_screening_rule == "pooled_oof" and pooled_labels:
            y_true = pd.concat([pd.Series(values) for values in pooled_labels], ignore_index=True)
            baseline_pred = _concat_predictions(pooled_baseline_predictions)
            candidate_pred = _concat_predictions(pooled_candidate_predictions)
            aggregate_baseline_score = _metric_score(
                y_true,
                baseline_pred,
                metric=metadata["primary_metric"],
                problem_type=metadata["problem_type"],
            )
            aggregate_candidate_score = _metric_score(
                y_true,
                candidate_pred,
                metric=metadata["primary_metric"],
                problem_type=metadata["problem_type"],
            )
            aggregate_gain = metric_improvement(
                candidate=aggregate_candidate_score,
                reference=aggregate_baseline_score,
                direction=metadata["metric_direction"],
            )
        summary = summarize_edge_screening(
            gains=list(fold_gains.values()),
            usable_feature_counts=list(usable_counts.values()),
            future_lookup_violation_count=future_violations,
            min_delta=options.edge_screening_min_delta,
            min_positive_folds=min_positive_folds,
            screening_rule=options.edge_screening_rule,
            aggregate_gain=aggregate_gain,
            aggregate_auto_score=aggregate_baseline_score,
            aggregate_candidate_score=aggregate_candidate_score,
            fold_auto_scores=fold_baseline_scores_for_trial,
            fold_ids=fold_ids_for_trial,
            max_relative_fold_degradation=options.edge_screening_max_relative_fold_degradation,
            materialization_failed=materialization_failed,
        )
        return {
            "scores_by_fold": block_scores,
            "gains_by_fold": fold_gains,
            "mean_score": float(np.mean([value for value in block_scores.values() if np.isfinite(value)]))
            if any(np.isfinite(value) for value in block_scores.values())
            else math.nan,
            "mean_gain": float(np.mean(finite_gains)) if finite_gains else math.nan,
            "aggregate_baseline_score": summary["aggregate_auto_score"],
            "aggregate_auto_score": summary["aggregate_auto_score"],
            "aggregate_candidate_score": summary["aggregate_candidate_score"],
            "aggregate_gain": summary["aggregate_gain"],
            "positive_fold_count": summary["positive_fold_count"],
            "total_fold_count": len(finite_gains),
            "worst_fold_relative_degradation": summary["worst_fold_relative_degradation"],
            "worst_fold_relative_degradation_fold": summary["worst_fold_relative_degradation_fold"],
            "screening_status": summary["screening_status"],
            "rejection_reason": summary["rejection_reason"],
            "usable_feature_count_min": min(usable_counts.values()) if usable_counts else 0,
            "usable_feature_count_max": max(usable_counts.values()) if usable_counts else 0,
            "future_lookup_violation_count": future_violations,
            "materialization_failed": materialization_failed,
            "materialization_error": error_reason,
        }

    def append_selection_trace(
        *,
        step: int,
        candidate_edge_id: str,
        candidate_edge_ids: Sequence[str],
        selected_after: Sequence[str],
        baseline_scores_by_fold: Mapping[int, float],
        trial: Mapping[str, Any],
        decision: str,
        stop_reason: str = "",
    ) -> None:
        baseline_values = [float(value) for value in baseline_scores_by_fold.values() if np.isfinite(value)]
        edge_selection_trace_rows.append({
            "dataset": dataset_name,
            "task": task_name,
            "strategy": options.edge_selection_strategy,
            "requested_selection_folds": requested_selection_folds,
            "effective_selection_folds": effective_selection_folds,
            "effective_fold_ids": "|".join(str(value) for value in effective_fold_ids),
            "edge_screening_rule": options.edge_screening_rule,
            "edge_screening_min_positive_folds": min_positive_folds,
            "edge_screening_min_positive_fraction": options.edge_screening_min_positive_fraction,
            "step": step,
            "candidate_edge_id": candidate_edge_id,
            "candidate_edge_ids": "|".join(candidate_edge_ids),
            "selected_edge_ids_after_step": "|".join(selected_after),
            "baseline_mean_score": float(np.mean(baseline_values)) if baseline_values else math.nan,
            "trial_mean_score": trial.get("mean_score", math.nan),
            "aggregate_baseline_score": trial.get("aggregate_baseline_score", math.nan),
            "aggregate_candidate_score": trial.get("aggregate_candidate_score", math.nan),
            "aggregate_gain": trial.get("aggregate_gain", math.nan),
            "fold_incremental_gains": json.dumps(
                {str(k): v for k, v in sorted(trial.get("gains_by_fold", {}).items())},
                sort_keys=True,
                default=str,
            ),
            "mean_incremental_gain": trial.get("mean_gain", math.nan),
            "positive_fold_count": trial.get("positive_fold_count", 0),
            "decision": decision,
            "stop_reason": stop_reason,
        })
    for fold in split_plan["folds"]:
        fold_id = int(fold["fold"])
        inner_train = train_targets.loc[fold["train_indices"]].reset_index(drop=True)
        inner_val = train_targets.loc[fold["validation_indices"]].reset_index(drop=True)
        dfs_train = None
        dfs_val = None

        if not external_auto_reference_only:
            run_counters["source_scan_count"] += 1
            run_counters["feature_materialization_count"] += 1
            dfs_train = timed(
                "materialize_dfs_fold_pair",
                lambda: materialize_declared_feature_frame_pair(
                    inner_train,
                    inner_val,
                    table_dict=table_dict,
                    features=dfs_features,
                    entity_key=metadata["entity_key"],
                    target_time_col=metadata["target_time_col"],
                ),
                fold=fold_id,
            )
            dfs_train, dfs_val = dfs_train

        run_counters["source_scan_count"] += 1
        run_counters["feature_materialization_count"] += 1
        auto_train = timed(
            "materialize_auto_fold_pair",
            lambda: materialize_declared_feature_frame_pair(
                inner_train,
                inner_val,
                table_dict=table_dict,
                features=auto_features,
                entity_key=metadata["entity_key"],
                target_time_col=metadata["target_time_col"],
            ),
            fold=fold_id,
        )
        auto_train, auto_val = auto_train
        matrices = {
            "auto_only": (
                auto_train,
                auto_val,
                _declared_model_columns(auto_features),
            ),
        }

        if not external_auto_reference_only:
            matrices["dfs_fallback"] = (
                dfs_train,
                dfs_val,
                _declared_model_columns(dfs_features),
            )

        base_variants_for_gate = (
            ("auto_only",)
            if external_auto_reference_only
            else BASE_VARIANTS
        )

        for variant in base_variants_for_gate:
            train_x, val_x, cols = matrices[variant]
            run_counters["decoder_fit_count"] += 1
            run_counters["decoder_prediction_count"] += 1
            if variant == "dfs_fallback":
                run_counters["dfs_decoder_fit_count"] += 1
            elif variant == "auto_only":
                run_counters["auto_decoder_fit_count"] += 1
            score = timed(
                f"score_{variant}",
                lambda train_x=train_x, val_x=val_x, cols=cols: score_matrix(
                    train_x=train_x,
                    val_x=val_x,
                    train_y=inner_train[metadata["label_col"]],
                    val_y=inner_val[metadata["label_col"]],
                    feature_cols=cols,
                    metadata=metadata,
                    options=options,
                ),
                fold=fold_id,
                variant=variant,
            )
            scores_by_variant[variant].append(score)
            fold_scores[(fold_id, variant)] = score
            fold_metrics.append({
                "dataset": dataset_name,
                "task": task_name,
                "fold": fold_id,
                "variant": variant,
                "metric": metadata["primary_metric"],
                "metric_direction": metadata["metric_direction"],
                "score": score,
                "mean_variant_score": np.nan,
                "fold_improvement_over_dfs": np.nan,
                "fold_improvement_over_auto": np.nan,
                "mean_improvement_over_dfs": np.nan,
                "mean_improvement_over_auto": np.nan,
                "globally_selected_variant": "",
                "fold_best_variant": "",
            })
        auto_feature_cols_by_fold[fold_id] = list(matrices["auto_only"][2])
        fold_contexts.append({
            "fold": fold_id,
            "inner_train": inner_train,
            "inner_val": inner_val,
            "auto_train": auto_train,
            "auto_val": auto_val,
        })
    min_positive_folds = resolve_edge_screening_min_positive_folds(
        options,
        effective_selection_folds=effective_selection_folds,
    )
    independent_screened_edges: list[dict[str, Any]] = []
    screened_edges: list[dict[str, Any]] = []
    selected_edge_ids: set[str] = set()
    edge_status_by_id: dict[str, dict[str, Any]] = {}
    rescue: dict[str, Any] = {
        "pair_screening": [],
        "pair_screening_fold_metrics": [],
        "selected_edges": [],
    }
    if options.enable_edge_screening:
        for edge in fdhg_edges:
            edge_id = str(edge.get("edge_id", ""))
            gains: list[float] = []
            usable_counts: list[int] = []
            fold_auto_scores_for_edge: list[float] = []
            fold_ids_for_edge: list[int] = []
            pooled_labels: list[Any] = []
            pooled_auto_predictions: list[Any] = []
            pooled_candidate_predictions: list[Any] = []
            future_violations = 0
            materialization_failed = False
            error_reason = ""
            for context in fold_contexts:
                fold_id = int(context["fold"])
                try:
                    run_counters["fdhg_mapping_fit_count"] += 1
                    run_counters["fdhg_materialization_count"] += 2
                    fdhg = timed(
                        "materialize_single_edge_fdhg",
                        lambda edge=edge, context=context, fold_id=fold_id: fit_transform_single_edge_fdhg_fold_cached(
                            inner_train_rows=context["inner_train"],
                            inner_validation_rows=context["inner_val"],
                            source_tables=table_dict,
                            task_metadata=metadata,
                            edge=edge,
                            lookup_source_columns_by_table=lookup_source_columns_by_table,
                            pit_lookup_cache=pit_lookup_cache,
                            fold=fold_id,
                            target_lookup_value_mapping=(
                                target_lookup_value_mapping
                            ),
                            continuous_fdhg_mode=options.continuous_fdhg_mode,
                            continuous_fdhg_bins=options.continuous_fdhg_bins,
                            continuous_fdhg_min_effective_bins=options.continuous_fdhg_min_effective_bins,
                            dataset=dataset_name,
                            task=task_name,
                        ),
                        fold=fold_id,
                        edge_id=edge_id,
                    )
                    fdhg_cols = feature_columns(fdhg["train_x"], join_keys, metadata)
                    _collect_continuous_discretization(
                        audit_rows=continuous_discretization_audit,
                        boundaries=continuous_discretization_boundaries,
                        materialization=fdhg,
                        fold=fold_id,
                        stage="materialize_single_edge_fdhg",
                    )
                    audit_rows, usable_cols = audit_residual_columns(
                        frame=fdhg["train_x"],
                        feature_cols=fdhg_cols,
                        fold=fold_id,
                        provenance=fdhg["feature_provenance"],
                    )
                    usable_count = len(usable_cols)
                    usable_counts.append(usable_count)
                    future_count = sum(
                        int(row.get("future_lookup_violation_count") or 0)
                        for row in fdhg.get("target_lookup_audit", [])
                    )
                    future_violations += future_count
                    if usable_count > 0 and future_count == 0:
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
                        if options.edge_screening_rule == "pooled_oof":
                            run_counters["decoder_fit_count"] += 2
                            run_counters["single_edge_decoder_fit_count"] += 1
                            run_counters["auto_decoder_fit_count"] += 1
                            run_counters["decoder_prediction_count"] += 2
                            auto_result = timed(
                                "score_single_edge_auto_oof",
                                lambda context=context, fold_id=fold_id: score_matrix_with_predictions(
                                    train_x=context["auto_train"],
                                    val_x=context["auto_val"],
                                    train_y=context["inner_train"][metadata["label_col"]],
                                    val_y=context["inner_val"][metadata["label_col"]],
                                    feature_cols=auto_feature_cols_by_fold[fold_id],
                                    metadata=metadata,
                                    options=options,
                                ),
                                fold=fold_id,
                                edge_id=edge_id,
                            )
                            candidate_result = timed(
                                "score_single_edge",
                                lambda single_train=single_train, single_val=single_val, usable_cols=usable_cols, context=context, fold_id=fold_id: score_matrix_with_predictions(
                                    train_x=single_train,
                                    val_x=single_val,
                                    train_y=context["inner_train"][metadata["label_col"]],
                                    val_y=context["inner_val"][metadata["label_col"]],
                                    feature_cols=[*auto_feature_cols_by_fold[fold_id], *usable_cols],
                                    metadata=metadata,
                                    options=options,
                                ),
                                fold=fold_id,
                                edge_id=edge_id,
                            )
                            single_score = float(candidate_result["score"])
                            pooled_labels.append(context["inner_val"][metadata["label_col"]].reset_index(drop=True))
                            pooled_auto_predictions.append(auto_result["prediction"])
                            pooled_candidate_predictions.append(candidate_result["prediction"])
                        else:
                            run_counters["decoder_fit_count"] += 1
                            run_counters["single_edge_decoder_fit_count"] += 1
                            run_counters["decoder_prediction_count"] += 1
                            single_score = timed(
                                "score_single_edge",
                                lambda single_train=single_train, single_val=single_val, usable_cols=usable_cols, context=context, fold_id=fold_id: score_matrix(
                                    train_x=single_train,
                                    val_x=single_val,
                                    train_y=context["inner_train"][metadata["label_col"]],
                                    val_y=context["inner_val"][metadata["label_col"]],
                                    feature_cols=[*auto_feature_cols_by_fold[fold_id], *usable_cols],
                                    metadata=metadata,
                                    options=options,
                                ),
                                fold=fold_id,
                                edge_id=edge_id,
                            )
                        gain = edge_fold_gain(
                            auto_score=fold_scores[(fold_id, "auto_only")],
                            auto_plus_single_edge_score=single_score,
                            direction=metadata["metric_direction"],
                        )
                    else:
                        single_score = math.nan
                        gain = math.nan
                    gains.append(gain)
                    fold_auto_scores_for_edge.append(float(fold_scores[(fold_id, "auto_only")]))
                    fold_ids_for_edge.append(fold_id)
                    single_edge_cache[(edge_id, fold_id)] = {
                        "fdhg": fdhg,
                        "usable_cols": usable_cols,
                        "score": single_score,
                        "gain": gain,
                        "usable_count": usable_count,
                        "future_lookup_violation_count": future_count,
                    }
                    edge_screening_fold_rows.append({
                        "dataset": dataset_name,
                        "task": task_name,
                        "edge_id": edge_id,
                        "fold": fold_id,
                        "requested_selection_folds": requested_selection_folds,
                        "effective_selection_folds": effective_selection_folds,
                        "effective_fold_ids": "|".join(str(value) for value in effective_fold_ids),
                        "auto_score": fold_scores[(fold_id, "auto_only")],
                        "auto_plus_single_edge_score": single_score,
                        "fold_gain": gain,
                        "metric": metadata["primary_metric"],
                        "metric_direction": metadata["metric_direction"],
                        "usable_feature_count": usable_count,
                        "future_lookup_violation_count": future_count,
                    })
                    del audit_rows
                except Exception as exc:
                    materialization_failed = True
                    error_reason = str(exc)
                    usable_counts.append(0)
                    edge_screening_fold_rows.append({
                        "dataset": dataset_name,
                        "task": task_name,
                        "edge_id": edge_id,
                        "fold": fold_id,
                        "requested_selection_folds": requested_selection_folds,
                        "effective_selection_folds": effective_selection_folds,
                        "effective_fold_ids": "|".join(str(value) for value in effective_fold_ids),
                        "auto_score": fold_scores[(fold_id, "auto_only")],
                        "auto_plus_single_edge_score": math.nan,
                        "fold_gain": math.nan,
                        "metric": metadata["primary_metric"],
                        "metric_direction": metadata["metric_direction"],
                        "usable_feature_count": 0,
                        "future_lookup_violation_count": 0,
                    })
            aggregate_auto_score = None
            aggregate_candidate_score = None
            aggregate_gain = None
            if options.edge_screening_rule == "pooled_oof" and pooled_labels:
                y_true = pd.concat([pd.Series(values) for values in pooled_labels], ignore_index=True)
                auto_pred = _concat_predictions(pooled_auto_predictions)
                candidate_pred = _concat_predictions(pooled_candidate_predictions)
                aggregate_auto_score = _metric_score(
                    y_true,
                    auto_pred,
                    metric=metadata["primary_metric"],
                    problem_type=metadata["problem_type"],
                )
                aggregate_candidate_score = _metric_score(
                    y_true,
                    candidate_pred,
                    metric=metadata["primary_metric"],
                    problem_type=metadata["problem_type"],
                )
                aggregate_gain = metric_improvement(
                    candidate=aggregate_candidate_score,
                    reference=aggregate_auto_score,
                    direction=metadata["metric_direction"],
                )
            summary = summarize_edge_screening(
                gains=gains,
                usable_feature_counts=usable_counts,
                future_lookup_violation_count=future_violations,
                min_delta=options.edge_screening_min_delta,
                min_positive_folds=min_positive_folds,
                screening_rule=options.edge_screening_rule,
                aggregate_gain=aggregate_gain,
                aggregate_auto_score=aggregate_auto_score,
                aggregate_candidate_score=aggregate_candidate_score,
                fold_auto_scores=fold_auto_scores_for_edge,
                fold_ids=fold_ids_for_edge,
                max_relative_fold_degradation=options.edge_screening_max_relative_fold_degradation,
                materialization_failed=materialization_failed,
            )
            if materialization_failed and error_reason:
                summary["materialization_error"] = error_reason
            row = {
                "dataset": dataset_name,
                "task": task_name,
                "requested_selection_folds": requested_selection_folds,
                "effective_selection_folds": effective_selection_folds,
                "effective_fold_ids": "|".join(str(value) for value in effective_fold_ids),
                **_edge_identity_row(edge),
                **summary,
                "screening_min_positive_fraction": options.edge_screening_min_positive_fraction,
            }
            edge_screening_rows.append(row)
            edge_status_by_id[edge_id] = row
            if row["screening_status"] == "screened_in":
                independent_screened_edges.append(dict(edge))
        if (
            options.edge_selection_strategy == "greedy"
            and options.enable_pairwise_rescue
            and not independent_screened_edges
            and len(fdhg_edges) >= 2
        ):
            rescue = evaluate_pairwise_rescue(
                dataset_name=dataset_name,
                task_name=task_name,
                table_dict=table_dict,
                metadata=metadata,
                fdhg_edges=fdhg_edges,
                join_keys=join_keys,
                options=options,
                fold_contexts=fold_contexts,
                fold_scores=fold_scores,
                auto_feature_cols_by_fold=auto_feature_cols_by_fold,
                lookup_source_columns_by_table=lookup_source_columns_by_table,
                pit_lookup_cache=pit_lookup_cache,
                timed=timed,
                run_counters=run_counters,
                min_positive_folds=min_positive_folds,
                target_lookup_value_mapping=(
                    target_lookup_value_mapping
                ),
            )
            pair_screening_rows = rescue["pair_screening"]
            pair_screening_fold_rows = rescue["pair_screening_fold_metrics"]
            continuous_discretization_audit.extend(rescue.get("continuous_discretization_audit", []))
            continuous_discretization_boundaries.update(rescue.get("continuous_discretization_boundaries", {}))
    else:
        independent_screened_edges = [dict(edge) for edge in fdhg_edges]
        for rank, edge in enumerate(independent_screened_edges, start=1):
            row = {
                "dataset": dataset_name,
                "task": task_name,
                "requested_selection_folds": requested_selection_folds,
                "effective_selection_folds": effective_selection_folds,
                "effective_fold_ids": "|".join(str(value) for value in effective_fold_ids),
                **_edge_identity_row(edge),
                "mean_gain": math.nan,
                "positive_fold_count": 0,
                "total_fold_count": 0,
                "worst_fold_gain": math.nan,
                "gain_std": math.nan,
                "screening_min_delta": options.edge_screening_min_delta,
                "screening_rule": options.edge_screening_rule,
                "screening_min_positive_fraction": options.edge_screening_min_positive_fraction,
                "screening_min_positive_folds": min_positive_folds,
                "screening_max_relative_fold_degradation": options.edge_screening_max_relative_fold_degradation,
                "screening_status": "screened_in",
                "rejection_reason": "",
                "usable_feature_count_min": 0,
                "usable_feature_count_max": 0,
                "future_lookup_violation_count": 0,
                "screening_rank": rank,
            }
            edge_screening_rows.append(row)
            edge_status_by_id[str(edge.get("edge_id", ""))] = row
    auto_only_scores_by_fold = {
        int(context["fold"]): float(fold_scores[(int(context["fold"]), "auto_only")])
        for context in fold_contexts
    }
    candidate_order = [str(edge.get("edge_id", "")) for edge in fdhg_edges]
    candidate_index = {edge_id: idx for idx, edge_id in enumerate(candidate_order)}
    edge_by_id = {str(edge.get("edge_id", "")): dict(edge) for edge in fdhg_edges}
    ranked_screened = sorted(
        [row for row in edge_screening_rows if row.get("screening_status") == "screened_in"],
        key=lambda row: (
            -float(row["aggregate_gain" if options.edge_screening_rule == "pooled_oof" else "mean_gain"])
            if np.isfinite(row.get("aggregate_gain" if options.edge_screening_rule == "pooled_oof" else "mean_gain", math.nan))
            else 0.0,
            candidate_index.get(str(row.get("edge_id", "")), 10**9),
            str(row["edge_id"]),
        ),
    )
    rank_by_edge_id = {str(row["edge_id"]): idx for idx, row in enumerate(ranked_screened, start=1)}
    for row in edge_screening_rows:
        row["screening_rank"] = rank_by_edge_id.get(str(row.get("edge_id", "")), "")
    independent_screened_edges = [
        edge_by_id[str(row["edge_id"])]
        for row in ranked_screened
        if str(row.get("edge_id", "")) in edge_by_id
    ]
    max_selected_edges = resolve_max_selected_fdhg_edges(options)
    edge_selection_step_count = 0
    edge_selection_stop_reason = ""

    def trial_selection_gain(trial: Mapping[str, Any]) -> float:
        key = "aggregate_gain" if options.edge_screening_rule == "pooled_oof" else "mean_gain"
        return float(trial.get(key, math.nan))

    def trial_forward_passes(trial: Mapping[str, Any]) -> bool:
        if options.edge_screening_rule == "pooled_oof":
            return bool(trial.get("screening_status") == "screened_in")
        return bool(
            np.isfinite(trial.get("mean_gain", math.nan))
            and float(trial["mean_gain"]) > float(options.edge_screening_min_delta)
            and int(trial.get("positive_fold_count", 0)) >= int(min_positive_folds)
        )

    def trial_backward_passes(trial: Mapping[str, Any]) -> bool:
        if options.edge_screening_rule == "pooled_oof":
            return bool(trial.get("screening_status") == "screened_in")
        return bool(
            np.isfinite(trial.get("mean_gain", math.nan))
            and float(trial["mean_gain"]) >= -1e-12
        )

    if options.edge_selection_strategy == "independent":
        screened_edges = [dict(edge) for edge in independent_screened_edges[:max_selected_edges]]
        selected_edge_ids = {str(edge.get("edge_id", "")) for edge in screened_edges}
        edge_selection_stop_reason = (
            "selection_budget_reached"
            if max_selected_edges is not None and len(independent_screened_edges) > max_selected_edges
            else "independent_screening_complete"
        )
        append_selection_trace(
            step=0,
            candidate_edge_id="",
            candidate_edge_ids=[],
            selected_after=[str(edge.get("edge_id", "")) for edge in screened_edges],
            baseline_scores_by_fold=auto_only_scores_by_fold,
            trial={
                "mean_score": math.nan,
                "gains_by_fold": {},
                "mean_gain": math.nan,
                "positive_fold_count": 0,
            },
            decision="selected_independent_screened_set",
            stop_reason=edge_selection_stop_reason,
        )
    elif options.edge_selection_strategy == "greedy":
        selected: list[dict[str, Any]] = []
        selected_ids: list[str] = []
        remaining = [dict(edge) for edge in fdhg_edges]
        current_scores = dict(auto_only_scores_by_fold)
        selected_pair = [
            dict(edge)
            for edge in rescue["selected_edges"]
        ] if (
            options.enable_edge_screening
            and not independent_screened_edges
            and len(fdhg_edges) >= 2
            and (max_selected_edges is None or max_selected_edges >= 2)
        ) else []
        if selected_pair:
            pair_ids = [
                str(edge.get("edge_id", ""))
                for edge in selected_pair
            ]
            pair_trial = evaluate_candidate_block(
                stage="materialize_greedy_initial_pair",
                candidate_edges=selected_pair,
                baseline_scores_by_fold=current_scores,
                baseline_edges=[],
            )
            if trial_forward_passes(pair_trial):
                selected = selected_pair
                selected_ids = pair_ids
                selected_id_set = set(selected_ids)
                remaining = [
                    edge
                    for edge in remaining
                    if str(edge.get("edge_id", "")) not in selected_id_set
                ]
                current_scores = dict(pair_trial["scores_by_fold"])
                edge_selection_step_count += 1
                append_selection_trace(
                    step=edge_selection_step_count,
                    candidate_edge_id="||".join(selected_ids),
                    candidate_edge_ids=selected_ids,
                    selected_after=selected_ids,
                    baseline_scores_by_fold=auto_only_scores_by_fold,
                    trial=pair_trial,
                    decision="accepted",
                )
            else:
                edge_selection_stop_reason = "initial_pair_failed_gate"
                append_selection_trace(
                    step=1,
                    candidate_edge_id="||".join(pair_ids),
                    candidate_edge_ids=pair_ids,
                    selected_after=[],
                    baseline_scores_by_fold=auto_only_scores_by_fold,
                    trial=pair_trial,
                    decision="rejected_stop",
                    stop_reason=edge_selection_stop_reason,
                )
                remaining = []
        while remaining and (max_selected_edges is None or len(selected) < max_selected_edges):
            step = edge_selection_step_count + 1
            trials: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
            for edge in remaining:
                edge_id = str(edge.get("edge_id", ""))
                trial_edges = [*selected, edge]
                trial = evaluate_candidate_block(
                    stage="materialize_greedy_trial_fdhg",
                    candidate_edges=trial_edges,
                    baseline_scores_by_fold=current_scores,
                    baseline_edges=selected,
                )
                trials.append((candidate_index[edge_id], dict(edge), trial, edge_id))
                append_selection_trace(
                    step=step,
                    candidate_edge_id=edge_id,
                    candidate_edge_ids=[str(item.get("edge_id", "")) for item in trial_edges],
                    selected_after=selected_ids,
                    baseline_scores_by_fold=current_scores,
                    trial=trial,
                    decision="tested",
                )
            best_passing_trial = _select_best_passing_greedy_trial(
                trials,
                trial_forward_passes=trial_forward_passes,
                trial_selection_gain=trial_selection_gain,
            )
            if best_passing_trial is None:
                best_index, best_edge, best_trial, best_edge_id = min(
                    trials,
                    key=lambda item: (
                        -trial_selection_gain(item[2])
                        if np.isfinite(trial_selection_gain(item[2]))
                        else math.inf,
                        item[0],
                    ),
                )
                edge_selection_stop_reason = "no_remaining_candidate_passed_gate"
                append_selection_trace(
                    step=step,
                    candidate_edge_id=best_edge_id,
                    candidate_edge_ids=[*selected_ids, best_edge_id],
                    selected_after=selected_ids,
                    baseline_scores_by_fold=current_scores,
                    trial=best_trial,
                    decision="rejected_stop",
                    stop_reason=edge_selection_stop_reason,
                )
                break

            best_index, best_edge, best_trial, best_edge_id = best_passing_trial
            previous_scores = dict(current_scores)
            selected.append(best_edge)
            selected_ids.append(best_edge_id)
            current_scores = dict(best_trial["scores_by_fold"])
            remaining = [
                edge for edge in remaining
                if str(edge.get("edge_id", "")) != best_edge_id
            ]
            edge_selection_step_count = step
            append_selection_trace(
                step=step,
                candidate_edge_id=best_edge_id,
                candidate_edge_ids=selected_ids,
                selected_after=selected_ids,
                baseline_scores_by_fold=previous_scores,
                trial=best_trial,
                decision="accepted",
            )
        if not edge_selection_stop_reason:
            edge_selection_stop_reason = (
                "selection_budget_reached"
                if max_selected_edges is not None and len(selected) >= max_selected_edges
                else "candidate_pool_exhausted"
            )
        screened_edges = [dict(edge) for edge in selected]
        selected_edge_ids = set(selected_ids)
    elif options.edge_selection_strategy == "greedy_backward":
        selected = [dict(edge) for edge in independent_screened_edges[:max_selected_edges]]
        selected_ids = [str(edge.get("edge_id", "")) for edge in selected]
        if not selected:
            current_scores = dict(auto_only_scores_by_fold)
            edge_selection_stop_reason = "empty_independent_screened_start"
        else:
            initial_trial = evaluate_candidate_block(
                stage="materialize_backward_initial_fdhg",
                candidate_edges=selected,
                baseline_scores_by_fold=auto_only_scores_by_fold,
                baseline_edges=[],
            )
            current_scores = dict(initial_trial["scores_by_fold"])
            append_selection_trace(
                step=0,
                candidate_edge_id="",
                candidate_edge_ids=selected_ids,
                selected_after=selected_ids,
                baseline_scores_by_fold=auto_only_scores_by_fold,
                trial=initial_trial,
                decision="backward_start_from_independent_screened_set",
            )
            while len(selected) > 1:
                step = edge_selection_step_count + 1
                removal_trials: list[tuple[int, dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]] = []
                for edge in selected:
                    edge_id = str(edge.get("edge_id", ""))
                    trial_edges = [
                        item for item in selected
                        if str(item.get("edge_id", "")) != edge_id
                    ]
                    trial = evaluate_candidate_block(
                        stage="materialize_backward_removal_trial_fdhg",
                        candidate_edges=trial_edges,
                        baseline_scores_by_fold=current_scores,
                        baseline_edges=selected,
                    )
                    removal_trials.append((candidate_index[edge_id], dict(edge), trial, edge_id, trial_edges))
                    append_selection_trace(
                        step=step,
                        candidate_edge_id=edge_id,
                        candidate_edge_ids=[str(item.get("edge_id", "")) for item in trial_edges],
                        selected_after=selected_ids,
                        baseline_scores_by_fold=current_scores,
                        trial=trial,
                        decision="tested_removal",
                    )
                best_index, _removed_edge, best_trial, removed_edge_id, best_trial_edges = min(
                    removal_trials,
                    key=lambda item: (
                        -trial_selection_gain(item[2]) if np.isfinite(trial_selection_gain(item[2])) else math.inf,
                        item[0],
                    ),
                )
                passes = trial_backward_passes(best_trial)
                if not passes:
                    edge_selection_stop_reason = "no_non_degrading_removal"
                    append_selection_trace(
                        step=step,
                        candidate_edge_id=removed_edge_id,
                        candidate_edge_ids=[str(item.get("edge_id", "")) for item in best_trial_edges],
                        selected_after=selected_ids,
                        baseline_scores_by_fold=current_scores,
                        trial=best_trial,
                        decision="rejected_stop",
                        stop_reason=edge_selection_stop_reason,
                    )
                    break
                previous_scores = dict(current_scores)
                selected = [dict(edge) for edge in best_trial_edges]
                selected_ids = [str(edge.get("edge_id", "")) for edge in selected]
                current_scores = dict(best_trial["scores_by_fold"])
                edge_selection_step_count = step
                append_selection_trace(
                    step=step,
                    candidate_edge_id=removed_edge_id,
                    candidate_edge_ids=selected_ids,
                    selected_after=selected_ids,
                    baseline_scores_by_fold=previous_scores,
                    trial=best_trial,
                    decision="removed",
                )
            if not edge_selection_stop_reason:
                edge_selection_stop_reason = "single_edge_remaining" if len(selected) == 1 else "candidate_pool_exhausted"
        screened_edges = [dict(edge) for edge in selected]
        selected_edge_ids = set(selected_ids)
    for row in edge_screening_rows:
        row["selected_for_combined_block"] = str(row.get("edge_id", "")) in selected_edge_ids

    if screened_edges:
        scores_by_variant["auto_plus_fdhg"] = []
        if len(screened_edges) == 1:
            screened_edge_id = str(screened_edges[0].get("edge_id", ""))
        else:
            screened_edge_id = ""
        for context in fold_contexts:
            fold_id = int(context["fold"])
            cached_single = single_edge_cache.get((screened_edge_id, fold_id)) if screened_edge_id else None
            if cached_single is not None:
                fdhg = cached_single["fdhg"]
                usable_fdhg_cols = list(cached_single["usable_cols"])
                score = float(cached_single["score"])
            else:
                run_counters["fdhg_mapping_fit_count"] += 1
                run_counters["fdhg_materialization_count"] += 2
                run_counters["target_lookup_count"] += 2
                fdhg = timed(
                    "materialize_combined_fdhg",
                    lambda context=context, fold_id=fold_id: fit_transform_fdhg_fold(
                        inner_train_rows=context["inner_train"],
                        inner_validation_rows=context["inner_val"],
                        source_tables=table_dict,
                        schema=None,
                        task_metadata=metadata,
                        candidate_edges=screened_edges,
                        max_edges=len(screened_edges),
                        fold=fold_id,
                        continuous_fdhg_mode=options.continuous_fdhg_mode,
                        continuous_fdhg_bins=options.continuous_fdhg_bins,
                        continuous_fdhg_min_effective_bins=options.continuous_fdhg_min_effective_bins,
                        dataset=dataset_name,
                        task=task_name,
                        fit_split="fold_train",
                        **(
                            {
                                "target_lookup_value_mapping":
                                    target_lookup_value_mapping
                            }
                            if target_lookup_value_mapping
                            is not None
                            else {}
                        ),
                    ),
                    fold=fold_id,
                    edge_count=len(screened_edges),
                )
            accepted_edge_audit.extend([
                row for row in fdhg["edge_audit"]
                if row.get("selection_status") == "accepted"
            ])
            rejected_edge_audit.extend([
                row for row in fdhg["edge_audit"]
                if row.get("selection_status") != "accepted"
            ])
            fdhg_feature_provenance.extend(fdhg["feature_provenance"])
            fdhg_target_lookup_audit.extend(fdhg.get("target_lookup_audit", []))
            fdhg_cols = feature_columns(fdhg["train_x"], join_keys, metadata)
            _collect_continuous_discretization(
                audit_rows=continuous_discretization_audit,
                boundaries=continuous_discretization_boundaries,
                materialization=fdhg,
                fold=fold_id,
                stage="materialize_combined_fdhg",
            )
            declared_residual_features.update(fdhg_cols)
            fdhg_audit_rows, usable_fdhg_cols = audit_residual_columns(
                frame=fdhg["train_x"],
                feature_cols=fdhg_cols,
                fold=fold_id,
                provenance=fdhg["feature_provenance"],
            )
            fdhg_fold_feature_audit.extend(fdhg_audit_rows)
            usable_by_fold[str(fold_id)] = len(usable_fdhg_cols)
            if cached_single is None:
                auto_fdhg_train = align_feature_blocks(
                    target_rows=context["inner_train"],
                    blocks=[("auto", context["auto_train"]), ("fdhg", fdhg["train_x"])],
                    join_keys=join_keys,
                    metadata=metadata,
                )
                auto_fdhg_val = align_feature_blocks(
                    target_rows=context["inner_val"],
                    blocks=[("auto", context["auto_val"]), ("fdhg", fdhg["validation_x"])],
                    join_keys=join_keys,
                    metadata=metadata,
                )
                run_counters["decoder_fit_count"] += 1
                run_counters["combined_decoder_fit_count"] += 1
                run_counters["decoder_prediction_count"] += 1
                score = timed(
                    "score_combined_fdhg",
                    lambda auto_fdhg_train=auto_fdhg_train, auto_fdhg_val=auto_fdhg_val, usable_fdhg_cols=usable_fdhg_cols, context=context, fold_id=fold_id: score_matrix(
                        train_x=auto_fdhg_train,
                        val_x=auto_fdhg_val,
                        train_y=context["inner_train"][metadata["label_col"]],
                        val_y=context["inner_val"][metadata["label_col"]],
                        feature_cols=[*auto_feature_cols_by_fold[fold_id], *usable_fdhg_cols],
                        metadata=metadata,
                        options=options,
                    ),
                    fold=fold_id,
                    edge_count=len(screened_edges),
                )
            scores_by_variant["auto_plus_fdhg"].append(score)
            fold_scores[(fold_id, "auto_plus_fdhg")] = score
            fold_metrics.append({
                "dataset": dataset_name,
                "task": task_name,
                "fold": fold_id,
                "variant": "auto_plus_fdhg",
                "metric": metadata["primary_metric"],
                "metric_direction": metadata["metric_direction"],
                "score": score,
                "mean_variant_score": np.nan,
                "fold_improvement_over_dfs": np.nan,
                "fold_improvement_over_auto": np.nan,
                "mean_improvement_over_dfs": np.nan,
                "mean_improvement_over_auto": np.nan,
                "globally_selected_variant": "",
                "fold_best_variant": "",
            })
    else:
        usable_by_fold = {}
    mean_scores = {
        variant: float(np.mean(values)) if values else math.nan
        for variant, values in scores_by_variant.items()
    }
    if external_auto_reference_only:
        auto_score = float(mean_scores["auto_only"])
        fdhg_score = float(
            mean_scores.get("auto_plus_fdhg", math.nan)
        )

        fdhg_gain = (
            metric_improvement(
                candidate=fdhg_score,
                reference=auto_score,
                direction=metadata["metric_direction"],
            )
            if np.isfinite(fdhg_score)
            else math.nan
        )

        fdhg_passed = bool(
            np.isfinite(fdhg_gain)
            and fdhg_gain > float(options.min_delta)
        )

        selected_variant = (
            "auto_plus_fdhg"
            if fdhg_passed
            else "auto_only"
        )

        selection = {
            "selected_variant": selected_variant,
            "selection_reason": (
                "selected_auto_plus_fdhg_over_auto_external_gate"
                if fdhg_passed
                else "selected_auto_external_gate"
            ),
            "gate_trials": [
                {
                    "variant": "auto_only",
                    "score": auto_score,
                    "improvement_over_auto": 0.0,
                    "admissible": True,
                },
                {
                    "variant": "auto_plus_fdhg",
                    "score": fdhg_score,
                    "improvement_over_auto": fdhg_gain,
                    "admissible": fdhg_passed,
                },
            ],
        }
    else:
        selection = select_joint_variant(
            mean_scores=mean_scores,
            metric_direction=metadata["metric_direction"],
            min_delta=options.min_delta,
        )
    for row in fold_metrics:
        fold_id = int(row["fold"])
        fold_best = _best_variant_for_scores(
            {variant: score for (fid, variant), score in fold_scores.items() if fid == fold_id},
            direction=metadata["metric_direction"],
        )
        row["mean_variant_score"] = mean_scores[row["variant"]]
        row["fold_improvement_over_dfs"] = (
            math.nan
            if external_auto_reference_only
            else metric_improvement(
                candidate=fold_scores[(fold_id, row["variant"])],
                reference=fold_scores[(fold_id, "dfs_fallback")],
                direction=metadata["metric_direction"],
            )
        )
        row["fold_improvement_over_auto"] = metric_improvement(
            candidate=fold_scores[(fold_id, row["variant"])],
            reference=fold_scores[(fold_id, "auto_only")],
            direction=metadata["metric_direction"],
        )
        row["mean_improvement_over_dfs"] = (
            math.nan
            if external_auto_reference_only
            else metric_improvement(
                candidate=mean_scores[row["variant"]],
                reference=mean_scores["dfs_fallback"],
                direction=metadata["metric_direction"],
            )
        )
        row["mean_improvement_over_auto"] = metric_improvement(
            candidate=mean_scores[row["variant"]],
            reference=mean_scores["auto_only"],
            direction=metadata["metric_direction"],
        )
        row["globally_selected_variant"] = selection["selected_variant"]
        row["fold_best_variant"] = fold_best
    selected_fdhg_edges = []
    for edge in fdhg_edges:
        edge_id = str(edge.get("edge_id", ""))
        status = edge_status_by_id.get(edge_id, {})
        selected_fdhg_edges.append({
            "dataset": dataset_name,
            "task": task_name,
            **_edge_identity_row(edge),
            "screening_status": status.get("screening_status", ""),
            "mean_incremental_gain": status.get("mean_gain", math.nan),
            "positive_fold_count": status.get("positive_fold_count", 0),
            "screening_rank": status.get("screening_rank", ""),
            "selected_for_combined_block": edge_id in selected_edge_ids,
            "rejection_reason": status.get("rejection_reason", ""),
        })
        if not screened_edges:
            rejected_edge_audit = selected_fdhg_edges
    run_counters["target_lookup_count"] += len(pit_lookup_cache)
    return {
        "fold_metrics": fold_metrics,
        "gate_trials": selection["gate_trials"],
        "selected_variant": selection["selected_variant"],
        "selection_reason": selection["selection_reason"],
        "mean_scores": mean_scores,
        "requested_selection_folds": requested_selection_folds,
        "effective_selection_folds": effective_selection_folds,
        "effective_fold_ids": effective_fold_ids,
        "edge_selection_strategy": options.edge_selection_strategy,
        "max_selected_fdhg_edges": options.max_selected_fdhg_edges,
        "ordered_candidate_edge_ids": [str(edge.get("edge_id", "")) for edge in fdhg_edges],
        "independent_screened_in_edge_ids": [
            str(edge.get("edge_id", "")) for edge in independent_screened_edges
        ],
        "strategy_selected_edge_ids": [
            str(edge.get("edge_id", "")) for edge in screened_edges
        ],
        "final_combined_block_edge_ids": [
            str(edge.get("edge_id", "")) for edge in screened_edges
        ],
        "edge_selection_trace": edge_selection_trace_rows,
        "edge_selection_step_count": edge_selection_step_count,
        "edge_selection_stop_reason": edge_selection_stop_reason,
        "pairwise_rescue_used": bool(screened_edges and pair_screening_rows and any(
            row.get("selected_initial_pair") for row in pair_screening_rows
        )),
        "selected_initial_pair": next(
            (
                str(row.get("pair_id", ""))
                for row in pair_screening_rows
                if row.get("selected_initial_pair")
            ),
            "",
        ),
        "pairwise_rescue_reason": _pairwise_rescue_reason(
            strategy=options.edge_selection_strategy,
            enabled=options.enable_pairwise_rescue,
            screened_edges=screened_edges,
            pair_screening_rows=pair_screening_rows,
        ),
        "selected_fdhg_edges": selected_fdhg_edges,
        "rejected_fdhg_edges": rejected_edge_audit,
        "screened_fdhg_edges": screened_edges,
        "edge_screening": edge_screening_rows,
        "edge_screening_fold_metrics": edge_screening_fold_rows,
        "pair_screening": pair_screening_rows,
        "pair_screening_fold_metrics": pair_screening_fold_rows,
        "edge_screening_enabled": options.enable_edge_screening,
        "edge_screening_rule": options.edge_screening_rule,
        "edge_screening_min_delta": options.edge_screening_min_delta,
        "edge_screening_min_positive_fraction": options.edge_screening_min_positive_fraction,
        "edge_screening_min_positive_folds": min_positive_folds,
        "edge_screening_max_relative_fold_degradation": options.edge_screening_max_relative_fold_degradation,
        "fdhg_screening_fallback": "auto_only" if not screened_edges else "",
        "fdhg_feature_provenance": fdhg_feature_provenance,
        "fdhg_fold_feature_audit": fdhg_fold_feature_audit,
        "fdhg_target_lookup_audit": fdhg_target_lookup_audit,
        "continuous_discretization_audit": continuous_discretization_audit,
        "continuous_discretization_boundaries": continuous_discretization_boundaries,
        "fdhg_usable_residual_features_by_fold": usable_by_fold,
        "fdhg_declared_residual_features": len(declared_residual_features),
        "run_counters": run_counters,
        "timing": timing_rows,
    }


def fit_transform_single_edge_fdhg_fold_cached(
    *,
    inner_train_rows: pd.DataFrame,
    inner_validation_rows: pd.DataFrame,
    source_tables: Mapping[str, Any],
    task_metadata: Mapping[str, Any],
    edge: Mapping[str, Any],
    lookup_source_columns_by_table: Mapping[str, Sequence[str]],
    pit_lookup_cache: dict[tuple[int, str, str], tuple[pd.DataFrame, dict[str, Any]]],
    fold: int,
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
    continuous_fdhg_mode: str = "exclude",
    continuous_fdhg_bins: int = 8,
    continuous_fdhg_min_effective_bins: int = 2,
    dataset: str = "",
    task: str = "",
) -> dict[str, Any]:
    fit_horizon = inner_train_rows[task_metadata["target_time_col"]].max()
    fitted = fit_afd_edges(
        inner_train_rows=inner_train_rows,
        source_tables=source_tables,
        schema=None,
        task_metadata=task_metadata,
        candidate_edges=[edge],
        max_edges=1,
        fold=fold,
        fit_horizon=fit_horizon,
        continuous_fdhg_mode=continuous_fdhg_mode,
        continuous_fdhg_bins=continuous_fdhg_bins,
        continuous_fdhg_min_effective_bins=continuous_fdhg_min_effective_bins,
        dataset=dataset,
        task=task,
        fit_split="fold_train",
    )
    accepted_fitted = [item for item in fitted if item.selection_status == "accepted"]

    def materialize_split(target_rows: pd.DataFrame, split: str) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        result = target_rows[[task_metadata["entity_key"], task_metadata["target_time_col"]]].reset_index(drop=True).copy()
        provenance: list[dict[str, Any]] = []
        lookup_audit: list[dict[str, Any]] = []
        for fitted_edge in accepted_fitted:
            table = source_tables[fitted_edge.source_table]
            table_df = _table_df(table)
            time_col = getattr(table, "time_col", None)
            if not time_col or str(time_col) not in table_df.columns:
                lookup_audit.append({
                    "fold": fitted_edge.fold,
                    "edge_id": fitted_edge.edge_id,
                    "target_row_count": len(target_rows),
                    "matched_target_rows": 0,
                    "unmatched_target_rows": len(target_rows),
                    "target_lookup_coverage": 0.0,
                    "maximum_lookup_source_time": None,
                    "maximum_target_time": str(pd.to_datetime(target_rows[task_metadata["target_time_col"]], errors="coerce").max()),
                    "future_lookup_violation_count": 0,
                    "mapping_fit_horizon": fitted_edge.fit_end_time,
                    "maximum_mapping_source_time": fitted_edge.maximum_source_time_used,
                    "rejection_reason": "missing_source_time_for_point_in_time_lookup",
                })
                continue
            row_entity_key = str(task_metadata["entity_key"])
            target_lookup_entity_key = str(
                edge.get("target_lookup_column") or row_entity_key
            )
            strict_before = bool(edge.get("strict_before", False))
            lookup_transform = str(
                edge.get(
                    "target_lookup_value_transform",
                    "",
                )
            )
            edge_target_lookup_value_mapping = (
                target_lookup_value_mapping
                if (
                    lookup_transform
                    == "dbinfer_inverse_entity_mapping"
                )
                else None
            )

            source_entity_key = resolve_source_lookup_entity_key(
                table=table,
                source_rows=table_df,
                target_entity_key=target_lookup_entity_key,
                edge_source_entity_column=str(edge.get("source_entity_column", "")),
            )

            # Cache identity must include target-side lookup semantics and
            # temporal exact-match policy, not only the source table/key.
            cache_key = (
                fold,
                split,
                (
                    f"{fitted_edge.source_table}:"
                    f"{source_entity_key}:"
                    f"{target_lookup_entity_key}:"
                    f"{int(strict_before)}:"
                    f"{lookup_transform}"
                ),
            )
            cached = pit_lookup_cache.get(cache_key)
            if cached is None:
                all_source_cols = list(
                    lookup_source_columns_by_table.get(
                        fitted_edge.source_table,
                        fitted_edge.lhs_columns,
                    )
                )
                for lookup_col in auto_fdhg_edge_lhs_lookup_columns(fitted_edge):
                    if lookup_col not in all_source_cols:
                        all_source_cols.append(lookup_col)

                source_view, audit = point_in_time_asof_join(
                    target_rows=target_rows,
                    source_rows=table_df,

                    # Preserve prediction-row identity.
                    entity_key=row_entity_key,

                    # Use the edge-specific relational lookup identity.
                    target_lookup_entity_key=target_lookup_entity_key,
                    source_entity_key=source_entity_key,

                    target_time_col=task_metadata["target_time_col"],
                    source_time_col=str(time_col),
                    source_columns=all_source_cols,
                    target_lookup_value_mapping=(
                        edge_target_lookup_value_mapping
                    ),

                    # strict_before=True means source_time < target_time.
                    allow_exact_matches=not strict_before,
                )
                cached = (source_view, audit)
                pit_lookup_cache[cache_key] = cached
            source_view, audit = cached
            source_view = apply_fitted_edge_discretization_to_lookup(source_view, fitted_edge)
            edge_audit = dict(audit)
            edge_audit.update({
                "fold": fitted_edge.fold,
                "edge_id": fitted_edge.edge_id,
                "mapping_fit_horizon": fitted_edge.fit_end_time,
                "maximum_mapping_source_time": fitted_edge.maximum_source_time_used,
                "rejection_reason": "",
            })
            if fitted_edge.maximum_source_time_used and fitted_edge.fit_end_time:
                if pd.Timestamp(fitted_edge.maximum_source_time_used) > pd.Timestamp(fitted_edge.fit_end_time):
                    raise AssertionError("maximum_mapping_source_time_exceeds_fit_horizon")
            lookup_audit.append(edge_audit)
            frame, rows = materialize_ambiguity_from_map(
                source_view[list(fitted_edge.lhs_columns)],
                fitted_edge=fitted_edge,
            )
            for col in frame.columns:
                result[col] = frame[col].to_numpy()
            provenance.extend(rows)
        return result, provenance, lookup_audit

    train_x, train_prov, train_lookup_audit = materialize_split(inner_train_rows, "train")
    val_x, val_prov, val_lookup_audit = materialize_split(inner_validation_rows, "validation")
    return {
        "fitted_edges": fitted,
        "edge_audit": [fitted_edge_to_audit_row(edge) for edge in fitted],
        "train_x": train_x,
        "validation_x": val_x,
        "feature_provenance": train_prov + val_prov,
        "target_lookup_audit": train_lookup_audit + val_lookup_audit,
        "continuous_discretization_audit": [
            row
            for item in fitted
            for row in ((item.continuous_discretization or {}).get("audit", []) if isinstance(item.continuous_discretization, Mapping) else [])
        ],
        "continuous_discretization_boundaries": {
            str(item.edge_id): (item.continuous_discretization or {}).get("boundaries", {})
            for item in fitted
            if isinstance(item.continuous_discretization, Mapping)
        },
    }


def compose_cached_single_edge_block(
    *,
    cached_items: Sequence[Mapping[str, Any]],
    join_keys: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not cached_items:
        raise ValueError("cannot_compose_empty_cached_fdhg_block")
    first = cached_items[0]["fdhg"]
    train_x = first["train_x"][list(join_keys)].reset_index(drop=True).copy()
    validation_x = first["validation_x"][list(join_keys)].reset_index(drop=True).copy()
    edge_audit: list[dict[str, Any]] = []
    feature_provenance: list[dict[str, Any]] = []
    target_lookup_audit: list[dict[str, Any]] = []
    for item in cached_items:
        fdhg = item["fdhg"]
        edge_audit.extend(fdhg.get("edge_audit", []))
        feature_provenance.extend(fdhg.get("feature_provenance", []))
        target_lookup_audit.extend(fdhg.get("target_lookup_audit", []))
        for split, out in (("train_x", train_x), ("validation_x", validation_x)):
            frame = fdhg[split]
            for col in feature_columns(frame, join_keys, metadata):
                if col not in out.columns:
                    out[col] = frame[col].to_numpy()
    return {
        "fitted_edges": [],
        "edge_audit": edge_audit,
        "train_x": train_x,
        "validation_x": validation_x,
        "feature_provenance": feature_provenance,
        "target_lookup_audit": target_lookup_audit,
        "continuous_discretization_audit": [
            row
            for item in cached_items
            for row in item.get("fdhg", {}).get("continuous_discretization_audit", [])
        ],
        "continuous_discretization_boundaries": {
            edge_id: boundaries
            for item in cached_items
            for edge_id, boundaries in item.get("fdhg", {}).get("continuous_discretization_boundaries", {}).items()
        },
    }


def _collect_continuous_discretization(
    *,
    audit_rows: list[dict[str, Any]],
    boundaries: dict[str, Any],
    materialization: Mapping[str, Any],
    fold: int | str,
    stage: str,
) -> None:
    for row in materialization.get("continuous_discretization_audit", []) or []:
        item = dict(row)
        item.setdefault("fold", fold)
        item["stage"] = stage
        if item not in audit_rows:
            audit_rows.append(item)
    for edge_id, edge_boundaries in (materialization.get("continuous_discretization_boundaries", {}) or {}).items():
        key = f"{stage}:fold={fold}:{edge_id}"
        boundaries[key] = edge_boundaries


def evaluate_pairwise_rescue(
    *,
    dataset_name: str,
    task_name: str,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    fdhg_edges: Sequence[Mapping[str, Any]],
    join_keys: Sequence[str],
    options: AutoFdhgOptions,
    fold_contexts: Sequence[Mapping[str, Any]],
    fold_scores: Mapping[tuple[int, str], float],
    auto_feature_cols_by_fold: Mapping[int, Sequence[str]],
    lookup_source_columns_by_table: Mapping[str, Sequence[str]],
    pit_lookup_cache: dict[tuple[int, str, str], tuple[pd.DataFrame, dict[str, Any]]],
    timed: Any,
    run_counters: dict[str, Any],
    min_positive_folds: int,
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    pair_fold_rows: list[dict[str, Any]] = []
    continuous_discretization_audit: list[dict[str, Any]] = []
    continuous_discretization_boundaries: dict[str, Any] = {}
    selected_edges: list[dict[str, Any]] = []
    edge_by_id = {str(edge.get("edge_id", "")): edge for edge in fdhg_edges}
    for pair_rank, (left, right) in enumerate(itertools.combinations(fdhg_edges, 2), start=1):
        left_id = str(left.get("edge_id", ""))
        right_id = str(right.get("edge_id", ""))
        pair_id = f"{left_id}||{right_id}"
        gains: list[float] = []
        usable_counts: list[int] = []
        fold_baseline_scores_for_pair: list[float] = []
        fold_ids_for_pair: list[int] = []
        pooled_labels: list[Any] = []
        pooled_auto_predictions: list[Any] = []
        pooled_pair_predictions: list[Any] = []
        future_violations = 0
        materialization_failed = False
        error_reason = ""
        for context in fold_contexts:
            fold_id = int(context["fold"])
            try:
                run_counters["fdhg_mapping_fit_count"] += 1
                run_counters["fdhg_materialization_count"] += 2
                run_counters["target_lookup_count"] += 2
                fdhg = timed(
                    "materialize_pair_fdhg",
                    lambda context=context, fold_id=fold_id, left=left, right=right: fit_transform_fdhg_fold(
                        inner_train_rows=context["inner_train"],
                        inner_validation_rows=context["inner_val"],
                        source_tables=table_dict,
                        schema=None,
                        task_metadata=metadata,
                        candidate_edges=[left, right],
                        max_edges=2,
                        fold=fold_id,
                        continuous_fdhg_mode=options.continuous_fdhg_mode,
                        continuous_fdhg_bins=options.continuous_fdhg_bins,
                        continuous_fdhg_min_effective_bins=options.continuous_fdhg_min_effective_bins,
                        dataset=dataset_name,
                        task=task_name,
                        fit_split="fold_train",
                        **(
                            {
                                "target_lookup_value_mapping":
                                    target_lookup_value_mapping
                            }
                            if target_lookup_value_mapping
                            is not None
                            else {}
                        ),
                    ),
                    fold=fold_id,
                    pair_id=pair_id,
                )
                fdhg_cols = feature_columns(fdhg["train_x"], join_keys, metadata)
                _collect_continuous_discretization(
                    audit_rows=continuous_discretization_audit,
                    boundaries=continuous_discretization_boundaries,
                    materialization=fdhg,
                    fold=fold_id,
                    stage="materialize_pair_fdhg",
                )
                _audit_rows, usable_cols = audit_residual_columns(
                    frame=fdhg["train_x"],
                    feature_cols=fdhg_cols,
                    fold=fold_id,
                    provenance=fdhg["feature_provenance"],
                )
                usable_count = len(usable_cols)
                usable_counts.append(usable_count)
                future_count = sum(
                    int(row.get("future_lookup_violation_count") or 0)
                    for row in fdhg.get("target_lookup_audit", [])
                )
                future_violations += future_count
                if usable_count > 0 and future_count == 0:
                    pair_train = align_feature_blocks(
                        target_rows=context["inner_train"],
                        blocks=[("auto", context["auto_train"]), ("fdhg", fdhg["train_x"])],
                        join_keys=join_keys,
                        metadata=metadata,
                    )
                    pair_val = align_feature_blocks(
                        target_rows=context["inner_val"],
                        blocks=[("auto", context["auto_val"]), ("fdhg", fdhg["validation_x"])],
                        join_keys=join_keys,
                        metadata=metadata,
                    )
                    run_counters["decoder_fit_count"] += 1
                    run_counters["combined_decoder_fit_count"] += 1
                    run_counters["decoder_prediction_count"] += 1
                    if options.edge_screening_rule == "pooled_oof":
                        run_counters["decoder_fit_count"] += 1
                        run_counters["auto_decoder_fit_count"] += 1
                        run_counters["decoder_prediction_count"] += 1
                        auto_result = timed(
                            "score_pair_auto_oof",
                            lambda context=context, fold_id=fold_id: score_matrix_with_predictions(
                                train_x=context["auto_train"],
                                val_x=context["auto_val"],
                                train_y=context["inner_train"][metadata["label_col"]],
                                val_y=context["inner_val"][metadata["label_col"]],
                                feature_cols=auto_feature_cols_by_fold[fold_id],
                                metadata=metadata,
                                options=options,
                            ),
                            fold=fold_id,
                            pair_id=pair_id,
                        )
                        pair_result = timed(
                            "score_pair_fdhg",
                            lambda pair_train=pair_train, pair_val=pair_val, usable_cols=usable_cols, context=context, fold_id=fold_id: score_matrix_with_predictions(
                                train_x=pair_train,
                                val_x=pair_val,
                                train_y=context["inner_train"][metadata["label_col"]],
                                val_y=context["inner_val"][metadata["label_col"]],
                                feature_cols=[*auto_feature_cols_by_fold[fold_id], *usable_cols],
                                metadata=metadata,
                                options=options,
                            ),
                            fold=fold_id,
                            pair_id=pair_id,
                        )
                        pair_score = float(pair_result["score"])
                        pooled_labels.append(context["inner_val"][metadata["label_col"]].reset_index(drop=True))
                        pooled_auto_predictions.append(auto_result["prediction"])
                        pooled_pair_predictions.append(pair_result["prediction"])
                    else:
                        pair_score = timed(
                            "score_pair_fdhg",
                            lambda pair_train=pair_train, pair_val=pair_val, usable_cols=usable_cols, context=context, fold_id=fold_id: score_matrix(
                                train_x=pair_train,
                                val_x=pair_val,
                                train_y=context["inner_train"][metadata["label_col"]],
                                val_y=context["inner_val"][metadata["label_col"]],
                                feature_cols=[*auto_feature_cols_by_fold[fold_id], *usable_cols],
                                metadata=metadata,
                                options=options,
                            ),
                            fold=fold_id,
                            pair_id=pair_id,
                        )
                    gain = edge_fold_gain(
                        auto_score=fold_scores[(fold_id, "auto_only")],
                        auto_plus_single_edge_score=pair_score,
                        direction=metadata["metric_direction"],
                    )
                else:
                    pair_score = math.nan
                    gain = math.nan
                gains.append(gain)
                fold_baseline_scores_for_pair.append(float(fold_scores[(fold_id, "auto_only")]))
                fold_ids_for_pair.append(fold_id)
                pair_fold_rows.append({
                    "dataset": dataset_name,
                    "task": task_name,
                    "pair_id": pair_id,
                    "pair_rank": pair_rank,
                    "edge_id_left": left_id,
                    "edge_id_right": right_id,
                    "fold": fold_id,
                    "effective_fold_ids": "|".join(str(value) for value in sorted(int(context["fold"]) for context in fold_contexts)),
                    "screening_rule": options.edge_screening_rule,
                    "auto_score": fold_scores[(fold_id, "auto_only")],
                    "auto_plus_pair_score": pair_score,
                    "fold_gain": gain,
                    "metric": metadata["primary_metric"],
                    "metric_direction": metadata["metric_direction"],
                    "usable_feature_count": usable_count,
                    "future_lookup_violation_count": future_count,
                })
            except Exception as exc:
                materialization_failed = True
                error_reason = str(exc)
                usable_counts.append(0)
                fold_baseline_scores_for_pair.append(float(fold_scores[(fold_id, "auto_only")]))
                fold_ids_for_pair.append(fold_id)
                pair_fold_rows.append({
                    "dataset": dataset_name,
                    "task": task_name,
                    "pair_id": pair_id,
                    "pair_rank": pair_rank,
                    "edge_id_left": left_id,
                    "edge_id_right": right_id,
                    "fold": fold_id,
                    "effective_fold_ids": "|".join(str(value) for value in sorted(int(context["fold"]) for context in fold_contexts)),
                    "screening_rule": options.edge_screening_rule,
                    "auto_score": fold_scores[(fold_id, "auto_only")],
                    "auto_plus_pair_score": math.nan,
                    "fold_gain": math.nan,
                    "metric": metadata["primary_metric"],
                    "metric_direction": metadata["metric_direction"],
                    "usable_feature_count": 0,
                    "future_lookup_violation_count": 0,
                })
        aggregate_baseline_score = None
        aggregate_candidate_score = None
        aggregate_gain = None
        if options.edge_screening_rule == "pooled_oof" and pooled_labels:
            y_true = pd.concat([pd.Series(values) for values in pooled_labels], ignore_index=True)
            auto_pred = _concat_predictions(pooled_auto_predictions)
            pair_pred = _concat_predictions(pooled_pair_predictions)
            aggregate_baseline_score = _metric_score(
                y_true,
                auto_pred,
                metric=metadata["primary_metric"],
                problem_type=metadata["problem_type"],
            )
            aggregate_candidate_score = _metric_score(
                y_true,
                pair_pred,
                metric=metadata["primary_metric"],
                problem_type=metadata["problem_type"],
            )
            aggregate_gain = metric_improvement(
                candidate=aggregate_candidate_score,
                reference=aggregate_baseline_score,
                direction=metadata["metric_direction"],
            )
        summary = summarize_pair_screening(
            gains=gains,
            usable_feature_counts=usable_counts,
            future_lookup_violation_count=future_violations,
            min_delta=options.edge_screening_min_delta,
            min_positive_folds=min_positive_folds,
            screening_rule=options.edge_screening_rule,
            aggregate_gain=aggregate_gain,
            aggregate_baseline_score=aggregate_baseline_score,
            aggregate_candidate_score=aggregate_candidate_score,
            fold_baseline_scores=fold_baseline_scores_for_pair,
            fold_ids=fold_ids_for_pair,
            max_relative_fold_degradation=options.edge_screening_max_relative_fold_degradation,
            materialization_failed=materialization_failed,
        )
        if materialization_failed and error_reason:
            summary["materialization_error"] = error_reason
        pair_rows.append({
            "dataset": dataset_name,
            "task": task_name,
            "pair_id": pair_id,
            "pair_rank": pair_rank,
            "edge_id_left": left_id,
            "edge_id_right": right_id,
            "effective_fold_ids": "|".join(str(value) for value in sorted(int(context["fold"]) for context in fold_contexts)),
            "screening_rule": options.edge_screening_rule,
            **summary,
            "selected_initial_pair": False,
        })
    accepted_pairs = [
        row for row in pair_rows
        if row.get("screening_status") == "screened_in"
        and (
            (
                options.edge_screening_rule == "pooled_oof"
                and np.isfinite(row.get("aggregate_gain", math.nan))
                and float(row["aggregate_gain"]) > float(options.edge_screening_min_delta)
            )
            or (
                options.edge_screening_rule != "pooled_oof"
                and np.isfinite(row.get("mean_gain", math.nan))
                and float(row["mean_gain"]) > 0.0
                and int(row.get("passing_fold_count", 0)) >= int(min_positive_folds)
            )
        )
    ]
    if accepted_pairs:
        winner = min(
            accepted_pairs,
            key=lambda row: (
                -float(row["aggregate_gain"] if options.edge_screening_rule == "pooled_oof" else row["mean_gain"]),
                int(row["pair_rank"]),
            ),
        )
        winner["selected_initial_pair"] = True
        selected_edges = [
            dict(edge_by_id[str(winner["edge_id_left"])]),
            dict(edge_by_id[str(winner["edge_id_right"])]),
        ]
    return {
        "pair_screening": pair_rows,
        "pair_screening_fold_metrics": pair_fold_rows,
        "continuous_discretization_audit": continuous_discretization_audit,
        "continuous_discretization_boundaries": continuous_discretization_boundaries,
        "selected_edges": selected_edges,
    }


def _pairwise_rescue_reason(
    *,
    strategy: str,
    enabled: bool,
    screened_edges: Sequence[Mapping[str, Any]],
    pair_screening_rows: Sequence[Mapping[str, Any]],
) -> str:
    if strategy != "greedy":
        return "not_greedy_strategy"
    if not enabled:
        return "disabled"
    if not pair_screening_rows:
        return "not_attempted_single_edge_passed" if screened_edges else "not_attempted_insufficient_candidates"
    if any(row.get("selected_initial_pair") for row in pair_screening_rows):
        return "selected_pair_passed_gate"
    return "no_pair_passed_gate"


def materialize_declared_feature_frame(
    targets: pd.DataFrame,
    *,
    table_dict: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    entity_key: str,
    target_time_col: str,
) -> pd.DataFrame:
    frame = materialize_feature_frame(
        targets,
        table_dict=table_dict,
        features=features,
        entity_key=entity_key,
        target_time_col=target_time_col,
    )
    for feature in features:
        output = feature.get("output_column")
        aux_columns = _auxiliary_columns(feature)
        if not output or output not in frame.columns:
            continue
        for aux in aux_columns:
            if aux not in frame.columns:
                frame[aux] = frame[output].isna().astype("int8")
    return frame


def materialize_declared_feature_frame_pair(
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    *,
    table_dict: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    entity_key: str,
    target_time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_targets.reset_index(drop=True).copy()
    validation = validation_targets.reset_index(drop=True).copy()
    train["__fdhg_pair_split"] = "train"
    train["__fdhg_pair_pos"] = np.arange(len(train), dtype=np.int64)
    validation["__fdhg_pair_split"] = "validation"
    validation["__fdhg_pair_pos"] = np.arange(len(validation), dtype=np.int64)
    combined = pd.concat([train, validation], ignore_index=True, sort=False)
    frame = materialize_declared_feature_frame(
        combined,
        table_dict=table_dict,
        features=features,
        entity_key=entity_key,
        target_time_col=target_time_col,
    )
    train_frame = (
        frame[frame["__fdhg_pair_split"] == "train"]
        .sort_values("__fdhg_pair_pos", kind="mergesort")
        .drop(columns=["__fdhg_pair_split", "__fdhg_pair_pos"])
        .reset_index(drop=True)
    )
    validation_frame = (
        frame[frame["__fdhg_pair_split"] == "validation"]
        .sort_values("__fdhg_pair_pos", kind="mergesort")
        .drop(columns=["__fdhg_pair_split", "__fdhg_pair_pos"])
        .reset_index(drop=True)
    )
    return train_frame, validation_frame


def audit_residual_columns(
    *,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    fold: int | str,
    provenance: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_feature = {
        str(row.get("feature_name")): row
        for row in provenance
        if row.get("feature_name")
    }
    rows: list[dict[str, Any]] = []
    usable_cols: list[str] = []
    for col in sorted(feature_cols):
        series = frame[col] if col in frame.columns else pd.Series(dtype="float64")
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        observed_count = int(series.notna().sum())
        finite_count = int(finite.shape[0])
        unique_count = int(finite.nunique(dropna=True)) if finite_count else 0
        variance = float(finite.var(ddof=0)) if finite_count else np.nan
        rejection_reason = ""
        if observed_count <= 0:
            rejection_reason = "all_values_missing"
        elif finite_count <= 0:
            rejection_reason = "zero_finite_values"
        elif unique_count <= 1:
            rejection_reason = (
                "constant_missing_indicator"
                if str(col).endswith("__is_missing")
                else "zero_variance"
            )
        elif np.isfinite(variance) and variance == 0.0:
            rejection_reason = "zero_variance"
        usable = rejection_reason == ""
        if usable:
            usable_cols.append(col)
        prov = by_feature.get(str(col), {})
        rows.append({
            "fold": fold,
            "edge_id": prov.get("edge_id", ""),
            "feature_name": col,
            "observed_count": observed_count,
            "finite_count": finite_count,
            "non_missing_rate": float(observed_count / max(1, len(series))),
            "unique_count": unique_count,
            "variance": variance,
            "usable": usable,
            "rejection_reason": rejection_reason,
        })
    return rows, usable_cols


def final_refit_and_evaluate(
    *,
    dataset_name: str = "",
    task_name: str = "",
    selected_variant: str,
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    dfs_features: Sequence[Mapping[str, Any]],
    auto_features: Sequence[Mapping[str, Any]],
    fdhg_edges: Sequence[Mapping[str, Any]],
    join_keys: Sequence[str],
    options: AutoFdhgOptions,
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    final_feature_audit: list[dict[str, Any]] = []
    continuous_discretization_audit: list[dict[str, Any]] = []
    continuous_discretization_boundaries: dict[str, Any] = {}
    final_usable_residual_features = 0
    fdhg_final_refit_skipped_reason = ""
    final_counters = {
        "official_validation_decoder_fit_count": 0,
        "official_validation_prediction_count": 0,
        "fdhg_mapping_fit_count": 0,
        "fdhg_materialization_count": 0,
        "target_lookup_count": 0,
    }
    if selected_variant == "dfs_fallback":
        fdhg_final_refit_skipped_reason = "globally_selected_variant_is_dfs_fallback"
        train_x, val_x = materialize_declared_feature_frame_pair(
            train_targets,
            validation_targets,
            table_dict=table_dict,
            features=dfs_features,
            entity_key=metadata["entity_key"],
            target_time_col=metadata["target_time_col"],
        )
        cols = _declared_model_columns(dfs_features)
    else:
        auto_train, auto_val = materialize_declared_feature_frame_pair(
            train_targets,
            validation_targets,
            table_dict=table_dict,
            features=auto_features,
            entity_key=metadata["entity_key"],
            target_time_col=metadata["target_time_col"],
        )
        if selected_variant == "auto_only":
            fdhg_final_refit_skipped_reason = "globally_selected_variant_is_auto_only"
            train_x, val_x = auto_train, auto_val
            cols = _declared_model_columns(auto_features)
        else:
            fit_horizon = train_targets[metadata["target_time_col"]].max()
            final_counters["fdhg_mapping_fit_count"] += 1
            fitted = fit_afd_edges(
                inner_train_rows=train_targets,
                source_tables=table_dict,
                schema=None,
                task_metadata=metadata,
                candidate_edges=fdhg_edges,
                max_edges=options.max_fdhg_edges,
                fold="full_train",
                fit_horizon=fit_horizon,
                continuous_fdhg_mode=options.continuous_fdhg_mode,
                continuous_fdhg_bins=options.continuous_fdhg_bins,
                continuous_fdhg_min_effective_bins=options.continuous_fdhg_min_effective_bins,
                dataset=dataset_name,
                task=task_name,
                fit_split="official_train",
            )
            continuous_discretization_audit.extend([
                row
                for edge in fitted
                for row in ((edge.continuous_discretization or {}).get("audit", []) if isinstance(edge.continuous_discretization, Mapping) else [])
            ])
            continuous_discretization_boundaries.update({
                str(edge.edge_id): (edge.continuous_discretization or {}).get("boundaries", {})
                for edge in fitted
                if isinstance(edge.continuous_discretization, Mapping)
            })
            source_entity_columns_by_edge = {
                str(edge.get("edge_id", "")): str(edge.get("source_entity_column", ""))
                for edge in fdhg_edges
                if edge.get("source_entity_column")
            }

            # Preserve the same event-row lookup semantics used during
            # train-only fold evaluation.  In particular, DBInfer targets
            # use a prediction-row identity (__row_id) that differs from
            # the relational lookup key (for example queryId/itemId).
            target_lookup_columns_by_edge = {
                str(edge.get("edge_id", "")): str(edge.get("target_lookup_column", ""))
                for edge in fdhg_edges
                if edge.get("target_lookup_column")
            }

            strict_before_by_edge = {
                str(edge.get("edge_id", "")): bool(edge.get("strict_before", False))
                for edge in fdhg_edges
                if "strict_before" in edge
            }

            target_lookup_value_mappings_by_edge = {
                str(edge.get("edge_id", "")):
                    target_lookup_value_mapping
                for edge in fdhg_edges
                if (
                    target_lookup_value_mapping is not None
                    and edge.get(
                        "target_lookup_value_transform"
                    )
                    == "dbinfer_inverse_entity_mapping"
                )
            }

            final_counters["fdhg_materialization_count"] += 1
            final_counters["target_lookup_count"] += 1
            fdhg_train, _, _ = materialize_ambiguity_features(
                fitted_edges=fitted,
                target_rows=train_targets,
                source_tables=table_dict,
                task_metadata=metadata,
                source_entity_columns_by_edge=source_entity_columns_by_edge,
                target_lookup_columns_by_edge=target_lookup_columns_by_edge,
                target_lookup_value_mappings_by_edge=(
                    target_lookup_value_mappings_by_edge
                ),
                strict_before_by_edge=strict_before_by_edge,
            )
            final_counters["fdhg_materialization_count"] += 1
            final_counters["target_lookup_count"] += 1
            fdhg_val, _, _ = materialize_ambiguity_features(
                fitted_edges=fitted,
                target_rows=validation_targets,
                source_tables=table_dict,
                task_metadata=metadata,
                source_entity_columns_by_edge=source_entity_columns_by_edge,
                target_lookup_columns_by_edge=target_lookup_columns_by_edge,
                target_lookup_value_mappings_by_edge=(
                    target_lookup_value_mappings_by_edge
                ),
                strict_before_by_edge=strict_before_by_edge,
            )
            fdhg_cols = feature_columns(fdhg_train, join_keys, metadata)
            final_feature_audit, usable_fdhg_cols = audit_residual_columns(
                frame=fdhg_train,
                feature_cols=fdhg_cols,
                fold="full_train",
                provenance=[],
            )
            final_usable_residual_features = len(usable_fdhg_cols)
            train_x = align_feature_blocks(
                target_rows=train_targets,
                blocks=[("auto", auto_train), ("fdhg", fdhg_train)],
                join_keys=join_keys,
                metadata=metadata,
            )
            val_x = align_feature_blocks(
                target_rows=validation_targets,
                blocks=[("auto", auto_val), ("fdhg", fdhg_val)],
                join_keys=join_keys,
                metadata=metadata,
            )
            cols = [*_declared_model_columns(auto_features), *usable_fdhg_cols]
    final_counters["official_validation_decoder_fit_count"] += 1
    model = _fit_model(
        train_x[cols] if cols else pd.DataFrame(index=train_x.index),
        train_targets[metadata["label_col"]],
        problem_type=metadata["problem_type"],
        options=_auto_options(options),
    )
    final_counters["official_validation_prediction_count"] += 1
    pred = _predict_model(
        model,
        val_x[cols] if cols else pd.DataFrame(index=val_x.index),
        problem_type=metadata["problem_type"],
    )
    score = _metric_score(
        validation_targets[metadata["label_col"]].reset_index(drop=True),
        pred,
        metric=metadata["primary_metric"],
        problem_type=metadata["problem_type"],
    )
    predictions = validation_targets[list(join_keys)].copy()
    predictions["prediction"] = pred
    predictions["label"] = validation_targets[metadata["label_col"]].to_numpy()
    return {
        "official_validation_score": score,
        "official_validation_metrics": {
            "primary_metric": metadata["primary_metric"],
            "metric_direction": metadata["metric_direction"],
            metadata["primary_metric"]: score,
            "selected_variant": selected_variant,
            "n_features": len(cols),
            "fdhg_final_refit_usable_features": final_usable_residual_features,
            "fdhg_final_refit_skipped_reason": fdhg_final_refit_skipped_reason,
        },
        "official_validation_predictions": predictions,
        "fdhg_final_refit_usable_features": final_usable_residual_features,
        "fdhg_final_refit_skipped_reason": fdhg_final_refit_skipped_reason,
        "fdhg_final_feature_audit": final_feature_audit,
        "continuous_discretization_audit": continuous_discretization_audit,
        "continuous_discretization_boundaries": continuous_discretization_boundaries,
        "run_counters": final_counters,
    }


def select_joint_variant(
    *,
    mean_scores: Mapping[str, float],
    metric_direction: str,
    min_delta: float,
) -> dict[str, Any]:
    best_variant = "dfs_fallback"
    best_score = float(mean_scores["dfs_fallback"])
    trials = []
    for candidate in ("auto_only", "auto_plus_fdhg"):
        if candidate not in mean_scores:
            continue
        candidate_score = float(mean_scores[candidate])
        improvement = metric_improvement(
            candidate=candidate_score,
            reference=best_score,
            direction=metric_direction,
        )
        passed = bool(
            improvement > min_delta
            and not math.isclose(improvement, min_delta, rel_tol=1e-12, abs_tol=1e-12)
        )
        trials.append({
            "variant": candidate,
            "reference_variant": best_variant,
            "mean_score": candidate_score,
            "reference_score": best_score,
            "improvement": improvement,
            "min_delta": min_delta,
            "passed": passed,
            "rejection_reason": "" if passed else "improvement_not_greater_than_min_delta",
            "complexity_rank": {"dfs_fallback": 0, "auto_only": 1, "auto_plus_fdhg": 2}[candidate],
        })
        if passed:
            best_variant = candidate
            best_score = candidate_score
    return {
        "selected_variant": best_variant,
        "selection_reason": f"selected_{best_variant}_by_train_only_joint_gate",
        "gate_trials": trials,
    }


def metric_improvement(*, candidate: float, reference: float, direction: str) -> float:
    if str(direction) in {"lower", "lower_is_better"}:
        return float(reference - candidate)
    if str(direction) in {"higher", "higher_is_better"}:
        return float(candidate - reference)
    raise ValueError(f"unsupported_metric_direction:{direction}")


def resolve_edge_screening_min_positive_folds(
    options: AutoFdhgOptions,
    *,
    effective_selection_folds: int | None = None,
) -> int:
    return resolve_edge_screening_min_positive_folds_for_effective(
        options,
        effective_selection_folds=(
            options.selection_folds
            if effective_selection_folds is None
            else effective_selection_folds
        ),
    )


def resolve_edge_screening_min_positive_folds_for_effective(
    options: AutoFdhgOptions,
    *,
    effective_selection_folds: int,
) -> int:
    value = options.edge_screening_min_positive_folds
    effective = max(1, int(effective_selection_folds))
    if options.edge_screening_rule in {"positive_fraction", "pooled_oof"}:
        fraction = Decimal(str(options.edge_screening_min_positive_fraction))
        resolved = int((Decimal(effective) * fraction).to_integral_value(rounding=ROUND_CEILING))
        resolved = max(1, resolved)
    else:
        resolved = int(math.ceil(options.selection_folds / 2)) if value is None else int(value)
    if resolved < 1:
        raise ValueError("edge_screening_min_positive_folds_must_be_at_least_1")
    if options.edge_screening_rule == "fixed_count" and resolved > int(options.selection_folds):
        raise ValueError("edge_screening_min_positive_folds_exceeds_selection_folds")
    return resolved


def resolve_max_selected_fdhg_edges(options: AutoFdhgOptions) -> int | None:
    value = options.max_selected_fdhg_edges
    if value is None:
        return None
    resolved = int(value)
    if resolved < 0:
        raise ValueError("max_selected_fdhg_edges_must_be_non_negative")
    return resolved


def edge_screening_delta_passes(*, delta: float, min_delta: float) -> bool:
    return bool(np.isfinite(delta) and float(delta) > float(min_delta))


def edge_screening_positive_fold_passes(*, delta: float) -> bool:
    return edge_screening_delta_passes(delta=delta, min_delta=0.0)


def edge_fold_gain(*, auto_score: float, auto_plus_single_edge_score: float, direction: str) -> float:
    if str(direction) in {"lower", "lower_is_better"}:
        return float(auto_score - auto_plus_single_edge_score)
    if str(direction) in {"higher", "higher_is_better"}:
        return float(auto_plus_single_edge_score - auto_score)
    raise ValueError(f"unsupported_metric_direction:{direction}")


def summarize_edge_screening(
    *,
    gains: Sequence[float],
    usable_feature_counts: Sequence[int],
    future_lookup_violation_count: int,
    min_delta: float,
    min_positive_folds: int,
    screening_rule: str = "fixed_count",
    aggregate_gain: float | None = None,
    aggregate_auto_score: float | None = None,
    aggregate_candidate_score: float | None = None,
    fold_auto_scores: Sequence[float] | None = None,
    fold_ids: Sequence[int | str] | None = None,
    max_relative_fold_degradation: float | None = None,
    materialization_failed: bool = False,
) -> dict[str, Any]:
    usable_counts = [int(value) for value in usable_feature_counts]
    finite_gains = [float(value) for value in gains if np.isfinite(value)]
    mean_gain = float(np.mean(finite_gains)) if finite_gains else math.nan
    positive_fold_count = int(
        sum(edge_screening_positive_fold_passes(delta=value) for value in finite_gains)
    )
    worst_fold_gain = float(np.min(finite_gains)) if finite_gains else math.nan
    if aggregate_gain is None:
        aggregate_gain = mean_gain
    relative_degradation = 0.0
    relative_degradation_fold = ""
    if fold_auto_scores is not None:
        auto_values = list(fold_auto_scores)
        ids = list(fold_ids or range(len(auto_values)))
        degradation_rows: list[tuple[float, Any]] = []
        for idx, gain in enumerate(gains):
            if idx >= len(auto_values) or not np.isfinite(gain) or not np.isfinite(auto_values[idx]):
                continue
            value = max(0.0, -float(gain)) / max(abs(float(auto_values[idx])), 1e-12)
            degradation_rows.append((value, ids[idx] if idx < len(ids) else idx))
        if degradation_rows:
            relative_degradation, relative_degradation_fold = max(
                degradation_rows,
                key=lambda item: (item[0], str(item[1])),
            )
    degradation_passed = (
        max_relative_fold_degradation is None
        or relative_degradation <= float(max_relative_fold_degradation)
    )
    if materialization_failed:
        status = "materialization_failed"
        reason = "materialization_error"
    elif not usable_counts or max(usable_counts) <= 0:
        status = "no_usable_features"
        reason = "no_usable_features"
    else:
        aggregate_passed = bool(np.isfinite(aggregate_gain) and float(aggregate_gain) > float(min_delta))
        mean_passed = bool(np.isfinite(mean_gain) and mean_gain > float(min_delta))
        if screening_rule == "pooled_oof":
            gain_passed = aggregate_passed
            folds_passed = True
        else:
            gain_passed = mean_passed
            folds_passed = positive_fold_count >= int(min_positive_folds)
        status = "screened_in" if gain_passed and folds_passed and degradation_passed else "screened_out"
        if status == "screened_in":
            reason = ""
        elif not gain_passed:
            reason = "non_positive_aggregate_gain" if screening_rule == "pooled_oof" else "non_positive_mean_gain"
        else:
            reason = "excessive_worst_fold_relative_degradation" if not degradation_passed else "insufficient_positive_folds"
    return {
        "mean_gain": mean_gain,
        "aggregate_gain": float(aggregate_gain) if aggregate_gain is not None else math.nan,
        "aggregate_baseline_score": float(aggregate_auto_score) if aggregate_auto_score is not None else math.nan,
        "aggregate_auto_score": float(aggregate_auto_score) if aggregate_auto_score is not None else math.nan,
        "aggregate_candidate_score": float(aggregate_candidate_score) if aggregate_candidate_score is not None else math.nan,
        "positive_fold_count": positive_fold_count,
        "total_fold_count": len(finite_gains),
        "worst_fold_gain": worst_fold_gain,
        "worst_fold_relative_degradation": relative_degradation,
        "worst_fold_relative_degradation_fold": relative_degradation_fold,
        "screening_rule": screening_rule,
        "gain_std": float(np.std(finite_gains, ddof=0)) if finite_gains else math.nan,
        "screening_min_delta": float(min_delta),
        "screening_min_positive_folds": int(min_positive_folds),
        "screening_max_relative_fold_degradation": (
            "" if max_relative_fold_degradation is None else float(max_relative_fold_degradation)
        ),
        "screening_status": status,
        "rejection_reason": reason,
        "usable_feature_count_min": min(usable_counts) if usable_counts else 0,
        "usable_feature_count_max": max(usable_counts) if usable_counts else 0,
        "future_lookup_violation_count": int(future_lookup_violation_count),
    }


def summarize_pair_screening(
    *,
    gains: Sequence[float],
    usable_feature_counts: Sequence[int],
    future_lookup_violation_count: int,
    min_delta: float,
    min_positive_folds: int,
    screening_rule: str = "fixed_count",
    aggregate_gain: float | None = None,
    aggregate_baseline_score: float | None = None,
    aggregate_candidate_score: float | None = None,
    fold_baseline_scores: Sequence[float] | None = None,
    fold_ids: Sequence[int | str] | None = None,
    max_relative_fold_degradation: float | None = None,
    materialization_failed: bool = False,
) -> dict[str, Any]:
    usable_counts = [int(value) for value in usable_feature_counts]
    finite_gains = [float(value) for value in gains if np.isfinite(value)]
    mean_gain = float(np.mean(finite_gains)) if finite_gains else math.nan
    if aggregate_gain is None:
        aggregate_gain = mean_gain
    positive_fold_count = int(
        sum(edge_screening_positive_fold_passes(delta=value) for value in finite_gains)
    )
    passing_fold_count = int(
        sum(edge_screening_delta_passes(delta=value, min_delta=min_delta) for value in finite_gains)
    )
    relative_degradation = 0.0
    relative_degradation_fold = ""
    if fold_baseline_scores is not None:
        baseline_values = list(fold_baseline_scores)
        ids = list(fold_ids or range(len(baseline_values)))
        degradation_rows: list[tuple[float, Any]] = []
        for idx, gain in enumerate(gains):
            if idx >= len(baseline_values) or not np.isfinite(gain) or not np.isfinite(baseline_values[idx]):
                continue
            value = max(0.0, -float(gain)) / max(abs(float(baseline_values[idx])), 1e-12)
            degradation_rows.append((value, ids[idx] if idx < len(ids) else idx))
        if degradation_rows:
            relative_degradation, relative_degradation_fold = max(
                degradation_rows,
                key=lambda item: (item[0], str(item[1])),
            )
    degradation_passed = (
        max_relative_fold_degradation is None
        or relative_degradation <= float(max_relative_fold_degradation)
    )
    if materialization_failed:
        status = "materialization_failed"
        reason = "materialization_error"
    elif not usable_counts or max(usable_counts) <= 0:
        status = "no_usable_features"
        reason = "no_usable_features"
    else:
        if screening_rule == "pooled_oof":
            gain_passed = bool(np.isfinite(aggregate_gain) and float(aggregate_gain) > float(min_delta))
            folds_passed = True
        else:
            gain_passed = bool(np.isfinite(mean_gain) and mean_gain > 0.0)
            folds_passed = passing_fold_count >= int(min_positive_folds)
        status = "screened_in" if gain_passed and folds_passed and degradation_passed else "screened_out"
        if status == "screened_in":
            reason = ""
        elif not gain_passed:
            reason = "non_positive_aggregate_gain" if screening_rule == "pooled_oof" else "non_positive_mean_gain"
        else:
            reason = "excessive_worst_fold_relative_degradation" if not degradation_passed else "insufficient_delta_passing_folds"
    return {
        "mean_gain": mean_gain,
        "aggregate_baseline_score": float(aggregate_baseline_score) if aggregate_baseline_score is not None else math.nan,
        "aggregate_candidate_score": float(aggregate_candidate_score) if aggregate_candidate_score is not None else math.nan,
        "aggregate_gain": float(aggregate_gain) if aggregate_gain is not None else math.nan,
        "positive_fold_count": positive_fold_count,
        "passing_fold_count": passing_fold_count,
        "total_fold_count": len(finite_gains),
        "worst_fold_gain": float(np.min(finite_gains)) if finite_gains else math.nan,
        "worst_fold_relative_degradation": relative_degradation,
        "worst_fold_relative_degradation_fold": relative_degradation_fold,
        "gain_std": float(np.std(finite_gains, ddof=0)) if finite_gains else math.nan,
        "screening_rule": screening_rule,
        "screening_min_delta": float(min_delta),
        "screening_min_positive_folds": int(min_positive_folds),
        "screening_max_relative_fold_degradation": (
            "" if max_relative_fold_degradation is None else float(max_relative_fold_degradation)
        ),
        "screening_status": status,
        "rejection_reason": reason,
        "usable_feature_count_min": min(usable_counts) if usable_counts else 0,
        "usable_feature_count_max": max(usable_counts) if usable_counts else 0,
        "future_lookup_violation_count": int(future_lookup_violation_count),
    }


def _best_variant_for_scores(scores: Mapping[str, float], *, direction: str) -> str:
    ordered = [variant for variant in VARIANTS if variant in scores]
    if str(direction) in {"lower", "lower_is_better"}:
        return min(ordered, key=lambda variant: (float(scores[variant]), ordered.index(variant)))
    if str(direction) in {"higher", "higher_is_better"}:
        return max(ordered, key=lambda variant: (float(scores[variant]), -ordered.index(variant)))
    raise ValueError(f"unsupported_metric_direction:{direction}")


def score_matrix(
    *,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    train_y: pd.Series,
    val_y: pd.Series,
    feature_cols: Sequence[str],
    metadata: Mapping[str, Any],
    options: AutoFdhgOptions,
) -> float:
    model = _fit_model(
        train_x[list(feature_cols)] if feature_cols else pd.DataFrame(index=train_x.index),
        train_y.reset_index(drop=True),
        problem_type=metadata["problem_type"],
        options=_auto_options(options),
    )
    pred = _predict_model(
        model,
        val_x[list(feature_cols)] if feature_cols else pd.DataFrame(index=val_x.index),
        problem_type=metadata["problem_type"],
    )
    return _metric_score(
        val_y.reset_index(drop=True),
        pred,
        metric=metadata["primary_metric"],
        problem_type=metadata["problem_type"],
    )


def score_matrix_with_predictions(
    *,
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    train_y: pd.Series,
    val_y: pd.Series,
    feature_cols: Sequence[str],
    metadata: Mapping[str, Any],
    options: AutoFdhgOptions,
) -> dict[str, Any]:
    model = _fit_model(
        train_x[list(feature_cols)] if feature_cols else pd.DataFrame(index=train_x.index),
        train_y.reset_index(drop=True),
        problem_type=metadata["problem_type"],
        options=_auto_options(options),
    )
    pred = _predict_model(
        model,
        val_x[list(feature_cols)] if feature_cols else pd.DataFrame(index=val_x.index),
        problem_type=metadata["problem_type"],
    )
    score = _metric_score(
        val_y.reset_index(drop=True),
        pred,
        metric=metadata["primary_metric"],
        problem_type=metadata["problem_type"],
    )
    return {"score": score, "prediction": pred}


def _concat_predictions(predictions: Sequence[Any]) -> Any:
    arrays = [np.asarray(pred) for pred in predictions]
    if not arrays:
        return np.asarray([])
    return np.concatenate(arrays, axis=0)


def align_feature_blocks(
    *,
    target_rows: pd.DataFrame,
    blocks: Sequence[tuple[str, pd.DataFrame]],
    join_keys: Sequence[str],
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    base_cols = list(dict.fromkeys([*join_keys, metadata["label_col"]]))
    out = target_rows[base_cols].reset_index(drop=True).copy()
    _assert_unique_keys(out, join_keys, "target")
    for name, block in blocks:
        _assert_unique_keys(block, join_keys, name)
        feature_cols = feature_columns(block, join_keys, metadata)
        merged = out[list(join_keys)].merge(
            block[list(join_keys) + feature_cols],
            on=list(join_keys),
            how="left",
            validate="one_to_one",
            sort=False,
            indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"missing_{name}_feature_rows")
        extra = block[list(join_keys)].merge(
            out[list(join_keys)],
            on=list(join_keys),
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not extra["_merge"].eq("both").all():
            raise ValueError(f"extra_{name}_feature_rows")
        for col in feature_cols:
            out[col] = merged[col].to_numpy()
        if metadata["label_col"] in block.columns:
            aligned_label = out[list(join_keys) + [metadata["label_col"]]].merge(
                block[list(join_keys) + [metadata["label_col"]]],
                on=list(join_keys),
                how="left",
                suffixes=("_target", "_block"),
                validate="one_to_one",
            )
            if not aligned_label[f"{metadata['label_col']}_target"].equals(
                aligned_label[f"{metadata['label_col']}_block"]
            ):
                raise ValueError(f"label_mismatch:{name}")
        time_col = metadata["target_time_col"]
        if time_col in join_keys and time_col in block.columns:
            # Presence in the key already verifies exact timestamp alignment.
            pass
    return out


def feature_columns(
    frame: pd.DataFrame,
    join_keys: Sequence[str],
    metadata: Mapping[str, Any],
) -> list[str]:
    excluded = set(join_keys) | {metadata.get("entity_key"), metadata.get("target_time_col"), metadata.get("label_col")}
    return sorted([col for col in frame.columns if col not in excluded])


def resolve_join_keys(dataset_name: str, task_name: str, metadata: Mapping[str, Any]) -> list[str]:
    if dataset_name == "rel-f1" and task_name == "driver-position":
        time_col = "date" if metadata.get("target_time_col") == "date" else str(metadata["target_time_col"])
        return [str(metadata["entity_key"]), time_col]
    if dataset_name == "rel-ratebeer" and task_name == "user-count":
        return [str(metadata["entity_key"]), str(metadata["target_time_col"])]
    return [str(metadata["entity_key"]), str(metadata["target_time_col"])]


def resolve_canonical_dfs_features(
    *,
    dataset_name: str,
    task_name: str,
    root: Path,
    metadata: Mapping[str, Any],
    explicit_config: Path | None = None,
) -> dict[str, Any]:
    requested = dataset_name
    canonical = canonical_relbench_dataset_name(dataset_name)
    requested_slug = f"{requested}_{task_name}"
    canonical_slug = f"{canonical}_{task_name}"
    if explicit_config is not None:
        result = _load_feature_config_source(
            explicit_config,
            metadata=metadata,
            source_type="explicit_override",
            requested_dataset_name=requested,
            canonical_dataset_name=canonical,
            task_name=task_name,
        )
        if result["features"]:
            return result
    config = root / "configs" / "reproduction" / "tasks.yaml"
    if config.exists():
        with config.open("r", encoding="utf-8") as fh:
            tasks = (yaml.safe_load(fh) or {}).get("tasks", {})
        entry = tasks.get(f"{dataset_name}/{task_name}")
        if entry and "dfs" in entry:
            features = _dfs_features_from_reproduction_config(entry["dfs"], metadata)
            if features:
                return {
                    "features": features,
                    "provenance": {
                        "source_type": "configs/reproduction/tasks.yaml",
                        "path": str(config),
                        "task_key": f"{dataset_name}/{task_name}",
                        "sha256": _file_sha256(config),
                    },
                    "blocker": "",
                }
    onboarding_dirs = [
        root,
        root / canonical_slug,
        root / requested_slug,
        root / "outputs" / "onboarding" / canonical_slug,
        root / "outputs" / "onboarding" / requested_slug,
    ]
    for directory in onboarding_dirs:
        loaded = _load_canonical_onboarding_artifact(
            directory,
            metadata=metadata,
            requested_dataset_name=requested,
            canonical_dataset_name=canonical,
            task_name=task_name,
        )
        if loaded["features"] or loaded["blocker"]:
            return loaded
    e2e_candidates = [
        root / "outputs" / "auto-fdhg-dfs" / requested_slug / "dfs_feature_config.json",
        root / "outputs" / "auto-fdhg-dfs" / canonical_slug / "dfs_feature_config.json",
        root / "outputs" / "e2e" / requested_slug / "dfs_corrected_canonical" / "dfs_feature_config.json",
        root / "outputs" / "e2e" / requested_slug / "dfs" / "dfs_feature_config.json",
        root / "outputs" / "e2e" / canonical_slug / "dfs_corrected_canonical" / "dfs_feature_config.json",
        root / "outputs" / "e2e" / canonical_slug / "dfs" / "dfs_feature_config.json",
    ]
    for path in e2e_candidates:
        if path.exists():
            result = _load_feature_config_source(
                path,
                metadata=metadata,
                source_type="canonical_e2e_artifact",
                requested_dataset_name=requested,
                canonical_dataset_name=canonical,
                task_name=task_name,
            )
            if result["features"]:
                return result
    return {
        "features": [],
        "provenance": {},
        "blocker": f"missing_canonical_dfs_source:{dataset_name}/{task_name}",
    }


def canonical_relbench_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name)
    return name if name.startswith("relbench-v1-") else f"relbench-v1-{name}"


def _load_feature_config_source(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    source_type: str,
    requested_dataset_name: str,
    canonical_dataset_name: str,
    task_name: str,
) -> dict[str, Any]:
    if not path.exists():
        return {"features": [], "provenance": {}, "blocker": ""}
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = _features_from_baseline_config(payload, metadata=metadata)
    return {
        "features": features,
        "provenance": _dfs_provenance(
            source_type=source_type,
            source_directory=path.parent,
            config_path=path,
            manifest_path=None,
            onboarding_manifest_path=None,
            requested_dataset_name=requested_dataset_name,
            canonical_dataset_name=canonical_dataset_name,
            task_name=task_name,
            features=features,
        ),
        "blocker": "" if features else f"empty_canonical_dfs_config:{path}",
    }


def _load_canonical_onboarding_artifact(
    directory: Path,
    *,
    metadata: Mapping[str, Any],
    requested_dataset_name: str,
    canonical_dataset_name: str,
    task_name: str,
) -> dict[str, Any]:
    config_path = directory / "baseline_feature_config.json"
    if not config_path.exists():
        return {"features": [], "provenance": {}, "blocker": ""}
    manifest_path = directory / "baseline_feature_manifest.csv"
    onboarding_manifest_path = directory / "onboarding_manifest.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    features = _features_from_baseline_config(payload, metadata=metadata)
    if not features:
        return {
            "features": [],
            "provenance": {},
            "blocker": f"empty_canonical_dfs_config:{config_path}",
        }
    blocker = _cross_check_onboarding_artifact(
        features=features,
        manifest_path=manifest_path,
        onboarding_manifest_path=onboarding_manifest_path,
        requested_dataset_name=requested_dataset_name,
        canonical_dataset_name=canonical_dataset_name,
        task_name=task_name,
    )
    provenance = _dfs_provenance(
        source_type="canonical_onboarding_artifact",
        source_directory=directory,
        config_path=config_path,
        manifest_path=manifest_path,
        onboarding_manifest_path=onboarding_manifest_path,
        requested_dataset_name=requested_dataset_name,
        canonical_dataset_name=canonical_dataset_name,
        task_name=task_name,
        features=features,
    )
    return {
        "features": [] if blocker else features,
        "provenance": provenance,
        "blocker": blocker,
    }


def _cross_check_onboarding_artifact(
    *,
    features: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    onboarding_manifest_path: Path,
    requested_dataset_name: str,
    canonical_dataset_name: str,
    task_name: str,
) -> str:
    expected_primary = {str(row["output_column"]) for row in features}
    expected_model = set(_declared_model_columns(features))
    if not manifest_path.exists():
        return f"missing_canonical_dfs_manifest:{manifest_path}"
    if not onboarding_manifest_path.exists():
        return f"missing_canonical_onboarding_manifest:{onboarding_manifest_path}"
    manifest = pd.read_csv(manifest_path)
    manifest_cols = set()
    for col in ("output_column", "feature_name"):
        if col in manifest.columns:
            manifest_cols.update(manifest[col].dropna().astype(str))
    if manifest_cols and not expected_primary.issubset(manifest_cols) and not expected_model.issubset(manifest_cols):
        return "canonical_dfs_config_manifest_disagree:feature_names"
    onboarding = json.loads(onboarding_manifest_path.read_text(encoding="utf-8"))
    artifact_dataset = str(
        onboarding.get("dataset")
        or onboarding.get("dataset_name")
        or onboarding.get("relbench_dataset")
        or ""
    )
    if artifact_dataset:
        normalized_artifact = canonical_relbench_dataset_name(
            artifact_dataset.removeprefix("relbench-v1-")
            if artifact_dataset.startswith("relbench-v1-")
            else artifact_dataset
        )
        if normalized_artifact != canonical_dataset_name:
            return "canonical_dfs_onboarding_manifest_disagree:dataset"
    artifact_task = str(onboarding.get("task") or onboarding.get("task_name") or "")
    if artifact_task and artifact_task != task_name:
        return "canonical_dfs_onboarding_manifest_disagree:task"
    artifact_strategy = str(onboarding.get("materialization_strategy") or "")
    if artifact_strategy and artifact_strategy != "grouped_temporal_sweep":
        return "canonical_dfs_onboarding_manifest_disagree:materialization_strategy"
    artifact_version = str(onboarding.get("implementation_version") or "")
    feature_versions = {str(row.get("implementation_version")) for row in features if row.get("implementation_version")}
    if artifact_version and feature_versions and artifact_version not in feature_versions:
        return "canonical_dfs_onboarding_manifest_disagree:implementation_version"
    del requested_dataset_name
    return ""


def _dfs_provenance(
    *,
    source_type: str,
    source_directory: Path,
    config_path: Path,
    manifest_path: Path | None,
    onboarding_manifest_path: Path | None,
    requested_dataset_name: str,
    canonical_dataset_name: str,
    task_name: str,
    features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    auxiliary = [aux for row in features for aux in _auxiliary_columns(row)]
    return {
        "source_type": source_type,
        "source_directory": str(source_directory),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
        "onboarding_manifest_path": str(onboarding_manifest_path) if onboarding_manifest_path is not None else "",
        "canonical_dataset_name": canonical_dataset_name,
        "requested_dataset_name": requested_dataset_name,
        "task": task_name,
        "declaration_count": len(features),
        "model_column_count": len(_declared_model_columns(features)),
        "canonical_feature_names": [str(row["output_column"]) for row in features],
        "canonical_auxiliary_columns": auxiliary,
        "materializer": "grouped_temporal_sweep",
        "materialization_strategy": "grouped_temporal_sweep",
        "fold_safe": True,
        "fixed_split_parquets_reused_in_gate": False,
        "sha256": _file_sha256(config_path) if config_path.exists() else "",
    }


def _dfs_features_from_reproduction_config(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = ("child_table", "child_time_col")
    if not all(config.get(key) for key in required):
        return []
    numeric_col = config.get("numeric_col")
    child_table = str(config["child_table"])
    child_fk = str(config.get("child_fk") or metadata["entity_key"])
    child_time_col = str(config["child_time_col"])
    base = {
        "kind": "relational",
        "feature_id": "",
        "child_table": child_table,
        "child_fk": child_fk,
        "child_event_time_col": child_time_col,
        "parent_table": "",
        "parent_key": metadata["entity_key"],
        "temporal_predicate": "child.event_time <= target.target_time",
    }
    specs = [
        ("count", None, f"f_{child_table}_count"),
        ("days_since_last", None, f"f_{child_table}_days_since_last"),
    ]
    if numeric_col:
        specs.extend([
            ("mean", str(numeric_col), f"f_{child_table}_{numeric_col}_mean"),
            ("std", str(numeric_col), f"f_{child_table}_{numeric_col}_std"),
            ("max", str(numeric_col), f"f_{child_table}_{numeric_col}_max"),
        ])
    features = []
    for idx, (agg, source_col, output) in enumerate(specs):
        row = dict(base)
        row.update({
            "feature_id": f"canonical_dfs::{idx}:{output}",
            "source_column": source_col,
            "aggregation": agg,
            "output_column": output,
            "origin": "canonical_dfs",
        })
        features.append(row)
    return features


def _features_from_baseline_config(payload: Any, *, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _raw_feature_rows(payload)
    features: list[dict[str, Any]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, Mapping):
            continue
        source_table = row.get("source_table") or row.get("child_table")
        join_key = row.get("join_key") or row.get("child_fk")
        event_time = row.get("child_event_time_col") or row.get("child_time_col")
        aggregation = row.get("aggregation")
        output_column = row.get("output_column") or row.get("feature_name")
        if not all([source_table, join_key, event_time, aggregation, output_column]):
            continue
        source_column = row.get("source_column")
        if source_column == "":
            source_column = None
        feature = {
            "kind": row.get("kind", "relational"),
            "feature_id": row.get("feature_id") or row.get("primitive_id") or f"canonical_dfs::{idx}:{output_column}",
            "child_table": str(source_table),
            "child_fk": str(join_key),
            "child_event_time_col": str(event_time),
            "parent_table": row.get("parent_table", ""),
            "parent_key": row.get("target_entity_key") or row.get("parent_key") or metadata["entity_key"],
            "source_column": source_column,
            "aggregation": str(aggregation),
            "output_column": str(output_column),
            "temporal_predicate": row.get("temporal_predicate", "child.event_time <= target.target_time"),
            "materialization_strategy": row.get("materialization_strategy", "grouped_temporal_sweep"),
            "origin": "canonical_dfs",
        }
        for key in (
            "primitive_id",
            "program_id",
            "leakage_safe",
            "temporal_safe",
            "implementation_version",
            "auxiliary_output_columns",
        ):
            if key in row:
                feature[key] = _normalize_auxiliary(row[key]) if key == "auxiliary_output_columns" else row[key]
        features.append(feature)
    return features


def _raw_feature_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, Mapping):
        return []
    for key in (
        "features",
        "baseline_features",
        "feature_declarations",
        "declarations",
        "selected_features",
        "dfs_features",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_auxiliary(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(item) for item in decoded]
        except json.JSONDecodeError:
            return [part for part in (v.strip() for v in value.split("|")) if part]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _auxiliary_columns(feature: Mapping[str, Any]) -> list[str]:
    return _normalize_auxiliary(feature.get("auxiliary_output_columns"))


def _declared_model_columns(features: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen = set()
    for feature in features:
        for col in [feature.get("output_column"), *_auxiliary_columns(feature)]:
            if col and col not in seen:
                seen.add(str(col))
                columns.append(str(col))
    return columns


def decoder_config(options: AutoFdhgOptions) -> dict[str, Any]:
    return {
        "decoder": options.selection_decoder,
        "random_seed": options.random_seed,
        "max_iter": 100,
        "min_samples_leaf": 1,
    }


def _auto_options(options: AutoFdhgOptions) -> AutoOnboardingOptions:
    return AutoOnboardingOptions(
        selection_folds=options.selection_folds,
        feature_budget=options.feature_budget,
        min_delta=options.min_delta,
        selection_decoder=options.selection_decoder,
        max_relations=options.max_relations,
        max_numeric_columns=options.max_numeric_columns,
        max_categorical_columns=options.max_categorical_columns,
    )


def _extract_feature_declarations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, Mapping):
        raw = payload.get("selected_features") or payload.get("features") or payload.get("dfs_features") or []
    else:
        raw = []
    features = []
    for idx, row in enumerate(raw):
        if not isinstance(row, Mapping):
            continue
        item = dict(row)
        if "output_column" not in item and "feature_name" in item:
            item["output_column"] = item["feature_name"]
        item.setdefault("feature_id", item.get("output_column", f"feature_{idx}"))
        features.append(item)
    return features


def _assert_unique_keys(frame: pd.DataFrame, join_keys: Sequence[str], name: str) -> None:
    missing = [key for key in join_keys if key not in frame.columns]
    if missing:
        raise ValueError(f"missing_{name}_join_keys:{','.join(missing)}")
    if frame.duplicated(list(join_keys)).any():
        raise ValueError(f"duplicate_{name}_keys")


def _relation_count(features: Sequence[Mapping[str, Any]]) -> int:
    return len({(row.get("child_table"), row.get("child_fk")) for row in features if row.get("kind") != "static_entity"})


def _public_folds(split_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **{k: v for k, v in fold.items() if k not in {"train_indices", "validation_indices"}},
            "single_unique_train_timestamp": int(fold.get("unique_train_timestamps", 0) or 0) < 2,
        }
        for fold in split_plan["folds"]
    ]


def selection_fold_metadata(
    *,
    split_plan: Mapping[str, Any],
    requested_selection_folds: int,
) -> dict[str, Any]:
    public_folds = _public_folds(split_plan)
    effective_fold_ids = [int(fold.get("fold", idx)) for idx, fold in enumerate(public_folds)]
    warnings: list[str] = []
    effective = len(public_folds)
    requested = int(requested_selection_folds)
    if effective < requested:
        warnings.append("effective_selection_folds_below_requested_selection_folds")
    if any(bool(fold.get("single_unique_train_timestamp")) for fold in public_folds):
        warnings.append("fold_with_fewer_than_two_unique_training_timestamps")
    return {
        "requested_selection_folds": requested,
        "effective_selection_folds": effective,
        "effective_fold_ids": effective_fold_ids,
        "fold_quality": [
            {
                "fold": fold.get("fold", ""),
                "unique_train_timestamps": fold.get("unique_train_timestamps", 0),
                "unique_validation_timestamps": fold.get("unique_validation_timestamps", 0),
                "train_rows": fold.get("train_rows", 0),
                "validation_rows": fold.get("validation_rows", 0),
                "single_unique_train_timestamp": bool(fold.get("single_unique_train_timestamp")),
            }
            for fold in public_folds
        ],
        "warnings": warnings,
    }


def _write_outputs(staging: Path, prepared: Mapping[str, Any]) -> None:
    _write_json(staging / "manifest.json", prepared["manifest"])
    _write_json(staging / "candidate_discovery.json", prepared.get("candidate_discovery", {}))
    _write_csv(
        staging / "candidate_column_audit.csv",
        prepared.get("candidate_discovery", {}).get("candidate_column_audit", []),
        fieldnames=[
            "source_table",
            "column",
            "dtype",
            "non_null_count",
            "cardinality",
            "unique_ratio",
            "actual_primary_key",
            "actual_foreign_key",
            "source_entity_column",
            "determinant_eligible",
            "dependent_eligible",
            "exclusion_reason",
            "eligibility_status",
            "original_column",
            "transformed_column",
            "requested_bins",
            "effective_bins",
        ],
    )
    _write_csv(staging / "fold_metrics.csv", prepared["gate"]["fold_metrics"])
    _write_csv(staging / "gate_trials.csv", prepared["gate"]["gate_trials"])
    _write_csv(staging / "run_timing.csv", prepared["gate"].get("timing", []))
    _write_csv(staging / "fdhg_edge_screening.csv", prepared["gate"].get("edge_screening", []))
    _write_csv(staging / "edge_selection_trace.csv", prepared["gate"].get("edge_selection_trace", []))
    _write_csv(
        staging / "fdhg_edge_screening_fold_metrics.csv",
        prepared["gate"].get("edge_screening_fold_metrics", []),
    )
    _write_csv(staging / "pair_screening.csv", prepared["gate"].get("pair_screening", []))
    _write_csv(
        staging / "pair_screening_fold_metrics.csv",
        prepared["gate"].get("pair_screening_fold_metrics", []),
    )
    continuous_audit = [
        *prepared["gate"].get("continuous_discretization_audit", []),
        *prepared["final"].get("continuous_discretization_audit", []),
    ]
    _write_csv(
        staging / "continuous_discretization_audit.csv",
        continuous_audit,
        fieldnames=[*CONTINUOUS_DISCRETIZATION_AUDIT_COLUMNS, "stage"],
    )
    _write_json(staging / "continuous_discretization_fold_boundaries.json", {
        "folds": prepared["gate"].get("continuous_discretization_boundaries", {}),
        "final_refit": prepared["final"].get("continuous_discretization_boundaries", {}),
    })
    _write_csv(staging / "fdhg_fold_feature_audit.csv", prepared["gate"]["fdhg_fold_feature_audit"])
    _write_csv(staging / "fdhg_target_lookup_audit.csv", prepared["gate"]["fdhg_target_lookup_audit"])
    _write_json(staging / "selected_variant.json", {
        "selected_variant": prepared["gate"]["selected_variant"],
        "primary_metric": prepared["metadata"]["primary_metric"],
        "metric_direction": _direction_label(prepared["metadata"]["metric_direction"]),
        "selection_folds": prepared["options"].selection_folds,
        "selection_folds_semantics": "requested maximum inner temporal folds; observed folds are effective_selection_folds",
        "requested_selection_folds": prepared["manifest"].get("requested_selection_folds", prepared["options"].selection_folds),
        "effective_selection_folds": prepared["manifest"].get("effective_selection_folds", 0),
        "effective_fold_ids": prepared["manifest"].get("effective_fold_ids", []),
        "temporal_fold_quality": prepared["manifest"].get("temporal_fold_quality", []),
        "manifest_warnings": prepared["manifest"].get("manifest_warnings", []),
        "min_delta": prepared["options"].min_delta,
        "mean_scores": prepared["gate"]["mean_scores"],
        "selection_reason": prepared["gate"].get("selection_reason", ""),
        "official_validation_was_used_for_selection": False,
        "test_split_accessed": False,
        "edge_screening_enabled": prepared["gate"].get("edge_screening_enabled", prepared["options"].enable_edge_screening),
        "edge_screening_rule": prepared["gate"].get("edge_screening_rule", prepared["options"].edge_screening_rule),
        "edge_screening_min_delta": prepared["gate"].get("edge_screening_min_delta", prepared["options"].edge_screening_min_delta),
        "edge_screening_min_positive_fraction": prepared["gate"].get(
            "edge_screening_min_positive_fraction",
            prepared["options"].edge_screening_min_positive_fraction,
        ),
        "edge_screening_max_relative_fold_degradation": prepared["gate"].get(
            "edge_screening_max_relative_fold_degradation",
            prepared["options"].edge_screening_max_relative_fold_degradation,
        ),
        "edge_screening_min_positive_folds": prepared["gate"].get(
            "edge_screening_min_positive_folds",
            resolve_edge_screening_min_positive_folds(
                prepared["options"],
                effective_selection_folds=prepared["manifest"].get(
                    "effective_selection_folds",
                    prepared["options"].selection_folds,
                ),
            ),
        ),
        "edge_selection_strategy": prepared["gate"].get(
            "edge_selection_strategy",
            prepared["options"].edge_selection_strategy,
        ),
        "max_selected_fdhg_edges": prepared["gate"].get(
            "max_selected_fdhg_edges",
            prepared["options"].max_selected_fdhg_edges,
        ),
        "candidate_discovery_protocol": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_protocol",
            "",
        ),
        "candidate_discovery_fold": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_fold",
            "",
        ),
        "candidate_discovery_fit_horizon": prepared.get("candidate_discovery", {}).get(
            "candidate_discovery_fit_horizon",
            "",
        ),
        "candidate_count_before_budget": prepared.get("candidate_discovery", {}).get(
            "candidate_count_before_budget",
            0,
        ),
        "candidate_count_after_budget": prepared.get("candidate_discovery", {}).get(
            "candidate_count_after_budget",
            len(prepared.get("accepted_fdhg_edges", [])),
        ),
        "candidate_count_after_candidate_budget": prepared.get("candidate_discovery", {}).get(
            "candidate_count_after_budget",
            len(prepared.get("accepted_fdhg_edges", [])),
        ),
        "candidate_rejection_reason_counts": prepared.get("candidate_discovery", {}).get(
            "rejection_reason_counts",
            {},
        ),
        "ordered_candidate_edge_ids": prepared.get("candidate_discovery", {}).get(
            "ordered_candidate_edge_ids",
            [],
        ),
        "candidate_fdhg_edge_count": len(prepared.get("accepted_fdhg_edges", [])),
        "fold_edge_instance_count": len(prepared["gate"].get("edge_screening_fold_metrics", [])),
        "screened_in_fdhg_edge_count": len(prepared["gate"].get("screened_fdhg_edges", [])),
        "screened_out_fdhg_edge_count": max(
            0,
            len(prepared.get("accepted_fdhg_edges", []))
            - len(prepared["gate"].get("screened_fdhg_edges", [])),
        ),
        "screened_in_edge_ids": [
            str(edge_id)
            for edge_id in prepared["gate"].get("strategy_selected_edge_ids", [])
        ],
        "independent_screened_in_edge_ids": prepared["gate"].get("independent_screened_in_edge_ids", []),
        "strategy_selected_edge_ids": prepared["gate"].get("strategy_selected_edge_ids", []),
        "final_combined_block_edge_ids": prepared["gate"].get("final_combined_block_edge_ids", []),
        "selected_screened_edge_count": len(prepared["gate"].get("screened_fdhg_edges", [])),
        "strategy_selected_edge_count": len(prepared["gate"].get("screened_fdhg_edges", [])),
        "edge_selection_step_count": prepared["gate"].get("edge_selection_step_count", 0),
        "edge_selection_stop_reason": prepared["gate"].get("edge_selection_stop_reason", ""),
        "fdhg_screening_fallback": prepared["gate"].get("fdhg_screening_fallback", ""),
        "pairwise_rescue_used": prepared["gate"].get("pairwise_rescue_used", False),
        "selected_initial_pair": prepared["gate"].get("selected_initial_pair", ""),
        "pairwise_rescue_reason": prepared["gate"].get("pairwise_rescue_reason", ""),
        "fdhg_declared_residual_features": prepared["gate"].get("fdhg_declared_residual_features", 0),
        "fdhg_usable_residual_features_by_fold": prepared["gate"].get("fdhg_usable_residual_features_by_fold", {}),
        "fdhg_final_refit_usable_features": prepared["final"].get("fdhg_final_refit_usable_features", 0),
        "fdhg_final_refit_skipped_reason": prepared["final"].get("fdhg_final_refit_skipped_reason", ""),
        "run_counters": prepared["manifest"].get("run_counters", {}),
        **prepared["manifest"].get("run_counters", {}),
    })
    _write_json(staging / "selected_auto_features.json", {"selected_features": prepared["auto_features"]})
    _write_csv(staging / "selected_fdhg_edges.csv", prepared["gate"]["selected_fdhg_edges"])
    _write_json(staging / "selected_fdhg_features.json", {"features": prepared["gate"]["fdhg_feature_provenance"]})
    _write_csv(staging / "dfs_feature_provenance.csv", _feature_provenance_rows(prepared["dfs_features"], "canonical_dfs", "dfs_fallback"))
    feature_rows = []
    feature_rows.extend(_feature_provenance_rows(prepared["dfs_features"], "canonical_dfs", "dfs_fallback"))
    feature_rows.extend(_feature_provenance_rows(prepared["auto_features"], "auto_temporal", "auto_only"))
    feature_rows.extend(_fdhg_feature_rows(prepared["gate"]["fdhg_feature_provenance"]))
    _write_csv(staging / "feature_provenance.csv", feature_rows)
    _write_json(staging / "official_validation_metrics.json", prepared["final"]["official_validation_metrics"])
    preds = prepared["final"]["official_validation_predictions"]
    if isinstance(preds, pd.DataFrame):
        preds.to_parquet(staging / "official_validation_predictions.parquet", index=False)


def _feature_provenance_rows(features: Sequence[Mapping[str, Any]], origin: str, variant: str) -> list[dict[str, Any]]:
    rows = []
    for feature in features:
        base = {
            "feature_name": feature.get("output_column", feature.get("feature_name", "")),
            "feature_origin": origin,
            "variant": variant,
            "source_table": feature.get("child_table", ""),
            "source_column": feature.get("source_column", ""),
            "aggregation": feature.get("aggregation", ""),
            "edge_id": "",
            "lhs_columns": "",
            "rhs_column": "",
            "fold_safe": True,
            "temporal_predicate": feature.get("temporal_predicate", "child.event_time <= target.target_time"),
            "selected": True,
        }
        rows.append(base)
        for aux in _auxiliary_columns(feature):
            aux_row = dict(base)
            aux_row["feature_name"] = aux
            aux_row["aggregation"] = f"{feature.get('aggregation', '')}_missing_indicator"
            rows.append(aux_row)
    return rows


def _fdhg_feature_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        name = row.get("feature_name", "")
        if name in seen:
            continue
        seen.add(name)
        out.append({
            "feature_name": name,
            "feature_origin": "fdhg_residual",
            "variant": "auto_plus_fdhg",
            "source_table": row.get("source_table", ""),
            "source_column": "",
            "aggregation": row.get("aggregation_or_statistic", ""),
            "edge_id": row.get("edge_id", ""),
            "lhs_columns": row.get("lhs_columns", ""),
            "rhs_column": row.get("rhs_column", ""),
            "fold_safe": True,
            "temporal_predicate": "fit_source_time <= fold_train_horizon",
            "selected": True,
        })
    return out


def _edge_identity_row(edge: Mapping[str, Any]) -> dict[str, Any]:
    lhs = edge.get("lhs_columns", "")
    if isinstance(lhs, str):
        lhs_text = lhs
    elif isinstance(lhs, Sequence):
        lhs_text = "|".join(str(value) for value in lhs)
    else:
        lhs_text = str(lhs)
    return {
        "edge_id": edge.get("edge_id", ""),
        "source_table": edge.get("source_table", ""),
        "lhs_columns": lhs_text,
        "rhs_column": edge.get("rhs_column", ""),
    }


def load_historical_candidate_edges(
    *,
    path: Path,
    table_dict: Mapping[str, Any],
    max_edges: int | None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    loaded = _read_candidate_edges_file(source)
    if not loaded:
        explicit_empty_pool = False

        if source.suffix.lower() == ".json":
            try:
                raw_payload = json.loads(
                    source.read_text(encoding="utf-8")
                )
            except Exception:
                raw_payload = None

            explicit_empty_pool = (
                isinstance(raw_payload, list)
                and len(raw_payload) == 0
            )

        if not explicit_empty_pool:
            raise ValueError(
                "historical_candidate_replay_empty_candidate_file"
            )

        provenance = {
            "candidate_discovery_protocol": (
                "historical_candidate_replay"
            ),
            "candidate_source": "historical_replay",
            "candidate_edges_file": str(source),
            "candidate_edges_file_sha256": _file_sha256(source),
            "loaded_candidate_edge_count": 0,
            "candidate_count_before_budget": 0,
            "candidate_count_after_budget": 0,
            "accepted_candidate_edge_count": 0,
            "rejected_candidate_edge_count": 0,
            "candidate_rediscovery_performed": False,
            "candidate_column_audit": [],
            "rejection_reason_counts": {},
            "ordered_candidate_edge_ids": [],
            "accepted_edges": [],
            "inner_validation_rows_used_for_candidate_discovery": 0,
            "official_validation_rows_used_for_candidate_discovery": 0,
            "test_rows_used_for_candidate_discovery": 0,
            "zero_candidate_pool": True,
        }

        return {
            "accepted_edges": [],
            "provenance": provenance,
        }

    seen: set[str] = set()
    edges: list[dict[str, Any]] = []
    for idx, raw in enumerate(loaded, start=1):
        edge = _normalize_candidate_edge_row(raw)
        edge_id = str(edge.get("edge_id", ""))
        if not edge_id:
            raise ValueError(f"historical_candidate_replay_missing_edge_id:row={idx}")
        if edge_id in seen:
            raise ValueError(f"historical_candidate_replay_duplicate_edge_id:{edge_id}")
        seen.add(edge_id)
        _validate_replayed_candidate_edge(edge=edge, table_dict=table_dict, row=idx)
        edges.append(edge)
    loaded_count = len(edges)
    budgeted = edges[: int(max_edges)] if max_edges is not None else list(edges)
    if not budgeted:
        raise ValueError("historical_candidate_replay_empty_after_budget")
    provenance = {
        "candidate_discovery_protocol": "historical_candidate_replay",
        "candidate_source": "historical_replay",
        "candidate_edges_file": str(source),
        "candidate_edges_file_sha256": _file_sha256(source),
        "loaded_candidate_edge_count": loaded_count,
        "candidate_count_before_budget": loaded_count,
        "candidate_count_after_budget": len(budgeted),
        "accepted_candidate_edge_count": len(budgeted),
        "rejected_candidate_edge_count": 0,
        "candidate_rediscovery_performed": False,
        "candidate_column_audit": [],
        "rejection_reason_counts": {},
        "ordered_candidate_edge_ids": [str(edge["edge_id"]) for edge in budgeted],
        "accepted_edges": budgeted,
        "inner_validation_rows_used_for_candidate_discovery": 0,
        "official_validation_rows_used_for_candidate_discovery": 0,
        "test_rows_used_for_candidate_discovery": 0,
    }
    return {"accepted_edges": budgeted, "provenance": provenance}


def _read_candidate_edges_file(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping):
            rows = (
                payload.get("accepted_edges")
                or payload.get("accepted_fdhg_edges")
                or payload.get("candidate_edges")
                or payload.get("edges")
                or []
            )
        else:
            rows = []
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("historical_candidate_replay_json_edges_must_be_objects")
        return [dict(row) for row in rows]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    raise ValueError("historical_candidate_replay_file_must_be_json_or_csv")


def _normalize_candidate_edge_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    required = ("edge_id", "source_table", "lhs_columns", "rhs_column")
    missing = [name for name in required if not str(out.get(name, "")).strip()]
    if missing:
        raise ValueError(f"historical_candidate_replay_missing_required_fields:{','.join(missing)}")
    out["edge_id"] = str(out["edge_id"])
    out["source_table"] = str(out["source_table"])
    out["rhs_column"] = str(out["rhs_column"])
    out["lhs_columns"] = tuple(_parse_replay_sequence(out.get("lhs_columns")))
    if "lhs_original_columns" in out and out["lhs_original_columns"] not in ("", None):
        out["lhs_original_columns"] = tuple(_parse_replay_sequence(out["lhs_original_columns"]))
    if "continuous_columns" in out:
        out["continuous_columns"] = _parse_replay_json_field(out["continuous_columns"], default=[])
    return out


def _parse_replay_sequence(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            parsed = json.loads(text)
            return [str(item) for item in parsed]
        return [part for part in (item.strip() for item in text.split("|")) if part]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return [str(value)]


def _parse_replay_json_field(value: Any, *, default: Any) -> Any:
    if value in ("", None):
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _validate_replayed_candidate_edge(
    *,
    edge: Mapping[str, Any],
    table_dict: Mapping[str, Any],
    row: int,
) -> None:
    table_name = str(edge["source_table"])
    if table_name not in table_dict:
        raise ValueError(f"historical_candidate_replay_unknown_source_table:{table_name}:row={row}")
    columns = set(map(str, _table_df(table_dict[table_name]).columns))
    for column in [*edge.get("lhs_columns", ()), edge.get("rhs_column", "")]:
        col = str(column)
        if col in columns:
            continue
        original = _replayed_endpoint_original(edge=edge, endpoint=col)
        if original and original in columns:
            continue
        raise ValueError(f"historical_candidate_replay_missing_source_column:{table_name}.{col}:row={row}")


def _replayed_endpoint_original(*, edge: Mapping[str, Any], endpoint: str) -> str:
    for spec in edge.get("continuous_columns", []) or []:
        if isinstance(spec, Mapping) and str(spec.get("transformed_column", "")) == str(endpoint):
            return str(spec.get("original_column", ""))
    lhs = list(edge.get("lhs_columns", ()) or ())
    if endpoint in lhs:
        originals = list(edge.get("lhs_original_columns", ()) or ())
        if len(originals) == len(lhs):
            return str(originals[lhs.index(endpoint)])
    if str(edge.get("rhs_column", "")) == str(endpoint):
        return str(edge.get("rhs_original_column", ""))
    return ""


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = _default_csv_fieldnames(path.name)
    if not rows:
        if fieldnames:
            with path.open("w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow(list(fieldnames))
        else:
            path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys([*fieldnames, *sorted({key for row in rows for key in row.keys()})])) if fieldnames else sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _default_csv_fieldnames(name: str) -> list[str]:
    defaults = {
        "fdhg_edge_screening.csv": [
            "dataset", "task", "edge_id", "source_table", "lhs_columns", "rhs_column",
            "requested_selection_folds", "effective_selection_folds", "effective_fold_ids",
            "mean_gain", "positive_fold_count", "total_fold_count", "worst_fold_gain",
            "worst_fold_relative_degradation", "worst_fold_relative_degradation_fold",
            "aggregate_gain", "aggregate_baseline_score", "aggregate_auto_score",
            "aggregate_candidate_score", "gain_std", "screening_rule", "screening_min_delta",
            "screening_min_positive_fraction", "screening_min_positive_folds",
            "screening_max_relative_fold_degradation",
            "screening_status", "rejection_reason", "usable_feature_count_min",
            "usable_feature_count_max", "future_lookup_violation_count", "screening_rank",
            "selected_for_combined_block",
        ],
        "fdhg_edge_screening_fold_metrics.csv": [
            "dataset", "task", "edge_id", "fold", "auto_score",
            "requested_selection_folds", "effective_selection_folds", "effective_fold_ids",
            "auto_plus_single_edge_score", "fold_gain", "metric", "metric_direction",
            "usable_feature_count", "future_lookup_violation_count",
        ],
        "pair_screening.csv": [
            "dataset", "task", "pair_id", "pair_rank", "edge_id_left", "edge_id_right",
            "effective_fold_ids", "screening_rule", "mean_gain", "aggregate_baseline_score",
            "aggregate_candidate_score", "aggregate_gain", "positive_fold_count",
            "passing_fold_count", "total_fold_count", "worst_fold_gain",
            "worst_fold_relative_degradation", "worst_fold_relative_degradation_fold",
            "gain_std", "screening_min_delta", "screening_min_positive_folds",
            "screening_max_relative_fold_degradation", "screening_status",
            "rejection_reason", "selected_initial_pair",
        ],
        "pair_screening_fold_metrics.csv": [
            "dataset", "task", "pair_id", "pair_rank", "edge_id_left", "edge_id_right",
            "fold", "effective_fold_ids", "screening_rule", "auto_score",
            "auto_plus_pair_score", "fold_gain", "metric",
            "metric_direction", "usable_feature_count", "future_lookup_violation_count",
        ],
        "selected_fdhg_edges.csv": [
            "dataset", "task", "edge_id", "source_table", "lhs_columns", "rhs_column",
            "screening_status", "mean_incremental_gain", "positive_fold_count",
            "screening_rank", "selected_for_combined_block", "rejection_reason",
        ],
        "fold_metrics.csv": [
            "dataset", "task", "fold", "variant", "metric", "metric_direction", "score",
            "mean_variant_score", "fold_improvement_over_dfs", "fold_improvement_over_auto",
            "mean_improvement_over_dfs", "mean_improvement_over_auto",
            "globally_selected_variant", "fold_best_variant",
        ],
        "gate_trials.csv": [
            "program_id", "variant", "mean_score", "metric", "metric_direction",
            "selected", "rejection_reason",
        ],
        "run_timing.csv": ["stage", "elapsed_seconds", "fold", "variant", "edge_id", "edge_count", "edge_ids", "pair_id"],
        "edge_selection_trace.csv": [
            "dataset", "task", "strategy", "step", "candidate_edge_id", "candidate_edge_ids",
            "requested_selection_folds", "effective_selection_folds", "effective_fold_ids",
            "edge_screening_rule", "edge_screening_min_positive_folds",
            "edge_screening_min_positive_fraction",
            "selected_edge_ids_after_step", "baseline_mean_score", "trial_mean_score",
            "aggregate_baseline_score", "aggregate_candidate_score", "aggregate_gain",
            "fold_incremental_gains", "mean_incremental_gain", "positive_fold_count",
            "decision", "stop_reason",
        ],
        "fdhg_fold_feature_audit.csv": [
            "fold", "edge_id", "feature_name", "observed_count", "finite_count",
            "non_missing_rate", "unique_count", "variance", "usable", "rejection_reason",
        ],
        "fdhg_target_lookup_audit.csv": [
            "fold", "edge_id", "target_row_count", "matched_target_rows",
            "unmatched_target_rows", "target_lookup_coverage", "maximum_lookup_source_time",
            "maximum_target_time", "future_lookup_violation_count", "mapping_fit_horizon",
            "maximum_mapping_source_time", "target_entity_key", "source_entity_column",
            "rejection_reason",
        ],
        "dfs_feature_provenance.csv": [
            "feature_name", "feature_origin", "variant", "source_table", "source_column",
            "aggregation", "edge_id", "lhs_columns", "rhs_column", "fold_safe",
            "temporal_predicate", "selected",
        ],
        "feature_provenance.csv": [
            "feature_name", "feature_origin", "variant", "source_table", "source_column",
            "aggregation", "edge_id", "lhs_columns", "rhs_column", "fold_safe",
            "temporal_predicate", "selected",
        ],
    }
    return list(defaults.get(name, []))


def _report(status: str, prepared: Mapping[str, Any], *, dry_run: bool, blockers: Sequence[str] = ()) -> AutoFdhgReport:
    gate = prepared.get("gate", {})
    final = prepared.get("final", {})
    selected_edge_rows = gate.get("selected_fdhg_edges", [])
    accepted_edge_count = (
        len([row for row in selected_edge_rows if row.get("selected_for_combined_block")])
        if selected_edge_rows
        else len(prepared.get("accepted_fdhg_edges", []))
    )
    return AutoFdhgReport(
        dataset=prepared["dataset_name"],
        task=prepared["task_name"],
        status=status,
        output_dir=prepared["output_dir"],
        blockers=tuple(blockers),
        dry_run=dry_run,
        selected_variant=gate.get("selected_variant"),
        metric=prepared["metadata"].get("primary_metric"),
        metric_direction=prepared["metadata"].get("metric_direction"),
        mean_scores=gate.get("mean_scores"),
        official_validation_score=final.get("official_validation_score"),
        dfs_features=len(prepared.get("dfs_features", [])),
        dfs_declarations=int(prepared.get("dfs_declaration_count", len(prepared.get("dfs_features", [])))),
        dfs_model_columns=int(prepared.get("dfs_model_column_count", len(_declared_model_columns(prepared.get("dfs_features", []))))),
        auto_features=len(prepared.get("auto_features", [])),
        fdhg_features=len(_fdhg_feature_rows(gate.get("fdhg_feature_provenance", []))),
        fdhg_declared_residual_features=int(gate.get("fdhg_declared_residual_features", 0)),
        fdhg_usable_residual_features_by_fold=gate.get("fdhg_usable_residual_features_by_fold", {}),
        fdhg_final_refit_usable_features=int(final.get("fdhg_final_refit_usable_features", 0)),
        accepted_edges=accepted_edge_count,
        candidate_edges=len(prepared.get("accepted_fdhg_edges", [])),
        screened_in_edges=len(gate.get("screened_fdhg_edges", [])),
        screened_out_edges=max(0, len(prepared.get("accepted_fdhg_edges", [])) - len(gate.get("screened_fdhg_edges", []))),
        expected_scans=prepared.get("workload", {}).get("expected_child_relation_scans", 0),
        expected_materializations=prepared.get("workload", {}).get("expected_matrix_materializations", 0),
        test_split_accessed=False,
    )


def _direction_label(direction: str) -> str:
    return "lower_is_better" if direction == "lower" else "higher_is_better"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
