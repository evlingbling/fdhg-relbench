#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
from relbench.datasets import get_dataset

from fdhg.onboarding.auto_fdhg import (
    AutoFdhgOptions,
    align_feature_blocks,
    audit_residual_columns,
    feature_columns,
    fit_afd_edges,
    materialize_ambiguity_features,
    materialize_declared_feature_frame_pair,
)


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def load_candidate_edges(path: Path):
    obj = load_json(path)

    if isinstance(obj, list):
        return obj

    if not isinstance(obj, dict):
        raise TypeError(
            f"Unexpected candidate JSON type: {type(obj)}"
        )

    for key in (
        "candidate_edges",
        "candidates",
        "edges",
        "fixed_candidate_edges",
    ):
        value = obj.get(key)
        if isinstance(value, list):
            return value

    raise KeyError(
        "Could not find candidate-edge list in "
        f"{path}. Top-level keys={list(obj)}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/final-gate-51task-v2"
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--task",
        required=True,
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "independent"],
        default="greedy",
    )
    parser.add_argument(
        "--export-root",
        default=(
            "outputs/model-robustness-feature-matrices"
        ),
    )

    args = parser.parse_args()

    root = Path(args.output_root)
    task_key = f"{args.dataset}_{args.task}"

    task_root = root / task_key

    canonical_root = (
        root
        / "_canonical_exports"
        / args.dataset
        / args.task
    )

    train_path = (
        canonical_root / "target_train.parquet"
    )

    val_path = (
        canonical_root / "target_validation.parquet"
    )

    tables_root = canonical_root / "tables"

    metadata_path = (
        task_root
        / "pipeline"
        / "resolved_task_metadata.json"
    )

    auto_dir = (
        task_root
        / "auto"
        / task_key
    )

    strategy_dir = (
        task_root
        / "strategies"
        / args.strategy
        / task_key
    )

    candidate_path = (
        task_root
        / "candidates"
        / "fixed_candidate_edges.json"
    )

    selected_edges_path = (
        strategy_dir
        / "selected_fdhg_edges.csv"
    )

    required_paths = [
        train_path,
        val_path,
        metadata_path,
        auto_dir / "selected_features.json",
        candidate_path,
        selected_edges_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    print("===== INPUTS =====")
    for path in required_paths:
        print(path)

    train_targets = pd.read_parquet(
        train_path
    ).reset_index(drop=True)

    val_targets = pd.read_parquet(
        val_path
    ).reset_index(drop=True)

    metadata = load_json(
        metadata_path
    )

    auto_spec = load_json(
        auto_dir / "selected_features.json"
    )

    auto_features = auto_spec[
        "selected_features"
    ]

    # Use the native RelBench Table objects, not bare pandas
    # DataFrames. The production materializers rely on both
    # table.df and schema metadata such as PK/FK/time columns.
    dataset_obj = get_dataset(
        args.dataset,
        download=False,
    )
    database = dataset_obj.get_db()
    table_dict = database.table_dict

    expected_tables = {
        path.stem
        for path in tables_root.glob("*.parquet")
    }
    loaded_tables = set(table_dict)

    missing_tables = sorted(
        expected_tables - loaded_tables
    )
    if missing_tables:
        raise ValueError(
            "RelBench DB missing canonical-export tables: "
            + ",".join(missing_tables)
        )

    print()
    print("===== DATA =====")
    print(
        "train targets:",
        train_targets.shape,
    )
    print(
        "validation targets:",
        val_targets.shape,
    )
    print(
        "tables:",
        sorted(table_dict),
    )
    print(
        "auto features:",
        len(auto_features),
    )

    entity_key = str(
        metadata["entity_key"]
    )

    target_time_col = str(
        metadata["target_time_col"]
    )

    label_col = str(
        metadata["label_col"]
    )

    join_keys = list(
        dict.fromkeys(
            [
                entity_key,
                target_time_col,
            ]
        )
    )

    #
    # AUTO
    #
    auto_train, auto_val = (
        materialize_declared_feature_frame_pair(
            train_targets,
            val_targets,
            table_dict=table_dict,
            features=auto_features,
            entity_key=entity_key,
            target_time_col=target_time_col,
        )
    )

    auto_cols = feature_columns(
        auto_train,
        join_keys,
        metadata,
    )

    print()
    print("===== AUTO =====")
    print(
        "train:",
        auto_train.shape,
    )
    print(
        "val:",
        auto_val.shape,
    )
    print(
        "model features:",
        len(auto_cols),
    )

    #
    # SELECTED FDHG EDGE IDs
    #
    selected_df = pd.read_csv(
        selected_edges_path
    )

    if (
        "selected_for_combined_block"
        not in selected_df.columns
    ):
        raise KeyError(
            "selected_for_combined_block missing "
            f"from {selected_edges_path}"
        )

    selected_mask = (
        selected_df[
            "selected_for_combined_block"
        ]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    selected_edge_ids = (
        selected_df.loc[
            selected_mask,
            "edge_id",
        ]
        .astype(str)
        .tolist()
    )

    if not selected_edge_ids:
        raise ValueError(
            f"No selected FDHG edges for "
            f"{args.strategy}"
        )

    print()
    print(
        "selected edge IDs:",
        selected_edge_ids,
    )

    candidates = load_candidate_edges(
        candidate_path
    )

    by_id = {
        str(edge.get("edge_id")): edge
        for edge in candidates
    }

    missing_specs = [
        edge_id
        for edge_id in selected_edge_ids
        if edge_id not in by_id
    ]

    if missing_specs:
        raise KeyError(
            "Selected edges missing from fixed "
            f"candidate pool: {missing_specs}"
        )

    selected_edges = [
        by_id[edge_id]
        for edge_id in selected_edge_ids
    ]

    #
    # FDHG full-train refit.
    #
    fit_horizon = (
        train_targets[
            target_time_col
        ].max()
    )

    options = AutoFdhgOptions()

    fitted = fit_afd_edges(
        inner_train_rows=train_targets,
        source_tables=table_dict,
        schema=None,
        task_metadata=metadata,
        candidate_edges=selected_edges,
        max_edges=len(selected_edges),
        fold="full_train",
        fit_horizon=fit_horizon,
        continuous_fdhg_mode=(
            options.continuous_fdhg_mode
        ),
        continuous_fdhg_bins=(
            options.continuous_fdhg_bins
        ),
        continuous_fdhg_min_effective_bins=(
            options
            .continuous_fdhg_min_effective_bins
        ),
        dataset=args.dataset,
        task=args.task,
        fit_split="official_train",
    )

    source_entity_columns_by_edge = {
        str(
            edge.get(
                "edge_id",
                "",
            )
        ): str(
            edge.get(
                "source_entity_column",
                "",
            )
        )
        for edge in selected_edges
        if edge.get(
            "source_entity_column"
        )
    }

    fdhg_train, _, _ = (
        materialize_ambiguity_features(
            fitted_edges=fitted,
            target_rows=train_targets,
            source_tables=table_dict,
            task_metadata=metadata,
            source_entity_columns_by_edge=(
                source_entity_columns_by_edge
            ),
        )
    )

    fdhg_val, _, _ = (
        materialize_ambiguity_features(
            fitted_edges=fitted,
            target_rows=val_targets,
            source_tables=table_dict,
            task_metadata=metadata,
            source_entity_columns_by_edge=(
                source_entity_columns_by_edge
            ),
        )
    )

    raw_fdhg_cols = feature_columns(
        fdhg_train,
        join_keys,
        metadata,
    )

    audit, usable_fdhg_cols = (
        audit_residual_columns(
            frame=fdhg_train,
            feature_cols=raw_fdhg_cols,
            fold="full_train",
            provenance=[],
        )
    )

    combined_train = align_feature_blocks(
        target_rows=train_targets,
        blocks=[
            ("auto", auto_train),
            ("fdhg", fdhg_train),
        ],
        join_keys=join_keys,
        metadata=metadata,
    )

    combined_val = align_feature_blocks(
        target_rows=val_targets,
        blocks=[
            ("auto", auto_val),
            ("fdhg", fdhg_val),
        ],
        join_keys=join_keys,
        metadata=metadata,
    )

    combined_cols = [
        *auto_cols,
        *usable_fdhg_cols,
    ]

    print()
    print(
        f"===== {args.strategy.upper()} ====="
    )
    print(
        "fitted edges:",
        len(fitted),
    )
    print(
        "raw FDHG columns:",
        len(raw_fdhg_cols),
    )
    print(
        "usable FDHG columns:",
        len(usable_fdhg_cols),
    )
    print(
        "combined features:",
        len(combined_cols),
    )

    #
    # Keep target metadata + label + model features.
    #
    base_cols = [
        column
        for column in (
            *join_keys,
            label_col,
        )
        if column in train_targets.columns
    ]

    auto_export_cols = list(
        dict.fromkeys(
            [
                *base_cols,
                *auto_cols,
            ]
        )
    )

    combined_export_cols = list(
        dict.fromkeys(
            [
                *base_cols,
                *combined_cols,
            ]
        )
    )

    export_root = (
        Path(args.export_root)
        / task_key
    )

    auto_out = (
        export_root / "auto"
    )

    strategy_out = (
        export_root
        / args.strategy
    )

    auto_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategy_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    auto_train[
        auto_export_cols
    ].to_parquet(
        auto_out / "train.parquet",
        index=False,
    )

    auto_val[
        auto_export_cols
    ].to_parquet(
        auto_out / "val.parquet",
        index=False,
    )

    combined_train[
        combined_export_cols
    ].to_parquet(
        strategy_out / "train.parquet",
        index=False,
    )

    combined_val[
        combined_export_cols
    ].to_parquet(
        strategy_out / "val.parquet",
        index=False,
    )

    audit_path = (
        strategy_out
        / "fdhg_final_feature_audit.csv"
    )

    pd.DataFrame(audit).to_csv(
        audit_path,
        index=False,
    )

    manifest = {
        "dataset": args.dataset,
        "task": args.task,
        "strategy": args.strategy,
        "entity_key": entity_key,
        "target_time_col": target_time_col,
        "label_col": label_col,
        "auto_feature_count": len(
            auto_cols
        ),
        "selected_edge_ids": (
            selected_edge_ids
        ),
        "fitted_edge_count": len(
            fitted
        ),
        "raw_fdhg_feature_count": len(
            raw_fdhg_cols
        ),
        "usable_fdhg_feature_count": len(
            usable_fdhg_cols
        ),
        "combined_feature_count": len(
            combined_cols
        ),
        "train_rows": len(
            train_targets
        ),
        "validation_rows": len(
            val_targets
        ),
    }

    with (
        export_root / "manifest.json"
    ).open("w") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            default=str,
        )

    print()
    print("===== EXPORTED =====")

    for path in (
        auto_out / "train.parquet",
        auto_out / "val.parquet",
        strategy_out / "train.parquet",
        strategy_out / "val.parquet",
        export_root / "manifest.json",
    ):
        print(path)

    print()
    print(json.dumps(
        manifest,
        indent=2,
    ))


if __name__ == "__main__":
    main()
