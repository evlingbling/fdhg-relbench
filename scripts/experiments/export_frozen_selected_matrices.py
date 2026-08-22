#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from relbench.datasets import get_dataset

from fdhg.onboarding.auto_fdhg import (
    _declared_model_columns,
    align_feature_blocks,
    audit_residual_columns,
    feature_columns,
    fit_afd_edges,
    materialize_ambiguity_features,
    materialize_declared_feature_frame_pair,
)


DEFAULT_ROOT = Path(
    "outputs/final-gate-51task-v2"
)

DEFAULT_EXPORT_ROOT = Path(
    "outputs/predictor-generalization/frozen-matrices"
)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


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
        f"Candidate edge list not found in {path}"
    )


def boolish(value) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def selected_edge_ids(
    *,
    selected_variant: Mapping[str, Any],
    selected_edges_csv: Path,
) -> list[str]:

    # Prefer the exact IDs recorded by the final selected artifact.
    ids = selected_variant.get(
        "final_combined_block_edge_ids"
    )

    if isinstance(ids, list):
        return [
            str(x)
            for x in ids
            if str(x)
        ]

    # Defensive fallback to the selected-edge CSV.
    df = pd.read_csv(selected_edges_csv)

    if "selected_for_combined_block" not in df.columns:
        raise KeyError(
            "selected_for_combined_block missing from "
            f"{selected_edges_csv}"
        )

    mask = df[
        "selected_for_combined_block"
    ].map(boolish)

    return (
        df.loc[mask, "edge_id"]
        .astype(str)
        .tolist()
    )


def task_paths(
    root: Path,
    dataset: str,
    task: str,
):
    slug = f"{dataset}_{task}"
    task_root = root / slug

    pipeline_manifest = (
        task_root
        / "pipeline"
        / "pipeline_manifest.json"
    )

    metadata_path = (
        task_root
        / "pipeline"
        / "resolved_task_metadata.json"
    )

    selected_dir = (
        task_root
        / "strategies"
        / "selected"
        / slug
    )

    joint_path = (
        task_root
        / "strategies"
        / "joint"
        / slug
        / "joint_selection.json"
    )

    candidate_path = (
        task_root
        / "candidates"
        / "fixed_candidate_edges.json"
    )

    return {
        "slug": slug,
        "task_root": task_root,
        "pipeline_manifest":
            pipeline_manifest,
        "metadata":
            metadata_path,
        "selected_dir":
            selected_dir,
        "joint":
            joint_path,
        "candidate":
            candidate_path,
    }


def verify_safety(
    joint: Mapping[str, Any],
    selected: Mapping[str, Any],
):
    if joint.get("test_split_accessed") is not False:
        raise RuntimeError(
            "REFUSING: joint test_split_accessed != False"
        )

    if (
        joint.get(
            "official_validation_was_used_for_selection"
        )
        is not False
    ):
        raise RuntimeError(
            "REFUSING: official validation used for selection"
        )

    if joint.get(
        "same_candidate_pool_verified"
    ) is not True:
        raise RuntimeError(
            "REFUSING: same_candidate_pool_verified != True"
        )

    if selected.get(
        "test_split_accessed"
    ) is not False:
        raise RuntimeError(
            "REFUSING: selected artifact test access != False"
        )


def load_tables(dataset: str):
    dataset_obj = get_dataset(
        dataset,
        download=False,
    )

    return dataset_obj.get_db().table_dict


def export_dfs(
    *,
    canonical_onboarding: Path,
    output_root: Path,
):
    src_train = (
        canonical_onboarding
        / "target_with_dfs_agg_train.parquet"
    )

    src_val = (
        canonical_onboarding
        / "target_with_dfs_agg_val.parquet"
    )

    if not src_train.exists():
        raise FileNotFoundError(src_train)

    if not src_val.exists():
        raise FileNotFoundError(src_val)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src_train,
        output_root / "train.parquet",
    )

    shutil.copy2(
        src_val,
        output_root / "val.parquet",
    )

    return (
        pd.read_parquet(
            output_root / "train.parquet"
        ),
        pd.read_parquet(
            output_root / "val.parquet"
        ),
        [],
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    ap.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
    )

    ap.add_argument(
        "--dataset",
        required=True,
    )

    ap.add_argument(
        "--task",
        required=True,
    )

    ap.add_argument(
        "--write",
        action="store_true",
    )

    args = ap.parse_args()

    paths = task_paths(
        args.output_root,
        args.dataset,
        args.task,
    )

    required = [
        paths["metadata"],
        paths["joint"],
        paths["candidate"],
        paths["selected_dir"]
            / "selected_variant.json",
        paths["selected_dir"]
            / "selected_auto_features.json",
        paths["selected_dir"]
            / "selected_fdhg_edges.csv",
        paths["selected_dir"]
            / "manifest.json",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    if paths["pipeline_manifest"].exists():
        pipeline = load_json(
            paths["pipeline_manifest"]
        )

        canonical_export = Path(
            pipeline["canonical_export_dir"]
        )

        canonical_onboarding = Path(
            pipeline["canonical_onboarding_dir"]
        )

    else:
        # Some completed parallel-worker artifacts predate
        # pipeline_manifest persistence. Their canonical inputs
        # remain under the same frozen worker output root.
        canonical_export = (
            args.output_root
            / "_canonical_exports"
            / args.dataset
            / args.task
        )

        canonical_onboarding = (
            args.output_root
            / "_canonical_onboarding"
            / f"relbench-v1-{args.dataset}_{args.task}"
        )

        print(
            "[INFO] pipeline_manifest.json missing; "
            "using deterministic canonical paths under "
            f"output_root={args.output_root}"
        )

    metadata = load_json(
        paths["metadata"]
    )

    joint = load_json(
        paths["joint"]
    )

    selected = load_json(
        paths["selected_dir"]
        / "selected_variant.json"
    )

    selected_manifest = load_json(
        paths["selected_dir"]
        / "manifest.json"
    )

    verify_safety(
        joint,
        selected,
    )

    variant = str(
        joint["selected_variant"]
    )

    train_path = (
        canonical_export
        / "target_train.parquet"
    )

    val_path = (
        canonical_export
        / "target_validation.parquet"
    )

    for p in (
        train_path,
        val_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    auto_obj = load_json(
        paths["selected_dir"]
        / "selected_auto_features.json"
    )

    auto_features = list(
        auto_obj.get(
            "selected_features",
            [],
        )
    )

    final_edge_ids = selected_edge_ids(
        selected_variant=selected,
        selected_edges_csv=(
            paths["selected_dir"]
            / "selected_fdhg_edges.csv"
        ),
    )

    continuous_mode = str(
        selected_manifest.get(
            "continuous_fdhg_mode",
            "exclude",
        )
    )

    continuous_bins = int(
        selected_manifest.get(
            "continuous_fdhg_bins",
            8,
        )
    )

    continuous_min_bins = int(
        selected_manifest.get(
            "continuous_fdhg_min_effective_bins",
            2,
        )
    )

    print()
    print(
        "FROZEN SELECTED MATRIX EXPORT"
    )
    print("=" * 100)

    print(
        f"dataset/task="
        f"{args.dataset}/{args.task}"
    )

    print(
        f"selected_variant={variant}"
    )

    print(
        f"selected_source_strategy="
        f"{selected.get('selected_source_strategy', '')}"
    )

    print(
        f"final_fdhg_edge_ids="
        f"{final_edge_ids}"
    )

    print(
        f"auto_features="
        f"{len(auto_features)}"
    )

    print(
        "continuous_fdhg="
        f"{continuous_mode}/"
        f"{continuous_bins}/"
        f"{continuous_min_bins}"
    )

    print(
        f"canonical_export="
        f"{canonical_export}"
    )

    print(
        f"canonical_onboarding="
        f"{canonical_onboarding}"
    )

    allowed = {
        "dfs_fallback",
        "auto_only",
        "auto_plus_fdhg_independent",
        "auto_plus_fdhg_greedy",
    }

    if variant not in allowed:
        raise RuntimeError(
            f"Unknown selected variant: {variant}"
        )

    if variant in {
        "dfs_fallback",
        "auto_only",
    } and final_edge_ids:
        print(
            "[INFO] ignoring stored FDHG IDs "
            f"because selected_variant={variant}"
        )

    if (
        variant.startswith(
            "auto_plus_fdhg_"
        )
        and not final_edge_ids
    ):
        raise RuntimeError(
            "FDHG-selected variant has zero "
            "final FDHG edge IDs"
        )

    if not args.write:
        print()
        print(
            "WRITE_ENABLED=0 "
            "(validated only; no materialization)"
        )
        return

    train_targets = pd.read_parquet(
        train_path
    ).reset_index(drop=True)

    val_targets = pd.read_parquet(
        val_path
    ).reset_index(drop=True)

    label_col = str(
        metadata["label_col"]
    )

    entity_key = str(
        metadata["entity_key"]
    )

    target_time_col = str(
        metadata["target_time_col"]
    )

    join_keys = list(
        dict.fromkeys(
            [
                entity_key,
                target_time_col,
            ]
        )
    )

    out = (
        args.export_root
        / paths["slug"]
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if variant == "dfs_fallback":
        train_out, val_out, model_cols = export_dfs(
            canonical_onboarding=
                canonical_onboarding,
            output_root=out,
        )

        # Use ONLY the canonical DFS model columns recorded
        # by the original final-gate artifact. Do not infer
        # predictors from every non-label column in the parquet.
        dfs_source = selected_manifest.get(
            "canonical_dfs_source",
            {}
        )

        dfs_feature_names = list(
            dfs_source.get(
                "canonical_feature_names",
                [],
            )
        )

        dfs_auxiliary_columns = list(
            dfs_source.get(
                "canonical_auxiliary_columns",
                [],
            )
        )

        model_cols = list(
            dict.fromkeys(
                [
                    *dfs_feature_names,
                    *dfs_auxiliary_columns,
                ]
            )
        )

        missing_dfs_cols = [
            c
            for c in model_cols
            if (
                c not in train_out.columns
                or c not in val_out.columns
            )
        ]

        if missing_dfs_cols:
            raise KeyError(
                "Canonical DFS model columns missing "
                f"from frozen parquet: {missing_dfs_cols}"
            )

        # Rewrite the copied canonical parquet to contain
        # only join keys + label + the recorded DFS columns.
        base_cols = [
            c
            for c in (
                *join_keys,
                label_col,
            )
            if c in train_out.columns
        ]

        dfs_export_cols = list(
            dict.fromkeys(
                [
                    *base_cols,
                    *model_cols,
                ]
            )
        )

        train_out = train_out[
            dfs_export_cols
        ].copy()

        val_out = val_out[
            dfs_export_cols
        ].copy()

        train_out.to_parquet(
            out / "train.parquet",
            index=False,
        )

        val_out.to_parquet(
            out / "val.parquet",
            index=False,
        )

        audit = []

    else:
        table_dict = load_tables(
            args.dataset
        )

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

        # Match production final-refit semantics exactly:
        # only columns declared by the frozen Auto program are
        # predictor inputs. Lookup/entity helper columns retained
        # in the materialized frame are NOT model features.
        auto_cols = list(
            _declared_model_columns(
                auto_features
            )
        )

        missing_auto_cols = [
            c
            for c in auto_cols
            if (
                c not in auto_train.columns
                or c not in auto_val.columns
            )
        ]

        if missing_auto_cols:
            raise KeyError(
                "Declared Auto model columns missing "
                f"after materialization: {missing_auto_cols}"
            )

        if variant == "auto_only":
            train_out = auto_train
            val_out = auto_val
            model_cols = auto_cols
            audit = []

        else:
            candidates = load_candidate_edges(
                paths["candidate"]
            )

            by_id = {
                str(
                    edge.get(
                        "edge_id"
                    )
                ): edge
                for edge in candidates
            }

            missing = [
                eid
                for eid in final_edge_ids
                if eid not in by_id
            ]

            if missing:
                raise KeyError(
                    "Selected edge IDs missing "
                    f"from frozen candidate pool: {missing}"
                )

            selected_edges = [
                by_id[eid]
                for eid in final_edge_ids
            ]

            fit_horizon = (
                train_targets[
                    target_time_col
                ].max()
            )

            fitted = fit_afd_edges(
                inner_train_rows=
                    train_targets,
                source_tables=
                    table_dict,
                schema=None,
                task_metadata=
                    metadata,
                candidate_edges=
                    selected_edges,
                max_edges=
                    len(selected_edges),
                fold="full_train",
                fit_horizon=
                    fit_horizon,
                continuous_fdhg_mode=
                    continuous_mode,
                continuous_fdhg_bins=
                    continuous_bins,
                continuous_fdhg_min_effective_bins=
                    continuous_min_bins,
                dataset=
                    args.dataset,
                task=
                    args.task,
                fit_split=
                    "official_train",
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

            target_lookup_columns_by_edge = {
                str(
                    edge.get(
                        "edge_id",
                        "",
                    )
                ): str(
                    edge.get(
                        "target_lookup_column",
                        "",
                    )
                )
                for edge in selected_edges
                if edge.get(
                    "target_lookup_column"
                )
            }

            strict_before_by_edge = {
                str(
                    edge.get(
                        "edge_id",
                        "",
                    )
                ): boolish(
                    edge.get(
                        "strict_before",
                        False,
                    )
                )
                for edge in selected_edges
                if "strict_before" in edge
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
                    target_lookup_columns_by_edge=(
                        target_lookup_columns_by_edge
                    ),
                    strict_before_by_edge=(
                        strict_before_by_edge
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
                    target_lookup_columns_by_edge=(
                        target_lookup_columns_by_edge
                    ),
                    strict_before_by_edge=(
                        strict_before_by_edge
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
                    feature_cols=
                        raw_fdhg_cols,
                    fold="full_train",
                    provenance=[],
                )
            )

            train_out = align_feature_blocks(
                target_rows=train_targets,
                blocks=[
                    (
                        "auto",
                        auto_train,
                    ),
                    (
                        "fdhg",
                        fdhg_train,
                    ),
                ],
                join_keys=join_keys,
                metadata=metadata,
            )

            val_out = align_feature_blocks(
                target_rows=val_targets,
                blocks=[
                    (
                        "auto",
                        auto_val,
                    ),
                    (
                        "fdhg",
                        fdhg_val,
                    ),
                ],
                join_keys=join_keys,
                metadata=metadata,
            )

            model_cols = [
                *auto_cols,
                *usable_fdhg_cols,
            ]

        base_cols = [
            c
            for c in (
                *join_keys,
                label_col,
            )
            if c in train_targets.columns
        ]

        export_cols = list(
            dict.fromkeys(
                [
                    *base_cols,
                    *model_cols,
                ]
            )
        )

        train_out[
            export_cols
        ].to_parquet(
            out / "train.parquet",
            index=False,
        )

        val_out[
            export_cols
        ].to_parquet(
            out / "val.parquet",
            index=False,
        )

    pd.DataFrame(
        audit
    ).to_csv(
        out
        / "fdhg_final_feature_audit.csv",
        index=False,
    )

    manifest = {
        "dataset":
            args.dataset,
        "task":
            args.task,
        "selected_variant":
            variant,
        "selected_source_strategy":
            selected.get(
                "selected_source_strategy",
                "",
            ),
        "label_col":
            label_col,
        "problem_type":
            metadata["problem_type"],
        "primary_metric":
            metadata["primary_metric"],
        "entity_key":
            entity_key,
        "target_time_col":
            target_time_col,
        "join_keys":
            join_keys,
        "model_feature_columns":
            model_cols,
        "model_feature_count":
            len(model_cols),
        "auto_feature_declaration_count":
            len(auto_features),
        "final_fdhg_edge_ids":
            (
                final_edge_ids
                if variant.startswith(
                    "auto_plus_fdhg_"
                )
                else []
            ),
        "continuous_fdhg_mode":
            continuous_mode,
        "continuous_fdhg_bins":
            continuous_bins,
        "continuous_fdhg_min_effective_bins":
            continuous_min_bins,
        "train_rows":
            len(train_out),
        "validation_rows":
            len(val_out),
        "test_split_accessed":
            False,
        "official_validation_used_for_selection":
            False,
        "same_candidate_pool_verified":
            True,
    }

    with (
        out / "manifest.json"
    ).open("w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            default=str,
        )

    print()
    print("EXPORTED")
    print(out / "train.parquet")
    print(out / "val.parquet")
    print(out / "manifest.json")
    print(
        f"model_feature_count="
        f"{len(model_cols)}"
    )


if __name__ == "__main__":
    main()
