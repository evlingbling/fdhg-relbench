from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def dataframe_fingerprint(
    df: pd.DataFrame,
    *,
    columns: list[str],
) -> str:
    """Return a deterministic SHA256 fingerprint for selected columns."""
    available = [column for column in columns if column in df.columns]
    if not available:
        return hashlib.sha256(b"").hexdigest()

    hashed = pd.util.hash_pandas_object(
        df[available],
        index=True,
    ).to_numpy()

    return hashlib.sha256(hashed.tobytes()).hexdigest()


def scalar_for_json(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    context: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{context} is missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def build_train_only_fit_table(
    *,
    entity_df: pd.DataFrame,
    target_train: pd.DataFrame,
    entity_key: str,
    target_entity_key: str,
    entity_time_col: str | None,
    target_time_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(
        entity_df,
        [entity_key],
        context="entity table",
    )
    require_columns(
        target_train,
        [target_entity_key],
        context="target train split",
    )

    target_keys = (
        target_train[target_entity_key]
        .dropna()
        .drop_duplicates()
    )

    source_rows = len(entity_df)
    source_entities = int(entity_df[entity_key].nunique(dropna=True))
    train_entities = int(target_keys.nunique(dropna=True))

    fit_df = entity_df[
        entity_df[entity_key].isin(set(target_keys.tolist()))
    ].copy()

    temporal_cutoff_applied = False
    n_rows_before_temporal_filter = len(fit_df)

    if bool(entity_time_col) != bool(target_time_col):
        raise ValueError(
            "entity_time_col and target_time_col must either both be "
            "provided or both be omitted."
        )

    if entity_time_col and target_time_col:
        require_columns(
            entity_df,
            [entity_time_col],
            context="entity table",
        )
        require_columns(
            target_train,
            [target_time_col],
            context="target train split",
        )

        cutoffs = target_train[
            [target_entity_key, target_time_col]
        ].copy()

        cutoffs[target_time_col] = pd.to_datetime(
            cutoffs[target_time_col],
            errors="coerce",
            utc=True,
        )
        cutoffs = cutoffs.dropna(
            subset=[target_entity_key, target_time_col]
        )

        # For each training entity, only rows observable by at least one
        # training prediction are eligible. The maximum train cutoff retains
        # all rows usable by any train example while excluding post-train
        # information.
        cutoffs = (
            cutoffs.groupby(
                target_entity_key,
                as_index=False,
                dropna=False,
            )[target_time_col]
            .max()
            .rename(
                columns={
                    target_entity_key: entity_key,
                    target_time_col: "__fdhg_train_cutoff",
                }
            )
        )

        fit_df[entity_time_col] = pd.to_datetime(
            fit_df[entity_time_col],
            errors="coerce",
            utc=True,
        )

        fit_df = fit_df.merge(
            cutoffs,
            on=entity_key,
            how="inner",
            validate="many_to_one",
        )

        fit_df = fit_df[
            fit_df[entity_time_col].notna()
            & (
                fit_df[entity_time_col]
                <= fit_df["__fdhg_train_cutoff"]
            )
        ].drop(columns=["__fdhg_train_cutoff"])

        temporal_cutoff_applied = True

    # Preserve source ordering so sampling remains reproducible.
    fit_df = fit_df.sort_index().reset_index(drop=True)

    if fit_df.empty:
        raise ValueError(
            "Train-only AFD fit table is empty. "
            "Check entity-key aliases and temporal columns."
        )

    manifest: dict[str, Any] = {
        "fit_split": "train",
        "fit_scope": (
            "train_entities_with_temporal_cutoff"
            if temporal_cutoff_applied
            else "train_entities"
        ),
        "entity_key": entity_key,
        "target_entity_key": target_entity_key,
        "entity_time_col": entity_time_col,
        "target_time_col": target_time_col,
        "temporal_cutoff_applied": temporal_cutoff_applied,
        "n_source_rows": int(source_rows),
        "n_source_entities": int(source_entities),
        "n_train_target_rows": int(len(target_train)),
        "n_train_entities": int(train_entities),
        "n_rows_after_entity_filter": int(
            n_rows_before_temporal_filter
        ),
        "n_fit_rows": int(len(fit_df)),
        "n_fit_entities": int(
            fit_df[entity_key].nunique(dropna=True)
        ),
        "max_target_train_time": (
            scalar_for_json(
                pd.to_datetime(
                    target_train[target_time_col],
                    errors="coerce",
                    utc=True,
                ).max()
            )
            if target_time_col
            else None
        ),
        "max_fit_entity_time": (
            scalar_for_json(fit_df[entity_time_col].max())
            if entity_time_col
            else None
        ),
        "fit_fingerprint_sha256": dataframe_fingerprint(
            fit_df,
            columns=[
                entity_key,
                *(
                    [entity_time_col]
                    if entity_time_col
                    else []
                ),
            ],
        ),
    }

    return fit_df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the train-visible entity table used for AFD discovery "
            "and ambiguity-map fitting."
        )
    )
    parser.add_argument("--entity-table", required=True)
    parser.add_argument("--target-train", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--entity-key", required=True)
    parser.add_argument(
        "--target-entity-key",
        required=True,
    )
    parser.add_argument("--entity-time-col", default=None)
    parser.add_argument("--target-time-col", default=None)
    args = parser.parse_args()

    entity_path = Path(args.entity_table)
    target_path = Path(args.target_train)
    output_path = Path(args.out)
    manifest_path = Path(args.manifest)

    if not entity_path.exists():
        raise FileNotFoundError(
            f"Entity table does not exist: {entity_path}"
        )
    if not target_path.exists():
        raise FileNotFoundError(
            f"Target train split does not exist: {target_path}"
        )

    entity_df = pd.read_parquet(entity_path)
    target_train = pd.read_parquet(target_path)

    fit_df, manifest = build_train_only_fit_table(
        entity_df=entity_df,
        target_train=target_train,
        entity_key=args.entity_key,
        target_entity_key=args.target_entity_key,
        entity_time_col=args.entity_time_col,
        target_time_col=args.target_time_col,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_output = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    fit_df.to_parquet(temporary_output, index=False)
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    temporary_output.replace(output_path)
    temporary_manifest.replace(manifest_path)

    success_path = output_path.parent / "_SUCCESS"
    success_path.write_text(
        "train-only AFD fit table complete\n",
        encoding="utf-8",
    )

    print("=== Train-only AFD fit table ===")
    print("entity table:", entity_path)
    print("target train:", target_path)
    print("output:", output_path)
    print("manifest:", manifest_path)
    print("source shape:", entity_df.shape)
    print("fit shape:", fit_df.shape)
    print(
        "temporal cutoff applied:",
        manifest["temporal_cutoff_applied"],
    )


if __name__ == "__main__":
    main()
