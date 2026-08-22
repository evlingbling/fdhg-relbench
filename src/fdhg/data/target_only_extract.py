from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from fdhg.data.relbench_loader import load_relbench_bundle


def get_task_registry_entry(dataset_name: str, task_name: str):
    from relbench.tasks import task_registry

    if dataset_name not in task_registry:
        raise KeyError(f"Unknown dataset_name={dataset_name}")

    if task_name not in task_registry[dataset_name]:
        raise KeyError(f"Unknown task_name={task_name} for dataset={dataset_name}")

    task_cls, args, kwargs = task_registry[dataset_name][task_name]
    return task_cls, args, kwargs


def get_task_metadata(dataset_name: str, task_name: str) -> Dict[str, Optional[str]]:
    task_cls, _, _ = get_task_registry_entry(dataset_name, task_name)

    metadata = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "task_class": task_cls.__name__,
        "entity_table": getattr(task_cls, "entity_table", None),
        "entity_col": getattr(task_cls, "entity_col", None),
        "target_col": getattr(task_cls, "target_col", None),
        "time_col": getattr(task_cls, "time_col", None),
        "task_type": str(getattr(task_cls, "task_type", None)),
    }
    return metadata


def is_entity_prediction_task(metadata: Dict[str, Optional[str]]) -> bool:
    """
    Entity-level classification/regression tasks have:
    entity_table, entity_col, target_col.

    Link-prediction tasks usually do not have these in the same format,
    so Step 2 excludes them.
    """
    return (
        metadata.get("entity_table") is not None
        and metadata.get("entity_col") is not None
        and metadata.get("target_col") is not None
        and "LINK_PREDICTION" not in str(metadata.get("task_type"))
    )


def normalize_name(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


def is_sequence_cell(x: Any) -> bool:
    """
    Detect non-scalar sequence cells such as numpy arrays, lists, or tuples.
    Strings and scalar numpy values should not be treated as sequences.
    """
    if x is None:
        return False

    if isinstance(x, (str, bytes)):
        return False

    if isinstance(x, (list, tuple)):
        return True

    # numpy ndarray-like object
    if hasattr(x, "shape") and hasattr(x, "__array__"):
        shape = getattr(x, "shape", None)

        # numpy scalar has shape == ()
        if shape == ():
            return False

        # real array/vector has shape like (3,), (2, 4), etc.
        try:
            return len(x) > 0
        except TypeError:
            return False

    return False


def find_sequence_columns(df: pd.DataFrame, max_check: int = 100) -> Dict[str, str]:
    """
    Find columns containing array/list/tuple-valued cells.
    Scalar numpy values are ignored.
    """
    out = {}

    for col in df.columns:
        sample = df[col].dropna().head(max_check)
        if sample.empty:
            continue

        for value in sample:
            if is_sequence_cell(value):
                out[col] = (
                    f"sequence_valued_column("
                    f"type={type(value).__name__}, "
                    f"shape={getattr(value, 'shape', None)})"
                )
                break

    return out


def scalarize_sequence_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert sequence-valued columns into multiple scalar columns.

    Example:
        category = np.array(["A", "B", "C"])

    becomes:
        category__0 = "A"
        category__1 = "B"
        category__2 = "C"
    """
    sequence_cols = find_sequence_columns(df)
    if not sequence_cols:
        return df, {}

    df = df.copy()
    info = {}

    for col in sequence_cols:
        non_null = df[col].dropna()

        max_len = 0
        for value in non_null:
            if is_sequence_cell(value):
                try:
                    max_len = max(max_len, len(value))
                except TypeError:
                    continue

        new_cols = []
        for idx in range(max_len):
            new_col = f"{col}__{idx}"

            def get_item(value, i=idx):
                if not is_sequence_cell(value):
                    return None
                try:
                    if i >= len(value):
                        return None
                    return value[i]
                except TypeError:
                    return None

            df[new_col] = df[col].map(get_item)
            new_cols.append(new_col)

        df = df.drop(columns=[col])

        info[col] = {
            "reason": sequence_cols[col],
            "max_len": max_len,
            "created_columns": new_cols,
        }

    return df, info


def resolve_entity_join_column(entity_df: pd.DataFrame, split_entity_col: str) -> str:
    """
    Resolve which column in the entity table should be joined with split_entity_col.

    Usually identical:
        customer_id -> customer_id

    Sometimes different:
        split OwnerUserId -> users.Id
        split PostId -> posts.Id
        split user -> users.user_id
    """
    entity_cols = list(entity_df.columns)

    # 1. Exact match.
    if split_entity_col in entity_cols:
        return split_entity_col

    # 2. Case-insensitive / punctuation-insensitive match.
    norm_split = normalize_name(split_entity_col)
    for col in entity_cols:
        if normalize_name(col) == norm_split:
            return col

    # 3. Common suffix match:
    # OwnerUserId -> UserId or Id
    for col in entity_cols:
        norm_col = normalize_name(col)
        if norm_split.endswith(norm_col) or norm_col.endswith(norm_split):
            return col

    # 4. Prefer common explicit ID columns when available.
    common_id_cols = ["id", "Id", "ID", "user_id", "UserId", "PostId"]
    for col in common_id_cols:
        if col in entity_cols:
            return col

    # 5. Unique ID-like fallback.
    id_like_cols = [
        col for col in entity_cols
        if "id" in normalize_name(col) or normalize_name(col) in {"user", "item", "product"}
    ]

    unique_candidates = []
    n = len(entity_df)
    for col in id_like_cols:
        try:
            nunique = entity_df[col].nunique(dropna=True)
        except TypeError:
            continue
        if n > 0 and nunique / n > 0.95:
            unique_candidates.append(col)

    if len(unique_candidates) == 1:
        return unique_candidates[0]

    raise ValueError(
        f"Could not resolve join column. split_entity_col={split_entity_col}, "
        f"entity_table_columns={entity_cols}, unique_id_candidates={unique_candidates}"
    )


def build_target_only_split(
    split_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    metadata: Dict[str, Optional[str]],
    split_name: str,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    Step 2 target-only extraction for one split.

    What this does:
    - join split table with entity table
    - separate target column as y
    - remove non-feature columns:
      target_col, time_col, entity_col, join key, split index

    What this does NOT do:
    - categorical encoding
    - scaling
    - imputation
    - high-cardinality filtering
    - TabPFN/TabICL inference
    """
    entity_col = metadata["entity_col"]
    target_col = metadata["target_col"]
    time_col = metadata["time_col"]

    if entity_col is None or target_col is None:
        raise ValueError("entity_col and target_col are required for entity prediction tasks.")

    if entity_col not in split_df.columns:
        raise KeyError(f"{entity_col} not found in {split_name} split columns: {list(split_df.columns)}")

    entity_join_col = resolve_entity_join_column(entity_df, entity_col)

    entity_unique = entity_df.drop_duplicates(subset=[entity_join_col]).copy()

    if len(entity_unique) != len(entity_df):
        print(
            f"[WARN] Entity table has duplicate rows for join column {entity_join_col}. "
            f"Using drop_duplicates for target-only extraction."
        )

    merged = split_df.merge(
        entity_unique,
        how="left",
        left_on=entity_col,
        right_on=entity_join_col,
        suffixes=("_split", "_entity"),
    )

    y = None
    if target_col in merged.columns:
        y = merged[target_col].copy()

    drop_cols = set()

    # Remove entity and join identifiers from predictive X.
    drop_cols.add(entity_col)
    drop_cols.add(entity_join_col)

    # Remove task time/cutoff column.
    if time_col is not None:
        drop_cols.add(time_col)

    # Remove target column.
    if target_col is not None:
        drop_cols.add(target_col)

    # Remove common split/index columns.
    for col in merged.columns:
        if col == "index" or col.endswith("_split"):
            drop_cols.add(col)

    # Remove duplicate versions of entity column after merge.
    for col in merged.columns:
        if normalize_name(col) == normalize_name(entity_col):
            if col != target_col:
                drop_cols.add(col)

    X = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")

    feature_cols = list(X.columns)

    info = {
        "split_name": split_name,
        "n_rows": len(split_df),
        "entity_join_col": entity_join_col,
        "n_missing_entity_matches": int(merged[entity_join_col].isna().sum()) if entity_join_col in merged else None,
        "feature_cols": feature_cols,
        "dropped_columns": {
            "non_feature_cols": sorted([c for c in drop_cols if c in merged.columns]),
        },
        "has_target": y is not None,
    }

    return X, y, info


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, path)


def extract_target_only(
    dataset_name: str,
    task_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    metadata = get_task_metadata(dataset_name, task_name)

    if not is_entity_prediction_task(metadata):
        raise ValueError(
            f"{dataset_name}/{task_name} is not an entity-level classification/regression task. "
            f"metadata={metadata}"
        )

    bundle = load_relbench_bundle(dataset_name, task_name)

    entity_table = metadata["entity_table"]
    if entity_table not in bundle.tables:
        raise KeyError(f"entity_table={entity_table} not found in database tables: {list(bundle.tables)}")

    entity_df = bundle.tables[entity_table]

    # Step 2: make entity table rectangular if it has sequence-valued cells.
    entity_df, sequence_scalarization_info = scalarize_sequence_columns(entity_df)

    task_out = output_dir / dataset_name / task_name
    task_out.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "metadata": metadata,
        "entity_table_shape_after_scalarization": entity_df.shape,
        "sequence_scalarization": sequence_scalarization_info,
        "splits": {},
    }

    for split_name, split_df in [
        ("train", bundle.train_table),
        ("val", bundle.val_table),
        ("test", bundle.test_table),
    ]:
        if split_df is None:
            summary["splits"][split_name] = {"available": False}
            continue

        X, y, info = build_target_only_split(
            split_df=split_df,
            entity_df=entity_df,
            metadata=metadata,
            split_name=split_name,
        )

        save_pickle(X, task_out / f"X_{split_name}.pkl")
        if y is not None:
            save_pickle(y, task_out / f"y_{split_name}.pkl")

        summary["splits"][split_name] = {
            "available": True,
            "X_shape": X.shape,
            "y_shape": None if y is None else y.shape,
            **info,
        }

    with open(task_out / "metadata.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def iter_entity_tasks():
    from relbench.tasks import task_registry

    for dataset_name in sorted(task_registry.keys()):
        for task_name in sorted(task_registry[dataset_name].keys()):
            metadata = get_task_metadata(dataset_name, task_name)
            if is_entity_prediction_task(metadata):
                yield dataset_name, task_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--all-entity", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/target_only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all_entity:
        results = []
        failed = []

        for dataset_name, task_name in iter_entity_tasks():
            print("\n" + "=" * 100)
            print(f"Extracting target-only features: {dataset_name}/{task_name}")
            print("=" * 100)

            try:
                summary = extract_target_only(
                    dataset_name=dataset_name,
                    task_name=task_name,
                    output_dir=output_dir,
                )
                print(json.dumps(summary, indent=2, default=str))
                results.append((dataset_name, task_name))
            except Exception as e:
                print("FAILED")
                print(type(e).__name__, str(e))
                failed.append((dataset_name, task_name, type(e).__name__, str(e)))

        print("\n" + "=" * 100)
        print("TARGET-ONLY EXTRACTION SUMMARY")
        print("=" * 100)
        print(f"Success: {len(results)}")
        print(f"Failed:  {len(failed)}")

        if failed:
            print("\nFailed tasks:")
            for dataset_name, task_name, err_type, err_msg in failed:
                print(f"- {dataset_name}/{task_name}: {err_type}: {err_msg}")

        return

    if args.dataset is None or args.task is None:
        raise ValueError("Provide either --all-entity or both --dataset and --task.")

    summary = extract_target_only(
        dataset_name=args.dataset,
        task_name=args.task,
        output_dir=output_dir,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
