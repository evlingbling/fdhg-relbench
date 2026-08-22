from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class RelBenchBundle:
    dataset_name: str
    task_name: str
    dataset: Any
    task: Any
    db: Any
    tables: Dict[str, pd.DataFrame]
    target_table: Optional[str]
    train_table: Optional[pd.DataFrame]
    val_table: Optional[pd.DataFrame]
    test_table: Optional[pd.DataFrame]


def _as_pandas_table(obj: Any) -> pd.DataFrame:
    """
    Convert common RelBench table-like objects to pandas DataFrame.
    """
    if obj is None:
        raise TypeError("Cannot convert None to pandas DataFrame")

    if isinstance(obj, pd.DataFrame):
        return obj

    if hasattr(obj, "df"):
        df = obj.df
        if isinstance(df, pd.DataFrame):
            return df

    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()

    if hasattr(obj, "to_df"):
        return obj.to_df()

    raise TypeError(f"Cannot convert object of type {type(obj)} to pandas DataFrame")


def _get_dataset(dataset_name: str) -> Any:
    try:
        from relbench.datasets import get_dataset
    except ImportError as exc:
        raise RuntimeError(
            "RelBench is required for dataset loading. "
            'Install the reproduction dependencies with: '
            'python -m pip install -e ".[relbench]"'
        ) from exc

    # Local-cache fast path:
    # RelBench/pooch can try to re-fetch db.zip when the registry hash is stale,
    # even if the extracted parquet cache already exists. For datasets with a
    # complete local cache, prefer process=False to avoid unnecessary re-download.
    from pathlib import Path

    cache_root = Path.home() / ".cache" / "relbench" / dataset_name
    db_dir = cache_root / "db"
    tasks_dir = cache_root / "tasks"

    if db_dir.exists() and tasks_dir.exists():
        try:
            return get_dataset(dataset_name, process=False)
        except TypeError:
            # Some RelBench versions may not support process=False.
            pass
        except Exception as e:
            print(f"[WARN] local-cache get_dataset(process=False) failed for {dataset_name}: {e}")

    return get_dataset(dataset_name)


def _get_task(dataset_name: str, task_name: str) -> Any:
    from relbench.tasks import get_task

    # RelBench versions may use either:
    # get_task(dataset_name, task_name)
    # or get_task(task_name)
    try:
        return get_task(dataset_name, task_name)
    except TypeError:
        return get_task(task_name)


def _extract_db(dataset: Any) -> Any:
    """
    Extract database object from a RelBench dataset object.
    """
    for attr in ["db", "database"]:
        if hasattr(dataset, attr):
            obj = getattr(dataset, attr)
            return obj() if callable(obj) else obj

    for method in ["get_db", "get_database"]:
        if hasattr(dataset, method):
            return getattr(dataset, method)()

    # Some APIs may return the database-like object directly.
    return dataset


def _extract_tables(db: Any) -> Dict[str, pd.DataFrame]:
    """
    Extract all database tables as pandas DataFrames.
    """
    for attr in ["tables", "table_dict"]:
        if hasattr(db, attr):
            raw_tables = getattr(db, attr)
            if isinstance(raw_tables, dict):
                return {
                    str(name): _as_pandas_table(table)
                    for name, table in raw_tables.items()
                }

    if hasattr(db, "table_names") and hasattr(db, "get_table"):
        out = {}
        for name in db.table_names:
            out[str(name)] = _as_pandas_table(db.get_table(name))
        return out

    raise AttributeError(
        "Could not extract tables from database object. "
        "Run an object inspection on db and update _extract_tables()."
    )


def _extract_split_table(task: Any, split: str) -> Optional[pd.DataFrame]:
    """
    Extract train/val/test task table from a RelBench task object.
    """
    method_candidates = [
        f"get_{split}_table",
        f"get_{split}",
        "get_table",
    ]

    for method_name in method_candidates:
        if hasattr(task, method_name):
            method = getattr(task, method_name)
            try:
                if method_name == "get_table":
                    obj = method(split)
                else:
                    obj = method()
                return _as_pandas_table(obj)
            except Exception:
                pass

    attr_candidates = [
        f"{split}_table",
        split,
    ]

    for attr in attr_candidates:
        if hasattr(task, attr):
            obj = getattr(task, attr)
            try:
                return _as_pandas_table(obj)
            except Exception:
                pass

    return None


def _extract_target_table_name(task: Any) -> Optional[str]:
    """
    Try to identify the target/entity table name.
    """
    for attr in ["target_table", "target_table_name", "entity_table", "table"]:
        if hasattr(task, attr):
            value = getattr(task, attr)
            if isinstance(value, str):
                return value

    if hasattr(task, "metadata"):
        metadata = task.metadata
        if isinstance(metadata, dict):
            for key in ["target_table", "target_table_name", "entity_table"]:
                if key in metadata:
                    return metadata[key]

    return None


TARGET_TABLE_OVERRIDES = {
    # Add overrides only if automatic detection fails.
    # Example:
    # ("rel-hm", "user-churn"): "users",
}


def load_relbench_bundle(dataset_name: str, task_name: str) -> RelBenchBundle:
    dataset = _get_dataset(dataset_name)
    task = _get_task(dataset_name, task_name)

    db = _extract_db(dataset)
    tables = _extract_tables(db)

    target_table = _extract_target_table_name(task)
    if target_table is None:
        target_table = TARGET_TABLE_OVERRIDES.get((dataset_name, task_name))

    train_table = _extract_split_table(task, "train")
    val_table = _extract_split_table(task, "val")
    test_table = _extract_split_table(task, "test")

    return RelBenchBundle(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset=dataset,
        task=task,
        db=db,
        tables=tables,
        target_table=target_table,
        train_table=train_table,
        val_table=val_table,
        test_table=test_table,
    )


