from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from fdhg.compiler.ambiguity import normalize_join_key_pair


DBINFER_ONBOARDING_VERSION = "dbinfer-v1"


AVS_REPEATER_RELATION_SPECS = (
    {
        "child_table": "Transaction",
        "child_fk": "id",
        "child_event_time_col": "date",
        "target_lookup_column": "id",
        "parent_table": "Customer",
        "parent_key": "id",
        "feature_namespace": "Transaction",
        "strict_before": False,
        "target_lookup_value_transform":
            "dbinfer_inverse_entity_mapping",
        "relation_orientation":
            "dbinfer_shared_declared_fk",
        "dbinfer_declared_fk_provenance": True,
        "verification_basis":
            "declared_shared_fk_plus_target_inverse_mapping",
        "verified": True,
    },
)


def discover_dbinfer_event_relations(
    *,
    table_dict: Mapping[str, pd.DataFrame],
    train_targets: pd.DataFrame,
    target_time_col: str,
    relation_specs: Sequence[Mapping[str, Any]],
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build audited temporal relation candidates for DBInfer event-row tasks.

    Unlike the RelBench entity-table resolver, DBInfer prediction rows may
    carry multiple relational lookup keys and need not correspond to a
    physical primary-key entity table.

    All checks use train targets only.
    """

    target_times = pd.to_datetime(
        train_targets[target_time_col],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []

    for spec in relation_specs:
        child_table = str(spec["child_table"])
        child_fk = str(spec["child_fk"])
        child_time_col = str(spec["child_event_time_col"])
        target_lookup_column = str(spec["target_lookup_column"])
        target_lookup_value_transform = spec.get(
            "target_lookup_value_transform"
        )
        relation_orientation = spec.get(
            "relation_orientation"
        )

        strict_before = bool(spec.get("strict_before", False))
        allow_exact_matches = not strict_before

        reasons: list[str] = []

        if child_table not in table_dict:
            reasons.append("missing_child_table")
            child = None
        else:
            child = table_dict[child_table]

        if target_lookup_column not in train_targets.columns:
            reasons.append("missing_target_lookup_column")

        if child is not None:
            if child_fk not in child.columns:
                reasons.append("missing_child_fk")

            if child_time_col not in child.columns:
                reasons.append("missing_child_event_time")

        referential_coverage = 0.0
        history_coverage = 0.0
        child_rows = 0

        if (
            child is not None
            and child_fk in child.columns
            and child_time_col in child.columns
            and target_lookup_column in train_targets.columns
        ):
            child_rows = len(child)

            target_lookup_values = (
                train_targets[target_lookup_column]
            )

            if (
                target_lookup_value_transform
                == "dbinfer_inverse_entity_mapping"
            ):
                if not target_lookup_value_mapping:
                    reasons.append(
                        "blocked_missing_dbinfer_inverse_entity_mapping"
                    )
                else:
                    target_lookup_values = (
                        target_lookup_values.map(
                            target_lookup_value_mapping
                        )
                    )

            target_keys, child_keys = normalize_join_key_pair(
                target_lookup_values,
                child[child_fk],
            )

            target_key_series = pd.Series(
                target_keys,
                index=train_targets.index,
            )
            child_key_series = pd.Series(
                child_keys,
                index=child.index,
            )

            target_non_null = target_key_series.dropna()
            child_non_null = child_key_series.dropna()

            if len(target_non_null):
                child_key_set = set(child_non_null.tolist())
                referential_coverage = float(
                    target_non_null.isin(child_key_set).mean()
                )

            child_times = pd.to_datetime(
                child[child_time_col],
                errors="coerce",
            )

            history_coverage = _training_history_coverage(
                target_keys=target_key_series,
                target_times=target_times,
                child_keys=child_key_series,
                child_times=child_times,
                strict_before=strict_before,
            )

            if not child_times.notna().any():
                reasons.append("missing_parseable_child_time")

            if not target_times.notna().any():
                reasons.append("missing_parseable_target_time")

            if history_coverage <= 0.0:
                reasons.append("no_training_history")

        row = {
            "child_table": child_table,
            "child_fk": child_fk,

            # These fields preserve compatibility with the existing
            # Auto candidate schema.  DBInfer may use logical/dummy parents.
            "parent_table": str(
                spec.get("parent_table", "__target_lookup__")
            ),
            "parent_key": str(
                spec.get("parent_key", target_lookup_column)
            ),

            "child_event_time_col": child_time_col,
            "target_lookup_column": target_lookup_column,
            "target_lookup_value_transform":
                target_lookup_value_transform,
            "relation_orientation":
                relation_orientation,
            "dbinfer_declared_fk_provenance": bool(
                spec.get(
                    "dbinfer_declared_fk_provenance",
                    False,
                )
            ),
            "verification_basis": spec.get(
                "verification_basis"
            ),
            "verified": bool(
                spec.get("verified", False)
            ),
            "feature_namespace": str(
                spec.get("feature_namespace", "")
            ),

            "strict_before": strict_before,
            "allow_exact_matches": allow_exact_matches,

            "referential_coverage": referential_coverage,
            "training_target_history_coverage": history_coverage,
            "cold_start_rate": 1.0 - history_coverage,
            "child_rows": child_rows,

            "target_named_column_present": False,

            "status": "rejected" if reasons else "accepted",
            "rejection_reasons": "|".join(reasons),
        }

        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["status"] != "accepted",
            -float(row["training_target_history_coverage"]),
            -float(row["referential_coverage"]),
            -int(row["child_rows"]),
            str(row["child_table"]),
            str(row["child_fk"]),
            str(row["target_lookup_column"]),
        )
    )

    rank = 1

    for row in rows:
        if row["status"] == "accepted":
            row["relation_rank"] = rank
            rank += 1
        else:
            row["relation_rank"] = ""

    return rows


def _training_history_coverage(
    *,
    target_keys: pd.Series,
    target_times: pd.Series,
    child_keys: pd.Series,
    child_times: pd.Series,
    strict_before: bool,
) -> float:
    """Fraction of train prediction rows with >=1 admissible historical row."""

    source = pd.DataFrame(
        {
            "_key": child_keys,
            "_time": child_times,
        }
    )

    source = source[
        source["_key"].notna()
        & source["_time"].notna()
    ]

    if source.empty:
        return 0.0

    first_source_time = source.groupby(
        "_key",
        sort=False,
        dropna=False,
    )["_time"].min()

    mapped_first_time = target_keys.map(first_source_time)

    valid = (
        mapped_first_time.notna()
        & target_times.notna()
    )

    if strict_before:
        hit = valid & (mapped_first_time < target_times)
    else:
        hit = valid & (mapped_first_time <= target_times)

    return float(hit.mean())


RETAILROCKET_CVR_RELATION_SPECS = (
    {
        "child_table": "ItemAvailability",
        "child_fk": "itemid",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "itemid",
        "parent_table": "Item",
        "parent_key": "itemid",
        "strict_before": False,
    },
    {
        "child_table": "ItemCategory",
        "child_fk": "itemid",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "itemid",
        "parent_table": "Item",
        "parent_key": "itemid",
        "strict_before": False,
    },
    {
        "child_table": "ItemProperty",
        "child_fk": "itemid",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "itemid",
        "parent_table": "Item",
        "parent_key": "itemid",
        "strict_before": False,
    },
    {
        "child_table": "View",
        "child_fk": "itemid",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "itemid",
        "parent_table": "Item",
        "parent_key": "itemid",
        "feature_namespace": "by_itemid",
        "strict_before": True,
    },
    {
        "child_table": "View",
        "child_fk": "visitorid",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "visitorid",
        "parent_table": "Visitor",
        "parent_key": "id",
        "feature_namespace": "by_visitorid",
        "strict_before": True,
    },
)


DIGINETICA_CTR_RELATION_SPECS = (
    {
        "child_table": "Query",
        "child_fk": "queryId",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "queryId",
        "parent_table": "Query",
        "parent_key": "queryId",
        "feature_namespace": "by_queryid",
        "strict_before": False,
    },
    {
        "child_table": "Click",
        "child_fk": "itemId",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "itemId",
        "parent_table": "Product",
        "parent_key": "itemId",
        "feature_namespace": "click_by_itemid",
        "strict_before": True,
    },
    {
        "child_table": "View",
        "child_fk": "itemId",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "itemId",
        "parent_table": "Product",
        "parent_key": "itemId",
        "feature_namespace": "view_by_itemid",
        "strict_before": True,
    },
    {
        "child_table": "Purchase",
        "child_fk": "itemId",
        "child_event_time_col": "timestamp",
        "target_lookup_column": "itemId",
        "parent_table": "Product",
        "parent_key": "itemId",
        "feature_namespace": "purchase_by_itemid",
        "strict_before": True,
    },
)


AVS_REPEATER_RELATION_SPECS = (
    {
        "child_table": "Transaction",
        "child_fk": "id",
        "child_event_time_col": "date",
        "target_lookup_column": "id",
        "parent_table": "Customer",
        "parent_key": "id",
        "strict_before": False,
    },
)

# ===== DBINFER RELBENCH-LIKE ADAPTER =====

from pathlib import Path as _Path
from typing import Any as _Any, Mapping as _Mapping

import pandas as _pd
import yaml as _yaml


class _DBInferTable:
    """Minimal table interface consumed by the existing Auto/FDHG pipeline."""

    def __init__(
        self,
        df: _pd.DataFrame,
        *,
        pkey_col: str | None = None,
        time_col: str | None = None,
        fkey_col_to_pkey_table: _Mapping[str, _Any] | None = None,
    ) -> None:
        self.df = df
        self.pkey_col = pkey_col
        self.time_col = time_col
        self.fkey_col_to_pkey_table = dict(
            fkey_col_to_pkey_table or {}
        )


class _DBInferDatabase:
    def __init__(self, table_dict: _Mapping[str, _DBInferTable]) -> None:
        self.table_dict = dict(table_dict)


class _DBInferDataset:
    def __init__(self, database: _DBInferDatabase) -> None:
        self._database = database

    def get_db(self) -> _DBInferDatabase:
        return self._database


class _DBInferTask:
    """RelBench-like prediction-task interface for exported DBInfer tasks."""

    def __init__(
        self,
        *,
        train: _pd.DataFrame,
        validation: _pd.DataFrame,
        entity_col: str,
        time_col: str,
        target_col: str,
        target_table: str,
        problem_type: str = "binary_classification",
        primary_metric: str = "roc_auc",
        metric_direction: str = "higher",
    ) -> None:
        self.entity_col = entity_col
        self.time_col = time_col
        self.target_col = target_col

        self.task_type = problem_type
        self.problem_type = problem_type

        self.primary_metric = primary_metric
        self.metric = primary_metric
        self.metrics = (primary_metric,)
        self.metric_direction = metric_direction

        self.entity_table = target_table
        self.target_table = target_table

        self._split_tables = {
            "train": _DBInferTable(train),
            "val": _DBInferTable(validation),
        }

    def get_table(self, split: str) -> _DBInferTable:
        if split not in self._split_tables:
            raise ValueError(
                "unsupported_dbinfer_task_split:"
                + str(split)
            )
        return self._split_tables[split]


def load_dbinfer_relbench_like_objects(
    dataset_name: str,
    task_name: str,
    download: bool = False,
):
    """
    Load an exported DBInfer task through the object interface expected by
    prepare_auto_onboarding().

    Only train and validation prediction rows are read.  The DBInfer test split
    is intentionally never opened here.
    """
    del download

    supported = {
        ("dbinfer-retailrocket", "cvr"),
        ("dbinfer-diginetica", "ctr"),
        ("dbinfer-avs", "repeater"),
    }

    if (dataset_name, task_name) not in supported:
        raise ValueError(
            "unsupported_dbinfer_task:"
            + str(dataset_name)
            + "/"
            + str(task_name)
        )

    repo_root = _Path(__file__).resolve().parents[3]
    export_root = (
        repo_root
        / "data"
        / "dbinfer-export"
        / dataset_name
        / task_name
    )
    config_path = export_root / "onboarding.yaml"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = _yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    table_dict: dict[str, _DBInferTable] = {}

    configured_tables = dict(
        config.get("tables") or {}
    )

    # Diginetica CTR has a ~92M-row QueryResult table. Prediction
    # rows are already exported separately under targets/{split}.parquet,
    # and QueryResult is intentionally not an Auto/FDHG source relation.
    # Load only the source tables used by the declared CTR relation specs.
    if (
        dataset_name == "dbinfer-diginetica"
        and task_name == "ctr"
    ):
        source_table_names = {
            "Query",
            "Click",
            "View",
            "Purchase",
        }

        configured_tables = {
            name: raw
            for name, raw in configured_tables.items()
            if name in source_table_names
        }

    for table_name, raw in sorted(
        configured_tables.items()
    ):
        table_path = _Path(str(raw["path"]))

        if not table_path.is_absolute():
            table_path = (
                config_path.parent / table_path
            ).resolve()

        df = _pd.read_parquet(table_path)

        # Preserve DBInfer schema semantics that would otherwise be lost
        # when parquet integer codes are passed into the generic compiler.
        #
        # Identifier columns must never be interpreted quantitatively.
        # Likewise categoryId is a categorical code, not a numeric magnitude.
        if (
            dataset_name == "dbinfer-retailrocket"
            and str(table_name) == "ItemCategory"
            and "category" in df.columns
        ):
            df = df.copy()
            df["category"] = df["category"].astype("string")

        if (
            dataset_name == "dbinfer-diginetica"
            and task_name == "ctr"
        ):
            # Preserve physical relation lookup keys as their original
            # integer dtype so DBInfer target/source joins remain exact.
            #
            # Only non-lookup identifier/category codes are converted away
            # from numeric magnitude semantics.
            semantic_string_columns_by_table = {
                "Query": {
                    "sessionId",
                    "userId",
                    "categoryId",
                },
                "Click": {
                    "queryId",
                },
                "View": {
                    "sessionId",
                    "userId",
                },
                "Purchase": {
                    "sessionId",
                    "userId",
                    "ordernumber",
                },
            }

            cast_columns = (
                semantic_string_columns_by_table.get(
                    str(table_name),
                    set(),
                )
                & set(df.columns)
            )

            if cast_columns:
                df = df.copy()

                for column in sorted(cast_columns):
                    df[column] = df[column].astype("string")

        table_dict[str(table_name)] = _DBInferTable(
            df,
            pkey_col=(
                None
                if raw.get("primary_key") is None
                else str(raw["primary_key"])
            ),
            time_col=(
                None
                if raw.get("event_time_col") is None
                else str(raw["event_time_col"])
            ),
            # DBInfer logical event relations are resolved separately by
            # discover_dbinfer_event_relations().  Physical FK metadata is not
            # used to infer the prediction-row relation semantics here.
            fkey_col_to_pkey_table={},
        )

    split_cfg = dict(config.get("split") or {})

    if bool(split_cfg.get("test_split_accessed", False)):
        raise ValueError(
            "dbinfer_export_declares_test_split_accessed"
        )

    train_path = _Path(
        str(split_cfg["train_target_path"])
    )
    validation_path = _Path(
        str(split_cfg["validation_target_path"])
    )

    if not train_path.is_absolute():
        train_path = (
            config_path.parent / train_path
        ).resolve()

    if not validation_path.is_absolute():
        validation_path = (
            config_path.parent / validation_path
        ).resolve()

    train = _pd.read_parquet(train_path)
    validation = _pd.read_parquet(validation_path)

    task_cfg = dict(config.get("task") or {})

    entity_col = str(task_cfg["entity_key"])
    time_col = str(task_cfg["target_time_col"])
    target_col = str(task_cfg["label_col"])
    target_table = str(task_cfg["target_table"])
    problem_type = str(task_cfg["problem_type"])
    primary_metric = str(task_cfg["primary_metric"])
    metric_direction = str(task_cfg["metric_direction"])

    required_target_columns = {
        entity_col,
        time_col,
        target_col,
    }

    # Logical lookup columns required by each supported DBInfer task.
    if (
        dataset_name == "dbinfer-retailrocket"
        and task_name == "cvr"
    ):
        required_target_columns.update(
            {"itemid", "visitorid"}
        )

    elif (
        dataset_name == "dbinfer-diginetica"
        and task_name == "ctr"
    ):
        required_target_columns.update(
            {"queryId", "itemId"}
        )

    for split_name, frame in (
        ("train", train),
        ("validation", validation),
    ):
        missing = sorted(
            required_target_columns - set(frame.columns)
        )
        if missing:
            raise ValueError(
                "missing_dbinfer_target_columns:"
                + split_name
                + ":"
                + ",".join(missing)
            )

    dataset = _DBInferDataset(
        _DBInferDatabase(table_dict)
    )

    task = _DBInferTask(
        train=train,
        validation=validation,
        entity_col=entity_col,
        time_col=time_col,
        target_col=target_col,
        target_table=target_table,
        problem_type=problem_type,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
    )

    if (
        dataset_name == "dbinfer-avs"
        and task_name == "repeater"
    ):
        from types import SimpleNamespace

        mapping_path = (
            config_path.parent
            / "entity_mapping.parquet"
        )

        if not mapping_path.exists():
            raise FileNotFoundError(
                "missing_dbinfer_entity_mapping:"
                + str(mapping_path)
            )

        mapping_frame = _pd.read_parquet(
            mapping_path,
            columns=[
                "raw_entity_id",
                "mapped_entity_id",
            ],
        )

        if (
            mapping_frame["raw_entity_id"].duplicated().any()
            or
            mapping_frame["mapped_entity_id"].duplicated().any()
        ):
            raise ValueError(
                "non_bijective_dbinfer_entity_mapping"
            )

        entity_mapping = dict(
            zip(
                mapping_frame["raw_entity_id"].tolist(),
                mapping_frame["mapped_entity_id"].tolist(),
            )
        )

        task._task_adapter = SimpleNamespace(
            entity_mapping=entity_mapping,
        )

        # AVS prediction rows use DBInfer's dense mapped entity IDs,
        # whereas Transaction.id stores the original raw customer IDs.
        # Expose the raw lookup value explicitly so downstream relational
        # feature materializers can perform exact joins without depending
        # on an in-memory adapter mapping.
        inverse_entity_mapping = {
            mapped: raw
            for raw, mapped in entity_mapping.items()
        }

        for split_name in ("train", "val"):
            split_frame = task.get_table(split_name).df
            raw_lookup = split_frame[entity_col].map(
                inverse_entity_mapping
            )

            if raw_lookup.isna().any():
                raise ValueError(
                    "incomplete_dbinfer_inverse_entity_mapping:"
                    + split_name
                )

            split_frame["__dbinfer_raw_entity_id"] = (
                raw_lookup
            )

    return (
        dataset,
        task,
        dataset_name + "-export-v1",
    )

