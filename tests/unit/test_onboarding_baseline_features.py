from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import fdhg.onboarding.pipeline as onboarding_pipeline
from fdhg.onboarding.pipeline import (
    _apply_features,
    _declared_baseline_features,
    _feature_name,
    onboard_dataset,
)
from tests.unit.onboarding_fixtures import write_onboarding_fixture


def _prepared(tmp_path: Path):
    config_path = write_onboarding_fixture(tmp_path)
    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=True,
    )
    assert report.status == "completed"
    train = pd.read_parquet(
        report.output_dir / "target_with_dfs_agg_train.parquet"
    )
    val = pd.read_parquet(
        report.output_dir / "target_with_dfs_agg_val.parquet"
    )
    return report.output_dir, train, val


def test_count_is_temporally_correct(tmp_path: Path) -> None:
    _, train, val = _prepared(tmp_path)

    assert train.loc[train["user_id"] == "u1", "f_events_count"].item() == 2.0
    assert train.loc[train["user_id"] == "u2", "f_events_count"].item() == 0.0
    assert val.loc[val["user_id"] == "u3", "f_events_count"].item() == 1.0


def test_numeric_mean_std_min_and_max_are_temporally_correct(
    tmp_path: Path,
) -> None:
    _, train, val = _prepared(tmp_path)
    u1 = train.loc[train["user_id"] == "u1"].iloc[0]

    assert u1["f_events_amount_mean"] == 15.0
    assert u1["f_events_amount_std"] == 5.0
    assert u1["f_events_amount_min"] == 10.0
    assert u1["f_events_amount_max"] == 20.0
    assert val.loc[val["user_id"] == "u3", "f_events_amount_mean"].item() == 7.0


def test_days_since_last_is_temporally_correct(tmp_path: Path) -> None:
    _, train, _ = _prepared(tmp_path)

    assert train.loc[train["user_id"] == "u1", "f_events_days_since_last"].item() == 5.0
    assert math.isnan(
        train.loc[train["user_id"] == "u2", "f_events_days_since_last"].item()
    )


def test_past_unique_values_is_temporally_correct(tmp_path: Path) -> None:
    _, train, val = _prepared(tmp_path)

    assert train.loc[train["user_id"] == "u1", "f_events_category_nunique"].item() == 2.0
    assert val.loc[val["user_id"] == "u3", "f_events_category_nunique"].item() == 1.0


def test_post_target_events_never_affect_features(tmp_path: Path) -> None:
    _, train, val = _prepared(tmp_path)

    assert train.loc[train["user_id"] == "u1", "f_events_amount_max"].item() != 999.0
    assert val.loc[val["user_id"] == "u3", "f_events_amount_max"].item() != 1000.0


def test_deterministic_output_column_order_and_identical_schema(
    tmp_path: Path,
) -> None:
    _, train, val = _prepared(tmp_path)

    assert list(train.columns) == list(val.columns)
    assert list(train.columns)[4:] == [
        "f_events_count",
        "f_events_count__is_missing",
        "f_events_amount_mean",
        "f_events_amount_mean__is_missing",
        "f_events_amount_std",
        "f_events_amount_std__is_missing",
        "f_events_amount_min",
        "f_events_amount_min__is_missing",
        "f_events_amount_max",
        "f_events_amount_max__is_missing",
        "f_events_days_since_last",
        "f_events_days_since_last__is_missing",
        "f_events_category_nunique",
        "f_events_category_nunique__is_missing",
    ]


def test_no_auxiliary_missing_columns_are_logical_primitives(
    tmp_path: Path,
) -> None:
    out, _, _ = _prepared(tmp_path)
    manifest = pd.read_csv(out / "baseline_feature_manifest.csv")
    primitive_ids = tuple(manifest["primitive_id"])

    assert all("__is_missing" not in item for item in primitive_ids)
    assert tuple(primitive_ids) == (
        "baseline::count",
        "baseline::numeric_mean",
        "baseline::numeric_std",
        "baseline::numeric_min",
        "baseline::numeric_max",
        "baseline::days_since_last",
        "baseline::history::past_unique_values",
    )


def _apply_features_reference(
    target,
    child,
    features,
    child_table,
    child_fk,
    child_time_col,
    entity_key,
    target_time_col,
):
    out = target.copy().reset_index(drop=True)
    child = child.copy()
    child[child_time_col] = pd.to_datetime(child[child_time_col], errors="coerce")
    target_times = pd.to_datetime(out[target_time_col], errors="coerce")
    for _, source_col, agg in features:
        col = _feature_name(child_table, source_col, agg)
        vals = []
        for idx, row in out.iterrows():
            target_time = target_times.iloc[idx]
            eligible = child[
                (child[child_fk] == row[entity_key])
                & (child[child_time_col] <= target_time)
            ]
            if agg == "count":
                value = float(len(eligible))
            elif agg == "days_since_last":
                if eligible.empty:
                    value = math.nan
                else:
                    value = (
                        target_time - eligible[child_time_col].max()
                    ).total_seconds() / 86400.0
            elif agg == "nunique":
                value = float(eligible[source_col].nunique(dropna=True))
            else:
                series = eligible[source_col].dropna()
                if series.empty:
                    value = math.nan
                elif agg == "mean":
                    value = float(series.mean())
                elif agg == "std":
                    value = float(series.std(ddof=0))
                elif agg == "min":
                    value = float(series.min())
                elif agg == "max":
                    value = float(series.max())
                else:
                    raise ValueError(f"unsupported aggregation:{agg}")
            vals.append(value)
        out[col] = vals
        out[f"{col}__is_missing"] = pd.Series(vals).isna()
    return out


def test_optimized_features_match_rowwise_reference() -> None:
    target = pd.DataFrame([
        {"user_id": "u2", "timestamp": pd.Timestamp("2026-01-03"), "label": 0},
        {"user_id": "u1", "timestamp": pd.Timestamp("2026-01-05"), "label": 1},
        {"user_id": "u1", "timestamp": pd.Timestamp("2026-01-05"), "label": 2},
        {"user_id": "u3", "timestamp": pd.Timestamp("2026-01-10"), "label": 3},
        {"user_id": "u1", "timestamp": pd.Timestamp("2026-01-01"), "label": 4},
        {"user_id": "u4", "timestamp": pd.Timestamp("2026-01-01"), "label": 5},
    ])
    child = pd.DataFrame([
        {"event_id": 1, "user_id": "u1", "created_at": pd.Timestamp("2026-01-01"), "amount": 10.0, "category": "a"},
        {"event_id": 2, "user_id": "u1", "created_at": pd.Timestamp("2026-01-05"), "amount": None, "category": "a"},
        {"event_id": 3, "user_id": "u1", "created_at": pd.Timestamp("2026-01-05"), "amount": 20.0, "category": "b"},
        {"event_id": 4, "user_id": "u1", "created_at": pd.Timestamp("2026-01-06"), "amount": 999.0, "category": "leak"},
        {"event_id": 5, "user_id": "u2", "created_at": pd.Timestamp("2026-01-02"), "amount": None, "category": "c"},
        {"event_id": 6, "user_id": "u2", "created_at": pd.Timestamp("2026-01-03"), "amount": 7.0, "category": "c"},
        {"event_id": 7, "user_id": "u4", "created_at": pd.Timestamp("2026-01-02"), "amount": 5.0, "category": None},
    ])
    features = _declared_baseline_features(
        child_table="events",
        child=child,
        child_fk="user_id",
        child_time_col="created_at",
        numeric_col="amount",
        child_primary_key="event_id",
    )

    optimized = _apply_features(
        target,
        child,
        features,
        "events",
        "user_id",
        "created_at",
        "user_id",
        "timestamp",
        child_primary_key="event_id",
    )
    reference = _apply_features_reference(
        target,
        child,
        features,
        "events",
        "user_id",
        "created_at",
        "user_id",
        "timestamp",
    )

    pd.testing.assert_frame_equal(optimized, reference)
    assert list(optimized["label"]) == [0, 1, 2, 3, 4, 5]
    u1_repeated = optimized.loc[
        (optimized["user_id"] == "u1")
        & (optimized["timestamp"] == pd.Timestamp("2026-01-05"))
    ]
    assert u1_repeated["f_events_count"].tolist() == [3.0, 3.0]
    assert u1_repeated["f_events_amount_std"].tolist() == [5.0, 5.0]
    assert optimized.loc[3, "f_events_count"] == 0.0
    assert optimized.loc[4, "f_events_count"] == 1.0
    assert optimized.loc[4, "f_events_amount_std"] == 0.0
    assert optimized.loc[4, "f_events_amount_min"] == 10.0
    assert optimized.loc[4, "f_events_amount_max"] == 10.0
    assert optimized.loc[5, "f_events_count"] == 0.0


def test_optimized_path_selected_for_large_synthetic_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = pd.DataFrame({
        "event_id": range(10000),
        "user_id": [f"u{idx % 100}" for idx in range(10000)],
        "created_at": pd.to_datetime("2026-01-01") + pd.to_timedelta(
            [idx % 50 for idx in range(10000)],
            unit="D",
        ),
        "amount": [float(idx % 17) for idx in range(10000)],
        "category": [f"c{idx % 5}" for idx in range(10000)],
    })
    target = pd.DataFrame({
        "user_id": [f"u{idx % 100}" for idx in range(1000)],
        "timestamp": pd.to_datetime("2026-02-01") + pd.to_timedelta(
            [idx % 10 for idx in range(1000)],
            unit="D",
        ),
        "label": [float(idx % 3) for idx in range(1000)],
    })

    def fail_reference(*args, **kwargs):
        raise AssertionError("row-wise reference path called")

    monkeypatch.setattr(
        pd,
        "merge",
        fail_reference,
    )
    monkeypatch.setattr(pd.DataFrame, "merge", fail_reference)

    result, train, val = _materialize_direct(target, target, child)
    assert result["workload"]["materialization_strategy"] == "grouped_temporal_sweep"
    assert not hasattr(onboarding_pipeline, "_apply_features_reference")
    assert len(train) == 1000
    assert list(train.columns) == list(val.columns)
    assert "f_events_count" in train.columns


def _materialize_direct(train, val, child):
    result = onboarding_pipeline.build_baseline_features(
        dataset="d",
        task="t",
        target_table="users",
        target_train=train,
        target_val=val,
        child_table="events",
        child=child,
        entity_key="user_id",
        child_fk="user_id",
        child_time_col="created_at",
        target_time_col="timestamp",
        label_col="label",
        numeric_col="amount",
        child_primary_key="event_id",
    )
    return result, result["train"], result["validation"]
