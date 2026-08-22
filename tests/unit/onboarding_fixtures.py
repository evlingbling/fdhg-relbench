from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def write_onboarding_fixture(
    tmp_path: Path,
    *,
    table_format: str = "parquet",
    orphan: bool = False,
    ambiguous_time: bool = False,
    missing_label: bool = False,
    invalid_split: bool = False,
) -> Path:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    users = pd.DataFrame([
        {
            "user_id": "u1",
            "signup_time": pd.Timestamp("2025-12-01"),
            "timestamp": pd.Timestamp("2026-01-10"),
            "future_spend": 100.0,
        },
        {
            "user_id": "u2",
            "signup_time": pd.Timestamp("2025-12-02"),
            "timestamp": pd.Timestamp("2026-01-20"),
            "future_spend": 50.0,
        },
        {
            "user_id": "u3",
            "signup_time": pd.Timestamp("2025-12-03"),
            "timestamp": pd.Timestamp("2026-02-10"),
            "future_spend": 80.0,
        },
        {
            "user_id": "u4",
            "signup_time": pd.Timestamp("2025-12-04"),
            "timestamp": pd.Timestamp("2026-02-15"),
            "future_spend": 20.0,
        },
    ])
    events = pd.DataFrame([
        {
            "event_id": "e1",
            "user_id": "u1",
            "created_at": pd.Timestamp("2026-01-01"),
            "amount": 10.0,
            "category": "a",
        },
        {
            "event_id": "e2",
            "user_id": "u1",
            "created_at": pd.Timestamp("2026-01-05"),
            "amount": 20.0,
            "category": "b",
        },
        {
            "event_id": "e3",
            "user_id": "u1",
            "created_at": pd.Timestamp("2026-01-11"),
            "amount": 999.0,
            "category": "z",
        },
        {
            "event_id": "e4",
            "user_id": "u2",
            "created_at": pd.Timestamp("2026-01-21"),
            "amount": 5.0,
            "category": "c",
        },
        {
            "event_id": "e5",
            "user_id": "u3",
            "created_at": pd.Timestamp("2026-02-01"),
            "amount": 7.0,
            "category": "a",
        },
        {
            "event_id": "e6",
            "user_id": "u3",
            "created_at": pd.Timestamp("2026-02-11"),
            "amount": 1000.0,
            "category": "leak",
        },
        {
            "event_id": "e7",
            "user_id": "u4",
            "created_at": pd.Timestamp("2026-01-15"),
            "amount": None,
            "category": "c",
        },
    ])
    if orphan:
        events.loc[len(events)] = {
            "event_id": "e8",
            "user_id": "missing",
            "created_at": pd.Timestamp("2026-01-02"),
            "amount": 3.0,
            "category": "orphan",
        }
    if ambiguous_time:
        events["updated_at"] = events["created_at"] + pd.Timedelta(days=1)
    if missing_label:
        users = users.drop(columns=["future_spend"])

    users_path = data_dir / f"users.{table_format}"
    events_path = data_dir / f"events.{table_format}"
    if table_format == "csv":
        users.to_csv(users_path, index=False)
        events.to_csv(events_path, index=False)
    else:
        users.to_parquet(users_path, index=False)
        events.to_parquet(events_path, index=False)

    event_time = None if ambiguous_time else "created_at"
    config = {
        "dataset": "example-commerce",
        "tables": {
            "users": {
                "path": str(users_path.relative_to(tmp_path)),
                "primary_key": "user_id",
            },
            "events": {
                "path": str(events_path.relative_to(tmp_path)),
                "primary_key": "event_id",
                "foreign_keys": [{
                    "column": "user_id",
                    "references": {
                        "table": "users",
                        "column": "user_id",
                    },
                }],
            },
        },
        "task": {
            "task_id": "user-spend",
            "target_table": "users",
            "entity_key": "user_id",
            "target_time_col": "timestamp",
            "label_col": "future_spend",
            "problem_type": "regression",
            "primary_metric": "rmse",
            "metric_direction": "lower",
        },
        "split": {
            "strategy": "temporal",
            "train_end": "2026-01-31",
            "validation_end": (
                "2026-01-15" if invalid_split else "2026-02-28"
            ),
        },
    }
    if event_time is not None:
        config["tables"]["events"]["event_time_col"] = event_time
    config_path = tmp_path / "example-commerce.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
