from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "prepare"
    / "build_afd_fit_table.py"
)

if not SCRIPT_PATH.exists():
    raise FileNotFoundError(
        f"Could not find AFD fit-table builder: {SCRIPT_PATH}"
    )

spec = importlib.util.spec_from_file_location(
    "build_afd_fit_table",
    SCRIPT_PATH,
)
assert spec is not None
assert spec.loader is not None

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

build_train_only_fit_table = (
    module.build_train_only_fit_table
)


def test_filters_entities_not_present_in_train() -> None:
    entity_df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "country": ["KR", "US", "JP", "GB"],
        }
    )

    target_train = pd.DataFrame(
        {
            "user_id": [1, 1, 3],
            "label": [0, 1, 0],
        }
    )

    fit_df, manifest = build_train_only_fit_table(
        entity_df=entity_df,
        target_train=target_train,
        entity_key="user_id",
        target_entity_key="user_id",
        entity_time_col=None,
        target_time_col=None,
    )

    assert fit_df["user_id"].tolist() == [1, 3]
    assert manifest["fit_split"] == "train"
    assert manifest["fit_scope"] == "train_entities"
    assert manifest["n_source_rows"] == 4
    assert manifest["n_train_entities"] == 2
    assert manifest["n_fit_rows"] == 2
    assert manifest["n_fit_entities"] == 2
    assert manifest["temporal_cutoff_applied"] is False


def test_filters_rows_after_each_entity_train_cutoff() -> None:
    entity_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "event_time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-10",
                    "2024-02-01",
                    "2024-01-02",
                    "2024-03-01",
                ],
                utc=True,
            ),
            "state": [
                "a",
                "b",
                "future-user-1",
                "x",
                "future-user-2",
            ],
        }
    )

    target_train = pd.DataFrame(
        {
            "user_id": [1, 2],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-02-01",
                ],
                utc=True,
            ),
            "label": [0, 1],
        }
    )

    fit_df, manifest = build_train_only_fit_table(
        entity_df=entity_df,
        target_train=target_train,
        entity_key="user_id",
        target_entity_key="user_id",
        entity_time_col="event_time",
        target_time_col="timestamp",
    )

    assert fit_df["state"].tolist() == ["a", "b", "x"]
    assert manifest["fit_split"] == "train"
    assert (
        manifest["fit_scope"]
        == "train_entities_with_temporal_cutoff"
    )
    assert manifest["temporal_cutoff_applied"] is True
    assert manifest["n_fit_rows"] == 3
    assert manifest["n_fit_entities"] == 2


def test_uses_maximum_training_cutoff_per_entity() -> None:
    entity_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "event_time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-10",
                    "2024-01-20",
                ],
                utc=True,
            ),
            "state": ["a", "b", "c"],
        }
    )

    target_train = pd.DataFrame(
        {
            "user_id": [1, 1],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-15",
                ],
                utc=True,
            ),
        }
    )

    fit_df, _ = build_train_only_fit_table(
        entity_df=entity_df,
        target_train=target_train,
        entity_key="user_id",
        target_entity_key="user_id",
        entity_time_col="event_time",
        target_time_col="timestamp",
    )

    assert fit_df["state"].tolist() == ["a", "b"]


def test_rejects_incomplete_temporal_configuration() -> None:
    entity_df = pd.DataFrame(
        {
            "user_id": [1],
            "event_time": pd.to_datetime(
                ["2024-01-01"],
                utc=True,
            ),
        }
    )

    target_train = pd.DataFrame(
        {
            "user_id": [1],
            "timestamp": pd.to_datetime(
                ["2024-01-02"],
                utc=True,
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="must either both be provided",
    ):
        build_train_only_fit_table(
            entity_df=entity_df,
            target_train=target_train,
            entity_key="user_id",
            target_entity_key="user_id",
            entity_time_col="event_time",
            target_time_col=None,
        )


def test_rejects_missing_entity_key() -> None:
    entity_df = pd.DataFrame(
        {
            "wrong_key": [1, 2],
            "country": ["KR", "US"],
        }
    )

    target_train = pd.DataFrame(
        {
            "user_id": [1],
        }
    )

    with pytest.raises(
        KeyError,
        match="missing required columns",
    ):
        build_train_only_fit_table(
            entity_df=entity_df,
            target_train=target_train,
            entity_key="user_id",
            target_entity_key="user_id",
            entity_time_col=None,
            target_time_col=None,
        )


def test_rejects_empty_train_visible_fit_table() -> None:
    entity_df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "country": ["KR", "US"],
        }
    )

    target_train = pd.DataFrame(
        {
            "user_id": [10, 11],
        }
    )

    with pytest.raises(
        ValueError,
        match="fit table is empty",
    ):
        build_train_only_fit_table(
            entity_df=entity_df,
            target_train=target_train,
            entity_key="user_id",
            target_entity_key="user_id",
            entity_time_col=None,
            target_time_col=None,
        )
