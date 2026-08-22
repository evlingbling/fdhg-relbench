from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


DFS_DIR = Path("outputs/dfs_agg/rel-amazon_item-churn_sample")
AMB_DIR = Path("outputs/ambiguity_features/rel-amazon_item-churn_sample")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
    not DFS_DIR.exists() or not AMB_DIR.exists(),
    reason=(
        "Integration artifacts are unavailable. "
        "Run the DFS and ambiguity builders first."
    ),
),
]


def assert_file(path: Path) -> None:
    assert path.exists(), f"Missing file: {path}"


def test_dfs_row_counts_and_alignment() -> None:
    expected_rows = {
        "train": 10000,
        "val": 2000,
        "test": 2000,
    }

    feature_cols_ref = None

    for split, n_expected in expected_rows.items():
        path = DFS_DIR / f"target_with_dfs_agg_{split}.parquet"
        assert_file(path)

        df = pd.read_parquet(path)
        assert len(df) == n_expected, f"{split}: expected {n_expected}, got {len(df)}"

        feature_cols = [c for c in df.columns if c.startswith("f_")]
        assert feature_cols, f"{split}: no generated feature columns found"

        if feature_cols_ref is None:
            feature_cols_ref = feature_cols
        else:
            assert feature_cols == feature_cols_ref, f"{split}: feature columns are not aligned"


def test_temporal_review_aggregation_is_cutoff_safe() -> None:
    for split in ["train", "val", "test"]:
        path = DFS_DIR / f"target_with_dfs_agg_{split}.parquet"
        assert_file(path)

        df = pd.read_parquet(path)

        col = "f_review_days_since_last"
        assert col in df.columns, f"{split}: missing {col}"

        non_missing = df[col].dropna()

        # days_since_last = target.timestamp - last_review_time.
        # If future reviews leaked in, this could become negative.
        assert (non_missing >= 0).all(), (
            f"{split}: found negative days_since_last; possible future leakage"
        )


def test_temporal_manifest_marks_features_safe() -> None:
    manifest_path = DFS_DIR / "dfs_agg_feature_manifest.csv"
    assert_file(manifest_path)

    manifest = pd.read_csv(manifest_path)
    assert "temporal_safe" in manifest.columns
    assert manifest["temporal_safe"].all(), (
        "Not all DFS aggregation features are marked temporal_safe"
    )


def test_ambiguity_features_alignment() -> None:
    expected_rows = {
        "train": 10000,
        "val": 2000,
        "test": 2000,
    }

    amb_cols_ref = None

    for split, n_expected in expected_rows.items():
        path = AMB_DIR / f"target_with_dfs_agg_amb_{split}.parquet"
        assert_file(path)

        df = pd.read_parquet(path)
        assert len(df) == n_expected, f"{split}: expected {n_expected}, got {len(df)}"

        amb_cols = [c for c in df.columns if c.startswith("f_amb__")]
        assert amb_cols, f"{split}: no ambiguity features found"

        if amb_cols_ref is None:
            amb_cols_ref = amb_cols
        else:
            assert amb_cols == amb_cols_ref, (
                f"{split}: ambiguity feature columns are not aligned"
            )


if __name__ == "__main__":
    test_dfs_row_counts_and_alignment()
    test_temporal_review_aggregation_is_cutoff_safe()
    test_temporal_manifest_marks_features_safe()
    test_ambiguity_features_alignment()
    print("OK: leakage guard MVP tests passed.")
