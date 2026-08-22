from __future__ import annotations

from pathlib import Path
import json
import pickle

import pandas as pd
import pytest


COMPILER_DIR = Path("outputs/feature_program_compiler/rel-amazon_item-churn_sample")
FEATURE_TABLE = Path("outputs/ambiguity_features/rel-amazon_item-churn_sample/target_with_dfs_agg_amb_train.parquet")
BASE_X_TRAIN = Path("artifacts/target_only/rel-amazon/item-churn/X_train.pkl")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
    not COMPILER_DIR.exists(),
    reason=(
        "Integration artifact is unavailable. "
        "Run the feature-program compiler first."
    ),
),
]



def test_feature_program_manifest_exists() -> None:
    assert (COMPILER_DIR / "feature_program_manifest.csv").exists()
    assert (COMPILER_DIR / "bfs_edge_trace.csv").exists()
    assert (COMPILER_DIR / "feature_program_compiler_summary.json").exists()


def test_feature_program_blocks_present() -> None:
    manifest = pd.read_csv(COMPILER_DIR / "feature_program_manifest.csv")

    blocks = set(manifest["block"])
    assert "pullup" in blocks
    assert "aggregation" in blocks
    assert "temporal_aggregation" in blocks
    assert "ambiguity" in blocks


def test_bfs_edge_types_present() -> None:
    edges = pd.read_csv(COMPILER_DIR / "bfs_edge_trace.csv")
    edge_types = set(edges["edge_type"])

    assert "fk_ind" in edge_types
    assert "inverse_fk" in edge_types
    assert "afd" in edge_types


def test_all_materialized_features_exist() -> None:
    manifest = pd.read_csv(COMPILER_DIR / "feature_program_manifest.csv")
    feature_table = pd.read_parquet(FEATURE_TABLE)

    with open(BASE_X_TRAIN, "rb") as f:
        base_x_train = pickle.load(f)

    # Final logical Z consists of:
    # target-only base features + generated DFS/ambiguity features.
    feature_cols = set(feature_table.columns).union(set(base_x_train.columns))

    missing = [
        f for f in manifest["materialized_feature"]
        if f not in feature_cols
    ]

    assert not missing, f"Missing materialized features: {missing}"


def test_summary_says_compiler_complete() -> None:
    with open(COMPILER_DIR / "feature_program_compiler_summary.json") as f:
        summary = json.load(f)

    assert summary["all_materialized_features_exist"] is True
    assert summary["n_pullup_programs"] > 0
    assert summary["n_aggregation_programs"] > 0
    assert summary["n_ambiguity_programs"] > 0


if __name__ == "__main__":
    test_feature_program_manifest_exists()
    test_feature_program_blocks_present()
    test_bfs_edge_types_present()
    test_all_materialized_features_exist()
    test_summary_says_compiler_complete()
    print("OK: feature program compiler MVP tests passed.")
