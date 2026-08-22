from __future__ import annotations

import math

import pandas as pd
import pytest

from fdhg.compiler.edge_reliability import compute_edge_reliability


def test_prompt_zip_tier_example() -> None:
    df = pd.DataFrame({
        "zip": ["06236", "06236", "06236", "06236", "07011"],
        "tier": ["gold", "gold", "silver", "silver", "bronze"],
    })
    stats = compute_edge_reliability(df, lhs_columns=["zip"], rhs_column="tier")
    assert stats["reliability_raw"] == pytest.approx(0.6)
    assert stats["non_singleton_coverage"] == pytest.approx(0.8)
    assert stats["reliability_non_singleton"] == pytest.approx(0.5)


def test_perfectly_deterministic_repeated_dependency() -> None:
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], "y": ["u", "u", "v", "v"]})
    stats = compute_edge_reliability(df, lhs_columns=["x"], rhs_column="y")
    assert stats["reliability_raw"] == pytest.approx(1.0)
    assert stats["reliability_non_singleton"] == pytest.approx(1.0)
    assert stats["reliability_loo"] == pytest.approx(1.0)
    assert stats["reliability_entropy"] == pytest.approx(1.0)


def test_fully_ambiguous_balanced_dependency() -> None:
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], "y": ["u", "v", "u", "v"]})
    stats = compute_edge_reliability(df, lhs_columns=["x"], rhs_column="y")
    assert stats["reliability_raw"] == pytest.approx(0.5)
    assert stats["reliability_non_singleton"] == pytest.approx(0.5)
    assert stats["conditional_entropy_normalized"] == pytest.approx(1.0)
    assert stats["reliability_entropy"] == pytest.approx(0.0)


def test_all_singleton_determinant_groups() -> None:
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": ["u", "v", "w"]})
    stats = compute_edge_reliability(df, lhs_columns=["x"], rhs_column="y")
    assert stats["reliability_raw"] == pytest.approx(1.0)
    assert stats["non_singleton_coverage"] == pytest.approx(0.0)
    assert math.isnan(stats["reliability_non_singleton"])
    assert math.isnan(stats["reliability_loo"])


def test_leave_one_out_can_differ_from_insample_mode() -> None:
    df = pd.DataFrame({"x": ["a", "a", "a", "a"], "y": ["u", "u", "v", "v"]})
    stats = compute_edge_reliability(df, lhs_columns=["x"], rhs_column="y")
    assert stats["reliability_raw"] == pytest.approx(0.5)
    assert stats["reliability_loo"] == pytest.approx(0.0)


def test_missing_values_are_excluded_like_fdhg_mapping() -> None:
    df = pd.DataFrame({
        "x": ["a", "a", None, "b", "b"],
        "y": ["u", None, "v", "w", "w"],
    })
    stats = compute_edge_reliability(df, lhs_columns=["x"], rhs_column="y")
    assert stats["total_support"] == 3
    assert stats["determinant_group_count"] == 2
    assert stats["reliability_raw"] == pytest.approx(1.0)


def test_tie_handling_is_deterministic_across_repeated_runs() -> None:
    df = pd.DataFrame({"x": ["a", "a", "a", "a"], "y": ["u", "u", "v", "v"]})
    values = [
        compute_edge_reliability(df, lhs_columns=["x"], rhs_column="y")["reliability_loo"]
        for _ in range(5)
    ]
    assert values == [values[0]] * 5
