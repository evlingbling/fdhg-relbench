from __future__ import annotations

import pandas as pd
import pytest

from fdhg.compiler.edge_reliability import compute_edge_reliability
from fdhg.onboarding.motivation_fd_violation import (
    _assert_leakage_counters_zero,
    aggregate_fd_violation_rows,
    corrupt_dependent_by_fd_aware_swaps,
    corrupt_dependent_by_permutation,
    edge_suitability_audit,
    parse_edge_spec,
    resolve_requested_edge,
)
from fdhg.onboarding.motivation_reliability_utility import (
    fold_train_source_view_for_edge,
)
from tests.unit.test_auto_fdhg import FakeRelBenchTable


def _fd_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "entity_id": ["e1", "e1", "e1", "e1", "e2", "e2", "e2", "e2"],
        "event_time": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
        ]),
        "x": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "y": ["red", "red", "red", "red", "blue", "blue", "blue", "blue"],
    })


def test_corruption_rate_zero_leaves_dependent_unchanged() -> None:
    rows = _fd_rows()
    corrupted, audit = corrupt_dependent_by_fd_aware_swaps(
        rows,
        lhs_columns=("x",),
        rhs_column="y",
        rate=0.0,
        seed=41,
    )

    assert corrupted["y"].equals(rows["y"])
    assert audit["effective_changed_row_rate"] == 0.0
    assert audit["requested_sampled_row_count"] == 0
    assert audit["actual_changed_row_count"] == 0
    assert audit["dependent_marginal_counts_preserved"] is True


def test_corruption_preserves_marginal_counts_and_fixed_columns() -> None:
    rows = _fd_rows()
    corrupted, audit = corrupt_dependent_by_fd_aware_swaps(
        rows,
        lhs_columns=("x",),
        rhs_column="y",
        rate=1.0,
        seed=41,
    )

    assert corrupted["y"].value_counts().sort_index().equals(rows["y"].value_counts().sort_index())
    assert audit["dependent_marginal_counts_preserved"] is True
    assert audit["unchanged_assignment_count"] == 0
    assert audit["actual_changed_row_count"] == audit["maximum_feasible_changed_row_count"]
    assert len(corrupted) == len(rows)
    for col in ["entity_id", "event_time", "x"]:
        assert corrupted[col].equals(rows[col])


def test_corruption_is_deterministic_and_seeded() -> None:
    rows = _fd_rows()
    first, _ = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=0.5, seed=41)
    second, _ = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=0.5, seed=41)
    third, _ = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=0.5, seed=42)

    assert first["y"].equals(second["y"])
    assert third["y"].value_counts().sort_index().equals(rows["y"].value_counts().sort_index())


def test_corruption_does_not_require_labels() -> None:
    rows = _fd_rows()[["entity_id", "event_time", "x", "y"]]
    corrupted, _ = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=0.25, seed=41)

    assert "label" not in corrupted.columns


def test_reliability_decreases_and_entropy_or_violation_increases() -> None:
    rows = _fd_rows()
    low, _ = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=0.0, seed=41)
    high, _ = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=1.0, seed=41)
    low_stats = compute_edge_reliability(low, lhs_columns=["x"], rhs_column="y")
    high_stats = compute_edge_reliability(high, lhs_columns=["x"], rhs_column="y")

    assert low_stats["reliability_loo"] == pytest.approx(1.0)
    assert high_stats["reliability_loo"] < low_stats["reliability_loo"]
    assert (
        high_stats["conditional_entropy_normalized"] > low_stats["conditional_entropy_normalized"]
        or 1.0 - high_stats["reliability_raw"] > 1.0 - low_stats["reliability_raw"]
    )


def test_fd_aware_swaps_handle_highly_imbalanced_binary_dependents() -> None:
    rows = pd.DataFrame({
        "x": ["a"] * 10 + ["b"] * 10,
        "y": [True] * 9 + [False] + [False] * 9 + [True],
        "fixed": list(range(20)),
    })
    corrupted, audit = corrupt_dependent_by_fd_aware_swaps(
        rows,
        lhs_columns=("x",),
        rhs_column="y",
        rate=1.0,
        seed=7,
    )
    assert audit["actual_changed_row_count"] == 8
    assert audit["effective_changed_rate_among_eligible"] == pytest.approx(0.4)
    assert audit["achieved_violation_rate"] == pytest.approx(0.5)
    assert corrupted["y"].value_counts().sort_index().equals(rows["y"].value_counts().sort_index())
    assert corrupted["fixed"].equals(rows["fixed"])


def test_fd_aware_swaps_handle_multiclass_dependents() -> None:
    rows = pd.DataFrame({
        "x": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        "y": ["red"] * 4 + ["blue"] * 4 + ["green"] * 4,
        "fixed": range(12),
    })
    low, low_audit = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=0.25, seed=11)
    high, high_audit = corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=1.0, seed=11)
    assert high_audit["actual_changed_row_count"] >= low_audit["actual_changed_row_count"]
    assert high_audit["induced_fd_violation_row_count"] >= low_audit["induced_fd_violation_row_count"]
    assert high["y"].value_counts().sort_index().equals(rows["y"].value_counts().sort_index())
    assert low["fixed"].equals(rows["fixed"])
    assert high["fixed"].equals(rows["fixed"])


def test_fd_aware_changed_counts_and_disagreement_are_monotonic() -> None:
    rows = _fd_rows()
    audits = [
        corrupt_dependent_by_fd_aware_swaps(rows, lhs_columns=("x",), rhs_column="y", rate=rate, seed=41)[1]
        for rate in (0.0, 0.25, 0.5, 1.0)
    ]
    changed = [audit["actual_changed_row_count"] for audit in audits]
    induced = [audit["induced_fd_violation_row_count"] for audit in audits]
    assert changed == sorted(changed)
    assert induced == sorted(induced)


def test_edge_suitability_audit_reports_feasibility_and_baseline() -> None:
    rows = pd.DataFrame({
        "x": ["a"] * 10 + ["b"] * 10,
        "y": [True] * 9 + [False] + [False] * 9 + [True],
    })
    audit = edge_suitability_audit(rows, lhs_columns=("x",), rhs_column="y")
    assert audit["dependent_cardinality"] == 2
    assert audit["dependent_min_class_frequency"] == 10
    assert audit["dependent_max_class_frequency"] == 10
    assert audit["dependent_minority_fraction"] == pytest.approx(0.5)
    assert audit["maximum_feasible_marginal_preserving_changed_row_rate"] == pytest.approx(0.4)
    assert audit["determinant_group_support"] == 2
    assert audit["baseline_violation_rate"] == pytest.approx(0.1)


def test_validation_and_future_source_rows_do_not_affect_fitted_reliability() -> None:
    train_targets = pd.DataFrame({
        "entity_id": ["e1"],
        "timestamp": pd.to_datetime(["2020-01-10"]),
        "label": [1],
    })
    validation_targets = pd.DataFrame({
        "entity_id": ["e2"],
        "timestamp": pd.to_datetime(["2020-01-10"]),
        "label": [0],
    })
    source = pd.DataFrame({
        "entity_id": ["e1", "e1", "e2"],
        "event_time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"]),
        "x": ["a", "future", "validation"],
        "y": ["red", "future_value", "validation_value"],
    })
    edge = {
        "edge_id": "events:x->y",
        "source_table": "events",
        "lhs_columns": ("x",),
        "rhs_column": "y",
        "source_entity_column": "entity_id",
    }
    view = fold_train_source_view_for_edge(
        table_dict={"events": FakeRelBenchTable(source, time_col="event_time")},
        metadata={"entity_key": "entity_id", "target_time_col": "timestamp", "label_col": "label"},
        train_targets=train_targets,
        validation_targets=validation_targets,
        edge=edge,
    )

    assert view["audit"]["future_row_violation_count"] == 0
    assert view["audit"]["validation_target_entity_overlap_in_reliability_rows"] == 0
    assert view["fit_rows"]["x"].tolist() == ["a"]
    stats = compute_edge_reliability(view["fit_rows"], lhs_columns=["x"], rhs_column="y")
    changed_source = source.copy()
    changed_source.loc[changed_source["entity_id"].eq("e2"), "y"] = "wild"
    changed_view = fold_train_source_view_for_edge(
        table_dict={"events": FakeRelBenchTable(changed_source, time_col="event_time")},
        metadata={"entity_key": "entity_id", "target_time_col": "timestamp", "label_col": "label"},
        train_targets=train_targets,
        validation_targets=validation_targets,
        edge=edge,
    )
    changed_stats = compute_edge_reliability(changed_view["fit_rows"], lhs_columns=["x"], rhs_column="y")
    assert stats["reliability_raw"] == changed_stats["reliability_raw"]


def test_parse_and_resolve_requested_edge_without_candidate_policy() -> None:
    table, lhs, rhs = parse_edge_spec("badges:Class->TagBased")
    assert table == "badges"
    assert lhs == ("Class",)
    assert rhs == "TagBased"
    prepared = {
        "table_dict": {
            "badges": FakeRelBenchTable(
                pd.DataFrame({"UserId": ["u1"], "Class": [1], "TagBased": [True]}),
                fkeys={"UserId": "users"},
            )
        },
        "accepted_relations": [
            {"status": "accepted", "child_table": "badges", "child_fk": "UserId", "parent_table": "users", "parent_key": "UserId"}
        ],
    }
    edge = resolve_requested_edge("badges:Class->TagBased", prepared)

    assert edge is not None
    assert edge["edge_id"] == "badges:Class->TagBased"
    assert edge["source_entity_column"] == "UserId"


def test_aggregate_keeps_edge_and_corruption_identity() -> None:
    rows = []
    for fold in range(3):
        for rate, reliability in [(0.0, 1.0), (1.0, 0.25)]:
            rows.append({
                "dataset": "rel-stack",
                "task": "user-badge",
                "edge_id": "badges:Class->TagBased",
                "source_table": "badges",
                "determinant": "Class",
                "dependent": "TagBased",
                "fold": fold,
                "corruption_seed": 41,
                "requested_corruption_rate": rate,
                "effective_changed_row_rate": rate,
                "effective_changed_rate_among_eligible": rate,
                "achieved_violation_rate": 1.0 - reliability,
                "induced_fd_violation_row_count": 0 if rate == 0.0 else 2,
                "reliability_loo": reliability,
                "reliability_raw": reliability,
                "reliability_entropy": reliability,
                "conditional_entropy_normalized": 1.0 - reliability,
                "violation_rate": 1.0 - reliability,
                "residual_nonzero_rate": 0.5,
                "residual_variance": 0.1,
                "delta_over_base": 0.0,
                "delta_relative_to_uncorrupted": 0.0,
                "edge_status": "ok",
            })
    aggregate = aggregate_fd_violation_rows(rows)

    assert [row["requested_corruption_rate"] for row in aggregate] == [0.0, 1.0]
    assert aggregate[0]["valid_fold_count"] == 3
    assert aggregate[1]["mean_reliability_loo"] < aggregate[0]["mean_reliability_loo"]


def test_leakage_counter_invariant_fails_closed() -> None:
    row = {
        "future_row_violation_count": 0,
        "inner_validation_row_usage_count": 0,
        "official_validation_row_usage_count": 0,
        "test_row_usage_count": 0,
    }
    _assert_leakage_counters_zero([row])
    with pytest.raises(AssertionError, match="test_row_usage_count"):
        _assert_leakage_counters_zero([{**row, "test_row_usage_count": 1}])
    # Exercise the public aggregate path with ok rows and finite counters.
    aggregate_row = {
        **row,
        "dataset": "d",
        "task": "t",
        "edge_id": "e:x->y",
        "source_table": "e",
        "determinant": "x",
        "dependent": "y",
        "requested_corruption_rate": 0.0,
        "effective_changed_row_rate": 0.0,
        "effective_changed_rate_among_eligible": 0.0,
        "achieved_violation_rate": 0.0,
        "induced_fd_violation_row_count": 0,
        "corruption_seed": 41,
        "reliability_loo": 1.0,
        "reliability_raw": 1.0,
        "reliability_entropy": 1.0,
        "conditional_entropy_normalized": 0.0,
        "violation_rate": 0.0,
        "residual_nonzero_rate": 0.0,
        "residual_variance": 0.0,
        "delta_over_base": 0.0,
        "delta_relative_to_uncorrupted": 0.0,
        "edge_status": "ok",
    }
    assert aggregate_fd_violation_rows([aggregate_row])
