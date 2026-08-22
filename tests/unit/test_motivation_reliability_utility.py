from __future__ import annotations

import pandas as pd

from fdhg.compiler.edge_reliability import compute_edge_reliability
from fdhg.compiler.fold_safe_fdhg import column_eligibility_audit
from fdhg.onboarding.motivation_reliability_utility import (
    MotivationOptions,
    aggregate_fold_rows,
    discover_earliest_fold_candidate_edges,
    train_source_view,
)
from tests.unit.test_auto_fdhg import FakeRelBenchTable


def _prepared_for_discovery(results: pd.DataFrame) -> dict:
    train = pd.DataFrame({
        "driver_id": ["d1", "d2", "d1"],
        "timestamp": pd.to_datetime(["2020-01-10", "2020-01-20", "2020-01-30"]),
        "position": [1.0, 1.0, 1.0],
    })
    return {
        "metadata": {
            "entity_key": "driver_id",
            "target_time_col": "timestamp",
            "label_col": "position",
            "primary_metric": "rmse",
            "metric_direction": "lower",
        },
        "train_df": train,
        "table_dict": {
            "results": FakeRelBenchTable(
                results,
                pkey_col="result_id",
                fkeys={"driver_id": "drivers"},
                time_col="race_date",
            )
        },
        "accepted_relations": [{
            "status": "accepted",
            "child_table": "results",
            "child_fk": "driver_id",
        }],
        "relations": [],
        "split_plan": {
            "folds": [
                {"fold": 0, "train_indices": [0], "validation_indices": [1]},
                {"fold": 1, "train_indices": [0, 1], "validation_indices": [2]},
            ]
        },
    }


def _stack_like_prepared() -> dict:
    train = pd.DataFrame({
        "UserId": ["u1", "u2", "u1"],
        "timestamp": pd.to_datetime(["2020-01-10", "2020-01-20", "2020-01-30"]),
        "label": [1, 0, 1],
    })
    badges = pd.DataFrame({
        "Id": [1, 2, 3, 4, 5],
        "UserId": ["u1", "u1", "u1", "u1", "u2"],
        "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-01"]),
        "Class": [1, 1, 2, 2, 1],
        "TagBased": [True, False, True, True, False],
        "Name": ["Nice Answer", "Scholar", "Nice Answer", "Teacher", "Visitor"],
    })
    post_history = pd.DataFrame({
        "Id": [10, 11, 12, 13, 14],
        "UserId": ["u1", "u1", "u1", "u1", "u2"],
        "CreationDate": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-01"]),
        "PostHistoryTypeId": [1, 1, 2, 2, 1],
        "ContentLicense": ["CC BY-SA", "MIT", "CC BY-SA", "CC BY-SA", "MIT"],
    })
    posts = pd.DataFrame({
        "Id": [20, 21, 22, 23, 24],
        "OwnerUserId": ["u1", "u1", "u1", "u1", "u2"],
        "CreationDate": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-01"]),
        "PostTypeId": [1, 1, 2, 2, 1],
        "ContentLicense": ["CC BY-SA", "MIT", "CC BY-SA", "CC BY-SA", "MIT"],
        "Body": [
            " ".join(["long"] * 40),
            " ".join(["text"] * 40),
            " ".join(["body"] * 40),
            " ".join(["words"] * 40),
            " ".join(["future"] * 40),
        ],
        "ExternalGuid": [
            "550e8400-e29b-41d4-a716-446655440000",
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400-e29b-41d4-a716-446655440002",
            "550e8400-e29b-41d4-a716-446655440003",
            "550e8400-e29b-41d4-a716-446655440004",
        ],
    })
    return {
        "metadata": {
            "entity_key": "UserId",
            "target_time_col": "timestamp",
            "label_col": "label",
            "primary_metric": "accuracy",
            "metric_direction": "higher",
        },
        "manifest": {"dataset": "rel-stack", "task": "user-badge"},
        "train_df": train,
        "table_dict": {
            "badges": FakeRelBenchTable(
                badges,
                pkey_col="Id",
                fkeys={"UserId": "users"},
                time_col="Date",
            ),
            "postHistory": FakeRelBenchTable(
                post_history,
                pkey_col="Id",
                fkeys={"UserId": "users"},
                time_col="CreationDate",
            ),
            "posts": FakeRelBenchTable(
                posts,
                pkey_col="Id",
                fkeys={"OwnerUserId": "users"},
                time_col="CreationDate",
            ),
        },
        "accepted_relations": [
            {"status": "accepted", "child_table": "badges", "child_fk": "UserId", "parent_table": "users", "parent_key": "UserId"},
            {"status": "accepted", "child_table": "postHistory", "child_fk": "UserId", "parent_table": "users", "parent_key": "UserId"},
            {"status": "accepted", "child_table": "posts", "child_fk": "OwnerUserId", "parent_table": "users", "parent_key": "UserId"},
        ],
        "relations": [],
        "split_plan": {
            "folds": [
                {"fold": 0, "train_indices": [0], "validation_indices": [1]},
                {"fold": 1, "train_indices": [0, 1], "validation_indices": [2]},
            ]
        },
    }


def test_candidate_discovery_uses_only_earliest_inner_train_snapshot() -> None:
    base = pd.DataFrame({
        "result_id": ["a", "b", "c", "v", "t", "future"],
        "driver_id": ["d1", "d1", "d1", "d2", "d3", "d1"],
        "race_date": pd.to_datetime([
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-01",
            "2020-01-01",
            "2020-01-25",
        ]),
        "x": ["p", "p", "q", "noise", "noise", "future_x"],
        "y": ["u", "v", "v", "changed_a", "changed_b", "future_y"],
    })
    changed = base.copy()
    changed.loc[changed["driver_id"].isin(["d2", "d3"]), ["x", "y"]] = "wild"
    changed.loc[changed["result_id"].eq("future"), ["x", "y"]] = "future_wild"
    first = discover_earliest_fold_candidate_edges(
        prepared=_prepared_for_discovery(base),
        edge_budget=4,
    )
    second = discover_earliest_fold_candidate_edges(
        prepared=_prepared_for_discovery(changed),
        edge_budget=4,
    )
    assert first["accepted_edges"]
    assert first["accepted_edges"] == second["accepted_edges"]
    assert [edge["edge_rank"] for edge in first["accepted_edges"]] == [edge["edge_rank"] for edge in second["accepted_edges"]]
    assert first["provenance"]["candidate_discovery_protocol"] == "fixed_from_earliest_inner_train_fold"
    assert first["provenance"]["candidate_discovery_fold"] == 0
    assert first["provenance"]["candidate_discovery_target_row_count"] == 1
    assert first["provenance"]["inner_validation_rows_used_for_candidate_discovery"] == 0
    assert first["provenance"]["official_validation_rows_used_for_candidate_discovery"] == 0
    assert first["provenance"]["test_rows_used_for_candidate_discovery"] == 0


def test_candidate_column_eligibility_is_schema_key_aware_and_categorical() -> None:
    prepared = _stack_like_prepared()
    discovery = discover_earliest_fold_candidate_edges(prepared=prepared, edge_budget=20)
    audit = {
        (row["source_table"], row["column"]): row
        for row in discovery["provenance"]["candidate_column_audit"]
    }

    assert audit[("badges", "Id")]["actual_primary_key"] is True
    assert audit[("badges", "Id")]["dependent_eligible"] is False
    assert audit[("badges", "UserId")]["actual_foreign_key"] is True
    assert audit[("badges", "UserId")]["dependent_eligible"] is False
    assert audit[("posts", "OwnerUserId")]["source_entity_column"] is True
    assert audit[("posts", "OwnerUserId")]["dependent_eligible"] is False

    assert audit[("posts", "PostTypeId")]["actual_primary_key"] is False
    assert audit[("posts", "PostTypeId")]["actual_foreign_key"] is False
    assert audit[("posts", "PostTypeId")]["determinant_eligible"] is True
    assert audit[("posts", "PostTypeId")]["dependent_eligible"] is True
    assert audit[("badges", "Class")]["dependent_eligible"] is True
    assert audit[("badges", "TagBased")]["dependent_eligible"] is True
    assert audit[("badges", "Name")]["dependent_eligible"] is True
    assert audit[("postHistory", "PostHistoryTypeId")]["dependent_eligible"] is True
    assert audit[("postHistory", "ContentLicense")]["dependent_eligible"] is True
    assert audit[("posts", "ContentLicense")]["dependent_eligible"] is True
    assert audit[("posts", "Body")]["dependent_eligible"] is False
    assert audit[("posts", "Body")]["exclusion_reason"].startswith("free_text")
    assert audit[("posts", "ExternalGuid")]["dependent_eligible"] is False
    assert audit[("posts", "ExternalGuid")]["exclusion_reason"] == "guid_like_excluded"


def test_stack_like_candidate_pairs_are_discovered_and_ordered_deterministically() -> None:
    prepared = _stack_like_prepared()
    first = discover_earliest_fold_candidate_edges(prepared=prepared, edge_budget=20)
    second = discover_earliest_fold_candidate_edges(prepared=prepared, edge_budget=20)
    first_pairs = [
        (edge["source_table"], edge["lhs_columns"][0], edge["rhs_column"])
        for edge in first["accepted_edges"]
    ]
    second_pairs = [
        (edge["source_table"], edge["lhs_columns"][0], edge["rhs_column"])
        for edge in second["accepted_edges"]
    ]

    assert first_pairs == second_pairs
    assert {
        ("badges", "Class", "TagBased"),
        ("badges", "TagBased", "Class"),
        ("postHistory", "PostHistoryTypeId", "ContentLicense"),
        ("postHistory", "ContentLicense", "PostHistoryTypeId"),
        ("posts", "PostTypeId", "ContentLicense"),
        ("posts", "ContentLicense", "PostTypeId"),
    }.issubset(set(first_pairs))
    assert first["provenance"]["candidate_pair_count_before_edge_validation"] >= 6
    assert first["provenance"]["accepted_candidate_edge_count"] > 0


def test_column_eligibility_helpers_do_not_reject_low_cardinality_id_name() -> None:
    table = FakeRelBenchTable(pd.DataFrame(), pkey_col="Id", fkeys={"UserId": "users"})
    audit = column_eligibility_audit(
        "PostHistoryTypeId",
        pd.Series([1, 1, 2, 2, 3], name="PostHistoryTypeId"),
        metadata={"entity_key": "UserId", "label_col": "label"},
        table=table,
        source_entity_column="UserId",
    )
    assert audit["dependent_eligible"] is True
    assert audit["determinant_eligible"] is True


def test_reliability_ignores_validation_and_test_dependent_values() -> None:
    train_targets = pd.DataFrame({
        "driver_id": ["d1"],
        "timestamp": pd.to_datetime(["2020-01-10"]),
    })
    validation_targets = pd.DataFrame({
        "driver_id": ["d2", "d3"],
        "timestamp": pd.to_datetime(["2020-01-10", "2020-01-10"]),
    })
    source = pd.DataFrame({
        "result_id": ["a", "b", "v", "t"],
        "driver_id": ["d1", "d1", "d2", "d3"],
        "race_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-01"]),
        "x": ["p", "p", "p", "p"],
        "y": ["u", "u", "bad", "bad"],
    })
    changed = source.copy()
    changed.loc[changed["driver_id"].isin(["d2", "d3"]), "y"] = ["worse", "worst"]
    metadata = {"entity_key": "driver_id", "target_time_col": "timestamp"}
    stats = []
    for frame in (source, changed):
        view = train_source_view(
            table=FakeRelBenchTable(frame, time_col="race_date"),
            metadata=metadata,
            train_targets=train_targets,
            validation_targets=validation_targets,
            relation={"child_table": "results", "child_fk": "driver_id", "parent_table": "drivers", "parent_key": "driver_id"},
        )
        stats.append(compute_edge_reliability(view["fit_rows"], lhs_columns=["x"], rhs_column="y"))
        assert view["audit"]["validation_target_entity_overlap_in_reliability_rows"] == 0
        assert view["audit"]["official_validation_row_usage_count"] == 0
        assert view["audit"]["test_row_usage_count"] == 0
    assert stats[0]["reliability_raw"] == stats[1]["reliability_raw"]
    assert stats[0]["reliability_loo"] == stats[1]["reliability_loo"]


def test_source_child_fk_can_differ_from_target_entity_key() -> None:
    view = train_source_view(
        table=FakeRelBenchTable(
            pd.DataFrame({
                "user_ref": ["u1", "u1", "u2"],
                "event_time": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-01"]),
                "x": ["a", "b", "c"],
                "y": ["m", "n", "o"],
            }),
            fkeys={"user_ref": "users"},
            time_col="event_time",
        ),
        metadata={"entity_key": "user_id", "target_time_col": "timestamp"},
        train_targets=pd.DataFrame({"user_id": ["u1"], "timestamp": pd.to_datetime(["2020-01-02"])}),
        validation_targets=pd.DataFrame({"user_id": ["u2"], "timestamp": pd.to_datetime(["2020-01-02"])}),
        relation={"child_table": "events", "child_fk": "user_ref", "parent_table": "users", "parent_key": "user_id"},
    )
    assert view["blocked_reason"] == ""
    assert view["fit_rows"]["user_ref"].tolist() == ["u1"]
    assert view["fit_rows"]["user_id"].tolist() == ["u1"]
    assert view["audit"]["source_entity_column"] == "user_ref"
    assert view["audit"]["source_entity_column_resolution"] == "accepted_relation_child_fk"


def test_static_table_excludes_validation_only_entities_and_remains_stable() -> None:
    train = pd.DataFrame({"driver_id": ["d1"], "timestamp": pd.to_datetime(["2020-01-01"])})
    validation = pd.DataFrame({"driver_id": ["d2"], "timestamp": pd.to_datetime(["2020-01-02"])})
    source = pd.DataFrame({
        "driver_id": ["d1", "d2", "d3"],
        "x": ["a", "a", "a"],
        "y": ["m", "bad", "bad"],
    })
    changed = source.copy()
    changed.loc[changed["driver_id"].isin(["d2", "d3"]), "y"] = ["worse", "worst"]
    stats = []
    for frame in (source, changed):
        view = train_source_view(
            table=FakeRelBenchTable(frame, fkeys={"driver_id": "drivers"}),
            metadata={"entity_key": "driver_id", "target_time_col": "timestamp"},
            train_targets=train,
            validation_targets=validation,
            relation={"child_table": "profiles", "child_fk": "driver_id", "parent_table": "drivers", "parent_key": "driver_id"},
        )
        assert view["blocked_reason"] == ""
        assert view["audit"]["reliability_fit_scope"] == "fold_train_static_entity_snapshot"
        assert view["fit_rows"]["driver_id"].tolist() == ["d1"]
        assert view["audit"]["validation_target_entity_overlap_in_reliability_rows"] == 0
        stats.append(compute_edge_reliability(view["fit_rows"], lhs_columns=["x"], rhs_column="y"))
    assert stats[0]["reliability_raw"] == stats[1]["reliability_raw"]


def test_unresolvable_entity_linkage_is_blocked() -> None:
    view = train_source_view(
        table=FakeRelBenchTable(pd.DataFrame({"other_id": ["d1"], "x": ["a"], "y": ["b"]})),
        metadata={"entity_key": "driver_id", "target_time_col": "timestamp"},
        train_targets=pd.DataFrame({"driver_id": ["d1"], "timestamp": pd.to_datetime(["2020-01-01"])}),
        validation_targets=pd.DataFrame({"driver_id": ["d2"], "timestamp": pd.to_datetime(["2020-01-02"])}),
    )
    assert view["blocked_reason"] == "blocked_unresolvable_entity_linkage"
    assert view["audit"]["reliability_fit_scope"] == "blocked_unresolvable_entity_linkage"


def test_screening_threshold_and_fold_count_are_configurable() -> None:
    rows = []
    for fold, delta in enumerate([0.05, 0.20, 0.30, -0.01]):
        rows.append({
            "dataset": "d",
            "task": "t",
            "seed": 7,
            "edge_id": "e",
            "edge_rank": 1,
            "determinant": "x",
            "dependent": "y",
            "source_table": "s",
            "relational_path": "s:x->y",
            "fold": fold,
            "edge_status": "ok",
            "delta": delta,
            "reliability_loo": 0.5,
            "reliability_raw": 0.5,
            "reliability_entropy": 0.5,
            "non_singleton_coverage": 1.0,
        })
    agg = aggregate_fold_rows(
        rows,
        options=MotivationOptions(
            selection_folds=4,
            edge_screening_min_delta=0.1,
            edge_screening_min_positive_folds=2,
        ),
    )[0]
    assert agg["positive_fold_count"] == 3
    assert agg["passing_fold_count"] == 2
    assert agg["screening_min_delta"] == 0.1
    assert agg["screening_min_positive_folds"] == 2
    assert agg["selected_by_existing_screening_rule"] is True
    assert agg["selected_by_two_of_three_rule"] == ""


def test_screening_threshold_boundaries_are_strict() -> None:
    def selected(delta: float) -> bool:
        rows = [{
            "dataset": "d",
            "task": "t",
            "seed": 7,
            "edge_id": "e",
            "edge_status": "ok",
            "delta": delta,
            "reliability_loo": 0.5,
            "reliability_raw": 0.5,
            "reliability_entropy": 0.5,
            "non_singleton_coverage": 1.0,
        }]
        return aggregate_fold_rows(
            rows,
            options=MotivationOptions(
                selection_folds=1,
                edge_screening_min_delta=0.1,
                edge_screening_min_positive_folds=1,
            ),
        )[0]["selected_by_existing_screening_rule"]

    assert selected(0.1) is False
    assert selected(0.1000000001) is True
    assert selected(0.0999999999) is False
