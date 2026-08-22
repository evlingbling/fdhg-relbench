from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fdhg.compiler.ambiguity import (
    FittedAmbiguityEdge,
    fit_ambiguity_map,
    materialize_ambiguity_from_map,
    normalize_lhs_frame,
    normalize_value,
)
import fdhg.onboarding.auto_fdhg as auto_fdhg_module
from fdhg.cli.export_fdhg_candidate_edges import main as export_candidate_edges_main
from fdhg.cli.auto_fdhg_relbench import main as cli_main
from fdhg.compiler.fold_safe_fdhg import (
    CONTINUOUS_MISSING_CATEGORY,
    apply_quantile_discretizer,
    discover_dmax1_edges_with_audit,
    discover_earliest_fold_candidate_edges,
    fit_quantile_discretizer,
    fit_afd_edges,
    fit_transform_fdhg_fold,
    materialize_ambiguity_features,
    point_in_time_asof_join,
)
from fdhg.onboarding.auto_fdhg import (
    AutoFdhgOptions,
    align_feature_blocks,
    audit_residual_columns,
    auto_fdhg_relbench,
    canonical_relbench_dataset_name,
    edge_fold_gain,
    evaluate_joint_gate,
    metric_improvement,
    prepare_auto_fdhg,
    resolve_canonical_dfs_features,
    resolve_edge_screening_min_positive_folds,
    select_joint_variant,
    selection_fold_metadata,
    summarize_edge_screening,
    fit_transform_single_edge_fdhg_fold_cached,
)


class FakeRelBenchTable:
    def __init__(self, df, *, pkey_col=None, fkeys=None, time_col=None) -> None:
        self.df = df
        self.pkey_col = pkey_col
        self.fkey_col_to_pkey_table = fkeys or {}
        self.time_col = time_col


class FakeDatabase:
    def __init__(self, table_dict) -> None:
        self.table_dict = table_dict


class FakeDataset:
    val_timestamp = pd.Timestamp("2020-01-01")
    test_timestamp = pd.Timestamp("2021-01-01")

    def __init__(self, table_dict) -> None:
        self._db = FakeDatabase(table_dict)

    def get_db(self):
        return self._db


class FakeTask:
    entity_col = "driver_id"
    time_col = "timestamp"
    target_col = "position"
    task_type = "regression"
    metrics = ["mae", "rmse"]

    def __init__(self, train, val) -> None:
        self.calls = []
        self._tables = {
            "train": FakeRelBenchTable(train),
            "val": FakeRelBenchTable(val),
        }

    def get_table(self, split):
        self.calls.append(split)
        if split == "test":
            raise AssertionError("test split accessed")
        return self._tables[split]


def fake_objects():
    drivers = pd.DataFrame({
        "driver_id": ["d1", "d2", "d3", "d4"],
        "team": ["a", "a", "b", "b"],
    })
    rows = []
    for driver, base, team in [
        ("d1", 1.0, "a"),
        ("d2", 8.0, "a"),
        ("d3", 3.0, "b"),
        ("d4", 10.0, "b"),
    ]:
        for idx in range(8):
            rows.append({
                "result_id": f"{driver}-{idx}",
                "driver_id": driver,
                "race_date": pd.Timestamp("2019-01-01") + pd.Timedelta(days=idx * 8),
                "position_value": base + (idx % 2),
                "grid": base + idx,
                "status": "ok" if idx % 3 else "dnf",
                "team": team,
                "position": base,
            })
    results = pd.DataFrame(rows)
    train_rows = []
    for timestamp in pd.date_range("2019-02-01", periods=20, freq="7D"):
        for driver, label in [("d1", 1.5), ("d2", 8.5), ("d3", 3.5), ("d4", 10.5)]:
            train_rows.append({"driver_id": driver, "timestamp": timestamp, "position": label})
    train = pd.DataFrame(train_rows)
    val = pd.DataFrame({
        "driver_id": ["d1", "d2", "d3", "d4"],
        "timestamp": pd.to_datetime(["2019-08-01"] * 4),
        "position": [1.5, 8.5, 3.5, 10.5],
    })
    table_dict = {
        "drivers": FakeRelBenchTable(drivers, pkey_col="driver_id"),
        "results": FakeRelBenchTable(
            results,
            pkey_col="result_id",
            fkeys={"driver_id": "drivers"},
            time_col="race_date",
        ),
    }
    return FakeDataset(table_dict), FakeTask(train, val), "0.fake"


def loader(dataset_name, task_name, download):
    assert dataset_name == "rel-f1"
    assert task_name == "driver-position"
    return fake_objects()


def loader_with_source_mutation(*, future_status: str):
    def _loader(dataset_name, task_name, download):
        dataset, task, version = loader(dataset_name, task_name, download)
        results = dataset.get_db().table_dict["results"].df.copy()
        latest = results["race_date"].max()
        results.loc[results["race_date"].eq(latest), "status"] = future_status
        dataset.get_db().table_dict["results"] = FakeRelBenchTable(
            results,
            pkey_col="result_id",
            fkeys={"driver_id": "drivers"},
            time_col="race_date",
        )
        return dataset, task, version

    return _loader


def loader_with_validation_mutation(*, validation_label: float, validation_time: str):
    def _loader(dataset_name, task_name, download):
        dataset, task, version = loader(dataset_name, task_name, download)
        val = task._tables["val"].df.copy()
        val["position"] = validation_label
        val["timestamp"] = pd.Timestamp(validation_time)
        task._tables["val"] = FakeRelBenchTable(val)
        return dataset, task, version

    return _loader


def loader_with_points_milliseconds(dataset_name, task_name, download):
    dataset, task, version = loader(dataset_name, task_name, download)
    results = dataset.get_db().table_dict["results"].df.copy()
    results["points"] = [float(idx % 10) for idx in range(len(results))]
    results["milliseconds"] = [float(1000 + idx * 7) for idx in range(len(results))]
    dataset.get_db().table_dict["results"] = FakeRelBenchTable(
        results,
        pkey_col="result_id",
        fkeys={"driver_id": "drivers"},
        time_col="race_date",
    )
    return dataset, task, version


def test_edge_fold_gain_lower_is_better() -> None:
    assert edge_fold_gain(
        auto_score=1.0,
        auto_plus_single_edge_score=0.75,
        direction="lower",
    ) == pytest.approx(0.25)


def test_edge_fold_gain_higher_is_better() -> None:
    assert edge_fold_gain(
        auto_score=0.7,
        auto_plus_single_edge_score=0.8,
        direction="higher",
    ) == pytest.approx(0.1)


def test_edge_screening_strict_mean_gain_tie_fails() -> None:
    summary = summarize_edge_screening(
        gains=[0.1, -0.1],
        usable_feature_counts=[1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=1,
    )
    assert summary["mean_gain"] == pytest.approx(0.0)
    assert summary["screening_status"] == "screened_out"
    assert summary["rejection_reason"] == "non_positive_mean_gain"


def test_edge_screening_majority_defaults_to_ceil_half() -> None:
    assert resolve_edge_screening_min_positive_folds(
        AutoFdhgOptions(selection_folds=3, edge_screening_min_positive_folds=None)
    ) == 2
    assert resolve_edge_screening_min_positive_folds(
        AutoFdhgOptions(selection_folds=4, edge_screening_min_positive_folds=None)
    ) == 2
    assert resolve_edge_screening_min_positive_folds(
        AutoFdhgOptions(selection_folds=5, edge_screening_min_positive_folds=None)
    ) == 3


def test_default_positive_fraction_resolves_expected_effective_fold_counts() -> None:
    options = AutoFdhgOptions(
        selection_folds=3,
        edge_screening_rule="positive_fraction",
    )
    assert resolve_edge_screening_min_positive_folds(
        options,
        effective_selection_folds=3,
    ) == 2
    assert resolve_edge_screening_min_positive_folds(
        options,
        effective_selection_folds=2,
    ) == 2
    assert resolve_edge_screening_min_positive_folds(
        options,
        effective_selection_folds=1,
    ) == 1


def test_positive_fraction_uses_effective_selection_folds_without_float_boundary_error() -> None:
    assert resolve_edge_screening_min_positive_folds(
        AutoFdhgOptions(
            selection_folds=3,
            edge_screening_rule="positive_fraction",
            edge_screening_min_positive_fraction=1 / 3,
        ),
        effective_selection_folds=3,
    ) == 1


def test_requested_three_effective_two_fold_metadata_warns() -> None:
    split_plan = {
        "protocol": "expanding_window",
        "folds": [
            {
                "fold": 0,
                "train_indices": [0],
                "validation_indices": [1],
                "unique_train_timestamps": 1,
                "unique_validation_timestamps": 1,
                "train_rows": 10,
                "validation_rows": 5,
            },
            {
                "fold": 1,
                "train_indices": [0, 1],
                "validation_indices": [2],
                "unique_train_timestamps": 2,
                "unique_validation_timestamps": 1,
                "train_rows": 15,
                "validation_rows": 5,
            },
        ],
    }
    meta = selection_fold_metadata(split_plan=split_plan, requested_selection_folds=3)
    assert meta["requested_selection_folds"] == 3
    assert meta["effective_selection_folds"] == 2
    assert meta["effective_fold_ids"] == [0, 1]
    assert "effective_selection_folds_below_requested_selection_folds" in meta["warnings"]
    assert "fold_with_fewer_than_two_unique_training_timestamps" in meta["warnings"]


def test_edge_screening_two_positive_folds_out_of_three_passes() -> None:
    summary = summarize_edge_screening(
        gains=[0.2, 0.1, -0.05],
        usable_feature_counts=[1, 1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=2,
    )
    assert summary["screening_status"] == "screened_in"
    assert summary["positive_fold_count"] == 2


def test_pooled_oof_screening_uses_aggregate_gain_and_keeps_fold_gains() -> None:
    summary = summarize_edge_screening(
        gains=[0.2, -0.01],
        usable_feature_counts=[1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=1,
        screening_rule="pooled_oof",
        aggregate_auto_score=0.50,
        aggregate_candidate_score=0.55,
        aggregate_gain=0.05,
    )
    assert summary["screening_status"] == "screened_in"
    assert summary["mean_gain"] == pytest.approx(0.095)
    assert summary["aggregate_gain"] == pytest.approx(0.05)
    assert summary["aggregate_candidate_score"] == pytest.approx(0.55)


def test_pooled_oof_does_not_require_positive_fold_fraction_but_fraction_rule_does() -> None:
    pooled = summarize_edge_screening(
        gains=[0.5, -0.1, -0.1],
        usable_feature_counts=[1, 1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=2,
        screening_rule="pooled_oof",
        aggregate_gain=0.01,
        aggregate_auto_score=0.50,
        aggregate_candidate_score=0.51,
    )
    fraction = summarize_edge_screening(
        gains=[0.5, -0.1, -0.1],
        usable_feature_counts=[1, 1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=2,
        screening_rule="positive_fraction",
    )
    assert pooled["positive_fold_count"] == 1
    assert pooled["screening_status"] == "screened_in"
    assert fraction["screening_status"] == "screened_out"
    assert fraction["rejection_reason"] == "insufficient_positive_folds"


def test_worst_fold_relative_degradation_uses_corresponding_fold_auto_score() -> None:
    summary = summarize_edge_screening(
        gains=[-0.1, -0.2],
        usable_feature_counts=[1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=1,
        screening_rule="pooled_oof",
        aggregate_gain=0.01,
        aggregate_auto_score=100.0,
        aggregate_candidate_score=100.01,
        fold_auto_scores=[10.0, 100.0],
        fold_ids=[7, 8],
    )
    assert summary["worst_fold_relative_degradation"] == pytest.approx(0.01)
    assert summary["worst_fold_relative_degradation_fold"] == 7


def test_pooled_oof_metric_interface_prediction_shapes() -> None:
    regression_y = pd.Series([1.0, 2.0, 3.0])
    regression_pred = auto_fdhg_module._concat_predictions([
        np.array([1.0, 2.0]),
        np.array([3.0]),
    ])
    assert auto_fdhg_module._metric_score(
        regression_y,
        regression_pred,
        metric="rmse",
        problem_type="regression",
    ) == pytest.approx(0.0)

    binary_y = pd.Series([0, 1, 1, 0])
    binary_pred = auto_fdhg_module._concat_predictions([
        np.array([0.1, 0.8]),
        np.array([0.9, 0.2]),
    ])
    assert auto_fdhg_module._metric_score(
        binary_y,
        binary_pred,
        metric="roc_auc",
        problem_type="binary_classification",
    ) == pytest.approx(1.0)

    multiclass_y = pd.Series([0, 1, 2, 1])
    multiclass_pred = auto_fdhg_module._concat_predictions([
        np.array([0, 1]),
        np.array([2, 1]),
    ])
    assert auto_fdhg_module._metric_score(
        multiclass_y,
        multiclass_pred,
        metric="accuracy",
        problem_type="multiclass_classification",
    ) == pytest.approx(1.0)


def test_edge_screening_one_positive_fold_out_of_three_fails() -> None:
    summary = summarize_edge_screening(
        gains=[0.2, -0.01, -0.01],
        usable_feature_counts=[1, 1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=2,
    )
    assert summary["screening_status"] == "screened_out"
    assert summary["rejection_reason"] == "insufficient_positive_folds"


def test_edge_screening_positive_mean_but_insufficient_positive_folds_fails() -> None:
    summary = summarize_edge_screening(
        gains=[0.5, -0.1, -0.1],
        usable_feature_counts=[1, 1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=2,
    )
    assert summary["mean_gain"] > 0
    assert summary["screening_status"] == "screened_out"
    assert summary["rejection_reason"] == "insufficient_positive_folds"


def test_edge_screening_sufficient_positive_folds_but_non_positive_mean_fails() -> None:
    summary = summarize_edge_screening(
        gains=[0.1, 0.1, -0.4],
        usable_feature_counts=[1, 1, 1],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=2,
    )
    assert summary["positive_fold_count"] == 2
    assert summary["mean_gain"] < 0
    assert summary["screening_status"] == "screened_out"
    assert summary["rejection_reason"] == "non_positive_mean_gain"


def test_edge_screening_no_usable_residual_features_fails() -> None:
    summary = summarize_edge_screening(
        gains=[],
        usable_feature_counts=[0, 0, 0],
        future_lookup_violation_count=0,
        min_delta=0.0,
        min_positive_folds=2,
    )
    assert summary["screening_status"] == "no_usable_features"
    assert summary["rejection_reason"] == "no_usable_features"


def feature_decl(*, output, agg, source=None):
    return {
        "kind": "relational",
        "feature_id": output,
        "child_table": "results",
        "child_fk": "driver_id",
        "child_event_time_col": "race_date",
        "parent_key": "driver_id",
        "source_column": source,
        "aggregation": agg,
        "output_column": output,
        "temporal_predicate": "child.event_time <= target.target_time",
    }


def write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    auto_root = tmp_path / "auto"
    auto_dir = auto_root / "rel-f1_driver-position"
    auto_dir.mkdir(parents=True)
    auto = {
        "selected_features": [
            feature_decl(output="f_results_position_value_mean", agg="mean", source="position_value"),
            feature_decl(output="f_results_grid_min", agg="min", source="grid"),
        ]
    }
    (auto_dir / "selected_features.json").write_text(json.dumps(auto), encoding="utf-8")
    config_dir = tmp_path / "configs" / "reproduction"
    config_dir.mkdir(parents=True)
    config = {
        "tasks": {
            "rel-f1/driver-position": {
                "dfs": {
                    "child_table": "results",
                    "child_fk": "driver_id",
                    "child_time_col": "race_date",
                    "numeric_col": "position_value",
                }
            }
        }
    }
    (config_dir / "tasks.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return auto_root, tmp_path


def f1_canonical_declarations():
    specs = [
        ("count", "", "f_results_count"),
        ("mean", "positionOrder", "f_results_positionOrder_mean"),
        ("std", "positionOrder", "f_results_positionOrder_std"),
        ("min", "positionOrder", "f_results_positionOrder_min"),
        ("max", "positionOrder", "f_results_positionOrder_max"),
        ("days_since_last", "", "f_results_days_since_last"),
    ]
    rows = []
    for idx, (agg, source, output) in enumerate(specs):
        rows.append({
            "primitive_id": f"baseline::{agg}",
            "program_id": "canonical_dfs",
            "source_table": "results",
            "join_key": "driverId",
            "child_event_time_col": "date",
            "target_entity_key": "driverId",
            "target_time_col": "date",
            "source_column": source,
            "aggregation": agg,
            "output_column": output,
            "temporal_predicate": "results.date <= drivers.date",
            "materialization_strategy": "grouped_temporal_sweep",
            "implementation_version": "onboarding-v1",
            "leakage_safe": True,
            "temporal_safe": True,
            "auxiliary_output_columns": [f"{output}__is_missing"],
        })
    return rows


def write_canonical_f1_onboarding(root: Path, *, disagree: bool = False) -> Path:
    directory = root / "outputs" / "onboarding" / "relbench-v1-rel-f1_driver-position"
    directory.mkdir(parents=True)
    declarations = f1_canonical_declarations()
    (directory / "baseline_feature_config.json").write_text(
        json.dumps({"features": declarations}),
        encoding="utf-8",
    )
    manifest_names = []
    for row in declarations:
        manifest_names.append(row["output_column"])
        manifest_names.extend(row["auxiliary_output_columns"])
    if disagree:
        manifest_names = ["wrong_feature"]
    pd.DataFrame({"output_column": manifest_names}).to_csv(
        directory / "baseline_feature_manifest.csv",
        index=False,
    )
    (directory / "onboarding_manifest.json").write_text(
        json.dumps({
            "dataset": "relbench-v1-rel-f1",
            "task": "driver-position",
            "materialization_strategy": "grouped_temporal_sweep",
            "implementation_version": "onboarding-v1",
        }),
        encoding="utf-8",
    )
    pd.DataFrame({"driverId": [], "date": [], "position": []}).to_parquet(
        directory / "target_with_dfs_agg_train.parquet",
        index=False,
    )
    pd.DataFrame({"driverId": [], "date": [], "position": []}).to_parquet(
        directory / "target_with_dfs_agg_val.parquet",
        index=False,
    )
    return directory


def metadata():
    return {
        "entity_key": "driver_id",
        "target_time_col": "timestamp",
        "label_col": "position",
        "problem_type": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower",
    }


def test_metric_improvement_and_strict_min_delta_gate() -> None:
    assert metric_improvement(candidate=3.0, reference=4.0, direction="lower") == 1.0
    assert metric_improvement(candidate=0.8, reference=0.7, direction="higher") == pytest.approx(0.1)
    tied = select_joint_variant(
        mean_scores={"dfs_fallback": 4.0, "auto_only": 4.0, "auto_plus_fdhg": 4.0},
        metric_direction="lower",
        min_delta=0.0,
    )
    assert tied["selected_variant"] == "dfs_fallback"
    equal_delta = select_joint_variant(
        mean_scores={"dfs_fallback": 4.0, "auto_only": 3.9, "auto_plus_fdhg": 3.8},
        metric_direction="lower",
        min_delta=0.1,
    )
    assert equal_delta["selected_variant"] == "auto_plus_fdhg"
    assert equal_delta["gate_trials"][0]["passed"] is False
    assert equal_delta["gate_trials"][1]["passed"] is True


def test_ambiguity_numeric_key_canonicalization() -> None:
    values = [
        1,
        1.0,
        Decimal("1.00"),
    ]
    assert {normalize_value(value) for value in values} == {"n:1"}
    assert normalize_value(np.int64(1)) == "n:1"
    assert normalize_value(np.float64(1.0)) == "n:1"
    assert normalize_value("1") == "s:1"
    assert normalize_value(True) == "b:1"
    assert normalize_value(False) == "b:0"
    assert normalize_value(None) == "__NULL__"
    assert normalize_value(float("inf")) == "n:+inf"
    assert normalize_value(float("-inf")) == "n:-inf"
    assert normalize_value(Decimal("1.2500")) == "n:1.25"


def test_ambiguity_int_fit_float_transform_round_trip() -> None:
    mapping = fit_ambiguity_map(
        pd.DataFrame({"wins": pd.Series([0, 1, 1], dtype="int64"), "points": ["a", "b", "b"]}),
        lhs_columns=("wins",),
        rhs_column="points",
    )
    edge = FittedAmbiguityEdge(
        edge_id="standings:wins->points",
        source_table="standings",
        lhs_columns=("wins",),
        rhs_column="points",
        mapping=mapping,
        fit_start_time=None,
        fit_end_time="2020-01-01",
        maximum_source_time_used="2020-01-01",
        support=3,
        coverage=1.0,
        confidence=1.0,
        conflict_rate=0.0,
        selection_status="accepted",
        rejection_reason="",
        fold=0,
    )
    frame, _ = materialize_ambiguity_from_map(
        pd.DataFrame({"wins": pd.Series([0.0, 1.0], dtype="float64")}),
        fitted_edge=edge,
    )
    cols = [col for col in frame.columns if col.endswith("__majority_confidence")]
    assert cols
    assert frame[cols[0]].tolist() == [1.0, 1.0]


def test_compound_lhs_canonicalization_is_stable_and_typed() -> None:
    left = pd.DataFrame({"a": [1, "1", None], "b": [np.float64(2.0), Decimal("2.00"), 3]})
    right = left.iloc[[2, 0, 1]].reset_index(drop=True)
    tokens = normalize_lhs_frame(left, ("a", "b")).tolist()
    assert tokens[0] == "n:1|n:2"
    assert tokens[1] == "s:1|n:2"
    assert tokens[2] == "__NULL__|n:3"
    assert sorted(tokens) == sorted(normalize_lhs_frame(right, ("a", "b")).tolist())


def test_gate_outcomes() -> None:
    assert select_joint_variant(
        mean_scores={"dfs_fallback": 4.0, "auto_only": 4.2, "auto_plus_fdhg": 4.1},
        metric_direction="lower",
        min_delta=0.0,
    )["selected_variant"] == "dfs_fallback"
    assert select_joint_variant(
        mean_scores={"dfs_fallback": 4.0, "auto_only": 3.8, "auto_plus_fdhg": 3.82},
        metric_direction="lower",
        min_delta=0.0,
    )["selected_variant"] == "auto_only"
    assert select_joint_variant(
        mean_scores={"dfs_fallback": 4.0, "auto_only": 3.8, "auto_plus_fdhg": 3.7},
        metric_direction="lower",
        min_delta=0.0,
    )["selected_variant"] == "auto_plus_fdhg"


def test_alignment_rejects_key_and_label_errors() -> None:
    target = pd.DataFrame({
        "driver_id": ["d1", "d2"],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "position": [1.0, 2.0],
    })
    block = target.iloc[[1, 0]].reset_index(drop=True).assign(f_a=[10.0, 20.0])
    aligned = align_feature_blocks(
        target_rows=target,
        blocks=[("auto", block)],
        join_keys=["driver_id", "timestamp"],
        metadata=metadata(),
    )
    assert aligned["f_a"].tolist() == [20.0, 10.0]
    duplicate = pd.concat([block, block.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate_auto_keys"):
        align_feature_blocks(
            target_rows=target,
            blocks=[("auto", duplicate)],
            join_keys=["driver_id", "timestamp"],
            metadata=metadata(),
        )
    bad_label = block.copy()
    bad_label.loc[0, "position"] = 999
    with pytest.raises(ValueError, match="label_mismatch:auto"):
        align_feature_blocks(
            target_rows=target,
            blocks=[("auto", bad_label)],
            join_keys=["driver_id", "timestamp"],
            metadata=metadata(),
        )


def test_fold_safe_fdhg_fit_uses_only_horizon() -> None:
    dataset, task, _ = fake_objects()
    table_dict = dataset.get_db().table_dict
    edge = {
        "edge_id": "results:team->status",
        "source_table": "results",
        "lhs_columns": ("team",),
        "rhs_column": "status",
    }
    fitted = fit_afd_edges(
        inner_train_rows=task.get_table("train").df.iloc[:10],
        source_tables=table_dict,
        schema=None,
        task_metadata=metadata(),
        candidate_edges=[edge],
        max_edges=1,
        fold=0,
        fit_horizon=pd.Timestamp("2019-02-15"),
    )
    assert fitted
    assert pd.Timestamp(fitted[0].maximum_source_time_used) <= pd.Timestamp("2019-02-15")
    assert fitted[0].rhs_column != "position"


def test_point_in_time_lookup_uses_target_timestamp_and_preserves_rows() -> None:
    target = pd.DataFrame({
        "driver_id": ["d1", "d1", "d2", "d1"],
        "timestamp": pd.to_datetime(["2020-01-05", "2020-01-15", "2020-01-10", "2019-12-31"]),
    })
    source = pd.DataFrame({
        "driver_id": ["d1", "d1", "d1", "d2"],
        "race_date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-02-01", "2020-01-01"]),
        "wins": [1, 2, 99, 7],
    })
    view, audit = point_in_time_asof_join(
        target_rows=target,
        source_rows=source,
        entity_key="driver_id",
        target_time_col="timestamp",
        source_time_col="race_date",
        source_columns=("wins",),
    )
    assert view["wins"].tolist() == [1, 2, 7, pytest.approx(float("nan"), nan_ok=True)]
    assert audit["target_row_count"] == 4
    assert audit["matched_target_rows"] == 3
    assert audit["future_lookup_violation_count"] == 0
    assert audit["maximum_lookup_source_time"] == "2020-01-10 00:00:00"


def test_point_in_time_lookup_is_order_stable_and_breaks_duplicate_ties() -> None:
    target = pd.DataFrame({
        "driver_id": ["d1", "d1"],
        "timestamp": pd.to_datetime(["2020-01-10", "2020-01-11"]),
    })
    source = pd.DataFrame({
        "driver_id": ["d1", "d1", "d1"],
        "race_date": pd.to_datetime(["2020-01-10", "2020-01-10", "2020-01-01"]),
        "wins": [1, 2, 0],
    })
    view, _ = point_in_time_asof_join(
        target_rows=target.iloc[[1, 0]].reset_index(drop=True),
        source_rows=source.iloc[[2, 0, 1]].reset_index(drop=True),
        entity_key="driver_id",
        target_time_col="timestamp",
        source_time_col="race_date",
        source_columns=("wins",),
    )
    assert view["wins"].tolist() == [2, 2]


def test_fdhg_transform_uses_pit_lhs_not_full_table_latest() -> None:
    table_dict = {
        "drivers": FakeRelBenchTable(pd.DataFrame({"driver_id": ["d1"]}), pkey_col="driver_id"),
        "results": FakeRelBenchTable(
            pd.DataFrame({
                "driver_id": ["d1", "d1", "d1"],
                "race_date": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-03-01"]),
                "wins": [1, 2, 99],
                "points": ["low", "high", "future"],
            }),
            fkeys={"driver_id": "drivers"},
            time_col="race_date",
        ),
    }
    edge = {
        "edge_id": "results:wins->points",
        "source_table": "results",
        "lhs_columns": ("wins",),
        "rhs_column": "points",
    }
    train = pd.DataFrame({
        "driver_id": ["d1", "d1"],
        "timestamp": pd.to_datetime(["2020-01-05", "2020-01-15"]),
        "position": [1.0, 2.0],
    })
    val = pd.DataFrame({
        "driver_id": ["d1"],
        "timestamp": pd.to_datetime(["2020-02-01"]),
        "position": [2.0],
    })
    result = fit_transform_fdhg_fold(
        inner_train_rows=train,
        inner_validation_rows=val,
        source_tables=table_dict,
        schema=None,
        task_metadata=metadata(),
        candidate_edges=[edge],
        max_edges=1,
        fold=0,
    )
    assert result["target_lookup_audit"]
    assert all(row["future_lookup_violation_count"] == 0 for row in result["target_lookup_audit"])
    assert max(pd.Timestamp(row["maximum_lookup_source_time"]) for row in result["target_lookup_audit"]) <= pd.Timestamp("2020-01-10")
    feature_cols = [col for col in result["validation_x"].columns if "majority_confidence" in col]
    assert feature_cols
    assert result["validation_x"][feature_cols[0]].notna().all()


def test_fdhg_materialization_uses_edge_source_entity_column_when_fk_name_differs() -> None:
    table_dict = {
        "users": FakeRelBenchTable(pd.DataFrame({"UserId": ["u1", "u2"]}), pkey_col="UserId"),
        "posts": FakeRelBenchTable(
            pd.DataFrame({
                "OwnerUserId": ["u1", "u1", "u2", "u2"],
                "CreationDate": pd.to_datetime([
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-01",
                    "2020-01-02",
                ]),
                "PostTypeId": [1, 2, 1, 2],
                "ContentLicense": ["MIT", "CC", "MIT", "GPL"],
            }),
            fkeys={"OwnerUserId": "users"},
            time_col="CreationDate",
        ),
    }
    task_metadata = {
        "entity_key": "UserId",
        "target_time_col": "timestamp",
        "label_col": "label",
        "problem_type": "binary_classification",
        "primary_metric": "accuracy",
        "metric_direction": "higher",
    }
    edge = {
        "edge_id": "posts:PostTypeId->ContentLicense",
        "source_table": "posts",
        "lhs_columns": ("PostTypeId",),
        "rhs_column": "ContentLicense",
        "source_entity_column": "OwnerUserId",
    }
    train = pd.DataFrame({
        "UserId": ["u1", "u2"],
        "timestamp": pd.to_datetime(["2020-01-03", "2020-01-03"]),
        "label": [1, 0],
    })
    val = pd.DataFrame({
        "UserId": ["u1", "u2"],
        "timestamp": pd.to_datetime(["2020-01-04", "2020-01-04"]),
        "label": [1, 0],
    })
    single = fit_transform_single_edge_fdhg_fold_cached(
        inner_train_rows=train,
        inner_validation_rows=val,
        source_tables=table_dict,
        task_metadata=task_metadata,
        edge=edge,
        lookup_source_columns_by_table={"posts": ["PostTypeId"]},
        pit_lookup_cache={},
        fold=0,
    )
    combined = fit_transform_fdhg_fold(
        inner_train_rows=train,
        inner_validation_rows=val,
        source_tables=table_dict,
        schema=None,
        task_metadata=task_metadata,
        candidate_edges=[edge],
        max_edges=1,
        fold=0,
    )
    for result in (single, combined):
        assert result["target_lookup_audit"]
        assert all(row["source_entity_column"] == "OwnerUserId" for row in result["target_lookup_audit"])
        assert all(row["future_lookup_violation_count"] == 0 for row in result["target_lookup_audit"])
        feature_cols = [col for col in result["train_x"].columns if "majority_confidence" in col]
        assert feature_cols
        assert result["train_x"][feature_cols[0]].notna().all()


def test_validation_lookup_does_not_update_fitted_mapping() -> None:
    table_dict = {
        "drivers": FakeRelBenchTable(pd.DataFrame({"driver_id": ["d1"]}), pkey_col="driver_id"),
        "results": FakeRelBenchTable(
            pd.DataFrame({
                "driver_id": ["d1", "d1"],
                "race_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                "wins": [1, 2],
                "points": ["train", "validation_only"],
            }),
            fkeys={"driver_id": "drivers"},
            time_col="race_date",
        ),
    }
    edge = {
        "edge_id": "results:wins->points",
        "source_table": "results",
        "lhs_columns": ("wins",),
        "rhs_column": "points",
    }
    fitted = fit_afd_edges(
        inner_train_rows=pd.DataFrame({"driver_id": ["d1"], "timestamp": [pd.Timestamp("2020-01-15")]}),
        source_tables=table_dict,
        schema=None,
        task_metadata=metadata(),
        candidate_edges=[edge],
        max_edges=1,
        fold=0,
        fit_horizon=pd.Timestamp("2020-01-15"),
        min_coverage=0.0,
    )
    val = pd.DataFrame({"driver_id": ["d1"], "timestamp": [pd.Timestamp("2020-02-15")]})
    frame, _, lookup_audit = materialize_ambiguity_features(
        fitted_edges=fitted,
        target_rows=val,
        source_tables=table_dict,
        task_metadata=metadata(),
    )
    feature_cols = [col for col in frame.columns if "majority_confidence" in col]
    assert feature_cols
    assert frame[feature_cols[0]].isna().all()
    assert lookup_audit[0]["maximum_lookup_source_time"] == "2020-02-01 00:00:00"
    assert fitted[0].maximum_source_time_used == "2020-01-01 00:00:00"


def test_missing_source_time_rejects_temporal_fdhg_edge() -> None:
    table_dict = {
        "drivers": FakeRelBenchTable(pd.DataFrame({"driver_id": ["d1"]}), pkey_col="driver_id"),
        "results": FakeRelBenchTable(
            pd.DataFrame({"driver_id": ["d1"], "wins": [1], "points": ["x"]}),
            fkeys={"driver_id": "drivers"},
            time_col=None,
        ),
    }
    fitted = fit_afd_edges(
        inner_train_rows=pd.DataFrame({"driver_id": ["d1"], "timestamp": [pd.Timestamp("2020-01-01")]}),
        source_tables=table_dict,
        schema=None,
        task_metadata=metadata(),
        candidate_edges=[{
            "edge_id": "results:wins->points",
            "source_table": "results",
            "lhs_columns": ("wins",),
            "rhs_column": "points",
        }],
        max_edges=1,
        fold=0,
        fit_horizon=pd.Timestamp("2020-01-01"),
    )
    assert fitted[0].selection_status == "rejected"
    assert fitted[0].rejection_reason == "missing_source_time_for_point_in_time_lookup"


def test_zero_support_fold_edge_is_rejected() -> None:
    table_dict = {
        "drivers": FakeRelBenchTable(pd.DataFrame({"driver_id": ["d1"]}), pkey_col="driver_id"),
        "results": FakeRelBenchTable(
            pd.DataFrame({
                "driver_id": ["d1"],
                "race_date": [pd.Timestamp("2020-01-01")],
                "number": [1],
                "rank": [None],
            }),
            fkeys={"driver_id": "drivers"},
            time_col="race_date",
        ),
    }
    edge = {
        "edge_id": "results:number->rank",
        "source_table": "results",
        "lhs_columns": ("number",),
        "rhs_column": "rank",
    }
    fitted = fit_afd_edges(
        inner_train_rows=pd.DataFrame({"driver_id": ["d1"], "timestamp": [pd.Timestamp("2019-01-01")]}),
        source_tables=table_dict,
        schema=None,
        task_metadata=metadata(),
        candidate_edges=[edge],
        max_edges=1,
        fold=0,
        fit_horizon=pd.Timestamp("2019-01-01"),
    )
    assert fitted[0].selection_status == "rejected"
    assert fitted[0].rejection_reason == "zero_support"


def test_residual_column_audit_rejects_unusable_train_columns() -> None:
    frame = pd.DataFrame({
        "all_missing": [None, None, None],
        "zero_finite": ["x", "y", "z"],
        "constant": [1.0, 1.0, 1.0],
        "constant__is_missing": [0, 0, 0],
        "usable": [0.0, 1.0, 2.0],
    })
    rows, usable = audit_residual_columns(
        frame=frame,
        feature_cols=list(frame.columns),
        fold=0,
        provenance=[],
    )
    reasons = {row["feature_name"]: row["rejection_reason"] for row in rows}
    assert usable == ["usable"]
    assert reasons["all_missing"] == "all_values_missing"
    assert reasons["zero_finite"] == "zero_finite_values"
    assert reasons["constant"] == "zero_variance"
    assert reasons["constant__is_missing"] == "constant_missing_indicator"


def test_prepare_dry_run_reports_artifacts_and_edges_without_test_access(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    prepared = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=2),
        object_loader=loader,
        include_gate=False,
    )
    assert not prepared["blockers"]
    assert prepared["manifest"]["test_split_accessed"] is False
    assert prepared["dfs_features"]
    assert len(prepared["auto_features"]) == 2
    assert len(prepared["accepted_fdhg_edges"]) <= 2
    folds = prepared["manifest"]["fold_boundaries"]
    assert len(folds) == 3
    assert all(row["unique_validation_timestamps"] > 1 for row in folds)


def test_auto_fdhg_strategies_share_ordered_candidate_ids(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    candidate_ids_by_strategy = {}
    for strategy in ("independent", "greedy", "greedy_backward"):
        prepared = prepare_auto_fdhg(
            dataset_name="rel-f1",
            task_name="driver-position",
            output_root=tmp_path / f"out-{strategy}",
            download=False,
            auto_output_root=auto_root,
            dfs_source_root=dfs_root,
            options=AutoFdhgOptions(
                selection_folds=3,
                max_fdhg_edges=3,
                edge_selection_strategy=strategy,
            ),
            object_loader=loader,
            include_gate=False,
        )
        candidate_ids_by_strategy[strategy] = prepared["manifest"]["ordered_candidate_edge_ids"]
        assert prepared["manifest"]["edge_selection_strategy"] == strategy
        assert prepared["candidate_discovery"]["strategy_name"] == strategy
        assert prepared["candidate_discovery"]["candidate_count_before_budget"] >= len(
            prepared["accepted_fdhg_edges"]
        )
    assert candidate_ids_by_strategy["independent"]
    assert candidate_ids_by_strategy["independent"] == candidate_ids_by_strategy["greedy"]
    assert candidate_ids_by_strategy["independent"] == candidate_ids_by_strategy["greedy_backward"]


def historical_candidate_edges() -> list[dict[str, Any]]:
    return [
        {
            "edge_id": "results:points->milliseconds",
            "source_table": "results",
            "lhs_columns": ["points"],
            "rhs_column": "milliseconds",
            "lhs_original_columns": ["points"],
            "rhs_original_column": "milliseconds",
            "source_entity_column": "driver_id",
            "source_entity_column_resolution": "accepted_relation_child_fk",
            "source_relation_id": "results:driver_id->drivers:driver_id",
            "edge_rank": 1,
            "historical_mean_gain": 0.001255,
        },
        {
            "edge_id": "results:status->team",
            "source_table": "results",
            "lhs_columns": ["status"],
            "rhs_column": "team",
            "source_entity_column": "driver_id",
            "edge_rank": 2,
        },
        {
            "edge_id": "results:team->status",
            "source_table": "results",
            "lhs_columns": ["team"],
            "rhs_column": "status",
            "source_entity_column": "driver_id",
            "edge_rank": 3,
        },
    ]


def write_historical_candidate_edges(path: Path) -> Path:
    path.write_text(json.dumps(historical_candidate_edges()), encoding="utf-8")
    return path


def test_historical_candidate_replay_preserves_numeric_endpoint_under_exclude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_discovery(**_kwargs):
        raise AssertionError("candidate discovery called during replay")

    monkeypatch.setattr(auto_fdhg_module, "discover_earliest_fold_candidate_edges", fail_discovery)
    auto_root, dfs_root = write_artifacts(tmp_path)
    replay_file = write_historical_candidate_edges(tmp_path / "historical_edges.json")
    prepared = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(
            selection_folds=3,
            max_fdhg_edges=3,
            continuous_fdhg_mode="exclude",
            fdhg_candidate_edges_file=replay_file,
        ),
        object_loader=loader_with_points_milliseconds,
        include_gate=True,
    )

    assert prepared["candidate_discovery"]["candidate_discovery_protocol"] == "historical_candidate_replay"
    assert prepared["candidate_discovery"]["candidate_rediscovery_performed"] is False
    assert prepared["candidate_discovery"]["loaded_candidate_edge_count"] == 3
    assert prepared["candidate_discovery"]["candidate_count_after_budget"] == 3
    assert prepared["accepted_fdhg_edges"][0]["edge_id"] == "results:points->milliseconds"
    assert prepared["accepted_fdhg_edges"][0]["historical_mean_gain"] == pytest.approx(0.001255)
    assert prepared["manifest"]["candidate_edges_file"] == str(replay_file)
    assert prepared["manifest"]["candidate_edges_file_sha256"]
    assert prepared["manifest"]["test_split_accessed"] is False
    assert prepared["gate"]["ordered_candidate_edge_ids"][0] == "results:points->milliseconds"


def test_historical_candidate_replay_preserves_exact_order_and_budget(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    replay_file = write_historical_candidate_edges(tmp_path / "historical_edges.json")
    prepared = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(
            selection_folds=3,
            max_fdhg_edges=2,
            fdhg_candidate_edges_file=replay_file,
        ),
        object_loader=loader_with_points_milliseconds,
        include_gate=False,
    )

    assert prepared["manifest"]["loaded_candidate_edge_count"] == 3
    assert prepared["manifest"]["ordered_candidate_edge_ids"] == [
        "results:points->milliseconds",
        "results:status->team",
    ]
    assert [edge["edge_rank"] for edge in prepared["accepted_fdhg_edges"]] == [1, 2]


def test_replay_candidate_list_is_identical_across_selection_strategies(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    replay_file = write_historical_candidate_edges(tmp_path / "historical_edges.json")
    prepared_by_strategy = {}
    for strategy in ("independent", "greedy", "greedy_backward"):
        prepared_by_strategy[strategy] = prepare_auto_fdhg(
            dataset_name="rel-f1",
            task_name="driver-position",
            output_root=tmp_path / f"out-{strategy}",
            download=False,
            auto_output_root=auto_root,
            dfs_source_root=dfs_root,
            options=AutoFdhgOptions(
                selection_folds=3,
                max_fdhg_edges=3,
                edge_selection_strategy=strategy,
                fdhg_candidate_edges_file=replay_file,
            ),
            object_loader=loader_with_points_milliseconds,
            include_gate=False,
        )

    candidate_lists = {
        strategy: prepared["manifest"]["ordered_candidate_edge_ids"]
        for strategy, prepared in prepared_by_strategy.items()
    }
    assert candidate_lists["independent"] == candidate_lists["greedy"] == candidate_lists["greedy_backward"]
    for strategy, prepared in prepared_by_strategy.items():
        assert prepared["manifest"]["edge_selection_strategy"] == strategy
        assert prepared["candidate_discovery"]["candidate_rediscovery_performed"] is False
        assert prepared["candidate_discovery"]["candidate_edges_file_sha256"] == prepared_by_strategy[
            "independent"
        ]["candidate_discovery"]["candidate_edges_file_sha256"]


def test_export_fdhg_candidate_edges_requires_complete_definitions(tmp_path: Path) -> None:
    output_dir = tmp_path / "historical_output"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps({"accepted_fdhg_edges": historical_candidate_edges()}),
        encoding="utf-8",
    )
    exported = tmp_path / "exported_edges.json"
    assert export_candidate_edges_main([
        "--input-output-dir",
        str(output_dir),
        "--output-file",
        str(exported),
    ]) == 0
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert [edge["edge_id"] for edge in payload] == [
        "results:points->milliseconds",
        "results:status->team",
        "results:team->status",
    ]

    incomplete = tmp_path / "incomplete_output"
    incomplete.mkdir()
    (incomplete / "candidate_discovery.json").write_text(
        json.dumps({"accepted_edges": [{"edge_id": "results:points->milliseconds"}]}),
        encoding="utf-8",
    )
    assert export_candidate_edges_main([
        "--input-output-dir",
        str(incomplete),
        "--output-file",
        str(tmp_path / "should_not_exist.json"),
    ]) == 1


def test_auto_fdhg_candidate_universe_ignores_future_source_mutations(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    prepared_a = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out-a",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=3),
        object_loader=loader_with_source_mutation(future_status="future-a"),
        include_gate=False,
    )
    prepared_b = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out-b",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=3),
        object_loader=loader_with_source_mutation(future_status="future-b"),
        include_gate=False,
    )
    assert prepared_a["manifest"]["ordered_candidate_edge_ids"]
    assert (
        prepared_a["manifest"]["ordered_candidate_edge_ids"]
        == prepared_b["manifest"]["ordered_candidate_edge_ids"]
    )
    assert prepared_a["manifest"]["candidate_count_before_budget"] == prepared_b["manifest"]["candidate_count_before_budget"]


def test_auto_fdhg_candidate_universe_ignores_validation_target_mutations(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    prepared_a = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out-a",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=3),
        object_loader=loader_with_validation_mutation(
            validation_label=-999.0,
            validation_time="2018-01-01",
        ),
        include_gate=False,
    )
    prepared_b = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out-b",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=3),
        object_loader=loader_with_validation_mutation(
            validation_label=999.0,
            validation_time="2030-01-01",
        ),
        include_gate=False,
    )
    assert prepared_a["manifest"]["ordered_candidate_edge_ids"]
    assert (
        prepared_a["manifest"]["ordered_candidate_edge_ids"]
        == prepared_b["manifest"]["ordered_candidate_edge_ids"]
    )
    assert prepared_a["manifest"]["candidate_discovery_fold"] == prepared_b["manifest"]["candidate_discovery_fold"]
    assert prepared_a["manifest"]["candidate_discovery_fit_horizon"] == prepared_b["manifest"]["candidate_discovery_fit_horizon"]


def test_auto_fdhg_candidate_audit_excludes_relation_fk_as_dependency(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    prepared = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=3),
        object_loader=loader,
        include_gate=False,
    )
    audit = {
        (row["source_table"], row["column"]): row
        for row in prepared["candidate_discovery"]["candidate_column_audit"]
    }
    fk = audit[("results", "driver_id")]
    assert fk["actual_foreign_key"] is True
    assert fk["source_entity_column"] is True
    assert fk["determinant_eligible"] is False
    assert fk["dependent_eligible"] is False


def test_default_continuous_fdhg_mode_excludes_high_cardinality_numeric() -> None:
    train = pd.DataFrame({
        "driver_id": ["d1", "d2"],
        "timestamp": pd.to_datetime(["2020-02-01", "2020-02-02"]),
        "position": [1.0, 2.0],
    })
    values = [*range(260), *range(40)]
    source = pd.DataFrame({
        "driver_id": ["d1"] * 150 + ["d2"] * 150,
        "race_date": pd.date_range("2020-01-01", periods=300, freq="h"),
        "hist_ctr": values,
        "status": ["low" if value < 150 else "high" for value in values],
    })
    prepared = {
        "metadata": metadata(),
        "train_df": train,
        "table_dict": {
            "results": FakeRelBenchTable(
                source,
                fkeys={"driver_id": "drivers"},
                time_col="race_date",
            )
        },
        "accepted_relations": [{
            "status": "accepted",
            "child_table": "results",
            "child_fk": "driver_id",
            "parent_table": "drivers",
            "parent_key": "driver_id",
        }],
        "split_plan": {"folds": [{"fold": 0, "train_indices": [0, 1], "validation_indices": []}]},
    }
    discovery = discover_earliest_fold_candidate_edges(prepared=prepared, edge_budget=8)
    audit = {
        row["column"]: row
        for row in discovery["provenance"]["candidate_column_audit"]
    }
    assert audit["hist_ctr"]["exclusion_reason"] == "high_cardinality_numeric_excluded"
    assert all("hist_ctr" not in edge["edge_id"] for edge in discovery["accepted_edges"])


def test_low_cardinality_float_categoricals_survive_pair_validation() -> None:
    source = pd.DataFrame([
        {
            "driver_id": "d1",
            "race_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx),
            "IsClick": float(click),
            "ObjectType": float(object_type),
            "Position": float(position),
        }
        for idx, (click, object_type, position) in enumerate(
            (click, object_type, position)
            for _repeat in range(2)
            for click in range(2)
            for object_type in range(3)
            for position in range(5)
        )
    ])
    accepted, rejected, audit_rows = discover_dmax1_edges_with_audit(
        table_dict={
            "search": FakeRelBenchTable(
                source,
                fkeys={"driver_id": "drivers"},
                time_col="race_date",
            )
        },
        metadata=metadata(),
        relations=[{
            "status": "accepted",
            "child_table": "search",
            "child_fk": "driver_id",
            "parent_table": "drivers",
            "parent_key": "driver_id",
        }],
        max_edges=None,
        continuous_fdhg_mode="exclude",
    )

    audit = {row["column"]: row for row in audit_rows}
    for column in ("IsClick", "ObjectType", "Position"):
        assert audit[column]["dtype"] == "float64"
        assert audit[column]["eligibility_status"] == "safe_categorical"
        assert audit[column]["determinant_eligible"] is True
        assert audit[column]["dependent_eligible"] is True
    edge_ids = {edge["edge_id"] for edge in accepted}
    expected = {
        f"search:{lhs}->{rhs}"
        for lhs in ("IsClick", "ObjectType", "Position")
        for rhs in ("IsClick", "ObjectType", "Position")
        if lhs != rhs
    }
    assert edge_ids == expected
    assert len(accepted) == 6
    assert not any(row["rejection_reason"] == "continuous_numeric_pair_excluded" for row in rejected)


def test_exclude_mode_only_treats_true_high_cardinality_numeric_as_continuous() -> None:
    source = pd.DataFrame({
        "driver_id": ["d1"] * 300,
        "race_date": pd.date_range("2020-01-01", periods=300, freq="h"),
        "safe_code": [float(value % 3) for value in range(300)],
        "status": ["a" if value % 2 == 0 else "b" for value in range(300)],
        "hist_ctr": [float(value) for value in [*range(270), *range(30)]],
    })
    accepted, rejected, audit_rows = discover_dmax1_edges_with_audit(
        table_dict={
            "search": FakeRelBenchTable(
                source,
                fkeys={"driver_id": "drivers"},
                time_col="race_date",
            )
        },
        metadata=metadata(),
        relations=[{
            "status": "accepted",
            "child_table": "search",
            "child_fk": "driver_id",
            "parent_table": "drivers",
            "parent_key": "driver_id",
        }],
        max_edges=None,
        continuous_fdhg_mode="exclude",
    )

    audit = {row["column"]: row for row in audit_rows}
    assert audit["safe_code"]["eligibility_status"] == "safe_categorical"
    assert audit["hist_ctr"]["eligibility_status"] == "high_cardinality_numeric_excluded"
    assert {edge["edge_id"] for edge in accepted} == {
        "search:safe_code->status",
        "search:status->safe_code",
    }
    assert all("hist_ctr" not in edge["edge_id"] for edge in accepted)
    assert not any(
        row.get("lhs_columns") in {"safe_code", "status"}
        and row.get("rhs_column") in {"safe_code", "status"}
        and row.get("rejection_reason") == "continuous_numeric_pair_excluded"
        for row in rejected
    )


def test_quantile_mode_uses_transformed_continuous_endpoint_not_raw_column() -> None:
    source = pd.DataFrame({
        "driver_id": ["d1"] * 300,
        "race_date": pd.date_range("2020-01-01", periods=300, freq="h"),
        "safe_code": [float(value % 3) for value in range(300)],
        "hist_ctr": [float(value) for value in [*range(270), *range(30)]],
    })
    accepted, rejected, audit_rows = discover_dmax1_edges_with_audit(
        table_dict={
            "search": FakeRelBenchTable(
                source,
                fkeys={"driver_id": "drivers"},
                time_col="race_date",
            )
        },
        metadata=metadata(),
        relations=[{
            "status": "accepted",
            "child_table": "search",
            "child_fk": "driver_id",
            "parent_table": "drivers",
            "parent_key": "driver_id",
        }],
        max_edges=None,
        continuous_fdhg_mode="quantile",
        continuous_fdhg_bins=4,
    )

    audit = {row["column"]: row for row in audit_rows}
    transformed = audit["hist_ctr"]["transformed_column"]
    assert audit["hist_ctr"]["eligibility_status"] == "continuous_discretization_eligible"
    assert transformed == "__fdhg_qbin__hist_ctr"
    assert accepted
    assert any(
        transformed in [*edge["lhs_columns"], edge["rhs_column"]]
        for edge in accepted
    )
    assert all(
        "hist_ctr" not in [*edge["lhs_columns"], edge["rhs_column"]]
        for edge in accepted
    )
    assert not any(
        row.get("lhs_columns") == "hist_ctr" or row.get("rhs_column") == "hist_ctr"
        for row in rejected
    )


def test_rel_avito_searchinfo_isuserloggedon_exclude_recovers_candidate_budget() -> None:
    rows = []
    for idx, (click, object_type, position, category) in enumerate(
        (click, object_type, position, category)
        for _repeat in range(2)
        for click in range(2)
        for object_type in range(3)
        for position in range(5)
        for category in range(4)
    ):
        rows.append({
            "SearchID": "s1",
            "SearchDate": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=idx),
            "IsClick": float(click),
            "ObjectType": float(object_type),
            "Position": float(position),
            "CategoryID": float(category),
        })
    search_stream = pd.DataFrame(rows)
    train = pd.DataFrame({
        "SearchID": ["s1", "s1", "s1"],
        "SearchDate": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]),
        "IsUserLoggedOn": [0, 1, 0],
    })
    prepared = {
        "metadata": {
            "entity_key": "SearchID",
            "target_time_col": "SearchDate",
            "label_col": "IsUserLoggedOn",
            "problem_type": "binary_classification",
            "primary_metric": "accuracy",
            "metric_direction": "higher",
        },
        "train_df": train,
        "table_dict": {
            "SearchStream": FakeRelBenchTable(
                search_stream,
                fkeys={"SearchID": "SearchInfo"},
                time_col="SearchDate",
            )
        },
        "accepted_relations": [{
            "status": "accepted",
            "child_table": "SearchStream",
            "child_fk": "SearchID",
            "parent_table": "SearchInfo",
            "parent_key": "SearchID",
        }],
        "split_plan": {
            "folds": [
                {"fold": 0, "train_indices": [0, 1, 2], "validation_indices": []},
            ]
        },
    }
    discovery = discover_earliest_fold_candidate_edges(
        prepared=prepared,
        edge_budget=8,
        continuous_fdhg_mode="exclude",
    )

    audit = {
        row["column"]: row
        for row in discovery["provenance"]["candidate_column_audit"]
    }
    for column, cardinality in {"IsClick": 2, "ObjectType": 3, "Position": 5}.items():
        assert audit[column]["dtype"] == "float64"
        assert audit[column]["cardinality"] == cardinality
        assert audit[column]["eligibility_status"] == "safe_categorical"
        assert audit[column]["determinant_eligible"] is True
        assert audit[column]["dependent_eligible"] is True
    assert discovery["provenance"]["candidate_count_before_budget"] >= 8
    assert discovery["provenance"]["candidate_count_after_budget"] == 8
    assert len(discovery["accepted_edges"]) == 8
    assert discovery["provenance"]["accepted_candidate_edge_count"] == 8
    assert discovery["provenance"]["rejection_reason_counts"].get("continuous_numeric_pair_excluded", 0) == 0


def test_quantile_mode_discovers_continuous_categorical_edges_and_audits_status() -> None:
    train = pd.DataFrame({
        "driver_id": ["d1", "d2"],
        "timestamp": pd.to_datetime(["2020-02-01", "2020-02-02"]),
        "position": [1.0, 2.0],
    })
    values = [*range(260), *range(40)]
    source = pd.DataFrame({
        "driver_id": ["d1"] * 150 + ["d2"] * 150,
        "race_date": pd.date_range("2020-01-01", periods=300, freq="h"),
        "hist_ctr": values,
        "status": ["low" if value % 2 == 0 else "high" for value in values],
    })
    prepared = {
        "metadata": metadata(),
        "train_df": train,
        "table_dict": {
            "results": FakeRelBenchTable(
                source,
                fkeys={"driver_id": "drivers"},
                time_col="race_date",
            )
        },
        "accepted_relations": [{
            "status": "accepted",
            "child_table": "results",
            "child_fk": "driver_id",
            "parent_table": "drivers",
            "parent_key": "driver_id",
        }],
        "split_plan": {"folds": [{"fold": 0, "train_indices": [0, 1], "validation_indices": []}]},
    }
    discovery = discover_earliest_fold_candidate_edges(
        prepared=prepared,
        edge_budget=8,
        continuous_fdhg_mode="quantile",
        continuous_fdhg_bins=4,
    )
    audit = {
        row["column"]: row
        for row in discovery["provenance"]["candidate_column_audit"]
    }
    assert audit["hist_ctr"]["eligibility_status"] == "continuous_discretization_eligible"
    assert audit["hist_ctr"]["transformed_column"] == "__fdhg_qbin__hist_ctr"
    assert any(
        edge["lhs_column_kind"] == "discretized_continuous"
        and edge["rhs_column_kind"] == "categorical"
        for edge in discovery["accepted_edges"]
    )


def test_quantile_boundaries_fit_only_on_fold_train_and_validation_perturbation_ignored() -> None:
    table_dict = {
        "drivers": FakeRelBenchTable(pd.DataFrame({"driver_id": ["d1"]}), pkey_col="driver_id"),
        "results": FakeRelBenchTable(
            pd.DataFrame({
                "driver_id": ["d1"] * 7,
                "race_date": pd.to_datetime([
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-03-01",
                    "2020-04-01",
                ]),
                "hist_ctr": [0.0, 1.0, 2.0, np.nan, 100.0, 9999.0, -9999.0],
                "status": ["a", "a", "b", "b", "c", "future", "future"],
            }),
            fkeys={"driver_id": "drivers"},
            time_col="race_date",
        ),
    }
    edge = {
        "edge_id": "results:__fdhg_qbin__hist_ctr->status",
        "source_table": "results",
        "lhs_columns": ("__fdhg_qbin__hist_ctr",),
        "rhs_column": "status",
        "continuous_columns": [{
            "original_column": "hist_ctr",
            "transformed_column": "__fdhg_qbin__hist_ctr",
        }],
    }
    train = pd.DataFrame({
        "driver_id": ["d1"],
        "timestamp": pd.to_datetime(["2020-01-05"]),
        "position": [1.0],
    })
    val = pd.DataFrame({
        "driver_id": ["d1"],
        "timestamp": pd.to_datetime(["2020-04-02"]),
        "position": [1.0],
    })
    result = fit_transform_fdhg_fold(
        inner_train_rows=train,
        inner_validation_rows=val,
        source_tables=table_dict,
        schema=None,
        task_metadata=metadata(),
        candidate_edges=[edge],
        max_edges=1,
        fold=0,
        continuous_fdhg_mode="quantile",
        continuous_fdhg_bins=2,
    )
    boundaries = result["continuous_discretization_boundaries"][edge["edge_id"]]["__fdhg_qbin__hist_ctr"]["boundaries"]
    assert boundaries == [1.5]
    assert result["continuous_discretization_audit"][0]["non_null_count"] == 4
    assert result["continuous_discretization_audit"][0]["missing_count"] == 1


def test_missing_values_use_stable_explicit_quantile_category() -> None:
    transformed = apply_quantile_discretizer(pd.Series([None, np.nan, 1.0, 2.0]), boundaries=[1.5])
    assert transformed.tolist() == [
        CONTINUOUS_MISSING_CATEGORY,
        CONTINUOUS_MISSING_CATEGORY,
        "bin_0",
        "bin_1",
    ]


def test_duplicate_quantiles_and_constant_columns_do_not_crash() -> None:
    spec = fit_quantile_discretizer(
        pd.Series([1.0, 1.0, 1.0, 2.0, 2.0]),
        requested_bins=8,
        min_effective_bins=2,
    )
    assert spec["requested_bins"] == 8
    assert spec["effective_bins"] >= 2
    constant = fit_quantile_discretizer(
        pd.Series([1.0, 1.0, 1.0]),
        requested_bins=8,
        min_effective_bins=2,
    )
    assert constant["accepted"] is False
    assert constant["rejection_reason"] == "continuous_discretization_effective_bins_below_minimum"


def test_zero_row_csv_outputs_keep_headers(tmp_path: Path) -> None:
    path = tmp_path / "pair_screening.csv"
    auto_fdhg_module._write_csv(path, [])
    text = path.read_text(encoding="utf-8")
    assert text.startswith("dataset,")
    assert len(text) > 1


def _synthetic_pair_rescue_inputs() -> dict:
    rows = []
    for fold in range(3):
        for idx in range(4):
            rows.append({
                "entity_id": f"e{fold}-{idx}",
                "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=fold * 10 + idx),
                "label": idx % 2,
                "fold": fold,
            })
    train_targets = pd.DataFrame(rows)
    folds = []
    for fold in range(3):
        validation_indices = train_targets.index[train_targets["fold"].eq(fold)].tolist()
        train_indices = train_targets.index[~train_targets["fold"].eq(fold)].tolist()
        folds.append({"fold": fold, "train_indices": train_indices, "validation_indices": validation_indices})
    return {
        "dataset_name": "synthetic",
        "task_name": "pair-rescue",
        "train_targets": train_targets.drop(columns=["fold"]),
        "table_dict": {},
        "metadata": {
            "entity_key": "entity_id",
            "target_time_col": "timestamp",
            "label_col": "label",
            "primary_metric": "accuracy",
            "metric_direction": "higher",
            "problem_type": "binary_classification",
        },
        "split_plan": {"folds": folds},
        "dfs_features": [],
        "auto_features": [],
        "fdhg_edges": [
            {"edge_id": "edge_a", "source_table": "events", "lhs_columns": ("a",), "rhs_column": "x"},
            {"edge_id": "edge_b", "source_table": "events", "lhs_columns": ("b",), "rhs_column": "y"},
            {"edge_id": "edge_c", "source_table": "events", "lhs_columns": ("c",), "rhs_column": "z"},
        ],
        "join_keys": ["entity_id", "timestamp"],
    }


def _patch_pair_rescue_scoring(monkeypatch: pytest.MonkeyPatch, *, single_a_score: float = 0.49) -> None:
    def passthrough_pair(train_targets, validation_targets, **_kwargs):
        return train_targets.reset_index(drop=True).copy(), validation_targets.reset_index(drop=True).copy()

    def fdhg_frame(target_rows: pd.DataFrame, edge_ids: Sequence[str]) -> pd.DataFrame:
        frame = target_rows[["entity_id", "timestamp"]].reset_index(drop=True).copy()
        for edge_id in edge_ids:
            frame[f"fdhg__{edge_id}"] = np.arange(len(frame), dtype=float) + len(edge_id)
        return frame

    def single_edge_materialization(*, inner_train_rows, inner_validation_rows, edge, fold, **_kwargs):
        edge_id = str(edge["edge_id"])
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, [edge_id]),
            "validation_x": fdhg_frame(inner_validation_rows, [edge_id]),
            "feature_provenance": [{
                "feature_name": f"fdhg__{edge_id}",
                "edge_id": edge_id,
                "source_table": "events",
            }],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    def pair_materialization(*, inner_train_rows, inner_validation_rows, candidate_edges, fold, **_kwargs):
        edge_ids = [str(edge["edge_id"]) for edge in candidate_edges]
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, edge_ids),
            "validation_x": fdhg_frame(inner_validation_rows, edge_ids),
            "feature_provenance": [
                {
                    "feature_name": f"fdhg__{edge_id}",
                    "edge_id": edge_id,
                    "source_table": "events",
                }
                for edge_id in edge_ids
            ],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    def score_by_features(*, feature_cols, **_kwargs):
        cols = set(feature_cols)
        if {"fdhg__edge_a", "fdhg__edge_b"}.issubset(cols):
            return 0.70
        if cols == {"fdhg__edge_a"}:
            return single_a_score
        if cols & {"fdhg__edge_b", "fdhg__edge_c"}:
            return 0.49
        return 0.50

    monkeypatch.setattr(auto_fdhg_module, "materialize_declared_feature_frame_pair", passthrough_pair)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_single_edge_fdhg_fold_cached", single_edge_materialization)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_fdhg_fold", pair_materialization)
    monkeypatch.setattr(auto_fdhg_module, "score_matrix", score_by_features)


def test_greedy_pairwise_rescue_selects_helpful_pair_when_singles_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pair_rescue_scoring(monkeypatch)
    kwargs = _synthetic_pair_rescue_inputs()
    gate = evaluate_joint_gate(
        **kwargs,
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=3,
            edge_selection_strategy="greedy",
            edge_screening_min_delta=0.0,
            edge_screening_min_positive_folds=2,
        ),
    )

    assert gate["pairwise_rescue_used"] is True
    assert gate["selected_initial_pair"] == "edge_a||edge_b"
    assert gate["pairwise_rescue_reason"] == "selected_pair_passed_gate"
    assert [edge["edge_id"] for edge in gate["screened_fdhg_edges"]] == ["edge_a", "edge_b"]
    assert gate["mean_scores"]["auto_plus_fdhg"] == pytest.approx(0.70)
    assert gate["selected_variant"] == "auto_plus_fdhg"
    assert {row["edge_id"] for row in gate["selected_fdhg_edges"] if row["selected_for_combined_block"]} == {
        "edge_a",
        "edge_b",
    }
    selected_pair_rows = [row for row in gate["pair_screening"] if row["selected_initial_pair"]]
    assert len(selected_pair_rows) == 1
    assert selected_pair_rows[0]["positive_fold_count"] == 3


def test_independent_remains_empty_without_pairwise_rescue(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pair_rescue_scoring(monkeypatch)
    kwargs = _synthetic_pair_rescue_inputs()
    gate = evaluate_joint_gate(
        **kwargs,
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=3,
            edge_selection_strategy="independent",
            edge_screening_min_delta=0.0,
            edge_screening_min_positive_folds=2,
        ),
    )

    assert gate["screened_fdhg_edges"] == []
    assert gate["pair_screening"] == []
    assert gate["pairwise_rescue_used"] is False
    assert gate["pairwise_rescue_reason"] == "not_greedy_strategy"
    assert "auto_plus_fdhg" not in gate["mean_scores"]
    assert gate["selected_variant"] == "auto_only"


def test_greedy_skips_pairwise_rescue_when_single_edge_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pair_rescue_scoring(monkeypatch, single_a_score=0.70)
    kwargs = _synthetic_pair_rescue_inputs()
    gate = evaluate_joint_gate(
        **kwargs,
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=3,
            edge_selection_strategy="greedy",
            edge_screening_min_delta=0.0,
            edge_screening_min_positive_folds=2,
        ),
    )

    assert [edge["edge_id"] for edge in gate["screened_fdhg_edges"]] == ["edge_a"]
    assert gate["pair_screening"] == []
    assert gate["pairwise_rescue_used"] is False
    assert gate["pairwise_rescue_reason"] == "not_attempted_single_edge_passed"


def test_greedy_pairwise_rescue_requires_delta_passing_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pair_rescue_scoring(monkeypatch)
    kwargs = _synthetic_pair_rescue_inputs()
    gate = evaluate_joint_gate(
        **kwargs,
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=3,
            edge_selection_strategy="greedy",
            edge_screening_min_delta=0.25,
            edge_screening_min_positive_folds=2,
        ),
    )

    assert gate["screened_fdhg_edges"] == []
    assert gate["pairwise_rescue_used"] is False
    assert gate["pairwise_rescue_reason"] == "no_pair_passed_gate"
    best = gate["pair_screening"][0]
    assert best["pair_id"] == "edge_a||edge_b"
    assert best["mean_gain"] == pytest.approx(0.20)
    assert best["positive_fold_count"] == 3
    assert best["passing_fold_count"] == 0


def _patch_strategy_dispatch_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    def passthrough_pair(train_targets, validation_targets, **_kwargs):
        return train_targets.reset_index(drop=True).copy(), validation_targets.reset_index(drop=True).copy()

    def fdhg_frame(target_rows: pd.DataFrame, edge_ids: Sequence[str]) -> pd.DataFrame:
        frame = target_rows[["entity_id", "timestamp"]].reset_index(drop=True).copy()
        for edge_id in edge_ids:
            frame[f"fdhg__{edge_id}"] = np.arange(len(frame), dtype=float) + len(edge_id)
        return frame

    def single_edge_materialization(*, inner_train_rows, inner_validation_rows, edge, fold, **_kwargs):
        edge_id = str(edge["edge_id"])
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, [edge_id]),
            "validation_x": fdhg_frame(inner_validation_rows, [edge_id]),
            "feature_provenance": [{
                "feature_name": f"fdhg__{edge_id}",
                "edge_id": edge_id,
                "source_table": "events",
            }],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    def block_materialization(*, inner_train_rows, inner_validation_rows, candidate_edges, fold, **_kwargs):
        edge_ids = [str(edge["edge_id"]) for edge in candidate_edges]
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, edge_ids),
            "validation_x": fdhg_frame(inner_validation_rows, edge_ids),
            "feature_provenance": [
                {
                    "feature_name": f"fdhg__{edge_id}",
                    "edge_id": edge_id,
                    "source_table": "events",
                }
                for edge_id in edge_ids
            ],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    def score_by_features(*, feature_cols, **_kwargs):
        edge_ids = frozenset(
            str(col).removeprefix("fdhg__")
            for col in feature_cols
            if str(col).startswith("fdhg__")
        )
        scores = {
            frozenset(): 0.50,
            frozenset({"edge_a"}): 0.56,
            frozenset({"edge_b"}): 0.58,
            frozenset({"edge_c"}): 0.57,
            frozenset({"edge_a", "edge_b"}): 0.55,
            frozenset({"edge_a", "edge_c"}): 0.54,
            frozenset({"edge_b", "edge_c"}): 0.70,
            frozenset({"edge_a", "edge_b", "edge_c"}): 0.69,
        }
        return scores[edge_ids]

    monkeypatch.setattr(auto_fdhg_module, "materialize_declared_feature_frame_pair", passthrough_pair)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_single_edge_fdhg_fold_cached", single_edge_materialization)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_fdhg_fold", block_materialization)
    monkeypatch.setattr(auto_fdhg_module, "score_matrix", score_by_features)


def _strategy_dispatch_gate(monkeypatch: pytest.MonkeyPatch, strategy: str) -> dict:
    _patch_strategy_dispatch_scoring(monkeypatch)
    kwargs = _synthetic_pair_rescue_inputs()
    return evaluate_joint_gate(
        **kwargs,
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=3,
            edge_selection_strategy=strategy,
            edge_screening_min_delta=0.0,
            edge_screening_min_positive_folds=2,
        ),
    )


def test_strategy_dispatch_selects_strategy_specific_edge_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    independent = _strategy_dispatch_gate(monkeypatch, "independent")
    greedy = _strategy_dispatch_gate(monkeypatch, "greedy")
    backward = _strategy_dispatch_gate(monkeypatch, "greedy_backward")

    assert independent["ordered_candidate_edge_ids"] == ["edge_a", "edge_b", "edge_c"]
    assert greedy["ordered_candidate_edge_ids"] == independent["ordered_candidate_edge_ids"]
    assert backward["ordered_candidate_edge_ids"] == independent["ordered_candidate_edge_ids"]
    assert independent["independent_screened_in_edge_ids"] == ["edge_b", "edge_c", "edge_a"]
    assert independent["strategy_selected_edge_ids"] == ["edge_b", "edge_c", "edge_a"]
    assert greedy["independent_screened_in_edge_ids"] == ["edge_b", "edge_c", "edge_a"]
    assert greedy["strategy_selected_edge_ids"] == ["edge_b", "edge_c"]
    assert backward["independent_screened_in_edge_ids"] == ["edge_b", "edge_c", "edge_a"]
    assert backward["strategy_selected_edge_ids"] == ["edge_b", "edge_c"]
    assert greedy["final_combined_block_edge_ids"] == ["edge_b", "edge_c"]
    assert backward["final_combined_block_edge_ids"] == ["edge_b", "edge_c"]
    assert [edge["edge_id"] for edge in greedy["screened_fdhg_edges"]] == ["edge_b", "edge_c"]
    assert [edge["edge_id"] for edge in backward["screened_fdhg_edges"]] == ["edge_b", "edge_c"]


def test_strategy_dispatch_integration_produces_distinct_selected_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_by_strategy = {
        strategy: tuple(_strategy_dispatch_gate(monkeypatch, strategy)["strategy_selected_edge_ids"])
        for strategy in ("independent", "greedy", "greedy_backward")
    }

    assert selected_by_strategy["independent"] == ("edge_b", "edge_c", "edge_a")
    assert selected_by_strategy["greedy"] == ("edge_b", "edge_c")
    assert selected_by_strategy["greedy_backward"] == ("edge_b", "edge_c")
    assert len(set(selected_by_strategy.values())) == 2


def _patch_pooled_strategy_scoring(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pooled_scores: Mapping[frozenset[str], float],
) -> None:
    def passthrough_pair(train_targets, validation_targets, **_kwargs):
        return train_targets.reset_index(drop=True).copy(), validation_targets.reset_index(drop=True).copy()

    def fdhg_frame(target_rows: pd.DataFrame, edge_ids: Sequence[str]) -> pd.DataFrame:
        frame = target_rows[["entity_id", "timestamp"]].reset_index(drop=True).copy()
        for edge_id in edge_ids:
            frame[f"fdhg__{edge_id}"] = np.arange(len(frame), dtype=float) + len(edge_id)
        return frame

    def single_edge_materialization(*, inner_train_rows, inner_validation_rows, edge, fold, **_kwargs):
        edge_id = str(edge["edge_id"])
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, [edge_id]),
            "validation_x": fdhg_frame(inner_validation_rows, [edge_id]),
            "feature_provenance": [{
                "feature_name": f"fdhg__{edge_id}",
                "edge_id": edge_id,
                "source_table": "events",
            }],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    def block_materialization(*, inner_train_rows, inner_validation_rows, candidate_edges, fold, **_kwargs):
        edge_ids = [str(edge["edge_id"]) for edge in candidate_edges]
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, edge_ids),
            "validation_x": fdhg_frame(inner_validation_rows, edge_ids),
            "feature_provenance": [
                {
                    "feature_name": f"fdhg__{edge_id}",
                    "edge_id": edge_id,
                    "source_table": "events",
                }
                for edge_id in edge_ids
            ],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    fold_scores = {
        frozenset(): [0.50, 0.50, 0.50],
        frozenset({"edge_a"}): [0.40, 0.40, 0.90],
        frozenset({"edge_b"}): [0.70, 0.70, 0.70],
        frozenset({"edge_c"}): [0.65, 0.65, 0.65],
        frozenset({"edge_a", "edge_b"}): [0.45, 0.45, 0.45],
        frozenset({"edge_a", "edge_c"}): [0.45, 0.45, 0.45],
        frozenset({"edge_b", "edge_c"}): [0.70, 0.70, 0.70],
        frozenset({"edge_a", "edge_b", "edge_c"}): [0.40, 0.40, 0.40],
    }

    def edge_set_from_features(feature_cols) -> frozenset[str]:
        return frozenset(
            str(col).removeprefix("fdhg__")
            for col in feature_cols
            if str(col).startswith("fdhg__")
        )

    def fold_id_from_val(val_x: pd.DataFrame) -> int:
        first = pd.Timestamp(val_x["timestamp"].iloc[0])
        return int((first - pd.Timestamp("2020-01-01")).days // 10)

    def score_by_features(*, feature_cols, val_x, **_kwargs):
        return fold_scores[edge_set_from_features(feature_cols)][fold_id_from_val(val_x)]

    def score_with_predictions(*, feature_cols, val_x, **_kwargs):
        edge_set = edge_set_from_features(feature_cols)
        return {
            "score": fold_scores[edge_set][fold_id_from_val(val_x)],
            "prediction": np.full(len(val_x), pooled_scores[edge_set], dtype=float),
        }

    def pooled_metric(_y_true, prediction, **_kwargs):
        return float(np.mean(np.asarray(prediction, dtype=float)))

    monkeypatch.setattr(auto_fdhg_module, "materialize_declared_feature_frame_pair", passthrough_pair)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_single_edge_fdhg_fold_cached", single_edge_materialization)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_fdhg_fold", block_materialization)
    monkeypatch.setattr(auto_fdhg_module, "score_matrix", score_by_features)
    monkeypatch.setattr(auto_fdhg_module, "score_matrix_with_predictions", score_with_predictions)
    monkeypatch.setattr(auto_fdhg_module, "_metric_score", pooled_metric)


def _pooled_strategy_gate(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    *,
    pooled_scores: Mapping[frozenset[str], float] | None = None,
) -> dict:
    _patch_pooled_strategy_scoring(
        monkeypatch,
        pooled_scores=pooled_scores or {
            frozenset(): 0.50,
            frozenset({"edge_a"}): 0.80,
            frozenset({"edge_b"}): 0.65,
            frozenset({"edge_c"}): 0.70,
            frozenset({"edge_a", "edge_b"}): 0.72,
            frozenset({"edge_a", "edge_c"}): 0.90,
            frozenset({"edge_b", "edge_c"}): 0.68,
            frozenset({"edge_a", "edge_b", "edge_c"}): 0.60,
        },
    )
    return evaluate_joint_gate(
        **_synthetic_pair_rescue_inputs(),
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=3,
            edge_selection_strategy=strategy,
            edge_screening_rule="pooled_oof",
            edge_screening_min_delta=0.0,
            edge_screening_min_positive_folds=3,
        ),
    )


def test_pooled_oof_independent_ranks_edges_by_pooled_aggregate_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _pooled_strategy_gate(monkeypatch, "independent")

    assert gate["strategy_selected_edge_ids"] == ["edge_a", "edge_c", "edge_b"]
    rows = {row["edge_id"]: row for row in gate["edge_screening"]}
    assert rows["edge_a"]["aggregate_gain"] == pytest.approx(0.30)
    assert rows["edge_b"]["mean_gain"] > rows["edge_a"]["mean_gain"]
    assert rows["edge_a"]["screening_rule"] == "pooled_oof"


def test_pooled_oof_greedy_forward_uses_pooled_incremental_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _pooled_strategy_gate(monkeypatch, "greedy")

    assert gate["strategy_selected_edge_ids"] == ["edge_a", "edge_c"]
    accepted = [row for row in gate["edge_selection_trace"] if row["decision"] == "accepted"]
    assert accepted[0]["candidate_edge_id"] == "edge_a"
    assert accepted[0]["aggregate_gain"] == pytest.approx(0.30)
    assert accepted[1]["candidate_edge_id"] == "edge_c"
    assert accepted[1]["aggregate_baseline_score"] == pytest.approx(0.80)
    assert accepted[1]["aggregate_candidate_score"] == pytest.approx(0.90)
    assert accepted[1]["aggregate_gain"] == pytest.approx(0.10)


def test_pooled_oof_greedy_backward_uses_pooled_removal_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _pooled_strategy_gate(monkeypatch, "greedy_backward")

    assert gate["strategy_selected_edge_ids"] == ["edge_a", "edge_c"]
    removed = [row for row in gate["edge_selection_trace"] if row["decision"] == "removed"]
    assert len(removed) == 1
    assert removed[0]["candidate_edge_id"] == "edge_b"
    assert removed[0]["aggregate_baseline_score"] == pytest.approx(0.60)
    assert removed[0]["aggregate_candidate_score"] == pytest.approx(0.90)
    assert removed[0]["aggregate_gain"] == pytest.approx(0.30)


def test_pooled_oof_pairwise_rescue_uses_pooled_pair_gain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _pooled_strategy_gate(
        monkeypatch,
        "greedy",
        pooled_scores={
            frozenset(): 0.50,
            frozenset({"edge_a"}): 0.49,
            frozenset({"edge_b"}): 0.49,
            frozenset({"edge_c"}): 0.49,
            frozenset({"edge_a", "edge_b"}): 0.80,
            frozenset({"edge_a", "edge_c"}): 0.70,
            frozenset({"edge_b", "edge_c"}): 0.75,
            frozenset({"edge_a", "edge_b", "edge_c"}): 0.60,
        },
    )

    assert gate["pairwise_rescue_used"] is True
    assert gate["selected_initial_pair"] == "edge_a||edge_b"
    best = [row for row in gate["pair_screening"] if row["selected_initial_pair"]][0]
    assert best["aggregate_baseline_score"] == pytest.approx(0.50)
    assert best["aggregate_candidate_score"] == pytest.approx(0.80)
    assert best["aggregate_gain"] == pytest.approx(0.30)
    assert best["screening_rule"] == "pooled_oof"


def _synthetic_budget_inputs(edge_count: int = 32) -> dict:
    kwargs = _synthetic_pair_rescue_inputs()
    kwargs["fdhg_edges"] = [
        {
            "edge_id": f"edge_{idx:02d}",
            "source_table": "events",
            "lhs_columns": (f"lhs_{idx:02d}",),
            "rhs_column": f"rhs_{idx:02d}",
        }
        for idx in range(edge_count)
    ]
    return kwargs


def _patch_budget_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    def passthrough_pair(train_targets, validation_targets, **_kwargs):
        return train_targets.reset_index(drop=True).copy(), validation_targets.reset_index(drop=True).copy()

    def fdhg_frame(target_rows: pd.DataFrame, edge_ids: Sequence[str]) -> pd.DataFrame:
        frame = target_rows[["entity_id", "timestamp"]].reset_index(drop=True).copy()
        for edge_id in edge_ids:
            frame[f"fdhg__{edge_id}"] = np.arange(len(frame), dtype=float) + len(edge_id)
        return frame

    def single_edge_materialization(*, inner_train_rows, inner_validation_rows, edge, fold, **_kwargs):
        edge_id = str(edge["edge_id"])
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, [edge_id]),
            "validation_x": fdhg_frame(inner_validation_rows, [edge_id]),
            "feature_provenance": [{
                "feature_name": f"fdhg__{edge_id}",
                "edge_id": edge_id,
                "source_table": "events",
            }],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    def block_materialization(*, inner_train_rows, inner_validation_rows, candidate_edges, fold, **_kwargs):
        return {
            "fitted_edges": [],
            "edge_audit": [],
            "train_x": fdhg_frame(inner_train_rows, [str(edge["edge_id"]) for edge in candidate_edges]),
            "validation_x": fdhg_frame(inner_validation_rows, [str(edge["edge_id"]) for edge in candidate_edges]),
            "feature_provenance": [],
            "target_lookup_audit": [{"future_lookup_violation_count": 0}],
        }

    def score_by_features(*, feature_cols, **_kwargs):
        edge_count = sum(1 for col in feature_cols if str(col).startswith("fdhg__"))
        return 0.50 + 0.01 * edge_count

    monkeypatch.setattr(auto_fdhg_module, "materialize_declared_feature_frame_pair", passthrough_pair)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_single_edge_fdhg_fold_cached", single_edge_materialization)
    monkeypatch.setattr(auto_fdhg_module, "fit_transform_fdhg_fold", block_materialization)
    monkeypatch.setattr(auto_fdhg_module, "score_matrix", score_by_features)


def test_selection_budget_caps_independent_without_truncating_candidate_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_budget_scoring(monkeypatch)
    gate = evaluate_joint_gate(
        **_synthetic_budget_inputs(edge_count=32),
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=32,
            max_selected_fdhg_edges=8,
            edge_selection_strategy="independent",
            edge_screening_min_delta=0.0,
            edge_screening_min_positive_folds=2,
        ),
    )

    assert len(gate["ordered_candidate_edge_ids"]) == 32
    assert gate["ordered_candidate_edge_ids"] == [f"edge_{idx:02d}" for idx in range(32)]
    assert len(gate["independent_screened_in_edge_ids"]) == 32
    assert gate["strategy_selected_edge_ids"] == [f"edge_{idx:02d}" for idx in range(8)]
    assert gate["edge_selection_stop_reason"] == "selection_budget_reached"


def test_selection_budget_caps_greedy_from_full_candidate_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_budget_scoring(monkeypatch)
    gate = evaluate_joint_gate(
        **_synthetic_budget_inputs(edge_count=32),
        options=AutoFdhgOptions(
            selection_folds=3,
            min_delta=-0.1,
            max_fdhg_edges=32,
            max_selected_fdhg_edges=8,
            edge_selection_strategy="greedy",
            edge_screening_min_delta=0.0,
            edge_screening_min_positive_folds=2,
        ),
    )

    assert len(gate["ordered_candidate_edge_ids"]) == 32
    assert len(gate["strategy_selected_edge_ids"]) == 8
    assert gate["strategy_selected_edge_ids"] == [f"edge_{idx:02d}" for idx in range(8)]
    assert gate["edge_selection_stop_reason"] == "selection_budget_reached"


def test_changing_selection_budget_does_not_change_ordered_candidate_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    edges = [
        {
            "edge_id": f"edge_{idx:02d}",
            "source_table": "events",
            "lhs_columns": (f"lhs_{idx:02d}",),
            "rhs_column": f"rhs_{idx:02d}",
        }
        for idx in range(32)
    ]

    def fake_discovery(*, prepared, edge_budget, **kwargs):
        accepted = edges[:edge_budget]
        return {
            "accepted_edges": accepted,
            "rejected_edges": [],
            "provenance": {
                "candidate_discovery_protocol": "unit_fake",
                "candidate_count_before_budget": len(edges),
                "candidate_count_after_budget": len(accepted),
                "candidate_column_audit": [],
                "rejection_reason_counts": {},
                "ordered_candidate_edge_ids": [edge["edge_id"] for edge in accepted],
            },
        }

    monkeypatch.setattr(auto_fdhg_module, "discover_earliest_fold_candidate_edges", fake_discovery)
    auto_root, dfs_root = write_artifacts(tmp_path)
    first = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out-a",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=32, max_selected_fdhg_edges=8),
        object_loader=loader,
        include_gate=False,
    )
    second = prepare_auto_fdhg(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out-b",
        download=False,
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=32, max_selected_fdhg_edges=16),
        object_loader=loader,
        include_gate=False,
    )

    assert first["manifest"]["candidate_count_after_candidate_budget"] == 32
    assert first["manifest"]["max_selected_fdhg_edges"] == 8
    assert second["manifest"]["max_selected_fdhg_edges"] == 16
    assert first["manifest"]["ordered_candidate_edge_ids"] == second["manifest"]["ordered_candidate_edge_ids"]


def test_study_outcome_like_discovery_remains_valid() -> None:
    train = pd.DataFrame({
        "StudyId": ["s1", "s2", "s1"],
        "timestamp": pd.to_datetime(["2020-01-10", "2020-01-20", "2020-01-30"]),
        "outcome": [1, 0, 1],
    })
    interventions = pd.DataFrame({
        "InterventionId": [1, 2, 3, 4],
        "StudyId": ["s1", "s1", "s1", "s2"],
        "StartDate": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01"]),
        "ArmClass": ["drug", "drug", "control", "drug"],
        "Phase": ["i", "ii", "ii", "iii"],
        "BriefTitle": [" ".join(["long"] * 40)] * 4,
    })
    prepared = {
        "metadata": {
            "entity_key": "StudyId",
            "target_time_col": "timestamp",
            "label_col": "outcome",
            "primary_metric": "accuracy",
            "metric_direction": "higher",
        },
        "train_df": train,
        "table_dict": {
            "interventions": FakeRelBenchTable(
                interventions,
                pkey_col="InterventionId",
                fkeys={"StudyId": "studies"},
                time_col="StartDate",
            )
        },
        "accepted_relations": [{
            "status": "accepted",
            "child_table": "interventions",
            "child_fk": "StudyId",
            "parent_table": "studies",
            "parent_key": "StudyId",
        }],
        "split_plan": {
            "folds": [
                {"fold": 0, "train_indices": [0], "validation_indices": [1]},
                {"fold": 1, "train_indices": [0, 1], "validation_indices": [2]},
            ]
        },
    }
    discovery = discover_earliest_fold_candidate_edges(prepared=prepared, edge_budget=8)
    assert discovery["accepted_edges"]
    assert discovery["provenance"]["candidate_discovery_protocol"] == "fixed_from_earliest_inner_train_fold"
    audit = {
        (row["source_table"], row["column"]): row
        for row in discovery["provenance"]["candidate_column_audit"]
    }
    assert audit[("interventions", "StudyId")]["dependent_eligible"] is False
    assert audit[("interventions", "BriefTitle")]["dependent_eligible"] is False


def test_canonical_f1_onboarding_artifact_resolution(tmp_path: Path, monkeypatch) -> None:
    directory = write_canonical_f1_onboarding(tmp_path)
    calls = []

    def fail_read_parquet(*args, **kwargs):
        calls.append(args[0])
        raise AssertionError("fixed split parquet read")

    monkeypatch.setattr(pd, "read_parquet", fail_read_parquet)
    resolved = resolve_canonical_dfs_features(
        dataset_name="rel-f1",
        task_name="driver-position",
        root=tmp_path,
        metadata={
            "entity_key": "driverId",
            "target_time_col": "date",
            "label_col": "position",
        },
    )
    assert not calls
    assert resolved["blocker"] == ""
    assert canonical_relbench_dataset_name("rel-f1") == "relbench-v1-rel-f1"
    assert resolved["provenance"]["source_type"] == "canonical_onboarding_artifact"
    assert resolved["provenance"]["source_directory"] == str(directory)
    assert resolved["provenance"]["requested_dataset_name"] == "rel-f1"
    assert resolved["provenance"]["canonical_dataset_name"] == "relbench-v1-rel-f1"
    assert resolved["provenance"]["declaration_count"] == 6
    assert resolved["provenance"]["model_column_count"] == 12
    assert resolved["provenance"]["materializer"] == "grouped_temporal_sweep"
    assert resolved["provenance"]["fixed_split_parquets_reused_in_gate"] is False
    assert len(resolved["features"]) == 6
    assert resolved["features"][0]["source_column"] is None
    assert resolved["features"][0]["auxiliary_output_columns"] == ["f_results_count__is_missing"]
    assert resolved["provenance"]["canonical_auxiliary_columns"] == [
        f"{row['output_column']}__is_missing" for row in f1_canonical_declarations()
    ]


def test_explicit_dfs_override_precedes_canonical_onboarding(tmp_path: Path) -> None:
    write_canonical_f1_onboarding(tmp_path)
    override = tmp_path / "override.json"
    override.write_text(
        json.dumps({
            "features": [{
                "source_table": "results",
                "join_key": "driverId",
                "child_event_time_col": "date",
                "source_column": "",
                "aggregation": "count",
                "output_column": "f_override_count",
            }]
        }),
        encoding="utf-8",
    )
    resolved = resolve_canonical_dfs_features(
        dataset_name="rel-f1",
        task_name="driver-position",
        root=tmp_path,
        metadata={"entity_key": "driverId", "target_time_col": "date", "label_col": "position"},
        explicit_config=override,
    )
    assert resolved["provenance"]["source_type"] == "explicit_override"
    assert [row["output_column"] for row in resolved["features"]] == ["f_override_count"]


def test_inconsistent_canonical_onboarding_artifact_blocks(tmp_path: Path) -> None:
    write_canonical_f1_onboarding(tmp_path, disagree=True)
    resolved = resolve_canonical_dfs_features(
        dataset_name="rel-f1",
        task_name="driver-position",
        root=tmp_path,
        metadata={"entity_key": "driverId", "target_time_col": "date", "label_col": "position"},
    )
    assert resolved["features"] == []
    assert resolved["blocker"] == "canonical_dfs_config_manifest_disagree:feature_names"


def test_canonical_source_resolution_is_deterministic(tmp_path: Path) -> None:
    write_canonical_f1_onboarding(tmp_path)
    kwargs = {
        "dataset_name": "rel-f1",
        "task_name": "driver-position",
        "root": tmp_path,
        "metadata": {"entity_key": "driverId", "target_time_col": "date", "label_col": "position"},
    }
    first = resolve_canonical_dfs_features(**kwargs)
    second = resolve_canonical_dfs_features(**kwargs)
    assert first["provenance"] == second["provenance"]
    assert first["features"] == second["features"]


def test_dfs_materialized_independently_for_each_fold(tmp_path: Path, monkeypatch) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    import fdhg.onboarding.auto_fdhg as module

    original = module.materialize_declared_feature_frame
    dfs_calls = []

    def counting_materializer(*args, **kwargs):
        features = kwargs.get("features", ())
        if any(str(row.get("feature_id", "")).startswith("canonical_dfs::") for row in features):
            dfs_calls.append(len(args[0]) if args else 0)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "materialize_declared_feature_frame", counting_materializer)
    report = auto_fdhg_relbench(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        object_loader=loader,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=1, enable_edge_screening=False),
        write=True,
    )
    assert report.status == "completed"
    assert len(dfs_calls) >= 3


def test_missing_canonical_dfs_source_blocks(tmp_path: Path) -> None:
    auto_root = tmp_path / "auto"
    (auto_root / "rel-f1_driver-position").mkdir(parents=True)
    (auto_root / "rel-f1_driver-position" / "selected_features.json").write_text(
        json.dumps({"selected_features": [feature_decl(output="f_results_count", agg="count")]}),
        encoding="utf-8",
    )
    report = auto_fdhg_relbench(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        auto_output_root=auto_root,
        dfs_source_root=tmp_path,
        object_loader=loader,
        write=False,
    )
    assert report.status == "blocked"
    assert "missing_canonical_dfs_source" in report.blockers[0]


def test_write_outputs_have_gate_variants_and_cli(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    report = auto_fdhg_relbench(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        object_loader=loader,
        options=AutoFdhgOptions(selection_folds=3, max_fdhg_edges=1, enable_edge_screening=False),
        write=True,
    )
    assert report.status == "completed"
    out = report.output_dir
    assert (out / "manifest.json").exists()
    fold_metrics = pd.read_csv(out / "fold_metrics.csv")
    assert set(fold_metrics["variant"]) == {"dfs_fallback", "auto_only", "auto_plus_fdhg"}
    assert fold_metrics.groupby("fold")["variant"].nunique().eq(3).all()
    assert {
        "fold_improvement_over_auto",
        "fold_improvement_over_dfs",
        "mean_improvement_over_auto",
        "mean_improvement_over_dfs",
        "globally_selected_variant",
        "fold_best_variant",
    }.issubset(fold_metrics.columns)
    assert "selected" not in fold_metrics.columns
    assert (out / "fdhg_fold_feature_audit.csv").exists()
    assert (out / "pair_screening.csv").exists()
    assert (out / "pair_screening_fold_metrics.csv").exists()
    selected = json.loads((out / "selected_variant.json").read_text(encoding="utf-8"))
    assert selected["official_validation_was_used_for_selection"] is False
    assert selected["test_split_accessed"] is False
    assert (out / "official_validation_predictions.parquet").exists()

    # CLI argument parsing smoke without invoking real RelBench loading.
    assert cli_main(["--dataset", "rel-f1", "--task", "driver-position", "--output-root", str(tmp_path / "cli"), "--dry-run"]) in {0, 2}


def test_no_screened_edges_falls_back_without_distinct_fdhg_variant(tmp_path: Path) -> None:
    auto_root, dfs_root = write_artifacts(tmp_path)
    report = auto_fdhg_relbench(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        auto_output_root=auto_root,
        dfs_source_root=dfs_root,
        object_loader=loader,
        options=AutoFdhgOptions(
            selection_folds=3,
            max_fdhg_edges=1,
            edge_screening_min_delta=1_000_000.0,
        ),
        write=True,
    )
    assert report.status == "completed"
    out = report.output_dir
    fold_metrics = pd.read_csv(out / "fold_metrics.csv")
    assert set(fold_metrics["variant"]) == {"dfs_fallback", "auto_only"}
    screening = pd.read_csv(out / "fdhg_edge_screening.csv")
    assert not screening["screening_status"].eq("screened_in").any()
    selected_edges = pd.read_csv(out / "selected_fdhg_edges.csv")
    assert selected_edges["selected_for_combined_block"].eq(False).all()
    selected = json.loads((out / "selected_variant.json").read_text(encoding="utf-8"))
    assert selected["fdhg_screening_fallback"] == "auto_only"
    assert selected["selected_screened_edge_count"] == 0
    assert selected["fold_edge_instance_count"] == 3


def test_greedy_trial_selector_skips_higher_gain_candidate_that_fails_gate() -> None:
    trials = [
        (
            0,
            {"edge_id": "edge_high_gain_but_unstable"},
            {
                "mean_gain": 0.0030,
                "positive_fold_count": 1,
                "screening_status": "screened_out",
            },
            "edge_high_gain_but_unstable",
        ),
        (
            1,
            {"edge_id": "edge_lower_gain_but_stable"},
            {
                "mean_gain": 0.0020,
                "positive_fold_count": 2,
                "screening_status": "screened_in",
            },
            "edge_lower_gain_but_stable",
        ),
    ]

    selected = auto_fdhg_module._select_best_passing_greedy_trial(
        trials,
        trial_forward_passes=lambda trial: (
            trial["screening_status"] == "screened_in"
        ),
        trial_selection_gain=lambda trial: trial["mean_gain"],
    )

    assert selected is not None
    assert selected[3] == "edge_lower_gain_but_stable"


def test_greedy_trial_selector_returns_none_when_no_candidate_passes_gate() -> None:
    trials = [
        (
            0,
            {"edge_id": "edge_a"},
            {
                "mean_gain": 0.0040,
                "positive_fold_count": 1,
                "screening_status": "screened_out",
            },
            "edge_a",
        ),
        (
            1,
            {"edge_id": "edge_b"},
            {
                "mean_gain": -0.0010,
                "positive_fold_count": 0,
                "screening_status": "screened_out",
            },
            "edge_b",
        ),
    ]

    selected = auto_fdhg_module._select_best_passing_greedy_trial(
        trials,
        trial_forward_passes=lambda trial: (
            trial["screening_status"] == "screened_in"
        ),
        trial_selection_gain=lambda trial: trial["mean_gain"],
    )

    assert selected is None


def test_export_fdhg_candidate_edges_accepts_explicit_empty_pool(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "zero_candidate_output"
    output_dir.mkdir()

    (output_dir / "candidate_discovery.json").write_text(
        json.dumps({"accepted_edges": []}),
        encoding="utf-8",
    )

    exported = tmp_path / "exported_empty_edges.json"

    assert export_candidate_edges_main([
        "--input-output-dir",
        str(output_dir),
        "--output-file",
        str(exported),
    ]) == 0

    assert json.loads(
        exported.read_text(encoding="utf-8")
    ) == []



def test_historical_candidate_replay_accepts_explicit_empty_pool(
    tmp_path: Path,
) -> None:
    from fdhg.onboarding.auto_fdhg import (
        load_historical_candidate_edges,
    )

    candidate_file = tmp_path / "empty_candidates.json"
    candidate_file.write_text(
        "[]",
        encoding="utf-8",
    )

    replay = load_historical_candidate_edges(
        path=candidate_file,
        table_dict={},
        max_edges=32,
    )

    assert replay["accepted_edges"] == []

    provenance = replay["provenance"]

    assert provenance["loaded_candidate_edge_count"] == 0
    assert provenance["candidate_count_before_budget"] == 0
    assert provenance["candidate_count_after_budget"] == 0
    assert provenance["accepted_candidate_edge_count"] == 0
    assert provenance["ordered_candidate_edge_ids"] == []
    assert provenance["candidate_rediscovery_performed"] is False
    assert provenance["zero_candidate_pool"] is True


def test_point_in_time_asof_join_supports_event_row_lookup_and_strict_before():
    import pandas as pd

    from fdhg.compiler.fold_safe_fdhg import (
        point_in_time_asof_join,
    )

    target = pd.DataFrame({
        "primary_key": [101, 102, 103],
        "user": [7, 7, 8],
        "timestamp": pd.to_datetime([
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
            "2026-01-02 10:00:00",
        ]),
    })

    source = pd.DataFrame({
        "user": [7, 7, 7, 8],
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00:00",
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
            "2026-01-02 10:00:00",
        ]),
        "value": [
            "u7_old",
            "u7_equal_first",
            "u7_equal_second",
            "u8_equal",
        ],
    })

    result, audit = point_in_time_asof_join(
        target_rows=target,
        source_rows=source,
        entity_key="primary_key",
        target_lookup_entity_key="user",
        source_entity_key="user",
        target_time_col="timestamp",
        source_time_col="timestamp",
        source_columns=["value"],
        allow_exact_matches=False,
    )

    assert result["value"].iloc[0] == "u7_old"
    assert result["value"].iloc[1] == "u7_equal_first"
    assert pd.isna(result["value"].iloc[2])

    assert audit["target_entity_key"] == "primary_key"
    assert audit["target_lookup_entity_column"] == "user"
    assert audit["source_entity_column"] == "user"
    assert audit["allow_exact_matches"] is False
    assert audit["temporal_predicate"] == "<"
    assert audit["future_lookup_violation_count"] == 0
    assert audit["exact_match_violation_count"] == 0


def test_materialize_ambiguity_features_uses_edge_specific_event_row_semantics():
    from types import SimpleNamespace

    import pandas as pd

    from fdhg.compiler.ambiguity import (
        FittedAmbiguityEdge,
        normalize_lhs_frame,
    )
    from fdhg.compiler.fold_safe_fdhg import (
        materialize_ambiguity_features,
    )

    target = pd.DataFrame({
        "primary_key": [101, 102, 103],
        "user": [7, 7, 8],
        "timestamp": pd.to_datetime([
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
            "2026-01-02 10:00:00",
        ]),
    })

    source = pd.DataFrame({
        "primary_key": [1, 2, 3, 4],
        "user": [7, 7, 7, 8],
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00:00",
            "2026-01-02 10:00:00",
            "2026-01-03 10:00:00",
            "2026-01-02 10:00:00",
        ]),
        "lhs": ["a", "b", "c", "d"],
    })

    mapping = pd.DataFrame({
        "lhs_norm": normalize_lhs_frame(
            source,
            ["lhs"],
        ),
        "majority_confidence": [1.0, 0.8, 0.6, 0.4],
        "entropy": [0.0, 0.2, 0.4, 0.6],
        "conflict_count": [1, 2, 2, 3],
        "support_count": [10, 8, 6, 4],
        "top1_margin": [1.0, 0.6, 0.4, 0.2],
    })

    edge = FittedAmbiguityEdge(
        edge_id="edge_self_history",
        source_table="event_interest",
        lhs_columns=("lhs",),
        rhs_column="rhs",
        mapping=mapping,
        fit_start_time=None,
        fit_end_time="2026-01-03 10:00:00",
        maximum_source_time_used="2026-01-03 10:00:00",
        support=28,
        coverage=1.0,
        confidence=0.7,
        conflict_rate=0.75,
        selection_status="accepted",
        rejection_reason="",
        fold=0,
    )

    table = SimpleNamespace(
        df=source,
        time_col="timestamp",
        fkey_col_to_pkey_table={
            "user": "users",
        },
    )

    result, _, audits = materialize_ambiguity_features(
        fitted_edges=[edge],
        target_rows=target,
        source_tables={
            "event_interest": table,
        },
        task_metadata={
            "entity_key": "primary_key",
            "target_time_col": "timestamp",
        },
        source_entity_columns_by_edge={
            "edge_self_history": "user",
        },
        target_lookup_columns_by_edge={
            "edge_self_history": "user",
        },
        strict_before_by_edge={
            "edge_self_history": True,
        },
    )

    assert result["primary_key"].tolist() == [101, 102, 103]

    audit = audits[0]

    assert audit["target_entity_key"] == "primary_key"
    assert audit["target_lookup_entity_column"] == "user"
    assert audit["source_entity_column"] == "user"
    assert audit["allow_exact_matches"] is False
    assert audit["temporal_predicate"] == "<"
    assert audit["temporal_lookup_violation_count"] == 0

    residual_cols = [
        col
        for col in result.columns
        if col.startswith("f_fdhg")
        and not col.endswith("__is_missing")
    ]

    assert residual_cols

    row_103 = result[
        result["primary_key"] == 103
    ].iloc[0]

    assert all(
        pd.isna(row_103[col])
        for col in residual_cols
    )


def test_event_relation_preserves_dbinfer_inverse_lookup_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DBInfer mapped->raw lookup semantics must survive relation rebuilding."""

    from types import SimpleNamespace

    import fdhg.onboarding.auto_relbench as auto_relbench_module
    import fdhg.onboarding.relbench_v1 as relbench_v1_module

    train = pd.DataFrame({
        "id": [0, 1],
        "timestamp": pd.to_datetime([
            "2020-01-03",
            "2020-01-03",
        ]),
        "repeater": [0, 1],
    })

    validation = pd.DataFrame({
        "id": [2],
        "timestamp": pd.to_datetime([
            "2020-01-04",
        ]),
        "repeater": [0],
    })

    transaction = FakeRelBenchTable(
        pd.DataFrame({
            "id": [100, 200, 300],
            "date": pd.to_datetime([
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
            ]),
            "dept": ["a", "b", "c"],
        }),
        time_col="date",
    )

    table_dict = {
        "Transaction": transaction,
    }

    verified_relation = {
        "parent_table": "Customer",
        "parent_column": "id",
        "child_table": "Transaction",
        "child_column": "id",
        "child_event_time_col": "date",
        "target_lookup_column": "id",
        "relation_orientation":
            "dbinfer_shared_declared_fk",
        "strict_before": False,
        "verified": True,
        "target_lookup_value_transform":
            "dbinfer_inverse_entity_mapping",
    }

    resolved = SimpleNamespace(
        entity_key="id",
        relation_entity_key="id",
        target_time_col="timestamp",
        entity_table="History",
        label_col="repeater",
        metric_direction="higher",
        relation_selection_method="single_verified_relation",
        relation_selection_reason="only_verified_relation",
        provenance={
            "relation": "relbench_v1:event_row_fallback",
        },
        candidate_relations_considered=[
            verified_relation,
        ],
        relation_screening=[{
            **verified_relation,
            "selected": True,
            "mean_inner_fold_score": 0.6,
        }],
    )

    monkeypatch.setattr(
        relbench_v1_module,
        "resolve_relbench_task_metadata",
        lambda **_kwargs: resolved,
    )

    # This regression is about propagation of verified relation semantics,
    # not recomputing history coverage.
    monkeypatch.setattr(
        auto_relbench_module,
        "_history_coverage",
        lambda **_kwargs: 1.0,
    )

    relations, _, _, _ = (
        auto_relbench_module
        ._discover_event_row_relation_candidates(
            dataset_name="dbinfer-avs",
            task_name="repeater",
            dataset=object(),
            task=object(),
            database=object(),
            table_dict=table_dict,
            train_targets=train,
            validation_targets=validation,
            selection_folds=3,
        )
    )

    accepted = [
        row
        for row in relations
        if row["status"] == "accepted"
    ]

    assert len(accepted) == 1

    relation = accepted[0]

    assert relation["child_table"] == "Transaction"
    assert relation["child_fk"] == "id"
    assert relation["target_lookup_column"] == "id"
    assert (
        relation["target_lookup_value_transform"]
        == "dbinfer_inverse_entity_mapping"
    )
