from __future__ import annotations

import enum
import inspect
import json
from pathlib import Path

import pandas as pd
import yaml

from fdhg.cli.auto_onboard_relbench import main as cli_main
from fdhg.onboarding.auto_relbench import (
    AutoOnboardingOptions,
    _choose_metric,
    _materialize_relation_features,
    auto_onboard_relbench,
    build_candidate_matrix_cache,
    classify_source_columns,
    discover_relation_candidates,
    generate_candidate_features,
    make_inner_temporal_splits,
    prepare_auto_onboarding,
    resolve_task_metadata,
    select_features,
)


class TaskType(enum.Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary"


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
    task_type = TaskType.REGRESSION
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


def fake_objects(
    *,
    extra_relation: bool = False,
    low_coverage: bool = False,
    no_timestamp: bool = False,
    unsafe_relation: bool = False,
    useful_count: bool = False,
    useful_last_numeric: bool = False,
):
    drivers = pd.DataFrame({
        "driver_id": ["d1", "d2", "d3", "d4"],
        "driver_name": ["a", "b", "c", "d"],
    })
    results = pd.DataFrame({
        "result_id": range(1, 13),
        "driver_id": [
            "d1", "d1", "d1", "d2", "d2", "d2",
            "d3", "d3", "d3", "d4", "d4", "d4",
        ],
        "race_date": pd.to_datetime([
            "2019-01-01", "2019-02-01", "2019-03-01",
            "2019-01-01", "2019-02-01", "2019-03-01",
            "2019-01-01", "2019-02-01", "2019-03-01",
            "2019-01-01", "2019-02-01", "2019-03-01",
        ]),
        "grid": [5] * 12,
        "position_value": [1.0, 2.0, 2.0, 8.0, 7.0, 8.0, 3.0, 4.0, 3.0, 10.0, 11.0, 10.0],
        "harmful_noise": [9, 1, 8, 2, 7, 3, 6, 4, 5, 1, 9, 2],
        "status": ["ok", "ok", "dnf", "ok", "ok", "ok", "ok", None, "ok", "dnf", "dnf", "ok"],
        "url": ["http://x"] * 12,
        "is_rookie": [False, False, False, True, True, False, False, False, False, True, True, True],
        "notes": ["long free text " * 20] * 12,
    })
    if useful_count:
        rows = []
        for driver, n_events in {"d1": 8, "d2": 2, "d3": 6, "d4": 1}.items():
            for idx in range(n_events):
                rows.append({
                    "event_id": f"{driver}-{idx}",
                    "driver_id": driver,
                    "event_time": pd.Timestamp("2019-01-01") + pd.Timedelta(days=idx),
                    "amount": float(idx + 1),
                    "kind": "a" if idx % 2 else "b",
                })
        results = pd.DataFrame(rows)
        results["result_id"] = range(len(results))
        results["race_date"] = results["event_time"]
        results["position_value"] = results.groupby("driver_id").cumcount() + 1.0
    if useful_last_numeric:
        results["mostly_missing_numeric"] = [None] * 10 + [1.0, None]
        results["constant_numeric"] = 1.0
        signal = results.pop("position_value")
        results["position_value"] = signal
    train = pd.DataFrame({
        "driver_id": ["d1", "d2", "d3", "d4", "d1", "d2", "d3", "d4"],
        "timestamp": pd.to_datetime([
            "2019-02-15", "2019-02-15", "2019-02-15", "2019-02-15",
            "2019-03-15", "2019-03-15", "2019-03-15", "2019-03-15",
        ]),
        "position": [1.5, 7.5, 3.5, 10.5, 2.0, 8.0, 3.0, 10.0],
    })
    val = pd.DataFrame({
        "driver_id": ["d1", "d2", "d3", "d4"],
        "timestamp": pd.to_datetime(["2019-04-15"] * 4),
        "position": [2.0, 8.0, 3.0, 10.0],
    })
    fkeys = {"driver_id": "drivers"}
    if low_coverage:
        results.loc[:8, "driver_id"] = "missing"
    table_dict = {
        "drivers": FakeRelBenchTable(drivers, pkey_col="driver_id"),
        "results": FakeRelBenchTable(
            results.assign(position=results["position_value"]) if unsafe_relation else results,
            pkey_col="result_id",
            fkeys=fkeys,
            time_col=None if no_timestamp else "race_date",
        ),
    }
    if extra_relation:
        table_dict["practice"] = FakeRelBenchTable(
            results.rename(columns={"race_date": "session_date"}).copy(),
            pkey_col="practice_id",
            fkeys=fkeys,
            time_col="session_date",
        )
    return FakeDataset(table_dict), FakeTask(train, val), "0.fake"


def loader(**kwargs):
    def _load(dataset_name, task_name, download):
        assert dataset_name == "rel-f1"
        assert task_name == "driver-position"
        return fake_objects(**kwargs)

    return _load


def test_task_metadata_enum_metric_inference_and_override(tmp_path: Path) -> None:
    dataset, task, _ = fake_objects()
    train = task.get_table("train").df
    val = task.get_table("val").df

    metadata = resolve_task_metadata(
        dataset_name="rel-f1",
        task_name="driver-position",
        task=task,
        train_df=train,
        validation_df=val,
    )

    assert metadata["problem_type"] == "regression"
    assert metadata["primary_metric"] == "rmse"
    assert metadata["metric_direction"] == "lower"

    override = tmp_path / "metadata.yaml"
    override.write_text(
        yaml.safe_dump({"tasks": {"rel-f1/driver-position": {"primary_metric": "mae"}}}),
        encoding="utf-8",
    )
    overridden = resolve_task_metadata(
        dataset_name="rel-f1",
        task_name="driver-position",
        task=task,
        train_df=train,
        validation_df=val,
        task_metadata_config=override,
    )
    assert overridden["primary_metric"] == "mae"


def test_relation_discovery_accepts_and_rejects_candidates() -> None:
    dataset, task, _ = fake_objects(extra_relation=True)
    metadata = resolve_task_metadata(
        dataset_name="rel-f1",
        task_name="driver-position",
        task=task,
        train_df=task.get_table("train").df,
        validation_df=task.get_table("val").df,
    )
    rows = discover_relation_candidates(
        table_dict=dataset.get_db().table_dict,
        train_targets=task.get_table("train").df,
        metadata=metadata,
        threshold=0.8,
    )
    assert [row["status"] for row in rows].count("accepted") == 2

    low, low_task, _ = fake_objects(low_coverage=True)
    low_rows = discover_relation_candidates(
        table_dict=low.get_db().table_dict,
        train_targets=low_task.get_table("train").df,
        metadata=metadata,
        threshold=0.8,
    )
    assert "referential_coverage_below_threshold" in low_rows[0]["rejection_reasons"]

    no_time, no_time_task, _ = fake_objects(no_timestamp=True)
    no_time_rows = discover_relation_candidates(
        table_dict=no_time.get_db().table_dict,
        train_targets=no_time_task.get_table("train").df,
        metadata=metadata,
        threshold=0.8,
    )
    assert "missing_child_event_time" in no_time_rows[0]["rejection_reasons"]

    unsafe, unsafe_task, _ = fake_objects(unsafe_relation=True)
    unsafe_rows = discover_relation_candidates(
        table_dict=unsafe.get_db().table_dict,
        train_targets=unsafe_task.get_table("train").df,
        metadata=metadata,
        threshold=0.8,
    )
    assert unsafe_rows[0]["status"] == "accepted"
    assert unsafe_rows[0]["target_named_column_present"] is True


def test_semantic_safety_excludes_identifiers_and_retains_safe_columns() -> None:
    dataset, task, _ = fake_objects()
    metadata = resolve_task_metadata(
        dataset_name="rel-f1",
        task_name="driver-position",
        task=task,
        train_df=task.get_table("train").df,
        validation_df=task.get_table("val").df,
    )
    relations = discover_relation_candidates(
        table_dict=dataset.get_db().table_dict,
        train_targets=task.get_table("train").df,
        metadata=metadata,
        threshold=0.8,
    )
    audit = classify_source_columns(
        table_dict=dataset.get_db().table_dict,
        relations=[row for row in relations if row["status"] == "accepted"],
        metadata=metadata,
        options=AutoOnboardingOptions(),
    )
    by_col = {row["column"]: row for row in audit}
    assert by_col["result_id"]["semantic_type"] == "identifier"
    assert by_col["driver_id"]["semantic_type"] == "identifier"
    assert by_col["driver_id"]["reason"] == "relation_child_fk"
    assert by_col["race_date"]["semantic_type"] == "timestamp"
    assert by_col["url"]["semantic_type"] == "identifier"
    assert by_col["is_rookie"]["semantic_type"] == "boolean"
    assert by_col["notes"]["semantic_type"] == "free_text"
    assert by_col["grid"]["accepted"] is True
    assert by_col["status"]["accepted"] is True


def test_target_name_collision_excludes_column_not_relation() -> None:
    dataset, task, _ = fake_objects(unsafe_relation=True)
    metadata = resolve_task_metadata(
        dataset_name="rel-f1",
        task_name="driver-position",
        task=task,
        train_df=task.get_table("train").df,
        validation_df=task.get_table("val").df,
    )
    relations = discover_relation_candidates(
        table_dict=dataset.get_db().table_dict,
        train_targets=task.get_table("train").df,
        metadata=metadata,
        threshold=0.8,
    )
    assert relations[0]["status"] == "accepted"
    assert relations[0]["target_named_column_present"] is True
    audit = classify_source_columns(
        table_dict=dataset.get_db().table_dict,
        relations=[relations[0]],
        metadata=metadata,
        options=AutoOnboardingOptions(),
    )
    position = [row for row in audit if row["column"] == "position"][0]
    assert position["accepted"] is False
    assert "target_name_collision_excluded" in position["reason"]
    assert "prediction_window_overlap" in position["reason"]
    assert any(
        row["column"] == "position_value"
        and row["accepted"]
        and "historical_source_safe" in row["reason"]
        for row in audit
    )


def test_inner_split_is_chronological_and_expanding() -> None:
    _, task, _ = fake_objects()
    split = make_inner_temporal_splits(
        task.get_table("train").df,
        time_col="timestamp",
        requested_folds=3,
    )

    assert split["protocol"] in {"single_holdout", "expanding_window"}
    for fold in split["folds"]:
        assert fold["train_rows"] > 0
        assert fold["validation_rows"] > 0
        assert pd.Timestamp(fold["cutoff_timestamp"]) < pd.Timestamp(
            fold["validation_end_timestamp"]
        )


def test_multi_timestamp_expanding_temporal_folds() -> None:
    targets = pd.DataFrame({
        "driver_id": [f"d{idx % 3}" for idx in range(40)],
        "timestamp": pd.to_datetime("2019-01-01") + pd.to_timedelta(
            [idx // 2 for idx in range(40)],
            unit="D",
        ),
        "position": [float(idx % 5) for idx in range(40)],
    })
    split = make_inner_temporal_splits(
        targets,
        time_col="timestamp",
        requested_folds=3,
    )

    assert split["protocol"] == "expanding_window"
    assert len(split["folds"]) == 3
    assert all(fold["unique_validation_timestamps"] > 1 for fold in split["folds"])
    for fold in split["folds"]:
        assert pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])


def test_candidate_matrix_materialized_once_per_fold_relation(monkeypatch) -> None:
    prepared = prepare_auto_onboarding(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=Path("unused"),
        download=False,
        task_metadata_config=None,
        options=AutoOnboardingOptions(feature_budget=20, max_numeric_columns=3),
        include_selection=False,
        object_loader=loader(extra_relation=True),
    )
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return _materialize_relation_features(*args, **kwargs)

    monkeypatch.setattr(
        "fdhg.onboarding.auto_relbench._materialize_relation_features",
        counted,
    )
    selected = select_features(
        train_targets=prepared["train_df"],
        table_dict=prepared["table_dict"],
        metadata=prepared["metadata"],
        candidates=prepared["candidate_features"],
        split_plan=prepared["split_plan"],
        options=AutoOnboardingOptions(feature_budget=20, max_numeric_columns=3),
    )
    accepted_relations = len({
        row["child_table"]
        for row in prepared["candidate_features"]
        if row["kind"] != "static_entity"
    })
    folds = len(prepared["split_plan"]["folds"])
    assert calls["count"] == folds * accepted_relations * 2
    assert selected["workload"]["child_relation_scan_count"] == calls["count"]
    assert selected["workload"]["model_trial_count"] > calls["count"]


def test_useful_last_numeric_column_survives_ranking() -> None:
    prepared = prepare_auto_onboarding(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=Path("unused"),
        download=False,
        task_metadata_config=None,
        options=AutoOnboardingOptions(feature_budget=4, max_numeric_columns=2),
        include_selection=False,
        object_loader=loader(useful_last_numeric=True),
    )
    candidate_sources = {
        row["source_column"]
        for row in prepared["candidate_features"]
        if row["kind"] == "numeric"
    }
    assert "position_value" in candidate_sources


def test_feature_selection_is_deterministic_and_budgeted() -> None:
    prepared = prepare_auto_onboarding(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=Path("unused"),
        download=False,
        task_metadata_config=None,
        options=AutoOnboardingOptions(feature_budget=2, max_numeric_columns=2),
        include_selection=False,
        object_loader=loader(),
    )
    first = select_features(
        train_targets=prepared["train_df"],
        table_dict=prepared["table_dict"],
        metadata=prepared["metadata"],
        candidates=prepared["candidate_features"],
        split_plan=prepared["split_plan"],
        options=AutoOnboardingOptions(feature_budget=2, max_numeric_columns=2),
    )
    second = select_features(
        train_targets=prepared["train_df"],
        table_dict=prepared["table_dict"],
        metadata=prepared["metadata"],
        candidates=prepared["candidate_features"],
        split_plan=prepared["split_plan"],
        options=AutoOnboardingOptions(feature_budget=2, max_numeric_columns=2),
    )
    assert [row["feature_id"] for row in first["selected_features"]] == [
        row["feature_id"] for row in second["selected_features"]
    ]
    assert len(first["selected_features"]) <= 2


def test_safe_fallback_when_min_delta_prevents_selection() -> None:
    prepared = prepare_auto_onboarding(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=Path("unused"),
        download=False,
        task_metadata_config=None,
        options=AutoOnboardingOptions(feature_budget=2, min_delta=999.0),
        include_selection=False,
        object_loader=loader(),
    )
    selected = select_features(
        train_targets=prepared["train_df"],
        table_dict=prepared["table_dict"],
        metadata=prepared["metadata"],
        candidates=prepared["candidate_features"],
        split_plan=prepared["split_plan"],
        options=AutoOnboardingOptions(feature_budget=2, min_delta=999.0),
    )
    assert selected["fallback"] is True
    assert selected["selected_features"]


def test_static_entity_fallback_when_no_relations() -> None:
    dataset, task, version = fake_objects(low_coverage=True)
    drivers = dataset.get_db().table_dict["drivers"].df.copy()
    drivers["age"] = [20.0, 30.0, 40.0, 50.0]
    drivers["team"] = ["a", "b", "a", "b"]
    dataset.get_db().table_dict["drivers"] = FakeRelBenchTable(
        drivers,
        pkey_col="driver_id",
    )

    prepared = prepare_auto_onboarding(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=Path("unused"),
        download=False,
        task_metadata_config=None,
        options=AutoOnboardingOptions(feature_budget=2),
        include_selection=False,
        object_loader=lambda *_: (dataset, task, version),
    )
    selected = select_features(
        train_targets=prepared["train_df"],
        table_dict=prepared["table_dict"],
        metadata=prepared["metadata"],
        candidates=[],
        split_plan=prepared["split_plan"],
        options=AutoOnboardingOptions(feature_budget=2),
    )
    assert selected["fallback"] is True
    assert selected["fallback_level"] == "static_entity_features"
    assert all(row["kind"] == "static_entity" for row in selected["selected_features"])


class RaisingValidationFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return RaisingValidationFrame

    def copy(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        if key == "position":
            raise AssertionError("validation label values accessed")
        return super().__getitem__(key)


def test_dry_run_does_not_write_or_use_validation_labels(tmp_path: Path) -> None:
    dataset, task, version = fake_objects()
    task._tables["val"] = FakeRelBenchTable(RaisingValidationFrame(task._tables["val"].df))
    report = auto_onboard_relbench(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        write=False,
        object_loader=lambda *_: (dataset, task, version),
    )

    assert report.status == "dry_run_ready"
    assert report.candidate_features > 0
    assert not report.output_dir.exists()


def test_callable_metric_normalization() -> None:
    def r2(y_true, y_pred):
        return 0.0

    def mae(y_true, y_pred):
        return 0.0

    def rmse(y_true, y_pred):
        return 0.0

    def average_precision(y_true, y_pred):
        return 0.0

    def auroc(y_true, y_pred):
        return 0.0

    def macro_f1(y_true, y_pred):
        return 0.0

    def accuracy(y_true, y_pred):
        return 0.0

    assert _choose_metric("regression", [r2, mae, rmse]) == "rmse"
    assert _choose_metric("binary", [average_precision, auroc]) == "roc_auc"
    assert _choose_metric("multiclass", [macro_f1, accuracy]) == "accuracy"


def test_end_to_end_auto_write_outputs_manifest(tmp_path: Path) -> None:
    report = auto_onboard_relbench(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        write=True,
        object_loader=loader(),
        options=AutoOnboardingOptions(feature_budget=2),
    )

    assert report.status == "completed"
    manifest_path = report.output_dir / "auto_onboarding_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["test_split_accessed"] is False
    assert manifest["materialization_strategy"] == "grouped_temporal_sweep"
    assert (report.output_dir / "official_validation_predictions.parquet").exists()
    trials = pd.read_csv(report.output_dir / "selection_trials.csv")
    required = {
        "phase",
        "fold",
        "candidate_added_or_removed",
        "selected_feature_ids_before_trial",
        "selected_feature_ids_after_trial",
        "metric",
        "score",
        "mean_score",
        "stability",
        "improvement",
        "accepted_decision",
        "decision_reason",
    }
    assert required.issubset(trials.columns)
    selected = json.loads(
        (report.output_dir / "selected_features.json").read_text(encoding="utf-8")
    )
    if selected["selected_features"]:
        row = selected["selected_features"][0]
        assert "selection_origin" in row
        assert "relation_rank" in row
        assert "column_semantic_type" in row or row["kind"] == "relation"
        assert "inner_fold_scores" in row
        assert "backward_cleanup_result" in row


def test_ratebeer_like_and_f1_like_synthetic_behaviors() -> None:
    ratebeer = prepare_auto_onboarding(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=Path("unused"),
        download=False,
        task_metadata_config=None,
        options=AutoOnboardingOptions(feature_budget=3),
        include_selection=True,
        object_loader=loader(useful_count=True),
    )
    assert ratebeer["selection"]["selected_features"]

    f1 = prepare_auto_onboarding(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=Path("unused"),
        download=False,
        task_metadata_config=None,
        options=AutoOnboardingOptions(feature_budget=3),
        include_selection=True,
        object_loader=loader(),
    )
    names = {row["output_column"] for row in f1["selection"]["selected_features"]}
    assert any("position_value_mean" in name or "position_value_std" in name for name in names)
    assert not {"f_results_count", "f_results_days_since_last", "f_results_position_value_min", "f_results_position_value_max"}.issubset(names)


def test_cli_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    import fdhg.cli.auto_onboard_relbench as cli

    monkeypatch.setattr(cli, "auto_onboard_relbench", lambda **kwargs: auto_onboard_relbench(
        **kwargs,
        object_loader=loader(),
    ))
    code = cli_main([
        "--dataset", "rel-f1",
        "--task", "driver-position",
        "--output-root", str(tmp_path / "out"),
        "--dry-run",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "STATUS dry_run_ready" in out
    assert "TEST_SPLIT_ACCESSED False" in out
    assert "EXPECTED_CHILD_RELATION_SCANS" in out


def test_no_dataset_specific_preferred_column_list_and_no_test_split_access(tmp_path: Path) -> None:
    import fdhg.onboarding.auto_relbench as module

    source = inspect.getsource(module)
    for forbidden in ("positionOrder", "points", "laps", "grid"):
        assert forbidden not in source

    dataset, task, version = fake_objects()
    report = auto_onboard_relbench(
        dataset_name="rel-f1",
        task_name="driver-position",
        output_root=tmp_path / "out",
        write=False,
        object_loader=lambda *_: (dataset, task, version),
    )
    assert report.status == "dry_run_ready"
    assert task.calls == ["train", "val"]
