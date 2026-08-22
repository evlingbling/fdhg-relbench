from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import yaml

from fdhg.onboarding.pipeline import onboard_dataset
from fdhg.onboarding.relbench_v1 import (
    _load_relbench_objects,
    export_relbench_v1,
)


class FakeRelBenchTable:
    def __init__(
        self,
        df,
        *,
        pkey_col=None,
        fkeys=None,
        time_col=None,
    ) -> None:
        self.df = df
        self.pkey_col = pkey_col
        self.fkey_col_to_pkey_table = fkeys or {}
        self.time_col = time_col


class FakeDatabase:
    def __init__(self, table_dict) -> None:
        self.table_dict = table_dict


class FakeDataset:
    val_timestamp = pd.Timestamp("2018-09-01")
    test_timestamp = pd.Timestamp("2020-01-01")

    def __init__(self, db) -> None:
        self._db = db

    def get_db(self):
        return self._db


class FakeTask:
    entity_col = "user_id"
    time_col = "timestamp"
    target_col = "num_ratings"

    def __init__(self, train, val) -> None:
        self._tables = {
            "train": FakeRelBenchTable(train),
            "val": FakeRelBenchTable(val),
        }
        self.calls = []

    def get_table(self, split):
        self.calls.append(split)
        if split == "test":
            raise AssertionError("test split accessed")
        return self._tables[split]


class FullyPublicFakeTask(FakeTask):
    task_type = "regression"
    primary_metric = "rmse"
    metric_direction = "lower"


def fake_objects(**kwargs):
    users = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "joined_at": pd.to_datetime([
            "2017-01-01",
            "2017-01-02",
            "2017-01-03",
        ]),
    })
    ratings = pd.DataFrame({
        "rating_id": [1, 2, 3, 4],
        "user_id": ["u1", "u1", "u2", "u3"],
        "created_at": pd.to_datetime([
            "2017-01-05",
            "2017-01-10",
            "2017-02-01",
            "2018-10-01",
        ]),
        "aroma": [3.0, 4.0, 5.0, 1.0],
        "style": ["ipa", "ipa", "stout", "lager"],
    })
    train = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "timestamp": pd.to_datetime(["2018-01-01", "2018-01-01"]),
        "num_ratings": [2.0, 1.0],
    })
    val = pd.DataFrame({
        "user_id": ["u3"],
        "timestamp": pd.to_datetime(["2019-01-01"]),
        "num_ratings": [1.0],
    })
    if kwargs.get("missing_label"):
        train = train.drop(columns=["num_ratings"])
    if kwargs.get("missing_time"):
        val = val.drop(columns=["timestamp"])
    if kwargs.get("missing_entity"):
        train = train.drop(columns=["user_id"])
    if kwargs.get("overlap"):
        val = train.iloc[[0]].copy()
    if kwargs.get("schema_mismatch"):
        val["extra"] = 1
    fkeys = {} if kwargs.get("missing_relation") else {"user_id": "users"}
    if kwargs.get("ambiguous_relation"):
        extra = FakeRelBenchTable(
            ratings.copy(),
            pkey_col="other_id",
            fkeys={"user_id": "users"},
            time_col="created_at",
        )
        table_dict = {
            "users": FakeRelBenchTable(users, pkey_col="user_id"),
            "beer_ratings": FakeRelBenchTable(
                ratings,
                pkey_col="rating_id",
                fkeys=fkeys,
                time_col="created_at",
            ),
            "other_events": extra,
        }
    else:
        table_dict = {
            "users": FakeRelBenchTable(users, pkey_col="user_id"),
            "beer_ratings": FakeRelBenchTable(
                ratings,
                pkey_col="rating_id",
                fkeys=fkeys,
                time_col=None if kwargs.get("missing_child_time") else "created_at",
            ),
        }
    task_cls = FullyPublicFakeTask if kwargs.get("public_metadata") else FakeTask
    task = task_cls(train, val)
    return FakeDataset(FakeDatabase(table_dict)), task, "0.0.fake"


def write_metadata_config(
    tmp_path: Path,
    *,
    child_table: str = "beer_ratings",
    child_fk: str = "user_id",
    child_event_time_col: str = "created_at",
    entity_key: str = "user_id",
    target_time_col: str = "timestamp",
    label_col: str = "num_ratings",
    include_relation: bool = True,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "metadata.yaml"
    row = {
        "entity_key": entity_key,
        "target_time_col": target_time_col,
        "label_col": label_col,
        "problem_type": "regression",
        "primary_metric": "rmse",
        "metric_direction": "lower",
    }
    if include_relation:
        row.update({
            "child_table": child_table,
            "child_fk": child_fk,
            "child_event_time_col": child_event_time_col,
        })
    path.write_text(
        yaml.safe_dump({"tasks": {"rel-ratebeer/user-count": row}}),
        encoding="utf-8",
    )
    return path


def loader(**kwargs):
    def _load(dataset_name, task_name, download):
        assert dataset_name == "rel-ratebeer"
        assert task_name == "user-count"
        assert download is True
        return fake_objects(**kwargs)

    return _load


def test_relbench_datasets_and_tasks_adapter_path(monkeypatch) -> None:
    calls = {}
    relbench = types.ModuleType("relbench")
    datasets = types.ModuleType("relbench.datasets")
    tasks = types.ModuleType("relbench.tasks")

    def get_dataset(name, download):
        calls["dataset"] = (name, download)
        return "dataset"

    def get_task(dataset, task, download):
        calls["task"] = (dataset, task, download)
        return "task"

    datasets.get_dataset = get_dataset
    tasks.get_task = get_task
    monkeypatch.setitem(sys.modules, "relbench", relbench)
    monkeypatch.setitem(sys.modules, "relbench.datasets", datasets)
    monkeypatch.setitem(sys.modules, "relbench.tasks", tasks)
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.2.3")

    dataset, task, version = _load_relbench_objects(
        "rel-ratebeer",
        "user-count",
        True,
    )

    assert dataset == "dataset"
    assert task == "task"
    assert version == "1.2.3"
    assert calls == {
        "dataset": ("rel-ratebeer", True),
        "task": ("rel-ratebeer", "user-count", True),
    }


def test_database_and_official_targets_export(tmp_path: Path) -> None:
    metadata = write_metadata_config(tmp_path)
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=True,
        task_metadata_config=metadata,
        object_loader=loader(),
    )

    assert report.status == "completed"
    assert report.table_names == ("beer_ratings", "users")
    assert report.train_rows == 2
    assert report.validation_rows == 1
    assert (report.output_dir / "tables" / "users.parquet").exists()
    assert (report.output_dir / "target_train.parquet").exists()
    assert (report.output_dir / "target_validation.parquet").exists()
    manifest = json.loads(
        (report.output_dir / "export_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["database_table_metadata"]["users"]["primary_key"] == "user_id"
    assert manifest["database_table_metadata"]["beer_ratings"]["foreign_keys"] == {
        "user_id": "users"
    }
    assert manifest["database_table_metadata"]["beer_ratings"]["time_column"] == "created_at"
    assert manifest["test_split_accessed"] is False
    assert manifest["entity_key_source"].startswith("explicit_metadata:")
    assert manifest["primary_metric_source"].startswith("explicit_metadata:")
    assert manifest["metric_direction_source"].startswith("explicit_metadata:")
    assert manifest["relation_source"].startswith("explicit_metadata:")
    assert manifest["task_metadata_resolution_status"] == "completed"
    assert manifest["official_validation_used_for_resolution"] is False
    assert manifest["relation_candidates"]


def test_public_task_metadata_resolves_without_dataset_specific_branch(
    tmp_path: Path,
) -> None:
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=False,
        object_loader=loader(public_metadata=True),
    )

    assert report.status == "dry_run_ready"


def test_missing_metadata_blocks_without_explicit_config(tmp_path: Path) -> None:
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=False,
        object_loader=loader(),
    )

    assert report.status == "blocked"
    assert "missing_task_metadata:problem_type" in report.blockers


def test_metadata_conflicting_with_official_schema_blocks(tmp_path: Path) -> None:
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=False,
        task_metadata_config=write_metadata_config(
            tmp_path,
            target_time_col="wrong_time",
        ),
        object_loader=loader(),
    )

    assert report.status == "blocked"
    assert "missing_target_timestamp" in report.blockers


def test_no_test_method_call(tmp_path: Path) -> None:
    _, task, _ = fake_objects()

    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "config.yaml",
        download=True,
        write=False,
        task_metadata_config=write_metadata_config(tmp_path),
        object_loader=lambda *_: (FakeDataset(fake_objects()[0].get_db()), task, "0"),
    )

    assert report.status == "dry_run_ready"
    assert task.calls == ["train", "val"]


def test_fail_closed_blockers(tmp_path: Path) -> None:
    cases = [
        ("missing_label", "missing_label"),
        ("missing_time", "missing_target_timestamp"),
        ("missing_entity", "missing_entity_key"),
        ("overlap", "train_validation_overlap"),
        ("schema_mismatch", "incompatible_target_schemas"),
        ("missing_relation", "invalid_explicit_relation"),
        ("missing_child_time", "invalid_explicit_relation"),
    ]
    for flag, blocker in cases:
        report = export_relbench_v1(
            dataset_name="rel-ratebeer",
            task_name="user-count",
            output_root=tmp_path / flag / "data",
            config_output=tmp_path / flag / "config.yaml",
            download=True,
            write=False,
            task_metadata_config=write_metadata_config(tmp_path / flag),
            object_loader=loader(**{flag: True}),
        )
        assert report.status == "blocked"
        assert blocker in report.blockers


def test_one_verified_relation_is_automatically_selected(tmp_path: Path) -> None:
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=False,
        task_metadata_config=write_metadata_config(
            tmp_path,
            include_relation=False,
        ),
        object_loader=loader(),
    )

    assert report.status == "dry_run_ready"
    assert report.child_relation == "beer_ratings.user_id->users.user_id"


def test_multiple_verified_relations_screened_without_explicit_selection(
    tmp_path: Path,
) -> None:
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=False,
        task_metadata_config=write_metadata_config(
            tmp_path,
            include_relation=False,
        ),
        object_loader=loader(ambiguous_relation=True),
    )

    assert report.status == "dry_run_ready"
    assert report.child_relation in {
        "beer_ratings.user_id->users.user_id",
        "other_events.user_id->users.user_id",
    }


def test_explicit_selection_among_multiple_verified_relations_succeeds(
    tmp_path: Path,
) -> None:
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=False,
        task_metadata_config=write_metadata_config(tmp_path),
        object_loader=loader(ambiguous_relation=True),
    )

    assert report.status == "dry_run_ready"
    assert report.child_relation == "beer_ratings.user_id->users.user_id"


def test_invalid_explicit_relation_blocks(tmp_path: Path) -> None:
    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=False,
        task_metadata_config=write_metadata_config(
            tmp_path,
            child_table="not_a_table",
        ),
        object_loader=loader(),
    )

    assert report.status == "blocked"
    assert "invalid_explicit_relation" in report.blockers


def test_generated_config_supports_official_pre_split_onboarding(
    tmp_path: Path,
) -> None:
    metadata = write_metadata_config(tmp_path)
    export = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=True,
        task_metadata_config=metadata,
        object_loader=loader(),
    )

    onboarding = onboard_dataset(
        config_path=export.config_path,
        output_root=tmp_path / "onboarding",
        write=True,
    )

    assert onboarding.status == "completed"
    resolved = yaml.safe_load(
        (onboarding.output_dir / "resolved_task_spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    task = resolved["tasks"]["relbench-v1-rel-ratebeer/user-count"]
    assert task["evaluation"]["drop_cols"] == ["timestamp", "user_id"]
    assert task["evaluation"]["drop_reasons"] == {
        "timestamp": "target_prediction_time",
        "user_id": "target_entity_key",
    }
    split_manifest = json.loads(
        (onboarding.output_dir / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert split_manifest["strategy"] == "official_pre_split"
    assert split_manifest["test_split_accessed"] is False


def test_exact_rerun_reuses_export(tmp_path: Path) -> None:
    kwargs = dict(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=True,
        task_metadata_config=write_metadata_config(tmp_path),
        object_loader=loader(),
    )
    first = export_relbench_v1(**kwargs)
    second = export_relbench_v1(**kwargs)

    assert first.status == "completed"
    assert second.status == "reused"


def test_source_identity_change_invalidates_reuse(tmp_path: Path) -> None:
    kwargs = dict(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=True,
        task_metadata_config=write_metadata_config(tmp_path),
    )
    assert export_relbench_v1(**kwargs, object_loader=loader()).status == "completed"
    report = export_relbench_v1(
        **kwargs,
        object_loader=lambda *_: fake_objects(overlap=False),
    )

    assert report.status == "reused"

    def changed_loader(*_):
        dataset, task, version = fake_objects()
        dataset.get_db().table_dict["beer_ratings"].df.loc[0, "aroma"] = 9.0
        return dataset, task, version

    changed = export_relbench_v1(**kwargs, object_loader=changed_loader)
    assert changed.status == "blocked"
    assert "conflicting_existing_output_identity" in changed.blockers


def test_interrupted_staging_does_not_publish_partial_output(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "data" / "rel-ratebeer" / "_user-count.staging"
    staging.mkdir(parents=True)

    report = export_relbench_v1(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        output_root=tmp_path / "data",
        config_output=tmp_path / "configs" / "ratebeer.yaml",
        download=True,
        write=True,
        task_metadata_config=write_metadata_config(tmp_path),
        object_loader=loader(),
    )

    assert report.status == "blocked"
    assert "partial_staging_output" in report.blockers
    assert not (tmp_path / "data" / "rel-ratebeer" / "user-count").exists()
