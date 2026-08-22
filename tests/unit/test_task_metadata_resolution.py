from __future__ import annotations

from pathlib import Path

import pytest

from fdhg.onboarding import auto_relbench
from fdhg.onboarding.relbench_v1 import (
    METRIC_POLICY_VERSION,
    _candidate_relation_fingerprint,
    _schema_fingerprint,
    resolve_relbench_task_metadata,
    resolved_metadata_reusable,
)
from tests.unit.test_relbench_v1_export import fake_objects


def _objects(**kwargs):
    dataset, task, _ = fake_objects(public_metadata=True, **kwargs)
    return dataset, task, dataset.get_db()


def test_single_verified_relation_selected_without_screening(tmp_path: Path) -> None:
    dataset, task, database = _objects()

    resolved = resolve_relbench_task_metadata(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata={},
        selection_folds=3,
        output_dir=tmp_path,
    )

    assert resolved.child_table == "beer_ratings"
    assert resolved.relation_selection_method == "single_verified_relation"
    assert resolved.relation_screening == ()
    assert resolved.official_validation_used_for_resolution is False
    assert resolved.test_split_accessed is False
    assert (tmp_path / "resolved_task_metadata.json").exists()


def test_multiple_verified_relations_selected_by_higher_is_better(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, task, database = _objects(ambiguous_relation=True)
    scores = {"beer_ratings": 0.6, "other_events": 0.8}

    def fake_score(*, features, **kwargs):
        child = features[0]["child_table"]
        return {
            "score": scores[child],
            "stability": 0.0,
            "trials": [{"fold": 0, "score": scores[child]}],
        }

    monkeypatch.setattr(auto_relbench, "_score_feature_set", fake_score)

    resolved = resolve_relbench_task_metadata(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata={
            "problem_type": "binary_classification",
            "primary_metric": "accuracy",
        },
        selection_folds=3,
        output_dir=None,
    )

    assert resolved.metric_direction == "higher"
    assert resolved.child_table == "other_events"
    assert resolved.relation_selection_method == "train_inner_fold_screening"
    assert len(resolved.relation_screening) == 2


def test_multiple_verified_relations_selected_by_lower_is_better(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, task, database = _objects(ambiguous_relation=True)
    scores = {"beer_ratings": 0.8, "other_events": 0.6}

    def fake_score(*, features, **kwargs):
        child = features[0]["child_table"]
        return {
            "score": scores[child],
            "stability": 0.0,
            "trials": [{"fold": 0, "score": scores[child]}],
        }

    monkeypatch.setattr(auto_relbench, "_score_feature_set", fake_score)

    resolved = resolve_relbench_task_metadata(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata={
            "problem_type": "regression",
            "primary_metric": "rmse",
        },
        selection_folds=3,
        output_dir=None,
    )

    assert resolved.metric_direction == "lower"
    assert resolved.child_table == "other_events"


def test_tied_relation_scores_use_lexical_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, task, database = _objects(ambiguous_relation=True)

    def fake_score(*, features, **kwargs):
        return {
            "score": 1.0,
            "stability": 0.0,
            "trials": [{"fold": 0, "score": 1.0}],
        }

    monkeypatch.setattr(auto_relbench, "_score_feature_set", fake_score)

    resolved = resolve_relbench_task_metadata(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata={
            "problem_type": "regression",
            "primary_metric": "rmse",
        },
        selection_folds=3,
        output_dir=None,
    )

    assert resolved.child_table == "beer_ratings"


def test_explicit_partial_override_and_invalid_relation() -> None:
    dataset, task, database = _objects()
    resolved = resolve_relbench_task_metadata(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata={"primary_metric": "mae"},
        selection_folds=3,
        output_dir=None,
    )

    assert resolved.primary_metric == "mae"
    assert resolved.metric_direction == "lower"
    assert resolved.provenance["primary_metric"].startswith("explicit_metadata:")
    with pytest.raises(ValueError, match="invalid_explicit_relation"):
        resolve_relbench_task_metadata(
            dataset_name="rel-ratebeer",
            task_name="user-count",
            dataset=dataset,
            task=task,
            database=database,
            explicit_metadata={
                "child_table": "missing",
                "child_fk": "user_id",
                "child_event_time_col": "created_at",
            },
            selection_folds=3,
            output_dir=None,
        )


def test_sklearn_metric_aliases_resolve_by_problem_type_priority() -> None:
    dataset, task, database = _objects()
    task.task_type = "binary_classification"
    task.metrics = ["roc_auc_score", "accuracy_score"]

    resolved = resolve_relbench_task_metadata(
        dataset_name="rel-ratebeer",
        task_name="user-count",
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata={},
        selection_folds=3,
        output_dir=None,
    )

    assert resolved.primary_metric == "roc_auc"
    assert resolved.metric_direction == "higher"
    assert resolved.provenance["primary_metric"] == "official_metric_declaration"


def test_dbinfer_placeholder_test_timestamp_is_not_used_as_real_boundary() -> None:
    dataset, task, database = _objects()
    dataset.test_timestamp = __import__("pandas").Timestamp("1970-01-02")

    resolved = resolve_relbench_task_metadata(
        dataset_name="dbinfer-avs",
        task_name="repeater",
        dataset=dataset,
        task=task,
        database=database,
        explicit_metadata={
            "problem_type": "binary_classification",
            "primary_metric": "roc_auc",
        },
        selection_folds=3,
        output_dir=None,
    )

    assert resolved.primary_metric == "roc_auc"


def test_unknown_official_metric_blocks_informatively() -> None:
    dataset, task, database = fake_objects()
    database = dataset.get_db()
    task.task_type = "regression"
    task.metrics = ["made_up_metric"]

    with pytest.raises(ValueError) as exc:
        resolve_relbench_task_metadata(
            dataset_name="rel-ratebeer",
            task_name="user-count",
            dataset=dataset,
            task=task,
            database=database,
            explicit_metadata={},
            selection_folds=3,
            output_dir=None,
        )

    message = str(exc.value)
    assert "unknown_primary_metric" in message
    assert "made_up_metric" in message
    assert "inspected_task_attributes" in message


def test_resume_fingerprint_reuse_and_mismatch() -> None:
    dataset, _task, database = _objects(ambiguous_relation=True)
    table_dict = database.table_dict
    candidates = [
        {
            "parent_table": "users",
            "parent_column": "user_id",
            "child_table": "beer_ratings",
            "child_column": "user_id",
            "child_event_time_col": "created_at",
            "verified": True,
        }
    ]
    payload = {
        "dataset": "rel-ratebeer",
        "task": "user-count",
        "schema_fingerprint": _schema_fingerprint(table_dict),
        "train_split_fingerprint": "train",
        "selection_folds": 3,
        "metric_policy_version": METRIC_POLICY_VERSION,
        "candidate_relation_fingerprint": _candidate_relation_fingerprint(candidates),
    }

    assert resolved_metadata_reusable(
        payload,
        dataset_name="rel-ratebeer",
        task_name="user-count",
        schema_fingerprint=payload["schema_fingerprint"],
        train_split_fingerprint="train",
        selection_folds=3,
        candidate_relation_fingerprint=payload["candidate_relation_fingerprint"],
    )
    assert not resolved_metadata_reusable(
        payload,
        dataset_name="rel-ratebeer",
        task_name="user-count",
        schema_fingerprint=payload["schema_fingerprint"],
        train_split_fingerprint="changed",
        selection_folds=3,
        candidate_relation_fingerprint=payload["candidate_relation_fingerprint"],
    )
