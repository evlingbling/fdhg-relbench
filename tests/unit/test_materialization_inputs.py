from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materialization_inputs import (
    load_rows_for_materialization_plan,
    resolve_materialization_inputs,
)
from fdhg.compiler.materializer import plan_candidate_materialization
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import CandidateProgram


def write_fixture(
    tmp_path: Path,
    *,
    train_extra: dict | None = None,
    val_extra: dict | None = None,
    source_split: str = "train",
    validation_split: str = "validation",
    identity: tuple[str, str] = ("rel-example", "pairwise"),
):
    dataset, task = identity
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    events = tmp_path / "events.parquet"
    train_row = {
        "user_id": "u1",
        "candidate_item_id": "i9",
        "timestamp": pd.Timestamp("2026-06-01"),
        "label": 1,
        "baseline_count_src": 2,
    }
    train_row.update(train_extra or {})
    val_row = {
        "user_id": "u1",
        "candidate_item_id": "i8",
        "timestamp": pd.Timestamp("2026-06-02"),
        "label": 0,
        "baseline_count_src": 3,
    }
    val_row.update(val_extra or {})
    pd.DataFrame([train_row]).to_parquet(train, index=False)
    pd.DataFrame([val_row]).to_parquet(val, index=False)
    pd.DataFrame([
        {
            "user_id": "u1",
            "item_id": "i1",
            "event_time": pd.Timestamp("2026-05-30"),
            "unused_source_col": 99,
        }
    ]).to_parquet(events, index=False)
    reproduction = tmp_path / "tasks.yaml"
    semantics = tmp_path / "task_semantics.yaml"
    reproduction.write_text(
        yaml.safe_dump({
            "tasks": {
                f"{dataset}/{task}": {
                    "problem_type": "binary",
                    "label_col": "label",
                    "target": {
                        "entity_key": "user_id",
                        "time_col": "timestamp",
                    },
                    "dfs": {
                        "child_table": "events",
                        "child_time_col": "event_time",
                        "numeric_col": "item_id",
                    },
                    "prepared_artifacts": {
                        "train_target": {
                            "dataset": dataset,
                            "task": task,
                            "split": "train",
                            "role": "target",
                            "table": "target",
                            "path": "train.parquet",
                        },
                        "validation_target": {
                            "dataset": dataset,
                            "task": task,
                            "split": validation_split,
                            "role": "target",
                            "table": "target",
                            "path": "val.parquet",
                        },
                        "source_tables": {
                            "events": {
                                "dataset": dataset,
                                "task": task,
                                "split": source_split,
                                "role": "source",
                                "path": "events.parquet",
                            }
                        },
                        "lowering_evidence": [
                            {
                                "dataset": dataset,
                                "task": task,
                                "program_id": "baseline",
                                "primitive_id": "baseline::count",
                                "source_table": "target",
                                "source_column": "baseline_count_src",
                                "output_column": "baseline_count",
                                "status": "proven",
                            }
                        ],
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    semantics.write_text(
        yaml.safe_dump({
            f"{dataset}/{task}": {
                "horizon_days": 30,
                "pairwise": {
                    "left_key": "user_id",
                    "right_key": "item_id",
                    "target_right_key": "candidate_item_id",
                    "left_history": {
                        "table": "events",
                        "key": "user_id",
                        "related_col": "item_id",
                        "time_col": "event_time",
                    },
                },
            }
        }),
        encoding="utf-8",
    )
    return reproduction, semantics


def load_spec(reproduction: Path, semantics: Path):
    return load_task_spec(
        dataset="rel-example",
        task="pairwise",
        reproduction_config=reproduction,
        semantics_config=semantics,
    )


def test_task_aware_input_resolution(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(tmp_path)
    spec = load_spec(reproduction, semantics)

    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    assert report.resolved
    assert report.inputs is not None
    assert report.inputs.train_target.path.name == "train.parquet"
    assert report.inputs.source_artifacts[0].table_name == "events"
    assert "user_id" in report.inputs.target_entity_columns


def test_explicit_artifact_identity(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(
        tmp_path,
        identity=("rel-other", "pairwise"),
    )
    spec = load_task_spec(
        dataset="rel-other",
        task="pairwise",
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    assert report.resolved
    assert report.inputs.dataset == "rel-other"


def test_missing_artifact(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(tmp_path)
    (tmp_path / "train.parquet").unlink()
    spec = load_spec(reproduction, semantics)

    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    assert not report.resolved
    assert "train.parquet" in report.blockers[0]


def test_missing_required_column(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(tmp_path)
    pd.DataFrame([{
        "user_id": "u1",
        "timestamp": pd.Timestamp("2026-06-01"),
        "label": 1,
    }]).to_parquet(tmp_path / "train.parquet", index=False)
    spec = load_spec(reproduction, semantics)

    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    assert not report.resolved
    assert "candidate_item_id" in report.blockers[0]


def test_train_validation_schema_mismatch(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(
        tmp_path,
        val_extra={"validation_only": 1},
    )
    spec = load_spec(reproduction, semantics)

    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    assert not report.resolved
    assert "schema mismatch" in report.blockers[0]


def test_ambiguous_split_rejection(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(
        tmp_path,
        validation_split="holdout",
    )
    spec = load_spec(reproduction, semantics)

    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    assert not report.resolved
    assert "split" in report.blockers[0]


def test_test_artifact_rejection(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(
        tmp_path,
        validation_split="test",
    )
    spec = load_spec(reproduction, semantics)

    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    assert not report.resolved
    assert "test/final" in report.blockers[0]


def test_loads_only_required_columns(tmp_path: Path) -> None:
    reproduction, semantics = write_fixture(
        tmp_path,
        train_extra={"unused_target_col": "x"},
        val_extra={"unused_target_col": "y"},
    )
    spec = load_spec(reproduction, semantics)
    compiled = build_candidate_program(spec)
    program = CandidateProgram(
        program_id="baseline",
        primitive_ids=["baseline::count"],
        families=["baseline"],
        description="baseline",
    )
    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )
    report = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    sources, train, val = load_rows_for_materialization_plan(
        inputs=report.inputs,
        plan=plan,
        evidence=report.inputs.evidence_for_program("baseline"),
    )

    assert sources == {}
    assert "unused_target_col" not in train[0]
    assert "unused_target_col" not in val[0]
