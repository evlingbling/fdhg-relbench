from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materializer import (
    TaskCandidateMaterializationRequest,
    materialize_task_candidates,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates
from fdhg.compiler.programs import build_configured_candidates


def write_task_fixture(tmp_path: Path):
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    events = tmp_path / "events.parquet"
    source_cols = {
        "src_" + str(index): index
        for index in range(15)
    }
    base_train = {
        "user_id": "u1",
        "candidate_item_id": "i9",
        "timestamp": pd.Timestamp("2026-06-01"),
        "label": 1,
        **source_cols,
    }
    base_val = {
        "user_id": "u1",
        "candidate_item_id": "i8",
        "timestamp": pd.Timestamp("2026-06-03"),
        "label": 0,
        **source_cols,
    }
    pd.DataFrame([base_train]).to_parquet(train, index=False)
    pd.DataFrame([base_val]).to_parquet(val, index=False)
    pd.DataFrame([
        {
            "user_id": "u1",
            "item_id": "i1",
            "event_time": pd.Timestamp("2026-05-31"),
        }
    ]).to_parquet(events, index=False)
    reproduction = tmp_path / "tasks.yaml"
    semantics = tmp_path / "task_semantics.yaml"
    task_body = {
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
                "dataset": "rel-example",
                "task": "pairwise",
                "split": "train",
                "role": "target",
                "table": "target",
                "path": "train.parquet",
            },
            "validation_target": {
                "dataset": "rel-example",
                "task": "pairwise",
                "split": "validation",
                "role": "target",
                "table": "target",
                "path": "val.parquet",
            },
            "source_tables": {
                "events": {
                    "dataset": "rel-example",
                    "task": "pairwise",
                    "split": "train",
                    "role": "source",
                    "path": "events.parquet",
                }
            },
            "lowering_evidence": [],
        },
    }
    reproduction.write_text(
        yaml.safe_dump({"tasks": {"rel-example/pairwise": task_body}}),
        encoding="utf-8",
    )
    semantics.write_text(
        yaml.safe_dump({
            "rel-example/pairwise": {
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
    spec = load_task_spec(
        dataset="rel-example",
        task="pairwise",
        reproduction_config=reproduction,
        semantics_config=semantics,
    )
    compiled = build_candidate_program(spec)
    programs = build_default_candidates(compiled)
    baseline_ids = [
        primitive.primitive_id
        for primitive in compiled.candidate_primitives
        if primitive.family.value == "baseline"
    ]
    evidence = []
    for program in programs:
        for index, primitive_id in enumerate(
            item
            for item in program.primitive_ids
            if item in baseline_ids
        ):
            evidence.append({
                "dataset": "rel-example",
                "task": "pairwise",
                "program_id": program.program_id,
                "primitive_id": primitive_id,
                "source_table": "target",
                "source_column": f"src_{index}",
                "output_column": f"out_{program.program_id}_{index}",
                "status": "proven",
            })
    task_body["prepared_artifacts"]["lowering_evidence"] = evidence
    reproduction.write_text(
        yaml.safe_dump({"tasks": {"rel-example/pairwise": task_body}}),
        encoding="utf-8",
    )
    return reproduction, semantics


def request(
    tmp_path: Path,
    reproduction: Path,
    semantics: Path,
    **kwargs,
) -> TaskCandidateMaterializationRequest:
    return TaskCandidateMaterializationRequest(
        dataset="rel-example",
        task="pairwise",
        output_root=tmp_path / "outputs" / "e2e",
        reproduction_config=reproduction,
        semantics_config=semantics,
        **kwargs,
    )


def test_deterministic_candidate_ordering(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    report = materialize_task_candidates(
        request(tmp_path, reproduction, semantics, write=False)
    )

    assert [outcome.program_id for outcome in report.outcomes] == sorted(
        outcome.program_id for outcome in report.outcomes
    )


def test_all_candidate_dry_run_no_writes(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    report = materialize_task_candidates(
        request(tmp_path, reproduction, semantics, write=False)
    )

    assert report.dry_run
    assert any(outcome.status == "dry_run_ready" for outcome in report.outcomes)
    assert any(outcome.status == "blocked" for outcome in report.outcomes)
    assert not (tmp_path / "outputs").exists()


def test_program_filtering(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    report = materialize_task_candidates(
        request(
            tmp_path,
            reproduction,
            semantics,
            program_ids=("baseline",),
            write=False,
        )
    )

    assert [outcome.program_id for outcome in report.outcomes] == ["baseline"]


def test_unknown_program_id(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    with pytest.raises(ValueError, match="unknown"):
        materialize_task_candidates(
            request(
                tmp_path,
                reproduction,
                semantics,
                program_ids=("missing",),
            )
        )


def test_custom_candidate_with_declared_primitives_and_order(
    tmp_path: Path,
) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    raw["tasks"]["rel-example/pairwise"]["candidate_programs"] = [{
        "program_id": "baseline_corrected_canonical",
        "primitive_ids": [
            "baseline::count",
            "baseline::numeric_mean",
            "baseline::numeric_std",
            "baseline::numeric_max",
            "baseline::days_since_last",
        ],
    }]
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = load_task_spec(
        dataset="rel-example",
        task="pairwise",
        reproduction_config=reproduction,
        semantics_config=semantics,
    )
    compiled = build_candidate_program(spec)

    programs = build_configured_candidates(
        compiled,
        reproduction_config=reproduction,
        semantics_config=semantics,
    )

    custom = {
        program.program_id: program
        for program in programs
    }["baseline_corrected_canonical"]
    assert custom.primitive_ids == [
        "baseline::count",
        "baseline::numeric_mean",
        "baseline::numeric_std",
        "baseline::numeric_max",
        "baseline::days_since_last",
    ]


def test_custom_candidate_unknown_primitive_rejected(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    raw["tasks"]["rel-example/pairwise"]["candidate_programs"] = [{
        "program_id": "bad",
        "primitive_ids": ["baseline::count", "missing::primitive"],
    }]
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = load_task_spec(
        dataset="rel-example",
        task="pairwise",
        reproduction_config=reproduction,
        semantics_config=semantics,
    )
    compiled = build_candidate_program(spec)

    with pytest.raises(ValueError, match="unknown primitive IDs"):
        build_configured_candidates(
            compiled,
            reproduction_config=reproduction,
            semantics_config=semantics,
        )


def test_custom_candidate_duplicate_primitive_rejected(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    raw["tasks"]["rel-example/pairwise"]["candidate_programs"] = [{
        "program_id": "bad",
        "primitive_ids": ["baseline::count", "baseline::count"],
    }]
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = load_task_spec(
        dataset="rel-example",
        task="pairwise",
        reproduction_config=reproduction,
        semantics_config=semantics,
    )
    compiled = build_candidate_program(spec)

    with pytest.raises(ValueError, match="duplicate primitive IDs"):
        build_configured_candidates(
            compiled,
            reproduction_config=reproduction,
            semantics_config=semantics,
        )


def test_default_ratebeer_user_count_baseline_remains_15_primitives() -> None:
    spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-count",
        reproduction_config=Path("configs/reproduction/tasks.yaml"),
        semantics_config=Path("configs/reproduction/task_semantics.yaml"),
    )
    compiled = build_candidate_program(spec)
    baseline = {
        program.program_id: program
        for program in build_default_candidates(compiled)
    }["baseline"]

    assert baseline.primitive_ids == [
        "baseline::count",
        "baseline::numeric_mean",
        "baseline::numeric_std",
        "baseline::numeric_max",
        "baseline::days_since_last",
        "baseline::history::window_count_short",
        "baseline::history::window_count_aligned",
        "baseline::history::window_count_long",
        "baseline::history::past_unique_values",
        "baseline::history::past_unique_neighbors",
        "baseline::history::mean_group_size",
        "baseline::history::max_group_size",
        "baseline::history::incoming_event_count",
        "baseline::history::past_unique_sources",
        "baseline::history::incoming_event_count_long",
    ]


def test_smoke_candidate_materialization_not_blocked_for_missing_evidence(
    tmp_path: Path,
) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    raw["tasks"]["rel-example/pairwise"]["candidate_programs"] = [{
        "program_id": "baseline_corrected_canonical",
        "primitive_ids": [
            "baseline::count",
            "baseline::numeric_mean",
            "baseline::numeric_std",
            "baseline::numeric_max",
            "baseline::days_since_last",
        ],
    }]
    evidence = []
    for index, primitive_id in enumerate(
        raw["tasks"]["rel-example/pairwise"]["candidate_programs"][0][
            "primitive_ids"
        ]
    ):
        evidence.append({
            "dataset": "rel-example",
            "task": "pairwise",
            "program_id": "baseline_corrected_canonical",
            "primitive_id": primitive_id,
            "source_table": "target",
            "source_column": f"src_{index}",
            "output_column": f"out_custom_{index}",
            "status": "proven",
        })
    raw["tasks"]["rel-example/pairwise"]["prepared_artifacts"][
        "lowering_evidence"
    ].extend(evidence)
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")

    report = materialize_task_candidates(
        request(
            tmp_path,
            reproduction,
            semantics,
            program_ids=("baseline_corrected_canonical",),
            write=False,
        )
    )

    assert report.input_resolved
    assert report.outcomes[0].status == "dry_run_ready"
    assert report.outcomes[0].blockers == ()


def test_baseline_only_mode(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    report = materialize_task_candidates(
        request(tmp_path, reproduction, semantics, baseline_only=True)
    )

    assert [outcome.program_id for outcome in report.outcomes] == ["baseline"]


def test_per_candidate_failure_isolation(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    evidence = raw["tasks"]["rel-example/pairwise"]["prepared_artifacts"][
        "lowering_evidence"
    ]
    raw["tasks"]["rel-example/pairwise"]["prepared_artifacts"][
        "lowering_evidence"
    ] = [
        row
        for row in evidence
        if row["program_id"] == "baseline"
    ]
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")

    report = materialize_task_candidates(
        request(tmp_path, reproduction, semantics, write=True)
    )

    statuses = {outcome.program_id: outcome.status for outcome in report.outcomes}
    assert statuses["baseline"] == "published"
    assert any(status == "blocked" for status in statuses.values())


def test_successful_publication_of_multiple_candidates(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    report = materialize_task_candidates(
        request(
            tmp_path,
            reproduction,
            semantics,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
            write=True,
        )
    )

    assert report.published_count == 2
    assert all(outcome.output_dir.exists() for outcome in report.outcomes)
    temporal = next(
        outcome
        for outcome in report.outcomes
        if outcome.program_id == "baseline_plus_pair_left_temporal"
    )
    assert (
        temporal.output_dir / "temporal_safety_audit.csv"
    ).exists()


def test_valid_reuse(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    first = materialize_task_candidates(
        request(tmp_path, reproduction, semantics, program_ids=("baseline",), write=True)
    )
    assert first.published_count == 1

    second = materialize_task_candidates(
        request(tmp_path, reproduction, semantics, program_ids=("baseline",), write=True)
    )

    assert second.reused_count == 1


def test_overwrite_behavior(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    materialize_task_candidates(
        request(tmp_path, reproduction, semantics, program_ids=("baseline",), write=True)
    )
    stale = (
        tmp_path
        / "outputs"
        / "e2e"
        / "rel-example_pairwise"
        / "candidates"
        / "baseline"
        / "extra.txt"
    )
    stale.write_text("stale", encoding="utf-8")

    report = materialize_task_candidates(
        request(
            tmp_path,
            reproduction,
            semantics,
            program_ids=("baseline",),
            write=True,
            overwrite=True,
        )
    )

    assert report.published_count == 1
    assert not stale.exists()


def test_discovery_compatibility(tmp_path: Path) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)
    report = materialize_task_candidates(
        request(tmp_path, reproduction, semantics, program_ids=("baseline",), write=True)
    )
    sweep = _load_sweep_module()

    discovered = sweep.discover_materialized_candidates(
        task_output_root=report.outcomes[0].output_dir.parents[1],
        configured_candidates=[],
    )

    assert discovered == ["baseline"]


def test_ratebeer_unresolved_provenance_remains_blocked(
    tmp_path: Path,
) -> None:
    report = materialize_task_candidates(
        TaskCandidateMaterializationRequest(
            dataset="rel-ratebeer",
            task="user-place-liked_pairwise",
            output_root=tmp_path / "outputs",
            reproduction_config=Path("configs/reproduction/tasks.yaml"),
            semantics_config=Path("configs/reproduction/task_semantics.yaml"),
            baseline_only=True,
            write=False,
        )
    )

    assert not report.input_resolved
    assert report.blocked_count == 1
    assert "missing_prepared_artifacts_config" in report.input_blockers


def test_non_pairwise_task_regression(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    pd.DataFrame([
        {
            "entity_id": "e1",
            "timestamp": pd.Timestamp("2026-06-01"),
            "label": 1,
        }
    ]).to_parquet(train, index=False)
    pd.DataFrame([
        {
            "entity_id": "e2",
            "timestamp": pd.Timestamp("2026-06-02"),
            "label": 0,
        }
    ]).to_parquet(val, index=False)
    reproduction = tmp_path / "tasks.yaml"
    reproduction.write_text(
        yaml.safe_dump({
            "tasks": {
                "rel-example/single": {
                    "problem_type": "binary",
                    "label_col": "label",
                    "target": {
                        "entity_key": "entity_id",
                        "time_col": "timestamp",
                    },
                    "dfs": {},
                    "prepared_artifacts": {
                        "train_target": {
                            "dataset": "rel-example",
                            "task": "single",
                            "split": "train",
                            "role": "target",
                            "table": "target",
                            "path": "train.parquet",
                        },
                        "validation_target": {
                            "dataset": "rel-example",
                            "task": "single",
                            "split": "validation",
                            "role": "target",
                            "table": "target",
                            "path": "val.parquet",
                        },
                        "source_tables": {},
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    semantics = tmp_path / "task_semantics.yaml"
    semantics.write_text("rel-example/single: {}\n", encoding="utf-8")

    report = materialize_task_candidates(
        TaskCandidateMaterializationRequest(
            dataset="rel-example",
            task="single",
            output_root=tmp_path / "outputs",
            reproduction_config=reproduction,
            semantics_config=semantics,
            baseline_only=True,
            write=True,
        )
    )

    assert report.published_count == 0
    assert report.blocked_count == 1
    assert "missing_proven_lowering_evidence" in report.outcomes[0].blockers[0]


def _load_sweep_module():
    path = Path("scripts/experiments/run_candidate_program_sweep.py")
    spec = importlib.util.spec_from_file_location(
        "run_candidate_program_sweep",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
