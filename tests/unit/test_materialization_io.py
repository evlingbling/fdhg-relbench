from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.ir import (
    CompiledTask,
    PairwiseHistorySpec,
    PairwiseSpec,
    Primitive,
    PrimitiveFamily,
    TaskSpec,
)
from fdhg.compiler.materialization_io import (
    materialization_plan_to_dict,
    primitive_bindings_to_records,
    temporal_audit_to_csv_text,
    temporal_audit_to_records,
    write_materialization_plan_json,
    write_primitive_bindings_json,
    write_temporal_safety_audit_csv,
)
from fdhg.compiler.materializer import (
    LoweringMode,
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import (
    CandidateProgram,
    build_default_candidates,
)


def complete_pairwise_task() -> TaskSpec:
    return TaskSpec(
        dataset="synthetic",
        task="pairwise",
        problem_type="binary",
        label_col="label",
        entity_key="left_id",
        target_time_col="timestamp",
        horizon_days=30,
        pairwise=PairwiseSpec(
            left_key="left_id",
            right_key="right_id",
            target_right_key="candidate_right_id",
            left_history=PairwiseHistorySpec(
                table="events",
                key="left_id",
                related_col="item_id",
                time_col="event_time",
            ),
            right_history=PairwiseHistorySpec(
                table="events",
                key="right_id",
                related_col="left_id",
                time_col="event_time",
            ),
            pair_history=PairwiseHistorySpec(
                table="events",
                left_key="left_id",
                right_key="right_id",
                time_col="event_time",
            ),
        ),
    )


def pairwise_primitive(
    primitive_id: str,
    *,
    operation: str,
    role: str,
    window_days: int | None = None,
    predicate: str = (
        "events.event_time < target.timestamp"
    ),
) -> Primitive:
    return Primitive(
        primitive_id=primitive_id,
        family=PrimitiveFamily.TEMPORAL,
        operation=operation,
        source_table="events",
        group_key=(
            "left_id" if role != "pair" else None
        ),
        event_time_col="event_time",
        window_days=window_days,
        temporal_predicate=predicate,
        metadata={"pairwise_role": role},
    )


def baseline_primitive() -> Primitive:
    return Primitive(
        primitive_id="baseline::count",
        family=PrimitiveFamily.BASELINE,
        operation="count",
        source_table="events",
        group_key="left_id",
        event_time_col="event_time",
    )


def structural_primitive() -> Primitive:
    return Primitive(
        primitive_id=(
            "structural::afd::majority_confidence"
        ),
        family=PrimitiveFamily.STRUCTURAL,
        operation="majority_confidence",
        group_key="left_id",
    )


def unsupported_primitive() -> Primitive:
    return pairwise_primitive(
        "temporal::pairwise::left::mystery",
        operation="mystery",
        role="left",
    )


def synthetic_plan():
    primitives = [
        baseline_primitive(),
        structural_primitive(),
        pairwise_primitive(
            "temporal::pairwise::left::days_since_last",
            operation="days_since_last",
            role="left",
        ),
        pairwise_primitive(
            "temporal::pairwise::right::unique_neighbors::30d",
            operation="past_unique_neighbors",
            role="right",
            window_days=30,
        ),
        pairwise_primitive(
            "temporal::pairwise::pair::days_since_last",
            operation="pair_days_since_last",
            role="pair",
        ),
        unsupported_primitive(),
    ]
    compiled = CompiledTask(
        task_spec=complete_pairwise_task(),
        candidate_primitives=primitives,
    )
    program = CandidateProgram(
        program_id="synthetic_candidate",
        primitive_ids=[
            primitive.primitive_id
            for primitive in primitives
        ],
        families=["baseline", "structural", "temporal"],
        description="Synthetic materialization plan.",
    )
    return plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )


def ratebeer_pairwise_plan():
    spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=Path(
            "configs/reproduction/tasks.yaml"
        ),
        semantics_config=Path(
            "configs/reproduction/task_semantics.yaml"
        ),
    )
    compiled = build_candidate_program(spec)
    program = next(
        item
        for item in build_default_candidates(compiled)
        if item.program_id
        == "baseline_plus_pairwise_temporal"
    )
    return plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={
            "beer_ratings",
            "place_ratings",
        },
    )


def test_plan_dict_conversion_is_deterministic() -> None:
    plan = synthetic_plan()

    first = materialization_plan_to_dict(plan)
    second = materialization_plan_to_dict(plan)

    assert first == second
    assert first["program_id"] == "synthetic_candidate"
    assert first["step_count"] == len(plan.steps)
    assert len(first["steps"]) == len(plan.steps)


def test_lowering_mode_error_and_warning_counts() -> None:
    plan_dict = materialization_plan_to_dict(
        synthetic_plan()
    )

    assert plan_dict["lowering_mode_counts"] == {
        "generate": 3,
        "passthrough": 1,
        "external": 1,
        "unsupported": 1,
    }
    assert plan_dict["error_count"] == 1
    assert plan_dict["warning_count"] == 2


def test_step_schema_preserves_null_fields() -> None:
    step = materialization_plan_to_dict(
        synthetic_plan()
    )["steps"][0]

    assert "source_left_key" in step
    assert "source_right_key" in step
    assert "target_left_key" in step
    assert "target_right_key" in step
    assert step["source_left_key"] is None


def test_one_binding_record_per_generated_output_column() -> None:
    plan = synthetic_plan()
    records = primitive_bindings_to_records(plan)

    expected = sum(
        len(step.output_columns)
        for step in plan.steps
        if step.lowering_mode == LoweringMode.GENERATE
    )
    assert len(records) == expected
    assert [
        record["output_column"]
        for record in records[:2]
    ] == [
        "f_pairwise__left__days_since_last",
        "f_pairwise__left__days_since_last__is_missing",
    ]


def test_missingness_columns_are_separate_bindings() -> None:
    records = primitive_bindings_to_records(
        synthetic_plan()
    )
    columns = {
        record["output_column"] for record in records
    }

    assert (
        "f_pairwise__pair__days_since_last"
        in columns
    )
    assert (
        "f_pairwise__pair__days_since_last__is_missing"
        in columns
    )


def test_passthrough_external_and_unsupported_do_not_invent_columns() -> None:
    plan = synthetic_plan()
    records = primitive_bindings_to_records(plan)
    primitive_ids = {
        record["primitive_id"] for record in records
    }

    assert "baseline::count" not in primitive_ids
    assert (
        "structural::afd::majority_confidence"
        not in primitive_ids
    )
    assert (
        "temporal::pairwise::left::mystery"
        not in primitive_ids
    )


def test_external_steps_remain_in_plan() -> None:
    plan_dict = materialization_plan_to_dict(
        synthetic_plan()
    )
    structural = next(
        step
        for step in plan_dict["steps"]
        if step["primitive_id"]
        == "structural::afd::majority_confidence"
    )

    assert structural["lowering_mode"] == "external"
    assert structural["requires_external_provider"]


def test_temporal_audit_preserves_errors_and_warnings() -> None:
    records = temporal_audit_to_records(synthetic_plan())

    warning_record = next(
        record
        for record in records
        if record["primitive_id"] == "baseline::count"
    )
    error_record = next(
        record
        for record in records
        if record["primitive_id"]
        == "temporal::pairwise::left::mystery"
    )

    assert json.loads(warning_record["warnings"]) == [
        "baseline primitive is expected to be present "
        "in the base candidate artifact"
    ]
    assert json.loads(error_record["errors"]) == [
        "unsupported materialization operation: mystery"
    ]


def test_right_and_pair_records_preserve_physical_keys() -> None:
    records = primitive_bindings_to_records(
        synthetic_plan()
    )
    right = next(
        record
        for record in records
        if record["pairwise_role"] == "right"
    )
    pair = next(
        record
        for record in records
        if record["pairwise_role"] == "pair"
    )

    assert right["source_group_key"] == "right_id"
    assert right["target_key"] == "candidate_right_id"
    assert right["related_col"] == "left_id"
    assert pair["source_left_key"] == "left_id"
    assert pair["source_right_key"] == "right_id"
    assert pair["target_left_key"] == "left_id"
    assert pair["target_right_key"] == "candidate_right_id"


def test_repeated_json_serialization_is_byte_identical(tmp_path: Path) -> None:
    plan = synthetic_plan()
    first = tmp_path / "first" / "materialization_plan.json"
    second = tmp_path / "second" / "materialization_plan.json"
    metadata = {
        "dataset": "synthetic",
        "task": "pairwise",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "source": "unit-test",
    }

    write_materialization_plan_json(
        plan,
        first,
        metadata=metadata,
    )
    write_materialization_plan_json(
        plan,
        second,
        metadata=metadata,
    )

    assert first.read_bytes() == second.read_bytes()


def test_repeated_csv_serialization_is_byte_identical(tmp_path: Path) -> None:
    plan = synthetic_plan()
    first = tmp_path / "first" / "temporal_safety_audit.csv"
    second = tmp_path / "second" / "temporal_safety_audit.csv"

    write_temporal_safety_audit_csv(plan, first)
    write_temporal_safety_audit_csv(plan, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("\n")


def test_writer_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "materialization_plan.json"
    write_materialization_plan_json(
        synthetic_plan(),
        path,
    )

    with pytest.raises(FileExistsError):
        write_materialization_plan_json(
            synthetic_plan(),
            path,
        )


def test_explicit_overwrite_works(tmp_path: Path) -> None:
    path = tmp_path / "materialization_plan.json"
    write_materialization_plan_json(
        synthetic_plan(),
        path,
    )
    write_materialization_plan_json(
        synthetic_plan(),
        path,
        metadata={"source": "overwrite-test"},
        overwrite=True,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["source"] == "overwrite-test"


def test_parent_directories_created_only_by_writer(tmp_path: Path) -> None:
    plan = synthetic_plan()
    path = tmp_path / "nested" / "materialization_plan.json"

    materialization_plan_to_dict(plan)
    primitive_bindings_to_records(plan)
    temporal_audit_to_records(plan)
    assert not path.parent.exists()

    write_materialization_plan_json(plan, path)
    assert path.exists()


def test_failed_write_leaves_no_partial_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "materialization_plan.json"

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError):
        write_materialization_plan_json(
            synthetic_plan(),
            path,
        )

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_plan_construction_still_writes_no_files(tmp_path: Path) -> None:
    before = sorted(tmp_path.iterdir())
    synthetic_plan()
    after = sorted(tmp_path.iterdir())

    assert after == before


def test_bindings_writer_can_include_metadata(tmp_path: Path) -> None:
    path = tmp_path / "primitive_column_bindings.json"

    write_primitive_bindings_json(
        synthetic_plan(),
        path,
        metadata={
            "dataset": "synthetic",
            "task": "pairwise",
            "compiler_version": "test",
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["compiler_version"] == "test"
    assert len(payload["records"]) == len(
        primitive_bindings_to_records(synthetic_plan())
    )


def test_audit_csv_writer_can_include_metadata(tmp_path: Path) -> None:
    path = tmp_path / "temporal_safety_audit.csv"

    write_temporal_safety_audit_csv(
        synthetic_plan(),
        path,
        metadata={
            "dataset": "synthetic",
            "task": "pairwise",
            "source": "unit-test",
        },
    )

    text = path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    first_row = text.splitlines()[1]
    assert header.endswith(
        "dataset,task,compiler_version,git_commit,"
        "created_at_utc,source"
    )
    assert first_row.endswith("synthetic,pairwise,,,,unit-test")


def test_ratebeer_pairwise_plan_serializes_expected_summary() -> None:
    plan = ratebeer_pairwise_plan()
    plan_dict = materialization_plan_to_dict(plan)

    assert plan_dict["step_count"] == 29
    assert plan_dict["lowering_mode_counts"]["generate"] == 14
    assert plan_dict["lowering_mode_counts"]["passthrough"] == 15
    assert plan_dict["lowering_mode_counts"]["unsupported"] == 0
    assert plan_dict["materializable"]
    assert plan_dict["temporally_safe"]


def test_serializer_has_no_artifact_or_model_dependency() -> None:
    import fdhg.compiler.materialization_io as module

    names = set(module.__dict__)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names


def test_temporal_audit_csv_text_is_deterministic() -> None:
    plan = synthetic_plan()

    first = temporal_audit_to_csv_text(plan)
    second = temporal_audit_to_csv_text(plan)

    assert first == second
    assert "errors,warnings" in first.splitlines()[0]


def test_binding_record_count_matches_ratebeer_outputs() -> None:
    plan = ratebeer_pairwise_plan()
    records = primitive_bindings_to_records(plan)

    assert len(records) == sum(
        len(step.output_columns) for step in plan.steps
    )
    assert len(records) == 17


def test_file_hash_helper_input_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "temporal_safety_audit.csv"
    write_temporal_safety_audit_csv(
        synthetic_plan(),
        path,
    )

    first = hashlib.sha256(path.read_bytes()).hexdigest()
    second = hashlib.sha256(path.read_bytes()).hexdigest()

    assert first == second
