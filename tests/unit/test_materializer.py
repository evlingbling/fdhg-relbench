from __future__ import annotations

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
from fdhg.compiler.materializer import (
    LoweringMode,
    PhysicalHistoryBinding,
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import (
    CandidateProgram,
    build_default_candidates,
)


def pairwise_task(
    *,
    right_history: PairwiseHistorySpec | None = None,
    pair_history: PairwiseHistorySpec | None = None,
) -> TaskSpec:
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
            right_history=right_history,
            pair_history=pair_history,
        ),
    )


def complete_pairwise_task() -> TaskSpec:
    return pairwise_task(
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
    )


def pairwise_primitive(
    primitive_id: str,
    *,
    operation: str,
    role: str,
    window_days: int | None = None,
    predicate: str | None = None,
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
        temporal_predicate=(
            predicate
            if predicate is not None
            else "events.event_time < target.timestamp"
        ),
        metadata={"pairwise_role": role},
    )


def baseline_primitive(
    primitive_id: str = "baseline::count",
) -> Primitive:
    return Primitive(
        primitive_id=primitive_id,
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


def program_for(
    primitives: list[Primitive],
    *,
    program_id: str = "candidate",
) -> tuple[CompiledTask, CandidateProgram]:
    compiled = CompiledTask(
        task_spec=complete_pairwise_task(),
        candidate_primitives=primitives,
    )
    program = CandidateProgram(
        program_id=program_id,
        primitive_ids=[
            primitive.primitive_id
            for primitive in primitives
        ],
        families=["baseline", "structural", "temporal"],
        description="Synthetic candidate.",
    )
    return compiled, program


def by_id(plan, primitive_id: str):
    return next(
        step
        for step in plan.steps
        if step.primitive_id == primitive_id
    )


def test_complete_pairwise_program_has_mixed_modes() -> None:
    primitives = [
        baseline_primitive(),
        structural_primitive(),
        pairwise_primitive(
            "temporal::pairwise::left::count::30d",
            operation="window_count",
            role="left",
            window_days=30,
        ),
        pairwise_primitive(
            "temporal::pairwise::right::count::30d",
            operation="window_count",
            role="right",
            window_days=30,
        ),
        pairwise_primitive(
            "temporal::pairwise::pair::prior_count",
            operation="prior_pair_count",
            role="pair",
        ),
    ]
    compiled, program = program_for(primitives)

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    assert [step.lowering_mode for step in plan.steps] == [
        LoweringMode.PASSTHROUGH,
        LoweringMode.EXTERNAL,
        LoweringMode.GENERATE,
        LoweringMode.GENERATE,
        LoweringMode.GENERATE,
    ]
    assert plan.materializable
    assert plan.temporally_safe
    assert plan.requires_external_provider


def test_baseline_primitives_are_passthrough() -> None:
    compiled, program = program_for([
        baseline_primitive()
    ])

    plan = plan_candidate_materialization(
        compiled,
        program,
    )

    step = plan.steps[0]
    assert step.lowering_mode == LoweringMode.PASSTHROUGH
    assert step.materializable
    assert step.temporally_safe
    assert not step.requires_external_provider
    assert "base candidate artifact" in step.warnings[0]


def test_pairwise_temporal_primitives_are_generated() -> None:
    compiled, program = program_for([
        pairwise_primitive(
            "temporal::pairwise::left::unique_neighbors::30d",
            operation="past_unique_neighbors",
            role="left",
            window_days=30,
        ),
        pairwise_primitive(
            "temporal::pairwise::pair::days_since_last",
            operation="pair_days_since_last",
            role="pair",
        ),
    ])

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    assert {
        step.lowering_mode
        for step in plan.steps
    } == {LoweringMode.GENERATE}
    assert plan.steps[0].source_table == "events"
    assert plan.steps[0].source_group_key == "left_id"
    assert plan.steps[0].target_key == "left_id"
    assert plan.steps[0].related_col == "item_id"
    assert plan.steps[1].output_columns == (
        "f_pairwise__pair__days_since_last",
        "f_pairwise__pair__days_since_last__is_missing",
    )


def test_structural_primitives_are_external() -> None:
    compiled, program = program_for([
        structural_primitive()
    ])

    plan = plan_candidate_materialization(
        compiled,
        program,
    )

    step = plan.steps[0]
    assert step.lowering_mode == LoweringMode.EXTERNAL
    assert step.materializable
    assert step.temporally_safe
    assert step.requires_external_provider
    assert plan.requires_external_provider


def test_unknown_operations_are_unsupported() -> None:
    compiled, program = program_for([
        pairwise_primitive(
            "temporal::pairwise::left::mystery",
            operation="mystery",
            role="left",
        )
    ])

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    step = plan.steps[0]
    assert step.lowering_mode == LoweringMode.UNSUPPORTED
    assert not step.materializable
    assert not plan.materializable
    assert "unsupported materialization operation" in (
        step.errors[0]
    )


def test_missing_primitive_id_is_blocking() -> None:
    compiled = CompiledTask(
        task_spec=complete_pairwise_task(),
        candidate_primitives=(baseline_primitive(),),
    )
    program = CandidateProgram(
        program_id="missing-primitive",
        primitive_ids=["baseline::count", "missing::primitive"],
        families=["baseline", "temporal"],
        description="Candidate references a missing primitive.",
    )

    plan = plan_candidate_materialization(
        compiled,
        program,
    )

    missing_step = by_id(plan, "missing::primitive")
    assert missing_step.lowering_mode == LoweringMode.UNSUPPORTED
    assert not missing_step.materializable
    assert not plan.materializable
    assert "primitive is not present in compiled task" in (
        missing_step.errors[0]
    )


def test_duplicate_primitive_ids_are_rejected() -> None:
    duplicate = baseline_primitive()
    compiled = CompiledTask(
        task_spec=complete_pairwise_task(),
        candidate_primitives=(
            duplicate,
            baseline_primitive(duplicate.primitive_id),
        ),
    )
    program = CandidateProgram(
        program_id="duplicate-primitive",
        primitive_ids=[duplicate.primitive_id],
        families=["baseline"],
        description="Candidate with duplicate compiled primitive IDs.",
    )

    with pytest.raises(
        ValueError,
        match="duplicate primitive_id values",
    ):
        plan_candidate_materialization(compiled, program)


def test_empty_candidate_program_has_sensible_status() -> None:
    compiled = CompiledTask(
        task_spec=complete_pairwise_task(),
        candidate_primitives=(),
    )
    program = CandidateProgram(
        program_id="empty",
        primitive_ids=[],
        families=[],
        description="Empty candidate.",
    )

    plan = plan_candidate_materialization(
        compiled,
        program,
    )

    assert plan.program_id == "empty"
    assert plan.steps == ()
    assert plan.audit_rows == ()
    assert plan.materializable
    assert plan.temporally_safe
    assert not plan.requires_external_provider


def test_candidate_level_status_comes_from_steps() -> None:
    compiled, program = program_for([
        baseline_primitive(),
        structural_primitive(),
        pairwise_primitive(
            "temporal::pairwise::left::count::30d",
            operation="window_count",
            role="left",
            window_days=30,
        ),
    ])

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    assert plan.materializable
    assert plan.temporally_safe
    assert plan.requires_external_provider


def test_strict_temporal_mismatch_blocks_only_generated_step() -> None:
    primitives = [
        baseline_primitive(),
        structural_primitive(),
        pairwise_primitive(
            "temporal::pairwise::left::count::30d",
            operation="window_count",
            role="left",
            window_days=30,
            predicate=(
                "events.event_time <= target.timestamp"
            ),
        ),
    ]
    compiled, program = program_for(primitives)

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    baseline = by_id(plan, "baseline::count")
    structural = by_id(
        plan,
        "structural::afd::majority_confidence",
    )
    generated = by_id(
        plan,
        "temporal::pairwise::left::count::30d",
    )

    assert baseline.materializable
    assert structural.materializable
    assert not generated.materializable
    assert not generated.temporally_safe
    assert "non-strict" in " ".join(
        generated.errors
    )
    assert not plan.materializable
    assert not plan.temporally_safe


def test_missing_right_history_is_a_generated_step_error() -> None:
    task = pairwise_task(
        right_history=None,
        pair_history=PairwiseHistorySpec(
            table="events",
            left_key="left_id",
            right_key="right_id",
            time_col="event_time",
        ),
    )
    prim = pairwise_primitive(
        "temporal::pairwise::right::count::30d",
        operation="window_count",
        role="right",
        window_days=30,
    )
    compiled = CompiledTask(task, [prim])
    program = CandidateProgram(
        "candidate",
        [prim.primitive_id],
        ["temporal"],
        "Synthetic.",
    )

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    assert plan.steps[0].lowering_mode == LoweringMode.GENERATE
    assert not plan.steps[0].materializable
    assert "right history source table is missing" in (
        " ".join(plan.steps[0].errors)
    )


def test_missing_pair_history_source_keys_are_errors() -> None:
    task = pairwise_task(
        right_history=PairwiseHistorySpec(
            table="events",
            key="right_id",
            related_col="left_id",
            time_col="event_time",
        ),
        pair_history=PairwiseHistorySpec(
            table="events",
            time_col="event_time",
        ),
    )
    prim = pairwise_primitive(
        "temporal::pairwise::pair::prior_count",
        operation="prior_pair_count",
        role="pair",
    )
    compiled = CompiledTask(task, [prim])
    program = CandidateProgram(
        "candidate",
        [prim.primitive_id],
        ["temporal"],
        "Synthetic.",
    )

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    errors = " ".join(plan.steps[0].errors)
    assert "source left key is missing" in errors
    assert "source right key is missing" in errors


def test_non_pairwise_candidate_planning_is_explicit() -> None:
    task = TaskSpec(
        dataset="synthetic",
        task="single",
        problem_type="binary",
        label_col="label",
        entity_key="entity_id",
        target_time_col="timestamp",
    )
    baseline = baseline_primitive()
    pairwise = pairwise_primitive(
        "temporal::pairwise::left::count::30d",
        operation="window_count",
        role="left",
        window_days=30,
    )
    compiled = CompiledTask(task, [baseline, pairwise])
    program = CandidateProgram(
        "candidate",
        [baseline.primitive_id, pairwise.primitive_id],
        ["baseline", "temporal"],
        "Synthetic.",
    )

    plan = plan_candidate_materialization(
        compiled,
        program,
    )

    assert plan.steps[0].lowering_mode == (
        LoweringMode.PASSTHROUGH
    )
    assert plan.steps[1].lowering_mode == (
        LoweringMode.UNSUPPORTED
    )
    assert "TaskSpec.pairwise" in " ".join(
        plan.steps[1].errors
    )
    assert not plan.materializable


def test_plan_construction_writes_no_files(tmp_path: Path) -> None:
    compiled, program = program_for([
        baseline_primitive(),
        pairwise_primitive(
            "temporal::pairwise::left::count::30d",
            operation="window_count",
            role="left",
            window_days=30,
        ),
    ])

    before = sorted(tmp_path.iterdir())

    plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )

    after = sorted(tmp_path.iterdir())
    assert after == before


def test_ratebeer_reference_binding_matches_corrected_semantics() -> None:
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
    reference = {
        "right": PhysicalHistoryBinding(
            role="right",
            source_table="place_ratings",
            source_group_key="place_id",
            source_left_key=None,
            source_right_key=None,
            source_event_time_col="created_at",
            target_key="candidate_place_id",
            target_left_key=None,
            target_right_key=None,
            target_time_col="timestamp",
            related_col="user_id",
        ),
        "pair": PhysicalHistoryBinding(
            role="pair",
            source_table="place_ratings",
            source_group_key=None,
            source_left_key="user_id",
            source_right_key="place_id",
            source_event_time_col="created_at",
            target_key=None,
            target_left_key="user_id",
            target_right_key="candidate_place_id",
            target_time_col="timestamp",
        ),
    }

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={
            "beer_ratings",
            "place_ratings",
        },
        reference_bindings=reference,
    )

    warnings = "\n".join(
        " ".join(row.warnings)
        for row in plan.audit_rows
    )
    errors = "\n".join(
        " ".join(row.errors)
        for row in plan.audit_rows
    )

    assert "reference binding mismatch" not in warnings
    assert "non-strict" not in errors


def test_complete_ratebeer_candidate_is_not_all_unsupported() -> None:
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

    plan = plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={
            "beer_ratings",
            "place_ratings",
        },
    )

    modes = {step.lowering_mode for step in plan.steps}
    assert LoweringMode.PASSTHROUGH in modes
    assert LoweringMode.GENERATE in modes
    assert LoweringMode.UNSUPPORTED not in modes
    assert plan.materializable
    assert plan.temporally_safe
    assert sum(
        step.lowering_mode == LoweringMode.GENERATE
        for step in plan.steps
    ) > 0
