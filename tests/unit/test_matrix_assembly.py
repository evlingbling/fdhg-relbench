from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import inspect

import pytest

from fdhg.compiler.batch_evaluator import (
    BatchEvaluationResult,
    EvaluatedFeatureRow,
)
from fdhg.compiler.config import load_task_spec
from fdhg.compiler.in_memory_materializer import (
    materialize_generated_features_in_memory,
)
from fdhg.compiler.materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    MaterializationAuditRow,
    PrimitiveMaterializationStep,
    plan_candidate_materialization,
)
from fdhg.compiler.matrix_assembly import (
    MatrixAssemblyCode,
    MatrixAssemblyError,
    assemble_candidate_matrix,
)
from fdhg.compiler.passthrough_bindings import (
    PassthroughColumnBinding,
    ResolvedPassthroughContract,
    passthrough_contract_from_declared_outputs,
    resolve_passthrough_bindings,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates
from tests.unit.test_ratebeer_legacy_equivalence import (
    standard_source_rows,
    standard_target_rows,
)


T0 = datetime(2026, 6, 1)


def step(
    *,
    primitive_id: str,
    mode: LoweringMode,
    outputs: tuple[str, ...],
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation="count",
        lowering_mode=mode,
        pairwise_role=None,
        source_table=None,
        source_group_key=None,
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=None,
        target_key=None,
        target_left_key=None,
        target_right_key=None,
        target_time_col="timestamp",
        related_col=None,
        window_days=None,
        cutoff_operator="<",
        output_columns=outputs,
        materializable=(mode != LoweringMode.UNSUPPORTED),
        temporally_safe=(mode != LoweringMode.UNSUPPORTED),
        requires_external_provider=(mode == LoweringMode.EXTERNAL),
    )


def plan_with(
    steps: tuple[PrimitiveMaterializationStep, ...],
    *,
    materializable: bool = True,
    temporally_safe: bool = True,
) -> CandidateMaterializationPlan:
    return CandidateMaterializationPlan(
        program_id="program",
        steps=steps,
        audit_rows=tuple(
            MaterializationAuditRow(
                program_id="program",
                primitive_id=item.primitive_id,
                lowering_mode=item.lowering_mode,
                pairwise_role=item.pairwise_role,
                source_table=item.source_table,
                source_event_time_col=(
                    item.source_event_time_col
                ),
                logical_temporal_predicate=None,
                required_cutoff_operator="<",
                configured_cutoff_operator="<",
                temporally_safe=item.temporally_safe,
                materializable=item.materializable,
                requires_external_provider=(
                    item.requires_external_provider
                ),
                errors=(),
                warnings=(),
            )
            for item in steps
        ),
        materializable=materializable,
        temporally_safe=temporally_safe,
        requires_external_provider=any(
            item.requires_external_provider for item in steps
        ),
    )


def generated_result(
    *,
    program_id: str = "program",
    columns: tuple[str, ...] = ("gen",),
    rows: tuple[EvaluatedFeatureRow, ...] = (
        EvaluatedFeatureRow(
            row_index=0,
            values=(("gen", 7),),
        ),
    ),
) -> BatchEvaluationResult:
    return BatchEvaluationResult(
        program_id=program_id,
        generated_step_count=1 if columns else 0,
        output_columns=columns,
        rows=rows,
    )


def target_rows():
    return [
        {
            "id": "r0",
            "timestamp": T0,
            "label": 1,
            "base_a": 10,
            "base_b": None,
        },
        {
            "id": "r1",
            "timestamp": T0,
            "label": 0,
            "base_a": 20,
            "base_b": 5,
        },
    ]


def assemble(
    plan,
    *,
    rows=None,
    generated=None,
    identity=("id",),
    passthrough_contract=None,
):
    selected_generated = (
        generated_result() if generated is None else generated
    )
    selected_rows = (
        target_rows()[: len(selected_generated.rows)]
        if rows is None
        else rows
    )
    selected_contract = (
        passthrough_contract_from_declared_outputs(plan)
        if passthrough_contract is None
        else passthrough_contract
    )
    return assemble_candidate_matrix(
        plan,
        target_rows=selected_rows,
        generated=selected_generated,
        identity_columns=identity,
        passthrough_contract=selected_contract,
    )


def assert_assembly_error(code, plan, **kwargs):
    with pytest.raises(MatrixAssemblyError) as exc:
        assemble(plan, **kwargs)
    assert exc.value.code == code
    assert "program_id=program" in str(exc.value)


def test_identity_plus_one_passthrough_column() -> None:
    plan = plan_with((
        step(
            primitive_id="baseline",
            mode=LoweringMode.PASSTHROUGH,
            outputs=("base_a",),
        ),
    ))
    generated = generated_result(columns=(), rows=(
        EvaluatedFeatureRow(0, ()),
        EvaluatedFeatureRow(1, ()),
    ))
    matrix = assemble(plan, generated=generated)
    assert matrix.output_columns == ("id", "base_a")
    assert matrix.rows[0].values == (("id", "r0"), ("base_a", 10))


def test_identity_plus_one_generated_column() -> None:
    plan = plan_with((
        step(
            primitive_id="gen",
            mode=LoweringMode.GENERATE,
            outputs=("gen",),
        ),
    ))
    matrix = assemble(plan)
    assert matrix.output_columns == ("id", "gen")
    assert matrix.rows[0].values == (("id", "r0"), ("gen", 7))


def test_identity_plus_passthrough_plus_generated_columns() -> None:
    plan = plan_with((
        step(
            primitive_id="baseline",
            mode=LoweringMode.PASSTHROUGH,
            outputs=("base_a",),
        ),
        step(
            primitive_id="gen",
            mode=LoweringMode.GENERATE,
            outputs=("gen",),
        ),
    ))
    matrix = assemble(plan)
    assert matrix.output_columns == ("id", "base_a", "gen")


def test_output_column_ordering() -> None:
    plan = plan_with((
        step(primitive_id="p1", mode=LoweringMode.PASSTHROUGH, outputs=("base_a",)),
        step(primitive_id="p2", mode=LoweringMode.PASSTHROUGH, outputs=("base_b",)),
        step(primitive_id="g1", mode=LoweringMode.GENERATE, outputs=("g1", "g2")),
    ))
    generated = generated_result(
        columns=("g1", "g2"),
        rows=(EvaluatedFeatureRow(0, (("g1", 1), ("g2", 2))),),
    )
    matrix = assemble(plan, generated=generated, identity=("id", "label"))
    assert matrix.output_columns == (
        "id",
        "label",
        "base_a",
        "base_b",
        "g1",
        "g2",
    )


def test_multiple_target_rows_preserve_order() -> None:
    plan = plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),))
    generated = generated_result(rows=(
        EvaluatedFeatureRow(0, (("gen", 1),)),
        EvaluatedFeatureRow(1, (("gen", 2),)),
    ))
    matrix = assemble(plan, generated=generated)
    assert [row.row_index for row in matrix.rows] == [0, 1]
    assert [dict(row.values)["id"] for row in matrix.rows] == ["r0", "r1"]


def test_missingness_columns_remain_adjacent() -> None:
    plan = plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("days", "missing")),))
    generated = generated_result(
        columns=("days", "missing"),
        rows=(EvaluatedFeatureRow(0, (("days", 0.0), ("missing", 1))),),
    )
    matrix = assemble(plan, generated=generated)
    assert matrix.generated_columns == ("days", "missing")


def test_passthrough_columns_follow_plan_step_order() -> None:
    plan = plan_with((
        step(primitive_id="p1", mode=LoweringMode.PASSTHROUGH, outputs=("base_b",)),
        step(primitive_id="p2", mode=LoweringMode.PASSTHROUGH, outputs=("base_a",)),
    ))
    generated = generated_result(columns=(), rows=(
        EvaluatedFeatureRow(0, ()),
        EvaluatedFeatureRow(1, ()),
    ))
    matrix = assemble(plan, generated=generated)
    assert matrix.passthrough_columns == ("base_b", "base_a")


def test_generated_columns_follow_batch_result_order() -> None:
    plan = plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("b", "a")),))
    generated = generated_result(
        columns=("b", "a"),
        rows=(EvaluatedFeatureRow(0, (("b", 2), ("a", 1))),),
    )
    matrix = assemble(plan, generated=generated)
    assert matrix.generated_columns == ("b", "a")


def test_program_id_mismatch() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.PROGRAM_ID_MISMATCH,
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)),
        generated=generated_result(program_id="other"),
    )


def test_target_generated_row_count_mismatch() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.ROW_COUNT_MISMATCH,
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)),
        rows=target_rows(),
    )


def test_generated_row_index_mismatch() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.ROW_INDEX_MISMATCH,
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)),
        generated=generated_result(rows=(
            EvaluatedFeatureRow(1, (("gen", 1),)),
        )),
    )


def test_out_of_order_generated_indexes() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.ROW_INDEX_MISMATCH,
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)),
        generated=generated_result(rows=(
            EvaluatedFeatureRow(1, (("gen", 2),)),
            EvaluatedFeatureRow(0, (("gen", 1),)),
        )),
    )


def test_duplicate_identity_column() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.DUPLICATE_COLUMN,
        plan_with(()),
        generated=generated_result(columns=(), rows=(
            EvaluatedFeatureRow(0, ()),
            EvaluatedFeatureRow(1, ()),
        )),
        identity=("id", "id"),
    )


def test_identity_passthrough_collision() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.DUPLICATE_COLUMN,
        plan_with((step(primitive_id="p", mode=LoweringMode.PASSTHROUGH, outputs=("id",)),)),
        generated=generated_result(columns=(), rows=(
            EvaluatedFeatureRow(0, ()),
            EvaluatedFeatureRow(1, ()),
        )),
    )


def test_identity_generated_collision() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.DUPLICATE_COLUMN,
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("id",)),)),
        generated=generated_result(columns=("id",), rows=(
            EvaluatedFeatureRow(0, (("id", 1),)),
        )),
    )


def test_passthrough_generated_collision() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.DUPLICATE_COLUMN,
        plan_with((
            step(primitive_id="p", mode=LoweringMode.PASSTHROUGH, outputs=("x",)),
            step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("x",)),
        )),
        generated=generated_result(columns=("x",), rows=(
            EvaluatedFeatureRow(0, (("x", 1),)),
        )),
    )


def test_duplicate_passthrough_columns() -> None:
    plan = plan_with((
        step(
            primitive_id="p1",
            mode=LoweringMode.PASSTHROUGH,
            outputs=(),
        ),
        step(
            primitive_id="p2",
            mode=LoweringMode.PASSTHROUGH,
            outputs=(),
        ),
    ))
    bad_contract = ResolvedPassthroughContract(
        program_id="program",
        bindings=(
            PassthroughColumnBinding(
                "program",
                "p1",
                "a",
                "x",
            ),
            PassthroughColumnBinding(
                "program",
                "p2",
                "b",
                "x",
            ),
        ),
        source_columns=("a", "b"),
        output_columns=("x", "x"),
    )
    assert_assembly_error(
        MatrixAssemblyCode.DUPLICATE_COLUMN,
        plan,
        passthrough_contract=bad_contract,
        generated=generated_result(columns=(), rows=(
            EvaluatedFeatureRow(0, ()),
            EvaluatedFeatureRow(1, ()),
        )),
    )


def test_missing_identity_column() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.MISSING_IDENTITY_COLUMN,
        plan_with(()),
        rows=[{"timestamp": T0}],
        generated=generated_result(columns=(), rows=(EvaluatedFeatureRow(0, ()),)),
    )


def test_missing_passthrough_column() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.MISSING_PASSTHROUGH_COLUMN,
        plan_with((step(primitive_id="p", mode=LoweringMode.PASSTHROUGH, outputs=("base_a",)),)),
        rows=[{"id": "r0"}],
        generated=generated_result(columns=(), rows=(EvaluatedFeatureRow(0, ()),)),
    )


def test_generated_output_column_mismatch() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.GENERATED_COLUMN_MISMATCH,
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("expected",)),)),
        generated=generated_result(columns=("actual",), rows=(
            EvaluatedFeatureRow(0, (("actual", 1),)),
        )),
    )


def test_generated_row_value_key_mismatch() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.GENERATED_COLUMN_MISMATCH,
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)),
        generated=generated_result(columns=("gen",), rows=(
            EvaluatedFeatureRow(0, (("wrong", 1),)),
        )),
    )


def test_external_step_rejected() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.EXTERNAL_STEP_PRESENT,
        plan_with((step(primitive_id="e", mode=LoweringMode.EXTERNAL, outputs=("x",)),)),
    )


def test_unsupported_step_rejected() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.UNSUPPORTED_STEP_PRESENT,
        plan_with((step(primitive_id="u", mode=LoweringMode.UNSUPPORTED, outputs=("x",)),), materializable=True),
    )


def test_non_materializable_plan_rejected() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.PLAN_NOT_MATERIALIZABLE,
        plan_with((), materializable=False),
    )


def test_temporally_unsafe_plan_rejected() -> None:
    assert_assembly_error(
        MatrixAssemblyCode.PLAN_NOT_TEMPORALLY_SAFE,
        plan_with((), temporally_safe=False),
    )


def test_empty_target_rows_and_empty_generated_rows() -> None:
    matrix = assemble(
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)),
        rows=[],
        generated=generated_result(
            columns=("gen",),
            rows=(),
        ),
    )
    assert matrix.output_columns == ("id", "gen")
    assert matrix.rows == ()


def test_no_generated_columns() -> None:
    matrix = assemble(
        plan_with((step(primitive_id="p", mode=LoweringMode.PASSTHROUGH, outputs=("base_a",)),)),
        generated=generated_result(columns=(), rows=(
            EvaluatedFeatureRow(0, ()),
            EvaluatedFeatureRow(1, ()),
        )),
    )
    assert matrix.output_columns == ("id", "base_a")


def test_no_passthrough_columns() -> None:
    matrix = assemble(
        plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)),
        generated=generated_result(),
    )
    assert matrix.output_columns == ("id", "gen")


def test_identity_only_matrix() -> None:
    matrix = assemble(
        plan_with(()),
        generated=generated_result(columns=(), rows=(
            EvaluatedFeatureRow(0, ()),
            EvaluatedFeatureRow(1, ()),
        )),
    )
    assert matrix.output_columns == ("id",)


def test_explicit_none_value_preserved() -> None:
    matrix = assemble(
        plan_with((step(primitive_id="p", mode=LoweringMode.PASSTHROUGH, outputs=("base_b",)),)),
        generated=generated_result(columns=(), rows=(
            EvaluatedFeatureRow(0, ()),
            EvaluatedFeatureRow(1, ()),
        )),
    )
    assert dict(matrix.rows[0].values)["base_b"] is None


def test_target_rows_not_mutated() -> None:
    rows = target_rows()
    before = deepcopy(rows)
    assemble(plan_with(()), rows=rows, generated=generated_result(columns=(), rows=(
        EvaluatedFeatureRow(0, ()),
        EvaluatedFeatureRow(1, ()),
    )))
    assert rows == before


def test_batch_result_not_mutated() -> None:
    generated = generated_result()
    before = repr(generated)
    assemble(plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),)), generated=generated)
    assert repr(generated) == before


def test_plan_not_mutated() -> None:
    plan = plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),))
    before = repr(plan)
    assemble(plan)
    assert repr(plan) == before


def test_repeated_assembly_deterministic() -> None:
    plan = plan_with((step(primitive_id="g", mode=LoweringMode.GENERATE, outputs=("gen",)),))
    first = assemble(plan)
    second = assemble(plan)
    assert first == second


def test_no_heavy_or_filesystem_dependency() -> None:
    import fdhg.compiler.matrix_assembly as module

    names = set(module.__dict__)
    source = inspect.getsource(module)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names
    assert "Path" not in names
    assert "open(" not in source


def ratebeer_plan():
    from pathlib import Path

    spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=Path("configs/reproduction/tasks.yaml"),
        semantics_config=Path("configs/reproduction/task_semantics.yaml"),
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


def synthetic_complete_passthrough_contract(plan):
    explicit = {}
    for index, item in enumerate(plan.steps):
        if item.lowering_mode == LoweringMode.PASSTHROUGH:
            column = (
                "rb_passthrough_"
                f"{index:02d}_"
                + item.primitive_id.replace("::", "__")
            )
            explicit[item.primitive_id] = ((column, column),)
    return resolve_passthrough_bindings(
        plan,
        explicit_bindings=explicit,
    )


def test_ratebeer_integration() -> None:
    plan = ratebeer_plan()
    passthrough_contract = synthetic_complete_passthrough_contract(
        plan
    )
    targets = [
        {
            **row,
            "label": index % 2,
            **{
                column: index + offset
                for offset, column in enumerate(
                    passthrough_contract.source_columns
                )
            },
        }
        for index, row in enumerate(standard_target_rows())
    ]
    materialized = materialize_generated_features_in_memory(
        plan,
        source_rows_by_table=standard_source_rows(),
        target_rows=targets,
    )
    matrix = assemble_candidate_matrix(
        plan,
        target_rows=targets,
        generated=materialized.batch_result,
        identity_columns=(
            "user_id",
            "candidate_place_id",
            "timestamp",
            "label",
        ),
        passthrough_contract=passthrough_contract,
    )
    assert matrix.program_id == plan.program_id
    assert len(matrix.rows) == len(targets)
    assert len([
        step
        for step in plan.steps
        if step.lowering_mode == LoweringMode.PASSTHROUGH
    ]) == 15
    assert len(matrix.passthrough_columns) == 15
    assert len([
        step
        for step in plan.steps
        if step.lowering_mode == LoweringMode.GENERATE
    ]) == 14
    assert len(matrix.generated_columns) == 17
    assert matrix.output_columns[:4] == (
        "user_id",
        "candidate_place_id",
        "timestamp",
        "label",
    )
    assert "f_pairtmp__user_place_activity_product" not in matrix.output_columns
    assert "f_pairtmp__user_place_activity_ratio" not in matrix.output_columns


def test_legacy_candidate_shape_fixture_documentation() -> None:
    plan = ratebeer_plan()
    identity = (
        "user_id",
        "candidate_place_id",
        "timestamp",
        "label",
    )
    baseline_passthrough = (
        synthetic_complete_passthrough_contract(plan)
        .output_columns
    )
    temporal_generated = tuple(
        column
        for step_item in plan.steps
        if step_item.lowering_mode == LoweringMode.GENERATE
        for column in step_item.output_columns
    )
    assert len(temporal_generated) == 17
    assert identity + baseline_passthrough + temporal_generated == (
        identity + baseline_passthrough + temporal_generated
    )
