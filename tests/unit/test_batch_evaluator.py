from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from fdhg.compiler.batch_evaluator import (
    BatchEvaluationCode,
    BatchEvaluationError,
    BatchEvaluationResult,
    evaluate_generated_plan_rows,
)
from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    MaterializationAuditRow,
    PrimitiveMaterializationStep,
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.primitive_evaluator import (
    PairKeyIndexKey,
    SingleKeyIndexKey,
    TemporalIndexRegistry,
)
from fdhg.compiler.programs import build_default_candidates
from fdhg.compiler.temporal_index import (
    PairKeyEvent,
    PairKeyTemporalIndex,
    SingleKeyEvent,
    SingleKeyTemporalIndex,
)


T0 = datetime(2026, 6, 1)


def single_step(
    *,
    primitive_id: str = "primitive::left::count",
    operation: str = "window_count",
    role: str = "left",
    outputs: tuple[str, ...] = ("count",),
    window_days: int | None = 30,
    lowering_mode: LoweringMode = LoweringMode.GENERATE,
) -> PrimitiveMaterializationStep:
    is_right = role == "right"
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation=operation,
        lowering_mode=lowering_mode,
        pairwise_role=role,
        source_table="events",
        source_group_key="right_id" if is_right else "left_id",
        source_left_key=None,
        source_right_key=None,
        source_event_time_col="event_time",
        target_key=(
            "candidate_right_id" if is_right else "left_id"
        ),
        target_left_key=None,
        target_right_key=None,
        target_time_col="timestamp",
        related_col="neighbor_id",
        window_days=window_days,
        cutoff_operator="<",
        output_columns=outputs,
        materializable=(
            lowering_mode != LoweringMode.UNSUPPORTED
        ),
        temporally_safe=(
            lowering_mode != LoweringMode.UNSUPPORTED
        ),
        requires_external_provider=(
            lowering_mode == LoweringMode.EXTERNAL
        ),
    )


def pair_step(
    *,
    primitive_id: str = "primitive::pair::prior_count",
    operation: str = "prior_pair_count",
    outputs: tuple[str, ...] = ("prior_count",),
    lowering_mode: LoweringMode = LoweringMode.GENERATE,
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation=operation,
        lowering_mode=lowering_mode,
        pairwise_role="pair",
        source_table="events",
        source_group_key=None,
        source_left_key="left_id",
        source_right_key="right_id",
        source_event_time_col="event_time",
        target_key=None,
        target_left_key="left_id",
        target_right_key="candidate_right_id",
        target_time_col="timestamp",
        related_col=None,
        window_days=None,
        cutoff_operator="<",
        output_columns=outputs,
        materializable=(
            lowering_mode != LoweringMode.UNSUPPORTED
        ),
        temporally_safe=(
            lowering_mode != LoweringMode.UNSUPPORTED
        ),
        requires_external_provider=(
            lowering_mode == LoweringMode.EXTERNAL
        ),
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
                primitive_id=step.primitive_id,
                lowering_mode=step.lowering_mode,
                pairwise_role=step.pairwise_role,
                source_table=step.source_table,
                source_event_time_col=(
                    step.source_event_time_col
                ),
                logical_temporal_predicate=None,
                required_cutoff_operator="<",
                configured_cutoff_operator="<",
                temporally_safe=step.temporally_safe,
                materializable=step.materializable,
                requires_external_provider=(
                    step.requires_external_provider
                ),
                errors=(),
                warnings=(),
            )
            for step in steps
        ),
        materializable=materializable,
        temporally_safe=temporally_safe,
        requires_external_provider=any(
            step.requires_external_provider for step in steps
        ),
    )


def registry() -> TemporalIndexRegistry:
    single_index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0 - timedelta(days=30), "a"),
        SingleKeyEvent("u1", T0 - timedelta(days=10), "a"),
        SingleKeyEvent("u1", T0 - timedelta(days=1), "b"),
        SingleKeyEvent("u1", T0, "at_target"),
        SingleKeyEvent("u2", T0 - timedelta(hours=12), "z"),
        SingleKeyEvent("p1", T0 - timedelta(days=30), "u1"),
        SingleKeyEvent("p1", T0 - timedelta(days=5), "u2"),
        SingleKeyEvent("p1", T0, "u3"),
    ])
    pair_index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", T0 - timedelta(days=60)),
        PairKeyEvent("u1", "p1", T0 - timedelta(days=3)),
        PairKeyEvent("u1", "p1", T0 - timedelta(days=3)),
        PairKeyEvent("u1", "p1", T0),
    ])
    return TemporalIndexRegistry(
        single_key_indexes={
            SingleKeyIndexKey(
                "events",
                "left_id",
                "event_time",
                "neighbor_id",
            ): single_index,
            SingleKeyIndexKey(
                "events",
                "right_id",
                "event_time",
                "neighbor_id",
            ): single_index,
        },
        pair_key_indexes={
            PairKeyIndexKey(
                "events",
                "left_id",
                "right_id",
                "event_time",
            ): pair_index
        },
    )


def target_rows():
    return [
        {
            "left_id": "u1",
            "candidate_right_id": "p1",
            "timestamp": T0,
        },
        {
            "left_id": "u2",
            "candidate_right_id": "missing",
            "timestamp": T0,
        },
    ]


def row_dict(result: BatchEvaluationResult, row_index: int):
    return dict(result.rows[row_index].values)


def test_one_row_and_one_generated_step() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((single_step(),)),
        target_rows=target_rows()[:1],
        indexes=registry(),
    )

    assert result.generated_step_count == 1
    assert result.output_columns == ("count",)
    assert row_dict(result, 0) == {"count": 3}


def test_multiple_rows_preserve_row_order() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((single_step(),)),
        target_rows=target_rows(),
        indexes=registry(),
    )

    assert [row.row_index for row in result.rows] == [0, 1]
    assert [dict(row.values)["count"] for row in result.rows] == [3, 1]


def test_multiple_generated_steps_preserve_step_order() -> None:
    plan = plan_with((
        single_step(primitive_id="first", outputs=("first",)),
        pair_step(primitive_id="second", outputs=("second",)),
    ))

    result = evaluate_generated_plan_rows(
        plan,
        target_rows=target_rows()[:1],
        indexes=registry(),
    )

    assert result.output_columns == ("first", "second")
    assert list(row_dict(result, 0)) == ["first", "second"]


def test_output_column_order_follows_plan_order() -> None:
    plan = plan_with((
        pair_step(
            operation="pair_days_since_last",
            outputs=("pair_days", "pair_missing"),
        ),
        single_step(outputs=("count",)),
    ))

    result = evaluate_generated_plan_rows(
        plan,
        target_rows=target_rows()[:1],
        indexes=registry(),
    )

    assert result.output_columns == (
        "pair_days",
        "pair_missing",
        "count",
    )


def test_missingness_columns_remain_adjacent() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((
            single_step(
                operation="days_since_last",
                outputs=("days", "missing"),
                window_days=None,
            ),
        )),
        target_rows=target_rows()[:1],
        indexes=registry(),
    )

    assert result.output_columns == ("days", "missing")
    assert result.rows[0].values == (
        ("days", 1.0),
        ("missing", 0),
    )


def test_passthrough_steps_produce_no_columns() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((
            single_step(
                lowering_mode=LoweringMode.PASSTHROUGH,
                outputs=(),
            ),
        )),
        target_rows=target_rows(),
        indexes=registry(),
    )

    assert result.generated_step_count == 0
    assert result.output_columns == ()
    assert result.rows[0].values == ()


def test_external_steps_are_rejected() -> None:
    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((
                single_step(
                    lowering_mode=LoweringMode.EXTERNAL,
                ),
            )),
            target_rows=target_rows(),
            indexes=registry(),
        )
    assert exc.value.code == BatchEvaluationCode.EXTERNAL_STEP_PRESENT


def test_unsupported_steps_are_rejected() -> None:
    step = single_step(
        lowering_mode=LoweringMode.UNSUPPORTED,
    )
    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((step,), materializable=True),
            target_rows=target_rows(),
            indexes=registry(),
        )
    assert exc.value.code == (
        BatchEvaluationCode.UNSUPPORTED_STEP_PRESENT
    )


def test_non_materializable_plan_rejected() -> None:
    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((single_step(),), materializable=False),
            target_rows=target_rows(),
            indexes=registry(),
        )
    assert exc.value.code == (
        BatchEvaluationCode.PLAN_NOT_MATERIALIZABLE
    )


def test_temporally_unsafe_plan_rejected() -> None:
    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((single_step(),), temporally_safe=False),
            target_rows=target_rows(),
            indexes=registry(),
        )
    assert exc.value.code == (
        BatchEvaluationCode.PLAN_NOT_TEMPORALLY_SAFE
    )


def test_duplicate_output_columns_rejected() -> None:
    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((
                single_step(outputs=("dup",)),
                pair_step(outputs=("dup",)),
            )),
            target_rows=target_rows(),
            indexes=registry(),
        )
    assert exc.value.code == (
        BatchEvaluationCode.DUPLICATE_OUTPUT_COLUMN
    )


def test_generated_step_with_no_outputs_rejected() -> None:
    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((single_step(outputs=()),)),
            target_rows=target_rows(),
            indexes=registry(),
        )
    assert exc.value.code == (
        BatchEvaluationCode.EMPTY_GENERATED_OUTPUT
    )


def test_scalar_output_mismatch_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatched(*args, **kwargs):
        return {"wrong": 1}

    monkeypatch.setattr(
        "fdhg.compiler.batch_evaluator.evaluate_generated_step",
        mismatched,
    )

    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((single_step(outputs=("count",)),)),
            target_rows=target_rows()[:1],
            indexes=registry(),
        )
    assert exc.value.code == (
        BatchEvaluationCode.SCALAR_OUTPUT_CONTRACT_MISMATCH
    )


def test_row_evaluation_error_contains_row_index() -> None:
    with pytest.raises(BatchEvaluationError) as exc:
        evaluate_generated_plan_rows(
            plan_with((single_step(outputs=("count",)),)),
            target_rows=[
                target_rows()[0],
                {"left_id": "u1"},
            ],
            indexes=registry(),
        )
    assert exc.value.code == BatchEvaluationCode.ROW_EVALUATION_FAILED
    assert "row_index=1" in str(exc.value)


def test_earlier_successful_rows_not_returned_on_later_failure() -> None:
    with pytest.raises(BatchEvaluationError):
        evaluate_generated_plan_rows(
            plan_with((single_step(outputs=("count",)),)),
            target_rows=[
                target_rows()[0],
                {"left_id": "u1"},
            ],
            indexes=registry(),
        )


def test_empty_target_rows() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((single_step(outputs=("count",)),)),
        target_rows=[],
        indexes=registry(),
    )
    assert result.output_columns == ("count",)
    assert result.rows == ()


def test_no_generated_steps_with_target_rows() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((
            single_step(
                lowering_mode=LoweringMode.PASSTHROUGH,
                outputs=(),
            ),
        )),
        target_rows=target_rows(),
        indexes=registry(),
    )
    assert result.output_columns == ()
    assert [row.values for row in result.rows] == [(), ()]


def test_target_mappings_are_not_mutated() -> None:
    rows = target_rows()
    before = [dict(row) for row in rows]
    evaluate_generated_plan_rows(
        plan_with((single_step(outputs=("count",)),)),
        target_rows=rows,
        indexes=registry(),
    )
    assert rows == before


def test_indexes_are_not_mutated() -> None:
    reg = registry()
    before = repr(reg)
    evaluate_generated_plan_rows(
        plan_with((single_step(outputs=("count",)),)),
        target_rows=target_rows(),
        indexes=reg,
    )
    assert repr(reg) == before


def test_repeated_calls_are_deterministic() -> None:
    plan = plan_with((single_step(outputs=("count",)),))
    first = evaluate_generated_plan_rows(
        plan,
        target_rows=target_rows(),
        indexes=registry(),
    )
    second = evaluate_generated_plan_rows(
        plan,
        target_rows=target_rows(),
        indexes=registry(),
    )
    assert first == second


def test_strict_target_time_exclusion_preserved() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((single_step(outputs=("count",)),)),
        target_rows=target_rows()[:1],
        indexes=registry(),
    )
    assert row_dict(result, 0)["count"] == 3


def test_lower_window_boundary_remains_included() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((single_step(outputs=("count",)),)),
        target_rows=target_rows()[:1],
        indexes=registry(),
    )
    assert row_dict(result, 0)["count"] == 3


def test_duplicate_events_remain_separately_counted() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((pair_step(outputs=("pair_count",)),)),
        target_rows=target_rows()[:1],
        indexes=registry(),
    )
    assert row_dict(result, 0)["pair_count"] == 3


def test_unique_related_values_remain_deduplicated() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((
            single_step(
                operation="past_unique_neighbors",
                outputs=("unique",),
            ),
        )),
        target_rows=target_rows()[:1],
        indexes=registry(),
    )
    assert row_dict(result, 0)["unique"] == 2


def test_datetime_recency_conversion_remains_correct() -> None:
    result = evaluate_generated_plan_rows(
        plan_with((
            single_step(
                operation="days_since_last",
                outputs=("days", "missing"),
                window_days=None,
            ),
        )),
        target_rows=[
            {
                "left_id": "u2",
                "candidate_right_id": "p1",
                "timestamp": T0,
            }
        ],
        indexes=registry(),
    )
    assert row_dict(result, 0) == {"days": 0.5, "missing": 0}


def test_numeric_recency_remains_expressed_in_days() -> None:
    numeric_index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", 7),
    ])
    reg = TemporalIndexRegistry(
        single_key_indexes={
            SingleKeyIndexKey(
                "events",
                "left_id",
                "event_time",
                "neighbor_id",
            ): numeric_index
        },
        pair_key_indexes={},
    )
    result = evaluate_generated_plan_rows(
        plan_with((
            single_step(
                operation="days_since_last",
                outputs=("days", "missing"),
                window_days=None,
            ),
        )),
        target_rows=[
            {
                "left_id": "u1",
                "candidate_right_id": "p1",
                "timestamp": 10,
            }
        ],
        indexes=reg,
    )
    assert row_dict(result, 0) == {"days": 3.0, "missing": 0}


def ratebeer_plan():
    from pathlib import Path

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


def ratebeer_registry() -> TemporalIndexRegistry:
    user_index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0 - timedelta(days=30), "beer_a"),
        SingleKeyEvent("u1", T0 - timedelta(days=10), "beer_a"),
        SingleKeyEvent("u1", T0 - timedelta(days=1), "beer_b"),
        SingleKeyEvent("u1", T0, "beer_c"),
    ])
    place_index = SingleKeyTemporalIndex([
        SingleKeyEvent("p1", T0 - timedelta(days=30), "u1"),
        SingleKeyEvent("p1", T0 - timedelta(days=5), "u2"),
        SingleKeyEvent("p1", T0, "u3"),
    ])
    pair_index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", T0 - timedelta(days=60)),
        PairKeyEvent("u1", "p1", T0 - timedelta(days=3)),
        PairKeyEvent("u1", "p1", T0 - timedelta(days=3)),
        PairKeyEvent("u1", "p1", T0),
    ])
    return TemporalIndexRegistry(
        single_key_indexes={
            SingleKeyIndexKey(
                "beer_ratings",
                "user_id",
                "updated_at",
                "beer_id",
            ): user_index,
            SingleKeyIndexKey(
                "place_ratings",
                "place_id",
                "created_at",
                "user_id",
            ): place_index,
        },
        pair_key_indexes={
            PairKeyIndexKey(
                "place_ratings",
                "user_id",
                "place_id",
                "created_at",
            ): pair_index
        },
    )


def test_ratebeer_real_plan_integration() -> None:
    result = evaluate_generated_plan_rows(
        ratebeer_plan(),
        target_rows=[
            {
                "user_id": "u1",
                "candidate_place_id": "p1",
                "timestamp": T0,
            },
            {
                "user_id": "u1",
                "candidate_place_id": "missing",
                "timestamp": T0,
            },
            {
                "user_id": "missing",
                "candidate_place_id": "p1",
                "timestamp": T0,
            },
        ],
        indexes=ratebeer_registry(),
    )

    assert result.generated_step_count == 14
    assert len(result.output_columns) == 17
    assert len(result.rows) == 3
    assert all(
        not column.startswith("baseline")
        for column in result.output_columns
    )
    row0 = row_dict(result, 0)
    row1 = row_dict(result, 1)
    row2 = row_dict(result, 2)
    assert row0["f_pairwise__left__count_30d"] == 3
    assert row0["f_pairwise__right__count_30d"] == 2
    assert row0["f_pairwise__pair__prior_count"] == 3
    assert row1[
        "f_pairwise__right__days_since_last__is_missing"
    ] == 1
    assert row1[
        "f_pairwise__pair__days_since_last__is_missing"
    ] == 1
    assert row2[
        "f_pairwise__left__days_since_last__is_missing"
    ] == 1
    assert row2[
        "f_pairwise__pair__days_since_last__is_missing"
    ] == 1


def test_batch_evaluator_has_no_heavy_or_filesystem_dependency() -> None:
    import fdhg.compiler.batch_evaluator as module

    names = set(module.__dict__)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names
    assert "Path" not in names
