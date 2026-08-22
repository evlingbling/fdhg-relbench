from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import inspect

import pytest

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materializer import (
    LoweringMode,
    PrimitiveMaterializationStep,
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.primitive_evaluator import (
    PairKeyIndexKey,
    PrimitiveEvaluationCode,
    PrimitiveEvaluationError,
    SingleKeyIndexKey,
    TemporalIndexRegistry,
    evaluate_generated_step,
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
    operation: str = "window_count",
    role: str = "left",
    outputs: tuple[str, ...] = ("out",),
    window_days: int | None = 30,
) -> PrimitiveMaterializationStep:
    is_right = role == "right"
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=f"primitive::{role}::{operation}",
        operation=operation,
        lowering_mode=LoweringMode.GENERATE,
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
        materializable=True,
        temporally_safe=True,
        requires_external_provider=False,
    )


def pair_step(
    *,
    operation: str = "prior_pair_count",
    outputs: tuple[str, ...] = ("out",),
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=f"primitive::pair::{operation}",
        operation=operation,
        lowering_mode=LoweringMode.GENERATE,
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
        materializable=True,
        temporally_safe=True,
        requires_external_provider=False,
    )


def registry() -> TemporalIndexRegistry:
    single_events = [
        SingleKeyEvent("u1", T0 - timedelta(days=31), "old"),
        SingleKeyEvent("u1", T0 - timedelta(days=30), "a"),
        SingleKeyEvent("u1", T0 - timedelta(days=10), "a"),
        SingleKeyEvent("u1", T0 - timedelta(days=1), "b"),
        SingleKeyEvent("u1", T0, "at_target"),
        SingleKeyEvent("p1", T0 - timedelta(days=30), "u1"),
        SingleKeyEvent("p1", T0 - timedelta(days=5), "u2"),
        SingleKeyEvent("p1", T0, "u3"),
    ]
    single_index = SingleKeyTemporalIndex(single_events)
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


def target_row():
    return {
        "left_id": "u1",
        "candidate_right_id": "p1",
        "timestamp": T0,
    }


def assert_error(
    step: PrimitiveMaterializationStep,
    code: PrimitiveEvaluationCode,
):
    with pytest.raises(PrimitiveEvaluationError) as exc:
        evaluate_generated_step(
            step,
            target_row=target_row(),
            indexes=registry(),
        )
    assert exc.value.code == code
    assert step.program_id in str(exc.value)
    assert step.primitive_id in str(exc.value)
    assert step.operation in str(exc.value)


def test_left_role_window_count() -> None:
    result = evaluate_generated_step(
        single_step(outputs=("left_count",)),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"left_count": 3}


def test_right_role_window_count() -> None:
    result = evaluate_generated_step(
        single_step(role="right", outputs=("right_count",)),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"right_count": 2}


def test_left_role_past_unique_neighbors() -> None:
    result = evaluate_generated_step(
        single_step(
            operation="past_unique_neighbors",
            outputs=("left_unique",),
        ),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"left_unique": 2}


def test_right_role_past_unique_neighbors() -> None:
    result = evaluate_generated_step(
        single_step(
            operation="past_unique_neighbors",
            role="right",
            outputs=("right_unique",),
        ),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"right_unique": 2}


def test_left_role_days_since_last_with_history() -> None:
    result = evaluate_generated_step(
        single_step(
            operation="days_since_last",
            outputs=("days", "missing"),
            window_days=None,
        ),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"days": 1.0, "missing": 0}


def test_right_role_days_since_last_with_history() -> None:
    result = evaluate_generated_step(
        single_step(
            operation="days_since_last",
            role="right",
            outputs=("days", "missing"),
            window_days=None,
        ),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"days": 5.0, "missing": 0}


def test_days_since_last_missing_history() -> None:
    row = {
        "left_id": "missing",
        "candidate_right_id": "p1",
        "timestamp": T0,
    }
    result = evaluate_generated_step(
        single_step(
            operation="days_since_last",
            outputs=("days", "missing"),
            window_days=None,
        ),
        target_row=row,
        indexes=registry(),
    )
    assert result == {"days": 0.0, "missing": 1}


def test_prior_pair_count() -> None:
    result = evaluate_generated_step(
        pair_step(outputs=("prior_count",)),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"prior_count": 3}


def test_pair_days_since_last_with_history() -> None:
    result = evaluate_generated_step(
        pair_step(
            operation="pair_days_since_last",
            outputs=("days", "missing"),
        ),
        target_row=target_row(),
        indexes=registry(),
    )
    assert result == {"days": 3.0, "missing": 0}


def test_pair_days_since_last_missing_history() -> None:
    row = {
        "left_id": "u1",
        "candidate_right_id": "missing",
        "timestamp": T0,
    }
    result = evaluate_generated_step(
        pair_step(
            operation="pair_days_since_last",
            outputs=("days", "missing"),
        ),
        target_row=row,
        indexes=registry(),
    )
    assert result == {"days": 0.0, "missing": 1}


def test_exact_target_time_events_remain_excluded() -> None:
    assert evaluate_generated_step(
        single_step(outputs=("count",)),
        target_row=target_row(),
        indexes=registry(),
    )["count"] == 3


def test_lower_window_boundary_remains_included() -> None:
    assert evaluate_generated_step(
        single_step(outputs=("count",)),
        target_row=target_row(),
        indexes=registry(),
    )["count"] == 3


def test_duplicate_source_events_remain_separately_counted() -> None:
    assert evaluate_generated_step(
        pair_step(outputs=("prior_count",)),
        target_row=target_row(),
        indexes=registry(),
    )["prior_count"] == 3


def test_duplicate_related_values_remain_deduplicated() -> None:
    assert evaluate_generated_step(
        single_step(
            operation="past_unique_neighbors",
            outputs=("unique",),
        ),
        target_row=target_row(),
        indexes=registry(),
    )["unique"] == 2


def test_target_left_right_values_use_physical_target_bindings() -> None:
    step = pair_step(outputs=("prior_count",))
    row = {
        "left_id": "u1",
        "candidate_right_id": "p1",
        "timestamp": T0,
        "right_id": "wrong-column-value",
    }
    result = evaluate_generated_step(
        step,
        target_row=row,
        indexes=registry(),
    )
    assert result == {"prior_count": 3}


def test_missing_target_key_fails_clearly() -> None:
    step = single_step(outputs=("count",))
    with pytest.raises(PrimitiveEvaluationError) as exc:
        evaluate_generated_step(
            step,
            target_row={"timestamp": T0},
            indexes=registry(),
        )
    assert exc.value.code == (
        PrimitiveEvaluationCode.MISSING_TARGET_VALUE
    )
    assert "program_id=program" in str(exc.value)


def test_missing_target_time_fails_clearly() -> None:
    step = single_step(outputs=("count",))
    with pytest.raises(PrimitiveEvaluationError) as exc:
        evaluate_generated_step(
            step,
            target_row={"left_id": "u1"},
            indexes=registry(),
        )
    assert exc.value.code == (
        PrimitiveEvaluationCode.MISSING_TARGET_TIME
    )


def test_missing_index_fails_clearly() -> None:
    step = single_step(outputs=("count",))
    empty = TemporalIndexRegistry({}, {})
    with pytest.raises(PrimitiveEvaluationError) as exc:
        evaluate_generated_step(
            step,
            target_row=target_row(),
            indexes=empty,
        )
    assert exc.value.code == PrimitiveEvaluationCode.MISSING_INDEX


def test_incomplete_single_key_binding_fails_clearly() -> None:
    assert_error(
        replace(single_step(outputs=("count",)), source_group_key=None),
        PrimitiveEvaluationCode.INCOMPLETE_BINDING,
    )


def test_incomplete_pair_binding_fails_clearly() -> None:
    assert_error(
        replace(pair_step(outputs=("count",)), source_left_key=None),
        PrimitiveEvaluationCode.INCOMPLETE_BINDING,
    )


def test_malformed_output_column_count_fails_clearly() -> None:
    assert_error(
        single_step(outputs=("only_value",), operation="days_since_last"),
        PrimitiveEvaluationCode.INVALID_OUTPUT_COLUMNS,
    )


def test_unsupported_operation_fails_clearly() -> None:
    assert_error(
        single_step(operation="mystery", outputs=("out",)),
        PrimitiveEvaluationCode.UNSUPPORTED_OPERATION,
    )


def test_passthrough_step_is_rejected() -> None:
    assert_error(
        replace(
            single_step(outputs=("count",)),
            lowering_mode=LoweringMode.PASSTHROUGH,
        ),
        PrimitiveEvaluationCode.NON_GENERATE_STEP,
    )


def test_external_step_is_rejected() -> None:
    assert_error(
        replace(
            single_step(outputs=("count",)),
            lowering_mode=LoweringMode.EXTERNAL,
        ),
        PrimitiveEvaluationCode.NON_GENERATE_STEP,
    )


def test_unsupported_step_is_rejected() -> None:
    assert_error(
        replace(
            single_step(outputs=("count",)),
            lowering_mode=LoweringMode.UNSUPPORTED,
        ),
        PrimitiveEvaluationCode.UNSUPPORTED_STEP,
    )


def test_target_row_mapping_is_not_mutated() -> None:
    row = target_row()
    before = dict(row)
    evaluate_generated_step(
        single_step(outputs=("count",)),
        target_row=row,
        indexes=registry(),
    )
    assert row == before


def test_repeated_evaluation_is_deterministic() -> None:
    step = single_step(outputs=("count",))
    first = evaluate_generated_step(
        step,
        target_row=target_row(),
        indexes=registry(),
    )
    second = evaluate_generated_step(
        step,
        target_row=target_row(),
        indexes=registry(),
    )
    assert first == second


def test_output_key_order_follows_step_output_columns() -> None:
    result = evaluate_generated_step(
        pair_step(
            operation="pair_days_since_last",
            outputs=("value_first", "missing_second"),
        ),
        target_row=target_row(),
        indexes=registry(),
    )
    assert list(result) == ["value_first", "missing_second"]


def test_datetime_recency_converts_to_fractional_days() -> None:
    idx = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0 - timedelta(hours=12)),
    ])
    reg = TemporalIndexRegistry(
        single_key_indexes={
            SingleKeyIndexKey(
                "events",
                "left_id",
                "event_time",
                "neighbor_id",
            ): idx
        },
        pair_key_indexes={},
    )
    result = evaluate_generated_step(
        single_step(
            operation="days_since_last",
            outputs=("days", "missing"),
            window_days=None,
        ),
        target_row=target_row(),
        indexes=reg,
    )
    assert result == {"days": 0.5, "missing": 0}


def test_numeric_recency_is_interpreted_as_days() -> None:
    idx = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", 7),
    ])
    reg = TemporalIndexRegistry(
        single_key_indexes={
            SingleKeyIndexKey(
                "events",
                "left_id",
                "event_time",
                "neighbor_id",
            ): idx
        },
        pair_key_indexes={},
    )
    result = evaluate_generated_step(
        single_step(
            operation="days_since_last",
            outputs=("days", "missing"),
            window_days=None,
        ),
        target_row={
            "left_id": "u1",
            "candidate_right_id": "p1",
            "timestamp": 10,
        },
        indexes=reg,
    )
    assert result == {"days": 3.0, "missing": 0}


def ratebeer_plan():
    spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=(
            __import__("pathlib").Path(
                "configs/reproduction/tasks.yaml"
            )
        ),
        semantics_config=(
            __import__("pathlib").Path(
                "configs/reproduction/task_semantics.yaml"
            )
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
        SingleKeyEvent("u1", T0 - timedelta(days=31), "old"),
        SingleKeyEvent("u1", T0 - timedelta(days=30), "beer_a"),
        SingleKeyEvent("u1", T0 - timedelta(days=10), "beer_a"),
        SingleKeyEvent("u1", T0 - timedelta(days=1), "beer_b"),
        SingleKeyEvent("u1", T0, "beer_c"),
    ])
    place_index = SingleKeyTemporalIndex([
        SingleKeyEvent("p1", T0 - timedelta(days=31), "old"),
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


def by_primitive_suffix(plan, suffix: str):
    return next(
        step
        for step in plan.steps
        if step.primitive_id.endswith(suffix)
    )


def test_ratebeer_style_scalar_fixture_produces_expected_values() -> None:
    plan = ratebeer_plan()
    reg = ratebeer_registry()
    row = {
        "user_id": "u1",
        "candidate_place_id": "p1",
        "timestamp": T0,
    }
    checks = {
        "left::count::30d": 3,
        "left::unique_neighbors::30d": 2,
        "right::count::30d": 2,
        "right::unique_neighbors::30d": 2,
        "pair::prior_count": 3,
    }

    for suffix, expected in checks.items():
        result = evaluate_generated_step(
            by_primitive_suffix(plan, suffix),
            target_row=row,
            indexes=reg,
        )
        assert next(iter(result.values())) == expected

    left_days = evaluate_generated_step(
        by_primitive_suffix(plan, "left::days_since_last"),
        target_row=row,
        indexes=reg,
    )
    right_days = evaluate_generated_step(
        by_primitive_suffix(plan, "right::days_since_last"),
        target_row=row,
        indexes=reg,
    )
    pair_days = evaluate_generated_step(
        by_primitive_suffix(plan, "pair::days_since_last"),
        target_row=row,
        indexes=reg,
    )

    assert list(left_days.values()) == [1.0, 0]
    assert list(right_days.values()) == [5.0, 0]
    assert list(pair_days.values()) == [3.0, 0]


def test_generic_evaluator_contains_no_ratebeer_column_names() -> None:
    import fdhg.compiler.primitive_evaluator as module

    source = inspect.getsource(module)
    assert "user_id" not in source
    assert "candidate_place_id" not in source
    assert "beer_id" not in source
    assert "place_id" not in source


def test_evaluator_has_no_heavy_or_filesystem_dependency() -> None:
    import fdhg.compiler.primitive_evaluator as module

    names = set(module.__dict__)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names
    assert "Path" not in names
