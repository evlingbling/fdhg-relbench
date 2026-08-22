from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import inspect

import pytest

from fdhg.compiler.batch_evaluator import BatchEvaluationCode
from fdhg.compiler.config import load_task_spec
from fdhg.compiler.in_memory_materializer import (
    InMemoryMaterializationCode,
    InMemoryMaterializationError,
    materialize_generated_features_in_memory,
)
from fdhg.compiler.index_builder import IndexBuildCode
from fdhg.compiler.materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    MaterializationAuditRow,
    PrimitiveMaterializationStep,
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates
from tests.fixtures.ratebeer_legacy_reference import (
    legacy_ratebeer_pairwise_features,
)
from tests.unit.test_ratebeer_legacy_equivalence import (
    standard_source_rows,
    standard_target_rows,
)


T0 = datetime(2026, 6, 1)


class OneShotRows:
    def __init__(self, rows):
        self.rows = list(rows)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1

        if self.iterations > 1:
            raise AssertionError("iterated more than once")

        return iter(self.rows)


def single_step(
    *,
    primitive_id: str = "primitive::left::count",
    operation: str = "window_count",
    outputs: tuple[str, ...] = ("count",),
    lowering_mode: LoweringMode = LoweringMode.GENERATE,
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation=operation,
        lowering_mode=lowering_mode,
        pairwise_role="left",
        source_table="events",
        source_group_key="left_id",
        source_left_key=None,
        source_right_key=None,
        source_event_time_col="event_time",
        target_key="left_id",
        target_left_key=None,
        target_right_key=None,
        target_time_col="timestamp",
        related_col="neighbor_id",
        window_days=30,
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


def pair_step() -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id="primitive::pair::prior",
        operation="prior_pair_count",
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
        output_columns=("prior_pair_count",),
        materializable=True,
        temporally_safe=True,
        requires_external_provider=False,
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


def source_rows():
    return {
        "events": [
            {
                "left_id": "u1",
                "right_id": "p1",
                "neighbor_id": "a",
                "event_time": T0 - timedelta(days=30),
            },
            {
                "left_id": "u1",
                "right_id": "p1",
                "neighbor_id": "a",
                "event_time": T0 - timedelta(days=10),
            },
            {
                "left_id": "u1",
                "right_id": "p1",
                "neighbor_id": "b",
                "event_time": T0 - timedelta(days=1),
            },
            {
                "left_id": "u1",
                "right_id": "p1",
                "neighbor_id": "target",
                "event_time": T0,
            },
        ]
    }


def target_rows():
    return [
        {
            "left_id": "u1",
            "candidate_right_id": "p1",
            "timestamp": T0,
        },
        {
            "left_id": "missing",
            "candidate_right_id": "p1",
            "timestamp": T0,
        },
    ]


def row_values(result, row_index: int):
    return dict(result.batch_result.rows[row_index].values)


def assert_error(code, plan, rows, targets):
    with pytest.raises(InMemoryMaterializationError) as exc:
        materialize_generated_features_in_memory(
            plan,
            source_rows_by_table=rows,
            target_rows=targets,
        )
    assert exc.value.code == code
    assert "program_id=program" in str(exc.value)
    return exc.value


def test_successful_one_table_materialization() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )

    assert result.program_id == "program"
    assert result.schema_report.valid
    assert result.single_index_count == 1
    assert result.pair_index_count == 0
    assert row_values(result, 0) == {"count": 3}


def test_successful_single_key_and_pair_key_composition() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((single_step(), pair_step())),
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )

    assert result.single_index_count == 1
    assert result.pair_index_count == 1
    assert row_values(result, 0)["prior_pair_count"] == 3


def test_schema_validation_executes_before_index_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("index construction should not run")

    monkeypatch.setattr(
        "fdhg.compiler.in_memory_materializer."
        "build_temporal_index_registry",
        fail_if_called,
    )
    error = assert_error(
        InMemoryMaterializationCode.SCHEMA_VALIDATION_FAILED,
        plan_with((single_step(),)),
        {"events": [{"left_id": "u1"}]},
        target_rows()[:1],
    )
    assert error.stage == "schema_validation"
    assert not called


def test_invalid_schema_prevents_index_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fdhg.compiler.in_memory_materializer."
        "build_temporal_index_registry",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("should not build")
        ),
    )
    assert_error(
        InMemoryMaterializationCode.SCHEMA_VALIDATION_FAILED,
        plan_with((single_step(),)),
        {"events": []},
        target_rows()[:1],
    )


def test_index_build_failure_is_wrapped_with_cause() -> None:
    error = assert_error(
        InMemoryMaterializationCode.INDEX_BUILD_FAILED,
        plan_with((single_step(),)),
        {
            "events": [
                {
                    "left_id": None,
                    "neighbor_id": "a",
                    "event_time": T0,
                }
            ]
        },
        target_rows()[:1],
    )
    assert error.underlying_code == IndexBuildCode.NULL_GROUP_KEY.value
    assert error.__cause__ is not None


def test_batch_evaluation_failure_is_wrapped_with_cause() -> None:
    error = assert_error(
        InMemoryMaterializationCode.BATCH_EVALUATION_FAILED,
        plan_with((single_step(),)),
        source_rows(),
        [
            {"left_id": "u1", "timestamp": T0},
            {"timestamp": T0},
        ],
    )
    assert error.underlying_code == (
        BatchEvaluationCode.ROW_EVALUATION_FAILED.value
    )
    assert error.__cause__ is not None


def test_source_schema_union_is_deterministic() -> None:
    rows = {
        "events": [
            {
                "left_id": "u1",
                "neighbor_id": "b",
                "event_time": T0 - timedelta(days=1),
            },
            {
                "neighbor_id": "a",
                "event_time": T0,
                "left_id": "u1",
            },
        ]
    }
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=rows,
        target_rows=target_rows()[:1],
    )
    assert result.schema_report.valid


def test_target_schema_union_is_deterministic() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=[
            {
                "left_id": "u1",
                "candidate_right_id": "p1",
                "timestamp": T0,
            },
            {
                "timestamp": T0 - timedelta(days=1),
                "left_id": "u1",
                "candidate_right_id": "p1",
            },
        ],
    )
    assert result.schema_report.valid


def test_missing_source_column_detected() -> None:
    error = assert_error(
        InMemoryMaterializationCode.SCHEMA_VALIDATION_FAILED,
        plan_with((single_step(),)),
        {"events": [{"left_id": "u1", "event_time": T0}]},
        target_rows()[:1],
    )
    assert "missing_source_column" in str(error)


def test_missing_target_column_detected() -> None:
    error = assert_error(
        InMemoryMaterializationCode.SCHEMA_VALIDATION_FAILED,
        plan_with((single_step(),)),
        source_rows(),
        [{"left_id": "u1"}],
    )
    assert "missing_target_column" in str(error)


def test_per_row_missing_source_value_reaches_index_builder() -> None:
    error = assert_error(
        InMemoryMaterializationCode.INDEX_BUILD_FAILED,
        plan_with((single_step(),)),
        {
            "events": [
                {
                    "left_id": "u1",
                    "neighbor_id": "a",
                    "event_time": T0,
                },
                {"left_id": "u1", "event_time": T0},
            ]
        },
        target_rows()[:1],
    )
    assert "row_index=1" in str(error)


def test_per_row_missing_target_value_reaches_batch_evaluator() -> None:
    error = assert_error(
        InMemoryMaterializationCode.BATCH_EVALUATION_FAILED,
        plan_with((single_step(),)),
        source_rows(),
        [
            {
                "left_id": "u1",
                "timestamp": T0,
            },
            {
                "candidate_right_id": "p1",
                "timestamp": T0,
            },
        ],
    )
    assert "row_index=1" in str(error)


def test_null_source_key_is_rejected() -> None:
    assert_error(
        InMemoryMaterializationCode.INDEX_BUILD_FAILED,
        plan_with((single_step(),)),
        {
            "events": [
                {
                    "left_id": None,
                    "neighbor_id": "a",
                    "event_time": T0,
                }
            ]
        },
        target_rows()[:1],
    )


def test_null_source_event_time_is_rejected() -> None:
    assert_error(
        InMemoryMaterializationCode.INDEX_BUILD_FAILED,
        plan_with((single_step(),)),
        {
            "events": [
                {
                    "left_id": "u1",
                    "neighbor_id": "a",
                    "event_time": None,
                }
            ]
        },
        target_rows()[:1],
    )


def test_strict_target_time_cutoff_preserved() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )
    assert row_values(result, 0)["count"] == 3


def test_inclusive_lower_window_boundary_preserved() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )
    assert row_values(result, 0)["count"] == 3


def test_duplicate_events_counted_separately() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((pair_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )
    assert row_values(result, 0)["prior_pair_count"] == 3


def test_duplicate_related_values_deduplicated() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((
            single_step(
                operation="past_unique_neighbors",
                outputs=("unique",),
            ),
        )),
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )
    assert row_values(result, 0)["unique"] == 2


def test_missing_history_flags_preserved() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((
            single_step(
                operation="days_since_last",
                outputs=("days", "missing"),
            ),
        )),
        source_rows_by_table=source_rows(),
        target_rows=target_rows(),
    )
    assert row_values(result, 1) == {"days": 0.0, "missing": 1}


def test_output_column_order_preserved() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((
            single_step(outputs=("first",)),
            pair_step(),
        )),
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )
    assert result.output_columns == ("first", "prior_pair_count")


def test_target_row_order_preserved() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=target_rows(),
    )
    assert [row.row_index for row in result.batch_result.rows] == [0, 1]


def test_source_rows_not_mutated() -> None:
    rows = source_rows()
    before = deepcopy(rows)
    materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=rows,
        target_rows=target_rows()[:1],
    )
    assert rows == before


def test_target_rows_not_mutated() -> None:
    targets = target_rows()
    before = deepcopy(targets)
    materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=targets,
    )
    assert targets == before


def test_plan_not_mutated() -> None:
    plan = plan_with((single_step(),))
    before = repr(plan)
    materialize_generated_features_in_memory(
        plan,
        source_rows_by_table=source_rows(),
        target_rows=target_rows()[:1],
    )
    assert repr(plan) == before


def test_repeated_calls_deterministic() -> None:
    plan = plan_with((single_step(), pair_step()))
    first = materialize_generated_features_in_memory(
        plan,
        source_rows_by_table=source_rows(),
        target_rows=target_rows(),
    )
    second = materialize_generated_features_in_memory(
        plan,
        source_rows_by_table=source_rows(),
        target_rows=target_rows(),
    )
    assert first == second


def test_source_iterables_consumed_once() -> None:
    rows = OneShotRows(source_rows()["events"])
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table={"events": rows},
        target_rows=target_rows()[:1],
    )
    assert result.single_index_count == 1
    assert rows.iterations == 1


def test_target_iterable_consumed_once() -> None:
    rows = OneShotRows(target_rows())
    result = materialize_generated_features_in_memory(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
        target_rows=rows,
    )
    assert result.target_row_count == 2
    assert rows.iterations == 1


def test_empty_required_source_table_fails() -> None:
    assert_error(
        InMemoryMaterializationCode.SCHEMA_VALIDATION_FAILED,
        plan_with((single_step(),)),
        {"events": []},
        target_rows()[:1],
    )


def test_empty_target_rows_fail_clearly() -> None:
    assert_error(
        InMemoryMaterializationCode.SCHEMA_VALIDATION_FAILED,
        plan_with((single_step(),)),
        source_rows(),
        [],
    )


def test_passthrough_only_plan_behavior() -> None:
    result = materialize_generated_features_in_memory(
        plan_with((
            single_step(
                lowering_mode=LoweringMode.PASSTHROUGH,
                outputs=(),
            ),
        )),
        source_rows_by_table={},
        target_rows=target_rows(),
    )
    assert result.single_index_count == 0
    assert result.pair_index_count == 0
    assert result.output_columns == ()
    assert [row.values for row in result.batch_result.rows] == [(), ()]


def test_external_plan_rejected() -> None:
    error = assert_error(
        InMemoryMaterializationCode.INDEX_BUILD_FAILED,
        plan_with((
            single_step(lowering_mode=LoweringMode.EXTERNAL),
        )),
        source_rows(),
        target_rows()[:1],
    )
    assert error.underlying_code == IndexBuildCode.EXTERNAL_STEP_PRESENT.value


def test_unsupported_plan_rejected() -> None:
    assert_error(
        InMemoryMaterializationCode.SCHEMA_VALIDATION_FAILED,
        plan_with((
            single_step(lowering_mode=LoweringMode.UNSUPPORTED),
        ), materializable=True),
        source_rows(),
        target_rows()[:1],
    )


def test_no_heavy_or_filesystem_dependency() -> None:
    import fdhg.compiler.in_memory_materializer as module

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


def test_ratebeer_integration() -> None:
    result = materialize_generated_features_in_memory(
        ratebeer_plan(),
        source_rows_by_table=standard_source_rows(),
        target_rows=standard_target_rows(),
    )
    assert result.schema_report.valid
    assert result.single_index_count == 2
    assert result.pair_index_count == 1
    assert result.generated_step_count == 14
    assert len(result.output_columns) == 17
    assert len(result.batch_result.rows) == len(standard_target_rows())
    row0 = dict(result.batch_result.rows[0].values)
    assert row0["f_pairwise__left__count_30d"] == 3


def test_legacy_equivalence_integration() -> None:
    result = materialize_generated_features_in_memory(
        ratebeer_plan(),
        source_rows_by_table=standard_source_rows(),
        target_rows=standard_target_rows(),
    )
    expected = legacy_ratebeer_pairwise_features(
        source_rows_by_table=standard_source_rows(),
        target_rows=standard_target_rows(),
    )

    for row_index, row in enumerate(result.batch_result.rows):
        actual = dict(row.values)

        for column in result.output_columns:
            assert actual[column] == expected[row_index][column]
