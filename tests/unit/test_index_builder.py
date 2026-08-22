from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path

import pytest

from fdhg.compiler.batch_evaluator import evaluate_generated_plan_rows
from fdhg.compiler.config import load_task_spec
from fdhg.compiler.index_builder import (
    IndexBuildCode,
    IndexBuildError,
    build_temporal_index_registry,
)
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
)
from fdhg.compiler.programs import build_default_candidates


T0 = datetime(2026, 6, 1)


def single_step(
    *,
    primitive_id: str = "primitive::left::count",
    operation: str = "window_count",
    role: str = "left",
    related_col: str | None = "neighbor_id",
    lowering_mode: LoweringMode = LoweringMode.GENERATE,
) -> PrimitiveMaterializationStep:
    is_right = role == "right"
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation=operation,
        lowering_mode=lowering_mode,
        pairwise_role=role,
        source_table="events" if not is_right else "places",
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
        related_col=related_col,
        window_days=30,
        cutoff_operator="<",
        output_columns=("out",),
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
    primitive_id: str = "primitive::pair::prior",
    lowering_mode: LoweringMode = LoweringMode.GENERATE,
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation="prior_pair_count",
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
        output_columns=("pair_out",),
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


def source_rows():
    return {
        "events": [
            {
                "left_id": "u1",
                "right_id": "p1",
                "neighbor_id": "n1",
                "event_time": T0 - timedelta(days=10),
            },
            {
                "left_id": "u1",
                "right_id": "p1",
                "neighbor_id": "n2",
                "event_time": T0 - timedelta(days=1),
            },
        ],
        "places": [
            {
                "right_id": "p1",
                "neighbor_id": "u1",
                "event_time": T0 - timedelta(days=5),
            }
        ],
    }


def assert_build_error(
    plan: CandidateMaterializationPlan,
    rows,
    code: IndexBuildCode,
) -> None:
    with pytest.raises(IndexBuildError) as exc:
        build_temporal_index_registry(
            plan,
            source_rows_by_table=rows,
        )
    assert exc.value.code == code
    assert "program_id=program" in str(exc.value)


def test_one_single_key_requirement() -> None:
    registry = build_temporal_index_registry(
        plan_with((single_step(),)),
        source_rows_by_table=source_rows(),
    )

    assert len(registry.single_key_indexes) == 1
    key = SingleKeyIndexKey(
        "events",
        "left_id",
        "event_time",
        "neighbor_id",
    )
    assert key in registry.single_key_indexes
    assert registry.single_key_indexes[key].count_before("u1", T0) == 2


def test_one_pair_key_requirement() -> None:
    registry = build_temporal_index_registry(
        plan_with((pair_step(),)),
        source_rows_by_table=source_rows(),
    )

    assert len(registry.pair_key_indexes) == 1
    key = PairKeyIndexKey(
        "events",
        "left_id",
        "right_id",
        "event_time",
    )
    assert registry.pair_key_indexes[key].count_before("u1", "p1", T0) == 2


def test_multiple_windows_reuse_one_single_key_index() -> None:
    registry = build_temporal_index_registry(
        plan_with((
            single_step(primitive_id="count30"),
            replace(single_step(primitive_id="count90"), window_days=90),
        )),
        source_rows_by_table=source_rows(),
    )
    assert len(registry.single_key_indexes) == 1


def test_count_unique_recency_reuse_compatible_index() -> None:
    registry = build_temporal_index_registry(
        plan_with((
            single_step(primitive_id="count", operation="window_count"),
            single_step(
                primitive_id="unique",
                operation="past_unique_neighbors",
            ),
            single_step(
                primitive_id="recency",
                operation="days_since_last",
            ),
        )),
        source_rows_by_table=source_rows(),
    )
    assert len(registry.single_key_indexes) == 1


def test_duplicate_pair_requirements_reuse_one_pair_index() -> None:
    registry = build_temporal_index_registry(
        plan_with((
            pair_step(primitive_id="pair_count"),
            replace(
                pair_step(primitive_id="pair_days"),
                operation="pair_days_since_last",
            ),
        )),
        source_rows_by_table=source_rows(),
    )
    assert len(registry.pair_key_indexes) == 1


def test_multiple_source_tables_build_separate_indexes() -> None:
    registry = build_temporal_index_registry(
        plan_with((
            single_step(),
            single_step(role="right", primitive_id="right"),
        )),
        source_rows_by_table=source_rows(),
    )
    assert len(registry.single_key_indexes) == 2


def test_unsorted_source_rows_produce_sorted_index_behavior() -> None:
    rows = source_rows()
    rows["events"] = list(reversed(rows["events"]))
    registry = build_temporal_index_registry(
        plan_with((single_step(),)),
        source_rows_by_table=rows,
    )
    index = next(iter(registry.single_key_indexes.values()))
    assert index.days_since_last_before("u1", T0) == 1.0


def test_missing_source_table() -> None:
    assert_build_error(
        plan_with((single_step(),)),
        {},
        IndexBuildCode.MISSING_SOURCE_TABLE,
    )


def test_missing_group_key_column() -> None:
    rows = source_rows()
    del rows["events"][0]["left_id"]
    assert_build_error(
        plan_with((single_step(),)),
        rows,
        IndexBuildCode.MISSING_SOURCE_COLUMN,
    )


def test_missing_related_column() -> None:
    rows = source_rows()
    del rows["events"][0]["neighbor_id"]
    assert_build_error(
        plan_with((single_step(),)),
        rows,
        IndexBuildCode.MISSING_SOURCE_COLUMN,
    )


def test_missing_event_time_column() -> None:
    rows = source_rows()
    del rows["events"][0]["event_time"]
    assert_build_error(
        plan_with((single_step(),)),
        rows,
        IndexBuildCode.MISSING_SOURCE_COLUMN,
    )


def test_missing_pair_left_column() -> None:
    rows = source_rows()
    del rows["events"][0]["left_id"]
    assert_build_error(
        plan_with((pair_step(),)),
        rows,
        IndexBuildCode.MISSING_SOURCE_COLUMN,
    )


def test_missing_pair_right_column() -> None:
    rows = source_rows()
    del rows["events"][0]["right_id"]
    assert_build_error(
        plan_with((pair_step(),)),
        rows,
        IndexBuildCode.MISSING_SOURCE_COLUMN,
    )


def test_null_group_key_rejected() -> None:
    rows = source_rows()
    rows["events"][0]["left_id"] = None
    assert_build_error(
        plan_with((single_step(),)),
        rows,
        IndexBuildCode.NULL_GROUP_KEY,
    )


def test_null_pair_key_rejected() -> None:
    rows = source_rows()
    rows["events"][0]["right_id"] = None
    assert_build_error(
        plan_with((pair_step(),)),
        rows,
        IndexBuildCode.NULL_PAIR_KEY,
    )


def test_null_event_time_rejected() -> None:
    rows = source_rows()
    rows["events"][0]["event_time"] = None
    assert_build_error(
        plan_with((single_step(),)),
        rows,
        IndexBuildCode.NULL_EVENT_TIME,
    )


def test_null_related_value_accepted_and_excluded_from_unique() -> None:
    rows = source_rows()
    rows["events"][0]["neighbor_id"] = None
    registry = build_temporal_index_registry(
        plan_with((single_step(),)),
        source_rows_by_table=rows,
    )
    index = next(iter(registry.single_key_indexes.values()))
    assert index.unique_related_before("u1", T0, timedelta(days=30)) == 1


def test_mixed_datetime_numeric_times_rejected() -> None:
    rows = source_rows()
    rows["events"][1]["event_time"] = 1
    assert_build_error(
        plan_with((single_step(),)),
        rows,
        IndexBuildCode.INVALID_EVENT_TIME_TYPE,
    )


def test_mixed_aware_naive_datetimes_rejected() -> None:
    rows = source_rows()
    rows["events"][1]["event_time"] = rows["events"][1][
        "event_time"
    ].replace(tzinfo=timezone.utc)
    assert_build_error(
        plan_with((single_step(),)),
        rows,
        IndexBuildCode.INVALID_EVENT_TIME_TYPE,
    )


def test_passthrough_only_plan_returns_empty_registry() -> None:
    registry = build_temporal_index_registry(
        plan_with((
            single_step(
                lowering_mode=LoweringMode.PASSTHROUGH,
            ),
        )),
        source_rows_by_table={},
    )
    assert registry.single_key_indexes == {}
    assert registry.pair_key_indexes == {}


def test_external_step_rejected() -> None:
    assert_build_error(
        plan_with((
            single_step(
                lowering_mode=LoweringMode.EXTERNAL,
            ),
        )),
        source_rows(),
        IndexBuildCode.EXTERNAL_STEP_PRESENT,
    )


def test_unsupported_step_rejected() -> None:
    step = single_step(
        lowering_mode=LoweringMode.UNSUPPORTED,
    )
    assert_build_error(
        plan_with((step,), materializable=True),
        source_rows(),
        IndexBuildCode.UNSUPPORTED_STEP_PRESENT,
    )


def test_non_materializable_plan_rejected() -> None:
    assert_build_error(
        plan_with((single_step(),), materializable=False),
        source_rows(),
        IndexBuildCode.PLAN_NOT_MATERIALIZABLE,
    )


def test_temporally_unsafe_plan_rejected() -> None:
    assert_build_error(
        plan_with((single_step(),), temporally_safe=False),
        source_rows(),
        IndexBuildCode.PLAN_NOT_TEMPORALLY_SAFE,
    )


def test_source_rows_are_not_mutated() -> None:
    rows = source_rows()
    before = deepcopy(rows)
    build_temporal_index_registry(
        plan_with((single_step(),)),
        source_rows_by_table=rows,
    )
    assert rows == before


def test_plan_is_not_mutated() -> None:
    plan = plan_with((single_step(),))
    before = repr(plan)
    build_temporal_index_registry(
        plan,
        source_rows_by_table=source_rows(),
    )
    assert repr(plan) == before


def test_repeated_builds_are_deterministic() -> None:
    plan = plan_with((single_step(), pair_step()))
    first = build_temporal_index_registry(
        plan,
        source_rows_by_table=source_rows(),
    )
    second = build_temporal_index_registry(
        plan,
        source_rows_by_table=source_rows(),
    )
    assert tuple(first.single_key_indexes) == tuple(
        second.single_key_indexes
    )
    assert tuple(first.pair_key_indexes) == tuple(
        second.pair_key_indexes
    )
    first_single = next(iter(first.single_key_indexes.values()))
    second_single = next(iter(second.single_key_indexes.values()))
    assert first_single.count_before("u1", T0) == (
        second_single.count_before("u1", T0)
    )
    first_pair = next(iter(first.pair_key_indexes.values()))
    second_pair = next(iter(second.pair_key_indexes.values()))
    assert first_pair.count_before("u1", "p1", T0) == (
        second_pair.count_before("u1", "p1", T0)
    )


def test_no_dataset_specific_column_names_in_implementation() -> None:
    import fdhg.compiler.index_builder as module

    source = inspect.getsource(module)
    assert "user_id" not in source
    assert "beer_id" not in source
    assert "place_id" not in source
    assert "candidate_place_id" not in source


def test_index_builder_has_no_heavy_or_filesystem_dependency() -> None:
    import fdhg.compiler.index_builder as module

    names = set(module.__dict__)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names
    assert "Path" not in names


def ratebeer_plan():
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


def ratebeer_rows():
    return {
        "beer_ratings": [
            {
                "user_id": "u1",
                "beer_id": "beer_a",
                "updated_at": T0 - timedelta(days=30),
            },
            {
                "user_id": "u1",
                "beer_id": "beer_a",
                "updated_at": T0 - timedelta(days=10),
            },
            {
                "user_id": "u1",
                "beer_id": "beer_b",
                "updated_at": T0 - timedelta(days=1),
            },
            {
                "user_id": "u1",
                "beer_id": "beer_c",
                "updated_at": T0,
            },
        ],
        "place_ratings": [
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=60),
            },
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=30),
            },
            {
                "user_id": "u2",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=5),
            },
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=3),
            },
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=3),
            },
            {
                "user_id": "u3",
                "place_id": "p1",
                "created_at": T0,
            },
        ],
    }


def test_ratebeer_integration_registry_and_batch_evaluation() -> None:
    plan = ratebeer_plan()
    registry = build_temporal_index_registry(
        plan,
        source_rows_by_table=ratebeer_rows(),
    )
    assert set(registry.single_key_indexes) == {
        SingleKeyIndexKey(
            "beer_ratings",
            "user_id",
            "updated_at",
            "beer_id",
        ),
        SingleKeyIndexKey(
            "place_ratings",
            "place_id",
            "created_at",
            "user_id",
        ),
    }
    assert set(registry.pair_key_indexes) == {
        PairKeyIndexKey(
            "place_ratings",
            "user_id",
            "place_id",
            "created_at",
        )
    }
    result = evaluate_generated_plan_rows(
        plan,
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
        indexes=registry,
    )
    row0 = dict(result.rows[0].values)
    row1 = dict(result.rows[1].values)
    row2 = dict(result.rows[2].values)
    assert result.generated_step_count == 14
    assert len(result.output_columns) == 17
    assert len(result.rows) == 3
    assert row0["f_pairwise__left__count_30d"] == 3
    assert row0["f_pairwise__right__count_30d"] == 4
    assert row0["f_pairwise__pair__prior_count"] == 4
    assert row1[
        "f_pairwise__right__days_since_last__is_missing"
    ] == 1
    assert row2[
        "f_pairwise__left__days_since_last__is_missing"
    ] == 1
