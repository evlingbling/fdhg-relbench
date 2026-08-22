from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import inspect
import random

from fdhg.compiler.batch_evaluator import evaluate_generated_plan_rows
from fdhg.compiler.config import load_task_spec
from fdhg.compiler.index_builder import build_temporal_index_registry
from fdhg.compiler.materializer import (
    LoweringMode,
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates
from tests.fixtures import ratebeer_legacy_reference
from tests.fixtures.ratebeer_legacy_reference import (
    legacy_ratebeer_pairwise_features,
)


T0 = datetime(2026, 6, 1, 12, 0, 0)
ACTIVITY_PRODUCT = "f_pairtmp__user_place_activity_product"
ACTIVITY_RATIO = "f_pairtmp__user_place_activity_ratio"


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


def generated_output_columns(plan) -> tuple[str, ...]:
    columns: list[str] = []

    for step in plan.steps:
        if step.lowering_mode == LoweringMode.GENERATE:
            columns.extend(step.output_columns)

    return tuple(columns)


def generic_records(source_rows, target_rows):
    plan = ratebeer_plan()
    registry = build_temporal_index_registry(
        plan,
        source_rows_by_table=source_rows,
    )
    result = evaluate_generated_plan_rows(
        plan,
        target_rows=target_rows,
        indexes=registry,
    )
    return (
        result.output_columns,
        tuple(dict(row.values) for row in result.rows),
        result,
    )


def assert_feature_records_equivalent(
    *,
    fixture_name: str,
    source_rows,
    target_rows,
) -> None:
    expected = legacy_ratebeer_pairwise_features(
        source_rows_by_table=source_rows,
        target_rows=target_rows,
    )
    columns, actual, _ = generic_records(
        source_rows,
        target_rows,
    )
    expected_columns = tuple(expected[0].keys()) if expected else columns

    assert expected_columns == columns

    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected, actual)
    ):
        assert set(expected_row) == set(actual_row)

        for column in columns:
            expected_value = expected_row[column]
            actual_value = actual_row[column]

            if isinstance(expected_value, float) or isinstance(
                actual_value,
                float,
            ):
                assert abs(expected_value - actual_value) <= 1e-12, (
                    f"fixture={fixture_name} row_index={row_index} "
                    f"target_row={target_rows[row_index]!r} "
                    f"column={column} legacy={expected_value!r} "
                    f"generic={actual_value!r}"
                )
            else:
                assert expected_value == actual_value, (
                    f"fixture={fixture_name} row_index={row_index} "
                    f"target_row={target_rows[row_index]!r} "
                    f"column={column} legacy={expected_value!r} "
                    f"generic={actual_value!r}"
                )

    assert len(expected) == len(actual)


def standard_source_rows():
    return {
        "beer_ratings": [
            {
                "user_id": "u1",
                "beer_id": "old",
                "updated_at": T0 - timedelta(days=366),
            },
            {
                "user_id": "u1",
                "beer_id": "beer_a",
                "updated_at": T0 - timedelta(days=90),
            },
            {
                "user_id": "u1",
                "beer_id": "beer_a",
                "updated_at": T0 - timedelta(days=30),
            },
            {
                "user_id": "u1",
                "beer_id": "beer_b",
                "updated_at": T0 - timedelta(days=1),
            },
            {
                "user_id": "u1",
                "beer_id": None,
                "updated_at": T0 - timedelta(hours=12),
            },
            {
                "user_id": "u1",
                "beer_id": "at_target",
                "updated_at": T0,
            },
            {
                "user_id": "u2",
                "beer_id": "beer_z",
                "updated_at": T0 - timedelta(days=5),
            },
        ],
        "place_ratings": [
            {
                "user_id": "old",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=366),
            },
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=90),
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
            {
                "user_id": "u2",
                "place_id": "p2",
                "created_at": T0 - timedelta(days=2),
            },
        ],
    }


def standard_target_rows():
    return [
        {
            "user_id": "u1",
            "candidate_place_id": "p1",
            "timestamp": T0,
        },
        {
            "user_id": "u1",
            "candidate_place_id": "missing_place",
            "timestamp": T0,
        },
        {
            "user_id": "missing_user",
            "candidate_place_id": "p1",
            "timestamp": T0,
        },
        {
            "user_id": "missing_user",
            "candidate_place_id": "missing_place",
            "timestamp": T0,
        },
    ]


def boundary_source_rows():
    epsilon = timedelta(microseconds=1)
    return {
        "beer_ratings": [
            {
                "user_id": "u1",
                "beer_id": "outside",
                "updated_at": T0 - timedelta(days=30) - epsilon,
            },
            {
                "user_id": "u1",
                "beer_id": "lower",
                "updated_at": T0 - timedelta(days=30),
            },
            {
                "user_id": "u1",
                "beer_id": "inside",
                "updated_at": T0 - epsilon,
            },
            {
                "user_id": "u1",
                "beer_id": "target",
                "updated_at": T0,
            },
        ],
        "place_ratings": [
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=30) - epsilon,
            },
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": T0 - timedelta(days=30),
            },
            {
                "user_id": "u2",
                "place_id": "p1",
                "created_at": T0 - epsilon,
            },
            {
                "user_id": "u3",
                "place_id": "p1",
                "created_at": T0,
            },
        ],
    }


def duplicate_timestamp_rows():
    ts = T0 - timedelta(days=3)
    return {
        "beer_ratings": [
            {
                "user_id": "u1",
                "beer_id": "same",
                "updated_at": ts,
            },
            {
                "user_id": "u1",
                "beer_id": "same",
                "updated_at": ts,
            },
            {
                "user_id": "u1",
                "beer_id": "other",
                "updated_at": ts,
            },
        ],
        "place_ratings": [
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": ts,
            },
            {
                "user_id": "u1",
                "place_id": "p1",
                "created_at": ts,
            },
            {
                "user_id": "u2",
                "place_id": "p1",
                "created_at": ts,
            },
        ],
    }


def test_exact_output_column_count_is_17() -> None:
    assert len(generated_output_columns(ratebeer_plan())) == 17


def test_exact_generated_step_count_is_14() -> None:
    plan = ratebeer_plan()
    assert sum(
        step.lowering_mode == LoweringMode.GENERATE
        for step in plan.steps
    ) == 14


def test_standard_fixture_exact_equivalence() -> None:
    assert_feature_records_equivalent(
        fixture_name="standard",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows(),
    )


def test_strict_target_time_exclusion() -> None:
    assert_feature_records_equivalent(
        fixture_name="boundary_target_exclusion",
        source_rows=boundary_source_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_inclusive_lower_window_boundary() -> None:
    _, actual, _ = generic_records(
        boundary_source_rows(),
        standard_target_rows()[:1],
    )
    assert actual[0]["f_pairwise__left__count_30d"] == 2
    assert_feature_records_equivalent(
        fixture_name="boundary_lower_inclusive",
        source_rows=boundary_source_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_just_outside_window_exclusion() -> None:
    _, actual, _ = generic_records(
        boundary_source_rows(),
        standard_target_rows()[:1],
    )
    assert actual[0]["f_pairwise__left__count_30d"] == 2


def test_duplicate_events_counted_separately() -> None:
    _, actual, _ = generic_records(
        duplicate_timestamp_rows(),
        standard_target_rows()[:1],
    )
    assert actual[0]["f_pairwise__left__count_30d"] == 3
    assert actual[0]["f_pairwise__pair__prior_count"] == 2
    assert_feature_records_equivalent(
        fixture_name="duplicate_events",
        source_rows=duplicate_timestamp_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_duplicate_related_values_deduplicated() -> None:
    _, actual, _ = generic_records(
        duplicate_timestamp_rows(),
        standard_target_rows()[:1],
    )
    assert actual[0]["f_pairwise__left__unique_neighbors_30d"] == 2


def test_null_related_values_excluded() -> None:
    _, actual, _ = generic_records(
        standard_source_rows(),
        standard_target_rows()[:1],
    )
    assert actual[0]["f_pairwise__left__unique_neighbors_30d"] == 2


def test_missing_user_history() -> None:
    _, actual, _ = generic_records(
        standard_source_rows(),
        [standard_target_rows()[2]],
    )
    assert actual[0]["f_pairwise__left__days_since_last__is_missing"] == 1
    assert_feature_records_equivalent(
        fixture_name="missing_user",
        source_rows=standard_source_rows(),
        target_rows=[standard_target_rows()[2]],
    )


def test_missing_place_history() -> None:
    _, actual, _ = generic_records(
        standard_source_rows(),
        [standard_target_rows()[1]],
    )
    assert actual[0]["f_pairwise__right__days_since_last__is_missing"] == 1


def test_missing_pair_history() -> None:
    _, actual, _ = generic_records(
        standard_source_rows(),
        [standard_target_rows()[1]],
    )
    assert actual[0]["f_pairwise__pair__days_since_last__is_missing"] == 1


def test_mixed_missing_history_combinations() -> None:
    assert_feature_records_equivalent(
        fixture_name="mixed_missing",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows(),
    )


def test_pair_count_equivalence() -> None:
    assert_feature_records_equivalent(
        fixture_name="pair_count",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_pair_recency_equivalence() -> None:
    assert_feature_records_equivalent(
        fixture_name="pair_recency",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_left_recency_equivalence() -> None:
    assert_feature_records_equivalent(
        fixture_name="left_recency",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_right_recency_equivalence() -> None:
    assert_feature_records_equivalent(
        fixture_name="right_recency",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_fractional_datetime_day_equivalence() -> None:
    _, actual, _ = generic_records(
        standard_source_rows(),
        standard_target_rows()[:1],
    )
    assert actual[0]["f_pairwise__left__days_since_last"] == 0.5
    assert_feature_records_equivalent(
        fixture_name="fractional_days",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows()[:1],
    )


def test_multiple_target_times() -> None:
    targets = [
        {
            "user_id": "u1",
            "candidate_place_id": "p1",
            "timestamp": T0 - timedelta(days=4),
        },
        {
            "user_id": "u1",
            "candidate_place_id": "p1",
            "timestamp": T0,
        },
    ]
    assert_feature_records_equivalent(
        fixture_name="multiple_target_times",
        source_rows=standard_source_rows(),
        target_rows=targets,
    )


def test_unsorted_source_rows() -> None:
    rows = standard_source_rows()
    rows["beer_ratings"] = list(reversed(rows["beer_ratings"]))
    rows["place_ratings"] = list(reversed(rows["place_ratings"]))
    assert_feature_records_equivalent(
        fixture_name="unsorted_source_rows",
        source_rows=rows,
        target_rows=standard_target_rows(),
    )


def test_repeated_runs_deterministic() -> None:
    first = generic_records(
        standard_source_rows(),
        standard_target_rows(),
    )
    second = generic_records(
        standard_source_rows(),
        standard_target_rows(),
    )
    assert first == second


def test_source_rows_not_mutated() -> None:
    rows = standard_source_rows()
    before = deepcopy(rows)
    assert_feature_records_equivalent(
        fixture_name="source_not_mutated",
        source_rows=rows,
        target_rows=standard_target_rows(),
    )
    assert rows == before


def test_target_rows_not_mutated() -> None:
    targets = standard_target_rows()
    before = deepcopy(targets)
    assert_feature_records_equivalent(
        fixture_name="target_not_mutated",
        source_rows=standard_source_rows(),
        target_rows=targets,
    )
    assert targets == before


def test_legacy_reference_independent_of_generic_modules() -> None:
    source = inspect.getsource(ratebeer_legacy_reference)
    forbidden = [
        "temporal_index",
        "primitive_evaluator",
        "batch_evaluator",
        "index_builder",
    ]
    for name in forbidden:
        assert name not in source


def test_activity_product_ratio_explicitly_absent() -> None:
    columns = generated_output_columns(ratebeer_plan())
    assert ACTIVITY_PRODUCT not in columns
    assert ACTIVITY_RATIO not in columns


def test_no_heavy_dependency_or_filesystem_writes(tmp_path) -> None:
    before = sorted(tmp_path.iterdir())
    import fdhg.compiler.batch_evaluator as batch_module

    helper_names = set(ratebeer_legacy_reference.__dict__)
    batch_names = set(batch_module.__dict__)
    assert "pandas" not in helper_names | batch_names
    assert "pyarrow" not in helper_names | batch_names
    assert "subprocess" not in helper_names | batch_names
    assert "tabpfn" not in helper_names | batch_names
    assert_feature_records_equivalent(
        fixture_name="no_writes",
        source_rows=standard_source_rows(),
        target_rows=standard_target_rows()[:1],
    )
    assert sorted(tmp_path.iterdir()) == before


def test_fixed_seed_randomized_equivalence() -> None:
    rng = random.Random(20260720)
    users = ["u1", "u2", "u3"]
    places = ["p1", "p2"]
    beers = ["b1", "b2", "b3", None]
    beer_rows = []
    place_rows = []

    for _ in range(40):
        beer_rows.append({
            "user_id": rng.choice(users),
            "beer_id": rng.choice(beers),
            "updated_at": T0 - timedelta(days=rng.randrange(0, 140)),
        })

    for _ in range(45):
        place_rows.append({
            "user_id": rng.choice(users),
            "place_id": rng.choice(places),
            "created_at": T0 - timedelta(days=rng.randrange(0, 140)),
        })

    targets = [
        {
            "user_id": rng.choice(users + ["missing_user"]),
            "candidate_place_id": rng.choice(places + ["missing_place"]),
            "timestamp": T0 - timedelta(days=rng.randrange(0, 30)),
        }
        for _ in range(12)
    ]
    assert_feature_records_equivalent(
        fixture_name="fixed_seed_randomized",
        source_rows={
            "beer_ratings": beer_rows,
            "place_ratings": place_rows,
        },
        target_rows=targets,
    )
