from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import (
    CandidateProgram,
    build_default_candidates,
)
from fdhg.compiler.schema_validation import (
    SchemaIssueCode,
    SchemaIssueScope,
    SchemaIssueSeverity,
    SchemaValidationReport,
    TableSchema,
    schema_validation_report_to_dict,
    validate_materialization_plan_schema,
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
            "events.event_time < target.timestamp"
        ),
        metadata={"pairwise_role": role},
    )


def baseline_primitive() -> Primitive:
    return Primitive(
        primitive_id="baseline::count",
        family=PrimitiveFamily.BASELINE,
        operation="count",
        source_table="missing_baseline_table",
        group_key="missing_baseline_col",
    )


def structural_primitive() -> Primitive:
    return Primitive(
        primitive_id=(
            "structural::afd::majority_confidence"
        ),
        family=PrimitiveFamily.STRUCTURAL,
        operation="majority_confidence",
    )


def unsupported_primitive() -> Primitive:
    return pairwise_primitive(
        "temporal::pairwise::left::mystery",
        operation="mystery",
        role="left",
    )


def make_plan(
    primitives: list[Primitive] | None = None,
):
    if primitives is None:
        primitives = [
            pairwise_primitive(
                "temporal::pairwise::left::unique_neighbors::30d",
                operation="past_unique_neighbors",
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

    compiled = CompiledTask(
        task_spec=complete_pairwise_task(),
        candidate_primitives=primitives,
    )
    program = CandidateProgram(
        program_id="synthetic",
        primitive_ids=[
            primitive.primitive_id
            for primitive in primitives
        ],
        families=["baseline", "structural", "temporal"],
        description="Synthetic schema validation plan.",
    )
    return plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={"events"},
    )


def valid_source_schema():
    return {
        "events": {
            "left_id",
            "right_id",
            "item_id",
            "event_time",
        }
    }


def valid_target_schema():
    return {
        "left_id",
        "candidate_right_id",
        "timestamp",
    }


def validate(plan, source=None, target=None):
    return validate_materialization_plan_schema(
        plan,
        source_schemas=(
            valid_source_schema()
            if source is None
            else source
        ),
        target_schema=(
            valid_target_schema()
            if target is None
            else target
        ),
    )


def issue_codes(report: SchemaValidationReport):
    return [issue.code for issue in report.issues]


def test_fully_valid_synthetic_pairwise_plan() -> None:
    report = validate(make_plan())

    assert report.valid
    assert report.checked_step_count == 3
    assert report.checked_table_count == 2
    assert report.issues == ()


def test_missing_source_table() -> None:
    report = validate(make_plan(), source={})

    assert not report.valid
    assert issue_codes(report) == [
        SchemaIssueCode.MISSING_SOURCE_TABLE,
        SchemaIssueCode.MISSING_SOURCE_TABLE,
        SchemaIssueCode.MISSING_SOURCE_TABLE,
    ]


def test_missing_source_group_key() -> None:
    report = validate(
        make_plan(),
        source={
            "events": {
                "right_id",
                "item_id",
                "event_time",
            }
        },
    )

    assert SchemaIssueCode.MISSING_SOURCE_COLUMN in issue_codes(
        report
    )
    assert report.issues[0].column_name == "left_id"


def test_missing_source_left_key() -> None:
    report = validate(
        make_plan([
            pairwise_primitive(
                "temporal::pairwise::pair::prior_count",
                operation="prior_pair_count",
                role="pair",
            )
        ]),
        source={"events": {"right_id", "event_time"}},
    )

    assert report.issues[0].column_name == "left_id"
    assert report.issues[0].code == (
        SchemaIssueCode.MISSING_SOURCE_COLUMN
    )


def test_missing_source_right_key() -> None:
    report = validate(
        make_plan([
            pairwise_primitive(
                "temporal::pairwise::pair::prior_count",
                operation="prior_pair_count",
                role="pair",
            )
        ]),
        source={"events": {"left_id", "event_time"}},
    )

    assert report.issues[0].column_name == "right_id"
    assert report.issues[0].code == (
        SchemaIssueCode.MISSING_SOURCE_COLUMN
    )


def test_missing_event_time_column() -> None:
    report = validate(
        make_plan(),
        source={
            "events": {
                "left_id",
                "right_id",
                "item_id",
            }
        },
    )

    assert any(
        issue.column_name == "event_time"
        for issue in report.issues
    )


def test_missing_related_column() -> None:
    report = validate(
        make_plan([
            pairwise_primitive(
                "temporal::pairwise::left::unique_neighbors::30d",
                operation="past_unique_neighbors",
                role="left",
                window_days=30,
            )
        ]),
        source={"events": {"left_id", "event_time"}},
    )

    assert report.issues[0].column_name == "item_id"


def test_missing_target_key() -> None:
    report = validate(
        make_plan([
            pairwise_primitive(
                "temporal::pairwise::right::count::30d",
                operation="window_count",
                role="right",
                window_days=30,
            )
        ]),
        target={"left_id", "timestamp"},
    )

    assert report.issues[0].column_name == (
        "candidate_right_id"
    )
    assert report.issues[0].code == (
        SchemaIssueCode.MISSING_TARGET_COLUMN
    )


def test_missing_target_left_key() -> None:
    report = validate(
        make_plan([
            pairwise_primitive(
                "temporal::pairwise::pair::prior_count",
                operation="prior_pair_count",
                role="pair",
            )
        ]),
        target={"candidate_right_id", "timestamp"},
    )

    assert report.issues[0].column_name == "left_id"


def test_missing_target_right_key() -> None:
    report = validate(
        make_plan([
            pairwise_primitive(
                "temporal::pairwise::pair::prior_count",
                operation="prior_pair_count",
                role="pair",
            )
        ]),
        target={"left_id", "timestamp"},
    )

    assert report.issues[0].column_name == (
        "candidate_right_id"
    )


def test_missing_target_time_column() -> None:
    report = validate(
        make_plan(),
        target={"left_id", "candidate_right_id"},
    )

    assert any(
        issue.column_name == "timestamp"
        for issue in report.issues
    )


def test_duplicate_requirements_yield_one_issue() -> None:
    plan = make_plan([
        pairwise_primitive(
            "temporal::pairwise::left::unique_neighbors::30d",
            operation="past_unique_neighbors",
            role="left",
            window_days=30,
        )
    ])
    step = replace(
        plan.steps[0],
        source_group_key="dup_col",
        related_col="dup_col",
    )
    plan = replace(plan, steps=(step,))
    report = validate(plan)

    assert len(report.issues) == 1
    assert report.issues[0].column_name == "dup_col"


def test_passthrough_steps_do_not_create_false_missing_issues() -> None:
    report = validate(make_plan([baseline_primitive()]))

    assert report.valid
    assert report.checked_step_count == 0
    assert report.issues == ()


def test_external_steps_are_reported_distinctly() -> None:
    report = validate(make_plan([structural_primitive()]))

    assert report.valid
    assert report.issues[0].severity == (
        SchemaIssueSeverity.WARNING
    )
    assert report.issues[0].scope == SchemaIssueScope.PROVIDER
    assert report.issues[0].code == (
        SchemaIssueCode.EXTERNAL_PROVIDER_REQUIRED
    )


def test_unsupported_steps_remain_blocking() -> None:
    report = validate(make_plan([unsupported_primitive()]))

    assert not report.valid
    assert report.issues[0].code == (
        SchemaIssueCode.UNSUPPORTED_STEP
    )


def test_issue_ordering_is_deterministic() -> None:
    report = validate(
        make_plan(),
        source={"events": set()},
        target=set(),
    )
    second = validate(
        make_plan(),
        source={"events": set()},
        target=set(),
    )

    assert report.issues == second.issues
    assert [
        (issue.primitive_id, issue.column_name)
        for issue in report.issues[:4]
    ] == [
        (
            "temporal::pairwise::left::unique_neighbors::30d",
            "left_id",
        ),
        (
            "temporal::pairwise::left::unique_neighbors::30d",
            "event_time",
        ),
        (
            "temporal::pairwise::left::unique_neighbors::30d",
            "item_id",
        ),
        (
            "temporal::pairwise::left::unique_neighbors::30d",
            "left_id",
        ),
    ]


def test_report_conversion_is_deterministic() -> None:
    report = validate(make_plan([structural_primitive()]))

    first = schema_validation_report_to_dict(report)
    second = schema_validation_report_to_dict(report)

    assert first == second
    assert first["issue_count"] == 1
    assert first["issues"][0]["code"] == (
        "external_provider_required"
    )


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


def test_ratebeer_valid_schema_case() -> None:
    report = validate_materialization_plan_schema(
        ratebeer_plan(),
        source_schemas={
            "beer_ratings": TableSchema(
                table_name="beer_ratings",
                columns=(
                    "user_id",
                    "beer_id",
                    "updated_at",
                ),
            ),
            "place_ratings": {
                "user_id",
                "place_id",
                "created_at",
            },
        },
        target_schema={
            "user_id",
            "candidate_place_id",
            "timestamp",
        },
    )

    assert report.program_id == (
        "baseline_plus_pairwise_temporal"
    )
    assert report.valid
    assert report.checked_step_count == 14
    assert report.checked_table_count == 3
    assert report.issues == ()


def test_ratebeer_fails_without_place_ratings_created_at() -> None:
    report = validate_materialization_plan_schema(
        ratebeer_plan(),
        source_schemas={
            "beer_ratings": {
                "user_id",
                "beer_id",
                "updated_at",
            },
            "place_ratings": {
                "user_id",
                "place_id",
            },
        },
        target_schema={
            "user_id",
            "candidate_place_id",
            "timestamp",
        },
    )

    assert not report.valid
    assert {
        issue.column_name for issue in report.issues
    } == {"created_at"}
    assert all(
        issue.code == SchemaIssueCode.MISSING_SOURCE_COLUMN
        for issue in report.issues
    )


def test_schema_validator_has_no_heavy_dependency() -> None:
    import fdhg.compiler.schema_validation as module

    names = set(module.__dict__)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names
