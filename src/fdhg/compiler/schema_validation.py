from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Collection, Mapping

from .materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    PrimitiveMaterializationStep,
)


class SchemaIssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class SchemaIssueScope(str, Enum):
    SOURCE = "source"
    TARGET = "target"
    PROVIDER = "provider"
    STEP = "step"


class SchemaIssueCode(str, Enum):
    MISSING_SOURCE_TABLE = "missing_source_table"
    MISSING_SOURCE_COLUMN = "missing_source_column"
    MISSING_TARGET_COLUMN = "missing_target_column"
    EXTERNAL_PROVIDER_REQUIRED = (
        "external_provider_required"
    )
    UNSUPPORTED_STEP = "unsupported_step"


@dataclass(frozen=True)
class TableSchema:
    table_name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class SchemaValidationIssue:
    program_id: str
    primitive_id: str
    severity: SchemaIssueSeverity
    scope: SchemaIssueScope
    table_name: str
    column_name: str | None
    code: SchemaIssueCode
    message: str


@dataclass(frozen=True)
class SchemaValidationReport:
    program_id: str
    valid: bool
    checked_step_count: int
    checked_table_count: int
    issues: tuple[SchemaValidationIssue, ...]


SchemaInput = Mapping[str, Collection[str] | TableSchema]


SOURCE_FIELD_ORDER = (
    "source_group_key",
    "source_left_key",
    "source_right_key",
    "source_event_time_col",
    "related_col",
)

TARGET_FIELD_ORDER = (
    "target_key",
    "target_left_key",
    "target_right_key",
    "target_time_col",
)


def validate_materialization_plan_schema(
    plan: CandidateMaterializationPlan,
    *,
    source_schemas: SchemaInput,
    target_schema: Collection[str] | TableSchema,
) -> SchemaValidationReport:
    source_columns = _normalize_source_schemas(
        source_schemas
    )
    target_columns = _normalize_columns(target_schema)
    issues: list[SchemaValidationIssue] = []
    seen: set[tuple[object, ...]] = set()
    checked_source_tables: set[str] = set()
    checked_step_count = 0

    for step in plan.steps:
        if step.lowering_mode == LoweringMode.GENERATE:
            checked_step_count += 1
            _validate_generated_step(
                step=step,
                source_columns=source_columns,
                target_columns=target_columns,
                issues=issues,
                seen=seen,
                checked_source_tables=checked_source_tables,
            )
        elif step.lowering_mode == LoweringMode.EXTERNAL:
            _add_issue(
                issues,
                seen,
                SchemaValidationIssue(
                    program_id=step.program_id,
                    primitive_id=step.primitive_id,
                    severity=SchemaIssueSeverity.WARNING,
                    scope=SchemaIssueScope.PROVIDER,
                    table_name="",
                    column_name=None,
                    code=(
                        SchemaIssueCode
                        .EXTERNAL_PROVIDER_REQUIRED
                    ),
                    message=(
                        "step requires an external provider "
                        "and was not validated against local "
                        "source schemas"
                    ),
                ),
            )
        elif step.lowering_mode == LoweringMode.UNSUPPORTED:
            _add_issue(
                issues,
                seen,
                SchemaValidationIssue(
                    program_id=step.program_id,
                    primitive_id=step.primitive_id,
                    severity=SchemaIssueSeverity.ERROR,
                    scope=SchemaIssueScope.STEP,
                    table_name="",
                    column_name=None,
                    code=SchemaIssueCode.UNSUPPORTED_STEP,
                    message=(
                        "unsupported materialization step "
                        "blocks schema validation"
                    ),
                ),
            )

    has_errors = any(
        issue.severity == SchemaIssueSeverity.ERROR
        for issue in issues
    )
    checked_table_count = len(checked_source_tables)

    if checked_step_count:
        checked_table_count += 1

    return SchemaValidationReport(
        program_id=plan.program_id,
        valid=not has_errors,
        checked_step_count=checked_step_count,
        checked_table_count=checked_table_count,
        issues=tuple(issues),
    )


def schema_validation_report_to_dict(
    report: SchemaValidationReport,
) -> dict[str, object]:
    return {
        "program_id": report.program_id,
        "valid": report.valid,
        "checked_step_count": report.checked_step_count,
        "checked_table_count": report.checked_table_count,
        "issue_count": len(report.issues),
        "issues": [
            {
                "program_id": issue.program_id,
                "primitive_id": issue.primitive_id,
                "severity": issue.severity.value,
                "scope": issue.scope.value,
                "table_name": issue.table_name,
                "column_name": issue.column_name,
                "code": issue.code.value,
                "message": issue.message,
            }
            for issue in report.issues
        ],
    }


def _validate_generated_step(
    *,
    step: PrimitiveMaterializationStep,
    source_columns: Mapping[str, frozenset[str]],
    target_columns: frozenset[str],
    issues: list[SchemaValidationIssue],
    seen: set[tuple[object, ...]],
    checked_source_tables: set[str],
) -> None:
    source_table = step.source_table

    if source_table is None:
        _add_source_table_issue(
            step=step,
            table_name="",
            issues=issues,
            seen=seen,
        )
    elif source_table not in source_columns:
        checked_source_tables.add(source_table)
        _add_source_table_issue(
            step=step,
            table_name=source_table,
            issues=issues,
            seen=seen,
        )
    else:
        checked_source_tables.add(source_table)
        for field_name in SOURCE_FIELD_ORDER:
            column = getattr(step, field_name)

            if (
                column is not None
                and column not in source_columns[source_table]
            ):
                _add_issue(
                    issues,
                    seen,
                    SchemaValidationIssue(
                        program_id=step.program_id,
                        primitive_id=step.primitive_id,
                        severity=SchemaIssueSeverity.ERROR,
                        scope=SchemaIssueScope.SOURCE,
                        table_name=source_table,
                        column_name=column,
                        code=(
                            SchemaIssueCode
                            .MISSING_SOURCE_COLUMN
                        ),
                        message=(
                            f"source column {source_table}."
                            f"{column} required by {field_name} "
                            "is missing"
                        ),
                    ),
                )

    for field_name in TARGET_FIELD_ORDER:
        column = getattr(step, field_name)

        if column is not None and column not in target_columns:
            _add_issue(
                issues,
                seen,
                SchemaValidationIssue(
                    program_id=step.program_id,
                    primitive_id=step.primitive_id,
                    severity=SchemaIssueSeverity.ERROR,
                    scope=SchemaIssueScope.TARGET,
                    table_name="target",
                    column_name=column,
                    code=SchemaIssueCode.MISSING_TARGET_COLUMN,
                    message=(
                        f"target column {column} required by "
                        f"{field_name} is missing"
                    ),
                ),
            )


def _add_source_table_issue(
    *,
    step: PrimitiveMaterializationStep,
    table_name: str,
    issues: list[SchemaValidationIssue],
    seen: set[tuple[object, ...]],
) -> None:
    _add_issue(
        issues,
        seen,
        SchemaValidationIssue(
            program_id=step.program_id,
            primitive_id=step.primitive_id,
            severity=SchemaIssueSeverity.ERROR,
            scope=SchemaIssueScope.SOURCE,
            table_name=table_name,
            column_name=None,
            code=SchemaIssueCode.MISSING_SOURCE_TABLE,
            message=(
                "source table required by generated step is "
                "missing"
            ),
        ),
    )


def _add_issue(
    issues: list[SchemaValidationIssue],
    seen: set[tuple[object, ...]],
    issue: SchemaValidationIssue,
) -> None:
    key = (
        issue.program_id,
        issue.primitive_id,
        issue.severity,
        issue.scope,
        issue.table_name,
        issue.column_name,
        issue.code,
    )

    if key in seen:
        return

    seen.add(key)
    issues.append(issue)


def _normalize_source_schemas(
    schemas: SchemaInput,
) -> dict[str, frozenset[str]]:
    normalized: dict[str, frozenset[str]] = {}

    for table_name, schema in schemas.items():
        if isinstance(schema, TableSchema):
            normalized[schema.table_name] = frozenset(
                schema.columns
            )
        else:
            normalized[str(table_name)] = frozenset(schema)

    return normalized


def _normalize_columns(
    schema: Collection[str] | TableSchema,
) -> frozenset[str]:
    if isinstance(schema, TableSchema):
        return frozenset(schema.columns)

    return frozenset(schema)
