from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .batch_evaluator import (
    BatchEvaluationError,
    BatchEvaluationResult,
    evaluate_generated_plan_rows,
)
from .index_builder import (
    IndexBuildError,
    build_temporal_index_registry,
)
from .materializer import CandidateMaterializationPlan
from .schema_validation import (
    SchemaValidationReport,
    TableSchema,
    validate_materialization_plan_schema,
)


class InMemoryMaterializationCode(str, Enum):
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    INDEX_BUILD_FAILED = "index_build_failed"
    BATCH_EVALUATION_FAILED = "batch_evaluation_failed"
    INVALID_SOURCE_ROWS = "invalid_source_rows"
    INVALID_TARGET_ROWS = "invalid_target_rows"


class InMemoryMaterializationError(ValueError):
    def __init__(
        self,
        *,
        code: InMemoryMaterializationCode,
        program_id: str,
        stage: str,
        message: str,
        underlying_code: str | None = None,
    ) -> None:
        self.code = code
        self.program_id = program_id
        self.stage = stage
        self.underlying_code = underlying_code
        detail = (
            f"{code.value}: program_id={program_id} "
            f"stage={stage}"
        )

        if underlying_code is not None:
            detail += f" underlying_code={underlying_code}"

        super().__init__(f"{detail}: {message}")


@dataclass(frozen=True)
class InMemoryMaterializationResult:
    program_id: str
    schema_report: SchemaValidationReport
    source_table_count: int
    target_row_count: int
    single_index_count: int
    pair_index_count: int
    generated_step_count: int
    output_columns: tuple[str, ...]
    batch_result: BatchEvaluationResult


SourceRowsByTable = Mapping[str, Iterable[Mapping[str, object]]]
TargetRows = Iterable[Mapping[str, object]]


def materialize_generated_features_in_memory(
    plan: CandidateMaterializationPlan,
    *,
    source_rows_by_table: SourceRowsByTable,
    target_rows: TargetRows,
) -> InMemoryMaterializationResult:
    source_rows = _materialize_source_rows(
        plan,
        source_rows_by_table,
    )
    target_row_tuple = _materialize_target_rows(
        plan,
        target_rows,
    )
    source_schemas = _source_schemas_from_rows(source_rows)
    target_schema = _target_schema_from_rows(target_row_tuple)
    schema_report = validate_materialization_plan_schema(
        plan,
        source_schemas=source_schemas,
        target_schema=target_schema,
    )

    if not schema_report.valid:
        _raise_schema_validation_failed(
            plan,
            schema_report,
        )

    try:
        registry = build_temporal_index_registry(
            plan,
            source_rows_by_table=source_rows,
        )
    except IndexBuildError as exc:
        raise InMemoryMaterializationError(
            code=(
                InMemoryMaterializationCode
                .INDEX_BUILD_FAILED
            ),
            program_id=plan.program_id,
            stage="index_build",
            underlying_code=exc.code.value,
            message=str(exc),
        ) from exc

    try:
        batch_result = evaluate_generated_plan_rows(
            plan,
            target_rows=target_row_tuple,
            indexes=registry,
        )
    except BatchEvaluationError as exc:
        raise InMemoryMaterializationError(
            code=(
                InMemoryMaterializationCode
                .BATCH_EVALUATION_FAILED
            ),
            program_id=plan.program_id,
            stage="batch_evaluation",
            underlying_code=exc.code.value,
            message=str(exc),
        ) from exc

    return InMemoryMaterializationResult(
        program_id=plan.program_id,
        schema_report=schema_report,
        source_table_count=len(source_rows),
        target_row_count=len(target_row_tuple),
        single_index_count=len(registry.single_key_indexes),
        pair_index_count=len(registry.pair_key_indexes),
        generated_step_count=batch_result.generated_step_count,
        output_columns=batch_result.output_columns,
        batch_result=batch_result,
    )


def _materialize_source_rows(
    plan: CandidateMaterializationPlan,
    source_rows_by_table: SourceRowsByTable,
) -> dict[str, tuple[Mapping[str, object], ...]]:
    try:
        return {
            str(table_name): tuple(rows)
            for table_name, rows in source_rows_by_table.items()
        }
    except TypeError as exc:
        raise InMemoryMaterializationError(
            code=(
                InMemoryMaterializationCode
                .INVALID_SOURCE_ROWS
            ),
            program_id=plan.program_id,
            stage="input_materialization",
            message=str(exc),
        ) from exc


def _materialize_target_rows(
    plan: CandidateMaterializationPlan,
    target_rows: TargetRows,
) -> tuple[Mapping[str, object], ...]:
    try:
        return tuple(target_rows)
    except TypeError as exc:
        raise InMemoryMaterializationError(
            code=(
                InMemoryMaterializationCode
                .INVALID_TARGET_ROWS
            ),
            program_id=plan.program_id,
            stage="input_materialization",
            message=str(exc),
        ) from exc


def _source_schemas_from_rows(
    source_rows: Mapping[str, tuple[Mapping[str, object], ...]],
) -> dict[str, TableSchema]:
    return {
        table_name: TableSchema(
            table_name=table_name,
            columns=_columns_from_rows(rows),
        )
        for table_name, rows in source_rows.items()
    }


def _target_schema_from_rows(
    target_rows: tuple[Mapping[str, object], ...],
) -> TableSchema:
    return TableSchema(
        table_name="target",
        columns=_columns_from_rows(target_rows),
    )


def _columns_from_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(str(column))

    return tuple(columns)


def _raise_schema_validation_failed(
    plan: CandidateMaterializationPlan,
    schema_report: SchemaValidationReport,
) -> None:
    details = "; ".join(
        f"code={issue.code.value} "
        f"table={issue.table_name} "
        f"column={issue.column_name} "
        f"primitive_id={issue.primitive_id}"
        for issue in schema_report.issues
    )
    raise InMemoryMaterializationError(
        code=(
            InMemoryMaterializationCode
            .SCHEMA_VALIDATION_FAILED
        ),
        program_id=plan.program_id,
        stage="schema_validation",
        underlying_code="schema_report_invalid",
        message=details or "schema validation failed",
    )
