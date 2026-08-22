from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    PrimitiveMaterializationStep,
)
from .primitive_evaluator import (
    PrimitiveEvaluationError,
    TemporalIndexRegistry,
    evaluate_generated_step,
)


class BatchEvaluationCode(str, Enum):
    PLAN_NOT_MATERIALIZABLE = "plan_not_materializable"
    PLAN_NOT_TEMPORALLY_SAFE = "plan_not_temporally_safe"
    EXTERNAL_STEP_PRESENT = "external_step_present"
    UNSUPPORTED_STEP_PRESENT = "unsupported_step_present"
    DUPLICATE_OUTPUT_COLUMN = "duplicate_output_column"
    EMPTY_GENERATED_OUTPUT = "empty_generated_output"
    ROW_EVALUATION_FAILED = "row_evaluation_failed"
    SCALAR_OUTPUT_CONTRACT_MISMATCH = (
        "scalar_output_contract_mismatch"
    )


class BatchEvaluationError(ValueError):
    def __init__(
        self,
        *,
        code: BatchEvaluationCode,
        program_id: str,
        message: str,
    ) -> None:
        self.code = code
        self.program_id = program_id
        super().__init__(
            f"{code.value}: program_id={program_id}: {message}"
        )


@dataclass(frozen=True)
class EvaluatedFeatureRow:
    row_index: int
    values: tuple[tuple[str, int | float], ...]


@dataclass(frozen=True)
class BatchEvaluationResult:
    program_id: str
    generated_step_count: int
    output_columns: tuple[str, ...]
    rows: tuple[EvaluatedFeatureRow, ...]


TargetRow = Mapping[str, object]


def evaluate_generated_plan_rows(
    plan: CandidateMaterializationPlan,
    *,
    target_rows: Sequence[TargetRow],
    indexes: TemporalIndexRegistry,
) -> BatchEvaluationResult:
    generated_steps = _validate_plan_ready(plan)
    output_columns = _collect_output_columns(
        plan,
        generated_steps,
    )
    materialized_rows = tuple(target_rows)
    evaluated_rows: list[EvaluatedFeatureRow] = []

    for row_index, target_row in enumerate(materialized_rows):
        row_values: list[tuple[str, int | float]] = []

        for step in generated_steps:
            try:
                step_values = evaluate_generated_step(
                    step,
                    target_row=target_row,
                    indexes=indexes,
                )
            except PrimitiveEvaluationError as exc:
                raise BatchEvaluationError(
                    code=(
                        BatchEvaluationCode
                        .ROW_EVALUATION_FAILED
                    ),
                    program_id=plan.program_id,
                    message=(
                        f"row_index={row_index} "
                        f"primitive_id={step.primitive_id} "
                        f"operation={step.operation}: {exc}"
                    ),
                ) from exc

            _validate_scalar_output(
                plan=plan,
                step=step,
                row_index=row_index,
                step_values=step_values,
            )
            row_values.extend(step_values.items())

        evaluated_rows.append(
            EvaluatedFeatureRow(
                row_index=row_index,
                values=tuple(row_values),
            )
        )

    return BatchEvaluationResult(
        program_id=plan.program_id,
        generated_step_count=len(generated_steps),
        output_columns=output_columns,
        rows=tuple(evaluated_rows),
    )


def _validate_plan_ready(
    plan: CandidateMaterializationPlan,
) -> tuple[PrimitiveMaterializationStep, ...]:
    if not plan.materializable:
        _raise(
            BatchEvaluationCode.PLAN_NOT_MATERIALIZABLE,
            plan.program_id,
            "plan is not materializable",
        )

    if not plan.temporally_safe:
        _raise(
            BatchEvaluationCode.PLAN_NOT_TEMPORALLY_SAFE,
            plan.program_id,
            "plan is not temporally safe",
        )

    for step in plan.steps:
        if step.lowering_mode == LoweringMode.EXTERNAL:
            _raise(
                BatchEvaluationCode.EXTERNAL_STEP_PRESENT,
                plan.program_id,
                f"external step present: "
                f"primitive_id={step.primitive_id} "
                f"operation={step.operation}",
            )

        if step.lowering_mode == LoweringMode.UNSUPPORTED:
            _raise(
                BatchEvaluationCode.UNSUPPORTED_STEP_PRESENT,
                plan.program_id,
                f"unsupported step present: "
                f"primitive_id={step.primitive_id} "
                f"operation={step.operation}",
            )

    return tuple(
        step
        for step in plan.steps
        if step.lowering_mode == LoweringMode.GENERATE
    )


def _collect_output_columns(
    plan: CandidateMaterializationPlan,
    generated_steps: tuple[PrimitiveMaterializationStep, ...],
) -> tuple[str, ...]:
    output_columns: list[str] = []
    seen: set[str] = set()

    for step in generated_steps:
        if not step.output_columns:
            _raise(
                BatchEvaluationCode.EMPTY_GENERATED_OUTPUT,
                plan.program_id,
                f"generated step has no output columns: "
                f"primitive_id={step.primitive_id} "
                f"operation={step.operation}",
            )

        for column in step.output_columns:
            if column in seen:
                _raise(
                    BatchEvaluationCode.DUPLICATE_OUTPUT_COLUMN,
                    plan.program_id,
                    f"duplicate output column {column!r} "
                    f"from primitive_id={step.primitive_id} "
                    f"operation={step.operation}",
                )
            seen.add(column)
            output_columns.append(column)

    return tuple(output_columns)


def _validate_scalar_output(
    *,
    plan: CandidateMaterializationPlan,
    step: PrimitiveMaterializationStep,
    row_index: int,
    step_values: Mapping[str, int | float],
) -> None:
    expected = tuple(step.output_columns)
    actual = tuple(step_values.keys())

    if actual != expected:
        _raise(
            BatchEvaluationCode.SCALAR_OUTPUT_CONTRACT_MISMATCH,
            plan.program_id,
            f"row_index={row_index} "
            f"primitive_id={step.primitive_id} "
            f"operation={step.operation}: "
            f"expected output columns {expected}, found {actual}",
        )


def _raise(
    code: BatchEvaluationCode,
    program_id: str,
    message: str,
) -> None:
    raise BatchEvaluationError(
        code=code,
        program_id=program_id,
        message=message,
    )
