from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .batch_evaluator import BatchEvaluationResult
from .materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    PrimitiveMaterializationStep,
)
from .passthrough_bindings import ResolvedPassthroughContract


class MatrixAssemblyCode(str, Enum):
    PROGRAM_ID_MISMATCH = "program_id_mismatch"
    ROW_COUNT_MISMATCH = "row_count_mismatch"
    ROW_INDEX_MISMATCH = "row_index_mismatch"
    DUPLICATE_COLUMN = "duplicate_column"
    MISSING_IDENTITY_COLUMN = "missing_identity_column"
    MISSING_PASSTHROUGH_COLUMN = "missing_passthrough_column"
    GENERATED_COLUMN_MISMATCH = "generated_column_mismatch"
    EXTERNAL_STEP_PRESENT = "external_step_present"
    UNSUPPORTED_STEP_PRESENT = "unsupported_step_present"
    PLAN_NOT_MATERIALIZABLE = "plan_not_materializable"
    PLAN_NOT_TEMPORALLY_SAFE = "plan_not_temporally_safe"


class MatrixAssemblyError(ValueError):
    def __init__(
        self,
        *,
        code: MatrixAssemblyCode,
        program_id: str,
        message: str,
    ) -> None:
        self.code = code
        self.program_id = program_id
        super().__init__(
            f"{code.value}: program_id={program_id}: {message}"
        )


@dataclass(frozen=True)
class CandidateMatrixRow:
    row_index: int
    values: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class CandidateMatrix:
    program_id: str
    identity_columns: tuple[str, ...]
    passthrough_columns: tuple[str, ...]
    generated_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    rows: tuple[CandidateMatrixRow, ...]


TargetRow = Mapping[str, object]


def assemble_candidate_matrix(
    plan: CandidateMaterializationPlan,
    *,
    target_rows: Sequence[TargetRow],
    generated: BatchEvaluationResult,
    identity_columns: Sequence[str],
    passthrough_contract: ResolvedPassthroughContract,
) -> CandidateMatrix:
    _validate_plan_ready(plan)
    _validate_program_id(plan, generated)
    _validate_contract_program_id(plan, passthrough_contract)
    identity = tuple(identity_columns)
    _reject_duplicates(
        plan.program_id,
        identity,
        "identity columns",
    )
    _validate_contract_primitives(plan, passthrough_contract)
    passthrough = passthrough_contract.output_columns
    _reject_duplicates(
        plan.program_id,
        passthrough,
        "passthrough columns",
    )
    expected_generated = _generated_columns_from_plan(plan)

    if generated.output_columns != expected_generated:
        _raise(
            MatrixAssemblyCode.GENERATED_COLUMN_MISMATCH,
            plan.program_id,
            "generated.output_columns do not match plan "
            f"generated columns: expected={expected_generated!r} "
            f"actual={generated.output_columns!r}",
        )

    _validate_column_partitions(
        plan.program_id,
        identity,
        passthrough,
        generated.output_columns,
    )
    _validate_row_alignment(plan, target_rows, generated)

    rows: list[CandidateMatrixRow] = []

    for row_index, target_row in enumerate(target_rows):
        values: list[tuple[str, object]] = []

        for column in identity:
            values.append((
                column,
                _target_value(
                    plan.program_id,
                    target_row,
                    row_index,
                    column,
                    MatrixAssemblyCode.MISSING_IDENTITY_COLUMN,
                ),
            ))

        for binding in passthrough_contract.bindings:
            values.append((
                binding.output_column,
                _target_value(
                    plan.program_id,
                    target_row,
                    row_index,
                    binding.source_column,
                    (
                        MatrixAssemblyCode
                        .MISSING_PASSTHROUGH_COLUMN
                    ),
                ),
            ))

        generated_values = tuple(
            generated.rows[row_index].values
        )

        if tuple(name for name, _ in generated_values) != (
            generated.output_columns
        ):
            _raise(
                MatrixAssemblyCode.GENERATED_COLUMN_MISMATCH,
                plan.program_id,
                f"generated row value keys do not match output "
                f"columns: row_index={row_index}",
            )

        values.extend(generated_values)
        rows.append(
            CandidateMatrixRow(
                row_index=row_index,
                values=tuple(values),
            )
        )

    output_columns = identity + passthrough + generated.output_columns
    return CandidateMatrix(
        program_id=plan.program_id,
        identity_columns=identity,
        passthrough_columns=passthrough,
        generated_columns=generated.output_columns,
        output_columns=output_columns,
        rows=tuple(rows),
    )


def _validate_plan_ready(
    plan: CandidateMaterializationPlan,
) -> None:
    if not plan.materializable:
        _raise(
            MatrixAssemblyCode.PLAN_NOT_MATERIALIZABLE,
            plan.program_id,
            "plan is not materializable",
        )

    if not plan.temporally_safe:
        _raise(
            MatrixAssemblyCode.PLAN_NOT_TEMPORALLY_SAFE,
            plan.program_id,
            "plan is not temporally safe",
        )

    for step in plan.steps:
        if step.lowering_mode == LoweringMode.EXTERNAL:
            _raise(
                MatrixAssemblyCode.EXTERNAL_STEP_PRESENT,
                plan.program_id,
                f"external step present: "
                f"primitive_id={step.primitive_id}",
            )

        if step.lowering_mode == LoweringMode.UNSUPPORTED:
            _raise(
                MatrixAssemblyCode.UNSUPPORTED_STEP_PRESENT,
                plan.program_id,
                f"unsupported step present: "
                f"primitive_id={step.primitive_id}",
            )


def _validate_program_id(
    plan: CandidateMaterializationPlan,
    generated: BatchEvaluationResult,
) -> None:
    if plan.program_id != generated.program_id:
        _raise(
            MatrixAssemblyCode.PROGRAM_ID_MISMATCH,
            plan.program_id,
            f"generated program_id={generated.program_id!r}",
        )


def _validate_contract_program_id(
    plan: CandidateMaterializationPlan,
    contract: ResolvedPassthroughContract,
) -> None:
    if plan.program_id != contract.program_id:
        _raise(
            MatrixAssemblyCode.PROGRAM_ID_MISMATCH,
            plan.program_id,
            f"passthrough_contract "
            f"program_id={contract.program_id!r}",
        )


def _validate_contract_primitives(
    plan: CandidateMaterializationPlan,
    contract: ResolvedPassthroughContract,
) -> None:
    step_modes = {
        step.primitive_id: step.lowering_mode
        for step in plan.steps
    }
    bound_ids = {
        binding.primitive_id
        for binding in contract.bindings
    }

    for binding in contract.bindings:
        mode = step_modes.get(binding.primitive_id)
        if mode is None:
            _raise(
                MatrixAssemblyCode.GENERATED_COLUMN_MISMATCH,
                plan.program_id,
                "passthrough contract references unknown "
                f"primitive_id={binding.primitive_id}",
            )
        if mode != LoweringMode.PASSTHROUGH:
            _raise(
                MatrixAssemblyCode.GENERATED_COLUMN_MISMATCH,
                plan.program_id,
                "passthrough contract references "
                f"non-passthrough primitive_id="
                f"{binding.primitive_id} lowering_mode={mode.value}",
            )

    for step in plan.steps:
        if (
            step.lowering_mode == LoweringMode.PASSTHROUGH
            and step.primitive_id not in bound_ids
        ):
            _raise(
                MatrixAssemblyCode.MISSING_PASSTHROUGH_COLUMN,
                plan.program_id,
                "passthrough contract has no binding for "
                f"primitive_id={step.primitive_id}",
            )


def _generated_columns_from_plan(
    plan: CandidateMaterializationPlan,
) -> tuple[str, ...]:
    columns: list[str] = []

    for step in plan.steps:
        if step.lowering_mode == LoweringMode.GENERATE:
            columns.extend(step.output_columns)

    return tuple(columns)


def _validate_column_partitions(
    program_id: str,
    identity: tuple[str, ...],
    passthrough: tuple[str, ...],
    generated: tuple[str, ...],
) -> None:
    seen: set[str] = set()

    for column in identity + passthrough + generated:
        if column in seen:
            _raise(
                MatrixAssemblyCode.DUPLICATE_COLUMN,
                program_id,
                f"duplicate column_name={column}",
            )
        seen.add(column)


def _validate_row_alignment(
    plan: CandidateMaterializationPlan,
    target_rows: Sequence[TargetRow],
    generated: BatchEvaluationResult,
) -> None:
    if len(target_rows) != len(generated.rows):
        _raise(
            MatrixAssemblyCode.ROW_COUNT_MISMATCH,
            plan.program_id,
            f"target_rows={len(target_rows)} "
            f"generated_rows={len(generated.rows)}",
        )

    seen: set[int] = set()

    for expected_index, row in enumerate(generated.rows):
        if row.row_index in seen:
            _raise(
                MatrixAssemblyCode.ROW_INDEX_MISMATCH,
                plan.program_id,
                f"duplicate row_index={row.row_index}",
            )
        seen.add(row.row_index)

        if row.row_index != expected_index:
            _raise(
                MatrixAssemblyCode.ROW_INDEX_MISMATCH,
                plan.program_id,
                f"expected row_index={expected_index} "
                f"actual row_index={row.row_index}",
            )


def _target_value(
    program_id: str,
    target_row: TargetRow,
    row_index: int,
    column: str,
    code: MatrixAssemblyCode,
) -> object:
    if column not in target_row:
        _raise(
            code,
            program_id,
            f"row_index={row_index} column_name={column}",
        )

    return target_row[column]


def _reject_duplicates(
    program_id: str,
    columns: tuple[str, ...],
    label: str,
) -> None:
    seen: set[str] = set()

    for column in columns:
        if column in seen:
            _raise(
                MatrixAssemblyCode.DUPLICATE_COLUMN,
                program_id,
                f"duplicate {label}: column_name={column}",
            )
        seen.add(column)


def _raise(
    code: MatrixAssemblyCode,
    program_id: str,
    message: str,
) -> None:
    raise MatrixAssemblyError(
        code=code,
        program_id=program_id,
        message=message,
    )
