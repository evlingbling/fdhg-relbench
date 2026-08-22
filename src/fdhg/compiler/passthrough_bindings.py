from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
)


class PassthroughBindingCode(str, Enum):
    PROGRAM_ID_MISMATCH = "program_id_mismatch"
    UNKNOWN_PRIMITIVE_ID = "unknown_primitive_id"
    NON_PASSTHROUGH_PRIMITIVE = "non_passthrough_primitive"
    MISSING_PASSTHROUGH_BINDING = "missing_passthrough_binding"
    DUPLICATE_SOURCE_BINDING = "duplicate_source_binding"
    DUPLICATE_OUTPUT_COLUMN = "duplicate_output_column"
    INVALID_BINDING = "invalid_binding"
    UNRESOLVED_PASSTHROUGH_STEP = "unresolved_passthrough_step"
    MISSING_SOURCE_COLUMN = "missing_source_column"


class PassthroughBindingError(ValueError):
    def __init__(
        self,
        *,
        code: PassthroughBindingCode,
        program_id: str,
        message: str,
    ) -> None:
        self.code = code
        self.program_id = program_id
        super().__init__(
            f"{code.value}: program_id={program_id}: {message}"
        )


@dataclass(frozen=True)
class PassthroughColumnBinding:
    program_id: str
    primitive_id: str
    source_column: str
    output_column: str


@dataclass(frozen=True)
class ResolvedPassthroughContract:
    program_id: str
    bindings: tuple[PassthroughColumnBinding, ...]
    source_columns: tuple[str, ...]
    output_columns: tuple[str, ...]


BindingInput = Sequence[tuple[str, str]]


def resolve_passthrough_bindings(
    plan: CandidateMaterializationPlan,
    *,
    explicit_bindings: Mapping[str, BindingInput],
) -> ResolvedPassthroughContract:
    steps_by_id = _steps_by_id(plan)
    passthrough_ids = tuple(
        step.primitive_id
        for step in plan.steps
        if step.lowering_mode == LoweringMode.PASSTHROUGH
    )

    for primitive_id in explicit_bindings:
        step = steps_by_id.get(primitive_id)
        if step is None:
            _raise(
                PassthroughBindingCode.UNKNOWN_PRIMITIVE_ID,
                plan.program_id,
                f"primitive_id={primitive_id}",
            )
        if step.lowering_mode != LoweringMode.PASSTHROUGH:
            _raise(
                PassthroughBindingCode.NON_PASSTHROUGH_PRIMITIVE,
                plan.program_id,
                f"primitive_id={primitive_id} "
                f"lowering_mode={step.lowering_mode.value}",
            )

    resolved: list[PassthroughColumnBinding] = []
    seen_output_columns: set[str] = set()
    seen_primitive_sources: set[tuple[str, str]] = set()

    for primitive_id in passthrough_ids:
        raw_bindings = explicit_bindings.get(primitive_id)
        if raw_bindings is None:
            _raise(
                PassthroughBindingCode
                .MISSING_PASSTHROUGH_BINDING,
                plan.program_id,
                f"primitive_id={primitive_id}",
            )
        if len(raw_bindings) == 0:
            _raise(
                PassthroughBindingCode
                .UNRESOLVED_PASSTHROUGH_STEP,
                plan.program_id,
                f"primitive_id={primitive_id}",
            )

        for index, pair in enumerate(raw_bindings):
            source_column, output_column = _parse_binding(
                plan.program_id,
                primitive_id,
                index,
                pair,
            )
            primitive_source = (primitive_id, source_column)
            if primitive_source in seen_primitive_sources:
                _raise(
                    PassthroughBindingCode
                    .DUPLICATE_SOURCE_BINDING,
                    plan.program_id,
                    f"primitive_id={primitive_id} "
                    f"source_column={source_column}",
                )
            seen_primitive_sources.add(primitive_source)

            if output_column in seen_output_columns:
                _raise(
                    PassthroughBindingCode
                    .DUPLICATE_OUTPUT_COLUMN,
                    plan.program_id,
                    f"primitive_id={primitive_id} "
                    f"output_column={output_column}",
                )
            seen_output_columns.add(output_column)
            resolved.append(
                PassthroughColumnBinding(
                    program_id=plan.program_id,
                    primitive_id=primitive_id,
                    source_column=source_column,
                    output_column=output_column,
                )
            )

    bindings = tuple(resolved)
    return ResolvedPassthroughContract(
        program_id=plan.program_id,
        bindings=bindings,
        source_columns=tuple(
            binding.source_column for binding in bindings
        ),
        output_columns=tuple(
            binding.output_column for binding in bindings
        ),
    )


def passthrough_contract_from_declared_outputs(
    plan: CandidateMaterializationPlan,
) -> ResolvedPassthroughContract:
    explicit: dict[str, tuple[tuple[str, str], ...]] = {}

    for step in plan.steps:
        if step.lowering_mode != LoweringMode.PASSTHROUGH:
            continue
        explicit[step.primitive_id] = tuple(
            (column, column)
            for column in step.output_columns
        )

    return resolve_passthrough_bindings(
        plan,
        explicit_bindings=explicit,
    )


def validate_passthrough_rows(
    contract: ResolvedPassthroughContract,
    *,
    target_rows: Sequence[Mapping[str, object]],
) -> None:
    for row_index, row in enumerate(target_rows):
        for binding in contract.bindings:
            if binding.source_column not in row:
                _raise(
                    PassthroughBindingCode.MISSING_SOURCE_COLUMN,
                    contract.program_id,
                    f"primitive_id={binding.primitive_id} "
                    f"row_index={row_index} "
                    f"source_column={binding.source_column} "
                    f"output_column={binding.output_column}",
                )


def passthrough_contract_to_records(
    contract: ResolvedPassthroughContract,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "program_id": binding.program_id,
            "primitive_id": binding.primitive_id,
            "source_column": binding.source_column,
            "output_column": binding.output_column,
        }
        for binding in contract.bindings
    )


def _steps_by_id(
    plan: CandidateMaterializationPlan,
):
    steps = {}

    for step in plan.steps:
        if step.primitive_id in steps:
            _raise(
                PassthroughBindingCode.INVALID_BINDING,
                plan.program_id,
                f"duplicate primitive_id={step.primitive_id}",
            )
        steps[step.primitive_id] = step

    return steps


def _parse_binding(
    program_id: str,
    primitive_id: str,
    index: int,
    pair: tuple[str, str],
) -> tuple[str, str]:
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
    ):
        _raise(
            PassthroughBindingCode.INVALID_BINDING,
            program_id,
            f"primitive_id={primitive_id} binding_index={index}",
        )

    source_column, output_column = pair
    if not isinstance(source_column, str) or source_column == "":
        _raise(
            PassthroughBindingCode.INVALID_BINDING,
            program_id,
            f"primitive_id={primitive_id} "
            f"binding_index={index} source_column={source_column!r}",
        )
    if not isinstance(output_column, str) or output_column == "":
        _raise(
            PassthroughBindingCode.INVALID_BINDING,
            program_id,
            f"primitive_id={primitive_id} "
            f"binding_index={index} output_column={output_column!r}",
        )

    return source_column, output_column


def _raise(
    code: PassthroughBindingCode,
    program_id: str,
    message: str,
) -> None:
    raise PassthroughBindingError(
        code=code,
        program_id=program_id,
        message=message,
    )
