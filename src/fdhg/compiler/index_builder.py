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
    PairKeyIndexKey,
    SingleKeyIndexKey,
    TemporalIndexRegistry,
)
from .temporal_index import (
    PairKeyEvent,
    PairKeyTemporalIndex,
    SingleKeyEvent,
    SingleKeyTemporalIndex,
)


class IndexBuildCode(str, Enum):
    PLAN_NOT_MATERIALIZABLE = "plan_not_materializable"
    PLAN_NOT_TEMPORALLY_SAFE = "plan_not_temporally_safe"
    EXTERNAL_STEP_PRESENT = "external_step_present"
    UNSUPPORTED_STEP_PRESENT = "unsupported_step_present"
    INCOMPLETE_SINGLE_BINDING = "incomplete_single_binding"
    INCOMPLETE_PAIR_BINDING = "incomplete_pair_binding"
    MISSING_SOURCE_TABLE = "missing_source_table"
    MISSING_SOURCE_COLUMN = "missing_source_column"
    NULL_GROUP_KEY = "null_group_key"
    NULL_PAIR_KEY = "null_pair_key"
    NULL_EVENT_TIME = "null_event_time"
    INVALID_EVENT_TIME_TYPE = "invalid_event_time_type"


class IndexBuildError(ValueError):
    def __init__(
        self,
        *,
        code: IndexBuildCode,
        program_id: str,
        message: str,
    ) -> None:
        self.code = code
        self.program_id = program_id
        super().__init__(
            f"{code.value}: program_id={program_id}: {message}"
        )


@dataclass(frozen=True)
class _SingleRequirement:
    key: SingleKeyIndexKey
    primitive_id: str


@dataclass(frozen=True)
class _PairRequirement:
    key: PairKeyIndexKey
    primitive_id: str


SourceRowsByTable = Mapping[str, Sequence[Mapping[str, object]]]


def build_temporal_index_registry(
    plan: CandidateMaterializationPlan,
    *,
    source_rows_by_table: SourceRowsByTable,
) -> TemporalIndexRegistry:
    _validate_plan_ready(plan)
    single_requirements, pair_requirements = _discover_requirements(
        plan
    )
    rows_by_table = {
        table: tuple(rows)
        for table, rows in source_rows_by_table.items()
    }

    for requirement in single_requirements:
        _require_source_table(
            plan,
            rows_by_table,
            requirement.key.source_table,
            requirement.primitive_id,
        )

    for requirement in pair_requirements:
        _require_source_table(
            plan,
            rows_by_table,
            requirement.key.source_table,
            requirement.primitive_id,
        )

    single_events = {
        requirement.key: []
        for requirement in single_requirements
    }
    pair_events = {
        requirement.key: []
        for requirement in pair_requirements
    }
    requirements_by_table: dict[
        str,
        list[_SingleRequirement | _PairRequirement],
    ] = {}

    for requirement in single_requirements:
        requirements_by_table.setdefault(
            requirement.key.source_table,
            [],
        ).append(requirement)

    for requirement in pair_requirements:
        requirements_by_table.setdefault(
            requirement.key.source_table,
            [],
        ).append(requirement)

    for table_name, requirements in requirements_by_table.items():
        for row_index, row in enumerate(rows_by_table[table_name]):
            for requirement in requirements:
                if isinstance(requirement, _SingleRequirement):
                    single_events[requirement.key].append(
                        _single_event_from_row(
                            plan=plan,
                            requirement=requirement,
                            row=row,
                            row_index=row_index,
                        )
                    )
                else:
                    pair_events[requirement.key].append(
                        _pair_event_from_row(
                            plan=plan,
                            requirement=requirement,
                            row=row,
                            row_index=row_index,
                        )
                    )

    try:
        single_indexes = {
            key: SingleKeyTemporalIndex(events)
            for key, events in single_events.items()
        }
        pair_indexes = {
            key: PairKeyTemporalIndex(events)
            for key, events in pair_events.items()
        }
    except (TypeError, ValueError) as exc:
        raise IndexBuildError(
            code=IndexBuildCode.INVALID_EVENT_TIME_TYPE,
            program_id=plan.program_id,
            message=f"invalid event-time values: {exc}",
        ) from exc

    return TemporalIndexRegistry(
        single_key_indexes=single_indexes,
        pair_key_indexes=pair_indexes,
    )


def _validate_plan_ready(
    plan: CandidateMaterializationPlan,
) -> None:
    if not plan.materializable:
        _raise(
            IndexBuildCode.PLAN_NOT_MATERIALIZABLE,
            plan.program_id,
            "plan is not materializable",
        )

    if not plan.temporally_safe:
        _raise(
            IndexBuildCode.PLAN_NOT_TEMPORALLY_SAFE,
            plan.program_id,
            "plan is not temporally safe",
        )

    for step in plan.steps:
        if step.lowering_mode == LoweringMode.EXTERNAL:
            _raise(
                IndexBuildCode.EXTERNAL_STEP_PRESENT,
                plan.program_id,
                f"external step present: "
                f"primitive_id={step.primitive_id}",
            )
        if step.lowering_mode == LoweringMode.UNSUPPORTED:
            _raise(
                IndexBuildCode.UNSUPPORTED_STEP_PRESENT,
                plan.program_id,
                f"unsupported step present: "
                f"primitive_id={step.primitive_id}",
            )


def _discover_requirements(
    plan: CandidateMaterializationPlan,
) -> tuple[
    tuple[_SingleRequirement, ...],
    tuple[_PairRequirement, ...],
]:
    single_requirements: list[_SingleRequirement] = []
    pair_requirements: list[_PairRequirement] = []
    seen_single: set[SingleKeyIndexKey] = set()
    seen_pair: set[PairKeyIndexKey] = set()

    for step in plan.steps:
        if step.lowering_mode != LoweringMode.GENERATE:
            continue

        if step.operation in {
            "window_count",
            "past_unique_neighbors",
            "days_since_last",
        }:
            key = _single_key_for_step(plan, step)

            if key not in seen_single:
                seen_single.add(key)
                single_requirements.append(
                    _SingleRequirement(
                        key=key,
                        primitive_id=step.primitive_id,
                    )
                )
            continue

        if step.operation in {
            "prior_pair_count",
            "pair_days_since_last",
        }:
            key = _pair_key_for_step(plan, step)

            if key not in seen_pair:
                seen_pair.add(key)
                pair_requirements.append(
                    _PairRequirement(
                        key=key,
                        primitive_id=step.primitive_id,
                    )
                )

    return tuple(single_requirements), tuple(pair_requirements)


def _single_key_for_step(
    plan: CandidateMaterializationPlan,
    step: PrimitiveMaterializationStep,
) -> SingleKeyIndexKey:
    if (
        step.source_table is None
        or step.source_group_key is None
        or step.source_event_time_col is None
    ):
        _raise(
            IndexBuildCode.INCOMPLETE_SINGLE_BINDING,
            plan.program_id,
            f"incomplete single-key binding: "
            f"primitive_id={step.primitive_id} "
            f"source_table={step.source_table!r}",
        )

    if (
        step.operation == "past_unique_neighbors"
        and step.related_col is None
    ):
        _raise(
            IndexBuildCode.INCOMPLETE_SINGLE_BINDING,
            plan.program_id,
            f"missing related_col for primitive_id="
            f"{step.primitive_id}",
        )

    return SingleKeyIndexKey(
        source_table=step.source_table,
        source_group_key=step.source_group_key,
        source_event_time_col=step.source_event_time_col,
        related_col=step.related_col,
    )


def _pair_key_for_step(
    plan: CandidateMaterializationPlan,
    step: PrimitiveMaterializationStep,
) -> PairKeyIndexKey:
    if (
        step.source_table is None
        or step.source_left_key is None
        or step.source_right_key is None
        or step.source_event_time_col is None
    ):
        _raise(
            IndexBuildCode.INCOMPLETE_PAIR_BINDING,
            plan.program_id,
            f"incomplete pair binding: primitive_id="
            f"{step.primitive_id} source_table="
            f"{step.source_table!r}",
        )

    return PairKeyIndexKey(
        source_table=step.source_table,
        source_left_key=step.source_left_key,
        source_right_key=step.source_right_key,
        source_event_time_col=step.source_event_time_col,
    )


def _single_event_from_row(
    *,
    plan: CandidateMaterializationPlan,
    requirement: _SingleRequirement,
    row: Mapping[str, object],
    row_index: int,
) -> SingleKeyEvent:
    key = requirement.key
    group_value = _required_value(
        plan=plan,
        primitive_id=requirement.primitive_id,
        row=row,
        row_index=row_index,
        table_name=key.source_table,
        column_name=key.source_group_key,
        null_code=IndexBuildCode.NULL_GROUP_KEY,
    )
    event_time = _required_value(
        plan=plan,
        primitive_id=requirement.primitive_id,
        row=row,
        row_index=row_index,
        table_name=key.source_table,
        column_name=key.source_event_time_col,
        null_code=IndexBuildCode.NULL_EVENT_TIME,
    )
    related = (
        _optional_value(
            plan=plan,
            primitive_id=requirement.primitive_id,
            row=row,
            row_index=row_index,
            table_name=key.source_table,
            column_name=key.related_col,
        )
        if key.related_col is not None
        else None
    )
    return SingleKeyEvent(
        group_key=group_value,
        event_time=event_time,
        related_value=related,
    )


def _pair_event_from_row(
    *,
    plan: CandidateMaterializationPlan,
    requirement: _PairRequirement,
    row: Mapping[str, object],
    row_index: int,
) -> PairKeyEvent:
    key = requirement.key
    left_value = _required_value(
        plan=plan,
        primitive_id=requirement.primitive_id,
        row=row,
        row_index=row_index,
        table_name=key.source_table,
        column_name=key.source_left_key,
        null_code=IndexBuildCode.NULL_PAIR_KEY,
    )
    right_value = _required_value(
        plan=plan,
        primitive_id=requirement.primitive_id,
        row=row,
        row_index=row_index,
        table_name=key.source_table,
        column_name=key.source_right_key,
        null_code=IndexBuildCode.NULL_PAIR_KEY,
    )
    event_time = _required_value(
        plan=plan,
        primitive_id=requirement.primitive_id,
        row=row,
        row_index=row_index,
        table_name=key.source_table,
        column_name=key.source_event_time_col,
        null_code=IndexBuildCode.NULL_EVENT_TIME,
    )
    return PairKeyEvent(
        left_key=left_value,
        right_key=right_value,
        event_time=event_time,
    )


def _required_value(
    *,
    plan: CandidateMaterializationPlan,
    primitive_id: str,
    row: Mapping[str, object],
    row_index: int,
    table_name: str,
    column_name: str,
    null_code: IndexBuildCode,
) -> object:
    if column_name not in row:
        _raise(
            IndexBuildCode.MISSING_SOURCE_COLUMN,
            plan.program_id,
            f"primitive_id={primitive_id} "
            f"source_table={table_name} "
            f"row_index={row_index} "
            f"column_name={column_name}",
        )

    value = row[column_name]

    if value is None:
        _raise(
            null_code,
            plan.program_id,
            f"primitive_id={primitive_id} "
            f"source_table={table_name} "
            f"row_index={row_index} "
            f"column_name={column_name}",
        )

    return value


def _optional_value(
    *,
    plan: CandidateMaterializationPlan,
    primitive_id: str,
    row: Mapping[str, object],
    row_index: int,
    table_name: str,
    column_name: str,
) -> object | None:
    if column_name not in row:
        _raise(
            IndexBuildCode.MISSING_SOURCE_COLUMN,
            plan.program_id,
            f"primitive_id={primitive_id} "
            f"source_table={table_name} "
            f"row_index={row_index} "
            f"column_name={column_name}",
        )

    return row[column_name]


def _require_source_table(
    plan: CandidateMaterializationPlan,
    rows_by_table: Mapping[str, tuple[Mapping[str, object], ...]],
    table_name: str,
    primitive_id: str,
) -> None:
    if table_name not in rows_by_table:
        _raise(
            IndexBuildCode.MISSING_SOURCE_TABLE,
            plan.program_id,
            f"primitive_id={primitive_id} "
            f"source_table={table_name}",
        )


def _raise(
    code: IndexBuildCode,
    program_id: str,
    message: str,
) -> None:
    raise IndexBuildError(
        code=code,
        program_id=program_id,
        message=message,
    )
