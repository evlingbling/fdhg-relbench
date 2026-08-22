from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from numbers import Real
from typing import Hashable, Mapping

from .materializer import (
    LoweringMode,
    PrimitiveMaterializationStep,
)
from .temporal_index import (
    PairKeyTemporalIndex,
    SingleKeyTemporalIndex,
    TimeValue,
    WindowValue,
)


class PrimitiveEvaluationCode(str, Enum):
    NON_GENERATE_STEP = "non_generate_step"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    MISSING_TARGET_VALUE = "missing_target_value"
    MISSING_TARGET_TIME = "missing_target_time"
    MISSING_INDEX = "missing_index"
    INVALID_OUTPUT_COLUMNS = "invalid_output_columns"
    INCOMPLETE_BINDING = "incomplete_binding"
    UNSUPPORTED_STEP = "unsupported_step"


class PrimitiveEvaluationError(ValueError):
    def __init__(
        self,
        *,
        code: PrimitiveEvaluationCode,
        step: PrimitiveMaterializationStep,
        message: str,
    ) -> None:
        self.code = code
        self.step = step
        super().__init__(
            f"{code.value}: program_id={step.program_id} "
            f"primitive_id={step.primitive_id} "
            f"operation={step.operation}: {message}"
        )


@dataclass(frozen=True)
class SingleKeyIndexKey:
    source_table: str
    source_group_key: str
    source_event_time_col: str
    related_col: str | None = None


@dataclass(frozen=True)
class PairKeyIndexKey:
    source_table: str
    source_left_key: str
    source_right_key: str
    source_event_time_col: str


@dataclass(frozen=True)
class TemporalIndexRegistry:
    single_key_indexes: Mapping[
        SingleKeyIndexKey,
        SingleKeyTemporalIndex,
    ]
    pair_key_indexes: Mapping[
        PairKeyIndexKey,
        PairKeyTemporalIndex,
    ]


TargetRow = Mapping[str, object]


def evaluate_generated_step(
    step: PrimitiveMaterializationStep,
    *,
    target_row: TargetRow,
    indexes: TemporalIndexRegistry,
) -> dict[str, int | float]:
    if step.lowering_mode == LoweringMode.UNSUPPORTED:
        _raise(
            PrimitiveEvaluationCode.UNSUPPORTED_STEP,
            step,
            "unsupported materialization steps cannot be evaluated",
        )

    if step.lowering_mode != LoweringMode.GENERATE:
        _raise(
            PrimitiveEvaluationCode.NON_GENERATE_STEP,
            step,
            "only generated materialization steps can be evaluated",
        )

    if step.operation == "window_count":
        _require_output_columns(step, 1)
        index = _single_key_index(step, indexes)
        group_value = _target_value(
            step,
            step.target_key,
            target_row,
        )
        target_time = _target_time(step, target_row)
        return {
            step.output_columns[0]: index.window_count_before(
                group_value,
                target_time,
                _window_for_target(step, target_time),
            )
        }

    if step.operation == "past_unique_neighbors":
        _require_output_columns(step, 1)
        index = _single_key_index(step, indexes)
        group_value = _target_value(
            step,
            step.target_key,
            target_row,
        )
        target_time = _target_time(step, target_row)
        return {
            step.output_columns[0]: index.unique_related_before(
                group_value,
                target_time,
                _window_for_target(step, target_time),
            )
        }

    if step.operation == "days_since_last":
        _require_output_columns(step, 2)
        index = _single_key_index(step, indexes)
        group_value = _target_value(
            step,
            step.target_key,
            target_row,
        )
        target_time = _target_time(step, target_row)
        days = index.days_since_last_before(
            group_value,
            target_time,
        )
        return _recency_outputs(step, days)

    if step.operation == "prior_pair_count":
        _require_output_columns(step, 1)
        index = _pair_key_index(step, indexes)
        left_value = _target_value(
            step,
            step.target_left_key,
            target_row,
        )
        right_value = _target_value(
            step,
            step.target_right_key,
            target_row,
        )
        target_time = _target_time(step, target_row)
        return {
            step.output_columns[0]: index.count_before(
                left_value,
                right_value,
                target_time,
            )
        }

    if step.operation == "pair_days_since_last":
        _require_output_columns(step, 2)
        index = _pair_key_index(step, indexes)
        left_value = _target_value(
            step,
            step.target_left_key,
            target_row,
        )
        right_value = _target_value(
            step,
            step.target_right_key,
            target_row,
        )
        target_time = _target_time(step, target_row)
        days = index.days_since_last_before(
            left_value,
            right_value,
            target_time,
        )
        return _recency_outputs(step, days)

    _raise(
        PrimitiveEvaluationCode.UNSUPPORTED_OPERATION,
        step,
        "operation is not supported by the scalar evaluator",
    )


def _single_key_index(
    step: PrimitiveMaterializationStep,
    indexes: TemporalIndexRegistry,
) -> SingleKeyTemporalIndex:
    if (
        step.source_table is None
        or step.source_group_key is None
        or step.source_event_time_col is None
        or step.target_key is None
    ):
        _raise(
            PrimitiveEvaluationCode.INCOMPLETE_BINDING,
            step,
            "single-key operation requires source_table, "
            "source_group_key, source_event_time_col, and target_key",
        )

    if (
        step.operation == "past_unique_neighbors"
        and step.related_col is None
    ):
        _raise(
            PrimitiveEvaluationCode.INCOMPLETE_BINDING,
            step,
            "past_unique_neighbors requires related_col",
        )

    key = SingleKeyIndexKey(
        source_table=step.source_table,
        source_group_key=step.source_group_key,
        source_event_time_col=step.source_event_time_col,
        related_col=step.related_col,
    )
    index = indexes.single_key_indexes.get(key)

    if index is None:
        _raise(
            PrimitiveEvaluationCode.MISSING_INDEX,
            step,
            f"missing single-key temporal index for {key}",
        )

    return index


def _pair_key_index(
    step: PrimitiveMaterializationStep,
    indexes: TemporalIndexRegistry,
) -> PairKeyTemporalIndex:
    if (
        step.source_table is None
        or step.source_left_key is None
        or step.source_right_key is None
        or step.source_event_time_col is None
        or step.target_left_key is None
        or step.target_right_key is None
    ):
        _raise(
            PrimitiveEvaluationCode.INCOMPLETE_BINDING,
            step,
            "pair operation requires source_table, source_left_key, "
            "source_right_key, source_event_time_col, target_left_key, "
            "and target_right_key",
        )

    key = PairKeyIndexKey(
        source_table=step.source_table,
        source_left_key=step.source_left_key,
        source_right_key=step.source_right_key,
        source_event_time_col=step.source_event_time_col,
    )
    index = indexes.pair_key_indexes.get(key)

    if index is None:
        _raise(
            PrimitiveEvaluationCode.MISSING_INDEX,
            step,
            f"missing pair-key temporal index for {key}",
        )

    return index


def _target_value(
    step: PrimitiveMaterializationStep,
    column: str | None,
    target_row: TargetRow,
) -> Hashable:
    if column is None:
        _raise(
            PrimitiveEvaluationCode.INCOMPLETE_BINDING,
            step,
            "target column binding is missing",
        )

    value = target_row.get(column)
    if value is None:
        _raise(
            PrimitiveEvaluationCode.MISSING_TARGET_VALUE,
            step,
            f"target value for column {column!r} is missing",
        )

    try:
        hash(value)
    except TypeError:
        _raise(
            PrimitiveEvaluationCode.MISSING_TARGET_VALUE,
            step,
            f"target value for column {column!r} is not hashable",
        )

    return value


def _target_time(
    step: PrimitiveMaterializationStep,
    target_row: TargetRow,
) -> TimeValue:
    value = target_row.get(step.target_time_col)

    if value is None:
        _raise(
            PrimitiveEvaluationCode.MISSING_TARGET_TIME,
            step,
            f"target time column {step.target_time_col!r} is missing",
        )

    if isinstance(value, datetime):
        return value

    if isinstance(value, Real) and not isinstance(value, bool):
        return value

    _raise(
        PrimitiveEvaluationCode.MISSING_TARGET_TIME,
        step,
        "target time must be datetime, int, or float",
    )


def _window_for_target(
    step: PrimitiveMaterializationStep,
    target_time: TimeValue,
) -> WindowValue:
    if step.window_days is None:
        _raise(
            PrimitiveEvaluationCode.INCOMPLETE_BINDING,
            step,
            "windowed operation requires window_days",
        )

    if step.window_days < 0:
        _raise(
            PrimitiveEvaluationCode.INCOMPLETE_BINDING,
            step,
            "window_days must be non-negative",
        )

    if isinstance(target_time, datetime):
        return timedelta(days=step.window_days)

    return step.window_days


def _recency_outputs(
    step: PrimitiveMaterializationStep,
    days: float | None,
) -> dict[str, int | float]:
    value_col, missing_col = step.output_columns

    if days is None:
        return {
            value_col: 0.0,
            missing_col: 1,
        }

    return {
        value_col: float(days),
        missing_col: 0,
    }


def _require_output_columns(
    step: PrimitiveMaterializationStep,
    expected_count: int,
) -> None:
    if len(step.output_columns) != expected_count:
        _raise(
            PrimitiveEvaluationCode.INVALID_OUTPUT_COLUMNS,
            step,
            f"operation requires {expected_count} output columns, "
            f"found {len(step.output_columns)}",
        )


def _raise(
    code: PrimitiveEvaluationCode,
    step: PrimitiveMaterializationStep,
    message: str,
) -> None:
    raise PrimitiveEvaluationError(
        code=code,
        step=step,
        message=message,
    )
