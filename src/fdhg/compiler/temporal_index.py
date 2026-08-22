from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from typing import Hashable, Iterable


TimeValue = datetime | int | float
WindowValue = timedelta | int | float


@dataclass(frozen=True)
class SingleKeyEvent:
    group_key: Hashable
    event_time: TimeValue
    related_value: Hashable | None = None


@dataclass(frozen=True)
class PairKeyEvent:
    left_key: Hashable
    right_key: Hashable
    event_time: TimeValue


@dataclass(frozen=True)
class _TimeModel:
    kind: str
    timezone_aware: bool | None = None


class SingleKeyTemporalIndex:
    """Immutable strict-prior temporal index for one grouping key.

    Count and recency queries use binary search over sorted event-time
    tuples. Unique-related queries binary-search the time window and
    scan only that bounded slice; null related values are excluded.
    """

    def __init__(
        self,
        events: Iterable[SingleKeyEvent],
    ) -> None:
        rows = list(events)
        time_model = _infer_time_model(
            [row.event_time for row in rows]
        )
        grouped: dict[
            Hashable,
            list[tuple[TimeValue, Hashable | None]],
        ] = {}

        for row in rows:
            _validate_time(row.event_time, time_model)
            grouped.setdefault(row.group_key, []).append(
                (row.event_time, row.related_value)
            )

        index: dict[
            Hashable,
            tuple[tuple[TimeValue, ...], tuple[Hashable | None, ...]],
        ] = {}

        for key, key_rows in grouped.items():
            sorted_rows = sorted(
                key_rows,
                key=lambda item: item[0],
            )
            index[key] = (
                tuple(row[0] for row in sorted_rows),
                tuple(row[1] for row in sorted_rows),
            )

        self._time_model = time_model
        self._index = dict(index)

    def count_before(
        self,
        group_key: Hashable,
        target_time: TimeValue,
    ) -> int:
        times, _ = self._events_for_key(group_key)
        cutoff = self._cutoff(times, target_time)
        return cutoff

    def window_count_before(
        self,
        group_key: Hashable,
        target_time: TimeValue,
        window: WindowValue,
    ) -> int:
        times, _ = self._events_for_key(group_key)
        start, end = self._window_bounds(
            times,
            target_time,
            window,
        )
        return end - start

    def unique_related_before(
        self,
        group_key: Hashable,
        target_time: TimeValue,
        window: WindowValue,
    ) -> int:
        times, related = self._events_for_key(group_key)
        start, end = self._window_bounds(
            times,
            target_time,
            window,
        )
        return len({
            value
            for value in related[start:end]
            if value is not None
        })

    def days_since_last_before(
        self,
        group_key: Hashable,
        target_time: TimeValue,
    ) -> float | None:
        times, _ = self._events_for_key(group_key)
        cutoff = self._cutoff(times, target_time)

        if cutoff == 0:
            return None

        return _duration_days(
            target_time,
            times[cutoff - 1],
            self._time_model,
        )

    def _events_for_key(
        self,
        group_key: Hashable,
    ) -> tuple[tuple[TimeValue, ...], tuple[Hashable | None, ...]]:
        return self._index.get(group_key, ((), ()))

    def _cutoff(
        self,
        times: tuple[TimeValue, ...],
        target_time: TimeValue,
    ) -> int:
        _validate_time(target_time, self._time_model)
        return bisect_left(times, target_time)

    def _window_bounds(
        self,
        times: tuple[TimeValue, ...],
        target_time: TimeValue,
        window: WindowValue,
    ) -> tuple[int, int]:
        _validate_time(target_time, self._time_model)
        _validate_window(window, self._time_model)
        lower = _subtract_window(
            target_time,
            window,
            self._time_model,
        )
        return bisect_left(times, lower), bisect_left(
            times,
            target_time,
        )


class PairKeyTemporalIndex:
    """Immutable strict-prior temporal index for key pairs."""

    def __init__(
        self,
        events: Iterable[PairKeyEvent],
    ) -> None:
        rows = list(events)
        time_model = _infer_time_model(
            [row.event_time for row in rows]
        )
        grouped: dict[
            tuple[Hashable, Hashable],
            list[TimeValue],
        ] = {}

        for row in rows:
            _validate_time(row.event_time, time_model)
            grouped.setdefault(
                (row.left_key, row.right_key),
                [],
            ).append(row.event_time)

        index = {
            key: tuple(sorted(times))
            for key, times in grouped.items()
        }

        self._time_model = time_model
        self._index = dict(index)

    def count_before(
        self,
        left_key: Hashable,
        right_key: Hashable,
        target_time: TimeValue,
    ) -> int:
        times = self._events_for_pair(left_key, right_key)
        return self._cutoff(times, target_time)

    def days_since_last_before(
        self,
        left_key: Hashable,
        right_key: Hashable,
        target_time: TimeValue,
    ) -> float | None:
        times = self._events_for_pair(left_key, right_key)
        cutoff = self._cutoff(times, target_time)

        if cutoff == 0:
            return None

        return _duration_days(
            target_time,
            times[cutoff - 1],
            self._time_model,
        )

    def _events_for_pair(
        self,
        left_key: Hashable,
        right_key: Hashable,
    ) -> tuple[TimeValue, ...]:
        return self._index.get((left_key, right_key), ())

    def _cutoff(
        self,
        times: tuple[TimeValue, ...],
        target_time: TimeValue,
    ) -> int:
        _validate_time(target_time, self._time_model)
        return bisect_left(times, target_time)


def _infer_time_model(
    values: list[TimeValue],
) -> _TimeModel:
    if not values:
        return _TimeModel(kind="empty")

    first = values[0]

    if isinstance(first, datetime):
        aware = _is_timezone_aware(first)
        model = _TimeModel(
            kind="datetime",
            timezone_aware=aware,
        )
    elif _is_numeric_time(first):
        model = _TimeModel(kind="numeric")
    else:
        raise TypeError(
            "event_time must be datetime, int, or float"
        )

    for value in values:
        _validate_time(value, model)

    return model


def _validate_time(
    value: TimeValue,
    model: _TimeModel,
) -> None:
    if model.kind == "empty":
        if isinstance(value, datetime) or _is_numeric_time(value):
            return
        raise TypeError(
            "target_time must be datetime, int, or float"
        )

    if model.kind == "datetime":
        if not isinstance(value, datetime):
            raise TypeError(
                "cannot mix datetime and numeric timestamps"
            )
        if _is_timezone_aware(value) != model.timezone_aware:
            raise ValueError(
                "cannot mix timezone-aware and timezone-naive "
                "datetimes"
            )
        return

    if model.kind == "numeric":
        if not _is_numeric_time(value):
            raise TypeError(
                "cannot mix numeric and datetime timestamps"
            )
        return

    raise TypeError(f"unknown time model: {model.kind}")


def _validate_window(
    window: WindowValue,
    model: _TimeModel,
) -> None:
    if model.kind == "datetime":
        if not isinstance(window, timedelta):
            raise TypeError(
                "datetime indexes require timedelta windows"
            )
        if window < timedelta(0):
            raise ValueError("window must be non-negative")
        return

    if model.kind == "numeric":
        if not _is_numeric_time(window):
            raise TypeError(
                "numeric indexes require numeric windows"
            )
        if window < 0:
            raise ValueError("window must be non-negative")
        return

    if model.kind == "empty":
        if isinstance(window, timedelta):
            if window < timedelta(0):
                raise ValueError("window must be non-negative")
            return
        if _is_numeric_time(window):
            if window < 0:
                raise ValueError("window must be non-negative")
            return
        raise TypeError(
            "window must be timedelta, int, or float"
        )


def _subtract_window(
    target_time: TimeValue,
    window: WindowValue,
    model: _TimeModel,
) -> TimeValue:
    if model.kind == "datetime":
        return target_time - window

    if model.kind == "numeric":
        return target_time - window

    if isinstance(target_time, datetime):
        if not isinstance(window, timedelta):
            raise TypeError(
                "datetime targets require timedelta windows"
            )
        return target_time - window

    if _is_numeric_time(target_time):
        if not _is_numeric_time(window):
            raise TypeError(
                "numeric targets require numeric windows"
            )
        return target_time - window

    raise TypeError(
        "target_time must be datetime, int, or float"
    )


def _duration_days(
    target_time: TimeValue,
    event_time: TimeValue,
    model: _TimeModel,
) -> float:
    _validate_time(target_time, model)
    _validate_time(event_time, model)
    delta = target_time - event_time

    if isinstance(delta, timedelta):
        return delta.total_seconds() / 86400.0

    return float(delta)


def _is_timezone_aware(value: datetime) -> bool:
    return (
        value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _is_numeric_time(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)
