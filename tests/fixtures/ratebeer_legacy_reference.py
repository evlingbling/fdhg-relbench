from __future__ import annotations

from datetime import timedelta
from typing import Mapping, Sequence


WINDOWS = (30, 90, 365)


def legacy_ratebeer_pairwise_features(
    *,
    source_rows_by_table: Mapping[str, Sequence[Mapping[str, object]]],
    target_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int | float], ...]:
    beer_rows = tuple(source_rows_by_table.get("beer_ratings", ()))
    place_rows = tuple(source_rows_by_table.get("place_ratings", ()))
    records: list[dict[str, int | float]] = []

    for target in target_rows:
        user_id = target["user_id"]
        place_id = target["candidate_place_id"]
        target_time = target["timestamp"]

        record: dict[str, int | float] = {}
        _add_single_history(
            record=record,
            prefix="f_pairwise__left",
            rows=beer_rows,
            group_col="user_id",
            group_value=user_id,
            related_col="beer_id",
            time_col="updated_at",
            target_time=target_time,
        )
        _add_single_history(
            record=record,
            prefix="f_pairwise__right",
            rows=place_rows,
            group_col="place_id",
            group_value=place_id,
            related_col="user_id",
            time_col="created_at",
            target_time=target_time,
        )
        _add_pair_history(
            record=record,
            rows=place_rows,
            left_col="user_id",
            left_value=user_id,
            right_col="place_id",
            right_value=place_id,
            time_col="created_at",
            target_time=target_time,
        )
        records.append(record)

    return tuple(records)


def _add_single_history(
    *,
    record: dict[str, int | float],
    prefix: str,
    rows: Sequence[Mapping[str, object]],
    group_col: str,
    group_value: object,
    related_col: str,
    time_col: str,
    target_time,
) -> None:
    matching = [
        row
        for row in rows
        if row.get(group_col) == group_value
        and row.get(time_col) is not None
        and row[time_col] < target_time
    ]

    for window_days in WINDOWS:
        lower = target_time - timedelta(days=window_days)
        window_rows = [
            row
            for row in matching
            if lower <= row[time_col] < target_time
        ]
        record[f"{prefix}__count_{window_days}d"] = len(
            window_rows
        )

        if window_days in {30, 90}:
            record[
                f"{prefix}__unique_neighbors_{window_days}d"
            ] = len({
                row.get(related_col)
                for row in window_rows
                if row.get(related_col) is not None
            })

    prior_times = sorted(row[time_col] for row in matching)

    if prior_times:
        record[f"{prefix}__days_since_last"] = _days_between(
            target_time,
            prior_times[-1],
        )
        record[f"{prefix}__days_since_last__is_missing"] = 0
    else:
        record[f"{prefix}__days_since_last"] = 0.0
        record[f"{prefix}__days_since_last__is_missing"] = 1


def _add_pair_history(
    *,
    record: dict[str, int | float],
    rows: Sequence[Mapping[str, object]],
    left_col: str,
    left_value: object,
    right_col: str,
    right_value: object,
    time_col: str,
    target_time,
) -> None:
    prior_times = sorted(
        row[time_col]
        for row in rows
        if row.get(left_col) == left_value
        and row.get(right_col) == right_value
        and row.get(time_col) is not None
        and row[time_col] < target_time
    )
    record["f_pairwise__pair__prior_count"] = len(prior_times)

    if prior_times:
        record["f_pairwise__pair__days_since_last"] = _days_between(
            target_time,
            prior_times[-1],
        )
        record[
            "f_pairwise__pair__days_since_last__is_missing"
        ] = 0
    else:
        record["f_pairwise__pair__days_since_last"] = 0.0
        record[
            "f_pairwise__pair__days_since_last__is_missing"
        ] = 1


def _days_between(target_time, event_time) -> float:
    delta = target_time - event_time

    if hasattr(delta, "total_seconds"):
        return delta.total_seconds() / 86400.0

    return float(delta)
