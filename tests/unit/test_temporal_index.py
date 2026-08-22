from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from fdhg.compiler.temporal_index import (
    PairKeyEvent,
    PairKeyTemporalIndex,
    SingleKeyEvent,
    SingleKeyTemporalIndex,
)


T0 = datetime(2026, 1, 1, 12, 0, 0)


def test_unsorted_single_key_events_are_sorted() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0 + timedelta(days=2)),
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u1", T0 + timedelta(days=1)),
    ])

    assert index.count_before("u1", T0 + timedelta(days=3)) == 3
    assert (
        index.days_since_last_before(
            "u1",
            T0 + timedelta(days=3),
        )
        == 1.0
    )


def test_unsorted_pair_key_events_are_sorted() -> None:
    index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", T0 + timedelta(days=3)),
        PairKeyEvent("u1", "p1", T0),
        PairKeyEvent("u1", "p1", T0 + timedelta(days=2)),
    ])

    assert index.count_before("u1", "p1", T0 + timedelta(days=4)) == 3
    assert (
        index.days_since_last_before(
            "u1",
            "p1",
            T0 + timedelta(days=4),
        )
        == 1.0
    )


def test_strict_cutoff_excludes_events_exactly_at_target_time() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u1", T0 + timedelta(days=1)),
    ])

    assert index.count_before("u1", T0 + timedelta(days=1)) == 1


def test_earlier_duplicate_timestamps_are_counted_separately() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u1", T0 + timedelta(days=1)),
    ])

    assert index.count_before("u1", T0 + timedelta(hours=1)) == 2


def test_unknown_single_key_returns_zero_or_none() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
    ])

    assert index.count_before("missing", T0) == 0
    assert index.window_count_before("missing", T0, timedelta(days=1)) == 0
    assert index.unique_related_before("missing", T0, timedelta(days=1)) == 0
    assert index.days_since_last_before("missing", T0) is None


def test_unknown_pair_returns_zero_or_none() -> None:
    index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", T0),
    ])

    assert index.count_before("missing", "p1", T0) == 0
    assert index.days_since_last_before("missing", "p1", T0) is None


def test_all_time_count_before_cutoff() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", 3),
        SingleKeyEvent("u1", 1),
        SingleKeyEvent("u1", 2),
    ])

    assert index.count_before("u1", 3) == 2


def test_window_lower_boundary_is_inclusive() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u1", T0 + timedelta(days=1)),
    ])

    assert (
        index.window_count_before(
            "u1",
            T0 + timedelta(days=1),
            timedelta(days=1),
        )
        == 1
    )


def test_window_upper_boundary_is_exclusive() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u1", T0 + timedelta(days=1)),
    ])

    assert (
        index.window_count_before(
            "u1",
            T0 + timedelta(days=1),
            timedelta(days=2),
        )
        == 1
    )


def test_empty_window_returns_zero() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
    ])

    assert index.window_count_before("u1", T0, timedelta(days=1)) == 0


def test_unique_related_values_are_deduplicated() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0, "beer1"),
        SingleKeyEvent("u1", T0 + timedelta(hours=1), "beer1"),
        SingleKeyEvent("u1", T0 + timedelta(hours=2), "beer2"),
    ])

    assert (
        index.unique_related_before(
            "u1",
            T0 + timedelta(days=1),
            timedelta(days=1),
        )
        == 2
    )


def test_null_related_values_are_excluded() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0, None),
        SingleKeyEvent("u1", T0 + timedelta(hours=1), "beer1"),
    ])

    assert (
        index.unique_related_before(
            "u1",
            T0 + timedelta(days=1),
            timedelta(days=1),
        )
        == 1
    )


def test_days_since_last_returns_correct_duration() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u1", T0 + timedelta(days=2, hours=12)),
    ])

    assert (
        index.days_since_last_before(
            "u1",
            T0 + timedelta(days=4),
        )
        == 1.5
    )


def test_pair_prior_count_is_correct() -> None:
    index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", T0),
        PairKeyEvent("u1", "p1", T0 + timedelta(days=1)),
        PairKeyEvent("u1", "p2", T0),
    ])

    assert index.count_before("u1", "p1", T0 + timedelta(days=2)) == 2


def test_pair_days_since_last_is_correct() -> None:
    index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", T0),
        PairKeyEvent("u1", "p1", T0 + timedelta(days=3)),
    ])

    assert (
        index.days_since_last_before(
            "u1",
            "p1",
            T0 + timedelta(days=5),
        )
        == 2.0
    )


def test_multiple_keys_remain_isolated() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
        SingleKeyEvent("u2", T0),
        SingleKeyEvent("u2", T0 + timedelta(days=1)),
    ])

    assert index.count_before("u1", T0 + timedelta(days=2)) == 1
    assert index.count_before("u2", T0 + timedelta(days=2)) == 2


def test_multiple_pairs_remain_isolated() -> None:
    index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", T0),
        PairKeyEvent("u1", "p2", T0),
        PairKeyEvent("u1", "p2", T0 + timedelta(days=1)),
    ])

    assert index.count_before("u1", "p1", T0 + timedelta(days=2)) == 1
    assert index.count_before("u1", "p2", T0 + timedelta(days=2)) == 2


def test_repeated_queries_are_deterministic() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0, "beer1"),
        SingleKeyEvent("u1", T0 + timedelta(days=1), "beer2"),
    ])

    first = index.window_count_before("u1", T0 + timedelta(days=2), timedelta(days=3))
    second = index.window_count_before("u1", T0 + timedelta(days=2), timedelta(days=3))
    assert first == second


def test_index_inputs_are_not_mutated() -> None:
    events = [
        SingleKeyEvent("u1", T0 + timedelta(days=1)),
        SingleKeyEvent("u1", T0),
    ]
    before = deepcopy(events)

    SingleKeyTemporalIndex(events)

    assert events == before


def test_mixed_datetime_numeric_timestamps_are_rejected() -> None:
    with pytest.raises(TypeError, match="mix datetime and numeric"):
        SingleKeyTemporalIndex([
            SingleKeyEvent("u1", T0),
            SingleKeyEvent("u1", 1),
        ])


def test_mixed_timezone_aware_and_naive_datetimes_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        PairKeyTemporalIndex([
            PairKeyEvent("u1", "p1", T0),
            PairKeyEvent(
                "u1",
                "p1",
                T0.replace(tzinfo=timezone.utc),
            ),
        ])


def test_negative_windows_are_rejected() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", 1),
    ])

    with pytest.raises(ValueError, match="non-negative"):
        index.window_count_before("u1", 2, -1)


def test_zero_width_windows_return_zero() -> None:
    index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", T0),
    ])

    assert index.window_count_before("u1", T0, timedelta(0)) == 0
    assert index.unique_related_before("u1", T0, timedelta(0)) == 0


def test_real_ratebeer_style_synthetic_history() -> None:
    target = datetime(2026, 6, 1)
    window = timedelta(days=30)
    user_index = SingleKeyTemporalIndex([
        SingleKeyEvent("u1", target - timedelta(days=31), "beer_old"),
        SingleKeyEvent("u1", target - timedelta(days=30), "beer_a"),
        SingleKeyEvent("u1", target - timedelta(days=10), "beer_a"),
        SingleKeyEvent("u1", target - timedelta(days=1), "beer_b"),
        SingleKeyEvent("u1", target, "beer_c"),
    ])
    place_index = SingleKeyTemporalIndex([
        SingleKeyEvent("p1", target - timedelta(days=31), "u_old"),
        SingleKeyEvent("p1", target - timedelta(days=30), "u1"),
        SingleKeyEvent("p1", target - timedelta(days=5), "u2"),
        SingleKeyEvent("p1", target, "u3"),
    ])
    pair_index = PairKeyTemporalIndex([
        PairKeyEvent("u1", "p1", target - timedelta(days=60)),
        PairKeyEvent("u1", "p1", target - timedelta(days=3)),
        PairKeyEvent("u1", "p1", target - timedelta(days=3)),
        PairKeyEvent("u1", "p1", target),
    ])

    assert user_index.window_count_before("u1", target, window) == 3
    assert user_index.unique_related_before("u1", target, window) == 2
    assert user_index.days_since_last_before("u1", target) == 1.0
    assert place_index.window_count_before("p1", target, window) == 2
    assert place_index.unique_related_before("p1", target, window) == 2
    assert place_index.days_since_last_before("p1", target) == 5.0
    assert pair_index.count_before("u1", "p1", target) == 3
    assert pair_index.days_since_last_before("u1", "p1", target) == 3.0


def test_temporal_index_has_no_heavy_or_filesystem_dependency() -> None:
    import fdhg.compiler.temporal_index as module

    names = set(module.__dict__)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names
    assert "Path" not in names
