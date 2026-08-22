from __future__ import annotations

from fdhg.cli.auto_fdhg_joint_relbench import (
    choose_joint_strategy,
)


def choose(
    *,
    auto=0.80,
    dfs=0.79,
    independent=None,
    greedy=None,
    independent_count=0,
    greedy_count=0,
    direction="higher",
    tolerance=0.001,
):
    return choose_joint_strategy(
        auto_score=auto,
        dfs_score=dfs,
        independent_score=independent,
        greedy_score=greedy,
        independent_count=independent_count,
        greedy_count=greedy_count,
        direction=direction,
        tolerance=tolerance,
        exact_tie_tolerance=1e-12,
    )


def test_sub_tolerance_fdhg_gain_keeps_auto():
    result = choose(
        independent=0.8005,
        independent_count=1,
    )

    assert result["baseline_variant"] == "auto_only"
    assert result["selected_variant"] == "auto_only"
    assert result["admissible_fdhg_variants"] == []


def test_only_greedy_admissible():
    result = choose(
        independent=0.8005,
        greedy=0.805,
        independent_count=1,
        greedy_count=3,
    )

    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_greedy"
    )


def test_only_independent_admissible():
    result = choose(
        independent=0.805,
        greedy=0.8005,
        independent_count=2,
        greedy_count=1,
    )

    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_independent"
    )


def test_better_fdhg_selected_beyond_tolerance():
    result = choose(
        independent=0.805,
        greedy=0.810,
        independent_count=2,
        greedy_count=4,
    )

    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_greedy"
    )


def test_sparser_greedy_selected_within_tolerance():
    result = choose(
        independent=0.8050,
        greedy=0.8055,
        independent_count=7,
        greedy_count=3,
    )

    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_greedy"
    )


def test_sparser_independent_selected_within_tolerance():
    result = choose(
        independent=0.8050,
        greedy=0.8055,
        independent_count=2,
        greedy_count=5,
    )

    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_independent"
    )


def test_equal_score_and_edges_selects_independent():
    result = choose(
        independent=0.805,
        greedy=0.805,
        independent_count=3,
        greedy_count=3,
    )

    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_independent"
    )


def test_dfs_selected_as_baseline_when_better():
    result = choose(
        auto=0.80,
        dfs=0.81,
        independent=0.805,
        greedy=0.806,
        independent_count=1,
        greedy_count=1,
    )

    assert result["baseline_variant"] == "dfs_fallback"
    assert result["selected_variant"] == "dfs_fallback"


def test_fdhg_must_improve_over_selected_dfs_baseline():
    result = choose(
        auto=0.80,
        dfs=0.81,
        independent=0.8115,
        greedy=0.8105,
        independent_count=2,
        greedy_count=1,
    )

    assert result["baseline_variant"] == "dfs_fallback"
    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_independent"
    )


def test_auto_preferred_when_dfs_within_tolerance():
    result = choose(
        auto=0.80,
        dfs=0.8005,
    )

    assert result["baseline_variant"] == "auto_only"
    assert result["selected_variant"] == "auto_only"


def test_lower_is_better_admissibility():
    result = choose_joint_strategy(
        auto_score=100.0,
        dfs_score=101.0,
        independent_score=99.95,
        greedy_score=99.7,
        independent_count=4,
        greedy_count=2,
        direction="lower",
        tolerance=0.1,
        exact_tie_tolerance=1e-12,
    )

    assert result["baseline_variant"] == "auto_only"
    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_greedy"
    )


def test_lower_is_better_dfs_baseline():
    result = choose_joint_strategy(
        auto_score=100.0,
        dfs_score=99.0,
        independent_score=99.5,
        greedy_score=98.5,
        independent_count=2,
        greedy_count=3,
        direction="lower",
        tolerance=0.1,
        exact_tie_tolerance=1e-12,
    )

    assert result["baseline_variant"] == "dfs_fallback"
    assert (
        result["selected_variant"]
        == "auto_plus_fdhg_greedy"
    )
