from __future__ import annotations

from collections.abc import Iterable
from math import isclose, isfinite
from typing import Literal


Direction = Literal["maximize", "minimize"]


def consistent_improvement_gate(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    direction: Direction,
    min_improvement: float = 0.0,
) -> bool:
    """
    Select the candidate only when it improves on every paired seed.

    Improvement is defined as:

    - maximize: candidate - baseline
    - minimize: baseline - candidate

    A tie is rejected when ``min_improvement == 0`` because the
    improvement must be strictly greater than the threshold.
    """
    pairs = list(zip(baseline, candidate, strict=True))

    if not pairs:
        raise ValueError("At least one seed is required.")

    if direction not in {"maximize", "minimize"}:
        raise ValueError(
            "direction must be either 'maximize' or 'minimize'."
        )

    if not isfinite(min_improvement):
        raise ValueError("min_improvement must be finite.")

    improvements: list[float] = []

    for baseline_value, candidate_value in pairs:
        baseline_float = float(baseline_value)
        candidate_float = float(candidate_value)

        if not isfinite(baseline_float):
            raise ValueError("Baseline scores must be finite.")

        if not isfinite(candidate_float):
            raise ValueError("Candidate scores must be finite.")

        if direction == "maximize":
            improvement = candidate_float - baseline_float
        else:
            improvement = baseline_float - candidate_float

        improvements.append(improvement)

    return all(
        improvement > min_improvement
        and not isclose(
            improvement,
            min_improvement,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for improvement in improvements
    )


def classification_gate(
    baseline: Iterable[float],
    candidate: Iterable[float],
) -> bool:
    """Select the candidate only if accuracy improves on every seed."""
    return consistent_improvement_gate(
        baseline,
        candidate,
        direction="maximize",
    )


def regression_gate(
    baseline_rmse: Iterable[float],
    candidate_rmse: Iterable[float],
) -> bool:
    """Select the candidate only if RMSE decreases on every seed."""
    return consistent_improvement_gate(
        baseline_rmse,
        candidate_rmse,
        direction="minimize",
    )


def lexicographic_improvement_gate(
    primary_baseline: Iterable[float],
    primary_candidate: Iterable[float],
    secondary_baseline: Iterable[float],
    secondary_candidate: Iterable[float],
    *,
    primary_direction: Direction,
    secondary_direction: Direction,
    tolerance: float = 1e-12,
) -> bool:
    """
    Apply a per-seed lexicographic gate.

    The candidate passes a seed when the primary metric improves, or
    when the primary metric is tied within tolerance and the secondary
    metric improves. Every seed must pass.
    """
    primary_pairs = list(
        zip(
            primary_baseline,
            primary_candidate,
            strict=True,
        )
    )
    secondary_pairs = list(
        zip(
            secondary_baseline,
            secondary_candidate,
            strict=True,
        )
    )

    if not primary_pairs:
        raise ValueError("At least one seed is required.")

    if len(primary_pairs) != len(secondary_pairs):
        raise ValueError(
            "Primary and secondary metrics must have equal lengths."
        )

    if primary_direction not in {"maximize", "minimize"}:
        raise ValueError("Invalid primary direction.")

    if secondary_direction not in {"maximize", "minimize"}:
        raise ValueError("Invalid secondary direction.")

    def improvement(
        baseline_value: float,
        candidate_value: float,
        direction: Direction,
    ) -> float:
        baseline_float = float(baseline_value)
        candidate_float = float(candidate_value)

        if not isfinite(baseline_float):
            raise ValueError("Baseline scores must be finite.")

        if not isfinite(candidate_float):
            raise ValueError("Candidate scores must be finite.")

        if direction == "maximize":
            return candidate_float - baseline_float

        return baseline_float - candidate_float

    seed_passes = []

    for primary_pair, secondary_pair in zip(
        primary_pairs,
        secondary_pairs,
        strict=True,
    ):
        primary_gain = improvement(
            primary_pair[0],
            primary_pair[1],
            primary_direction,
        )
        secondary_gain = improvement(
            secondary_pair[0],
            secondary_pair[1],
            secondary_direction,
        )

        primary_tied = isclose(
            primary_gain,
            0.0,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )

        seed_passes.append(
            primary_gain > tolerance
            or (
                primary_tied
                and secondary_gain > tolerance
            )
        )

    return all(seed_passes)
