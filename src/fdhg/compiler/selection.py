from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .programs import CandidateProgram


@dataclass(frozen=True)
class ProgramScore:
    program_id: str
    result_variant: str
    n_runs: int
    n_features_mean: float
    primary_metric: str
    primary_mean: float
    primary_std: float
    secondary_metric: str | None
    secondary_mean: float | None

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "result_variant": self.result_variant,
            "n_runs": self.n_runs,
            "n_features_mean": self.n_features_mean,
            "primary_metric": self.primary_metric,
            "primary_mean": self.primary_mean,
            "primary_std": self.primary_std,
            "secondary_metric": self.secondary_metric,
            "secondary_mean": self.secondary_mean,
        }


@dataclass(frozen=True)
class CandidateValidationResult:
    dataset: str
    task: str
    program_id: str
    primary_metric: str
    metric_direction: str
    validation_score: float | None
    baseline_program_id: str | None = None
    baseline_score: float | None = None
    split: str = "validation"
    n_features: int | None = None
    eligible: bool = True
    rejection_reasons: tuple[str, ...] = ()
    evidence_location: str = ""
    materializable: bool | None = None
    leakage_safe: bool | None = None
    temporally_safe: bool | None = None
    provenance_complete: bool | None = None


@dataclass(frozen=True)
class CandidateSelectionPolicy:
    dataset: str
    task: str
    primary_metric: str
    metric_direction: str
    baseline_program_id: str = "baseline"
    min_improvement: float = 0.0


@dataclass(frozen=True)
class RankedCandidate:
    program_id: str
    validation_score: float
    improvement_over_baseline: float
    n_features: int | None
    added_features: int | None
    family_complexity: tuple[int, ...]
    evidence_location: str


@dataclass(frozen=True)
class RejectedCandidate:
    program_id: str
    rejection_reasons: tuple[str, ...]
    evidence_location: str


@dataclass(frozen=True)
class CandidateSelectionDecision:
    selected_program_id: str
    selected_score: float
    baseline_program_id: str
    baseline_score: float
    improvement_over_baseline: float
    metric: str
    metric_direction: str
    fallback_occurred: bool
    fallback_reason: str | None
    ranked_candidates: tuple[RankedCandidate, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    evidence_locations: tuple[str, ...]


CANONICAL_VALIDATION_COLUMNS = frozenset({
    "dataset",
    "task",
    "program_id",
    "split",
    "primary_metric",
    "metric_direction",
    "score",
    "n_features",
    "eligible",
    "rejection_reason",
    "evidence_location",
    "materializable",
    "leakage_safe",
    "temporally_safe",
    "provenance_complete",
})


def load_program_score(
    *,
    program_id: str,
    result_root: Path,
    result_variant: str,
    primary_metric: str,
    secondary_metric: str | None,
    seeds: list[int],
) -> ProgramScore:
    rows = []

    for seed in seeds:
        path = (
            result_root
            / result_variant
            / f"seed{seed}"
            / "metrics.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)

        if len(frame) != 1:
            raise ValueError(
                f"Expected one metric row in {path}, "
                f"found {len(frame)}"
            )

        rows.append(frame.iloc[0].to_dict())

    runs = pd.DataFrame(rows)

    if primary_metric not in runs.columns:
        raise KeyError(
            f"Metric {primary_metric!r} missing for "
            f"{program_id}. Available columns: "
            f"{sorted(runs.columns)}"
        )

    primary_values = pd.to_numeric(
        runs[primary_metric],
        errors="raise",
    )

    secondary_mean = None

    if secondary_metric is not None:
        if secondary_metric not in runs.columns:
            raise KeyError(
                f"Metric {secondary_metric!r} missing for "
                f"{program_id}. Available columns: "
                f"{sorted(runs.columns)}"
            )

        secondary_mean = float(
            pd.to_numeric(
                runs[secondary_metric],
                errors="raise",
            ).mean()
        )

    if "n_features" not in runs.columns:
        raise KeyError(
            f"n_features missing for {program_id}. "
            f"Available columns: {sorted(runs.columns)}"
        )

    return ProgramScore(
        program_id=program_id,
        result_variant=result_variant,
        n_runs=len(runs),
        n_features_mean=float(
            pd.to_numeric(
                runs["n_features"],
                errors="raise",
            ).mean()
        ),
        primary_metric=primary_metric,
        primary_mean=float(primary_values.mean()),
        primary_std=float(primary_values.std(ddof=1)),
        secondary_metric=secondary_metric,
        secondary_mean=secondary_mean,
    )


def load_candidate_validation_results(
    path: Path,
) -> tuple[CandidateValidationResult, ...]:
    """
    Read a canonical, task-scoped validation summary.

    The loader deliberately requires dataset, task, program_id, split,
    and metric columns in the file. It never derives identity from the
    filename or directory layout.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = CANONICAL_VALIDATION_COLUMNS - fieldnames

        if missing:
            raise ValueError(
                f"{path} is missing canonical validation columns: "
                + ", ".join(sorted(missing))
            )

        records = []

        for row_index, row in enumerate(reader, start=2):
            score = _optional_float(
                row["score"],
                location=f"{path}:{row_index}:score",
            )
            baseline_score = _optional_float(
                row.get("baseline_score", ""),
                location=f"{path}:{row_index}:baseline_score",
            )

            records.append(
                CandidateValidationResult(
                    dataset=row["dataset"],
                    task=row["task"],
                    program_id=row["program_id"],
                    split=row["split"],
                    primary_metric=row["primary_metric"],
                    metric_direction=row["metric_direction"],
                    validation_score=score,
                    baseline_program_id=(
                        row.get("baseline_program_id") or None
                    ),
                    baseline_score=baseline_score,
                    n_features=_optional_nonnegative_int(
                        row["n_features"],
                        location=(
                            f"{path}:{row_index}:n_features"
                        ),
                    ),
                    eligible=_parse_bool(
                        row["eligible"],
                        location=f"{path}:{row_index}:eligible",
                    ),
                    rejection_reasons=_split_reasons(
                        row["rejection_reason"]
                    ),
                    evidence_location=_require_nonempty(
                        row["evidence_location"],
                        location=(
                            f"{path}:{row_index}:"
                            "evidence_location"
                        ),
                    ),
                    materializable=_parse_bool(
                        row["materializable"],
                        location=(
                            f"{path}:{row_index}:materializable"
                        ),
                    ),
                    leakage_safe=_parse_bool(
                        row["leakage_safe"],
                        location=(
                            f"{path}:{row_index}:leakage_safe"
                        ),
                    ),
                    temporally_safe=_parse_bool(
                        row["temporally_safe"],
                        location=(
                            f"{path}:{row_index}:temporally_safe"
                        ),
                    ),
                    provenance_complete=_parse_bool(
                        row["provenance_complete"],
                        location=(
                            f"{path}:{row_index}:"
                            "provenance_complete"
                        ),
                    ),
                )
            )

    return _reject_duplicate_loaded_records(tuple(records))


def select_candidate_program(
    programs: Sequence[CandidateProgram],
    validation_results: Sequence[CandidateValidationResult],
    policy: CandidateSelectionPolicy,
) -> CandidateSelectionDecision:
    if policy.metric_direction not in {"higher", "lower"}:
        raise ValueError(
            "metric_direction must be 'higher' or 'lower', "
            f"got {policy.metric_direction!r}"
        )

    program_tuple = tuple(programs)
    result_tuple = tuple(validation_results)
    program_by_id = _dedupe_programs(program_tuple)
    records_by_program = _dedupe_validation_records(
        result_tuple,
        policy=policy,
    )

    baseline_record = records_by_program.get(
        policy.baseline_program_id
    )
    if baseline_record is None:
        raise ValueError(
            "No validation result for baseline program "
            f"{policy.baseline_program_id!r}"
        )

    baseline_rejections = _record_rejection_reasons(
        baseline_record,
        policy=policy,
    )
    if baseline_rejections:
        raise ValueError(
            "Baseline validation result is unusable: "
            + ", ".join(baseline_rejections)
        )

    baseline_score = _require_score(
        baseline_record,
        reason="baseline validation result has no score",
    )
    baseline_features = _feature_count(
        program_by_id.get(policy.baseline_program_id),
        baseline_record,
    )
    ranked = []
    rejected: list[RejectedCandidate] = []

    for program in sorted(
        program_tuple,
        key=lambda item: item.program_id,
    ):
        if program.program_id == policy.baseline_program_id:
            continue

        record = records_by_program.get(program.program_id)
        if record is None:
            mismatched_rejection = (
                _rejection_for_mismatched_program_records(
                    result_tuple,
                    program.program_id,
                    policy=policy,
                )
            )
            if mismatched_rejection is not None:
                rejected.append(mismatched_rejection)
                continue

            rejected.append(
                RejectedCandidate(
                    program_id=program.program_id,
                    rejection_reasons=(
                        "missing_validation_result",
                    ),
                    evidence_location="",
                )
            )
            continue

        baseline_conflicts = _baseline_metadata_conflicts(
            record,
            baseline_record=baseline_record,
            policy=policy,
        )
        if baseline_conflicts:
            rejected.append(
                RejectedCandidate(
                    program_id=program.program_id,
                    rejection_reasons=baseline_conflicts,
                    evidence_location=record.evidence_location,
                )
            )
            continue

        reasons = _record_rejection_reasons(
            record,
            policy=policy,
        )
        if reasons:
            rejected.append(
                _rejection_for_mismatched_program_records(
                    result_tuple,
                    program.program_id,
                    policy=policy,
                )
                or RejectedCandidate(
                    program_id=program.program_id,
                    rejection_reasons=tuple(reasons),
                    evidence_location=record.evidence_location,
                )
            )
            continue

        score = _require_score(
            record,
            reason="candidate validation result has no score",
        )
        candidate_features = _feature_count(program, record)
        ranked.append(
            RankedCandidate(
                program_id=program.program_id,
                validation_score=score,
                improvement_over_baseline=_improvement(
                    candidate_score=score,
                    baseline_score=baseline_score,
                    metric_direction=policy.metric_direction,
                ),
                n_features=candidate_features,
                added_features=_added_features(
                    candidate_features,
                    baseline_features,
                ),
                family_complexity=_family_complexity(program),
                evidence_location=record.evidence_location,
            )
        )

    for record in sorted(
        result_tuple,
        key=lambda item: (
            item.dataset,
            item.task,
            item.program_id,
            item.evidence_location,
        ),
    ):
        if record.program_id in program_by_id:
            continue

        reasons = (
            ["unknown_program_id"]
            + list(_record_rejection_reasons(record, policy=policy))
        )
        rejected.append(
            RejectedCandidate(
                program_id=record.program_id,
                rejection_reasons=tuple(dict.fromkeys(reasons)),
                evidence_location=record.evidence_location,
            )
        )

    ranked_candidates = tuple(
        sorted(
            ranked,
            key=lambda item: _ranking_key(
                item,
                metric_direction=policy.metric_direction,
            ),
        )
    )

    if not ranked_candidates:
        selected_program_id = policy.baseline_program_id
        selected_score = baseline_score
        improvement = 0.0
        fallback_occurred = True
        fallback_reason = "no_eligible_fdhg_candidate"
    else:
        best = ranked_candidates[0]
        improvement = best.improvement_over_baseline

        if _passes_final_gate(
            improvement,
            min_improvement=policy.min_improvement,
        ):
            selected_program_id = best.program_id
            selected_score = best.validation_score
            fallback_occurred = False
            fallback_reason = None
        else:
            selected_program_id = policy.baseline_program_id
            selected_score = baseline_score
            fallback_occurred = True
            fallback_reason = (
                "best_fdhg_candidate_does_not_improve_baseline"
            )

    evidence_locations = tuple(
        sorted({
            location
            for location in (
                [baseline_record.evidence_location]
                + [
                    candidate.evidence_location
                    for candidate in ranked_candidates
                ]
                + [
                    candidate.evidence_location
                    for candidate in rejected
                ]
            )
            if location
        })
    )

    return CandidateSelectionDecision(
        selected_program_id=selected_program_id,
        selected_score=selected_score,
        baseline_program_id=policy.baseline_program_id,
        baseline_score=baseline_score,
        improvement_over_baseline=improvement,
        metric=policy.primary_metric,
        metric_direction=policy.metric_direction,
        fallback_occurred=fallback_occurred,
        fallback_reason=fallback_reason,
        ranked_candidates=ranked_candidates,
        rejected_candidates=tuple(
            sorted(
                rejected,
                key=lambda item: (
                    item.program_id,
                    item.rejection_reasons,
                    item.evidence_location,
                ),
            )
        ),
        evidence_locations=evidence_locations,
    )


def _reject_duplicate_loaded_records(
    records: tuple[CandidateValidationResult, ...],
) -> tuple[CandidateValidationResult, ...]:
    seen: set[CandidateValidationResult] = set()
    duplicates: list[CandidateValidationResult] = []

    for record in records:
        if record in seen:
            duplicates.append(record)
        seen.add(record)

    if duplicates:
        duplicate_ids = sorted({
            record.program_id for record in duplicates
        })
        raise ValueError(
            "duplicate validation rows: "
            + ", ".join(duplicate_ids)
        )

    return records


def _require_nonempty(
    value: str | None,
    *,
    location: str,
) -> str:
    if value is None or value.strip() == "":
        raise ValueError(f"Expected non-empty value at {location}")
    return value


def _baseline_metadata_conflicts(
    record: CandidateValidationResult,
    *,
    baseline_record: CandidateValidationResult,
    policy: CandidateSelectionPolicy,
) -> tuple[str, ...]:
    conflicts = []

    if (
        record.baseline_program_id is not None
        and record.baseline_program_id != ""
        and record.baseline_program_id != policy.baseline_program_id
    ):
        conflicts.append("baseline_program_id_conflict")

    if (
        record.baseline_score is not None
        and baseline_record.validation_score is not None
        and record.baseline_score != baseline_record.validation_score
    ):
        conflicts.append("baseline_score_conflict")

    return tuple(conflicts)


def _passes_final_gate(
    improvement: float,
    *,
    min_improvement: float,
) -> bool:
    return improvement > 0.0 and improvement >= min_improvement


def _rejection_for_mismatched_program_records(
    records: Sequence[CandidateValidationResult],
    program_id: str,
    *,
    policy: CandidateSelectionPolicy,
) -> RejectedCandidate | None:
    matching_program_records = sorted(
        (
            record
            for record in records
            if record.program_id == program_id
        ),
        key=lambda item: (
            item.dataset,
            item.task,
            item.primary_metric,
            item.metric_direction,
            item.split,
            item.evidence_location,
        ),
    )

    if not matching_program_records:
        return None

    reasons = []
    locations = []

    for record in matching_program_records:
        reasons.extend(
            _record_rejection_reasons(record, policy=policy)
        )
        if record.evidence_location:
            locations.append(record.evidence_location)

    if not reasons:
        reasons.append("missing_validation_result")

    return RejectedCandidate(
        program_id=program_id,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        evidence_location="|".join(sorted(set(locations))),
    )


def _optional_nonnegative_int(
    value: str | None,
    *,
    location: str,
) -> int | None:
    parsed = _optional_int(value, location=location)
    if parsed is not None and parsed < 0:
        raise ValueError(
            f"Expected non-negative integer at {location}, "
            f"got {parsed!r}"
        )
    return parsed


def _optional_float(
    value: str | None,
    *,
    location: str,
) -> float | None:
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite float at {location}")
    return parsed


def _optional_int(
    value: str | None,
    *,
    location: str,
) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Expected integer at {location}, got {value!r}"
        ) from exc


def _parse_bool(value: str, *, location: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(
        f"Expected boolean at {location}, got {value!r}"
    )


def _split_reasons(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(
        reason.strip()
        for reason in value.replace(";", "|").split("|")
        if reason.strip()
    )


def _dedupe_programs(
    programs: Sequence[CandidateProgram],
) -> dict[str, CandidateProgram]:
    by_id: dict[str, CandidateProgram] = {}
    duplicates = []

    for program in programs:
        if program.program_id in by_id:
            duplicates.append(program.program_id)
        by_id[program.program_id] = program

    if duplicates:
        raise ValueError(
            "duplicate candidate program IDs: "
            + ", ".join(sorted(set(duplicates)))
        )

    return by_id


def _dedupe_validation_records(
    records: Sequence[CandidateValidationResult],
    *,
    policy: CandidateSelectionPolicy,
) -> dict[str, CandidateValidationResult]:
    by_program: dict[str, CandidateValidationResult] = {}

    for record in records:
        if (
            record.dataset != policy.dataset
            or record.task != policy.task
        ):
            continue

        existing = by_program.get(record.program_id)
        if existing is None:
            by_program[record.program_id] = record
            continue

        if existing != record:
            raise ValueError(
                "conflicting duplicate validation results for "
                f"{record.program_id!r}"
            )

    return by_program


def _record_rejection_reasons(
    record: CandidateValidationResult,
    *,
    policy: CandidateSelectionPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if record.dataset != policy.dataset or record.task != policy.task:
        reasons.append("task_mismatch")

    if record.primary_metric != policy.primary_metric:
        reasons.append("metric_mismatch")

    if record.metric_direction != policy.metric_direction:
        reasons.append("metric_direction_mismatch")

    if _is_test_or_final_split(record.split):
        reasons.append("test_or_final_split_evidence")
    elif record.split.strip().lower() not in {
        "validation",
        "valid",
        "val",
        "dev",
    }:
        reasons.append("non_validation_split")

    if record.validation_score is None:
        reasons.append("missing_validation_score")
    elif not math.isfinite(record.validation_score):
        reasons.append("missing_validation_score")

    if not record.eligible:
        reasons.append("ineligible")

    if record.materializable is None:
        reasons.append("unknown_materialization_feasibility")
    elif not record.materializable:
        reasons.append("infeasible")

    if record.leakage_safe is None:
        reasons.append("unknown_leakage_safety")
    elif not record.leakage_safe:
        reasons.append("leakage_violation")

    if record.temporally_safe is None:
        reasons.append("unknown_temporal_safety")
    elif not record.temporally_safe:
        reasons.append("temporal_safety_violation")

    if record.provenance_complete is None:
        reasons.append("unknown_provenance_completeness")
    elif not record.provenance_complete:
        reasons.append("incomplete_provenance_contract")

    reasons.extend(record.rejection_reasons)
    return tuple(dict.fromkeys(reasons))


def _is_test_or_final_split(split: str) -> bool:
    normalized = split.strip().lower().replace("-", "_")
    return "test" in normalized or "final" in normalized


def _require_score(
    record: CandidateValidationResult,
    *,
    reason: str,
) -> float:
    if record.validation_score is None:
        raise ValueError(reason)
    if not math.isfinite(record.validation_score):
        raise ValueError(reason)
    return record.validation_score


def _feature_count(
    program: CandidateProgram | None,
    record: CandidateValidationResult,
) -> int | None:
    if record.n_features is not None:
        return record.n_features
    if program is not None:
        return len(program.primitive_ids)
    return None


def _added_features(
    candidate_features: int | None,
    baseline_features: int | None,
) -> int | None:
    if candidate_features is None or baseline_features is None:
        return None
    return max(0, candidate_features - baseline_features)


def _family_complexity(
    program: CandidateProgram,
) -> tuple[int, ...]:
    order = {
        "temporal": 1,
        "structural": 2,
        "coverage": 3,
        "baseline": 4,
    }
    return tuple(
        sorted(
            order.get(family, 100)
            for family in set(program.families)
            if family != "baseline"
        )
    ) + (len(set(program.families) - {"baseline"}),)


def _ranking_key(
    candidate: RankedCandidate,
    *,
    metric_direction: str,
) -> tuple[float, int, tuple[int, ...], str]:
    if metric_direction == "higher":
        score_key = -candidate.validation_score
    else:
        score_key = candidate.validation_score

    added_features = (
        candidate.added_features
        if candidate.added_features is not None
        else 10**12
    )
    return (
        score_key,
        added_features,
        candidate.family_complexity,
        candidate.program_id,
    )


def _improvement(
    *,
    candidate_score: float,
    baseline_score: float,
    metric_direction: str,
) -> float:
    if metric_direction == "higher":
        return candidate_score - baseline_score
    return baseline_score - candidate_score


def select_program(
    scores: list[ProgramScore],
    *,
    metric_direction: str,
    primary_tolerance: float = 1e-12,
    secondary_tolerance: float = 1e-12,
) -> ProgramScore:
    """
    Select a candidate program with tolerance-aware lexicographic
    ordering.

    Ordering:
      1. primary validation metric
      2. secondary validation metric
      3. fewer physical features
      4. deterministic program-id tie break

    Metric differences within tolerance are treated as ties.
    """
    if not scores:
        raise ValueError("No candidate program scores")

    if metric_direction not in {"lower", "higher"}:
        raise ValueError(
            "metric_direction must be 'lower' or 'higher', "
            f"got {metric_direction!r}"
        )

    candidates = list(scores)

    primary_values = [
        score.primary_mean
        for score in candidates
    ]

    if metric_direction == "higher":
        best_primary = max(primary_values)

        candidates = [
            score
            for score in candidates
            if (
                best_primary - score.primary_mean
                <= primary_tolerance
            )
        ]
    else:
        best_primary = min(primary_values)

        candidates = [
            score
            for score in candidates
            if (
                score.primary_mean - best_primary
                <= primary_tolerance
            )
        ]

    candidates_with_secondary = [
        score
        for score in candidates
        if score.secondary_mean is not None
    ]

    if candidates_with_secondary:
        secondary_values = [
            score.secondary_mean
            for score in candidates_with_secondary
        ]

        if metric_direction == "higher":
            best_secondary = max(secondary_values)

            candidates = [
                score
                for score in candidates_with_secondary
                if (
                    best_secondary
                    - float(score.secondary_mean)
                    <= secondary_tolerance
                )
            ]
        else:
            best_secondary = min(secondary_values)

            candidates = [
                score
                for score in candidates_with_secondary
                if (
                    float(score.secondary_mean)
                    - best_secondary
                    <= secondary_tolerance
                )
            ]

    return min(
        candidates,
        key=lambda score: (
            score.n_features_mean,
            score.program_id,
        ),
    )


# Backward-compatible alias for any older scripts.
def select_lower_is_better(
    scores: list[ProgramScore],
) -> ProgramScore:
    return select_program(
        scores,
        metric_direction="lower",
    )
