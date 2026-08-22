from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO


CANONICAL_EXPORT_COLUMNS = (
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
    "baseline_program_id",
    "baseline_score",
)

SEED_EXPORT_COLUMNS = (
    *CANONICAL_EXPORT_COLUMNS,
    "seed",
)


@dataclass(frozen=True)
class CandidateSafetyEvidence:
    program_id: str
    materializable: bool | None
    leakage_safe: bool | None
    temporally_safe: bool | None
    provenance_complete: bool | None
    evidence_locations: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateValidationExportRecord:
    dataset: str
    task: str
    program_id: str
    split: str
    primary_metric: str
    metric_direction: str
    score: float | None
    n_features: int | None
    eligible: bool
    rejection_reasons: tuple[str, ...]
    evidence_location: str
    materializable: bool | None
    leakage_safe: bool | None
    temporally_safe: bool | None
    provenance_complete: bool | None
    baseline_program_id: str | None = None
    baseline_score: float | None = None
    seed: int | None = None

    def to_csv_row(self, *, include_seed: bool = False) -> dict[str, str]:
        row = {
            "dataset": self.dataset,
            "task": self.task,
            "program_id": self.program_id,
            "split": self.split,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "score": _format_optional_float(self.score),
            "n_features": (
                ""
                if self.n_features is None
                else str(self.n_features)
            ),
            "eligible": "true" if self.eligible else "false",
            "rejection_reason": "|".join(
                self.rejection_reasons
            ),
            "evidence_location": self.evidence_location,
            "materializable": _format_optional_bool(
                self.materializable
            ),
            "leakage_safe": _format_optional_bool(
                self.leakage_safe
            ),
            "temporally_safe": _format_optional_bool(
                self.temporally_safe
            ),
            "provenance_complete": _format_optional_bool(
                self.provenance_complete
            ),
            "baseline_program_id": (
                self.baseline_program_id or ""
            ),
            "baseline_score": _format_optional_float(
                self.baseline_score
            ),
        }
        if include_seed:
            row["seed"] = "" if self.seed is None else str(self.seed)
        return row


@dataclass(frozen=True)
class ValidationExportReport:
    dataset: str
    task: str
    split: str
    aggregate_records: tuple[CandidateValidationExportRecord, ...]
    seed_records: tuple[CandidateValidationExportRecord, ...]
    rejected_program_ids: tuple[str, ...]
    notes: tuple[str, ...]


def build_validation_export_records(
    *,
    dataset: str,
    task: str,
    split: str,
    primary_metric: str,
    metric_direction: str,
    candidate_program_ids: Sequence[str],
    expected_seeds: Sequence[int],
    aggregate_rows: Sequence[Mapping[str, object]],
    seed_rows: Sequence[Mapping[str, object]],
    selected_program_id: str,
    baseline_program_id: str = "dfs",
    safety_evidence_by_program: (
        Mapping[str, CandidateSafetyEvidence] | None
    ) = None,
) -> ValidationExportReport:
    if _is_test_or_final_split(split) or not _is_validation_split(split):
        raise ValueError(
            "canonical validation export requires validation split"
        )

    expected_seed_tuple = tuple(sorted(int(seed) for seed in expected_seeds))
    if len(expected_seed_tuple) != len(set(expected_seed_tuple)):
        raise ValueError("duplicate expected seeds")

    aggregate_by_program = _index_aggregate_rows(
        aggregate_rows,
        primary_metric=primary_metric,
    )
    seed_by_program = _index_seed_rows(
        seed_rows,
        primary_metric=primary_metric,
    )
    safety = dict(safety_evidence_by_program or {})

    baseline_score = None
    if baseline_program_id in aggregate_by_program:
        baseline_score = _score_from_aggregate(
            aggregate_by_program[baseline_program_id],
            primary_metric=primary_metric,
        )

    aggregate_records = []
    seed_records_out = []

    for program_id in sorted(set(candidate_program_ids)):
        program_seed_rows = tuple(
            sorted(
                seed_by_program.get(program_id, ()),
                key=lambda row: int(row["seed"]),
            )
        )
        _reject_duplicate_seed_rows(program_id, program_seed_rows)

        aggregate = aggregate_by_program.get(program_id)
        program_safety = safety.get(program_id)
        reasons = []

        if aggregate is None:
            score = None
            n_features = None
            reasons.append("metric_failure")
        else:
            score = _score_from_aggregate(
                aggregate,
                primary_metric=primary_metric,
            )
            n_features = _feature_count_from_aggregate(aggregate)

        available_seeds = tuple(
            int(row["seed"]) for row in program_seed_rows
        )
        missing_seeds = tuple(
            seed
            for seed in expected_seed_tuple
            if seed not in available_seeds
        )
        if missing_seeds:
            reasons.append("missing_seeds")

        reasons.extend(
            _stability_rejection_reasons(
                program_id=program_id,
                baseline_program_id=baseline_program_id,
                selected_program_id=selected_program_id,
                primary_metric=primary_metric,
                metric_direction=metric_direction,
                expected_seeds=expected_seed_tuple,
                seed_rows_by_program=seed_by_program,
                aggregate_rows_by_program=aggregate_by_program,
            )
        )

        materializable, leakage_safe, temporally_safe, provenance_complete = (
            _safety_tuple(program_safety)
        )
        reasons.extend(
            _safety_rejection_reasons(program_safety)
        )

        if program_id == baseline_program_id:
            reasons = [
                reason
                for reason in reasons
                if not reason.startswith("failed_")
                and reason not in {
                    "insufficient_seed_wins",
                    "unstable_paired_deltas",
                    "nonpositive_paired_delta",
                }
            ]

        eligible = (
            score is not None
            and not missing_seeds
            and not reasons
        )

        aggregate_records.append(
            CandidateValidationExportRecord(
                dataset=dataset,
                task=task,
                program_id=program_id,
                split=split,
                primary_metric=primary_metric,
                metric_direction=metric_direction,
                score=score,
                n_features=n_features,
                eligible=eligible,
                rejection_reasons=tuple(dict.fromkeys(reasons)),
                evidence_location=_aggregate_evidence_location(
                    aggregate,
                    program_seed_rows,
                    program_safety,
                ),
                materializable=materializable,
                leakage_safe=leakage_safe,
                temporally_safe=temporally_safe,
                provenance_complete=provenance_complete,
                baseline_program_id=baseline_program_id,
                baseline_score=baseline_score,
            )
        )

        for seed_row in program_seed_rows:
            seed_records_out.append(
                CandidateValidationExportRecord(
                    dataset=dataset,
                    task=task,
                    program_id=program_id,
                    split=split,
                    primary_metric=primary_metric,
                    metric_direction=metric_direction,
                    score=_score_from_seed(
                        seed_row,
                        primary_metric=primary_metric,
                    ),
                    n_features=_feature_count_from_seed(seed_row),
                    eligible=eligible,
                    rejection_reasons=tuple(dict.fromkeys(reasons)),
                    evidence_location=str(
                        seed_row.get(
                            "evidence_location",
                            "",
                        )
                    ),
                    materializable=materializable,
                    leakage_safe=leakage_safe,
                    temporally_safe=temporally_safe,
                    provenance_complete=provenance_complete,
                    baseline_program_id=baseline_program_id,
                    baseline_score=baseline_score,
                    seed=int(seed_row["seed"]),
                )
            )

    return ValidationExportReport(
        dataset=dataset,
        task=task,
        split=split,
        aggregate_records=tuple(
            sorted(aggregate_records, key=_record_sort_key)
        ),
        seed_records=tuple(
            sorted(seed_records_out, key=_seed_record_sort_key)
        ),
        rejected_program_ids=tuple(
            record.program_id
            for record in sorted(
                aggregate_records,
                key=_record_sort_key,
            )
            if not record.eligible
        ),
        notes=(
            "selection export contains validation split only",
            "eligibility combines stability gate and explicit safety evidence",
        ),
    )


def inspect_candidate_safety_evidence(
    *,
    dataset: str,
    task: str,
    program_id: str,
    artifact_dir: Path,
    baseline_program_id: str = "dfs",
) -> CandidateSafetyEvidence:
    locations = [f"artifact-dir:{artifact_dir}"]
    reasons = []
    train_path = artifact_dir / "target_with_dfs_agg_train.parquet"
    val_path = artifact_dir / "target_with_dfs_agg_val.parquet"
    has_artifacts = train_path.exists() and val_path.exists()

    materializable = has_artifacts
    if not has_artifacts:
        reasons.append("missing_candidate_artifacts")

    if program_id == baseline_program_id:
        if has_artifacts:
            locations.append("baseline_policy:dfs_reference")
            return CandidateSafetyEvidence(
                program_id=program_id,
                materializable=True,
                leakage_safe=True,
                temporally_safe=True,
                provenance_complete=True,
                evidence_locations=tuple(locations),
                rejection_reasons=tuple(reasons),
            )

    temporal = _read_temporal_safety(artifact_dir)
    leakage = _read_leakage_safety(artifact_dir)
    provenance = _read_provenance_complete(artifact_dir)

    if temporal is None:
        reasons.append("missing_temporal_safety_evidence")
    if leakage is None:
        reasons.append("missing_leakage_safety_evidence")
    if provenance is None:
        reasons.append("missing_provenance_evidence")

    if (
        dataset == "rel-ratebeer"
        and task == "user-place-liked_pairwise"
        and program_id != baseline_program_id
    ):
        provenance = False
        reasons.append("incomplete_passthrough_provenance")

    return CandidateSafetyEvidence(
        program_id=program_id,
        materializable=materializable,
        leakage_safe=leakage,
        temporally_safe=temporal,
        provenance_complete=provenance,
        evidence_locations=tuple(locations),
        rejection_reasons=tuple(dict.fromkeys(reasons)),
    )


def write_validation_export_csv(
    records: Sequence[CandidateValidationExportRecord],
    handle: TextIO,
    *,
    include_seed: bool = False,
) -> None:
    columns = (
        SEED_EXPORT_COLUMNS
        if include_seed
        else CANONICAL_EXPORT_COLUMNS
    )
    writer = csv.DictWriter(
        handle,
        fieldnames=list(columns),
        lineterminator="\n",
    )
    writer.writeheader()
    for record in sorted(records, key=_seed_record_sort_key):
        writer.writerow(
            record.to_csv_row(include_seed=include_seed)
        )


def _index_aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    primary_metric: str,
) -> dict[str, Mapping[str, object]]:
    out = {}
    for row in rows:
        program_id = str(row.get("candidate", row.get("program_id", "")))
        if not program_id:
            continue
        if primary_metric not in row and "primary_mean" not in row:
            continue
        existing = out.get(program_id)
        if existing is not None and existing != row:
            raise ValueError(
                f"conflicting aggregate rows for {program_id}"
            )
        out[program_id] = row
    return out


def _index_seed_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    primary_metric: str,
) -> dict[str, tuple[Mapping[str, object], ...]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        program_id = str(row.get("candidate", row.get("program_id", "")))
        if not program_id:
            continue
        if "seed" not in row or primary_metric not in row:
            continue
        grouped.setdefault(program_id, []).append(row)
    return {
        program_id: tuple(rows)
        for program_id, rows in grouped.items()
    }


def _reject_duplicate_seed_rows(
    program_id: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seed for {program_id}")


def _stability_rejection_reasons(
    *,
    program_id: str,
    baseline_program_id: str,
    selected_program_id: str,
    primary_metric: str,
    metric_direction: str,
    expected_seeds: tuple[int, ...],
    seed_rows_by_program: Mapping[str, tuple[Mapping[str, object], ...]],
    aggregate_rows_by_program: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    if program_id == baseline_program_id:
        return ()

    baseline_rows = {
        int(row["seed"]): row
        for row in seed_rows_by_program.get(baseline_program_id, ())
    }
    candidate_rows = {
        int(row["seed"]): row
        for row in seed_rows_by_program.get(program_id, ())
    }
    paired_seeds = [
        seed
        for seed in expected_seeds
        if seed in baseline_rows and seed in candidate_rows
    ]
    if len(paired_seeds) < len(expected_seeds):
        return ("missing_seeds",)

    lower_is_better = metric_direction == "lower"
    gains = []
    for seed in paired_seeds:
        baseline_score = _score_from_seed(
            baseline_rows[seed],
            primary_metric=primary_metric,
        )
        candidate_score = _score_from_seed(
            candidate_rows[seed],
            primary_metric=primary_metric,
        )
        gains.append(
            baseline_score - candidate_score
            if lower_is_better
            else candidate_score - baseline_score
        )

    wins = sum(1 for gain in gains if gain > 0)
    minimum_delta = min(gains)
    mean_gain = _mean_gain_from_aggregates(
        program_id=program_id,
        baseline_program_id=baseline_program_id,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        aggregate_rows_by_program=aggregate_rows_by_program,
    )
    minimum_mean_gain = _minimum_mean_gain(primary_metric)
    reasons = []

    if mean_gain is None or not math.isfinite(mean_gain):
        reasons.append("metric_failure")
    elif mean_gain <= minimum_mean_gain:
        reasons.append("failed_minimum_paired_delta")

    if wins < 3:
        reasons.append("insufficient_seed_wins")

    if minimum_delta <= 0:
        reasons.append("nonpositive_paired_delta")

    if wins < len(expected_seeds):
        reasons.append("unstable_paired_deltas")

    if program_id != selected_program_id:
        reasons.append("failed_stability_gate")

    return tuple(dict.fromkeys(reasons))


def _safety_tuple(
    safety: CandidateSafetyEvidence | None,
) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    if safety is None:
        return None, None, None, None
    return (
        safety.materializable,
        safety.leakage_safe,
        safety.temporally_safe,
        safety.provenance_complete,
    )


def _safety_rejection_reasons(
    safety: CandidateSafetyEvidence | None,
) -> tuple[str, ...]:
    if safety is None:
        return (
            "missing_materialization_evidence",
            "missing_leakage_safety_evidence",
            "missing_temporal_safety_evidence",
            "missing_provenance_evidence",
            "incomplete_safety_evidence",
        )
    reasons = list(safety.rejection_reasons)
    if safety.materializable is None:
        reasons.append("missing_materialization_evidence")
    elif not safety.materializable:
        reasons.append("missing_candidate_artifacts")
    if safety.leakage_safe is None:
        reasons.append("missing_leakage_safety_evidence")
    elif not safety.leakage_safe:
        reasons.append("leakage_audit_failure")
    if safety.temporally_safe is None:
        reasons.append("missing_temporal_safety_evidence")
    elif not safety.temporally_safe:
        reasons.append("temporal_audit_failure")
    if safety.provenance_complete is None:
        reasons.append("missing_provenance_evidence")
    elif not safety.provenance_complete:
        reasons.append("incomplete_provenance")
    if any(
        value is not True
        for value in (
            safety.materializable,
            safety.leakage_safe,
            safety.temporally_safe,
            safety.provenance_complete,
        )
    ):
        reasons.append("incomplete_safety_evidence")
    return tuple(dict.fromkeys(reasons))


def _aggregate_evidence_location(
    aggregate: Mapping[str, object] | None,
    seed_rows: Sequence[Mapping[str, object]],
    safety: CandidateSafetyEvidence | None,
) -> str:
    locations = []
    if aggregate is not None:
        location = aggregate.get("evidence_location")
        if location:
            locations.append(str(location))
    for row in seed_rows:
        location = row.get("evidence_location")
        if location:
            locations.append(str(location))
    if safety is not None:
        locations.extend(safety.evidence_locations)
    return "|".join(sorted(set(locations)))


def _score_from_aggregate(
    row: Mapping[str, object],
    *,
    primary_metric: str,
) -> float:
    value = row.get("primary_mean", row.get(primary_metric))
    return _finite_float(value)


def _score_from_seed(
    row: Mapping[str, object],
    *,
    primary_metric: str,
) -> float:
    return _finite_float(row[primary_metric])


def _feature_count_from_aggregate(
    row: Mapping[str, object],
) -> int | None:
    value = row.get("n_features_mean", row.get("n_features"))
    if value is None or value == "":
        return None
    return int(round(_finite_float(value)))


def _feature_count_from_seed(
    row: Mapping[str, object],
) -> int | None:
    value = row.get("n_features")
    if value is None or value == "":
        return None
    return int(round(_finite_float(value)))


def _mean_gain_from_aggregates(
    *,
    program_id: str,
    baseline_program_id: str,
    primary_metric: str,
    metric_direction: str,
    aggregate_rows_by_program: Mapping[str, Mapping[str, object]],
) -> float | None:
    baseline = aggregate_rows_by_program.get(baseline_program_id)
    candidate = aggregate_rows_by_program.get(program_id)
    if baseline is None or candidate is None:
        return None
    baseline_score = _score_from_aggregate(
        baseline,
        primary_metric=primary_metric,
    )
    candidate_score = _score_from_aggregate(
        candidate,
        primary_metric=primary_metric,
    )
    if metric_direction == "lower":
        return baseline_score - candidate_score
    return candidate_score - baseline_score


def _minimum_mean_gain(primary_metric: str) -> float:
    return {
        "accuracy": 0.001,
        "roc_auc": 0.0,
        "average_precision": 0.0,
        "macro_f1": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "mse": 0.0,
        "log_loss": 0.0,
    }.get(primary_metric, 0.0)


def _finite_float(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite metric value: {value!r}")
    return parsed


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return repr(float(value))


def _record_sort_key(
    record: CandidateValidationExportRecord,
) -> tuple[str, str, str, str, str]:
    return (
        record.dataset,
        record.task,
        record.program_id,
        record.split,
        record.primary_metric,
    )


def _seed_record_sort_key(
    record: CandidateValidationExportRecord,
) -> tuple[str, str, str, str, str, int]:
    return (
        record.dataset,
        record.task,
        record.program_id,
        record.split,
        record.primary_metric,
        -1 if record.seed is None else record.seed,
    )


def _is_validation_split(split: str) -> bool:
    return split.strip().lower() in {
        "validation",
        "valid",
        "val",
        "dev",
    }


def _is_test_or_final_split(split: str) -> bool:
    normalized = split.strip().lower().replace("-", "_")
    return "test" in normalized or "final" in normalized


def _read_temporal_safety(artifact_dir: Path) -> bool | None:
    path = artifact_dir / "temporal_safety_audit.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    strict = _read_candidate_safety_audit_rows(rows, "temporal_safety")
    if strict is not None:
        return strict
    if not rows or "temporally_safe" not in rows[0]:
        return None
    return all(
        str(row.get("temporally_safe", "")).lower() == "true"
        for row in rows
    )


def _read_leakage_safety(artifact_dir: Path) -> bool | None:
    path = artifact_dir / "leakage_safety_audit.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    strict = _read_candidate_safety_audit_rows(rows, "leakage_safety")
    if strict is not None:
        return strict
    if not rows or "leakage_safe" not in rows[0]:
        return None
    return all(
        str(row.get("leakage_safe", "")).lower() == "true"
        for row in rows
    )


def _read_provenance_complete(artifact_dir: Path) -> bool | None:
    path = artifact_dir / "lowering_provenance_audit.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    strict = _read_candidate_safety_audit_rows(rows, "lowering_provenance")
    if strict is not None:
        return strict
    if not rows:
        return None
    if "realized" in rows[0]:
        return all(
            str(row.get("realized", "")).lower() == "true"
            for row in rows
        )
    if "provenance_complete" in rows[0]:
        return all(
            str(row.get("provenance_complete", "")).lower()
            == "true"
            for row in rows
        )
    return None


def _read_candidate_safety_audit_rows(
    rows: list[dict[str, str]],
    audit_type: str,
) -> bool | None:
    if not rows:
        return None
    required = {"audit_type", "passed", "dataset", "task", "program_id"}
    if not required <= set(rows[0]):
        return None
    matching = [
        row for row in rows if row.get("audit_type") == audit_type
    ]
    if not matching:
        return None
    return all(
        str(row.get("passed", "")).strip().lower() == "true"
        for row in matching
    )
