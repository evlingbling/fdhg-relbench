from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO

from .selection import CandidateValidationResult


CANONICAL_VALIDATION_COLUMNS = (
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

AGGREGATE_VALIDATION_REQUIRED_COLUMNS = frozenset({
    "dataset",
    "task",
    "program_id",
    "split",
    "primary_metric",
    "metric_direction",
    "validation_score",
    "n_features",
    "eligible",
    "rejection_reason",
    "evidence_location",
    "materializable",
    "leakage_safe",
    "temporally_safe",
    "provenance_complete",
})

SEED_VALIDATION_REQUIRED_COLUMNS = frozenset({
    "dataset",
    "task",
    "program_id",
    "split",
    "seed",
    "primary_metric",
    "metric_direction",
    "validation_score",
    "n_features",
    "eligible",
    "rejection_reason",
    "evidence_location",
    "materializable",
    "leakage_safe",
    "temporally_safe",
    "provenance_complete",
})

SAFETY_COLUMNS = (
    "materializable",
    "leakage_safe",
    "temporally_safe",
    "provenance_complete",
)


@dataclass(frozen=True)
class ValidationArtifactAdapter:
    name: str
    required_columns: tuple[str, ...]
    row_interpretation: str
    split_policy: str
    safety_policy: str


@dataclass(frozen=True)
class NormalizedCandidateRecord:
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

    def to_candidate_validation_result(
        self,
    ) -> CandidateValidationResult:
        return CandidateValidationResult(
            dataset=self.dataset,
            task=self.task,
            program_id=self.program_id,
            split=self.split,
            primary_metric=self.primary_metric,
            metric_direction=self.metric_direction,
            validation_score=self.score,
            baseline_program_id=self.baseline_program_id,
            baseline_score=self.baseline_score,
            n_features=self.n_features,
            eligible=self.eligible,
            rejection_reasons=self.rejection_reasons,
            evidence_location=self.evidence_location,
            materializable=self.materializable,
            leakage_safe=self.leakage_safe,
            temporally_safe=self.temporally_safe,
            provenance_complete=self.provenance_complete,
        )

    def to_csv_row(self) -> dict[str, str]:
        return {
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
            "eligible": _format_bool(self.eligible),
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


@dataclass(frozen=True)
class RejectedSourceRecord:
    source_path: str
    row_number: int | None
    reason: str
    evidence_location: str
    row_identity: str


@dataclass(frozen=True)
class ValidationNormalizationReport:
    source_path: str
    adapter_name: str
    source_schema: tuple[str, ...]
    normalized_records: tuple[NormalizedCandidateRecord, ...]
    rejected_records: tuple[RejectedSourceRecord, ...]
    validation_only: bool
    supported: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SourceArtifactAuditRow:
    source_path: str
    schema: str
    validation_only_status: str
    task_identity_explicit: bool
    program_identity_explicit: bool
    safety_evidence_available: bool
    adapter_supported: bool
    reason: str


EXPLICIT_AGGREGATE_VALIDATION_ADAPTER = ValidationArtifactAdapter(
    name="explicit_aggregate_validation_v1",
    required_columns=tuple(
        sorted(AGGREGATE_VALIDATION_REQUIRED_COLUMNS)
    ),
    row_interpretation=(
        "one row is one task-scoped aggregate validation metric "
        "for one candidate program"
    ),
    split_policy=(
        "split must be explicit; test/final/unclear split rows "
        "are emitted as ineligible audit records"
    ),
    safety_policy=(
        "materialization, leakage, temporal-safety, and "
        "provenance fields must be explicit booleans"
    ),
)

EXPLICIT_SEED_VALIDATION_ADAPTER = ValidationArtifactAdapter(
    name="explicit_seed_validation_v1",
    required_columns=tuple(
        sorted(SEED_VALIDATION_REQUIRED_COLUMNS)
    ),
    row_interpretation=(
        "rows are unique seed-level validation metrics; seeds are "
        "averaged only within identical dataset/task/program/split/"
        "metric/direction groups"
    ),
    split_policy=(
        "all seeds for a program must share validation split; "
        "validation/test mixing is a conflict"
    ),
    safety_policy=(
        "safety booleans must be explicit and identical across "
        "contributing seeds"
    ),
)


def normalize_validation_artifact(
    path: Path,
    *,
    dataset: str | None = None,
    task: str | None = None,
) -> ValidationNormalizationReport:
    rows, fieldnames = _read_csv_rows(path)
    fields = set(fieldnames)

    if SEED_VALIDATION_REQUIRED_COLUMNS <= fields:
        report = _normalize_seed_rows(
            path=path,
            fieldnames=fieldnames,
            rows=rows,
            dataset=dataset,
            task=task,
        )
    elif AGGREGATE_VALIDATION_REQUIRED_COLUMNS <= fields:
        report = _normalize_aggregate_rows(
            path=path,
            fieldnames=fieldnames,
            rows=rows,
            dataset=dataset,
            task=task,
        )
    else:
        missing = sorted(
            AGGREGATE_VALIDATION_REQUIRED_COLUMNS - fields
        )
        report = ValidationNormalizationReport(
            source_path=str(path),
            adapter_name="unsupported",
            source_schema=tuple(fieldnames),
            normalized_records=(),
            rejected_records=(
                RejectedSourceRecord(
                    source_path=str(path),
                    row_number=None,
                    reason=(
                        "unsupported_schema_missing_columns:"
                        + "|".join(missing)
                    ),
                    evidence_location=f"csv-schema:{path}",
                    row_identity="",
                ),
            ),
            validation_only=False,
            supported=False,
            notes=(
                "No adapter selected; schema lacks explicit "
                "selector identity and safety evidence.",
            ),
        )

    return report


def audit_validation_sources(
    paths: Sequence[Path],
) -> tuple[SourceArtifactAuditRow, ...]:
    rows = []
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            _, fieldnames = _read_csv_rows(path)
            fields = set(fieldnames)
            supported = (
                SEED_VALIDATION_REQUIRED_COLUMNS <= fields
                or AGGREGATE_VALIDATION_REQUIRED_COLUMNS <= fields
            )
            split_status = _audit_split_status(fields)
            rows.append(
                SourceArtifactAuditRow(
                    source_path=str(path),
                    schema=",".join(fieldnames),
                    validation_only_status=split_status,
                    task_identity_explicit=(
                        {"dataset", "task"} <= fields
                    ),
                    program_identity_explicit=(
                        "program_id" in fields
                    ),
                    safety_evidence_available=(
                        set(SAFETY_COLUMNS) <= fields
                    ),
                    adapter_supported=supported,
                    reason=_audit_reason(
                        path=path,
                        fields=fields,
                        supported=supported,
                    ),
                )
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            rows.append(
                SourceArtifactAuditRow(
                    source_path=str(path),
                    schema="unavailable",
                    validation_only_status="unavailable",
                    task_identity_explicit=False,
                    program_identity_explicit=False,
                    safety_evidence_available=False,
                    adapter_supported=False,
                    reason=str(exc),
                )
            )
    return tuple(rows)


def records_to_candidate_validation_results(
    records: Sequence[NormalizedCandidateRecord],
) -> tuple[CandidateValidationResult, ...]:
    return tuple(
        record.to_candidate_validation_result()
        for record in records
    )


def write_canonical_validation_csv(
    records: Sequence[NormalizedCandidateRecord],
    handle: TextIO,
) -> None:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(CANONICAL_VALIDATION_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for record in sorted(
        records,
        key=lambda item: (
            item.dataset,
            item.task,
            item.program_id,
            item.split,
            item.primary_metric,
            item.evidence_location,
        ),
    ):
        writer.writerow(record.to_csv_row())


def _normalize_aggregate_rows(
    *,
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[tuple[int, dict[str, str]]],
    dataset: str | None,
    task: str | None,
) -> ValidationNormalizationReport:
    normalized = []
    rejected = []
    seen: dict[
        tuple[str, str, str, str, str, str],
        NormalizedCandidateRecord,
    ] = {}

    for row_number, row in rows:
        record = _record_from_row(
            path=path,
            row=row,
            row_number=row_number,
        )
        key = _record_key(record)
        existing = seen.get(key)
        if existing is not None:
            if existing != record:
                raise ValueError(
                    "conflicting metric rows for "
                    + "/".join(key)
                )
            continue
        seen[key] = record

        if dataset is not None and record.dataset != dataset:
            rejected.append(
                _rejected_from_record(
                    path=path,
                    record=record,
                    row_number=row_number,
                    reason="task_mismatch",
                )
            )
            continue
        if task is not None and record.task != task:
            rejected.append(
                _rejected_from_record(
                    path=path,
                    record=record,
                    row_number=row_number,
                    reason="task_mismatch",
                )
            )
            continue

        normalized.append(_apply_split_policy(record))

    return ValidationNormalizationReport(
        source_path=str(path),
        adapter_name=EXPLICIT_AGGREGATE_VALIDATION_ADAPTER.name,
        source_schema=tuple(fieldnames),
        normalized_records=tuple(
            sorted(normalized, key=_normalized_sort_key)
        ),
        rejected_records=tuple(
            sorted(rejected, key=_rejected_sort_key)
        ),
        validation_only=all(
            _is_validation_split(record.split)
            for record in normalized
        ),
        supported=True,
        notes=(
            EXPLICIT_AGGREGATE_VALIDATION_ADAPTER.row_interpretation,
            EXPLICIT_AGGREGATE_VALIDATION_ADAPTER.safety_policy,
        ),
    )


def _normalize_seed_rows(
    *,
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[tuple[int, dict[str, str]]],
    dataset: str | None,
    task: str | None,
) -> ValidationNormalizationReport:
    seed_records = []
    rejected = []

    for row_number, row in rows:
        seed_record = _seed_record_from_row(
            path=path,
            row=row,
            row_number=row_number,
        )
        if dataset is not None and seed_record.record.dataset != dataset:
            rejected.append(
                _rejected_from_record(
                    path=path,
                    record=seed_record.record,
                    row_number=row_number,
                    reason="task_mismatch",
                )
            )
            continue
        if task is not None and seed_record.record.task != task:
            rejected.append(
                _rejected_from_record(
                    path=path,
                    record=seed_record.record,
                    row_number=row_number,
                    reason="task_mismatch",
                )
            )
            continue
        seed_records.append(seed_record)

    _reject_validation_test_seed_mixing(seed_records)

    groups: dict[
        tuple[str, str, str, str, str, str],
        list[_SeedRecord],
    ] = {}
    for seed_record in seed_records:
        groups.setdefault(
            _record_key(seed_record.record),
            [],
        ).append(seed_record)

    normalized = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item.seed)
        seeds = [item.seed for item in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                "duplicate seed for " + "/".join(key)
            )
        _require_compatible_seed_group(key, group)
        normalized.append(_aggregate_seed_group(group))

    return ValidationNormalizationReport(
        source_path=str(path),
        adapter_name=EXPLICIT_SEED_VALIDATION_ADAPTER.name,
        source_schema=tuple(fieldnames),
        normalized_records=tuple(
            sorted(normalized, key=_normalized_sort_key)
        ),
        rejected_records=tuple(
            sorted(rejected, key=_rejected_sort_key)
        ),
        validation_only=all(
            _is_validation_split(record.split)
            for record in normalized
        ),
        supported=True,
        notes=(
            EXPLICIT_SEED_VALIDATION_ADAPTER.row_interpretation,
            EXPLICIT_SEED_VALIDATION_ADAPTER.safety_policy,
        ),
    )


@dataclass(frozen=True)
class _SeedRecord:
    seed: int
    row_number: int
    record: NormalizedCandidateRecord


def _seed_record_from_row(
    *,
    path: Path,
    row: dict[str, str],
    row_number: int,
) -> _SeedRecord:
    seed = _parse_nonnegative_int(
        row["seed"],
        location=f"{path}:{row_number}:seed",
    )
    return _SeedRecord(
        seed=seed,
        row_number=row_number,
        record=_record_from_row(
            path=path,
            row=row,
            row_number=row_number,
        ),
    )


def _record_from_row(
    *,
    path: Path,
    row: dict[str, str],
    row_number: int,
) -> NormalizedCandidateRecord:
    evidence_location = row["evidence_location"].strip()
    if not evidence_location:
        evidence_location = f"csv:{path}:{row_number}"

    return NormalizedCandidateRecord(
        dataset=_require_field(row, "dataset", path, row_number),
        task=_require_field(row, "task", path, row_number),
        program_id=_require_field(
            row,
            "program_id",
            path,
            row_number,
        ),
        split=_require_field(row, "split", path, row_number),
        primary_metric=_require_field(
            row,
            "primary_metric",
            path,
            row_number,
        ),
        metric_direction=_parse_direction(
            _require_field(
                row,
                "metric_direction",
                path,
                row_number,
            ),
            location=f"{path}:{row_number}:metric_direction",
        ),
        score=_parse_optional_float(
            row["validation_score"],
            location=f"{path}:{row_number}:validation_score",
        ),
        n_features=_parse_nonnegative_int(
            row["n_features"],
            location=f"{path}:{row_number}:n_features",
        ),
        eligible=_parse_bool(
            row["eligible"],
            location=f"{path}:{row_number}:eligible",
        ),
        rejection_reasons=_split_reasons(
            row.get("rejection_reason")
        ),
        evidence_location=evidence_location,
        materializable=_parse_optional_bool(
            row["materializable"],
            location=f"{path}:{row_number}:materializable",
        ),
        leakage_safe=_parse_optional_bool(
            row["leakage_safe"],
            location=f"{path}:{row_number}:leakage_safe",
        ),
        temporally_safe=_parse_optional_bool(
            row["temporally_safe"],
            location=f"{path}:{row_number}:temporally_safe",
        ),
        provenance_complete=_parse_optional_bool(
            row["provenance_complete"],
            location=f"{path}:{row_number}:provenance_complete",
        ),
        baseline_program_id=(
            row.get("baseline_program_id") or None
        ),
        baseline_score=_parse_optional_float(
            row.get("baseline_score"),
            location=f"{path}:{row_number}:baseline_score",
        ),
    )


def _apply_split_policy(
    record: NormalizedCandidateRecord,
) -> NormalizedCandidateRecord:
    reasons = list(record.rejection_reasons)
    eligible = record.eligible
    if _is_test_or_final_split(record.split):
        eligible = False
        reasons.append("test_or_final_split_evidence")
    elif not _is_validation_split(record.split):
        eligible = False
        reasons.append("non_validation_split")
    return NormalizedCandidateRecord(
        **{
            **record.__dict__,
            "eligible": eligible,
            "rejection_reasons": tuple(dict.fromkeys(reasons)),
        }
    )


def _aggregate_seed_group(
    group: Sequence[_SeedRecord],
) -> NormalizedCandidateRecord:
    first = group[0].record
    scores = [item.record.score for item in group]
    if any(score is None for score in scores):
        score = None
    else:
        score = sum(float(score) for score in scores) / len(scores)
    evidence_locations = "|".join(
        sorted({
            item.record.evidence_location
            for item in group
        })
    )
    reasons = []
    eligible = True
    for item in group:
        eligible = eligible and item.record.eligible
        reasons.extend(item.record.rejection_reasons)
    return _apply_split_policy(
        NormalizedCandidateRecord(
            dataset=first.dataset,
            task=first.task,
            program_id=first.program_id,
            split=first.split,
            primary_metric=first.primary_metric,
            metric_direction=first.metric_direction,
            score=score,
            n_features=first.n_features,
            eligible=eligible,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
            evidence_location=evidence_locations,
            materializable=first.materializable,
            leakage_safe=first.leakage_safe,
            temporally_safe=first.temporally_safe,
            provenance_complete=first.provenance_complete,
            baseline_program_id=first.baseline_program_id,
            baseline_score=first.baseline_score,
        )
    )


def _require_compatible_seed_group(
    key: tuple[str, str, str, str, str, str],
    group: Sequence[_SeedRecord],
) -> None:
    first = group[0].record
    for item in group[1:]:
        record = item.record
        if record.n_features != first.n_features:
            raise ValueError(
                "conflicting feature counts for " + "/".join(key)
            )
        if (
            record.materializable,
            record.leakage_safe,
            record.temporally_safe,
            record.provenance_complete,
        ) != (
            first.materializable,
            first.leakage_safe,
            first.temporally_safe,
            first.provenance_complete,
        ):
            raise ValueError(
                "conflicting safety evidence for " + "/".join(key)
            )
        if record.baseline_program_id != first.baseline_program_id:
            raise ValueError(
                "conflicting baseline program metadata for "
                + "/".join(key)
            )
        if record.baseline_score != first.baseline_score:
            raise ValueError(
                "conflicting baseline score metadata for "
                + "/".join(key)
            )


def _reject_validation_test_seed_mixing(
    seed_records: Sequence[_SeedRecord],
) -> None:
    splits_by_identity: dict[tuple[str, str, str], set[str]] = {}
    for seed_record in seed_records:
        record = seed_record.record
        key = (record.dataset, record.task, record.program_id)
        splits_by_identity.setdefault(key, set()).add(record.split)

    for key, splits in sorted(splits_by_identity.items()):
        has_validation = any(
            _is_validation_split(split) for split in splits
        )
        has_test = any(
            _is_test_or_final_split(split) for split in splits
        )
        if has_validation and has_test:
            raise ValueError(
                "mixed validation and test seed evidence for "
                + "/".join(key)
            )


def _read_csv_rows(
    path: Path,
) -> tuple[list[tuple[int, dict[str, str]]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return (
            [
                (row_number, dict(row))
                for row_number, row in enumerate(reader, start=2)
            ],
            tuple(reader.fieldnames),
        )


def _record_key(
    record: NormalizedCandidateRecord,
) -> tuple[str, str, str, str, str, str]:
    return (
        record.dataset,
        record.task,
        record.program_id,
        record.split,
        record.primary_metric,
        record.metric_direction,
    )


def _normalized_sort_key(
    record: NormalizedCandidateRecord,
) -> tuple[str, str, str, str, str, str]:
    return (
        record.dataset,
        record.task,
        record.program_id,
        record.split,
        record.primary_metric,
        record.evidence_location,
    )


def _rejected_sort_key(
    record: RejectedSourceRecord,
) -> tuple[str, int, str, str]:
    return (
        record.source_path,
        record.row_number or -1,
        record.reason,
        record.evidence_location,
    )


def _rejected_from_record(
    *,
    path: Path,
    record: NormalizedCandidateRecord,
    row_number: int,
    reason: str,
) -> RejectedSourceRecord:
    return RejectedSourceRecord(
        source_path=str(path),
        row_number=row_number,
        reason=reason,
        evidence_location=record.evidence_location,
        row_identity="/".join(_record_key(record)),
    )


def _require_field(
    row: dict[str, str],
    field: str,
    path: Path,
    row_number: int,
) -> str:
    value = row.get(field, "")
    if value.strip() == "":
        raise ValueError(
            f"missing required value at {path}:{row_number}:{field}"
        )
    return value


def _parse_optional_float(
    value: str | None,
    *,
    location: str,
) -> float | None:
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"expected finite float at {location}")
    return parsed


def _parse_nonnegative_int(
    value: str | None,
    *,
    location: str,
) -> int:
    if value is None or value.strip() == "":
        raise ValueError(f"expected integer at {location}")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(
            f"expected non-negative integer at {location}"
        )
    return parsed


def _parse_bool(
    value: str | None,
    *,
    location: str,
) -> bool:
    if value is None:
        raise ValueError(f"expected boolean at {location}")
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"expected boolean at {location}")


def _parse_optional_bool(
    value: str | None,
    *,
    location: str,
) -> bool | None:
    if value is None or value.strip() == "":
        return None
    return _parse_bool(value, location=location)


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return _format_bool(value)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return repr(float(value))


def _split_reasons(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(
        reason.strip()
        for reason in value.replace(";", "|").split("|")
        if reason.strip()
    )


def _parse_direction(value: str, *, location: str) -> str:
    if value not in {"higher", "lower"}:
        raise ValueError(
            f"expected metric direction higher/lower at {location}"
        )
    return value


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


def _audit_split_status(fields: set[str]) -> str:
    if "split" in fields:
        return "explicit_split_column"
    return "unclear_or_absent_split"


def _audit_reason(
    *,
    path: Path,
    fields: set[str],
    supported: bool,
) -> str:
    if supported:
        return "supported_explicit_validation_schema"
    if not {"dataset", "task"} <= fields:
        return "missing_explicit_dataset_task_identity"
    if "program_id" not in fields:
        if "selected_program_id" in fields:
            return "post_selection_summary_not_candidate_evidence"
        if "selected_candidate" in fields:
            return "selected_candidate_summary_not_program_id"
        return "missing_explicit_program_identity"
    if "split" not in fields:
        return "missing_explicit_validation_split"
    if not set(SAFETY_COLUMNS) <= fields:
        return "missing_explicit_safety_evidence"
    return f"unsupported_schema:{path.name}"


def iter_default_audit_paths() -> Iterable[Path]:
    for root in (Path("results/compiler"), Path("results/paper_tables")):
        if root.exists():
            yield from sorted(root.glob("*.csv"))
