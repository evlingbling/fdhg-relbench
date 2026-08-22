from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
)
from .passthrough_provenance import (
    PassthroughProvenanceReport,
    PROVEN,
)


AUDIT_COLUMNS = (
    "dataset",
    "task",
    "program_id",
    "audit_type",
    "primitive_id",
    "status",
    "passed",
    "source_table",
    "source_column",
    "output_column",
    "rejection_reason",
    "evidence_location",
    "notes",
)


@dataclass(frozen=True)
class CandidateSafetyAuditRow:
    dataset: str
    task: str
    program_id: str
    audit_type: str
    primitive_id: str
    status: str
    passed: bool
    source_table: str | None = None
    source_column: str | None = None
    output_column: str | None = None
    rejection_reason: str = ""
    evidence_location: str = ""
    notes: tuple[str, ...] = ()

    def to_csv_row(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "program_id": self.program_id,
            "audit_type": self.audit_type,
            "primitive_id": self.primitive_id,
            "status": self.status,
            "passed": "true" if self.passed else "false",
            "source_table": self.source_table or "",
            "source_column": self.source_column or "",
            "output_column": self.output_column or "",
            "rejection_reason": self.rejection_reason,
            "evidence_location": self.evidence_location,
            "notes": "|".join(self.notes),
        }


@dataclass(frozen=True)
class TemporalSafetyAudit:
    rows: tuple[CandidateSafetyAuditRow, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)


@dataclass(frozen=True)
class LeakageSafetyAudit:
    rows: tuple[CandidateSafetyAuditRow, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)


@dataclass(frozen=True)
class LoweringProvenanceAudit:
    rows: tuple[CandidateSafetyAuditRow, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)


@dataclass(frozen=True)
class CandidateSafetyAuditReport:
    dataset: str
    task: str
    program_id: str
    temporal: TemporalSafetyAudit
    leakage: LeakageSafetyAudit
    provenance: LoweringProvenanceAudit


@dataclass(frozen=True)
class ExplicitLoweringEvidence:
    dataset: str
    task: str
    program_id: str
    primitive_id: str
    source_table: str | None
    source_column: str | None
    output_column: str | None
    status: str
    evidence_location: str
    notes: tuple[str, ...] = ()


def audit_temporal_safety(
    *,
    dataset: str,
    task: str,
    plan: CandidateMaterializationPlan,
) -> TemporalSafetyAudit:
    rows = []
    for step in plan.steps:
        rows.append(
            _temporal_row(
                dataset=dataset,
                task=task,
                step=step,
            )
        )
    return TemporalSafetyAudit(
        rows=tuple(sorted(rows, key=_row_sort_key))
    )


def audit_leakage_safety(
    *,
    dataset: str,
    task: str,
    program_id: str,
    feature_columns: Sequence[str] | None,
    label_col: str,
    candidate_id_columns: Sequence[str] = (),
    surrogate_key_columns: Sequence[str] = (),
    target_aggregate_columns: Sequence[str] = (),
    cross_fitted_target_aggregates: bool = False,
) -> LeakageSafetyAudit:
    rows: list[CandidateSafetyAuditRow] = []

    if feature_columns is None:
        rows.append(
            _audit_row(
                dataset,
                task,
                program_id,
                "leakage_safety",
                "__schema__",
                "missing_feature_schema",
                False,
                "missing_feature_schema_evidence",
                "feature schema is required for leakage audit",
            )
        )
        return LeakageSafetyAudit(rows=tuple(rows))

    feature_set = set(feature_columns)

    checks = [
        (
            "label_column",
            label_col in feature_set,
            "label_column_in_features",
            label_col,
        ),
    ]

    for column in surrogate_key_columns:
        checks.append((
            "surrogate_key",
            column in feature_set,
            "surrogate_key_in_features",
            column,
        ))

    for column in candidate_id_columns:
        checks.append((
            "candidate_id",
            column in feature_set,
            "candidate_id_feature_leakage",
            column,
        ))

    for column in target_aggregate_columns:
        checks.append((
            "target_aware_aggregate",
            column in feature_set and not cross_fitted_target_aggregates,
            "target_aggregate_without_cross_fitting",
            column,
        ))

    for status, failed, reason, column in checks:
        rows.append(
            CandidateSafetyAuditRow(
                dataset=dataset,
                task=task,
                program_id=program_id,
                audit_type="leakage_safety",
                primitive_id=column,
                status=status,
                passed=not failed,
                source_column=column,
                output_column=column,
                rejection_reason=reason if failed else "",
                evidence_location="feature-schema:provided",
            )
        )

    if not rows:
        rows.append(
            _audit_row(
                dataset,
                task,
                program_id,
                "leakage_safety",
                "__schema__",
                "no_leakage_columns_detected",
                True,
                "",
                "explicit feature schema checked",
            )
        )

    return LeakageSafetyAudit(
        rows=tuple(sorted(rows, key=_row_sort_key))
    )


def audit_lowering_provenance(
    *,
    dataset: str,
    task: str,
    plan: CandidateMaterializationPlan,
    passthrough_report: PassthroughProvenanceReport | None = None,
    explicit_evidence: Sequence[ExplicitLoweringEvidence] = (),
) -> LoweringProvenanceAudit:
    explicit_by_primitive = _index_explicit_evidence(
        dataset=dataset,
        task=task,
        program_id=plan.program_id,
        evidence=explicit_evidence,
    )
    passthrough_by_primitive = {
        row.primitive_id: row
        for row in (
            passthrough_report.binding_evidence
            if passthrough_report is not None
            else ()
        )
    }
    rows = []

    for step in plan.steps:
        explicit = explicit_by_primitive.get(step.primitive_id)
        if explicit is not None:
            rows.append(
                CandidateSafetyAuditRow(
                    dataset=dataset,
                    task=task,
                    program_id=plan.program_id,
                    audit_type="lowering_provenance",
                    primitive_id=step.primitive_id,
                    status=explicit.status,
                    passed=explicit.status == "proven",
                    source_table=explicit.source_table,
                    source_column=explicit.source_column,
                    output_column=explicit.output_column,
                    rejection_reason=(
                        ""
                        if explicit.status == "proven"
                        else "explicit_provenance_not_proven"
                    ),
                    evidence_location=explicit.evidence_location,
                    notes=explicit.notes,
                )
            )
            continue

        if step.lowering_mode == LoweringMode.GENERATE:
            rows.append(
                CandidateSafetyAuditRow(
                    dataset=dataset,
                    task=task,
                    program_id=plan.program_id,
                    audit_type="lowering_provenance",
                    primitive_id=step.primitive_id,
                    status="native_lowering",
                    passed=step.materializable and bool(step.output_columns),
                    source_table=step.source_table,
                    source_column=step.source_event_time_col,
                    output_column="|".join(step.output_columns),
                    rejection_reason=(
                        ""
                        if step.materializable and step.output_columns
                        else "native_lowering_incomplete"
                    ),
                    evidence_location="materialization-plan",
                )
            )
            continue

        if step.lowering_mode == LoweringMode.PASSTHROUGH:
            evidence = passthrough_by_primitive.get(step.primitive_id)
            if evidence is None:
                rows.append(
                    _provenance_failure_row(
                        dataset,
                        task,
                        plan.program_id,
                        step.primitive_id,
                        "missing_passthrough_provenance",
                    )
                )
                continue
            rows.append(
                CandidateSafetyAuditRow(
                    dataset=dataset,
                    task=task,
                    program_id=plan.program_id,
                    audit_type="lowering_provenance",
                    primitive_id=step.primitive_id,
                    status=evidence.status,
                    passed=evidence.status == PROVEN,
                    source_column=evidence.source_column,
                    output_column=evidence.output_column,
                    rejection_reason=(
                        ""
                        if evidence.status == PROVEN
                        else f"passthrough_{evidence.status}"
                    ),
                    evidence_location=evidence.evidence_location,
                    notes=evidence.notes,
                )
            )
            continue

        if step.lowering_mode == LoweringMode.EXTERNAL:
            rows.append(
                _provenance_failure_row(
                    dataset,
                    task,
                    plan.program_id,
                    step.primitive_id,
                    "missing_existing_backend_binding",
                )
            )
            continue

        rows.append(
            _provenance_failure_row(
                dataset,
                task,
                plan.program_id,
                step.primitive_id,
                "unsupported_lowering_mode",
            )
        )

    return LoweringProvenanceAudit(
        rows=tuple(sorted(rows, key=_row_sort_key))
    )


def build_candidate_safety_audit_report(
    *,
    dataset: str,
    task: str,
    plan: CandidateMaterializationPlan,
    feature_columns: Sequence[str] | None,
    label_col: str,
    candidate_id_columns: Sequence[str] = (),
    surrogate_key_columns: Sequence[str] = (),
    target_aggregate_columns: Sequence[str] = (),
    cross_fitted_target_aggregates: bool = False,
    passthrough_report: PassthroughProvenanceReport | None = None,
    explicit_evidence: Sequence[ExplicitLoweringEvidence] = (),
) -> CandidateSafetyAuditReport:
    return CandidateSafetyAuditReport(
        dataset=dataset,
        task=task,
        program_id=plan.program_id,
        temporal=audit_temporal_safety(
            dataset=dataset,
            task=task,
            plan=plan,
        ),
        leakage=audit_leakage_safety(
            dataset=dataset,
            task=task,
            program_id=plan.program_id,
            feature_columns=feature_columns,
            label_col=label_col,
            candidate_id_columns=candidate_id_columns,
            surrogate_key_columns=surrogate_key_columns,
            target_aggregate_columns=target_aggregate_columns,
            cross_fitted_target_aggregates=(
                cross_fitted_target_aggregates
            ),
        ),
        provenance=audit_lowering_provenance(
            dataset=dataset,
            task=task,
            plan=plan,
            passthrough_report=passthrough_report,
            explicit_evidence=explicit_evidence,
        ),
    )


def write_audit_csv(
    rows: Sequence[CandidateSafetyAuditRow],
    handle: TextIO,
) -> None:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(AUDIT_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in sorted(rows, key=_row_sort_key):
        writer.writerow(row.to_csv_row())


def _temporal_row(
    *,
    dataset: str,
    task: str,
    step,
) -> CandidateSafetyAuditRow:
    if step.lowering_mode in {LoweringMode.PASSTHROUGH, LoweringMode.EXTERNAL}:
        return CandidateSafetyAuditRow(
            dataset=dataset,
            task=task,
            program_id=step.program_id,
            audit_type="temporal_safety",
            primitive_id=step.primitive_id,
            status="not_applicable",
            passed=True,
            evidence_location="materialization-plan",
            notes=("non-temporal provider primitive",),
        )

    reasons = []
    if not step.source_event_time_col or not step.target_time_col:
        reasons.append("missing_timestamp_requirement")
    if step.cutoff_operator != "<":
        reasons.append("invalid_as_of_direction")
        reasons.append("future_event_access")
    if step.window_days is not None and step.window_days <= 0:
        reasons.append("invalid_temporal_window")
    if not step.temporally_safe:
        reasons.append("materializer_temporal_policy_failed")

    return CandidateSafetyAuditRow(
        dataset=dataset,
        task=task,
        program_id=step.program_id,
        audit_type="temporal_safety",
        primitive_id=step.primitive_id,
        status="temporal_primitive",
        passed=not reasons,
        source_table=step.source_table,
        source_column=step.source_event_time_col,
        output_column="|".join(step.output_columns),
        rejection_reason="|".join(dict.fromkeys(reasons)),
        evidence_location="materialization-plan",
        notes=("strict event_time < prediction_time required",),
    )


def _index_explicit_evidence(
    *,
    dataset: str,
    task: str,
    program_id: str,
    evidence: Sequence[ExplicitLoweringEvidence],
) -> dict[str, ExplicitLoweringEvidence]:
    out = {}
    for record in evidence:
        if (
            record.dataset != dataset
            or record.task != task
            or record.program_id != program_id
        ):
            continue
        existing = out.get(record.primitive_id)
        if existing is not None and existing != record:
            raise ValueError(
                "conflicting provenance evidence for "
                f"{record.primitive_id}"
            )
        out[record.primitive_id] = record
    return out


def _provenance_failure_row(
    dataset: str,
    task: str,
    program_id: str,
    primitive_id: str,
    reason: str,
) -> CandidateSafetyAuditRow:
    return CandidateSafetyAuditRow(
        dataset=dataset,
        task=task,
        program_id=program_id,
        audit_type="lowering_provenance",
        primitive_id=primitive_id,
        status="missing",
        passed=False,
        rejection_reason=reason,
        evidence_location="materialization-plan",
    )


def _audit_row(
    dataset: str,
    task: str,
    program_id: str,
    audit_type: str,
    primitive_id: str,
    status: str,
    passed: bool,
    rejection_reason: str,
    note: str,
) -> CandidateSafetyAuditRow:
    return CandidateSafetyAuditRow(
        dataset=dataset,
        task=task,
        program_id=program_id,
        audit_type=audit_type,
        primitive_id=primitive_id,
        status=status,
        passed=passed,
        rejection_reason=rejection_reason,
        evidence_location="static-audit",
        notes=(note,),
    )


def _row_sort_key(row: CandidateSafetyAuditRow):
    return (
        row.dataset,
        row.task,
        row.program_id,
        row.audit_type,
        row.primitive_id,
        row.status,
        row.evidence_location,
    )
