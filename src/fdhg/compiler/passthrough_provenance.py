from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
)
from .passthrough_bindings import (
    PassthroughColumnBinding,
    ResolvedPassthroughContract,
)


PROVEN = "proven"
PARTIAL = "partial"
MISSING = "missing"
CONFLICTING = "conflicting"

VALID_EVIDENCE_STATUSES = frozenset({
    PROVEN,
    PARTIAL,
    MISSING,
    CONFLICTING,
})


@dataclass(frozen=True)
class PassthroughBindingEvidence:
    program_id: str
    primitive_id: str
    source_column: str | None
    output_column: str | None
    evidence_kind: str
    evidence_location: str
    status: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PassthroughProvenanceReport:
    program_id: str
    passthrough_step_count: int
    binding_evidence: tuple[PassthroughBindingEvidence, ...]
    proven_bindings: tuple[PassthroughBindingEvidence, ...]
    unresolved_primitive_ids: tuple[str, ...]
    conflicting_primitive_ids: tuple[str, ...]
    complete: bool


def build_passthrough_provenance_report(
    plan: CandidateMaterializationPlan,
    *,
    evidence_records: Sequence[PassthroughBindingEvidence],
) -> PassthroughProvenanceReport:
    passthrough_ids = tuple(
        step.primitive_id
        for step in plan.steps
        if step.lowering_mode == LoweringMode.PASSTHROUGH
    )
    passthrough_id_set = frozenset(passthrough_ids)
    steps_by_id = {step.primitive_id: step for step in plan.steps}

    deduped = _dedupe_evidence(tuple(evidence_records))
    evidence_by_primitive: dict[
        str,
        list[PassthroughBindingEvidence],
    ] = {}

    for record in deduped:
        _validate_evidence_record(
            plan=plan,
            steps_by_id=steps_by_id,
            passthrough_id_set=passthrough_id_set,
            record=record,
        )
        evidence_by_primitive.setdefault(
            record.primitive_id,
            [],
        ).append(record)

    rows: list[PassthroughBindingEvidence] = []
    unresolved: list[str] = []
    conflicts: list[str] = []

    for primitive_id in passthrough_ids:
        records = tuple(
            evidence_by_primitive.get(primitive_id, ())
        )

        if not records:
            rows.append(
                _missing_record(
                    program_id=plan.program_id,
                    primitive_id=primitive_id,
                )
            )
            unresolved.append(primitive_id)
            continue

        if _records_conflict(records):
            rows.append(
                _conflict_record(
                    program_id=plan.program_id,
                    primitive_id=primitive_id,
                    records=records,
                )
            )
            conflicts.append(primitive_id)
            continue

        merged = _merge_compatible_evidence(
            program_id=plan.program_id,
            primitive_id=primitive_id,
            records=records,
        )
        rows.append(merged)

        if merged.status != PROVEN:
            unresolved.append(primitive_id)

    complete = (
        len(conflicts) == 0
        and len(unresolved) == 0
        and len(rows) == len(passthrough_ids)
        and all(record.status == PROVEN for record in rows)
    )

    return PassthroughProvenanceReport(
        program_id=plan.program_id,
        passthrough_step_count=len(passthrough_ids),
        binding_evidence=tuple(rows),
        proven_bindings=tuple(
            row for row in rows if row.status == PROVEN
        ),
        unresolved_primitive_ids=tuple(unresolved),
        conflicting_primitive_ids=tuple(conflicts),
        complete=complete,
    )


def provenance_report_to_contract(
    report: PassthroughProvenanceReport,
) -> ResolvedPassthroughContract:
    if not report.complete:
        raise ValueError(
            "cannot build passthrough contract from incomplete "
            f"provenance report: program_id={report.program_id}"
        )

    if report.conflicting_primitive_ids:
        raise ValueError(
            "cannot build passthrough contract from conflicting "
            f"provenance report: program_id={report.program_id}"
        )

    bindings: list[PassthroughColumnBinding] = []

    for record in report.proven_bindings:
        if record.status != PROVEN:
            raise ValueError(
                "cannot build passthrough contract from "
                f"non-proven evidence: primitive_id="
                f"{record.primitive_id}"
            )
        if record.source_column is None:
            raise ValueError(
                "proven evidence is missing source column: "
                f"primitive_id={record.primitive_id}"
            )
        if record.output_column is None:
            raise ValueError(
                "proven evidence is missing output column: "
                f"primitive_id={record.primitive_id}"
            )

        bindings.append(
            PassthroughColumnBinding(
                program_id=record.program_id,
                primitive_id=record.primitive_id,
                source_column=record.source_column,
                output_column=record.output_column,
            )
        )

    return ResolvedPassthroughContract(
        program_id=report.program_id,
        bindings=tuple(bindings),
        source_columns=tuple(
            binding.source_column for binding in bindings
        ),
        output_columns=tuple(
            binding.output_column for binding in bindings
        ),
    )


def _dedupe_evidence(
    evidence_records: tuple[PassthroughBindingEvidence, ...],
) -> tuple[PassthroughBindingEvidence, ...]:
    seen: set[PassthroughBindingEvidence] = set()
    deduped: list[PassthroughBindingEvidence] = []

    for record in evidence_records:
        if record in seen:
            continue
        seen.add(record)
        deduped.append(record)

    return tuple(deduped)


def _validate_evidence_record(
    *,
    plan: CandidateMaterializationPlan,
    steps_by_id,
    passthrough_id_set: frozenset[str],
    record: PassthroughBindingEvidence,
) -> None:
    if record.program_id != plan.program_id:
        raise ValueError(
            "evidence program_id does not match plan: "
            f"expected={plan.program_id!r} "
            f"actual={record.program_id!r}"
        )

    step = steps_by_id.get(record.primitive_id)

    if step is None:
        raise ValueError(
            "evidence references unknown primitive_id: "
            f"{record.primitive_id}"
        )

    if record.primitive_id not in passthrough_id_set:
        raise ValueError(
            "evidence references non-passthrough primitive_id: "
            f"{record.primitive_id}"
        )

    if record.status not in VALID_EVIDENCE_STATUSES:
        raise ValueError(
            "unknown passthrough evidence status: "
            f"{record.status!r}"
        )

    if record.status == PROVEN and record.source_column is None:
        raise ValueError(
            "proven passthrough evidence requires source_column: "
            f"primitive_id={record.primitive_id}"
        )

    if record.status == PROVEN and record.output_column is None:
        raise ValueError(
            "proven passthrough evidence requires output_column: "
            f"primitive_id={record.primitive_id}"
        )


def _records_conflict(
    records: tuple[PassthroughBindingEvidence, ...],
) -> bool:
    if any(record.status == CONFLICTING for record in records):
        return True

    source_columns = {
        record.source_column
        for record in records
        if record.source_column is not None
    }
    output_columns = {
        record.output_column
        for record in records
        if record.output_column is not None
    }

    return len(source_columns) > 1 or len(output_columns) > 1


def _merge_compatible_evidence(
    *,
    program_id: str,
    primitive_id: str,
    records: tuple[PassthroughBindingEvidence, ...],
) -> PassthroughBindingEvidence:
    ordered = sorted(
        records,
        key=lambda record: (
            record.evidence_kind,
            record.evidence_location,
            record.source_column or "",
            record.output_column or "",
            record.notes,
        ),
    )
    source_column = _single_non_null(
        record.source_column for record in ordered
    )
    output_column = _single_non_null(
        record.output_column for record in ordered
    )
    has_proven_record = any(
        record.status == PROVEN for record in ordered
    )
    if (
        source_column is None
        and output_column is None
        and all(record.status == MISSING for record in ordered)
    ):
        status = MISSING
    elif (
        has_proven_record
        and source_column is not None
        and output_column is not None
    ):
        status = PROVEN
    else:
        status = PARTIAL
    evidence_kinds = tuple(
        sorted({record.evidence_kind for record in ordered})
    )
    evidence_locations = tuple(
        sorted({record.evidence_location for record in ordered})
    )
    notes = tuple(
        sorted({
            note
            for record in ordered
            for note in record.notes
        })
    )

    return PassthroughBindingEvidence(
        program_id=program_id,
        primitive_id=primitive_id,
        source_column=source_column,
        output_column=output_column,
        evidence_kind=" + ".join(evidence_kinds),
        evidence_location=" | ".join(evidence_locations),
        status=status,
        notes=notes,
    )


def _single_non_null(values) -> str | None:
    non_null = tuple(
        value for value in values if value is not None
    )
    return non_null[0] if non_null else None


def _missing_record(
    *,
    program_id: str,
    primitive_id: str,
) -> PassthroughBindingEvidence:
    return PassthroughBindingEvidence(
        program_id=program_id,
        primitive_id=primitive_id,
        source_column=None,
        output_column=None,
        evidence_kind="none",
        evidence_location="local-repository",
        status=MISSING,
        notes=("no reliable local binding evidence found",),
    )


def _conflict_record(
    *,
    program_id: str,
    primitive_id: str,
    records: tuple[PassthroughBindingEvidence, ...],
) -> PassthroughBindingEvidence:
    locations = tuple(
        sorted({
            record.evidence_location
            for record in records
        })
    )
    notes = tuple(
        f"{record.status}:{record.source_column}->{record.output_column}"
        for record in sorted(
            records,
            key=lambda item: (
                item.status,
                item.source_column or "",
                item.output_column or "",
                item.evidence_location,
            ),
        )
    )

    return PassthroughBindingEvidence(
        program_id=program_id,
        primitive_id=primitive_id,
        source_column=None,
        output_column=None,
        evidence_kind="conflict",
        evidence_location=" | ".join(locations),
        status=CONFLICTING,
        notes=notes,
    )
