from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    PrimitiveMaterializationStep,
)


PLAN_STEP_FIELDS = (
    "primitive_id",
    "operation",
    "lowering_mode",
    "pairwise_role",
    "source_table",
    "source_group_key",
    "source_left_key",
    "source_right_key",
    "source_event_time_col",
    "target_key",
    "target_left_key",
    "target_right_key",
    "target_time_col",
    "related_col",
    "window_days",
    "cutoff_operator",
    "output_columns",
    "materializable",
    "temporally_safe",
    "requires_external_provider",
    "errors",
    "warnings",
)

BINDING_FIELDS = (
    "program_id",
    "primitive_id",
    "operation",
    "lowering_mode",
    "pairwise_role",
    "output_column",
    "source_table",
    "source_group_key",
    "source_left_key",
    "source_right_key",
    "source_event_time_col",
    "target_key",
    "target_left_key",
    "target_right_key",
    "target_time_col",
    "related_col",
    "window_days",
    "cutoff_operator",
)

TEMPORAL_AUDIT_FIELDS = (
    "program_id",
    "primitive_id",
    "lowering_mode",
    "pairwise_role",
    "logical_temporal_predicate",
    "required_cutoff_operator",
    "configured_cutoff_operator",
    "source_table",
    "source_event_time_col",
    "target_time_col",
    "materializable",
    "temporally_safe",
    "requires_external_provider",
    "error_count",
    "warning_count",
    "errors",
    "warnings",
)

PROVENANCE_METADATA_FIELDS = (
    "dataset",
    "task",
    "compiler_version",
    "git_commit",
    "created_at_utc",
    "source",
)


def materialization_plan_to_dict(
    plan: CandidateMaterializationPlan,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    lowering_counts = Counter(
        step.lowering_mode.value for step in plan.steps
    )
    result: dict[str, object] = {
        "program_id": plan.program_id,
        "materializable": plan.materializable,
        "temporally_safe": plan.temporally_safe,
        "requires_external_provider": (
            plan.requires_external_provider
        ),
        "step_count": len(plan.steps),
        "lowering_mode_counts": {
            mode.value: lowering_counts.get(mode.value, 0)
            for mode in LoweringMode
        },
        "error_count": sum(
            len(step.errors) for step in plan.steps
        ),
        "warning_count": sum(
            len(step.warnings) for step in plan.steps
        ),
        "steps": [_step_to_dict(step) for step in plan.steps],
    }

    if metadata is not None:
        result["metadata"] = dict(metadata)

    return result


def primitive_bindings_to_records(
    plan: CandidateMaterializationPlan,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for step in plan.steps:
        for output_column in step.output_columns:
            records.append({
                "program_id": step.program_id,
                "primitive_id": step.primitive_id,
                "operation": step.operation,
                "lowering_mode": step.lowering_mode.value,
                "pairwise_role": step.pairwise_role,
                "output_column": output_column,
                "source_table": step.source_table,
                "source_group_key": step.source_group_key,
                "source_left_key": step.source_left_key,
                "source_right_key": step.source_right_key,
                "source_event_time_col": (
                    step.source_event_time_col
                ),
                "target_key": step.target_key,
                "target_left_key": step.target_left_key,
                "target_right_key": step.target_right_key,
                "target_time_col": step.target_time_col,
                "related_col": step.related_col,
                "window_days": step.window_days,
                "cutoff_operator": step.cutoff_operator,
            })

    return records


def temporal_audit_to_records(
    plan: CandidateMaterializationPlan,
    *,
    metadata: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for row in plan.audit_rows:
        record: dict[str, object] = {
            "program_id": row.program_id,
            "primitive_id": row.primitive_id,
            "lowering_mode": row.lowering_mode.value,
            "pairwise_role": row.pairwise_role,
            "logical_temporal_predicate": (
                row.logical_temporal_predicate
            ),
            "required_cutoff_operator": (
                row.required_cutoff_operator
            ),
            "configured_cutoff_operator": (
                row.configured_cutoff_operator
            ),
            "source_table": row.source_table,
            "source_event_time_col": (
                row.source_event_time_col
            ),
            "target_time_col": _target_time_for_row(plan, row),
            "materializable": row.materializable,
            "temporally_safe": row.temporally_safe,
            "requires_external_provider": (
                row.requires_external_provider
            ),
            "error_count": len(row.errors),
            "warning_count": len(row.warnings),
            "errors": _messages_to_json(row.errors),
            "warnings": _messages_to_json(row.warnings),
        }

        if metadata is not None:
            record.update(_metadata_record(metadata))

        records.append(record)

    return records


def write_materialization_plan_json(
    plan: CandidateMaterializationPlan,
    output_path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    payload = materialization_plan_to_dict(
        plan,
        metadata=metadata,
    )
    _write_json(payload, output_path, overwrite=overwrite)


def write_primitive_bindings_json(
    plan: CandidateMaterializationPlan,
    output_path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    records = primitive_bindings_to_records(plan)
    payload: object = records

    if metadata is not None:
        payload = {
            "metadata": dict(metadata),
            "records": records,
        }

    _write_json(payload, output_path, overwrite=overwrite)


def write_temporal_safety_audit_csv(
    plan: CandidateMaterializationPlan,
    output_path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> None:
    csv_text = temporal_audit_to_csv_text(
        plan,
        metadata=metadata,
    )
    _atomic_write_text(
        output_path,
        csv_text,
        overwrite=overwrite,
    )


def temporal_audit_to_csv_text(
    plan: CandidateMaterializationPlan,
    *,
    metadata: Mapping[str, object] | None = None,
) -> str:
    buffer = io.StringIO(newline="")
    fieldnames = list(TEMPORAL_AUDIT_FIELDS)

    if metadata is not None:
        fieldnames.extend(PROVENANCE_METADATA_FIELDS)

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(
        temporal_audit_to_records(
            plan,
            metadata=metadata,
        )
    )
    return buffer.getvalue()


def _step_to_dict(
    step: PrimitiveMaterializationStep,
) -> dict[str, object]:
    return {
        "primitive_id": step.primitive_id,
        "operation": step.operation,
        "lowering_mode": step.lowering_mode.value,
        "pairwise_role": step.pairwise_role,
        "source_table": step.source_table,
        "source_group_key": step.source_group_key,
        "source_left_key": step.source_left_key,
        "source_right_key": step.source_right_key,
        "source_event_time_col": step.source_event_time_col,
        "target_key": step.target_key,
        "target_left_key": step.target_left_key,
        "target_right_key": step.target_right_key,
        "target_time_col": step.target_time_col,
        "related_col": step.related_col,
        "window_days": step.window_days,
        "cutoff_operator": step.cutoff_operator,
        "output_columns": list(step.output_columns),
        "materializable": step.materializable,
        "temporally_safe": step.temporally_safe,
        "requires_external_provider": (
            step.requires_external_provider
        ),
        "errors": list(step.errors),
        "warnings": list(step.warnings),
    }


def _messages_to_json(messages: Sequence[str]) -> str:
    return json.dumps(
        list(messages),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _metadata_record(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        field: metadata.get(field)
        for field in PROVENANCE_METADATA_FIELDS
    }


def _target_time_for_row(
    plan: CandidateMaterializationPlan,
    row,
) -> str | None:
    for step in plan.steps:
        if step.primitive_id == row.primitive_id:
            return step.target_time_col
    return None


def _write_json(
    payload: object,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> None:
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_write_text(
        output_path,
        text,
        overwrite=overwrite,
    )


def _atomic_write_text(
    output_path: str | Path,
    text: str,
    *,
    overwrite: bool,
) -> None:
    path = Path(output_path)

    if path.exists() and not overwrite:
        raise FileExistsError(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink()
            except FileNotFoundError:
                pass
