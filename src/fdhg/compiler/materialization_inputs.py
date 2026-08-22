from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_safety import ExplicitLoweringEvidence
from .config import load_yaml
from .ir import TaskSpec
from .materializer import CandidateMaterializationPlan, LoweringMode


VALID_TRAIN_SPLITS = frozenset({"train", "training"})
VALID_VALIDATION_SPLITS = frozenset({"validation", "val"})
REJECTED_SPLITS = frozenset({
    "test",
    "heldout_test",
    "held_out_test",
    "final",
    "paper_final",
})


@dataclass(frozen=True)
class PreparedArtifactSpec:
    dataset: str
    task: str
    split: str
    role: str
    table_name: str
    path: Path
    columns: tuple[str, ...]
    evidence_location: str


@dataclass(frozen=True)
class ResolvedMaterializationInputs:
    dataset: str
    task: str
    train_target: PreparedArtifactSpec
    validation_target: PreparedArtifactSpec
    source_artifacts: tuple[PreparedArtifactSpec, ...]
    target_entity_columns: tuple[str, ...]
    label_column: str
    prediction_time_column: str
    explicit_lowering_evidence: tuple[ExplicitLoweringEvidence, ...]
    evidence_locations: tuple[str, ...]

    def source_by_table(self) -> dict[str, PreparedArtifactSpec]:
        return {
            artifact.table_name: artifact
            for artifact in self.source_artifacts
        }

    def evidence_for_program(
        self,
        program_id: str,
    ) -> tuple[ExplicitLoweringEvidence, ...]:
        return tuple(
            record
            for record in self.explicit_lowering_evidence
            if record.program_id == program_id
        )


@dataclass(frozen=True)
class MaterializationInputReport:
    dataset: str
    task: str
    resolved: bool
    inputs: ResolvedMaterializationInputs | None
    blockers: tuple[str, ...]
    evidence_locations: tuple[str, ...]


def resolve_materialization_inputs(
    task_spec: TaskSpec,
    *,
    reproduction_config: Path,
    semantics_config: Path | None = None,
) -> MaterializationInputReport:
    """Resolve task-scoped prepared artifacts without reading full data."""

    key = f"{task_spec.dataset}/{task_spec.task}"
    evidence_locations: list[str] = []
    blockers: list[str] = []
    raw_task = _task_config_entry(reproduction_config, key)
    semantics = (
        load_yaml(semantics_config).get(key, {})
        if semantics_config is not None and semantics_config.exists()
        else {}
    )
    prepared = _first_mapping(
        raw_task.get("prepared_artifacts"),
        semantics.get("prepared_artifacts"),
    )
    if prepared is None:
        return MaterializationInputReport(
            dataset=task_spec.dataset,
            task=task_spec.task,
            resolved=False,
            inputs=None,
            blockers=("missing_prepared_artifacts_config",),
            evidence_locations=(),
        )

    evidence_locations.append(
        f"config:{reproduction_config}:{key}:prepared_artifacts"
    )
    try:
        if str(prepared.get("provider", "")) == "onboarding":
            inputs = _inputs_from_onboarding_manifest(
                task_spec=task_spec,
                prepared=prepared,
                config_path=reproduction_config,
                config_key=key,
            )
            return MaterializationInputReport(
                dataset=task_spec.dataset,
                task=task_spec.task,
                resolved=True,
                inputs=inputs,
                blockers=(),
                evidence_locations=inputs.evidence_locations,
            )
        train_target = _artifact_from_config(
            task_spec=task_spec,
            raw=prepared.get("train_target"),
            expected_role="target",
            expected_split_group="train",
            config_path=reproduction_config,
            config_key=key,
            config_name="train_target",
        )
        validation_target = _artifact_from_config(
            task_spec=task_spec,
            raw=prepared.get("validation_target"),
            expected_role="target",
            expected_split_group="validation",
            config_path=reproduction_config,
            config_key=key,
            config_name="validation_target",
        )
        source_artifacts = _source_artifacts_from_config(
            task_spec=task_spec,
            raw=prepared.get("source_tables", {}),
            config_path=reproduction_config,
            config_key=key,
        )
        evidence = _explicit_evidence_from_config(
            task_spec=task_spec,
            raw=prepared.get("lowering_evidence", ()),
            config_path=reproduction_config,
            config_key=key,
        )
        _validate_target_role_columns(
            task_spec=task_spec,
            train_target=train_target,
            validation_target=validation_target,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        blockers.append(str(exc))
        return MaterializationInputReport(
            dataset=task_spec.dataset,
            task=task_spec.task,
            resolved=False,
            inputs=None,
            blockers=tuple(blockers),
            evidence_locations=tuple(evidence_locations),
        )

    inputs = ResolvedMaterializationInputs(
        dataset=task_spec.dataset,
        task=task_spec.task,
        train_target=train_target,
        validation_target=validation_target,
        source_artifacts=tuple(sorted(
            source_artifacts,
            key=lambda item: item.table_name,
        )),
        target_entity_columns=_target_entity_columns(task_spec),
        label_column=task_spec.label_col,
        prediction_time_column=task_spec.target_time_col,
        explicit_lowering_evidence=evidence,
        evidence_locations=tuple(sorted({
            *evidence_locations,
            train_target.evidence_location,
            validation_target.evidence_location,
            *(item.evidence_location for item in source_artifacts),
            *(item.evidence_location for item in evidence),
        })),
    )
    return MaterializationInputReport(
        dataset=task_spec.dataset,
        task=task_spec.task,
        resolved=True,
        inputs=inputs,
        blockers=(),
        evidence_locations=inputs.evidence_locations,
    )


def _inputs_from_onboarding_manifest(
    *,
    task_spec: TaskSpec,
    prepared: Mapping[str, Any],
    config_path: Path,
    config_key: str,
) -> ResolvedMaterializationInputs:
    raw_manifest = prepared.get("onboarding_manifest", {})
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("onboarding_manifest must be a mapping")
    manifest_path = _resolve_path(config_path, raw_manifest.get("path"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset") != task_spec.dataset
        or manifest.get("task") != task_spec.task
        or manifest.get("status") != "completed"
    ):
        raise ValueError("onboarding manifest dataset/task/status mismatch")
    base = manifest_path.parent
    file_hashes = manifest.get("file_hashes", {})
    if not isinstance(file_hashes, Mapping):
        raise ValueError("onboarding manifest missing file_hashes")
    for name, expected in sorted(file_hashes.items()):
        actual_path = base / str(name)
        if not actual_path.exists():
            raise FileNotFoundError(actual_path)
        if _file_sha256(actual_path) != str(expected):
            raise ValueError(f"onboarding file hash mismatch: {name}")
    for audit in (
        "temporal_safety_audit.csv",
        "leakage_safety_audit.csv",
    ):
        _verify_audit_passed(base / audit)
    train_target = _artifact_from_manifest(
        task_spec=task_spec,
        raw=manifest.get("train_target"),
        base=base,
        expected_split_group="train",
        evidence_location=f"onboarding-manifest:{manifest_path}:train_target",
    )
    validation_target = _artifact_from_manifest(
        task_spec=task_spec,
        raw=manifest.get("validation_target"),
        base=base,
        expected_split_group="validation",
        evidence_location=(
            f"onboarding-manifest:{manifest_path}:validation_target"
        ),
    )
    _validate_target_role_columns(
        task_spec=task_spec,
        train_target=train_target,
        validation_target=validation_target,
    )
    evidence = _explicit_evidence_from_config(
        task_spec=task_spec,
        raw=manifest.get("lowering_evidence", ()),
        config_path=manifest_path,
        config_key=config_key,
    )
    return ResolvedMaterializationInputs(
        dataset=task_spec.dataset,
        task=task_spec.task,
        train_target=train_target,
        validation_target=validation_target,
        source_artifacts=(),
        target_entity_columns=_target_entity_columns(task_spec),
        label_column=task_spec.label_col,
        prediction_time_column=task_spec.target_time_col,
        explicit_lowering_evidence=evidence,
        evidence_locations=tuple(sorted({
            f"config:{config_path}:{config_key}:prepared_artifacts",
            f"onboarding-manifest:{manifest_path}",
            train_target.evidence_location,
            validation_target.evidence_location,
            *(item.evidence_location for item in evidence),
        })),
    )


def _artifact_from_manifest(
    *,
    task_spec: TaskSpec,
    raw,
    base: Path,
    expected_split_group: str,
    evidence_location: str,
) -> PreparedArtifactSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("onboarding manifest artifact row is missing")
    if raw.get("dataset") != task_spec.dataset or raw.get("task") != task_spec.task:
        raise ValueError("onboarding artifact dataset/task mismatch")
    split = _normalize_split(str(raw.get("split", "")))
    _validate_split(split, expected_split_group, "onboarding_manifest")
    role = str(raw.get("role", ""))
    if role != "target":
        raise ValueError("onboarding artifact role mismatch: expected target")
    path = base / str(raw["path"])
    columns = _read_schema_columns(path)
    return PreparedArtifactSpec(
        dataset=task_spec.dataset,
        task=task_spec.task,
        split=split,
        role=role,
        table_name=str(raw.get("table", "target_with_dfs_agg")),
        path=path,
        columns=columns,
        evidence_location=evidence_location,
    )


def _verify_audit_passed(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(str(row.get("passed", "")).lower() != "true" for row in rows):
        raise ValueError(f"onboarding safety audit failed: {path.name}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows_for_materialization_plan(
    *,
    inputs: ResolvedMaterializationInputs,
    plan: CandidateMaterializationPlan,
    evidence: Sequence[ExplicitLoweringEvidence] = (),
) -> tuple[
    dict[str, tuple[Mapping[str, object], ...]],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    source_columns = _required_source_columns(plan)
    target_columns = _required_target_columns(
        inputs=inputs,
        plan=plan,
        evidence=evidence,
    )
    train_rows = _read_parquet_records(
        inputs.train_target.path,
        columns=target_columns,
        artifact=inputs.train_target,
    )
    validation_rows = _read_parquet_records(
        inputs.validation_target.path,
        columns=target_columns,
        artifact=inputs.validation_target,
    )
    source_by_table = inputs.source_by_table()
    source_rows: dict[str, tuple[Mapping[str, object], ...]] = {}
    for table_name, columns in sorted(source_columns.items()):
        if table_name not in source_by_table:
            raise FileNotFoundError(
                f"missing source artifact for table {table_name!r}"
            )
        artifact = source_by_table[table_name]
        source_rows[table_name] = _read_parquet_records(
            artifact.path,
            columns=tuple(sorted(columns)),
            artifact=artifact,
        )
    return source_rows, train_rows, validation_rows


def _task_config_entry(config_path: Path, key: str) -> Mapping[str, Any]:
    config = load_yaml(config_path)
    tasks = config.get("tasks", config)
    if not isinstance(tasks, Mapping) or key not in tasks:
        raise KeyError(f"task {key!r} missing from {config_path}")
    raw = tasks[key]
    if not isinstance(raw, Mapping):
        raise ValueError(f"task {key!r} must be a mapping")
    return raw


def _first_mapping(*values) -> Mapping[str, Any] | None:
    for value in values:
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError("prepared_artifacts must be a mapping")
        return value
    return None


def _artifact_from_config(
    *,
    task_spec: TaskSpec,
    raw,
    expected_role: str,
    expected_split_group: str,
    config_path: Path,
    config_key: str,
    config_name: str,
) -> PreparedArtifactSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{config_name} artifact config is missing")
    dataset = str(raw.get("dataset", ""))
    task = str(raw.get("task", ""))
    if dataset != task_spec.dataset or task != task_spec.task:
        raise ValueError(
            f"{config_name} dataset/task mismatch: {dataset}/{task}"
        )
    split = _normalize_split(str(raw.get("split", "")))
    _validate_split(split, expected_split_group, config_name)
    role = str(raw.get("role", ""))
    if role != expected_role:
        raise ValueError(
            f"{config_name} role mismatch: expected {expected_role}"
        )
    table_name = str(raw.get("table", raw.get("table_name", "")))
    if not table_name:
        raise ValueError(f"{config_name} table is required")
    path = _resolve_path(config_path, raw.get("path"))
    columns = _read_schema_columns(path)
    return PreparedArtifactSpec(
        dataset=dataset,
        task=task,
        split=split,
        role=role,
        table_name=table_name,
        path=path,
        columns=columns,
        evidence_location=(
            f"config:{config_path}:{config_key}:"
            f"prepared_artifacts.{config_name}"
        ),
    )


def _source_artifacts_from_config(
    *,
    task_spec: TaskSpec,
    raw,
    config_path: Path,
    config_key: str,
) -> tuple[PreparedArtifactSpec, ...]:
    if not isinstance(raw, Mapping):
        raise ValueError("source_tables must be a mapping")
    artifacts = []
    for table_name, item in sorted(raw.items()):
        artifact = _artifact_from_config(
            task_spec=task_spec,
            raw={**item, "table": item.get("table", table_name)},
            expected_role="source",
            expected_split_group="train",
            config_path=config_path,
            config_key=config_key,
            config_name=f"source_tables.{table_name}",
        )
        if artifact.table_name != str(table_name):
            raise ValueError(
                f"source artifact key/table mismatch for {table_name}"
            )
        artifacts.append(artifact)
    return tuple(artifacts)


def _explicit_evidence_from_config(
    *,
    task_spec: TaskSpec,
    raw,
    config_path: Path,
    config_key: str,
) -> tuple[ExplicitLoweringEvidence, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("lowering_evidence must be a list")
    records = []
    seen = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError("lowering_evidence rows must be mappings")
        if (
            item.get("dataset") != task_spec.dataset
            or item.get("task") != task_spec.task
        ):
            raise ValueError(
                "lowering_evidence dataset/task mismatch"
            )
        record = ExplicitLoweringEvidence(
            dataset=str(item["dataset"]),
            task=str(item["task"]),
            program_id=str(item["program_id"]),
            primitive_id=str(item["primitive_id"]),
            source_table=(
                None
                if item.get("source_table") is None
                else str(item.get("source_table"))
            ),
            source_column=(
                None
                if item.get("source_column") is None
                else str(item.get("source_column"))
            ),
            output_column=(
                None
                if item.get("output_column") is None
                else str(item.get("output_column"))
            ),
            status=str(item["status"]),
            evidence_location=(
                f"config:{config_path}:{config_key}:"
                f"prepared_artifacts.lowering_evidence[{index}]"
            ),
            notes=tuple(str(note) for note in item.get("notes", ())),
        )
        key = (record.program_id, record.primitive_id)
        if key in seen:
            raise ValueError(
                "duplicate lowering_evidence for "
                f"{record.program_id}/{record.primitive_id}"
            )
        seen.add(key)
        records.append(record)
    return tuple(sorted(
        records,
        key=lambda item: (item.program_id, item.primitive_id),
    ))


def _validate_target_role_columns(
    *,
    task_spec: TaskSpec,
    train_target: PreparedArtifactSpec,
    validation_target: PreparedArtifactSpec,
) -> None:
    required = set(_target_entity_columns(task_spec))
    required.add(task_spec.label_col)
    required.add(task_spec.target_time_col)
    for artifact in (train_target, validation_target):
        missing = sorted(required - set(artifact.columns))
        if missing:
            raise ValueError(
                f"{artifact.split} target artifact missing columns: "
                + ", ".join(missing)
            )
    if set(train_target.columns) != set(validation_target.columns):
        missing_train = sorted(
            set(validation_target.columns) - set(train_target.columns)
        )
        missing_validation = sorted(
            set(train_target.columns) - set(validation_target.columns)
        )
        raise ValueError(
            "train/validation target schema mismatch: "
            f"missing_train={missing_train} "
            f"missing_validation={missing_validation}"
        )


def _target_entity_columns(task_spec: TaskSpec) -> tuple[str, ...]:
    columns = [task_spec.entity_key]
    if task_spec.pairwise is not None:
        columns.append(task_spec.pairwise.target_right_key)
    return tuple(dict.fromkeys(columns))


def _required_source_columns(
    plan: CandidateMaterializationPlan,
) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for step in plan.steps:
        if step.lowering_mode != LoweringMode.GENERATE:
            continue
        if step.source_table is None:
            continue
        columns = required.setdefault(step.source_table, set())
        for column in (
            step.source_group_key,
            step.source_left_key,
            step.source_right_key,
            step.source_event_time_col,
            step.related_col,
        ):
            if column:
                columns.add(column)
    return required


def _required_target_columns(
    *,
    inputs: ResolvedMaterializationInputs,
    plan: CandidateMaterializationPlan,
    evidence: Sequence[ExplicitLoweringEvidence],
) -> tuple[str, ...]:
    columns = set(inputs.target_entity_columns)
    columns.add(inputs.label_column)
    columns.add(inputs.prediction_time_column)
    for step in plan.steps:
        for column in (
            step.target_key,
            step.target_left_key,
            step.target_right_key,
            step.target_time_col,
        ):
            if column:
                columns.add(column)
    for record in evidence:
        if record.source_column:
            columns.add(record.source_column)
    return tuple(sorted(columns))


def _read_parquet_records(
    path: Path,
    *,
    columns: Sequence[str],
    artifact: PreparedArtifactSpec,
) -> tuple[Mapping[str, object], ...]:
    missing = sorted(set(columns) - set(artifact.columns))
    if missing:
        raise ValueError(
            f"{artifact.role} artifact {artifact.path} missing "
            "required columns: "
            + ", ".join(missing)
        )
    import pandas as pd

    frame = pd.read_parquet(path, columns=list(columns))
    return tuple(frame.to_dict("records"))


def _read_schema_columns(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    columns = tuple(str(name) for name in schema.names)
    if len(columns) != len(set(columns)):
        raise ValueError(f"duplicate physical columns in {path}")
    return columns


def _resolve_path(config_path: Path, raw_path) -> Path:
    if raw_path is None:
        raise ValueError("artifact path is required")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _normalize_split(split: str) -> str:
    return split.strip().lower().replace("-", "_")


def _validate_split(
    split: str,
    expected_group: str,
    config_name: str,
) -> None:
    if not split:
        raise ValueError(f"{config_name} split is required")
    if split in REJECTED_SPLITS:
        raise ValueError(f"{config_name} test/final split rejected")
    valid = (
        VALID_TRAIN_SPLITS
        if expected_group == "train"
        else VALID_VALIDATION_SPLITS
    )
    if split not in valid:
        raise ValueError(
            f"{config_name} split {split!r} is not {expected_group}"
        )
