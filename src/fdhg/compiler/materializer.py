from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

from .ir import (
    CompiledTask,
    PairwiseHistorySpec,
    Primitive,
    PrimitiveFamily,
)
from .programs import CandidateProgram, build_configured_candidates


STRICT_PRIOR_EVENT_OPERATOR = "<"

PAIRWISE_GENERATED_OPERATIONS = {
    "window_count",
    "past_unique_neighbors",
    "days_since_last",
    "prior_pair_count",
    "pair_days_since_last",
}


class LoweringMode(str, Enum):
    GENERATE = "generate"
    PASSTHROUGH = "passthrough"
    EXTERNAL = "external"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PhysicalHistoryBinding:
    role: str
    source_table: str | None
    source_group_key: str | None
    source_left_key: str | None
    source_right_key: str | None
    source_event_time_col: str | None
    target_key: str | None
    target_left_key: str | None
    target_right_key: str | None
    target_time_col: str
    related_col: str | None = None


@dataclass(frozen=True)
class PrimitiveMaterializationStep:
    program_id: str
    primitive_id: str
    operation: str
    lowering_mode: LoweringMode
    pairwise_role: str | None
    source_table: str | None
    source_group_key: str | None
    source_left_key: str | None
    source_right_key: str | None
    source_event_time_col: str | None
    target_key: str | None
    target_left_key: str | None
    target_right_key: str | None
    target_time_col: str
    related_col: str | None
    window_days: int | None
    cutoff_operator: str
    output_columns: tuple[str, ...]
    materializable: bool
    temporally_safe: bool
    requires_external_provider: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterializationAuditRow:
    program_id: str
    primitive_id: str
    lowering_mode: LoweringMode
    pairwise_role: str | None
    source_table: str | None
    source_event_time_col: str | None
    logical_temporal_predicate: str | None
    required_cutoff_operator: str
    configured_cutoff_operator: str | None
    temporally_safe: bool
    materializable: bool
    requires_external_provider: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CandidateMaterializationPlan:
    program_id: str
    steps: tuple[PrimitiveMaterializationStep, ...]
    audit_rows: tuple[MaterializationAuditRow, ...]
    materializable: bool
    temporally_safe: bool
    requires_external_provider: bool


@dataclass(frozen=True)
class CandidateMaterializationRequest:
    """Explicit, non-experimental candidate materialization request.

    The orchestration layer consumes already-loaded train/validation target
    rows and source rows. It does not fetch RelBench data, run training, or
    launch an experiment sweep.
    """

    compiled: CompiledTask
    program: CandidateProgram
    output_dir: Path
    source_rows_by_table: Mapping[str, Iterable[Mapping[str, object]]]
    train_target_rows: Iterable[Mapping[str, object]]
    validation_target_rows: Iterable[Mapping[str, object]]
    write: bool = False
    overwrite: bool = False
    allow_reuse: bool = True
    validation_split: str = "validation"
    passthrough_provenance_report: object | None = None
    explicit_lowering_evidence: Sequence[object] = ()
    candidate_id_columns: Sequence[str] = ()
    surrogate_key_columns: Sequence[str] = (
        "__row_id",
        "primary_key",
        "__fdhg_row_id",
    )
    target_aggregate_columns: Sequence[str] = ()
    cross_fitted_target_aggregates: bool = False


@dataclass(frozen=True)
class CandidateMaterializationResult:
    dataset: str
    task: str
    program_id: str
    output_dir: Path
    plan: CandidateMaterializationPlan
    dry_run: bool
    reused: bool
    materializable: bool
    temporally_safe: bool
    leakage_safe: bool
    provenance_complete: bool
    selector_ready: bool
    train_artifact: Path | None
    validation_artifact: Path | None
    manifest_path: Path | None
    bindings_path: Path | None
    audit_paths: tuple[Path, ...]
    feature_columns: tuple[str, ...]
    train_row_count: int
    validation_row_count: int
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskCandidateMaterializationRequest:
    dataset: str
    task: str
    output_root: Path
    reproduction_config: Path
    semantics_config: Path | None = None
    program_ids: Sequence[str] = ()
    exclude_program_ids: Sequence[str] = ()
    baseline_only: bool = False
    write: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class TaskCandidateMaterializationOutcome:
    program_id: str
    output_dir: Path
    primitive_count: int
    lowering_feasible: bool
    status: str
    blockers: tuple[str, ...]
    result: CandidateMaterializationResult | None = None


@dataclass(frozen=True)
class TaskCandidateMaterializationReport:
    dataset: str
    task: str
    output_root: Path
    dry_run: bool
    input_resolved: bool
    input_blockers: tuple[str, ...]
    evidence_locations: tuple[str, ...]
    outcomes: tuple[TaskCandidateMaterializationOutcome, ...]

    @property
    def published_count(self) -> int:
        return sum(
            outcome.status == "published" for outcome in self.outcomes
        )

    @property
    def reused_count(self) -> int:
        return sum(
            outcome.status == "reused" for outcome in self.outcomes
        )

    @property
    def blocked_count(self) -> int:
        return sum(
            outcome.status == "blocked" for outcome in self.outcomes
        )

    @property
    def failed_count(self) -> int:
        return sum(
            outcome.status == "failed" for outcome in self.outcomes
        )


def plan_candidate_materialization(
    compiled: CompiledTask,
    program: CandidateProgram,
    *,
    available_source_tables: Collection[str] | None = None,
    cutoff_operator: str = STRICT_PRIOR_EVENT_OPERATOR,
    reference_bindings: (
        Mapping[str, PhysicalHistoryBinding] | None
    ) = None,
) -> CandidateMaterializationPlan:
    """Build a read-only materialization plan for logical primitives."""
    primitive_by_id = _index_primitives_by_id(
        compiled.candidate_primitives
    )

    steps: list[PrimitiveMaterializationStep] = []
    audit_rows: list[MaterializationAuditRow] = []

    for primitive_id in program.primitive_ids:
        primitive = primitive_by_id.get(primitive_id)

        if primitive is None:
            step = _missing_primitive_step(
                program_id=program.program_id,
                primitive_id=primitive_id,
                target_time_col=(
                    compiled.task_spec.target_time_col
                ),
                cutoff_operator=cutoff_operator,
            )
        else:
            step = _step_for_primitive(
                compiled=compiled,
                program_id=program.program_id,
                primitive=primitive,
                available_source_tables=(
                    available_source_tables
                ),
                cutoff_operator=cutoff_operator,
                reference_bindings=reference_bindings,
            )

        steps.append(step)
        audit_rows.append(
            _audit_row(
                program_id=program.program_id,
                primitive=primitive,
                step=step,
                cutoff_operator=cutoff_operator,
            )
        )

    return CandidateMaterializationPlan(
        program_id=program.program_id,
        steps=tuple(steps),
        audit_rows=tuple(audit_rows),
        materializable=all(
            step.materializable for step in steps
        ),
        temporally_safe=all(
            step.temporally_safe for step in steps
        ),
        requires_external_provider=any(
            step.requires_external_provider
            for step in steps
        ),
    )


def materialize_candidate_program(
    request: CandidateMaterializationRequest,
) -> CandidateMaterializationResult:
    """Materialize one candidate and emit selector safety artifacts.

    When ``request.write`` is false, this function performs pure planning and
    static lowering checks only. When writing, artifacts are staged in a private
    sibling directory and atomically published only after train/validation
    artifacts, manifests, bindings, and all candidate-local safety audits are
    complete and passing.
    """

    dataset = request.compiled.task_spec.dataset
    task = request.compiled.task_spec.task
    _reject_test_split(request.validation_split)

    source_rows = _tuple_source_rows(request.source_rows_by_table)
    train_rows = tuple(request.train_target_rows)
    validation_rows = tuple(request.validation_target_rows)
    plan = plan_candidate_materialization(
        request.compiled,
        request.program,
        available_source_tables=set(source_rows),
    )
    feature_columns = _feature_columns_from_plan_and_evidence(
        plan,
        request.explicit_lowering_evidence,
        request.passthrough_provenance_report,
    )

    if not request.write:
        return CandidateMaterializationResult(
            dataset=dataset,
            task=task,
            program_id=request.program.program_id,
            output_dir=request.output_dir,
            plan=plan,
            dry_run=True,
            reused=False,
            materializable=plan.materializable,
            temporally_safe=plan.temporally_safe,
            leakage_safe=False,
            provenance_complete=False,
            selector_ready=False,
            train_artifact=None,
            validation_artifact=None,
            manifest_path=None,
            bindings_path=None,
            audit_paths=(),
            feature_columns=feature_columns,
            train_row_count=len(train_rows),
            validation_row_count=len(validation_rows),
            failure_reasons=("dry_run_no_artifacts_written",),
        )

    if request.output_dir.exists():
        if (
            request.allow_reuse
            and not request.overwrite
            and _candidate_dir_matches(
                request.output_dir,
                dataset=dataset,
                task=task,
                program_id=request.program.program_id,
            )
        ):
            return _reused_materialization_result(
                request=request,
                plan=plan,
                feature_columns=feature_columns,
                train_row_count=len(train_rows),
                validation_row_count=len(validation_rows),
            )
        if not request.overwrite:
            raise FileExistsError(request.output_dir)

    _validate_lowering_ready(
        dataset=dataset,
        task=task,
        plan=plan,
        feature_columns=feature_columns,
        label_col=request.compiled.task_spec.label_col,
        candidate_id_columns=request.candidate_id_columns,
        surrogate_key_columns=request.surrogate_key_columns,
        target_aggregate_columns=request.target_aggregate_columns,
        cross_fitted_target_aggregates=(
            request.cross_fitted_target_aggregates
        ),
        passthrough_report=request.passthrough_provenance_report,
        explicit_evidence=request.explicit_lowering_evidence,
    )

    return _materialize_candidate_program_write(
        request=request,
        plan=plan,
        source_rows=source_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        feature_columns=feature_columns,
    )


def _materialize_candidate_program_write(
    *,
    request: CandidateMaterializationRequest,
    plan: CandidateMaterializationPlan,
    source_rows: Mapping[str, tuple[Mapping[str, object], ...]],
    train_rows: tuple[Mapping[str, object], ...],
    validation_rows: tuple[Mapping[str, object], ...],
    feature_columns: tuple[str, ...],
) -> CandidateMaterializationResult:
    from .candidate_safety import (
        build_candidate_safety_audit_report,
        write_audit_csv,
    )
    from .in_memory_materializer import (
        materialize_generated_features_in_memory,
    )

    dataset = request.compiled.task_spec.dataset
    task = request.compiled.task_spec.task
    output_dir = request.output_dir
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f"_{output_dir.name}.",
            suffix=".tmp",
            dir=parent,
        )
    )

    try:
        generated_plan = _generated_only_plan(plan)
        train_generated = materialize_generated_features_in_memory(
            generated_plan,
            source_rows_by_table=source_rows,
            target_rows=train_rows,
        )
        validation_generated = materialize_generated_features_in_memory(
            generated_plan,
            source_rows_by_table=source_rows,
            target_rows=validation_rows,
        )
        train_out = _combine_output_rows(
            target_rows=train_rows,
            generated_rows=train_generated.batch_result.rows,
            plan=plan,
            explicit_evidence=request.explicit_lowering_evidence,
            passthrough_report=request.passthrough_provenance_report,
        )
        validation_out = _combine_output_rows(
            target_rows=validation_rows,
            generated_rows=validation_generated.batch_result.rows,
            plan=plan,
            explicit_evidence=request.explicit_lowering_evidence,
            passthrough_report=request.passthrough_provenance_report,
        )
        train_path = staging / "target_with_dfs_agg_train.parquet"
        val_path = staging / "target_with_dfs_agg_val.parquet"
        _write_parquet_rows(train_out, train_path)
        _write_parquet_rows(validation_out, val_path)
        _write_candidate_program_json(
            request.compiled,
            request.program,
            staging / "candidate_program.json",
        )
        bindings = _binding_records(
            dataset=dataset,
            task=task,
            plan=plan,
            explicit_evidence=request.explicit_lowering_evidence,
            passthrough_report=request.passthrough_provenance_report,
        )
        _write_candidate_manifest_csv(
            dataset=dataset,
            task=task,
            plan=plan,
            bindings=bindings,
            path=staging / "candidate_manifest.csv",
        )
        _write_json_deterministic(
            {
                "dataset": dataset,
                "task": task,
                "program_id": plan.program_id,
                "records": bindings,
            },
            staging / "primitive_column_bindings.json",
        )
        feature_columns = _feature_columns_from_bindings(bindings)
        report = build_candidate_safety_audit_report(
            dataset=dataset,
            task=task,
            plan=plan,
            feature_columns=feature_columns,
            label_col=request.compiled.task_spec.label_col,
            candidate_id_columns=request.candidate_id_columns,
            surrogate_key_columns=request.surrogate_key_columns,
            target_aggregate_columns=request.target_aggregate_columns,
            cross_fitted_target_aggregates=(
                request.cross_fitted_target_aggregates
            ),
            passthrough_report=request.passthrough_provenance_report,
            explicit_evidence=request.explicit_lowering_evidence,
        )
        audit_paths = _write_safety_audits(
            staging=staging,
            report=report,
            write_audit_csv=write_audit_csv,
        )
        failure_reasons = _audit_failure_reasons(report)
        materializable = _artifacts_schema_valid(
            train_path=train_path,
            validation_path=val_path,
            feature_columns=feature_columns,
        )
        if not materializable:
            failure_reasons = (
                *failure_reasons,
                "materialized_artifact_schema_invalid",
            )
        selector_ready = (
            materializable
            and report.temporal.passed
            and report.leakage.passed
            and report.provenance.passed
        )
        manifest = _materialization_manifest(
            dataset=dataset,
            task=task,
            plan=plan,
            train_path=train_path,
            validation_path=val_path,
            train_rows=train_out,
            validation_rows=validation_out,
            feature_columns=feature_columns,
            status="success" if selector_ready else "failed",
            failure_reasons=failure_reasons,
            bindings=bindings,
            audit_paths=audit_paths,
        )
        _write_json_deterministic(
            manifest,
            staging / "materialization_manifest.json",
        )
        _validate_complete_staging(
            staging,
            require_selector_ready=selector_ready,
        )
        if not selector_ready:
            raise ValueError(
                "candidate safety audits did not pass: "
                + ", ".join(failure_reasons)
            )
        _publish_staging(
            staging,
            output_dir,
            overwrite=request.overwrite,
        )
        staging = None
        return CandidateMaterializationResult(
            dataset=dataset,
            task=task,
            program_id=plan.program_id,
            output_dir=output_dir,
            plan=plan,
            dry_run=False,
            reused=False,
            materializable=True,
            temporally_safe=True,
            leakage_safe=True,
            provenance_complete=True,
            selector_ready=True,
            train_artifact=(
                output_dir / "target_with_dfs_agg_train.parquet"
            ),
            validation_artifact=(
                output_dir / "target_with_dfs_agg_val.parquet"
            ),
            manifest_path=output_dir / "materialization_manifest.json",
            bindings_path=output_dir / "primitive_column_bindings.json",
            audit_paths=tuple(
                output_dir / path.name for path in audit_paths
            ),
            feature_columns=feature_columns,
            train_row_count=len(train_out),
            validation_row_count=len(validation_out),
        )
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def materialize_task_candidates(
    request: TaskCandidateMaterializationRequest,
) -> TaskCandidateMaterializationReport:
    """Resolve prepared inputs and materialize configured/default candidates.

    This is task-aware orchestration only. It compiles candidate programs,
    loads prepared parquet inputs, and delegates per-candidate publication to
    ``materialize_candidate_program``. It never runs model training,
    evaluation, or candidate sweeps.
    """

    from .config import load_task_spec
    from .materialization_inputs import (
        load_rows_for_materialization_plan,
        resolve_materialization_inputs,
    )
    from .planner import build_candidate_program

    task_spec = load_task_spec(
        dataset=request.dataset,
        task=request.task,
        reproduction_config=request.reproduction_config,
        semantics_config=request.semantics_config,
    )
    compiled = build_candidate_program(task_spec)
    programs = _filter_task_programs(
        build_configured_candidates(
            compiled,
            reproduction_config=request.reproduction_config,
            semantics_config=request.semantics_config,
        ),
        program_ids=request.program_ids,
        exclude_program_ids=request.exclude_program_ids,
        baseline_only=request.baseline_only,
    )
    input_report = resolve_materialization_inputs(
        task_spec,
        reproduction_config=request.reproduction_config,
        semantics_config=request.semantics_config,
    )
    output_root = request.output_root / f"{request.dataset}_{request.task}"

    outcomes: list[TaskCandidateMaterializationOutcome] = []
    if not input_report.resolved or input_report.inputs is None:
        for program in programs:
            outcomes.append(
                TaskCandidateMaterializationOutcome(
                    program_id=program.program_id,
                    output_dir=(
                        output_root / "candidates" / program.program_id
                    ),
                    primitive_count=len(program.primitive_ids),
                    lowering_feasible=False,
                    status="blocked",
                    blockers=input_report.blockers,
                )
            )
        return TaskCandidateMaterializationReport(
            dataset=request.dataset,
            task=request.task,
            output_root=request.output_root,
            dry_run=not request.write,
            input_resolved=False,
            input_blockers=input_report.blockers,
            evidence_locations=input_report.evidence_locations,
            outcomes=tuple(outcomes),
        )

    available_sources = {
        artifact.table_name
        for artifact in input_report.inputs.source_artifacts
    }
    for program in programs:
        output_dir = output_root / "candidates" / program.program_id
        plan = plan_candidate_materialization(
            compiled,
            program,
            available_source_tables=available_sources,
        )
        evidence = input_report.inputs.evidence_for_program(
            program.program_id
        )
        blockers = _task_candidate_blockers(
            plan=plan,
            evidence=evidence,
        )
        if blockers:
            outcomes.append(
                TaskCandidateMaterializationOutcome(
                    program_id=program.program_id,
                    output_dir=output_dir,
                    primitive_count=len(program.primitive_ids),
                    lowering_feasible=False,
                    status="blocked",
                    blockers=blockers,
                )
            )
            continue
        if not request.write:
            result = materialize_candidate_program(
                CandidateMaterializationRequest(
                    compiled=compiled,
                    program=program,
                    output_dir=output_dir,
                    source_rows_by_table={
                        table: ()
                        for table in available_sources
                    },
                    train_target_rows=(),
                    validation_target_rows=(),
                    write=False,
                    explicit_lowering_evidence=evidence,
                    candidate_id_columns=_candidate_id_columns(task_spec),
                )
            )
            outcomes.append(
                TaskCandidateMaterializationOutcome(
                    program_id=program.program_id,
                    output_dir=output_dir,
                    primitive_count=len(program.primitive_ids),
                    lowering_feasible=plan.materializable,
                    status="dry_run_ready",
                    blockers=(),
                    result=result,
                )
            )
            continue
        try:
            source_rows, train_rows, validation_rows = (
                load_rows_for_materialization_plan(
                    inputs=input_report.inputs,
                    plan=plan,
                    evidence=evidence,
                )
            )
            result = materialize_candidate_program(
                CandidateMaterializationRequest(
                    compiled=compiled,
                    program=program,
                    output_dir=output_dir,
                    source_rows_by_table=source_rows,
                    train_target_rows=train_rows,
                    validation_target_rows=validation_rows,
                    write=True,
                    overwrite=request.overwrite,
                    explicit_lowering_evidence=evidence,
                    candidate_id_columns=_candidate_id_columns(task_spec),
                )
            )
            status = "reused" if result.reused else "published"
            outcomes.append(
                TaskCandidateMaterializationOutcome(
                    program_id=program.program_id,
                    output_dir=output_dir,
                    primitive_count=len(program.primitive_ids),
                    lowering_feasible=True,
                    status=status,
                    blockers=(),
                    result=result,
                )
            )
        except Exception as exc:
            outcomes.append(
                TaskCandidateMaterializationOutcome(
                    program_id=program.program_id,
                    output_dir=output_dir,
                    primitive_count=len(program.primitive_ids),
                    lowering_feasible=plan.materializable,
                    status="failed",
                    blockers=(str(exc),),
                )
            )

    return TaskCandidateMaterializationReport(
        dataset=request.dataset,
        task=request.task,
        output_root=request.output_root,
        dry_run=not request.write,
        input_resolved=True,
        input_blockers=(),
        evidence_locations=input_report.evidence_locations,
        outcomes=tuple(outcomes),
    )


def _filter_task_programs(
    programs: Sequence[CandidateProgram],
    *,
    program_ids: Sequence[str],
    exclude_program_ids: Sequence[str],
    baseline_only: bool,
) -> tuple[CandidateProgram, ...]:
    by_id = {program.program_id: program for program in programs}
    if len(by_id) != len(programs):
        raise ValueError("duplicate candidate program IDs")
    unknown = sorted(
        (set(program_ids) | set(exclude_program_ids)) - set(by_id)
    )
    if unknown:
        raise ValueError(
            "unknown candidate program IDs: " + ", ".join(unknown)
        )
    selected = list(programs)
    if baseline_only:
        selected = [
            program for program in selected if program.program_id == "baseline"
        ]
    if program_ids:
        requested = set(program_ids)
        selected = [
            program for program in selected if program.program_id in requested
        ]
    if exclude_program_ids:
        excluded = set(exclude_program_ids)
        selected = [
            program for program in selected if program.program_id not in excluded
        ]
    return tuple(sorted(selected, key=lambda program: program.program_id))


def _task_candidate_blockers(
    *,
    plan: CandidateMaterializationPlan,
    evidence: Sequence[object],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not plan.materializable:
        blockers.extend(
            error
            for step in plan.steps
            for error in step.errors
        )
    evidence_ids = {
        str(getattr(record, "primitive_id"))
        for record in evidence
        if getattr(record, "status", "") == "proven"
        and getattr(record, "source_column", None)
        and getattr(record, "output_column", None)
    }
    for step in plan.steps:
        if step.lowering_mode in {
            LoweringMode.PASSTHROUGH,
            LoweringMode.EXTERNAL,
        } and step.primitive_id not in evidence_ids:
            blockers.append(
                f"missing_proven_lowering_evidence:{step.primitive_id}"
            )
    return tuple(dict.fromkeys(blockers))


def _candidate_id_columns(task_spec) -> tuple[str, ...]:
    if task_spec.pairwise is None:
        return ()
    return (task_spec.pairwise.target_right_key,)


def _reject_test_split(split: str) -> None:
    normalized = split.strip().lower().replace("-", "_")
    if normalized in {"test", "heldout_test", "held_out_test", "final"}:
        raise ValueError(
            "candidate materialization accepts train/validation only"
        )
    if normalized not in {"validation", "val"}:
        raise ValueError(
            f"unsupported validation split {split!r}; expected validation"
        )


def _generated_only_plan(
    plan: CandidateMaterializationPlan,
) -> CandidateMaterializationPlan:
    steps = tuple(
        step
        for step in plan.steps
        if step.lowering_mode == LoweringMode.GENERATE
    )
    audit_rows = tuple(
        row
        for row in plan.audit_rows
        if row.lowering_mode == LoweringMode.GENERATE
    )
    return CandidateMaterializationPlan(
        program_id=plan.program_id,
        steps=steps,
        audit_rows=audit_rows,
        materializable=all(step.materializable for step in steps),
        temporally_safe=all(step.temporally_safe for step in steps),
        requires_external_provider=False,
    )


def _tuple_source_rows(
    rows_by_table: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    return {
        str(table): tuple(rows)
        for table, rows in rows_by_table.items()
    }


def _validate_lowering_ready(
    *,
    dataset: str,
    task: str,
    plan: CandidateMaterializationPlan,
    feature_columns: Sequence[str],
    label_col: str,
    candidate_id_columns: Sequence[str],
    surrogate_key_columns: Sequence[str],
    target_aggregate_columns: Sequence[str],
    cross_fitted_target_aggregates: bool,
    passthrough_report: object | None,
    explicit_evidence: Sequence[object],
) -> None:
    from .candidate_safety import build_candidate_safety_audit_report

    if not plan.materializable:
        raise ValueError(
            "candidate plan is not materializable: "
            + "; ".join(
                error
                for step in plan.steps
                for error in step.errors
            )
        )
    if any(step.lowering_mode == LoweringMode.UNSUPPORTED for step in plan.steps):
        raise ValueError("candidate plan contains unsupported primitives")

    report = build_candidate_safety_audit_report(
        dataset=dataset,
        task=task,
        plan=plan,
        feature_columns=feature_columns,
        label_col=label_col,
        candidate_id_columns=candidate_id_columns,
        surrogate_key_columns=surrogate_key_columns,
        target_aggregate_columns=target_aggregate_columns,
        cross_fitted_target_aggregates=cross_fitted_target_aggregates,
        passthrough_report=passthrough_report,
        explicit_evidence=explicit_evidence,
    )
    if not report.provenance.passed:
        raise ValueError(
            "candidate lowering provenance is incomplete: "
            + ", ".join(_audit_failure_reasons(report))
        )


def _feature_columns_from_plan_and_evidence(
    plan: CandidateMaterializationPlan,
    explicit_evidence: Sequence[object],
    passthrough_report: object | None,
) -> tuple[str, ...]:
    records = _binding_records(
        dataset="",
        task="",
        plan=plan,
        explicit_evidence=explicit_evidence,
        passthrough_report=passthrough_report,
    )
    return _feature_columns_from_bindings(records)


def _combine_output_rows(
    *,
    target_rows: tuple[Mapping[str, object], ...],
    generated_rows,
    plan: CandidateMaterializationPlan,
    explicit_evidence: Sequence[object],
    passthrough_report: object | None,
) -> list[dict[str, object]]:
    provider_outputs = _provider_output_bindings(
        plan=plan,
        explicit_evidence=explicit_evidence,
        passthrough_report=passthrough_report,
    )
    generated_by_index = {
        row.row_index: dict(row.values)
        for row in generated_rows
    }
    output_rows: list[dict[str, object]] = []
    for index, target_row in enumerate(target_rows):
        out = dict(target_row)
        out.update(generated_by_index.get(index, {}))
        for source_column, output_column in provider_outputs:
            if source_column not in target_row:
                raise ValueError(
                    "provider source column missing from target rows: "
                    f"{source_column}"
                )
            out[output_column] = target_row[source_column]
        output_rows.append(out)
    return output_rows


def _provider_output_bindings(
    *,
    plan: CandidateMaterializationPlan,
    explicit_evidence: Sequence[object],
    passthrough_report: object | None,
) -> tuple[tuple[str, str], ...]:
    evidence_by_primitive = _evidence_by_primitive(
        explicit_evidence=explicit_evidence,
        passthrough_report=passthrough_report,
    )
    bindings: list[tuple[str, str]] = []
    for step in plan.steps:
        if step.lowering_mode == LoweringMode.GENERATE:
            continue
        evidence = evidence_by_primitive.get(step.primitive_id)
        source_column = getattr(evidence, "source_column", None)
        output_column = getattr(evidence, "output_column", None)
        if not source_column or not output_column:
            raise ValueError(
                "provider primitive lacks proven source/output columns: "
                f"{step.primitive_id}"
            )
        bindings.append((str(source_column), str(output_column)))
    return tuple(bindings)


def _binding_records(
    *,
    dataset: str,
    task: str,
    plan: CandidateMaterializationPlan,
    explicit_evidence: Sequence[object],
    passthrough_report: object | None,
) -> list[dict[str, object]]:
    evidence_by_primitive = _evidence_by_primitive(
        explicit_evidence=explicit_evidence,
        passthrough_report=passthrough_report,
    )
    records: list[dict[str, object]] = []
    for step in plan.steps:
        if step.lowering_mode == LoweringMode.GENERATE:
            output_columns = step.output_columns
            source_columns = tuple(
                column
                for column in (
                    step.source_event_time_col,
                    step.source_group_key,
                    step.source_left_key,
                    step.source_right_key,
                    step.related_col,
                )
                if column
            )
            evidence_location = "materialization-plan"
            status = "native_lowering"
        else:
            evidence = evidence_by_primitive.get(step.primitive_id)
            output = getattr(evidence, "output_column", None)
            source = getattr(evidence, "source_column", None)
            output_columns = (str(output),) if output else ()
            source_columns = (str(source),) if source else ()
            evidence_location = (
                str(getattr(evidence, "evidence_location", ""))
                if evidence is not None
                else ""
            )
            status = (
                str(getattr(evidence, "status", "missing"))
                if evidence is not None
                else "missing"
            )

        records.append({
            "dataset": dataset,
            "task": task,
            "program_id": plan.program_id,
            "primitive_id": step.primitive_id,
            "lowering_mode": step.lowering_mode.value,
            "source_table": step.source_table,
            "source_columns": list(source_columns),
            "output_columns": list(output_columns),
            "evidence_location": evidence_location,
            "status": status,
        })
    return sorted(
        records,
        key=lambda row: (str(row["primitive_id"]), str(row["lowering_mode"])),
    )


def _evidence_by_primitive(
    *,
    explicit_evidence: Sequence[object],
    passthrough_report: object | None,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for record in getattr(passthrough_report, "proven_bindings", ()) or ():
        out[str(getattr(record, "primitive_id"))] = record
    for record in explicit_evidence:
        primitive_id = str(getattr(record, "primitive_id"))
        existing = out.get(primitive_id)
        if existing is not None and existing != record:
            raise ValueError(
                "conflicting lowering evidence for "
                f"{primitive_id}"
            )
        out[primitive_id] = record
    return out


def _feature_columns_from_bindings(
    bindings: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for record in bindings:
        for column in record["output_columns"]:
            name = str(column)
            if name in seen:
                raise ValueError(
                    f"duplicate materialized feature column: {name}"
                )
            seen.add(name)
            columns.append(name)
    return tuple(columns)


def _write_parquet_rows(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    import pandas as pd

    pd.DataFrame(list(rows)).to_parquet(path, index=False)


def _write_candidate_program_json(
    compiled: CompiledTask,
    program: CandidateProgram,
    path: Path,
) -> None:
    _write_json_deterministic(
        {
            "compiled_task": compiled.to_dict(),
            "candidate_program": program.to_dict(),
        },
        path,
    )


def _write_candidate_manifest_csv(
    *,
    dataset: str,
    task: str,
    plan: CandidateMaterializationPlan,
    bindings: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    fields = (
        "dataset",
        "task",
        "program_id",
        "primitive_id",
        "lowering_mode",
        "source_table",
        "source_columns",
        "output_columns",
        "status",
        "evidence_location",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in bindings:
            writer.writerow({
                "dataset": dataset,
                "task": task,
                "program_id": plan.program_id,
                "primitive_id": record["primitive_id"],
                "lowering_mode": record["lowering_mode"],
                "source_table": record["source_table"] or "",
                "source_columns": "|".join(record["source_columns"]),
                "output_columns": "|".join(record["output_columns"]),
                "status": record["status"],
                "evidence_location": record["evidence_location"],
            })


def _write_safety_audits(
    *,
    staging: Path,
    report,
    write_audit_csv,
) -> tuple[Path, ...]:
    paths = (
        staging / "temporal_safety_audit.csv",
        staging / "leakage_safety_audit.csv",
        staging / "lowering_provenance_audit.csv",
    )
    for path, audit in zip(
        paths,
        (report.temporal, report.leakage, report.provenance),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            write_audit_csv(audit.rows, handle)
    return paths


def _audit_failure_reasons(report) -> tuple[str, ...]:
    reasons: list[str] = []
    for audit in (report.temporal, report.leakage, report.provenance):
        for row in audit.rows:
            if not row.passed:
                reasons.append(row.rejection_reason or row.status)
    return tuple(dict.fromkeys(reason for reason in reasons if reason))


def _artifacts_schema_valid(
    *,
    train_path: Path,
    validation_path: Path,
    feature_columns: Sequence[str],
) -> bool:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return train_path.exists() and validation_path.exists()
    if not train_path.exists() or not validation_path.exists():
        return False
    train_schema = pq.read_schema(train_path)
    validation_schema = pq.read_schema(validation_path)
    train_columns = set(train_schema.names)
    validation_columns = set(validation_schema.names)
    return all(
        column in train_columns and column in validation_columns
        for column in feature_columns
    )


def _materialization_manifest(
    *,
    dataset: str,
    task: str,
    plan: CandidateMaterializationPlan,
    train_path: Path,
    validation_path: Path,
    train_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str],
    status: str,
    failure_reasons: Sequence[str],
    bindings: Sequence[Mapping[str, object]],
    audit_paths: Sequence[Path],
) -> dict[str, object]:
    target_columns = _target_columns(
        train_rows=train_rows,
        validation_rows=validation_rows,
        feature_columns=feature_columns,
    )
    evidence_locations = tuple(
        sorted({
            str(record["evidence_location"])
            for record in bindings
            if record.get("evidence_location")
        })
    )
    return {
        "dataset": dataset,
        "task": task,
        "program_id": plan.program_id,
        "train_artifact": train_path.name,
        "validation_artifact": validation_path.name,
        "train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "feature_count": len(feature_columns),
        "target_columns": list(target_columns),
        "feature_columns": list(feature_columns),
        "prediction_time_column": (
            plan.steps[0].target_time_col if plan.steps else ""
        ),
        "materialization_status": status,
        "failure_reasons": list(failure_reasons),
        "source_evidence_locations": list(evidence_locations),
        "train_schema_sha256": _schema_hash(train_path),
        "validation_schema_sha256": _schema_hash(validation_path),
        "audit_files": [path.name for path in audit_paths],
    }


def _target_columns(
    *,
    train_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    feature_set = set(feature_columns)
    columns: list[str] = []
    seen: set[str] = set()
    for row in (*train_rows, *validation_rows):
        for column in row:
            name = str(column)
            if name in feature_set or name in seen:
                continue
            seen.add(name)
            columns.append(name)
    return tuple(columns)


def _schema_hash(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ""
    if not path.exists():
        return ""
    schema = pq.read_schema(path)
    payload = "\n".join(
        f"{field.name}:{field.type}" for field in schema
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_deterministic(payload: object, path: Path) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    try:
        return asdict(value)
    except TypeError:
        return str(value)


def _validate_complete_staging(
    staging: Path,
    *,
    require_selector_ready: bool,
) -> None:
    required = (
        "target_with_dfs_agg_train.parquet",
        "target_with_dfs_agg_val.parquet",
        "candidate_program.json",
        "candidate_manifest.csv",
        "materialization_manifest.json",
        "primitive_column_bindings.json",
        "temporal_safety_audit.csv",
        "leakage_safety_audit.csv",
        "lowering_provenance_audit.csv",
    )
    missing = [name for name in required if not (staging / name).is_file()]
    if missing:
        raise ValueError(
            "staged candidate directory is incomplete: "
            + ", ".join(missing)
        )
    if require_selector_ready:
        manifest = json.loads(
            (staging / "materialization_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if manifest.get("materialization_status") != "success":
            raise ValueError("staged manifest does not report success")


def _publish_staging(
    staging: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    backup: Path | None = None
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(output_dir)
        backup = _unique_backup_path(output_dir)
        output_dir.replace(backup)
    try:
        staging.replace(output_dir)
    except OSError:
        if backup is not None:
            backup.replace(output_dir)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _unique_backup_path(output_dir: Path) -> Path:
    while True:
        candidate = (
            output_dir.parent
            / f"_{output_dir.name}.{uuid.uuid4().hex}.backup"
        )
        if not candidate.exists():
            return candidate


def _candidate_dir_matches(
    output_dir: Path,
    *,
    dataset: str,
    task: str,
    program_id: str,
) -> bool:
    manifest_path = output_dir / "materialization_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if (
        manifest.get("dataset") != dataset
        or manifest.get("task") != task
        or manifest.get("program_id") != program_id
        or manifest.get("materialization_status") != "success"
    ):
        return False
    return all(
        (output_dir / name).is_file()
        for name in (
            "target_with_dfs_agg_train.parquet",
            "target_with_dfs_agg_val.parquet",
            "candidate_program.json",
            "candidate_manifest.csv",
            "primitive_column_bindings.json",
            "temporal_safety_audit.csv",
            "leakage_safety_audit.csv",
            "lowering_provenance_audit.csv",
        )
    )


def _reused_materialization_result(
    *,
    request: CandidateMaterializationRequest,
    plan: CandidateMaterializationPlan,
    feature_columns: tuple[str, ...],
    train_row_count: int,
    validation_row_count: int,
) -> CandidateMaterializationResult:
    output_dir = request.output_dir
    return CandidateMaterializationResult(
        dataset=request.compiled.task_spec.dataset,
        task=request.compiled.task_spec.task,
        program_id=request.program.program_id,
        output_dir=output_dir,
        plan=plan,
        dry_run=False,
        reused=True,
        materializable=True,
        temporally_safe=True,
        leakage_safe=True,
        provenance_complete=True,
        selector_ready=True,
        train_artifact=output_dir / "target_with_dfs_agg_train.parquet",
        validation_artifact=output_dir / "target_with_dfs_agg_val.parquet",
        manifest_path=output_dir / "materialization_manifest.json",
        bindings_path=output_dir / "primitive_column_bindings.json",
        audit_paths=(
            output_dir / "temporal_safety_audit.csv",
            output_dir / "leakage_safety_audit.csv",
            output_dir / "lowering_provenance_audit.csv",
        ),
        feature_columns=feature_columns,
        train_row_count=train_row_count,
        validation_row_count=validation_row_count,
    )


def _index_primitives_by_id(
    primitives: tuple[Primitive, ...] | list[Primitive],
) -> dict[str, Primitive]:
    primitive_by_id: dict[str, Primitive] = {}
    duplicate_ids: list[str] = []

    for primitive in primitives:
        if primitive.primitive_id in primitive_by_id:
            duplicate_ids.append(primitive.primitive_id)
        else:
            primitive_by_id[primitive.primitive_id] = primitive

    if duplicate_ids:
        duplicates = ", ".join(sorted(set(duplicate_ids)))
        raise ValueError(
            "duplicate primitive_id values in compiled task: "
            f"{duplicates}"
        )

    return primitive_by_id


def resolve_pairwise_binding(
    *,
    compiled: CompiledTask,
    primitive: Primitive,
) -> PhysicalHistoryBinding:
    pairwise = compiled.task_spec.pairwise

    if pairwise is None:
        raise ValueError(
            "Cannot resolve pairwise binding without PairwiseSpec"
        )

    role = primitive.metadata.get("pairwise_role")
    target_time_col = compiled.task_spec.target_time_col

    if role == "left":
        return _single_history_binding(
            role="left",
            history=pairwise.left_history,
            target_key=pairwise.left_key,
            target_time_col=target_time_col,
        )

    if role == "right":
        return _single_history_binding(
            role="right",
            history=pairwise.right_history,
            target_key=pairwise.target_right_key,
            target_time_col=target_time_col,
        )

    if role == "pair":
        history = pairwise.pair_history

        return PhysicalHistoryBinding(
            role="pair",
            source_table=(
                history.table if history is not None else None
            ),
            source_group_key=None,
            source_left_key=(
                history.left_key
                if history is not None
                else None
            ),
            source_right_key=(
                history.right_key
                if history is not None
                else None
            ),
            source_event_time_col=(
                history.time_col
                if history is not None
                else None
            ),
            target_key=None,
            target_left_key=pairwise.left_key,
            target_right_key=pairwise.target_right_key,
            target_time_col=target_time_col,
        )

    return PhysicalHistoryBinding(
        role=str(role) if role is not None else "",
        source_table=primitive.source_table,
        source_group_key=primitive.group_key,
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=primitive.event_time_col,
        target_key=primitive.metadata.get("target_key"),
        target_left_key=primitive.metadata.get(
            "target_left_key"
        ),
        target_right_key=primitive.metadata.get(
            "target_right_key"
        ),
        target_time_col=target_time_col,
        related_col=primitive.numeric_col,
    )


def _step_for_primitive(
    *,
    compiled: CompiledTask,
    program_id: str,
    primitive: Primitive,
    available_source_tables: Collection[str] | None,
    cutoff_operator: str,
    reference_bindings: (
        Mapping[str, PhysicalHistoryBinding] | None
    ),
) -> PrimitiveMaterializationStep:
    if primitive.family == PrimitiveFamily.BASELINE:
        return _provider_step(
            program_id=program_id,
            primitive=primitive,
            target_time_col=(
                compiled.task_spec.target_time_col
            ),
            cutoff_operator=cutoff_operator,
            lowering_mode=LoweringMode.PASSTHROUGH,
            requires_external_provider=False,
            warning=(
                "baseline primitive is expected to be present "
                "in the base candidate artifact"
            ),
        )

    if primitive.family == PrimitiveFamily.STRUCTURAL:
        return _provider_step(
            program_id=program_id,
            primitive=primitive,
            target_time_col=(
                compiled.task_spec.target_time_col
            ),
            cutoff_operator=cutoff_operator,
            lowering_mode=LoweringMode.EXTERNAL,
            requires_external_provider=True,
            warning=(
                "structural primitive requires an external "
                "structural lowerer or artifact provider"
            ),
        )

    if primitive.operation not in PAIRWISE_GENERATED_OPERATIONS:
        return _unsupported_step(
            program_id=program_id,
            primitive=primitive,
            target_time_col=(
                compiled.task_spec.target_time_col
            ),
            cutoff_operator=cutoff_operator,
        )

    if compiled.task_spec.pairwise is None:
        return _unsupported_step(
            program_id=program_id,
            primitive=primitive,
            target_time_col=(
                compiled.task_spec.target_time_col
            ),
            cutoff_operator=cutoff_operator,
            reason=(
                "pairwise generated materialization requires "
                "TaskSpec.pairwise"
            ),
        )

    binding = resolve_pairwise_binding(
        compiled=compiled,
        primitive=primitive,
    )
    return _generated_step(
        program_id=program_id,
        primitive=primitive,
        binding=binding,
        available_source_tables=available_source_tables,
        cutoff_operator=cutoff_operator,
        reference_bindings=reference_bindings,
    )


def _single_history_binding(
    *,
    role: str,
    history: PairwiseHistorySpec | None,
    target_key: str | None,
    target_time_col: str,
) -> PhysicalHistoryBinding:
    return PhysicalHistoryBinding(
        role=role,
        source_table=(
            history.table if history is not None else None
        ),
        source_group_key=(
            history.key if history is not None else None
        ),
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=(
            history.time_col if history is not None else None
        ),
        target_key=target_key,
        target_left_key=None,
        target_right_key=None,
        target_time_col=target_time_col,
        related_col=(
            history.related_col
            if history is not None
            else None
        ),
    )


def _generated_step(
    *,
    program_id: str,
    primitive: Primitive,
    binding: PhysicalHistoryBinding,
    available_source_tables: Collection[str] | None,
    cutoff_operator: str,
    reference_bindings: (
        Mapping[str, PhysicalHistoryBinding] | None
    ),
) -> PrimitiveMaterializationStep:
    binding_errors = _validate_binding(
        primitive=primitive,
        binding=binding,
        available_source_tables=available_source_tables,
    )
    safety_errors = _validate_temporal_policy(
        primitive=primitive,
        binding=binding,
        cutoff_operator=cutoff_operator,
    )
    warnings = _compare_reference_binding(
        binding=binding,
        reference=(
            reference_bindings or {}
        ).get(binding.role),
    )

    errors = tuple(binding_errors + safety_errors)

    return PrimitiveMaterializationStep(
        program_id=program_id,
        primitive_id=primitive.primitive_id,
        operation=primitive.operation,
        lowering_mode=LoweringMode.GENERATE,
        pairwise_role=binding.role or None,
        source_table=binding.source_table,
        source_group_key=binding.source_group_key,
        source_left_key=binding.source_left_key,
        source_right_key=binding.source_right_key,
        source_event_time_col=binding.source_event_time_col,
        target_key=binding.target_key,
        target_left_key=binding.target_left_key,
        target_right_key=binding.target_right_key,
        target_time_col=binding.target_time_col,
        related_col=binding.related_col,
        window_days=primitive.window_days,
        cutoff_operator=cutoff_operator,
        output_columns=_output_columns(primitive, binding.role),
        materializable=not errors,
        temporally_safe=not safety_errors,
        requires_external_provider=False,
        errors=errors,
        warnings=tuple(warnings),
    )


def _validate_binding(
    *,
    primitive: Primitive,
    binding: PhysicalHistoryBinding,
    available_source_tables: Collection[str] | None,
) -> list[str]:
    errors: list[str] = []
    role = binding.role

    if role not in {"left", "right", "pair"}:
        errors.append("unknown pairwise role")

    if binding.source_table is None:
        errors.append(f"{role} history source table is missing")
    elif (
        available_source_tables is not None
        and binding.source_table not in available_source_tables
    ):
        errors.append(
            f"{role} history source table is not configured: "
            f"{binding.source_table}"
        )

    if binding.source_event_time_col is None:
        errors.append(
            f"{role} history event-time column is missing"
        )

    if role in {"left", "right"}:
        if binding.source_group_key is None:
            errors.append(f"{role} history source key is missing")
        if binding.target_key is None:
            errors.append(f"{role} target key is missing")
        if (
            primitive.operation == "past_unique_neighbors"
            and binding.related_col is None
        ):
            errors.append(
                f"{role} related column is required for "
                "past_unique_neighbors"
            )

    if role == "pair":
        if binding.source_left_key is None:
            errors.append("pair history source left key is missing")
        if binding.source_right_key is None:
            errors.append("pair history source right key is missing")
        if binding.target_left_key is None:
            errors.append("pair target left key is missing")
        if binding.target_right_key is None:
            errors.append("pair target right key is missing")

    return errors


def _validate_temporal_policy(
    *,
    primitive: Primitive,
    binding: PhysicalHistoryBinding,
    cutoff_operator: str,
) -> list[str]:
    errors: list[str] = []

    if cutoff_operator != STRICT_PRIOR_EVENT_OPERATOR:
        errors.append(
            "configured cutoff operator is not strict prior-event '<'"
        )

    if primitive.event_time_col is None:
        errors.append(
            "logical primitive has no source event-time binding"
        )
    elif (
        binding.source_event_time_col is not None
        and primitive.event_time_col
        != binding.source_event_time_col
    ):
        errors.append(
            "logical primitive event-time binding does not "
            "match physical history binding"
        )

    if (
        primitive.source_table is not None
        and binding.source_table is not None
        and primitive.source_table != binding.source_table
    ):
        errors.append(
            "logical primitive source table does not match "
            "physical history binding"
        )

    predicate = primitive.temporal_predicate or ""

    if "<=" in predicate:
        errors.append(
            "logical temporal predicate is non-strict and "
            "does not match strict source_event_time < target_time"
        )
    elif primitive.temporal_predicate is None:
        errors.append(
            "logical temporal predicate is missing"
        )
    elif "<" not in predicate:
        errors.append(
            "logical temporal predicate does not expose a "
            "strict prior-event comparison"
        )

    if (
        predicate
        and binding.source_event_time_col is not None
        and binding.source_event_time_col not in predicate
    ):
        errors.append(
            "logical temporal predicate does not reference "
            "the resolved source event-time column"
        )

    if (
        predicate
        and binding.target_time_col not in predicate
    ):
        errors.append(
            "logical temporal predicate does not reference "
            "the target time column"
        )

    return errors


def _compare_reference_binding(
    *,
    binding: PhysicalHistoryBinding,
    reference: PhysicalHistoryBinding | None,
) -> list[str]:
    if reference is None:
        return []

    warnings: list[str] = []

    comparisons = [
        ("source_table", binding.source_table, reference.source_table),
        (
            "source_event_time_col",
            binding.source_event_time_col,
            reference.source_event_time_col,
        ),
        (
            "source_group_key",
            binding.source_group_key,
            reference.source_group_key,
        ),
        (
            "source_left_key",
            binding.source_left_key,
            reference.source_left_key,
        ),
        (
            "source_right_key",
            binding.source_right_key,
            reference.source_right_key,
        ),
    ]

    for name, actual, expected in comparisons:
        if expected is not None and actual != expected:
            warnings.append(
                "reference binding mismatch for "
                f"{binding.role}.{name}: "
                f"configured={actual!r}, reference={expected!r}"
            )

    return warnings


def _output_columns(
    primitive: Primitive,
    role: str,
) -> tuple[str, ...]:
    prefix = f"f_pairwise__{role}"
    operation = primitive.operation

    if operation == "window_count":
        return (
            f"{prefix}__count_{primitive.window_days}d",
        )

    if operation == "past_unique_neighbors":
        return (
            f"{prefix}__unique_neighbors_"
            f"{primitive.window_days}d",
        )

    if operation == "days_since_last":
        return (
            f"{prefix}__days_since_last",
            f"{prefix}__days_since_last__is_missing",
        )

    if operation == "prior_pair_count":
        return ("f_pairwise__pair__prior_count",)

    if operation == "pair_days_since_last":
        return (
            "f_pairwise__pair__days_since_last",
            "f_pairwise__pair__days_since_last__is_missing",
        )

    return ()


def _provider_step(
    *,
    program_id: str,
    primitive: Primitive,
    target_time_col: str,
    cutoff_operator: str,
    lowering_mode: LoweringMode,
    requires_external_provider: bool,
    warning: str,
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id=program_id,
        primitive_id=primitive.primitive_id,
        operation=primitive.operation,
        lowering_mode=lowering_mode,
        pairwise_role=primitive.metadata.get(
            "pairwise_role"
        ),
        source_table=primitive.source_table,
        source_group_key=primitive.group_key,
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=primitive.event_time_col,
        target_key=primitive.metadata.get("target_key"),
        target_left_key=primitive.metadata.get(
            "target_left_key"
        ),
        target_right_key=primitive.metadata.get(
            "target_right_key"
        ),
        target_time_col=target_time_col,
        related_col=primitive.numeric_col,
        window_days=primitive.window_days,
        cutoff_operator=cutoff_operator,
        output_columns=(),
        materializable=True,
        temporally_safe=True,
        requires_external_provider=(
            requires_external_provider
        ),
        warnings=(warning,),
    )


def _unsupported_step(
    *,
    program_id: str,
    primitive: Primitive,
    target_time_col: str,
    cutoff_operator: str,
    reason: str | None = None,
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id=program_id,
        primitive_id=primitive.primitive_id,
        operation=primitive.operation,
        lowering_mode=LoweringMode.UNSUPPORTED,
        pairwise_role=primitive.metadata.get(
            "pairwise_role"
        ),
        source_table=primitive.source_table,
        source_group_key=primitive.group_key,
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=primitive.event_time_col,
        target_key=primitive.metadata.get("target_key"),
        target_left_key=primitive.metadata.get(
            "target_left_key"
        ),
        target_right_key=primitive.metadata.get(
            "target_right_key"
        ),
        target_time_col=target_time_col,
        related_col=primitive.numeric_col,
        window_days=primitive.window_days,
        cutoff_operator=cutoff_operator,
        output_columns=(),
        materializable=False,
        temporally_safe=False,
        requires_external_provider=False,
        errors=(
            reason
            or f"unsupported materialization operation: "
            f"{primitive.operation}",
        ),
    )


def _missing_primitive_step(
    *,
    program_id: str,
    primitive_id: str,
    target_time_col: str,
    cutoff_operator: str,
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id=program_id,
        primitive_id=primitive_id,
        operation="",
        lowering_mode=LoweringMode.UNSUPPORTED,
        pairwise_role=None,
        source_table=None,
        source_group_key=None,
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=None,
        target_key=None,
        target_left_key=None,
        target_right_key=None,
        target_time_col=target_time_col,
        related_col=None,
        window_days=None,
        cutoff_operator=cutoff_operator,
        output_columns=(),
        materializable=False,
        temporally_safe=False,
        requires_external_provider=False,
        errors=("primitive is not present in compiled task",),
    )


def _audit_row(
    *,
    program_id: str,
    primitive: Primitive | None,
    step: PrimitiveMaterializationStep,
    cutoff_operator: str,
) -> MaterializationAuditRow:
    predicate = (
        primitive.temporal_predicate
        if primitive is not None
        else None
    )

    return MaterializationAuditRow(
        program_id=program_id,
        primitive_id=step.primitive_id,
        lowering_mode=step.lowering_mode,
        pairwise_role=step.pairwise_role,
        source_table=step.source_table,
        source_event_time_col=step.source_event_time_col,
        logical_temporal_predicate=predicate,
        required_cutoff_operator=STRICT_PRIOR_EVENT_OPERATOR,
        configured_cutoff_operator=cutoff_operator,
        temporally_safe=step.temporally_safe,
        materializable=step.materializable,
        requires_external_provider=(
            step.requires_external_provider
        ),
        errors=step.errors,
        warnings=step.warnings,
    )
