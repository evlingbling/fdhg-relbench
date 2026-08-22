from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd

from .config import load_task_spec
from .materializer import (
    TaskCandidateMaterializationReport,
    TaskCandidateMaterializationRequest,
    materialize_task_candidates,
)
from .planner import build_candidate_program
from .programs import CandidateProgram, build_configured_candidates
from .selection import (
    CandidateSelectionDecision,
    CandidateSelectionPolicy,
    load_candidate_validation_results,
    select_candidate_program,
)
from .validation_export import (
    CandidateSafetyEvidence,
    build_validation_export_records,
    inspect_candidate_safety_evidence,
    write_validation_export_csv,
)


PIPELINE_STAGES = (
    "resolve",
    "compile",
    "materialize",
    "discover",
    "validate",
    "export",
    "select",
    "report",
)


@dataclass(frozen=True)
class EvaluationRequest:
    dataset: str
    task: str
    program_id: str
    seed: int
    artifact_dir: Path
    result_root: Path
    primary_metric: str
    metric_direction: str


@dataclass(frozen=True)
class EvaluationResult:
    request: EvaluationRequest
    status: str
    score: float | None
    n_features: int | None
    evidence_location: str
    rejection_reason: str = ""
    command: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()


class CandidateEvaluator(Protocol):
    def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        ...


@dataclass(frozen=True)
class TaskPipelineRequest:
    dataset: str
    task: str
    output_root: Path
    result_root: Path
    seeds: tuple[int, ...]
    mode: str = "dry-run"
    program_ids: tuple[str, ...] = ()
    exclude_program_ids: tuple[str, ...] = ()
    baseline_only: bool = False
    write_materialization: bool = False
    run_validation: bool = False
    select: bool = False
    overwrite: bool = False
    overwrite_pipeline_output: bool = False
    reproduction_config: Path = Path("configs/reproduction/tasks.yaml")
    semantics_config: Path = Path("configs/reproduction/task_semantics.yaml")


@dataclass(frozen=True)
class TaskPipelineStageResult:
    stage: str
    status: str
    inputs: Mapping[str, object]
    outputs: Mapping[str, object]
    blockers: tuple[str, ...] = ()
    evidence_locations: tuple[str, ...] = ()
    failure_reason: str = ""


@dataclass(frozen=True)
class TaskPipelineReport:
    dataset: str
    task: str
    requested_mode: str
    pipeline_status: str
    stages: tuple[TaskPipelineStageResult, ...]
    candidate_ids: tuple[str, ...]
    discovered_candidates: tuple[str, ...]
    evaluation_results: tuple[EvaluationResult, ...]
    canonical_validation_path: Path | None
    selection_decision: CandidateSelectionDecision | None
    warnings: tuple[str, ...]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "requested_mode": self.requested_mode,
            "pipeline_status": self.pipeline_status,
            "candidate_ids": list(self.candidate_ids),
            "discovered_candidates": list(self.discovered_candidates),
            "canonical_validation_path": (
                None
                if self.canonical_validation_path is None
                else str(self.canonical_validation_path)
            ),
            "selection_decision": (
                None
                if self.selection_decision is None
                else asdict(self.selection_decision)
            ),
            "evaluation_results": [
                _evaluation_result_to_dict(result)
                for result in self.evaluation_results
            ],
            "stages": [asdict(stage) for stage in self.stages],
            "warnings": list(self.warnings),
            "failures": list(self.failures),
        }


def run_task_pipeline(
    request: TaskPipelineRequest,
    *,
    evaluator: CandidateEvaluator | None = None,
) -> TaskPipelineReport:
    mode = _normalize_mode(request)
    stages: list[TaskPipelineStageResult] = []
    warnings: list[str] = []
    failures: list[str] = []
    discovered: tuple[str, ...] = ()
    evaluation_results: tuple[EvaluationResult, ...] = ()
    canonical_path: Path | None = None
    decision: CandidateSelectionDecision | None = None

    task_spec = load_task_spec(
        dataset=request.dataset,
        task=request.task,
        reproduction_config=request.reproduction_config,
        semantics_config=request.semantics_config,
    )
    stages.append(TaskPipelineStageResult(
        stage="resolve",
        status="completed",
        inputs={"dataset": request.dataset, "task": request.task},
        outputs={
            "primary_metric": task_spec.primary_metric,
            "metric_direction": task_spec.metric_direction,
        },
    ))
    compiled = build_candidate_program(task_spec)
    programs = _filter_programs(
        build_configured_candidates(
            compiled,
            reproduction_config=request.reproduction_config,
            semantics_config=request.semantics_config,
        ),
        program_ids=request.program_ids,
        exclude_program_ids=request.exclude_program_ids,
        baseline_only=request.baseline_only,
    )
    candidate_ids = tuple(program.program_id for program in programs)
    stages.append(TaskPipelineStageResult(
        stage="compile",
        status="completed",
        inputs={"candidate_primitive_count": len(compiled.candidate_primitives)},
        outputs={"candidate_ids": candidate_ids},
    ))

    materialization_report: TaskCandidateMaterializationReport | None = None
    if _includes_stage(mode, "materialize"):
        materialization_report = materialize_task_candidates(
            TaskCandidateMaterializationRequest(
                dataset=request.dataset,
                task=request.task,
                output_root=request.output_root,
                reproduction_config=request.reproduction_config,
                semantics_config=request.semantics_config,
                program_ids=request.program_ids,
                exclude_program_ids=request.exclude_program_ids,
                baseline_only=request.baseline_only,
                write=request.write_materialization and mode != "dry-run",
                overwrite=request.overwrite,
            )
        )
        status = _materialization_stage_status(materialization_report)
        candidate_blockers = _materialization_candidate_blockers(
            materialization_report,
        )
        stages.append(TaskPipelineStageResult(
            stage="materialize",
            status=status,
            inputs={"write": request.write_materialization and mode != "dry-run"},
            outputs={
                "published": materialization_report.published_count,
                "reused": materialization_report.reused_count,
                "blocked": materialization_report.blocked_count,
                "failed": materialization_report.failed_count,
                "candidate_blockers": candidate_blockers,
            },
            blockers=(
                materialization_report.input_blockers
                + tuple(candidate_blockers)
            ),
            evidence_locations=materialization_report.evidence_locations,
        ))
    else:
        stages.append(_skipped("materialize"))

    if _includes_stage(mode, "discover"):
        discovered = discover_strict_materialized_candidates(
            dataset=request.dataset,
            task=request.task,
            output_root=request.output_root,
            candidate_ids=candidate_ids,
        )
        status = "completed" if discovered else "blocked"
        blockers = () if discovered else ("no_strict_materialized_candidates",)
        stages.append(TaskPipelineStageResult(
            stage="discover",
            status=status,
            inputs={"candidate_ids": candidate_ids},
            outputs={"discovered_candidates": discovered},
            blockers=blockers,
        ))
    else:
        stages.append(_skipped("discover"))

    if _includes_stage(mode, "validate"):
        if evaluator is None and request.run_validation:
            stages.append(TaskPipelineStageResult(
                stage="validate",
                status="blocked",
                inputs={"run_validation": True},
                outputs={},
                blockers=("no_candidate_evaluator_configured",),
            ))
        elif not request.run_validation:
            stages.append(TaskPipelineStageResult(
                stage="validate",
                status="skipped",
                inputs={"run_validation": False},
                outputs={},
                blockers=("validation_not_requested",),
            ))
        else:
            evaluation_results = _run_validation(
                request=request,
                task_spec=task_spec,
                discovered=discovered,
                evaluator=evaluator,
            )
            validation_counts = _validation_result_counts(
                evaluation_results,
            )
            validation_blockers = _validation_result_blockers(
                evaluation_results,
            )
            failures.extend(
                validation_blockers
                if validation_counts["failed"] > 0
                else ()
            )
            stages.append(TaskPipelineStageResult(
                stage="validate",
                status=_validation_stage_status(evaluation_results),
                inputs={"seeds": request.seeds, "candidates": discovered},
                outputs={
                    "completed": validation_counts["completed"],
                    "reused": validation_counts["reused"],
                    "blocked": validation_counts["blocked"],
                    "failed": validation_counts["failed"],
                    "evaluation_result_count": validation_counts["total"],
                    "candidate_seed_blockers": validation_blockers,
                },
                blockers=validation_blockers,
                evidence_locations=tuple(sorted({
                    result.evidence_location
                    for result in evaluation_results
                    if result.evidence_location
                })),
            ))
    else:
        stages.append(_skipped("validate"))

    if _includes_stage(mode, "export"):
        if not evaluation_results:
            stages.append(TaskPipelineStageResult(
                stage="export",
                status="blocked",
                inputs={},
                outputs={},
                blockers=("missing_validation_results",),
            ))
        else:
            canonical_path = _canonical_validation_path(request)
            _write_canonical_export(
                request=request,
                task_spec=task_spec,
                discovered=discovered,
                evaluation_results=evaluation_results,
                output_path=canonical_path,
            )
            stages.append(TaskPipelineStageResult(
                stage="export",
                status="completed",
                inputs={"evaluation_result_count": len(evaluation_results)},
                outputs={"canonical_validation_path": str(canonical_path)},
            ))
    else:
        stages.append(_skipped("export"))

    if _includes_stage(mode, "select") or request.select:
        selection_path = canonical_path or _canonical_validation_path(request)
        if not selection_path.exists():
            stages.append(TaskPipelineStageResult(
                stage="select",
                status="blocked",
                inputs={"canonical_validation_path": str(selection_path)},
                outputs={},
                blockers=("canonical_validation_missing",),
            ))
        else:
            records = load_candidate_validation_results(selection_path)
            decision = select_candidate_program(
                programs,
                records,
                CandidateSelectionPolicy(
                    dataset=request.dataset,
                    task=request.task,
                    primary_metric=_require_metric(task_spec.primary_metric),
                    metric_direction=_require_metric(task_spec.metric_direction),
                    baseline_program_id="baseline",
                ),
            )
            stages.append(TaskPipelineStageResult(
                stage="select",
                status="completed",
                inputs={"canonical_validation_path": str(selection_path)},
                outputs={
                    "selected_program_id": decision.selected_program_id,
                    "fallback": decision.fallback_occurred,
                },
                evidence_locations=decision.evidence_locations,
            ))
    else:
        stages.append(_skipped("select"))

    report = TaskPipelineReport(
        dataset=request.dataset,
        task=request.task,
        requested_mode=mode,
        pipeline_status=_pipeline_status(stages),
        stages=tuple(stages),
        candidate_ids=candidate_ids,
        discovered_candidates=discovered,
        evaluation_results=evaluation_results,
        canonical_validation_path=canonical_path,
        selection_decision=decision,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )
    if mode != "dry-run" and _includes_stage(mode, "report"):
        _write_pipeline_report(
            request=request,
            report=report,
        )
    return report


def discover_strict_materialized_candidates(
    *,
    dataset: str,
    task: str,
    output_root: Path,
    candidate_ids: Sequence[str],
) -> tuple[str, ...]:
    candidates_root = output_root / f"{dataset}_{task}" / "candidates"
    discovered = []
    for program_id in sorted(set(candidate_ids)):
        if program_id.startswith("_"):
            continue
        artifact_dir = candidates_root / program_id
        if not artifact_dir.is_dir():
            continue
        if not _strict_candidate_dir_valid(
            dataset=dataset,
            task=task,
            program_id=program_id,
            artifact_dir=artifact_dir,
        ):
            continue
        discovered.append(program_id)
    return tuple(discovered)


def _strict_candidate_dir_valid(
    *,
    dataset: str,
    task: str,
    program_id: str,
    artifact_dir: Path,
) -> bool:
    manifest_path = artifact_dir / "materialization_manifest.json"
    if not manifest_path.exists():
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
    safety = inspect_candidate_safety_evidence(
        dataset=dataset,
        task=task,
        program_id=program_id,
        artifact_dir=artifact_dir,
        baseline_program_id="__no_baseline_exemption__",
    )
    return all((
        safety.materializable is True,
        safety.leakage_safe is True,
        safety.temporally_safe is True,
        safety.provenance_complete is True,
    ))


def _materialization_stage_status(
    report: TaskCandidateMaterializationReport,
) -> str:
    if not report.input_resolved:
        return "blocked"
    if report.failed_count:
        return "failed"
    if report.blocked_count:
        return "blocked"
    valid = {"published", "reused", "dry_run_ready"}
    if all(outcome.status in valid for outcome in report.outcomes):
        return "completed"
    return "blocked"


def _materialization_candidate_blockers(
    report: TaskCandidateMaterializationReport,
) -> tuple[str, ...]:
    rows = []
    for outcome in report.outcomes:
        if not outcome.blockers:
            continue
        rows.append(
            f"{outcome.program_id}:{'|'.join(outcome.blockers)}"
        )
    return tuple(rows)


def _run_validation(
    *,
    request: TaskPipelineRequest,
    task_spec,
    discovered: Sequence[str],
    evaluator: CandidateEvaluator,
) -> tuple[EvaluationResult, ...]:
    results = []
    for program_id in sorted(discovered):
        artifact_dir = (
            request.output_root
            / f"{request.dataset}_{request.task}"
            / "candidates"
            / program_id
        )
        for seed in sorted(request.seeds):
            results.append(evaluator.evaluate(EvaluationRequest(
                dataset=request.dataset,
                task=request.task,
                program_id=program_id,
                seed=int(seed),
                artifact_dir=artifact_dir,
                result_root=request.result_root,
                primary_metric=_require_metric(task_spec.primary_metric),
                metric_direction=_require_metric(task_spec.metric_direction),
            )))
    return tuple(results)


def _validation_stage_status(
    results: Sequence[EvaluationResult],
) -> str:
    counts = _validation_result_counts(results)
    if counts["total"] == 0:
        return "blocked"
    if counts["failed"]:
        return "failed"
    if counts["blocked"]:
        return "blocked"
    if counts["completed"] + counts["reused"] == counts["total"]:
        return "completed"
    return "blocked"


def _validation_result_counts(
    results: Sequence[EvaluationResult],
) -> dict[str, int]:
    counts = {
        "completed": 0,
        "reused": 0,
        "blocked": 0,
        "failed": 0,
        "total": len(results),
    }
    for result in results:
        if result.status == "completed":
            if _successful_validation_result_valid(result):
                counts["completed"] += 1
            else:
                counts["failed"] += 1
        elif result.status == "reused":
            if _successful_validation_result_valid(result):
                counts["reused"] += 1
            else:
                counts["failed"] += 1
        elif result.status == "failed":
            counts["failed"] += 1
        elif result.status == "blocked":
            counts["blocked"] += 1
        else:
            counts["blocked"] += 1
    return counts


def _validation_result_blockers(
    results: Sequence[EvaluationResult],
) -> tuple[str, ...]:
    if not results:
        return ("no_evaluation_results",)
    blockers = []
    for result in results:
        reason = _validation_result_rejection_reason(result)
        if not reason:
            continue
        blockers.append(
            f"{result.request.program_id}:seed{result.request.seed}:{reason}"
        )
    return tuple(blockers)


def _validation_result_rejection_reason(
    result: EvaluationResult,
) -> str:
    if result.status == "failed":
        return result.rejection_reason or "evaluation_failed"
    if result.status == "blocked":
        return result.rejection_reason or "evaluation_blocked"
    if result.status in {"completed", "reused"}:
        if not _successful_validation_result_valid(result):
            return "invalid_validation_result"
        return ""
    return result.rejection_reason or f"unexpected_status:{result.status}"


def _successful_validation_result_valid(
    result: EvaluationResult,
) -> bool:
    if result.score is None or not math.isfinite(float(result.score)):
        return False
    n_features = result.n_features
    if isinstance(n_features, bool) or n_features is None:
        return False
    if not isinstance(n_features, int) or n_features < 0:
        return False
    if (
        result.request.dataset == ""
        or result.request.task == ""
        or result.request.program_id == ""
    ):
        return False
    return True


def _write_canonical_export(
    *,
    request: TaskPipelineRequest,
    task_spec,
    discovered: Sequence[str],
    evaluation_results: Sequence[EvaluationResult],
    output_path: Path,
) -> None:
    if _under_paper_tables(output_path):
        raise ValueError("refusing to write under results/paper_tables")
    if (
        output_path.exists()
        and not request.overwrite
        and not request.overwrite_pipeline_output
    ):
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for result in evaluation_results:
        if result.status not in {"completed", "reused"} or result.score is None:
            continue
        seed_rows.append({
            "candidate": result.request.program_id,
            "seed": result.request.seed,
            _require_metric(task_spec.primary_metric): result.score,
            "n_features": result.n_features,
            "evidence_location": result.evidence_location,
        })
    aggregate_rows = _aggregate_seed_rows(
        seed_rows,
        primary_metric=_require_metric(task_spec.primary_metric),
    )
    selected_program_id = _stable_selected_program(
        aggregate_rows=aggregate_rows,
        seed_rows=seed_rows,
        primary_metric=_require_metric(task_spec.primary_metric),
        metric_direction=_require_metric(task_spec.metric_direction),
    )
    safety = {
        program_id: inspect_candidate_safety_evidence(
            dataset=request.dataset,
            task=request.task,
            program_id=program_id,
            artifact_dir=(
                request.output_root
                / f"{request.dataset}_{request.task}"
                / "candidates"
                / program_id
            ),
            baseline_program_id="__no_baseline_exemption__",
        )
        for program_id in discovered
    }
    export_report = build_validation_export_records(
        dataset=request.dataset,
        task=request.task,
        split="validation",
        primary_metric=_require_metric(task_spec.primary_metric),
        metric_direction=_require_metric(task_spec.metric_direction),
        candidate_program_ids=tuple(discovered),
        expected_seeds=request.seeds,
        aggregate_rows=aggregate_rows,
        seed_rows=seed_rows,
        selected_program_id=selected_program_id,
        baseline_program_id="baseline",
        safety_evidence_by_program=safety,
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        write_validation_export_csv(
            export_report.aggregate_records,
            handle,
        )


def _aggregate_seed_rows(
    seed_rows: Sequence[Mapping[str, object]],
    *,
    primary_metric: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in seed_rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    out = []
    for candidate, rows in sorted(grouped.items()):
        scores = [float(row[primary_metric]) for row in rows]
        features = [float(row["n_features"]) for row in rows]
        out.append({
            "candidate": candidate,
            "primary_mean": sum(scores) / len(scores),
            "n_features_mean": sum(features) / len(features),
            "evidence_location": "|".join(sorted(
                str(row["evidence_location"]) for row in rows
            )),
        })
    return out


def _stable_selected_program(
    *,
    aggregate_rows: Sequence[Mapping[str, object]],
    seed_rows: Sequence[Mapping[str, object]],
    primary_metric: str,
    metric_direction: str,
) -> str:
    # Keep this narrow: it mirrors the current sweep gate inputs and delegates
    # the final production decision to the validation-aware selector.
    from scripts.experiments.run_candidate_program_sweep import (
        apply_stability_gate,
    )

    if not aggregate_rows:
        return "baseline"
    summary = pd.DataFrame([
        {
            "candidate": (
                "dfs" if row["candidate"] == "baseline" else row["candidate"]
            ),
            "primary_mean": row["primary_mean"],
            "n_features_mean": row["n_features_mean"],
        }
        for row in aggregate_rows
    ])
    seeds = pd.DataFrame([
        {
            **row,
            "candidate": (
                "dfs" if row["candidate"] == "baseline" else row["candidate"]
            ),
        }
        for row in seed_rows
    ])
    try:
        selected, _ = apply_stability_gate(
            candidate_summary=summary,
            seed_results=seeds,
            primary_metric=primary_metric,
        )
    except ValueError:
        return "baseline"
    selected = str(selected)
    return "baseline" if selected == "dfs" else selected


def _filter_programs(
    programs: Sequence[CandidateProgram],
    *,
    program_ids: Sequence[str],
    exclude_program_ids: Sequence[str],
    baseline_only: bool,
) -> tuple[CandidateProgram, ...]:
    by_id = {program.program_id: program for program in programs}
    unknown = sorted((set(program_ids) | set(exclude_program_ids)) - set(by_id))
    if unknown:
        raise ValueError("unknown candidate program IDs: " + ", ".join(unknown))
    selected = list(programs)
    if baseline_only:
        selected = [program for program in selected if program.program_id == "baseline"]
    if program_ids:
        wanted = set(program_ids)
        selected = [program for program in selected if program.program_id in wanted]
    if exclude_program_ids:
        excluded = set(exclude_program_ids)
        selected = [program for program in selected if program.program_id not in excluded]
    return tuple(sorted(selected, key=lambda item: item.program_id))


def _normalize_mode(request: TaskPipelineRequest) -> str:
    mode = request.mode
    valid = {
        "dry-run",
        "materialize-only",
        "validation-only",
        "selection-only",
        "through-materialization",
        "through-validation",
        "full",
    }
    if mode not in valid:
        raise ValueError(f"unknown pipeline mode: {mode}")
    return mode


def _includes_stage(mode: str, stage: str) -> bool:
    if stage in {"resolve", "compile"}:
        return True
    stages_by_mode = {
        "dry-run": {"materialize", "discover"},
        "materialize-only": {"materialize", "report"},
        "validation-only": {"discover", "validate", "export", "report"},
        "selection-only": {"select", "report"},
        "through-materialization": {"materialize", "discover", "report"},
        "through-validation": {"materialize", "discover", "validate", "export", "report"},
        "full": {"materialize", "discover", "validate", "export", "select", "report"},
    }
    return stage in stages_by_mode[mode]


def _canonical_validation_path(request: TaskPipelineRequest) -> Path:
    return (
        request.result_root
        / f"{request.dataset}_{request.task}"
        / "canonical_validation.csv"
    )


def _pipeline_report_path(request: TaskPipelineRequest) -> Path:
    return (
        request.result_root
        / f"{request.dataset}_{request.task}"
        / "task_pipeline_report.json"
    )


def _write_pipeline_report(
    *,
    request: TaskPipelineRequest,
    report: TaskPipelineReport,
) -> None:
    path = _pipeline_report_path(request)
    if _under_paper_tables(path):
        raise ValueError("refusing to write under results/paper_tables")
    if path.exists() and not request.overwrite_pipeline_output:
        existing = path.read_text(encoding="utf-8")
        existing_hash = hashlib.sha256(existing.encode("utf-8")).hexdigest()
        raise FileExistsError(
            f"{path} exists; existing_sha256={existing_hash}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), sort_keys=True, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def _under_paper_tables(path: Path) -> bool:
    paper = Path("results/paper_tables").resolve()
    resolved = path.resolve()
    return resolved == paper or paper in resolved.parents


def _skipped(stage: str) -> TaskPipelineStageResult:
    return TaskPipelineStageResult(
        stage=stage,
        status="skipped",
        inputs={},
        outputs={},
    )


def _pipeline_status(stages: Sequence[TaskPipelineStageResult]) -> str:
    if any(stage.status == "failed" for stage in stages):
        return "failed"
    if any(stage.status == "blocked" for stage in stages):
        return "blocked"
    return "completed"


def _require_metric(value: str | None) -> str:
    if not value:
        raise ValueError("task primary metric and direction are required")
    return value


def _evaluation_result_to_dict(result: EvaluationResult) -> dict[str, object]:
    payload = asdict(result)
    payload["request"]["artifact_dir"] = str(result.request.artifact_dir)
    payload["request"]["result_root"] = str(result.request.result_root)
    return payload
