from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .candidate_safety import AUDIT_COLUMNS
from .config import load_yaml
from .task_pipeline import EvaluationRequest, EvaluationResult
from .validation_export import inspect_candidate_safety_evidence


@dataclass(frozen=True)
class CandidateEvaluatorConfig:
    reproduction_config: Path = Path("configs/reproduction/tasks.yaml")
    python_executable: Path | None = None
    device: str = "cpu"
    timeout_seconds: int | None = None
    overwrite: bool = False
    evaluator_backend: str = "tabpfn"
    evaluator_version: str = "candidate-evaluator-v1"
    canonicalization_version: str = "canonical-metrics-v1"


@dataclass(frozen=True)
class EvaluationArtifactRecord:
    dataset: str
    task: str
    program_id: str
    seed: int
    split: str
    primary_metric: str
    metric_direction: str
    candidate_artifact_path: str
    result_dir: str
    original_metrics_path: str
    canonical_metrics_path: str
    command: tuple[str, ...]
    return_code: int | None
    status: str
    input_artifact_hashes: Mapping[str, str]
    evaluator_config_hash: str


class ProcessRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: int | None,
    ) -> subprocess.CompletedProcess:
        ...


class SubprocessCandidateEvaluator:
    def __init__(
        self,
        *,
        config: CandidateEvaluatorConfig,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.config = config
        self.process_runner = process_runner or _default_process_runner

    def evaluate(
        self,
        request: EvaluationRequest,
    ) -> EvaluationResult:
        result_dir = _result_dir(request)
        metrics_path = result_dir / "metrics.csv"
        canonical_metrics_path = result_dir / "canonical_validation_metrics.csv"
        manifest_path = result_dir / "evaluation_manifest.json"
        staging = _staging_dir(result_dir)
        staging_metrics = staging / "metrics.csv"
        staging_canonical = staging / "canonical_validation_metrics.csv"
        staging_manifest = staging / "evaluation_manifest.json"
        command: list[str] = []
        try:
            _validate_result_root(request.result_root)
            _validate_candidate_dir(request)
            command = _build_command(
                request=request,
                config=self.config,
                result_dir=staging,
            )
            hashes = _input_hashes(request.artifact_dir)
            script_hash = _file_sha256(Path(command[2]))
            task_config_hash = _file_sha256(self.config.reproduction_config)
            config_hash = _config_hash(self.config, command)
            reused = None
            if not self.config.overwrite:
                reused = load_completed_evaluation(
                    request=request,
                    config=self.config,
                    manifest_path=manifest_path,
                    original_metrics_path=metrics_path,
                    canonical_metrics_path=canonical_metrics_path,
                    command=command,
                    input_hashes=hashes,
                    evaluator_script_hash=script_hash,
                    task_config_hash=task_config_hash,
                    evaluator_config_hash=config_hash,
                )
            if reused is not None:
                score, n_features = _read_metrics(
                    canonical_metrics_path,
                    request=request,
                )
                return EvaluationResult(
                    request=request,
                    status="reused",
                    score=score,
                    n_features=n_features,
                    evidence_location=str(canonical_metrics_path),
                    command=tuple(command),
                    environment=_environment_record(self.config),
                )
            if result_dir.exists() and not self.config.overwrite:
                raise ValueError(
                    f"stale evaluation output exists: {result_dir}"
                )
            result_dir.parent.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            stdout_path = staging / "stdout.log"
            stderr_path = staging / "stderr.log"
            env = _environment(self.config)
            try:
                process = self.process_runner(
                    command,
                    cwd=Path.cwd(),
                    env=env,
                    timeout=self.config.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                stdout_path.write_text(str(exc.stdout or ""), encoding="utf-8")
                stderr_path.write_text(str(exc.stderr or ""), encoding="utf-8")
                _write_manifest(
                    staging_manifest,
                    request=request,
                    command=command,
                    return_code=None,
                    status="failed",
                    original_metrics_path=staging_metrics,
                    canonical_metrics_path=staging_canonical,
                    input_hashes=hashes,
                    evaluator_script_hash=script_hash,
                    task_config_hash=task_config_hash,
                    evaluator_config_hash=config_hash,
                    canonicalization_version=self.config.canonicalization_version,
                    device=self.config.device,
                )
                return EvaluationResult(
                    request=request,
                    status="failed",
                    score=None,
                    n_features=None,
                    evidence_location=str(staging_canonical),
                    rejection_reason="evaluation_timeout",
                    command=tuple(command),
                    environment=_environment_record(self.config),
                )
            stdout_path.write_text(process.stdout or "", encoding="utf-8")
            stderr_path.write_text(process.stderr or "", encoding="utf-8")
            if process.returncode != 0:
                _write_manifest(
                    staging_manifest,
                    request=request,
                    command=command,
                    return_code=process.returncode,
                    status="failed",
                    original_metrics_path=staging_metrics,
                    canonical_metrics_path=staging_canonical,
                    input_hashes=hashes,
                    evaluator_script_hash=script_hash,
                    task_config_hash=task_config_hash,
                    evaluator_config_hash=config_hash,
                    canonicalization_version=self.config.canonicalization_version,
                    device=self.config.device,
                )
                return EvaluationResult(
                    request=request,
                    status="failed",
                    score=None,
                    n_features=None,
                    evidence_location=str(staging_canonical),
                    rejection_reason=f"subprocess_failed:{process.returncode}",
                    command=tuple(command),
                    environment=_environment_record(self.config),
                )
            _write_canonical_metrics(
                original_metrics_path=staging_metrics,
                canonical_metrics_path=staging_canonical,
                request=request,
            )
            score, n_features = _read_metrics(
                staging_canonical,
                request=request,
            )
            _write_manifest(
                staging_manifest,
                request=request,
                command=command,
                return_code=process.returncode,
                status="completed",
                original_metrics_path=metrics_path,
                canonical_metrics_path=canonical_metrics_path,
                input_hashes=hashes,
                evaluator_script_hash=script_hash,
                task_config_hash=task_config_hash,
                evaluator_config_hash=config_hash,
                canonicalization_version=self.config.canonicalization_version,
                device=self.config.device,
            )
            _publish_evaluation_dir(
                staging,
                result_dir,
                overwrite=self.config.overwrite,
            )
            return EvaluationResult(
                request=request,
                status="completed",
                score=score,
                n_features=n_features,
                evidence_location=str(canonical_metrics_path),
                command=tuple(command),
                environment=_environment_record(self.config),
            )
        except Exception as exc:
            return EvaluationResult(
                request=request,
                status="failed",
                score=None,
                n_features=None,
                evidence_location=str(metrics_path),
                rejection_reason=str(exc),
                command=tuple(command),
                environment=_environment_record(self.config),
            )


def load_completed_evaluation(
    *,
    request: EvaluationRequest,
    config: CandidateEvaluatorConfig,
    manifest_path: Path,
    original_metrics_path: Path,
    canonical_metrics_path: Path,
    command: Sequence[str],
    input_hashes: Mapping[str, str],
    evaluator_script_hash: str,
    task_config_hash: str,
    evaluator_config_hash: str,
) -> EvaluationArtifactRecord | None:
    if (
        not manifest_path.exists()
        or not original_metrics_path.exists()
        or not canonical_metrics_path.exists()
    ):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": request.dataset,
        "task": request.task,
        "program_id": request.program_id,
        "seed": request.seed,
        "split": "validation",
        "primary_metric": request.primary_metric,
        "metric_direction": request.metric_direction,
        "candidate_artifact_path": str(request.artifact_dir),
        "original_metrics_path": str(original_metrics_path),
        "canonical_metrics_path": str(canonical_metrics_path),
        "command": list(command),
        "status": "completed",
        "input_artifact_hashes": dict(input_hashes),
        "evaluator_script_hash": evaluator_script_hash,
        "task_config_hash": task_config_hash,
        "evaluator_config_hash": evaluator_config_hash,
        "canonicalization_version": config.canonicalization_version,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return None
    return EvaluationArtifactRecord(
        dataset=request.dataset,
        task=request.task,
        program_id=request.program_id,
        seed=request.seed,
        split="validation",
        primary_metric=request.primary_metric,
        metric_direction=request.metric_direction,
        candidate_artifact_path=str(request.artifact_dir),
        result_dir=str(manifest_path.parent),
        original_metrics_path=str(original_metrics_path),
        canonical_metrics_path=str(canonical_metrics_path),
        command=tuple(command),
        return_code=int(manifest["return_code"]),
        status="completed",
        input_artifact_hashes=input_hashes,
        evaluator_config_hash=evaluator_config_hash,
    )


def _build_command(
    *,
    request: EvaluationRequest,
    config: CandidateEvaluatorConfig,
    result_dir: Path,
) -> list[str]:
    task_config = _task_config(
        config.reproduction_config,
        request.dataset,
        request.task,
    )
    evaluator = _evaluator_script(
        task_config["problem_type"],
        backend=config.evaluator_backend,
    )
    command = [
        str(config.python_executable or sys.executable),
        "-u",
        evaluator,
        "--train-parquet",
        str(request.artifact_dir / "target_with_dfs_agg_train.parquet"),
        "--val-parquet",
        str(request.artifact_dir / "target_with_dfs_agg_val.parquet"),
        "--output-dir",
        str(result_dir),
        "--dataset",
        request.dataset,
        "--task",
        request.task,
        "--variant",
        _variant_alias(request.program_id),
        "--label-col",
        str(task_config["label_col"]),
        "--seed",
        str(request.seed),
    ]
    drop_cols = ",".join(task_config.get("evaluation", {}).get("drop_cols", ()))
    if drop_cols:
        command.extend(["--drop-cols", drop_cols])
    if config.evaluator_backend != "catboost":
        command.extend(["--device", config.device])
    return command


def _validate_candidate_dir(request: EvaluationRequest) -> None:
    if request.program_id.startswith("_"):
        raise ValueError("private candidate directories are not evaluable")
    manifest_path = request.artifact_dir / "materialization_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset") != request.dataset
        or manifest.get("task") != request.task
        or manifest.get("program_id") != request.program_id
        or manifest.get("materialization_status") != "success"
    ):
        raise ValueError("candidate materialization manifest identity mismatch")
    for name in (
        "target_with_dfs_agg_train.parquet",
        "target_with_dfs_agg_val.parquet",
    ):
        if not (request.artifact_dir / name).exists():
            raise FileNotFoundError(request.artifact_dir / name)
    safety = inspect_candidate_safety_evidence(
        dataset=request.dataset,
        task=request.task,
        program_id=request.program_id,
        artifact_dir=request.artifact_dir,
        baseline_program_id="__no_baseline_exemption__",
    )
    if not all((
        safety.materializable is True,
        safety.leakage_safe is True,
        safety.temporally_safe is True,
        safety.provenance_complete is True,
    )):
        raise ValueError("candidate safety audits are incomplete or failed")


def _read_metrics(
    metrics_path: Path,
    *,
    request: EvaluationRequest,
) -> tuple[float, int]:
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one metric row in {metrics_path}")
    row = rows[0]
    for column in ("dataset", "task", "program_id", "split", "seed"):
        if column not in row:
            raise ValueError(f"metrics.csv missing required column {column}")
    if row["dataset"] != request.dataset or row["task"] != request.task:
        raise ValueError("metrics dataset/task mismatch")
    if _program_alias(row["program_id"]) != request.program_id:
        raise ValueError("metrics program_id mismatch")
    if int(row["seed"]) != request.seed:
        raise ValueError("metrics seed mismatch")
    if row["split"].strip().lower() != "validation":
        raise ValueError("metrics split is not validation")
    if request.primary_metric not in row:
        raise ValueError(
            f"metrics missing primary metric {request.primary_metric}"
        )
    score = float(row[request.primary_metric])
    if not math.isfinite(score):
        raise ValueError("primary metric is not finite")
    if "n_features" not in row:
        raise ValueError("metrics missing n_features")
    n_features = _validate_n_features(row["n_features"])
    return score, n_features


def _write_canonical_metrics(
    *,
    original_metrics_path: Path,
    canonical_metrics_path: Path,
    request: EvaluationRequest,
) -> None:
    if not original_metrics_path.exists():
        raise FileNotFoundError(original_metrics_path)
    with original_metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one evaluator metric row in {original_metrics_path}"
        )
    row = dict(rows[0])
    if row.get("dataset") != request.dataset or row.get("task") != request.task:
        raise ValueError("original metrics dataset/task mismatch")
    if _program_alias(row.get("variant", row.get("program_id", ""))) != request.program_id:
        raise ValueError("original metrics program/variant mismatch")
    if int(row.get("seed", "-1")) != request.seed:
        raise ValueError("original metrics seed mismatch")
    if "split" in row and row["split"].strip().lower() not in {
        "",
        "val",
        "validation",
    }:
        raise ValueError("original metrics split is not validation")
    row["program_id"] = request.program_id
    row["split"] = "validation"
    ordered = []
    for name in (
        "dataset",
        "task",
        "program_id",
        "split",
        "seed",
    ):
        if name not in ordered:
            ordered.append(name)
    for name in original_fieldnames:
        if name not in ordered and name != "variant":
            ordered.append(name)
    if request.primary_metric not in row:
        raise ValueError(
            f"original metrics missing primary metric {request.primary_metric}"
        )
    _validate_n_features(row.get("n_features"))
    score = float(row[request.primary_metric])
    if not math.isfinite(score):
        raise ValueError("primary metric is not finite")
    with canonical_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, lineterminator="\n")
        writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in ordered})


def _validate_n_features(value: object) -> int:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed != int(parsed):
        raise ValueError("n_features must be a nonnegative integer")
    return int(parsed)


def _write_manifest(
    path: Path,
    *,
    request: EvaluationRequest,
    command: Sequence[str],
    return_code: int | None,
    status: str,
    original_metrics_path: Path,
    canonical_metrics_path: Path,
    input_hashes: Mapping[str, str],
    evaluator_script_hash: str,
    task_config_hash: str,
    evaluator_config_hash: str,
    canonicalization_version: str,
    device: str,
) -> None:
    payload = {
        "dataset": request.dataset,
        "task": request.task,
        "program_id": request.program_id,
        "seed": request.seed,
        "split": "validation",
        "primary_metric": request.primary_metric,
        "metric_direction": request.metric_direction,
        "candidate_artifact_path": str(request.artifact_dir),
        "legacy_variant_alias": _variant_alias(request.program_id),
        "command": list(command),
        "return_code": return_code,
        "device": device,
        "status": status,
        "original_metrics_path": str(original_metrics_path),
        "canonical_metrics_path": str(canonical_metrics_path),
        "authoritative_metrics_path": str(canonical_metrics_path),
        "input_artifact_hashes": dict(input_hashes),
        "evaluator_script_hash": evaluator_script_hash,
        "task_config_hash": task_config_hash,
        "evaluator_config_hash": evaluator_config_hash,
        "canonicalization_version": canonicalization_version,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _input_hashes(artifact_dir: Path) -> dict[str, str]:
    names = (
        "target_with_dfs_agg_train.parquet",
        "target_with_dfs_agg_val.parquet",
        "materialization_manifest.json",
        "temporal_safety_audit.csv",
        "leakage_safety_audit.csv",
        "lowering_provenance_audit.csv",
    )
    return {
        name: _file_sha256(artifact_dir / name)
        for name in names
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(
    config: CandidateEvaluatorConfig,
    command: Sequence[str],
) -> str:
    payload = {
        "device": config.device,
        "timeout_seconds": config.timeout_seconds,
        "evaluator_backend": config.evaluator_backend,
        "evaluator_version": config.evaluator_version,
        "canonicalization_version": config.canonicalization_version,
        "python_executable": str(config.python_executable or sys.executable),
        "command": list(command),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _default_process_runner(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int | None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(env),
        timeout=timeout,
        check=False,
        text=True,
        capture_output=True,
    )


def _environment(config: CandidateEvaluatorConfig) -> dict[str, str]:
    env = dict(os.environ)
    env["FDHG_EVALUATION_SPLIT"] = "validation"
    env["FDHG_EVALUATOR_DEVICE"] = config.device
    return env


def _environment_record(config: CandidateEvaluatorConfig) -> tuple[str, ...]:
    return (
        "FDHG_EVALUATION_SPLIT=validation",
        f"FDHG_EVALUATOR_DEVICE={config.device}",
    )


def _task_config(
    path: Path,
    dataset: str,
    task: str,
) -> Mapping[str, object]:
    config = load_yaml(path)
    tasks = config.get("tasks", config)
    key = f"{dataset}/{task}"
    if key not in tasks:
        raise KeyError(f"task {key!r} missing from {path}")
    return tasks[key]


def _evaluator_script(problem_type: str, *, backend: str) -> str:
    if problem_type == "binary":
        return "scripts/evaluate/evaluate_binary_tabpfn.py"
    if problem_type == "regression":
        return "scripts/evaluate/evaluate_regression_tabpfn.py"
    if problem_type == "multiclass":
        if backend == "catboost":
            return "scripts/evaluate/evaluate_multiclass_catboost.py"
        return "scripts/evaluate/evaluate_multiclass_tabpfn.py"
    raise ValueError(f"unsupported problem_type {problem_type!r}")


def _variant_alias(program_id: str) -> str:
    return "dfs" if program_id == "baseline" else program_id


def _program_alias(program_id: str) -> str:
    return "baseline" if program_id == "dfs" else program_id


def _result_dir(request: EvaluationRequest) -> Path:
    return (
        request.result_root
        / f"{request.dataset}_{request.task}"
        / "evaluations"
        / request.program_id
        / f"seed{request.seed}"
    )


def _validate_result_root(result_root: Path) -> None:
    paper = Path("results/paper_tables").resolve()
    resolved = result_root.resolve()
    if resolved == paper or paper in resolved.parents:
        raise ValueError("refusing to write under results/paper_tables")


def _publish_evaluation_dir(
    staging: Path,
    result_dir: Path,
    *,
    overwrite: bool,
) -> None:
    backup: Path | None = None
    if result_dir.exists():
        if not overwrite:
            raise FileExistsError(result_dir)
        backup = result_dir.parent / f"_{result_dir.name}.backup"
        if backup.exists():
            shutil.rmtree(backup)
        result_dir.replace(backup)
    try:
        staging.replace(result_dir)
    except OSError:
        if backup is not None and backup.exists():
            backup.replace(result_dir)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _staging_dir(result_dir: Path) -> Path:
    return result_dir.parent / f"_{result_dir.name}.staging"
