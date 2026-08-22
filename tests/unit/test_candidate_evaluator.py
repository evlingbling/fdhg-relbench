from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

import fdhg.compiler.candidate_evaluator as ce
from fdhg.compiler.candidate_evaluator import (
    CandidateEvaluatorConfig,
    SubprocessCandidateEvaluator,
)
from fdhg.compiler.task_pipeline import EvaluationRequest


def write_candidate(
    tmp_path: Path,
    *,
    dataset: str = "rel-example",
    task: str = "pairwise",
    program_id: str = "baseline",
    safety_passed: bool = True,
) -> tuple[Path, Path]:
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    pd.DataFrame([{"label": 1, "f": 1}]).to_parquet(
        artifact / "target_with_dfs_agg_train.parquet",
        index=False,
    )
    pd.DataFrame([{"label": 0, "f": 2}]).to_parquet(
        artifact / "target_with_dfs_agg_val.parquet",
        index=False,
    )
    (artifact / "materialization_manifest.json").write_text(
        json.dumps({
            "dataset": dataset,
            "task": task,
            "program_id": program_id,
            "materialization_status": "success",
        }),
        encoding="utf-8",
    )
    for name, audit_type in [
        ("temporal_safety_audit.csv", "temporal_safety"),
        ("leakage_safety_audit.csv", "leakage_safety"),
        ("lowering_provenance_audit.csv", "lowering_provenance"),
    ]:
        (artifact / name).write_text(
            "dataset,task,program_id,audit_type,primitive_id,status,passed,source_table,source_column,output_column,rejection_reason,evidence_location,notes\n"
            f"{dataset},{task},{program_id},{audit_type},p,ok,{str(safety_passed).lower()},,,,,audit,\n",
            encoding="utf-8",
        )
    config = tmp_path / "tasks.yaml"
    config.write_text(
        yaml.safe_dump({
            "tasks": {
                f"{dataset}/{task}": {
                    "problem_type": "binary",
                    "label_col": "label",
                    "evaluation": {"drop_cols": ["label"]},
                }
            }
        }),
        encoding="utf-8",
    )
    return artifact, config


def request(tmp_path: Path, artifact: Path, *, program_id: str = "baseline"):
    return EvaluationRequest(
        dataset="rel-example",
        task="pairwise",
        program_id=program_id,
        seed=41,
        artifact_dir=artifact,
        result_root=tmp_path / "results",
        primary_metric="roc_auc",
        metric_direction="higher",
    )


def runner_write_metrics(score=0.75, *, split=None, variant="dfs", n_features="4"):
    calls = []

    def run(argv, *, cwd, env, timeout):
        calls.append((list(argv), cwd, dict(env), timeout))
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        seed = argv[argv.index("--seed") + 1]
        out_dir.mkdir(parents=True, exist_ok=True)
        split_columns = ",split" if split is not None else ""
        split_values = f",{split}" if split is not None else ""
        (out_dir / "metrics.csv").write_text(
            f"dataset,task,variant,seed,roc_auc,n_features{split_columns}\n"
            f"rel-example,pairwise,{variant},{seed},{score},{n_features}{split_values}\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="ok",
            stderr="",
        )

    run.calls = calls
    return run


def evaluator(config_path: Path, runner, **kwargs):
    return SubprocessCandidateEvaluator(
        config=CandidateEvaluatorConfig(
            reproduction_config=config_path,
            python_executable=Path("/usr/bin/python3"),
            **kwargs,
        ),
        process_runner=runner,
    )


def write_task_config(
    tmp_path: Path,
    *,
    problem_type: str,
    drop_cols: tuple[str, ...] = ("label",),
) -> Path:
    config = tmp_path / f"{problem_type}.yaml"
    config.write_text(
        yaml.safe_dump({
            "tasks": {
                "rel-example/pairwise": {
                    "problem_type": problem_type,
                    "label_col": "label",
                    "evaluation": {"drop_cols": list(drop_cols)},
                }
            }
        }),
        encoding="utf-8",
    )
    return config


def test_deterministic_command_construction_and_no_shell(
    tmp_path: Path,
) -> None:
    artifact, config = write_candidate(tmp_path)
    runner = runner_write_metrics()

    result = evaluator(config, runner).evaluate(request(tmp_path, artifact))

    argv = result.command
    assert argv[:3] == (
        "/usr/bin/python3",
        "-u",
        "scripts/evaluate/evaluate_binary_tabpfn.py",
    )
    assert "--device" in argv
    assert "cpu" in argv
    assert not isinstance(argv, str)


def test_constructed_commands_match_supported_problem_types(tmp_path: Path) -> None:
    artifact, _ = write_candidate(tmp_path)
    req = request(tmp_path, artifact)

    regression = ce._build_command(
        request=req,
        config=CandidateEvaluatorConfig(
            reproduction_config=write_task_config(tmp_path, problem_type="regression"),
            python_executable=Path("/usr/bin/python3"),
        ),
        result_dir=tmp_path / "regression",
    )
    assert regression[:3] == [
        "/usr/bin/python3",
        "-u",
        "scripts/evaluate/evaluate_regression_tabpfn.py",
    ]
    assert "--device" in regression

    multiclass = ce._build_command(
        request=req,
        config=CandidateEvaluatorConfig(
            reproduction_config=write_task_config(tmp_path, problem_type="multiclass"),
            python_executable=Path("/usr/bin/python3"),
        ),
        result_dir=tmp_path / "multiclass",
    )
    assert multiclass[2] == "scripts/evaluate/evaluate_multiclass_tabpfn.py"
    assert "--device" in multiclass

    catboost = ce._build_command(
        request=req,
        config=CandidateEvaluatorConfig(
            reproduction_config=write_task_config(tmp_path, problem_type="multiclass"),
            python_executable=Path("/usr/bin/python3"),
            evaluator_backend="catboost",
        ),
        result_dir=tmp_path / "catboost",
    )
    assert catboost[2] == "scripts/evaluate/evaluate_multiclass_catboost.py"
    assert "--device" not in catboost


def test_cuda_device_forwarding(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    result = evaluator(
        config,
        runner_write_metrics(),
        device="cuda:0",
    ).evaluate(request(tmp_path, artifact))

    assert result.command[-1] == "cuda:0"


def test_optional_drop_cols_omitted_when_empty(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    config.write_text(
        yaml.safe_dump({
            "tasks": {
                "rel-example/pairwise": {
                    "problem_type": "binary",
                    "label_col": "label",
                    "evaluation": {"drop_cols": []},
                }
            }
        }),
        encoding="utf-8",
    )

    result = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "completed"
    assert "--drop-cols" not in result.command


def test_strict_candidate_identity_validation(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path, dataset="rel-other")

    result = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "failed"
    assert "identity mismatch" in result.rejection_reason


def test_missing_candidate_artifact(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    (artifact / "target_with_dfs_agg_val.parquet").unlink()

    result = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "failed"
    assert "target_with_dfs_agg_val.parquet" in result.rejection_reason


def test_failed_safety_audit(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path, safety_passed=False)

    result = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "failed"
    assert "safety" in result.rejection_reason


def test_validation_split_only_and_test_metric_rejection(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)

    result = evaluator(
        config,
        runner_write_metrics(split="test"),
    ).evaluate(request(tmp_path, artifact))

    assert result.status == "failed"
    assert "split" in result.rejection_reason


def test_completed_run_reuse(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    first = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )
    assert first.status == "completed"

    def fail(*args, **kwargs):
        raise AssertionError("should reuse")

    second = evaluator(config, fail).evaluate(request(tmp_path, artifact))

    assert second.status == "reused"
    assert second.score == first.score


def test_stale_manifest_and_changed_hash_rejected(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    ev = evaluator(config, runner_write_metrics())
    assert ev.evaluate(request(tmp_path, artifact)).status == "completed"
    (artifact / "materialization_manifest.json").write_text(
        json.dumps({
            "dataset": "rel-example",
            "task": "pairwise",
            "program_id": "baseline",
            "materialization_status": "success",
            "changed": True,
        }),
        encoding="utf-8",
    )

    result = ev.evaluate(request(tmp_path, artifact))

    assert result.status == "failed"
    assert "stale evaluation output" in result.rejection_reason


def test_changed_evaluator_config_rejected(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    assert evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    ).status == "completed"

    result = evaluator(
        config,
        runner_write_metrics(),
        device="cuda",
    ).evaluate(request(tmp_path, artifact))

    assert result.status == "failed"
    assert "stale evaluation output" in result.rejection_reason


def test_changed_task_config_hash_rejected(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    assert evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    ).status == "completed"
    config.write_text(
        yaml.safe_dump({
            "tasks": {
                "rel-example/pairwise": {
                    "problem_type": "binary",
                    "label_col": "label",
                    "evaluation": {"drop_cols": ["label"], "unused": "changed"},
                }
            }
        }),
        encoding="utf-8",
    )

    result = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "failed"
    assert "stale evaluation output" in result.rejection_reason


def test_evaluator_script_hash_in_reuse_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, config = write_candidate(tmp_path)
    script = tmp_path / "evaluate.py"
    script.write_text("print('v1')\n", encoding="utf-8")
    monkeypatch.setattr(ce, "_evaluator_script", lambda problem_type, *, backend: str(script))
    assert evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    ).status == "completed"
    script.write_text("print('v2')\n", encoding="utf-8")

    result = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "failed"
    assert "stale evaluation output" in result.rejection_reason


def test_missing_primary_metric_nonfinite_and_duplicate_rows(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)

    def bad(argv, *, cwd, env, timeout):
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.csv").write_text(
            "dataset,task,variant,seed,n_features\n"
            "rel-example,pairwise,dfs,41,4\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    result = evaluator(config, bad).evaluate(request(tmp_path, artifact))

    assert result.status == "failed"
    assert "primary metric" in result.rejection_reason

    def nonfinite(argv, *, cwd, env, timeout):
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.csv").write_text(
            "dataset,task,variant,seed,roc_auc,n_features\n"
            "rel-example,pairwise,dfs,41,inf,4\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    result = evaluator(
        config,
        nonfinite,
        overwrite=True,
    ).evaluate(request(tmp_path, artifact))
    assert "finite" in result.rejection_reason

    def duplicate(argv, *, cwd, env, timeout):
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.csv").write_text(
            "dataset,task,variant,seed,roc_auc,n_features\n"
            "rel-example,pairwise,dfs,41,0.7,4\n"
            "rel-example,pairwise,dfs,41,0.7,4\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    result = evaluator(
        config,
        duplicate,
        overwrite=True,
    ).evaluate(request(tmp_path, artifact))
    assert "exactly one evaluator metric row" in result.rejection_reason


def test_failed_subprocess_timeout_stdout_stderr(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)

    def fail(argv, *, cwd, env, timeout):
        return subprocess.CompletedProcess(
            argv,
            2,
            stdout="out",
            stderr="err",
        )

    result = evaluator(config, fail).evaluate(request(tmp_path, artifact))
    assert result.status == "failed"
    assert "subprocess_failed:2" in result.rejection_reason
    final_dir = (
        tmp_path
        / "results"
        / "rel-example_pairwise"
        / "evaluations"
        / "baseline"
        / "seed41"
    )
    staging_dir = final_dir.parent / "_seed41.staging"
    assert not final_dir.exists()
    assert (staging_dir / "stdout.log").read_text(encoding="utf-8") == "out"
    assert (staging_dir / "stderr.log").read_text(encoding="utf-8") == "err"

    def timeout(argv, *, cwd, env, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    result = evaluator(
        config,
        timeout,
        overwrite=True,
        timeout_seconds=1,
    ).evaluate(request(tmp_path, artifact))
    assert result.rejection_reason == "evaluation_timeout"


def test_failed_overwrite_preserves_prior_completed_evaluation(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    assert evaluator(config, runner_write_metrics(score=0.75)).evaluate(
        request(tmp_path, artifact)
    ).status == "completed"
    result_dir = (
        tmp_path
        / "results"
        / "rel-example_pairwise"
        / "evaluations"
        / "baseline"
        / "seed41"
    )
    prior = (result_dir / "canonical_validation_metrics.csv").read_text(
        encoding="utf-8"
    )

    def fail(argv, *, cwd, env, timeout):
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.csv").write_text(
            "dataset,task,variant,seed,roc_auc,n_features\n"
            "rel-example,pairwise,dfs,41,0.10,4\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 5, stdout="", stderr="err")

    result = evaluator(config, fail, overwrite=True).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "failed"
    assert (result_dir / "canonical_validation_metrics.csv").read_text(
        encoding="utf-8"
    ) == prior


def test_baseline_dfs_alias_and_no_paper_table_writes(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)

    result = evaluator(config, runner_write_metrics(variant="dfs")).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "completed"

    bad_request = EvaluationRequest(
        dataset="rel-example",
        task="pairwise",
        program_id="baseline",
        seed=41,
        artifact_dir=artifact,
        result_root=Path("results/paper_tables"),
        primary_metric="roc_auc",
        metric_direction="higher",
    )
    result = evaluator(config, runner_write_metrics()).evaluate(bad_request)
    assert "results/paper_tables" in result.rejection_reason


def test_no_network_or_shell_access(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)
    runner = runner_write_metrics()

    result = evaluator(config, runner).evaluate(request(tmp_path, artifact))

    argv, _, env, _ = runner.calls[0]
    assert result.status == "completed"
    assert isinstance(argv, list)
    assert env["FDHG_EVALUATION_SPLIT"] == "validation"


def test_original_metrics_preserved_and_canonical_metrics_written(
    tmp_path: Path,
) -> None:
    artifact, config = write_candidate(tmp_path)

    result = evaluator(config, runner_write_metrics()).evaluate(
        request(tmp_path, artifact)
    )

    result_dir = (
        tmp_path
        / "results"
        / "rel-example_pairwise"
        / "evaluations"
        / "baseline"
        / "seed41"
    )
    original = (result_dir / "metrics.csv").read_text(encoding="utf-8")
    canonical = (result_dir / "canonical_validation_metrics.csv").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (result_dir / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert result.status == "completed"
    assert "variant" in original
    assert "program_id" in canonical
    assert ",validation," in canonical
    assert manifest["original_metrics_path"] == str(result_dir / "metrics.csv")
    assert manifest["canonical_metrics_path"] == str(
        result_dir / "canonical_validation_metrics.csv"
    )
    assert manifest["authoritative_metrics_path"] == str(
        result_dir / "canonical_validation_metrics.csv"
    )


@pytest.mark.parametrize("value", ["4", "4.0"])
def test_integer_valued_n_features_accepted(
    tmp_path: Path,
    value: str,
) -> None:
    artifact, config = write_candidate(tmp_path)

    result = evaluator(config, runner_write_metrics(n_features=value)).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "completed"
    assert result.n_features == 4


def test_fractional_n_features_rejected(tmp_path: Path) -> None:
    artifact, config = write_candidate(tmp_path)

    result = evaluator(config, runner_write_metrics(n_features="4.5")).evaluate(
        request(tmp_path, artifact)
    )

    assert result.status == "failed"
    assert "n_features" in result.rejection_reason
