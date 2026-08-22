from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fdhg.compiler.materializer import (
    TaskCandidateMaterializationRequest,
    materialize_task_candidates,
)
from fdhg.compiler.task_pipeline import (
    EvaluationResult,
    TaskPipelineRequest,
    discover_strict_materialized_candidates,
    run_task_pipeline,
)
import fdhg.onboarding.pipeline as onboarding_pipeline
from fdhg.onboarding.pipeline import BASELINE_AUTO, onboard_dataset
from tests.unit.onboarding_fixtures import write_onboarding_fixture


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, request):
        self.calls.append(request)
        return EvaluationResult(
            request=request,
            status="completed",
            score=12.5,
            n_features=6,
            evidence_location=f"fake:{request.program_id}:{request.seed}",
            command=("fake",),
            environment=("cpu",),
        )


def _onboarded(tmp_path: Path):
    config_path = write_onboarding_fixture(tmp_path)
    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "onboarding",
        write=True,
    )
    assert report.status == "completed"
    return config_path, report


def test_baseline_auto_custom_candidate_is_generated_from_primitives(
    tmp_path: Path,
) -> None:
    _, report = _onboarded(tmp_path)
    manifest = json.loads(
        (report.output_dir / "onboarding_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    program = manifest["candidate_programs"][0]
    assert program["program_id"] == BASELINE_AUTO
    assert program["primitive_ids"] == [
        "baseline::count",
        "baseline::numeric_mean",
        "baseline::numeric_std",
        "baseline::numeric_min",
        "baseline::numeric_max",
        "baseline::days_since_last",
        "baseline::history::past_unique_values",
    ]


def test_dry_run_does_not_materialize_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_onboarding_fixture(tmp_path)

    def fail_materialization(*args, **kwargs):
        raise AssertionError("feature materialization called during dry-run")

    monkeypatch.setattr(
        onboarding_pipeline,
        "_apply_features_optimized",
        fail_materialization,
    )

    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "onboarding",
        write=False,
    )

    assert report.status == "dry_run_ready"
    assert "f_events_count" in report.planned_feature_columns
    assert report.workload["materialization_strategy"] == "grouped_temporal_sweep"
    assert report.workload["materialization_executed"] is False
    assert not report.output_dir.exists()


def test_task_materialization_and_strict_discovery_succeed(
    tmp_path: Path,
) -> None:
    _, onboarded = _onboarded(tmp_path)
    reproduction = onboarded.output_dir / "resolved_task_spec.yaml"

    report = materialize_task_candidates(
        TaskCandidateMaterializationRequest(
            dataset="example-commerce",
            task="user-spend",
            output_root=tmp_path / "compiler",
            reproduction_config=reproduction,
            semantics_config=tmp_path / "missing.yaml",
            program_ids=(BASELINE_AUTO,),
            write=True,
        )
    )

    assert report.published_count == 1
    discovered = discover_strict_materialized_candidates(
        dataset="example-commerce",
        task="user-spend",
        output_root=tmp_path / "compiler",
        candidate_ids=(BASELINE_AUTO,),
    )
    assert discovered == (BASELINE_AUTO,)


def test_end_to_end_synthetic_pipeline_with_fake_evaluator_and_reuse(
    tmp_path: Path,
) -> None:
    config_path, onboarded = _onboarded(tmp_path)
    reproduction = onboarded.output_dir / "resolved_task_spec.yaml"
    evaluator = FakeEvaluator()
    request = TaskPipelineRequest(
        dataset="example-commerce",
        task="user-spend",
        output_root=tmp_path / "compiler",
        result_root=tmp_path / "results",
        seeds=(41,),
        program_ids=(BASELINE_AUTO,),
        mode="through-validation",
        write_materialization=True,
        run_validation=True,
        reproduction_config=reproduction,
        semantics_config=tmp_path / "missing.yaml",
    )

    report = run_task_pipeline(request, evaluator=evaluator)

    assert report.pipeline_status == "completed"
    assert evaluator.calls and evaluator.calls[0].program_id == BASELINE_AUTO
    canonical = (
        tmp_path
        / "results"
        / "example-commerce_user-spend"
        / "canonical_validation.csv"
    )
    rows = pd.read_csv(canonical)
    assert rows.loc[0, "split"] == "validation"
    assert rows.loc[0, "program_id"] == BASELINE_AUTO
    assert rows.loc[0, "score"] == 12.5

    reused = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "onboarding",
        write=True,
    )
    assert reused.status == "reused"


def test_existing_ratebeer_prepared_task_remains_unchanged() -> None:
    from fdhg.compiler.config import load_task_spec
    from fdhg.compiler.planner import build_candidate_program
    from fdhg.compiler.programs import build_declared_candidates

    spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-count",
        reproduction_config=Path(
            "tests/fixtures/configs/task_pipeline_smoke_server.yaml"
        ),
        semantics_config=Path("configs/reproduction/task_semantics.yaml"),
    )
    compiled = build_candidate_program(spec)
    declared = build_declared_candidates(
        compiled,
        declarations=[{
            "program_id": "baseline_corrected_canonical",
            "primitive_ids": [
                "baseline::count",
                "baseline::numeric_mean",
                "baseline::numeric_std",
                "baseline::numeric_max",
                "baseline::days_since_last",
            ],
        }],
    )

    assert declared[0].program_id == "baseline_corrected_canonical"
    assert len(declared[0].primitive_ids) == 5
