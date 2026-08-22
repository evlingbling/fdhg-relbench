from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from fdhg.compiler.candidate_evaluator import (
    CandidateEvaluatorConfig,
    SubprocessCandidateEvaluator,
)
from fdhg.compiler.materializer import (
    TaskCandidateMaterializationOutcome,
    TaskCandidateMaterializationReport,
)
from fdhg.compiler.selection import (
    CandidateValidationResult,
    CandidateSelectionPolicy,
    select_candidate_program,
)
from fdhg.compiler.task_pipeline import (
    EvaluationResult,
    TaskPipelineRequest,
    discover_strict_materialized_candidates,
    run_task_pipeline,
)
from fdhg.compiler.validation_export import write_validation_export_csv
from tests.unit.test_task_candidate_materialization import write_task_fixture


class FakeEvaluator:
    def __init__(self, scores, *, fail=None):
        self.scores = scores
        self.fail = set(fail or ())
        self.calls = []

    def evaluate(self, request):
        self.calls.append(request)
        key = (request.program_id, request.seed)
        if key in self.fail:
            return EvaluationResult(
                request=request,
                status="failed",
                score=None,
                n_features=None,
                evidence_location=f"fake:{request.program_id}:{request.seed}",
                rejection_reason=f"failed:{request.program_id}:{request.seed}",
                command=("fake-evaluator",),
                environment=("cpu",),
            )
        return EvaluationResult(
            request=request,
            status="completed",
            score=self.scores[key],
            n_features=2 if request.program_id == "baseline" else 4,
            evidence_location=f"fake:{request.program_id}:{request.seed}",
            command=("fake-evaluator",),
            environment=("cpu",),
        )


class ScriptedEvaluator:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def evaluate(self, request):
        self.calls.append(request)
        status, score, n_features, reason = self.results[
            (request.program_id, request.seed)
        ]
        return EvaluationResult(
            request=request,
            status=status,
            score=score,
            n_features=n_features,
            evidence_location=f"scripted:{request.program_id}:{request.seed}",
            rejection_reason=reason,
            command=("scripted",),
            environment=("cpu",),
        )


def fixture_with_metric(tmp_path: Path):
    reproduction, semantics = write_task_fixture(tmp_path)
    raw = yaml.safe_load(semantics.read_text(encoding="utf-8"))
    raw["rel-example/pairwise"]["primary_metric"] = "roc_auc"
    raw["rel-example/pairwise"]["metric_direction"] = "higher"
    semantics.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return reproduction, semantics


def request(tmp_path: Path, reproduction: Path, semantics: Path, **kwargs):
    seeds = kwargs.pop("seeds", (41, 42, 43, 44))
    return TaskPipelineRequest(
        dataset="rel-example",
        task="pairwise",
        output_root=tmp_path / "outputs" / "e2e",
        result_root=tmp_path / "results" / "compiler",
        seeds=seeds,
        reproduction_config=reproduction,
        semantics_config=semantics,
        **kwargs,
    )


def good_scores():
    scores = {}
    for seed in (41, 42, 43, 44):
        scores[("baseline", seed)] = 0.70
        scores[("baseline_plus_pair_left_temporal", seed)] = 0.76
        scores[("baseline_plus_pairwise_temporal", seed)] = 0.72
    return scores


def materialize_for_validation(
    tmp_path: Path,
    reproduction: Path,
    semantics: Path,
    *,
    program_ids=("baseline",),
) -> None:
    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            program_ids=program_ids,
        )
    )
    assert report.pipeline_status == "completed"


def stage(report, name: str):
    return [item for item in report.stages if item.stage == name][0]


def test_default_dry_run_performs_no_writes(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    report = run_task_pipeline(request(tmp_path, reproduction, semantics))

    assert report.requested_mode == "dry-run"
    assert report.pipeline_status == "blocked"
    assert not (tmp_path / "outputs").exists()
    assert [stage.stage for stage in report.stages] == [
        "resolve",
        "compile",
        "materialize",
        "discover",
        "validate",
        "export",
        "select",
    ]


def test_materialization_only(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            program_ids=("baseline",),
        )
    )

    assert report.pipeline_status == "completed"
    assert report.stages[2].stage == "materialize"
    assert report.stages[2].outputs["published"] == 1


def test_input_resolved_blocked_candidate_yields_blocked_stage(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    raw["tasks"]["rel-example/pairwise"]["prepared_artifacts"][
        "lowering_evidence"
    ] = []
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            program_ids=("baseline",),
        )
    )

    materialize = report.stages[2]
    assert materialize.stage == "materialize"
    assert materialize.status == "blocked"
    assert report.pipeline_status == "blocked"
    assert any(
        blocker.startswith("baseline:missing_proven_lowering_evidence")
        for blocker in materialize.blockers
    )


def test_input_resolved_failed_candidate_yields_failed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    def fake_materialize(request):
        return TaskCandidateMaterializationReport(
            dataset=request.dataset,
            task=request.task,
            output_root=request.output_root,
            dry_run=False,
            input_resolved=True,
            input_blockers=(),
            evidence_locations=(),
            outcomes=(
                TaskCandidateMaterializationOutcome(
                    program_id="baseline",
                    output_dir=request.output_root / "candidate",
                    primitive_count=1,
                    lowering_feasible=False,
                    status="failed",
                    blockers=("boom",),
                ),
            ),
        )

    monkeypatch.setattr(
        "fdhg.compiler.task_pipeline.materialize_task_candidates",
        fake_materialize,
    )

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            program_ids=("baseline",),
        )
    )

    assert report.stages[2].status == "failed"
    assert report.pipeline_status == "failed"
    assert "baseline:boom" in report.stages[2].blockers


def test_all_published_reused_or_dry_run_ready_yields_completed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    def fake_materialize(request):
        return TaskCandidateMaterializationReport(
            dataset=request.dataset,
            task=request.task,
            output_root=request.output_root,
            dry_run=not request.write,
            input_resolved=True,
            input_blockers=(),
            evidence_locations=(),
            outcomes=(
                TaskCandidateMaterializationOutcome(
                    program_id="a",
                    output_dir=request.output_root / "a",
                    primitive_count=1,
                    lowering_feasible=True,
                    status="published",
                    blockers=(),
                ),
                TaskCandidateMaterializationOutcome(
                    program_id="b",
                    output_dir=request.output_root / "b",
                    primitive_count=1,
                    lowering_feasible=True,
                    status="reused",
                    blockers=(),
                ),
            ),
        )

    monkeypatch.setattr(
        "fdhg.compiler.task_pipeline.materialize_task_candidates",
        fake_materialize,
    )

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            program_ids=("baseline",),
        )
    )

    assert report.stages[2].status == "completed"
    assert report.pipeline_status == "completed"

    def fake_dry_run(request):
        return TaskCandidateMaterializationReport(
            dataset=request.dataset,
            task=request.task,
            output_root=request.output_root,
            dry_run=True,
            input_resolved=True,
            input_blockers=(),
            evidence_locations=(),
            outcomes=(
                TaskCandidateMaterializationOutcome(
                    program_id="baseline",
                    output_dir=request.output_root / "baseline",
                    primitive_count=1,
                    lowering_feasible=True,
                    status="dry_run_ready",
                    blockers=(),
                ),
            ),
        )

    monkeypatch.setattr(
        "fdhg.compiler.task_pipeline.materialize_task_candidates",
        fake_dry_run,
    )
    dry = run_task_pipeline(
        request(tmp_path, reproduction, semantics, mode="dry-run")
    )
    assert dry.stages[2].status == "completed"


def test_candidate_level_blockers_written_to_pipeline_report(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    raw["tasks"]["rel-example/pairwise"]["prepared_artifacts"][
        "lowering_evidence"
    ] = []
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
        )
    )

    report_path = (
        tmp_path
        / "results"
        / "compiler"
        / "rel-example_pairwise"
        / "task_pipeline_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    materialize = [
        stage
        for stage in payload["stages"]
        if stage["stage"] == "materialize"
    ][0]
    assert report.pipeline_status == "blocked"
    assert materialize["status"] == "blocked"
    assert any(
        item.startswith("baseline:missing_proven_lowering_evidence")
        for item in materialize["outputs"]["candidate_blockers"]
    )


def test_validation_only_with_existing_candidates(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
        )
    )

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="validation-only",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
        ),
        evaluator=FakeEvaluator(good_scores()),
    )

    assert report.canonical_validation_path.exists()
    assert len(report.evaluation_results) == 8


def test_one_failed_evaluation_yields_failed_validation_and_pipeline(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    materialize_for_validation(tmp_path, reproduction, semantics)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="through-validation",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("failed", None, None, "subprocess_failed:1"),
        }),
    )

    validate = stage(report, "validate")
    export = stage(report, "export")
    assert validate.status == "failed"
    assert report.pipeline_status == "failed"
    assert validate.outputs["failed"] == 1
    assert validate.outputs["evaluation_result_count"] == 1
    assert "baseline:seed41:subprocess_failed:1" in validate.blockers
    assert export.status == "completed"
    assert report.canonical_validation_path.exists()
    text = report.canonical_validation_path.read_text(encoding="utf-8")
    assert "metric_failure|missing_seeds" in text


def test_mixed_completed_and_failed_validation_yields_failed(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    materialize_for_validation(
        tmp_path,
        reproduction,
        semantics,
        program_ids=("baseline", "baseline_plus_pair_left_temporal"),
    )

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="through-validation",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("completed", 0.7, 2, ""),
            ("baseline_plus_pair_left_temporal", 41): (
                "failed",
                None,
                None,
                "subprocess_failed:1",
            ),
        }),
    )

    validate = stage(report, "validate")
    assert validate.status == "failed"
    assert report.pipeline_status == "failed"
    assert validate.outputs["completed"] == 1
    assert validate.outputs["failed"] == 1


def test_one_blocked_evaluation_yields_blocked_validation_and_pipeline(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    materialize_for_validation(tmp_path, reproduction, semantics)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="through-validation",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("blocked", None, None, "missing_evaluator_config"),
        }),
    )

    validate = stage(report, "validate")
    assert validate.status == "blocked"
    assert report.pipeline_status == "blocked"
    assert validate.outputs["blocked"] == 1
    assert "baseline:seed41:missing_evaluator_config" in validate.blockers


def test_mixed_completed_and_blocked_validation_yields_blocked(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    materialize_for_validation(
        tmp_path,
        reproduction,
        semantics,
        program_ids=("baseline", "baseline_plus_pair_left_temporal"),
    )

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="through-validation",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("completed", 0.7, 2, ""),
            ("baseline_plus_pair_left_temporal", 41): (
                "blocked",
                None,
                None,
                "candidate_missing",
            ),
        }),
    )

    validate = stage(report, "validate")
    assert validate.status == "blocked"
    assert report.pipeline_status == "blocked"
    assert validate.outputs["completed"] == 1
    assert validate.outputs["blocked"] == 1


def test_all_completed_reused_or_mixed_success_validation_completed(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    materialize_for_validation(
        tmp_path,
        reproduction,
        semantics,
        program_ids=("baseline", "baseline_plus_pair_left_temporal"),
    )

    completed = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="validation-only",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("completed", 0.7, 2, ""),
        }),
    )
    assert stage(completed, "validate").status == "completed"

    reused = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="validation-only",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("reused", 0.7, 2, ""),
        }),
    )
    assert stage(reused, "validate").status == "completed"
    assert stage(reused, "validate").outputs["reused"] == 1

    mixed = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="validation-only",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("completed", 0.7, 2, ""),
            ("baseline_plus_pair_left_temporal", 41): ("reused", 0.8, 4, ""),
        }),
    )
    validate = stage(mixed, "validate")
    assert validate.status == "completed"
    assert validate.outputs["completed"] == 1
    assert validate.outputs["reused"] == 1


def test_validation_report_json_preserves_counts_and_blockers(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    materialize_for_validation(tmp_path, reproduction, semantics)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="through-validation",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): ("failed", None, None, "subprocess_failed:1"),
        }),
    )

    report_path = (
        tmp_path
        / "results"
        / "compiler"
        / "rel-example_pairwise"
        / "task_pipeline_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    validate = [
        item
        for item in payload["stages"]
        if item["stage"] == "validate"
    ][0]
    assert report.pipeline_status == "failed"
    assert payload["pipeline_status"] == "failed"
    assert validate["outputs"]["failed"] == 1
    assert validate["outputs"]["evaluation_result_count"] == 1
    assert "baseline:seed41:subprocess_failed:1" in validate["blockers"]


def test_no_validation_results_is_blocked_when_validation_requested(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="validation-only",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({}),
    )

    validate = stage(report, "validate")
    assert validate.status == "blocked"
    assert report.pipeline_status == "blocked"
    assert "no_evaluation_results" in validate.blockers


def test_successful_regression_like_result_with_five_features_completed(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    raw = yaml.safe_load(semantics.read_text(encoding="utf-8"))
    raw["rel-example/pairwise"]["primary_metric"] = "rmse"
    raw["rel-example/pairwise"]["metric_direction"] = "lower"
    semantics.write_text(yaml.safe_dump(raw), encoding="utf-8")
    materialize_for_validation(tmp_path, reproduction, semantics)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="validation-only",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): (
                "completed",
                45.639053051904675,
                5,
                "",
            ),
        }),
    )

    validate = stage(report, "validate")
    assert validate.status == "completed"
    assert report.evaluation_results[0].score == 45.639053051904675
    assert report.evaluation_results[0].n_features == 5


def test_reused_regression_like_result_with_five_features_completed(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    raw = yaml.safe_load(semantics.read_text(encoding="utf-8"))
    raw["rel-example/pairwise"]["primary_metric"] = "rmse"
    raw["rel-example/pairwise"]["metric_direction"] = "lower"
    semantics.write_text(yaml.safe_dump(raw), encoding="utf-8")
    materialize_for_validation(tmp_path, reproduction, semantics)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="validation-only",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline",),
            seeds=(41,),
        ),
        evaluator=ScriptedEvaluator({
            ("baseline", 41): (
                "reused",
                45.639053051904675,
                5,
                "",
            ),
        }),
    )

    validate = stage(report, "validate")
    assert validate.status == "completed"
    assert validate.outputs["reused"] == 1


def test_selection_only_with_existing_canonical_csv(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    canonical = (
        tmp_path
        / "results"
        / "compiler"
        / "rel-example_pairwise"
        / "canonical_validation.csv"
    )
    canonical.parent.mkdir(parents=True)
    rows = [
        CandidateValidationResult(
            dataset="rel-example",
            task="pairwise",
            program_id="baseline",
            primary_metric="roc_auc",
            metric_direction="higher",
            validation_score=0.70,
            split="validation",
            n_features=2,
            eligible=True,
            evidence_location="baseline",
            materializable=True,
            leakage_safe=True,
            temporally_safe=True,
            provenance_complete=True,
        ),
        CandidateValidationResult(
            dataset="rel-example",
            task="pairwise",
            program_id="baseline_plus_pair_left_temporal",
            primary_metric="roc_auc",
            metric_direction="higher",
            validation_score=0.75,
            baseline_program_id="baseline",
            baseline_score=0.70,
            split="validation",
            n_features=4,
            eligible=True,
            evidence_location="fdhg",
            materializable=True,
            leakage_safe=True,
            temporally_safe=True,
            provenance_complete=True,
        ),
    ]
    with canonical.open("w", encoding="utf-8", newline="") as handle:
        from fdhg.compiler.validation_results import NormalizedCandidateRecord
        import csv

        fieldnames = list(NormalizedCandidateRecord(
            dataset="x",
            task="x",
            program_id="x",
            split="validation",
            primary_metric="roc_auc",
            metric_direction="higher",
            score=0.0,
            n_features=1,
            eligible=True,
            rejection_reasons=(),
            evidence_location="x",
            materializable=True,
            leakage_safe=True,
            temporally_safe=True,
            provenance_complete=True,
        ).to_csv_row().keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset": row.dataset,
                "task": row.task,
                "program_id": row.program_id,
                "split": row.split,
                "primary_metric": row.primary_metric,
                "metric_direction": row.metric_direction,
                "score": row.validation_score,
                "n_features": row.n_features,
                "eligible": "true",
                "rejection_reason": "",
                "evidence_location": row.evidence_location,
                "materializable": "true",
                "leakage_safe": "true",
                "temporally_safe": "true",
                "provenance_complete": "true",
                "baseline_program_id": row.baseline_program_id or "",
                "baseline_score": "" if row.baseline_score is None else row.baseline_score,
            })

    report = run_task_pipeline(
        request(tmp_path, reproduction, semantics, mode="selection-only")
    )

    assert report.selection_decision.selected_program_id == (
        "baseline_plus_pair_left_temporal"
    )


def test_full_synthetic_pipeline_selects_best_fdhg(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="full",
            write_materialization=True,
            run_validation=True,
            program_ids=(
                "baseline",
                "baseline_plus_pair_left_temporal",
                "baseline_plus_pairwise_temporal",
            ),
        ),
        evaluator=FakeEvaluator(good_scores()),
    )

    assert report.selection_decision.selected_program_id == (
        "baseline_plus_pair_left_temporal"
    )
    assert report.pipeline_status == "completed"


def test_full_pipeline_with_production_evaluator_fake_runner(
    tmp_path: Path,
) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    def runner(argv, *, cwd, env, timeout):
        import subprocess

        program = argv[argv.index("--variant") + 1]
        seed = int(argv[argv.index("--seed") + 1])
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        canonical_program = "baseline" if program == "dfs" else program
        score = (
            0.70
            if canonical_program == "baseline"
            else 0.76
            if canonical_program == "baseline_plus_pair_left_temporal"
            else 0.71
        )
        (out_dir / "metrics.csv").write_text(
            "dataset,task,variant,seed,roc_auc,n_features\n"
            f"rel-example,pairwise,{program},{seed},{score},4\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    evaluator = SubprocessCandidateEvaluator(
        config=CandidateEvaluatorConfig(
            reproduction_config=reproduction,
            python_executable=Path("/usr/bin/python3"),
            device="cpu",
        ),
        process_runner=runner,
    )

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="full",
            write_materialization=True,
            run_validation=True,
            program_ids=(
                "baseline",
                "baseline_plus_pair_left_temporal",
                "baseline_plus_pairwise_temporal",
            ),
        ),
        evaluator=evaluator,
    )

    assert report.selection_decision.selected_program_id == (
        "baseline_plus_pair_left_temporal"
    )


def test_one_candidate_blocked_while_another_succeeds(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    raw = yaml.safe_load(reproduction.read_text(encoding="utf-8"))
    evidence = raw["tasks"]["rel-example/pairwise"]["prepared_artifacts"][
        "lowering_evidence"
    ]
    raw["tasks"]["rel-example/pairwise"]["prepared_artifacts"][
        "lowering_evidence"
    ] = [row for row in evidence if row["program_id"] == "baseline"]
    reproduction.write_text(yaml.safe_dump(raw), encoding="utf-8")

    report = run_task_pipeline(
        request(tmp_path, reproduction, semantics, mode="materialize-only", write_materialization=True)
    )

    assert report.stages[2].outputs["published"] == 1
    assert report.stages[2].outputs["blocked"] > 0


def test_one_seed_evaluation_failure_blocks_export_selection(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="materialize-only",
            write_materialization=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
        )
    )

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="through-validation",
            run_validation=True,
            overwrite_pipeline_output=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
        ),
        evaluator=FakeEvaluator(good_scores(), fail={("baseline_plus_pair_left_temporal", 44)}),
    )

    assert (
        "baseline_plus_pair_left_temporal:seed44:"
        "failed:baseline_plus_pair_left_temporal:44"
    ) in report.failures
    assert stage(report, "validate").status == "failed"
    assert report.pipeline_status == "failed"
    assert report.canonical_validation_path.exists()


def test_strict_candidate_discovery(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    run_task_pipeline(
        request(tmp_path, reproduction, semantics, mode="materialize-only", write_materialization=True, program_ids=("baseline",))
    )
    private = tmp_path / "outputs" / "e2e" / "rel-example_pairwise" / "candidates" / "_x"
    private.mkdir(parents=True)

    assert discover_strict_materialized_candidates(
        dataset="rel-example",
        task="pairwise",
        output_root=tmp_path / "outputs" / "e2e",
        candidate_ids=("baseline", "_x"),
    ) == ("baseline",)


def test_dfs_fallback_and_exact_tie_fallback(tmp_path: Path) -> None:
    programs = [
        type("P", (), {"program_id": "baseline", "families": ["baseline"], "estimated_feature_count": 2})(),
        type("P", (), {"program_id": "fdhg", "families": ["temporal"], "estimated_feature_count": 4})(),
    ]
    records = [
        CandidateValidationResult("d", "t", "baseline", "roc_auc", "higher", 0.7, n_features=2, evidence_location="b", materializable=True, leakage_safe=True, temporally_safe=True, provenance_complete=True),
        CandidateValidationResult("d", "t", "fdhg", "roc_auc", "higher", 0.7, baseline_program_id="baseline", baseline_score=0.7, n_features=4, evidence_location="f", materializable=True, leakage_safe=True, temporally_safe=True, provenance_complete=True),
    ]

    decision = select_candidate_program(
        programs,
        records,
        CandidateSelectionPolicy("d", "t", "roc_auc", "higher", baseline_program_id="baseline"),
    )

    assert decision.fallback_occurred


def test_lower_is_better_metric(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    raw = yaml.safe_load(semantics.read_text(encoding="utf-8"))
    raw["rel-example/pairwise"]["primary_metric"] = "log_loss"
    raw["rel-example/pairwise"]["metric_direction"] = "lower"
    semantics.write_text(yaml.safe_dump(raw), encoding="utf-8")
    scores = {}
    for seed in (41, 42, 43, 44):
        scores[("baseline", seed)] = 0.70
        scores[("baseline_plus_pair_left_temporal", seed)] = 0.60

    report = run_task_pipeline(
        request(
            tmp_path,
            reproduction,
            semantics,
            mode="full",
            write_materialization=True,
            run_validation=True,
            program_ids=("baseline", "baseline_plus_pair_left_temporal"),
        ),
        evaluator=FakeEvaluator(scores),
    )

    assert report.selection_decision.selected_program_id == (
        "baseline_plus_pair_left_temporal"
    )


def test_output_overwrite_refusal(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    report_path = (
        tmp_path / "results" / "compiler" / "rel-example_pairwise" / "task_pipeline_report.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_task_pipeline(
            request(tmp_path, reproduction, semantics, mode="materialize-only", write_materialization=True, program_ids=("baseline",))
        )


def test_no_test_metric_selection(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    canonical = tmp_path / "results" / "compiler" / "rel-example_pairwise" / "canonical_validation.csv"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        "dataset,task,program_id,split,primary_metric,metric_direction,score,n_features,eligible,rejection_reason,evidence_location,materializable,leakage_safe,temporally_safe,provenance_complete,baseline_program_id,baseline_score\n"
        "rel-example,pairwise,baseline,test,roc_auc,higher,0.7,2,true,,b,true,true,true,true,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        run_task_pipeline(request(tmp_path, reproduction, semantics, mode="selection-only"))


def test_ratebeer_blockers_reported(tmp_path: Path) -> None:
    report = run_task_pipeline(TaskPipelineRequest(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        output_root=tmp_path / "outputs",
        result_root=tmp_path / "results",
        seeds=(41, 42, 43, 44),
        mode="dry-run",
        baseline_only=True,
        reproduction_config=Path("configs/reproduction/tasks.yaml"),
        semantics_config=Path("configs/reproduction/task_semantics.yaml"),
    ))

    assert report.pipeline_status == "blocked"
    assert "missing_prepared_artifacts_config" in report.stages[2].blockers


def test_deterministic_report(tmp_path: Path) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    first = run_task_pipeline(request(tmp_path, reproduction, semantics))
    second = run_task_pipeline(request(tmp_path, reproduction, semantics))

    assert first.to_dict() == second.to_dict()
