from __future__ import annotations

from pathlib import Path

from fdhg.cli.run_task_pipeline import main
from fdhg.compiler.task_pipeline import EvaluationResult
from tests.unit.test_task_pipeline import fixture_with_metric


def test_cli_default_dry_run_no_writes(tmp_path: Path, capsys) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--result-root",
        str(tmp_path / "results" / "compiler"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--program-id",
        "baseline",
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert "REQUESTED_MODE dry-run" in out
    assert "materialize\t" in out
    assert not (tmp_path / "outputs").exists()


def test_cli_full_blocks_without_evaluator(tmp_path: Path, capsys) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--result-root",
        str(tmp_path / "results" / "compiler"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--program-id",
        "baseline",
        "--through-validation",
        "--write-materialization",
        "--run-validation",
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert "validate\tblocked\tno_candidate_evaluator_configured" in out


def test_cli_evaluator_configuration(tmp_path: Path, capsys, monkeypatch) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    class FakeProductionEvaluator:
        def __init__(self, *, config):
            self.config = config

        def evaluate(self, request):
            return EvaluationResult(
                request=request,
                status="failed",
                score=None,
                n_features=None,
                evidence_location="fake",
                rejection_reason="fake",
                command=("fake",),
                environment=(f"device={self.config.device}",),
            )

    monkeypatch.setattr(
        "fdhg.cli.run_task_pipeline.SubprocessCandidateEvaluator",
        FakeProductionEvaluator,
    )

    code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--result-root",
        str(tmp_path / "results" / "compiler"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--program-id",
        "baseline",
        "--through-validation",
        "--write-materialization",
        "--run-validation",
        "--evaluator",
        "existing-script",
        "--device",
        "cuda:0",
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert "PIPELINE_STATUS failed" in out
    assert "validate\tfailed" in out
    assert "baseline:seed41:fake" in out


def test_cli_selection_only_existing_csv(tmp_path: Path, capsys) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)
    canonical = (
        tmp_path
        / "results"
        / "compiler"
        / "rel-example_pairwise"
        / "canonical_validation.csv"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_text(
        "dataset,task,program_id,split,primary_metric,metric_direction,score,n_features,eligible,rejection_reason,evidence_location,materializable,leakage_safe,temporally_safe,provenance_complete,baseline_program_id,baseline_score\n"
        "rel-example,pairwise,baseline,validation,roc_auc,higher,0.7,2,true,,b,true,true,true,true,,\n",
        encoding="utf-8",
    )

    code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--result-root",
        str(tmp_path / "results" / "compiler"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--baseline-only",
        "--selection-only",
    ])

    assert code == 0
    assert "SELECTED_PROGRAM_ID baseline" in capsys.readouterr().out


def test_cli_refuses_paper_table_outputs(tmp_path: Path, capsys) -> None:
    reproduction, semantics = fixture_with_metric(tmp_path)

    code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--result-root",
        "results/paper_tables",
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--materialize-only",
        "--write-materialization",
        "--program-id",
        "baseline",
    ])

    assert code == 1
    assert "results/paper_tables" in capsys.readouterr().err
