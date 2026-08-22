from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

from fdhg.cli import normalize_validation_results
from fdhg.cli import select_candidate


def header() -> str:
    return (
        "dataset,task,program_id,split,primary_metric,"
        "metric_direction,validation_score,n_features,eligible,"
        "rejection_reason,evidence_location,materializable,"
        "leakage_safe,temporally_safe,provenance_complete,"
        "baseline_program_id,baseline_score"
    )


def row(
    program_id: str,
    score: str,
    *,
    provenance_complete: str = "true",
) -> str:
    return (
        "rel-ratebeer,user-place-liked_pairwise,"
        f"{program_id},validation,roc_auc,higher,{score},"
        "15,true,,source.csv,true,true,true,"
        f"{provenance_complete},baseline,0.70"
    )


def write_source(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "source.csv"
    path.write_text(
        "\n".join([header(), *rows]),
        encoding="utf-8",
    )
    return path


def test_cli_prints_canonical_csv_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_source(tmp_path, [row("baseline", "0.70")])

    exit_code = normalize_validation_results.main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--source",
        str(source),
    ])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.startswith("dataset,task,program_id")
    assert "rel-ratebeer,user-place-liked_pairwise,baseline" in output


def test_cli_audit_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "unsupported.csv"
    source.write_text(
        "task,selected_candidate,primary_metric\n"
        "rel-ratebeer_user-place-liked_pairwise,fdhg,roc_auc\n",
        encoding="utf-8",
    )

    exit_code = normalize_validation_results.main([
        "--audit",
        "--source",
        str(source),
    ])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "adapter_supported" in output
    assert "False" in output


def test_cli_refuses_overwrite_by_default(tmp_path: Path) -> None:
    source = write_source(tmp_path, [row("baseline", "0.70")])
    output = tmp_path / "canonical.csv"
    output.write_text("existing\n", encoding="utf-8")

    exit_code = normalize_validation_results.main([
        "--source",
        str(source),
        "--output",
        str(output),
    ])

    assert exit_code == 1
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_cli_writes_explicit_tmp_output(tmp_path: Path) -> None:
    source = write_source(tmp_path, [row("baseline", "0.70")])
    output = Path("/tmp/fdhg_validation_results_test.csv")
    if output.exists():
        output.unlink()

    exit_code = normalize_validation_results.main([
        "--source",
        str(source),
        "--output",
        str(output),
    ])

    try:
        assert exit_code == 0
        assert output.exists()
        assert output.read_text(encoding="utf-8").startswith(
            "dataset,task,program_id"
        )
    finally:
        if output.exists():
            output.unlink()


def test_cli_selector_integration(tmp_path: Path) -> None:
    source = write_source(
        tmp_path,
        [
            row("baseline", "0.70"),
            row("baseline_plus_pairwise_temporal", "0.75"),
        ],
    )
    canonical = tmp_path / "canonical.csv"

    normalize_exit = normalize_validation_results.main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--source",
        str(source),
        "--output",
        str(canonical),
    ])
    select_exit = select_candidate.main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--validation-results",
        str(canonical),
    ])

    assert normalize_exit == 0
    assert select_exit == 0


def test_cli_ratebeer_incomplete_provenance_falls_back(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_source(
        tmp_path,
        [
            row("baseline", "0.70"),
            row(
                "baseline_plus_pairwise_temporal",
                "0.99",
                provenance_complete="false",
            ),
        ],
    )
    canonical = tmp_path / "canonical.csv"

    normalize_validation_results.main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--source",
        str(source),
        "--output",
        str(canonical),
    ])
    exit_code = select_candidate.main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--validation-results",
        str(canonical),
    ])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "SELECTED_PROGRAM_ID baseline" in output


def test_cli_no_experiment_training_materialization_gpu_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_source(tmp_path, [row("baseline", "0.70")])

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(socket, "socket", fail)

    exit_code = normalize_validation_results.main([
        "--source",
        str(source),
    ])

    assert exit_code == 0
    assert "dataset,task,program_id" in capsys.readouterr().out
