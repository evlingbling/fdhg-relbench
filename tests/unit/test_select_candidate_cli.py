from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

from fdhg.cli import select_candidate


DATASET = "rel-ratebeer"
TASK = "user-place-liked_pairwise"


def csv_text(rows: list[str]) -> str:
    header = (
        "dataset,task,program_id,split,primary_metric,"
        "metric_direction,score,n_features,eligible,"
        "rejection_reason,evidence_location,"
        "materializable,leakage_safe,temporally_safe,"
        "provenance_complete,baseline_program_id,baseline_score"
    )
    return "\n".join([header, *rows])


def row(
    program_id: str,
    score: str,
    *,
    task: str = TASK,
    split: str = "validation",
    n_features: str = "20",
    eligible: str = "true",
    evidence: str | None = None,
    materializable: str = "true",
    leakage_safe: str = "true",
    temporally_safe: str = "true",
    provenance_complete: str = "true",
    baseline_score: str = "0.70",
) -> str:
    return (
        f"{DATASET},{task},{program_id},{split},roc_auc,higher,"
        f"{score},{n_features},{eligible},,"
        f"{evidence or 'validation.csv'},"
        f"{materializable},{leakage_safe},{temporally_safe},"
        f"{provenance_complete},baseline,{baseline_score}"
    )


def write_validation_csv(
    tmp_path: Path,
    rows: list[str],
) -> Path:
    path = tmp_path / "validation.csv"
    path.write_text(csv_text(rows), encoding="utf-8")
    return path


def cli_args(path: Path) -> list[str]:
    return [
        "--dataset",
        DATASET,
        "--task",
        TASK,
        "--validation-results",
        str(path),
    ]


def test_cli_selects_best_fdhg_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_validation_csv(
        tmp_path,
        [
            row("baseline", "0.70", n_features="15"),
            row(
                "baseline_plus_pairwise_temporal",
                "0.74",
                n_features="30",
            ),
            row(
                "baseline_plus_pair_left_temporal",
                "0.76",
                n_features="22",
            ),
        ],
    )

    exit_code = select_candidate.main(cli_args(path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "SELECTED_PROGRAM_ID baseline_plus_pair_left_temporal"
        in output
    )
    assert "FALLBACK_OCCURRED False" in output


def test_cli_falls_back_to_baseline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_validation_csv(
        tmp_path,
        [
            row("baseline", "0.70", n_features="15"),
            row(
                "baseline_plus_pairwise_temporal",
                "0.69",
                n_features="30",
            ),
        ],
    )

    exit_code = select_candidate.main(cli_args(path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "SELECTED_PROGRAM_ID baseline" in output
    assert "FALLBACK_OCCURRED True" in output


def test_cli_exact_tie_falls_back(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_validation_csv(
        tmp_path,
        [
            row("baseline", "0.70", n_features="15"),
            row(
                "baseline_plus_pairwise_temporal",
                "0.70",
                n_features="30",
            ),
        ],
    )

    exit_code = select_candidate.main(cli_args(path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "SELECTED_PROGRAM_ID baseline" in output


def test_cli_rejects_malformed_canonical_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "validation.csv"
    path.write_text(
        "dataset,task,program_id,score\n"
        f"{DATASET},{TASK},baseline,0.70\n",
        encoding="utf-8",
    )

    exit_code = select_candidate.main(cli_args(path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing canonical validation columns" in captured.err


def test_cli_reports_task_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_validation_csv(
        tmp_path,
        [
            row("baseline", "0.70", n_features="15"),
            row(
                "baseline_plus_pairwise_temporal",
                "0.80",
                task="other-task",
                n_features="30",
            ),
        ],
    )

    exit_code = select_candidate.main(cli_args(path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "SELECTED_PROGRAM_ID baseline" in output
    assert "task_mismatch" in output


def test_cli_rejects_test_split_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = write_validation_csv(
        tmp_path,
        [
            row("baseline", "0.70", n_features="15"),
            row(
                "baseline_plus_pairwise_temporal",
                "0.99",
                split="paper-final-test",
                n_features="30",
            ),
        ],
    )

    exit_code = select_candidate.main(cli_args(path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "SELECTED_PROGRAM_ID baseline" in output
    assert "test_or_final_split_evidence" in output


def test_cli_has_no_write_experiment_gpu_or_network_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_validation_csv(
        tmp_path,
        [
            row("baseline", "0.70", n_features="15"),
            row(
                "baseline_plus_pairwise_temporal",
                "0.74",
                n_features="30",
            ),
        ],
    )

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(Path, "unlink", fail)
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(socket, "socket", fail)

    exit_code = select_candidate.main(cli_args(path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "SELECTED_PROGRAM_ID baseline_plus_pairwise_temporal"
        in output
    )
