from __future__ import annotations

from pathlib import Path

import pytest

from fdhg.cli import audit_candidate_safety


def cli_args(candidate_root: Path) -> list[str]:
    return [
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--program-id",
        "baseline",
        "--candidate-root",
        str(candidate_root),
    ]


def test_cli_print_only_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = audit_candidate_safety.main(cli_args(tmp_path))

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "TEMPORAL_SAFETY" in output
    assert "LOWERING_PROVENANCE" in output


def test_cli_overwrite_refusal(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    output.mkdir()
    (output / "existing.txt").write_text("x", encoding="utf-8")

    exit_code = audit_candidate_safety.main([
        *cli_args(tmp_path),
        "--output-dir",
        str(output),
    ])

    assert exit_code == 1


def test_cli_explicit_tmp_output(tmp_path: Path) -> None:
    output = Path("/tmp/fdhg_candidate_safety_audit_test")
    if output.exists():
        for child in output.iterdir():
            child.unlink()
        output.rmdir()

    exit_code = audit_candidate_safety.main([
        *cli_args(tmp_path),
        "--output-dir",
        str(output),
    ])

    try:
        assert exit_code == 0
        assert (output / "temporal_safety_audit.csv").exists()
        assert (output / "leakage_safety_audit.csv").exists()
        assert (output / "lowering_provenance_audit.csv").exists()
    finally:
        if output.exists():
            for child in output.iterdir():
                child.unlink()
            output.rmdir()


def test_cli_rejects_results_paper_tables_output(tmp_path: Path) -> None:
    exit_code = audit_candidate_safety.main([
        *cli_args(tmp_path),
        "--output-dir",
        "results/paper_tables/safety_audit",
        "--overwrite",
    ])

    assert exit_code == 1
