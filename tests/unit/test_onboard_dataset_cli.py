from __future__ import annotations

from pathlib import Path

import pytest

from fdhg.cli.onboard_dataset import main
import fdhg.onboarding.pipeline as onboarding_pipeline
from tests.unit.onboarding_fixtures import write_onboarding_fixture


def test_onboard_dataset_cli_dry_run_no_writes(
    tmp_path: Path,
    capsys,
) -> None:
    config = write_onboarding_fixture(tmp_path)

    code = main([
        "--config", str(config),
        "--output-root", str(tmp_path / "out"),
        "--dry-run",
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert "STATUS dry_run_ready" in out
    assert not (tmp_path / "out").exists()


def test_onboard_dataset_cli_dry_run_does_not_materialize(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_onboarding_fixture(tmp_path)

    def fail_materialization(*args, **kwargs):
        raise AssertionError("feature materialization called during dry-run")

    monkeypatch.setattr(
        onboarding_pipeline,
        "_apply_features_optimized",
        fail_materialization,
    )

    code = main([
        "--config", str(config),
        "--output-root", str(tmp_path / "out"),
        "--dry-run",
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert "STATUS dry_run_ready" in out


def test_onboard_dataset_cli_write(tmp_path: Path, capsys) -> None:
    config = write_onboarding_fixture(tmp_path)

    code = main([
        "--config", str(config),
        "--output-root", str(tmp_path / "out"),
        "--write",
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert "STATUS completed" in out
    assert (
        tmp_path
        / "out"
        / "example-commerce_user-spend"
        / "onboarding_manifest.json"
    ).exists()


def test_onboard_dataset_cli_blocked(tmp_path: Path, capsys) -> None:
    config = write_onboarding_fixture(tmp_path, orphan=True)

    code = main([
        "--config", str(config),
        "--output-root", str(tmp_path / "out"),
        "--write",
    ])

    out = capsys.readouterr().out
    assert code == 2
    assert "STATUS blocked" in out
    assert "referential_integrity_failure" in out
