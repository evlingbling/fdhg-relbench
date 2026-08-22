from __future__ import annotations

from pathlib import Path

from fdhg.cli.materialize_task_candidates import main
from tests.unit.test_task_candidate_materialization import write_task_fixture


def test_cli_dry_run_no_manual_json_rows(
    tmp_path: Path,
    capsys,
) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    exit_code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--program-id",
        "baseline",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "DRY_RUN True" in out
    assert "baseline\tdry_run_ready" in out
    assert not (tmp_path / "outputs").exists()


def test_cli_write_publishes_candidate(
    tmp_path: Path,
    capsys,
) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    exit_code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--program-id",
        "baseline",
        "--write",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PUBLISHED_COUNT 1" in out
    assert (
        tmp_path
        / "outputs"
        / "e2e"
        / "rel-example_pairwise"
        / "candidates"
        / "baseline"
        / "materialization_manifest.json"
    ).exists()


def test_cli_program_filter_and_exclude(
    tmp_path: Path,
    capsys,
) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    exit_code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--exclude-program-id",
        "baseline",
    ])

    assert exit_code == 0
    assert "\nbaseline\t" not in capsys.readouterr().out


def test_cli_unknown_program_id(
    tmp_path: Path,
    capsys,
) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    exit_code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--program-id",
        "unknown",
    ])

    assert exit_code == 1
    assert "unknown" in capsys.readouterr().err


def test_cli_missing_prepared_artifacts_fails_closed(
    tmp_path: Path,
    capsys,
) -> None:
    reproduction = tmp_path / "tasks.yaml"
    semantics = tmp_path / "task_semantics.yaml"
    reproduction.write_text(
        """
tasks:
  rel-example/single:
    problem_type: binary
    label_col: label
    target:
      entity_key: entity_id
      time_col: timestamp
    dfs: {}
""",
        encoding="utf-8",
    )
    semantics.write_text("rel-example/single: {}\n", encoding="utf-8")

    exit_code = main([
        "--dataset",
        "rel-example",
        "--task",
        "single",
        "--output-root",
        str(tmp_path / "outputs"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--baseline-only",
        "--write",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "INPUT_RESOLVED False" in out
    assert "missing_prepared_artifacts_config" in out
    assert not (tmp_path / "outputs").exists()


def test_cli_does_not_call_training_or_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reproduction, semantics = write_task_fixture(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("training/evaluation must not run")

    monkeypatch.setattr(
        "scripts.experiments.run_candidate_program_sweep.apply_stability_gate",
        fail,
        raising=False,
    )

    exit_code = main([
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--output-root",
        str(tmp_path / "outputs" / "e2e"),
        "--reproduction-config",
        str(reproduction),
        "--semantics-config",
        str(semantics),
        "--program-id",
        "baseline",
    ])

    assert exit_code == 0
