from __future__ import annotations

import json
from pathlib import Path

import pytest

from fdhg.cli import compile_materialization_plan as cli
from fdhg.compiler.programs import CandidateProgram


CONFIG = "configs/reproduction/tasks.yaml"
SEMANTICS = "configs/reproduction/task_semantics.yaml"


def ratebeer_args(output_dir: Path, *extra: str) -> list[str]:
    return [
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--program",
        "baseline_plus_pairwise_temporal",
        "--reproduction-config",
        CONFIG,
        "--semantics-config",
        SEMANTICS,
        "--output-dir",
        str(output_dir),
        "--compiler-version",
        "unit-test",
        "--git-commit",
        "abc123",
        "--created-at-utc",
        "2026-01-01T00:00:00Z",
        "--source",
        "unit-test",
        *extra,
    ]


def output_files(output_dir: Path) -> set[str]:
    return {
        path.name for path in output_dir.iterdir()
    }


def file_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }


def hidden_sibling_dirs(tmp_path: Path) -> list[Path]:
    return [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name.startswith(".plan.")
    ]


def test_successful_ratebeer_plan_only_cli_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "plan"

    exit_code = cli.main(ratebeer_args(output_dir))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PROGRAM_ID baseline_plus_pairwise_temporal" in (
        captured.out
    )
    assert "STEP_COUNT 29" in captured.out
    assert output_dir.is_dir()


def test_exactly_three_output_files_created(tmp_path: Path) -> None:
    output_dir = tmp_path / "plan"

    assert cli.main(ratebeer_args(output_dir)) == 0

    assert output_files(output_dir) == {
        "materialization_plan.json",
        "primitive_column_bindings.json",
        "temporal_safety_audit.csv",
    }


def test_json_and_csv_contents_match_selected_program(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "plan"

    assert cli.main(ratebeer_args(output_dir)) == 0

    plan = json.loads(
        (
            output_dir / "materialization_plan.json"
        ).read_text(encoding="utf-8")
    )
    bindings = json.loads(
        (
            output_dir / "primitive_column_bindings.json"
        ).read_text(encoding="utf-8")
    )
    audit_text = (
        output_dir / "temporal_safety_audit.csv"
    ).read_text(encoding="utf-8")

    assert plan["program_id"] == (
        "baseline_plus_pairwise_temporal"
    )
    assert plan["step_count"] == 29
    assert plan["lowering_mode_counts"]["generate"] == 14
    assert len(bindings["records"]) == 17
    assert audit_text.count("\n") == 30
    assert (
        "baseline_plus_pairwise_temporal"
        in audit_text
    )


def test_unknown_program_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ratebeer_args(tmp_path / "plan")
    args[args.index("--program") + 1] = "missing"

    exit_code = cli.main(args)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "unknown program ID" in captured.err
    assert not (tmp_path / "plan").exists()


def test_unsafe_plan_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_planner = cli.plan_candidate_materialization

    def unsafe_plan(*args, **kwargs):
        plan = real_planner(*args, **kwargs)
        return type(plan)(
            program_id=plan.program_id,
            steps=plan.steps,
            audit_rows=plan.audit_rows,
            materializable=True,
            temporally_safe=False,
            requires_external_provider=False,
        )

    monkeypatch.setattr(
        cli,
        "plan_candidate_materialization",
        unsafe_plan,
    )

    exit_code = cli.main(ratebeer_args(tmp_path / "plan"))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not temporally safe" in captured.err
    assert not (tmp_path / "plan").exists()


def test_non_materializable_plan_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_planner = cli.plan_candidate_materialization

    def non_materializable_plan(*args, **kwargs):
        plan = real_planner(*args, **kwargs)
        return type(plan)(
            program_id=plan.program_id,
            steps=plan.steps,
            audit_rows=plan.audit_rows,
            materializable=False,
            temporally_safe=True,
            requires_external_provider=False,
        )

    monkeypatch.setattr(
        cli,
        "plan_candidate_materialization",
        non_materializable_plan,
    )

    exit_code = cli.main(ratebeer_args(tmp_path / "plan"))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not materializable" in captured.err
    assert not (tmp_path / "plan").exists()


def test_external_provider_plan_requires_opt_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ratebeer_args(tmp_path / "plan")
    args[args.index("--program") + 1] = (
        "baseline_plus_structural_pairwise_temporal"
    )

    exit_code = cli.main(args)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "requires an external provider" in captured.err
    assert not (tmp_path / "plan").exists()


def test_external_provider_plan_can_be_allowed(
    tmp_path: Path,
) -> None:
    args = ratebeer_args(
        tmp_path / "plan",
        "--allow-external-provider",
    )
    args[args.index("--program") + 1] = (
        "baseline_plus_structural_pairwise_temporal"
    )

    assert cli.main(args) == 0
    plan = json.loads(
        (
            tmp_path
            / "plan"
            / "materialization_plan.json"
        ).read_text(encoding="utf-8")
    )
    assert plan["requires_external_provider"]


def test_overwrite_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "plan"

    assert cli.main(ratebeer_args(output_dir)) == 0
    assert cli.main(ratebeer_args(output_dir)) == 1


def test_explicit_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "plan"

    assert cli.main(ratebeer_args(output_dir)) == 0
    args = ratebeer_args(output_dir, "--overwrite")
    args[args.index("--source") + 1] = "overwrite-test"

    assert cli.main(args) == 0
    assert output_files(output_dir) == {
        "materialization_plan.json",
        "primitive_column_bindings.json",
        "temporal_safety_audit.csv",
    }
    plan = json.loads(
        (
            output_dir / "materialization_plan.json"
        ).read_text(encoding="utf-8")
    )
    assert plan["metadata"]["source"] == "overwrite-test"
    assert hidden_sibling_dirs(tmp_path) == []


def test_final_rename_failure_restores_original_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "plan"
    assert cli.main(ratebeer_args(output_dir)) == 0
    original = file_bytes(output_dir)
    real_replace = cli._replace_path

    def fail_final_staging_replace(source: Path, destination: Path):
        if (
            destination == output_dir
            and source.name.endswith(".tmp")
        ):
            raise OSError("simulated final rename failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        cli,
        "_replace_path",
        fail_final_staging_replace,
    )

    exit_code = cli.main(
        ratebeer_args(output_dir, "--overwrite")
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "simulated final rename failure" in captured.err
    assert output_dir.is_dir()
    assert file_bytes(output_dir) == original
    assert hidden_sibling_dirs(tmp_path) == []


def test_restore_failure_preserves_backup_and_reports_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "plan"
    assert cli.main(ratebeer_args(output_dir)) == 0
    original = file_bytes(output_dir)
    real_replace = cli._replace_path

    def fail_final_and_restore(source: Path, destination: Path):
        if (
            destination == output_dir
            and source.name.endswith(".tmp")
        ):
            raise OSError("simulated final rename failure")
        if (
            destination == output_dir
            and source.name.endswith(".backup")
        ):
            raise OSError("simulated restore failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        cli,
        "_replace_path",
        fail_final_and_restore,
    )

    exit_code = cli.main(
        ratebeer_args(output_dir, "--overwrite")
    )

    captured = capsys.readouterr()
    backups = [
        path
        for path in hidden_sibling_dirs(tmp_path)
        if path.name.endswith(".backup")
    ]
    staging = [
        path
        for path in hidden_sibling_dirs(tmp_path)
        if path.name.endswith(".tmp")
    ]

    assert exit_code == 1
    assert "failed to restore previous output" in captured.err
    assert "simulated final rename failure" in captured.err
    assert "simulated restore failure" in captured.err
    assert len(backups) == 1
    assert str(backups[0]) in captured.err
    assert file_bytes(backups[0]) == original
    assert not output_dir.exists()
    assert staging == []


def test_partial_output_cleanup_on_simulated_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_staged_write(*args, **kwargs):
        staging = kwargs["staging"]
        (staging / "materialization_plan.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        raise OSError("simulated staged write failure")

    monkeypatch.setattr(
        cli,
        "_write_staged_files",
        fail_staged_write,
    )

    output_dir = tmp_path / "plan"
    exit_code = cli.main(ratebeer_args(output_dir))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "simulated staged write failure" in captured.err
    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_duplicate_program_ids_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    program = CandidateProgram(
        program_id="baseline_plus_pairwise_temporal",
        primitive_ids=[],
        families=[],
        description="duplicate",
    )

    monkeypatch.setattr(
        cli,
        "build_default_candidates",
        lambda compiled: [program, program],
    )

    exit_code = cli.main(ratebeer_args(tmp_path / "plan"))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "duplicate candidate program IDs" in captured.err
    assert not (tmp_path / "plan").exists()


def test_no_parquet_files_created(tmp_path: Path) -> None:
    output_dir = tmp_path / "plan"

    assert cli.main(ratebeer_args(output_dir)) == 0

    assert not list(tmp_path.rglob("*.parquet"))


def test_no_files_written_outside_output_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "plan"

    assert cli.main(ratebeer_args(output_dir)) == 0

    outside = [
        path
        for path in tmp_path.iterdir()
        if path != output_dir
    ]
    assert outside == []


def test_cli_has_no_gpu_model_or_parquet_dependency() -> None:
    names = set(cli.__dict__)

    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names


def test_unknown_dataset_task_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ratebeer_args(tmp_path / "plan")
    args[args.index("--task") + 1] = "missing-task"

    exit_code = cli.main(args)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing-task" in captured.err
    assert not (tmp_path / "plan").exists()


def test_invalid_output_path_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("x", encoding="utf-8")

    exit_code = cli.main(
        ratebeer_args(output_path, "--overwrite")
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not-a-directory" in captured.err
