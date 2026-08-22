from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Sequence

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materialization_io import (
    write_materialization_plan_json,
    write_primitive_bindings_json,
    write_temporal_safety_audit_csv,
)
from fdhg.compiler.materializer import (
    LoweringMode,
    plan_candidate_materialization,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import (
    CandidateProgram,
    build_default_candidates,
)


PLAN_JSON_NAME = "materialization_plan.json"
BINDINGS_JSON_NAME = "primitive_column_bindings.json"
AUDIT_CSV_NAME = "temporal_safety_audit.csv"


class CliError(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a task and write read-only "
            "materialization provenance artifacts."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reproduction-config",
        default="configs/reproduction/tasks.yaml",
    )
    parser.add_argument(
        "--semantics-config",
        default=(
            "configs/reproduction/"
            "task_semantics.yaml"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--allow-external-provider",
        action="store_true",
    )
    parser.add_argument("--compiler-version")
    parser.add_argument("--git-commit")
    parser.add_argument("--created-at-utc")
    parser.add_argument("--source")
    return parser


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    _validate_output_path(output_dir, overwrite=args.overwrite)

    task_spec = load_task_spec(
        dataset=args.dataset,
        task=args.task,
        reproduction_config=Path(
            args.reproduction_config
        ),
        semantics_config=Path(args.semantics_config),
    )
    compiled = build_candidate_program(task_spec)
    programs = build_default_candidates(compiled)
    program = _select_program(programs, args.program)

    plan = plan_candidate_materialization(
        compiled,
        program,
    )
    _validate_plan(
        plan,
        allow_external_provider=(
            args.allow_external_provider
        ),
    )

    metadata = {
        "dataset": args.dataset,
        "task": args.task,
        "compiler_version": args.compiler_version,
        "git_commit": args.git_commit,
        "created_at_utc": args.created_at_utc,
        "source": args.source,
    }
    paths = _write_output_set(
        plan=plan,
        output_dir=output_dir,
        metadata=metadata,
        overwrite=args.overwrite,
    )
    _print_summary(plan, paths)
    return 0


def _select_program(
    programs: Sequence[CandidateProgram],
    program_id: str,
) -> CandidateProgram:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for program in programs:
        if program.program_id in seen:
            duplicates.add(program.program_id)
        seen.add(program.program_id)

    if duplicates:
        raise CliError(
            "duplicate candidate program IDs: "
            + ", ".join(sorted(duplicates))
        )

    for program in programs:
        if program.program_id == program_id:
            return program

    available = ", ".join(
        program.program_id for program in programs
    )
    raise CliError(
        f"unknown program ID {program_id!r}; "
        f"available programs: {available}"
    )


def _validate_plan(
    plan,
    *,
    allow_external_provider: bool,
) -> None:
    if not plan.materializable:
        raise CliError(
            f"plan is not materializable: {plan.program_id}"
        )

    if not plan.temporally_safe:
        raise CliError(
            f"plan is not temporally safe: {plan.program_id}"
        )

    if (
        plan.requires_external_provider
        and not allow_external_provider
    ):
        raise CliError(
            "plan requires an external provider; rerun with "
            "--allow-external-provider to write provenance"
        )


def _validate_output_path(
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if not overwrite:
            raise FileExistsError(output_dir)

    parent = output_dir.parent
    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError(parent)


def _write_output_set(
    *,
    plan,
    output_dir: Path,
    metadata: dict[str, object],
    overwrite: bool,
) -> dict[str, Path]:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=parent,
        )
    )
    final_paths = {
        "plan": output_dir / PLAN_JSON_NAME,
        "bindings": output_dir / BINDINGS_JSON_NAME,
        "audit": output_dir / AUDIT_CSV_NAME,
    }

    try:
        _write_staged_files(
            plan=plan,
            staging=staging,
            metadata=metadata,
        )
        _validate_staged_files(staging)
        _finalize_staged_directory(
            staging,
            output_dir,
            overwrite=overwrite,
        )
        staging = None
        return final_paths
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _write_staged_files(
    *,
    plan,
    staging: Path,
    metadata: dict[str, object],
) -> None:
    write_materialization_plan_json(
        plan,
        staging / PLAN_JSON_NAME,
        metadata=metadata,
    )
    write_primitive_bindings_json(
        plan,
        staging / BINDINGS_JSON_NAME,
        metadata=metadata,
    )
    write_temporal_safety_audit_csv(
        plan,
        staging / AUDIT_CSV_NAME,
        metadata=metadata,
    )


def _finalize_staged_directory(
    staging: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    backup: Path | None = None

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(output_dir)
        backup = _unique_backup_path(output_dir)
        _replace_path(output_dir, backup)

    try:
        _replace_path(staging, output_dir)
    except OSError as finalize_error:
        if backup is None:
            raise

        try:
            _replace_path(backup, output_dir)
        except OSError as restore_error:
            raise CliError(
                "failed to finalize output directory: "
                f"{finalize_error}; failed to restore previous "
                f"output directory from backup {backup}: "
                f"{restore_error}"
            ) from restore_error

        raise

    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _unique_backup_path(output_dir: Path) -> Path:
    parent = output_dir.parent

    while True:
        candidate = (
            parent
            / f".{output_dir.name}.{uuid.uuid4().hex}.backup"
        )
        if not candidate.exists():
            return candidate


def _replace_path(source: Path, destination: Path) -> None:
    source.replace(destination)


def _validate_staged_files(staging: Path) -> None:
    missing = [
        name
        for name in [
            PLAN_JSON_NAME,
            BINDINGS_JSON_NAME,
            AUDIT_CSV_NAME,
        ]
        if not (staging / name).is_file()
    ]

    if missing:
        raise CliError(
            "staged output set is incomplete: "
            + ", ".join(missing)
        )


def _print_summary(
    plan,
    paths: dict[str, Path],
) -> None:
    counts = Counter(
        step.lowering_mode for step in plan.steps
    )
    lines = [
        ("PROGRAM_ID", plan.program_id),
        ("STEP_COUNT", len(plan.steps)),
        ("GENERATE", counts[LoweringMode.GENERATE]),
        ("PASSTHROUGH", counts[LoweringMode.PASSTHROUGH]),
        ("EXTERNAL", counts[LoweringMode.EXTERNAL]),
        ("UNSUPPORTED", counts[LoweringMode.UNSUPPORTED]),
        ("MATERIALIZABLE", plan.materializable),
        ("TEMPORALLY_SAFE", plan.temporally_safe),
        (
            "REQUIRES_EXTERNAL_PROVIDER",
            plan.requires_external_provider,
        ),
        ("PLAN_JSON", paths["plan"]),
        ("BINDINGS_JSON", paths["bindings"]),
        ("AUDIT_CSV", paths["audit"]),
    ]

    for key, value in lines:
        print(key, value)


if __name__ == "__main__":
    raise SystemExit(main())
