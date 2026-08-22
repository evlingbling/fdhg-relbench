from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.candidate_safety import ExplicitLoweringEvidence
from fdhg.compiler.materializer import (
    CandidateMaterializationRequest,
    materialize_candidate_program,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import CandidateProgram, build_default_candidates


class CliError(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (
        CliError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one compiler candidate and emit "
            "candidate-local safety audits. Dry-run by default."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--reproduction-config",
        default="configs/reproduction/tasks.yaml",
    )
    parser.add_argument(
        "--semantics-config",
        default="configs/reproduction/task_semantics.yaml",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--source-rows-json")
    parser.add_argument("--train-target-json")
    parser.add_argument("--validation-target-json")
    parser.add_argument("--explicit-evidence-json")
    return parser


def run(args: argparse.Namespace):
    task_spec = load_task_spec(
        dataset=args.dataset,
        task=args.task,
        reproduction_config=Path(args.reproduction_config),
        semantics_config=Path(args.semantics_config),
    )
    compiled = build_candidate_program(task_spec)
    program = _select_program(
        build_default_candidates(compiled),
        args.program_id,
    )
    output_dir = (
        Path(args.output_root)
        / f"{args.dataset}_{args.task}"
        / "candidates"
        / args.program_id
    )

    if args.audit_only:
        if args.write:
            raise CliError("--audit-only is print-only in this command")
        return materialize_candidate_program(
            CandidateMaterializationRequest(
                compiled=compiled,
                program=program,
                output_dir=output_dir,
                source_rows_by_table={},
                train_target_rows=(),
                validation_target_rows=(),
                write=False,
            )
        )

    if args.write:
        source_rows = _read_json_object(
            args.source_rows_json,
            "--source-rows-json",
        )
        train_rows = _read_json_array(
            args.train_target_json,
            "--train-target-json",
        )
        validation_rows = _read_json_array(
            args.validation_target_json,
            "--validation-target-json",
        )
        evidence = _read_explicit_evidence(args.explicit_evidence_json)
    else:
        source_rows = {}
        train_rows = ()
        validation_rows = ()
        evidence = ()

    return materialize_candidate_program(
        CandidateMaterializationRequest(
            compiled=compiled,
            program=program,
            output_dir=output_dir,
            source_rows_by_table=source_rows,
            train_target_rows=train_rows,
            validation_target_rows=validation_rows,
            write=bool(args.write),
            overwrite=bool(args.overwrite),
            explicit_lowering_evidence=evidence,
            candidate_id_columns=(
                (task_spec.pairwise.target_right_key,)
                if task_spec.pairwise is not None
                else ()
            ),
        )
    )


def _select_program(
    programs: Sequence[CandidateProgram],
    program_id: str,
) -> CandidateProgram:
    for program in programs:
        if program.program_id == program_id:
            return program
    available = ", ".join(program.program_id for program in programs)
    raise CliError(
        f"unknown program ID {program_id!r}; available: {available}"
    )


def _read_json_object(
    path_value: str | None,
    option_name: str,
) -> Mapping[str, object]:
    if path_value is None:
        raise CliError(f"{option_name} is required with --write")
    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CliError(f"{option_name} must contain a JSON object")
    return payload


def _read_json_array(
    path_value: str | None,
    option_name: str,
) -> Sequence[Mapping[str, object]]:
    if path_value is None:
        raise CliError(f"{option_name} is required with --write")
    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise CliError(f"{option_name} must contain a JSON array")
    return payload


def _read_explicit_evidence(
    path_value: str | None,
) -> tuple[ExplicitLoweringEvidence, ...]:
    if path_value is None:
        return ()
    payload = json.loads(Path(path_value).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise CliError("--explicit-evidence-json must contain a JSON array")
    return tuple(
        ExplicitLoweringEvidence(
            dataset=str(row["dataset"]),
            task=str(row["task"]),
            program_id=str(row["program_id"]),
            primitive_id=str(row["primitive_id"]),
            source_table=(
                None
                if row.get("source_table") is None
                else str(row.get("source_table"))
            ),
            source_column=(
                None
                if row.get("source_column") is None
                else str(row.get("source_column"))
            ),
            output_column=(
                None
                if row.get("output_column") is None
                else str(row.get("output_column"))
            ),
            status=str(row["status"]),
            evidence_location=str(row["evidence_location"]),
            notes=tuple(str(note) for note in row.get("notes", ())),
        )
        for row in payload
    )


def _print_result(result) -> None:
    lines = [
        ("DATASET", result.dataset),
        ("TASK", result.task),
        ("PROGRAM_ID", result.program_id),
        ("DRY_RUN", result.dry_run),
        ("REUSED", result.reused),
        ("MATERIALIZABLE", result.materializable),
        ("TEMPORALLY_SAFE", result.temporally_safe),
        ("LEAKAGE_SAFE", result.leakage_safe),
        ("PROVENANCE_COMPLETE", result.provenance_complete),
        ("SELECTOR_READY", result.selector_ready),
        ("OUTPUT_DIR", result.output_dir),
        ("TRAIN_ARTIFACT", result.train_artifact or ""),
        ("VALIDATION_ARTIFACT", result.validation_artifact or ""),
        ("FEATURE_COUNT", len(result.feature_columns)),
        ("TRAIN_ROW_COUNT", result.train_row_count),
        ("VALIDATION_ROW_COUNT", result.validation_row_count),
        ("FAILURE_REASONS", "|".join(result.failure_reasons)),
    ]
    for key, value in lines:
        print(key, value)


if __name__ == "__main__":
    raise SystemExit(main())
