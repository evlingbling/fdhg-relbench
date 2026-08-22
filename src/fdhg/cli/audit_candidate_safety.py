from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.compiler.candidate_safety import (
    build_candidate_safety_audit_report,
    write_audit_csv,
)
from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materializer import plan_candidate_materialization
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates


class CliError(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run(args)
        if args.output_dir:
            _write_report(
                report,
                Path(args.output_dir),
                overwrite=args.overwrite,
            )
        else:
            _print_report(report)
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

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate task-scoped, candidate-scoped static safety "
            "audits without running materialization or training."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reproduction-config",
        default="configs/reproduction/tasks.yaml",
    )
    parser.add_argument(
        "--semantics-config",
        default="configs/reproduction/task_semantics.yaml",
    )
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
    plan = plan_candidate_materialization(compiled, program)
    candidate_root = Path(args.candidate_root)
    feature_columns = _read_feature_columns(candidate_root)

    return build_candidate_safety_audit_report(
        dataset=args.dataset,
        task=args.task,
        plan=plan,
        feature_columns=feature_columns,
        label_col=task_spec.label_col,
        candidate_id_columns=(
            [task_spec.pairwise.target_right_key]
            if task_spec.pairwise is not None
            else []
        ),
        surrogate_key_columns=(
            "__row_id",
            "primary_key",
            "__fdhg_row_id",
        ),
    )


def _select_program(programs, program_id: str):
    for program in programs:
        if program.program_id == program_id:
            return program
    available = ", ".join(program.program_id for program in programs)
    raise CliError(
        f"unknown program ID {program_id!r}; available: {available}"
    )


def _read_feature_columns(candidate_root: Path) -> tuple[str, ...] | None:
    parquet_path = candidate_root / "target_with_dfs_agg_train.parquet"
    if not parquet_path.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    schema = pq.read_schema(parquet_path)
    return tuple(schema.names)


def _print_report(report) -> None:
    for name, audit in [
        ("TEMPORAL_SAFETY", report.temporal),
        ("LEAKAGE_SAFETY", report.leakage),
        ("LOWERING_PROVENANCE", report.provenance),
    ]:
        print(name)
        for row in audit.rows:
            print(
                "\t".join([
                    row.dataset,
                    row.task,
                    row.program_id,
                    row.audit_type,
                    row.primitive_id,
                    row.status,
                    str(row.passed),
                    row.rejection_reason,
                    row.evidence_location,
                ])
            )


def _write_report(report, output_dir: Path, *, overwrite: bool) -> None:
    _validate_output_dir(output_dir, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, audit in [
        ("temporal_safety_audit.csv", report.temporal),
        ("leakage_safety_audit.csv", report.leakage),
        ("lowering_provenance_audit.csv", report.provenance),
    ]:
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            write_audit_csv(audit.rows, handle)


def _validate_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    paper_tables = Path("results/paper_tables").resolve()
    resolved = output_dir.resolve()
    if resolved == paper_tables or paper_tables in resolved.parents:
        raise CliError("refusing to write under results/paper_tables")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(output_dir)
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise NotADirectoryError(output_dir.parent)


if __name__ == "__main__":
    raise SystemExit(main())
