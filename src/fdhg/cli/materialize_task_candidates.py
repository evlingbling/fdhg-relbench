from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.compiler.materializer import (
    TaskCandidateMaterializationRequest,
    materialize_task_candidates,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = materialize_task_candidates(
            TaskCandidateMaterializationRequest(
                dataset=args.dataset,
                task=args.task,
                output_root=Path(args.output_root),
                reproduction_config=Path(args.reproduction_config),
                semantics_config=Path(args.semantics_config),
                program_ids=tuple(args.program_id or ()),
                exclude_program_ids=tuple(args.exclude_program_id or ()),
                baseline_only=bool(args.baseline_only),
                write=bool(args.write),
                overwrite=bool(args.overwrite),
            )
        )
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
    _print_report(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize task default compiler candidates from explicit "
            "prepared artifacts. Dry-run by default."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--program-id", action="append")
    parser.add_argument("--exclude-program-id", action="append")
    parser.add_argument("--baseline-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
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


def _print_report(report) -> None:
    print("DATASET", report.dataset)
    print("TASK", report.task)
    print("DRY_RUN", report.dry_run)
    print("INPUT_RESOLVED", report.input_resolved)
    print("PUBLISHED_COUNT", report.published_count)
    print("REUSED_COUNT", report.reused_count)
    print("BLOCKED_COUNT", report.blocked_count)
    print("FAILED_COUNT", report.failed_count)
    print("EVIDENCE_LOCATIONS", "|".join(report.evidence_locations))
    if report.input_blockers:
        print("INPUT_BLOCKERS", "|".join(report.input_blockers))
    print(
        "\t".join([
            "PROGRAM_ID",
            "STATUS",
            "PRIMITIVE_COUNT",
            "LOWERING_FEASIBLE",
            "OUTPUT_DIR",
            "BLOCKERS",
        ])
    )
    for outcome in report.outcomes:
        print(
            "\t".join([
                outcome.program_id,
                outcome.status,
                str(outcome.primitive_count),
                str(outcome.lowering_feasible),
                str(outcome.output_dir),
                "|".join(outcome.blockers),
            ])
        )


if __name__ == "__main__":
    raise SystemExit(main())
