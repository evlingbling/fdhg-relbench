from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.compiler.candidate_evaluator import (
    CandidateEvaluatorConfig,
    SubprocessCandidateEvaluator,
)
from fdhg.compiler.task_pipeline import (
    TaskPipelineRequest,
    run_task_pipeline,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        evaluator = None
        if args.evaluator == "existing-script":
            evaluator = SubprocessCandidateEvaluator(
                config=CandidateEvaluatorConfig(
                    reproduction_config=Path(args.reproduction_config),
                    python_executable=(
                        Path(args.python_executable)
                        if args.python_executable
                        else None
                    ),
                    device=args.device,
                    timeout_seconds=args.evaluation_timeout,
                    overwrite=args.overwrite_evaluations,
                )
            )
        report = run_task_pipeline(
            TaskPipelineRequest(
                dataset=args.dataset,
                task=args.task,
                output_root=Path(args.output_root),
                result_root=Path(args.result_root),
                seeds=tuple(args.seeds),
                mode=args.mode,
                program_ids=tuple(args.program_id or ()),
                exclude_program_ids=tuple(args.exclude_program_id or ()),
                baseline_only=args.baseline_only,
                write_materialization=args.write_materialization,
                run_validation=args.run_validation,
                select=args.select,
                overwrite=args.overwrite,
                overwrite_pipeline_output=args.overwrite_pipeline_output,
                reproduction_config=Path(args.reproduction_config),
                semantics_config=Path(args.semantics_config),
            ),
            evaluator=evaluator,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
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
            "Run the validation-driven compiler task pipeline. "
            "Dry-run by default."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44])
    parser.add_argument("--program-id", action="append")
    parser.add_argument("--exclude-program-id", action="append")
    parser.add_argument("--baseline-only", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    modes.add_argument(
        "--materialize-only",
        dest="mode",
        action="store_const",
        const="materialize-only",
    )
    modes.add_argument(
        "--validation-only",
        dest="mode",
        action="store_const",
        const="validation-only",
    )
    modes.add_argument(
        "--selection-only",
        dest="mode",
        action="store_const",
        const="selection-only",
    )
    modes.add_argument(
        "--through-materialization",
        dest="mode",
        action="store_const",
        const="through-materialization",
    )
    modes.add_argument(
        "--through-validation",
        dest="mode",
        action="store_const",
        const="through-validation",
    )
    modes.add_argument("--full", dest="mode", action="store_const", const="full")
    parser.set_defaults(mode="dry-run")
    parser.add_argument("--write-materialization", action="store_true")
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument(
        "--evaluator",
        choices=("existing-script",),
        help=(
            "Production validation evaluator using the existing evaluator "
            "scripts. Omit to keep validation blocked/fail-closed."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-timeout", type=int)
    parser.add_argument("--python-executable")
    parser.add_argument("--overwrite-evaluations", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-pipeline-output", action="store_true")
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
    print("REQUESTED_MODE", report.requested_mode)
    print("PIPELINE_STATUS", report.pipeline_status)
    print("CANDIDATE_IDS", "|".join(report.candidate_ids))
    print("DISCOVERED_CANDIDATES", "|".join(report.discovered_candidates))
    print("CANONICAL_VALIDATION_PATH", report.canonical_validation_path or "")
    if report.selection_decision is not None:
        print("SELECTED_PROGRAM_ID", report.selection_decision.selected_program_id)
        print("FALLBACK_OCCURRED", report.selection_decision.fallback_occurred)
    print("STAGES")
    for stage in report.stages:
        print(
            "\t".join([
                stage.stage,
                stage.status,
                "|".join(stage.blockers),
                stage.failure_reason,
            ])
        )


if __name__ == "__main__":
    raise SystemExit(main())
