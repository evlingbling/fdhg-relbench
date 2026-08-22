from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates
from fdhg.compiler.selection import (
    CandidateSelectionDecision,
    CandidateSelectionPolicy,
    load_candidate_validation_results,
    select_candidate_program,
)


class CliError(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        decision = run(args)
    except (
        CliError,
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_decision(decision)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a compiler candidate from validation evidence "
            "without running materialization or training."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--validation-results",
        required=True,
        help=(
            "Canonical validation CSV with explicit dataset, task, "
            "program_id, split, metric, score, and evidence columns."
        ),
    )
    parser.add_argument(
        "--reproduction-config",
        default="configs/reproduction/tasks.yaml",
    )
    parser.add_argument(
        "--semantics-config",
        default="configs/reproduction/task_semantics.yaml",
    )
    parser.add_argument(
        "--baseline-program-id",
        default="baseline",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
    )
    return parser


def run(args: argparse.Namespace) -> CandidateSelectionDecision:
    task_spec = load_task_spec(
        dataset=args.dataset,
        task=args.task,
        reproduction_config=Path(args.reproduction_config),
        semantics_config=Path(args.semantics_config),
    )

    if task_spec.primary_metric is None:
        raise CliError(
            f"primary metric is not configured for "
            f"{args.dataset}/{args.task}"
        )

    if task_spec.metric_direction is None:
        raise CliError(
            f"metric direction is not configured for "
            f"{args.dataset}/{args.task}"
        )

    compiled = build_candidate_program(task_spec)
    programs = build_default_candidates(compiled)
    records = load_candidate_validation_results(
        Path(args.validation_results)
    )
    policy = CandidateSelectionPolicy(
        dataset=args.dataset,
        task=args.task,
        primary_metric=task_spec.primary_metric,
        metric_direction=task_spec.metric_direction,
        baseline_program_id=args.baseline_program_id,
        min_improvement=args.min_improvement,
    )
    return select_candidate_program(
        programs,
        records,
        policy,
    )


def print_decision(
    decision: CandidateSelectionDecision,
) -> None:
    print(f"SELECTED_PROGRAM_ID {decision.selected_program_id}")
    print(f"SELECTED_SCORE {decision.selected_score}")
    print(f"BASELINE_PROGRAM_ID {decision.baseline_program_id}")
    print(f"BASELINE_SCORE {decision.baseline_score}")
    print(
        "IMPROVEMENT_OVER_BASELINE "
        f"{decision.improvement_over_baseline}"
    )
    print(f"METRIC {decision.metric}")
    print(f"METRIC_DIRECTION {decision.metric_direction}")
    print(f"FALLBACK_OCCURRED {decision.fallback_occurred}")
    print(f"FALLBACK_REASON {decision.fallback_reason or ''}")

    print("RANKED_CANDIDATES")
    for candidate in decision.ranked_candidates:
        print(
            "\t".join([
                candidate.program_id,
                str(candidate.validation_score),
                str(candidate.improvement_over_baseline),
                str(candidate.n_features),
                str(candidate.added_features),
                candidate.evidence_location,
            ])
        )

    print("REJECTED_CANDIDATES")
    for candidate in decision.rejected_candidates:
        print(
            "\t".join([
                candidate.program_id,
                "|".join(candidate.rejection_reasons),
                candidate.evidence_location,
            ])
        )

    print("EVIDENCE_LOCATIONS")
    for location in decision.evidence_locations:
        print(location)


if __name__ == "__main__":
    raise SystemExit(main())
