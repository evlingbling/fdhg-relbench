from __future__ import annotations

import argparse
from pathlib import Path

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.existing_backend import (
    resolve_existing_artifacts,
)
from fdhg.compiler.discovery import (
    discover_existing_artifacts,
)
from fdhg.compiler.manifest import (
    write_candidate_artifacts,
)
from fdhg.compiler.planner import (
    build_candidate_program,
)
from fdhg.compiler.programs import (
    build_block_candidates,
)
from fdhg.compiler.provenance import (
    validate_realized_primitives,
)
from fdhg.compiler.selected_manifest import (
    write_selection_artifacts,
)
from fdhg.compiler.selection import (
    load_program_score,
    select_program,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a relational task into an FDHG "
            "residual feature program."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--config",
        default="configs/reproduction/tasks.yaml",
    )
    parser.add_argument(
        "--semantics-config",
        default=(
            "configs/reproduction/"
            "task_semantics.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--plan-only",
        action="store_true",
    )
    parser.add_argument(
        "--select-existing",
        action="store_true",
        help=(
            "Select among existing canonical candidate "
            "artifacts and metric runs."
        ),
    )

    parser.add_argument(
        "--discover-existing",
        action="store_true",
        help=(
            "Automatically discover existing result variants, "
            "artifacts, and logical primitive bindings."
        ),
    )

    args = parser.parse_args()

    if args.select_existing and args.discover_existing:
        parser.error(
            "--select-existing and --discover-existing "
            "cannot be used together"
        )

    task_spec = load_task_spec(
        dataset=args.dataset,
        task=args.task,
        reproduction_config=Path(args.config),
        semantics_config=Path(args.semantics_config),
    )

    compiled = build_candidate_program(task_spec)
    output_dir = Path(args.output)

    write_candidate_artifacts(
        compiled,
        output_dir,
    )

    backend = None
    realized_bindings = None

    if args.discover_existing:
        backend = discover_existing_artifacts(
            dataset=args.dataset,
            task=args.task,
            seeds=list(task_spec.seeds),
        )
    elif args.select_existing:
        backend = resolve_existing_artifacts(
            args.dataset,
            args.task,
        )

    if backend is not None:
        realized_bindings = {
            program_id: list(
                artifact.realized_primitive_ids
            )
            for program_id, artifact in backend.items()
        }

    programs = build_block_candidates(
        compiled,
        realized_primitive_ids_by_program=(
            realized_bindings
        ),
    )

    print("\n=== FDHG compiler plan ===")
    print("dataset/task:", args.dataset, args.task)
    print("problem_type:", task_spec.problem_type)
    print("entity_key:", task_spec.entity_key)
    print("child_table:", task_spec.child_table)
    print("child_time_col:", task_spec.child_time_col)
    print("horizon_days:", task_spec.horizon_days)
    print("feature_budget:", task_spec.feature_budget)
    print(
        "candidate primitives:",
        len(compiled.candidate_primitives),
    )
    print("candidate programs:", len(programs))

    if backend is not None:
        scores = []

        for program in programs:
            if program.program_id not in backend:
                continue

            artifact = backend[program.program_id]

            if task_spec.primary_metric is None:
                raise ValueError(
                    "primary_metric is missing from "
                    "task semantics"
                )

            scores.append(
                load_program_score(
                    program_id=program.program_id,
                    result_root=artifact.result_root,
                    result_variant=(
                        artifact.result_variant
                    ),
                    primary_metric=(
                        task_spec.primary_metric
                    ),
                    secondary_metric=(
                        task_spec.secondary_metric
                    ),
                    seeds=list(task_spec.seeds),
                )
            )

        if task_spec.metric_direction is None:
            raise ValueError(
                "metric_direction is missing from "
                "task semantics"
            )

        selected = select_program(
            scores,
            metric_direction=(
                task_spec.metric_direction
            ),
        )

        selected_artifact = backend[
            selected.program_id
        ]

        selected_program = next(
            program
            for program in programs
            if program.program_id == selected.program_id
        )

        provenance = validate_realized_primitives(
            artifact_dir=selected_artifact.artifact_dir,
            primitive_ids=selected_program.primitive_ids,
            primitive_column_bindings=(
                selected_artifact
                .primitive_column_bindings
            ),
        )

        provenance_path = (
            output_dir / "lowering_provenance_audit.csv"
        )
        provenance.to_csv(
            provenance_path,
            index=False,
        )

        write_selection_artifacts(
            compiled=compiled,
            programs=programs,
            scores=scores,
            selected=selected,
            output_dir=output_dir,
        )

        print("WROTE", provenance_path)
        print("\n=== Selected program ===")
        print("program_id:", selected.program_id)
        print(
            "result_variant:",
            selected.result_variant,
        )
        print(
            "primary:",
            selected.primary_metric,
            selected.primary_mean,
        )
        print(
            "n_features:",
            selected.n_features_mean,
        )
        return

    if args.plan_only:
        return

    raise NotImplementedError(
        "Native compiler lowering is not connected yet. "
        "Use --plan-only or --select-existing."
    )


if __name__ == "__main__":
    main()
