from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .ir import CompiledTask
from .programs import CandidateProgram
from .selection import ProgramScore


def write_selection_artifacts(
    *,
    compiled: CompiledTask,
    programs: list[CandidateProgram],
    scores: list[ProgramScore],
    selected: ProgramScore,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    program_lookup = {
        program.program_id: program
        for program in programs
    }

    selected_program = program_lookup[
        selected.program_id
    ]

    trace = pd.DataFrame(
        [score.to_dict() for score in scores]
    ).sort_values(
        [
            "primary_mean",
            "secondary_mean",
            "n_features_mean",
        ],
        na_position="last",
    )

    trace["selected"] = trace["program_id"].eq(
        selected.program_id
    )

    trace.to_csv(
        output_dir / "selection_trace.csv",
        index=False,
    )

    selected_payload = {
        "compiler_version": compiled.compiler_version,
        "dataset": compiled.task_spec.dataset,
        "task": compiled.task_spec.task,
        "primary_metric": selected.primary_metric,
        "secondary_metric": selected.secondary_metric,
        "selection_policy": {
            "ordering": [
                "primary_metric",
                "secondary_metric",
                "fewer_physical_features",
                "program_id",
            ],
            "primary_tolerance": 1e-12,
            "secondary_tolerance": 1e-12,
        },
        "selected_program_id": selected.program_id,
        "selected_result_variant": (
            selected.result_variant
        ),
        "selected_score": selected.primary_mean,
        "selected_n_features": (
            selected.n_features_mean
        ),
        "selected_primitive_ids": (
            selected_program.primitive_ids
        ),
        "selected_families": (
            selected_program.families
        ),
        "fallback_to_baseline": (
            selected.program_id == "baseline"
        ),
    }

    (
        output_dir / "selected_program.json"
    ).write_text(
        json.dumps(
            selected_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    selected_manifest = pd.DataFrame([
        primitive.to_dict()
        for primitive in compiled.candidate_primitives
        if primitive.primitive_id
        in set(selected_program.primitive_ids)
    ])

    if (
        not selected_manifest.empty
        and "metadata" in selected_manifest.columns
    ):
        selected_manifest["metadata"] = (
            selected_manifest["metadata"].map(
                lambda value: json.dumps(
                    value,
                    sort_keys=True,
                )
            )
        )

    selected_manifest.to_csv(
        output_dir / "selected_manifest.csv",
        index=False,
    )

    print(
        "WROTE",
        output_dir / "selection_trace.csv",
    )
    print(
        "WROTE",
        output_dir / "selected_program.json",
    )
    print(
        "WROTE",
        output_dir / "selected_manifest.csv",
    )
