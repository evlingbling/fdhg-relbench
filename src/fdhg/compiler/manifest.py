from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .ir import CompiledTask


def write_candidate_artifacts(
    compiled: CompiledTask,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    program_path = output_dir / "candidate_program.json"
    manifest_path = output_dir / "candidate_manifest.csv"
    safety_path = output_dir / "temporal_safety_audit.csv"

    program_path.write_text(
        json.dumps(
            compiled.to_dict(),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    rows = [
        primitive.to_dict()
        for primitive in compiled.candidate_primitives
    ]

    manifest = pd.DataFrame(rows)

    if not manifest.empty:
        manifest["metadata"] = manifest["metadata"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )

    manifest.to_csv(manifest_path, index=False)

    safety_columns = [
        "primitive_id",
        "family",
        "source_table",
        "group_key",
        "event_time_col",
        "window_days",
        "temporal_predicate",
        "temporally_safe",
    ]

    safety = manifest[
        [
            column
            for column in safety_columns
            if column in manifest.columns
        ]
    ].copy()

    safety.to_csv(safety_path, index=False)

    print("WROTE", program_path)
    print("WROTE", manifest_path)
    print("WROTE", safety_path)
