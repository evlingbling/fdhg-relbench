from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_realized_primitives(
    *,
    artifact_dir: Path,
    primitive_ids: list[str],
    primitive_column_bindings: (
        dict[str, tuple[str, ...]]
    ),
) -> pd.DataFrame:
    parquet_path = (
        artifact_dir
        / "target_with_dfs_agg_train.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    frame = pd.read_parquet(parquet_path)
    columns = set(frame.columns)

    rows = []

    for primitive_id in primitive_ids:
        expected_columns = primitive_column_bindings.get(
            primitive_id
        )

        if not expected_columns:
            rows.append({
                "primitive_id": primitive_id,
                "realized": False,
                "expected_columns": "",
                "matched_columns": "",
                "artifact_dir": str(artifact_dir),
                "failure_reason": (
                    "missing_physical_binding"
                ),
            })
            continue

        matched_columns = [
            column
            for column in expected_columns
            if column in columns
        ]

        realized = (
            len(matched_columns)
            == len(expected_columns)
        )

        rows.append({
            "primitive_id": primitive_id,
            "realized": realized,
            "expected_columns": "|".join(
                expected_columns
            ),
            "matched_columns": "|".join(
                matched_columns
            ),
            "artifact_dir": str(artifact_dir),
            "failure_reason": (
                ""
                if realized
                else "physical_column_missing"
            ),
        })

    audit = pd.DataFrame(rows)

    missing = audit.loc[
        ~audit["realized"],
        [
            "primitive_id",
            "failure_reason",
            "expected_columns",
        ],
    ]

    if not missing.empty:
        raise ValueError(
            "Selected compiler primitives are not "
            f"realized in {parquet_path}:\n"
            + missing.to_string(index=False)
        )

    return audit
