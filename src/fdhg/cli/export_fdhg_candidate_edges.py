from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_EDGE_FIELDS = ("edge_id", "source_table", "lhs_columns", "rhs_column")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export complete FDHG candidate edge definitions for historical replay."
    )
    parser.add_argument("--input-output-dir", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        edges = load_exportable_candidate_edges(args.input_output_dir)
        write_candidate_edges(args.output_file, edges)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"EXPORTED_FDHG_CANDIDATE_EDGES {len(edges)}")
    print(f"OUTPUT_FILE {args.output_file}")
    return 0


def load_exportable_candidate_edges(input_output_dir: Path) -> list[dict[str, Any]]:
    root = Path(input_output_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    explicit_empty_candidate_pool = False

    for filename, keys in (
        (
            "candidate_discovery.json",
            (
                "accepted_edges",
                "candidate_edges",
                "accepted_fdhg_edges",
                "edges",
            ),
        ),
        ("manifest.json", ("accepted_fdhg_edges",)),
    ):
        path = root / filename
        if not path.exists():
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue

        for key in keys:
            if key not in payload:
                continue

            value = payload.get(key)
            if not isinstance(value, list):
                continue

            if not value:
                explicit_empty_candidate_pool = True
                continue

            rows = [
                dict(row)
                for row in value
                if isinstance(row, Mapping)
            ]

            if rows:
                candidates.append(
                    (f"{filename}:{key}", rows)
                )

    for source, edges in candidates:
        if _complete_edge_definitions(edges):
            return edges

        missing = sorted({
            field
            for edge in edges
            for field in REQUIRED_EDGE_FIELDS
            if not str(edge.get(field, "")).strip()
        })

        raise ValueError(
            "incomplete_fdhg_candidate_edge_definitions:"
            f"{source}:missing={','.join(missing)}"
        )

    if explicit_empty_candidate_pool:
        return []

    raise ValueError(
        "no_complete_fdhg_candidate_edge_definitions_found:"
        "need manifest.json accepted_fdhg_edges or "
        "candidate_discovery.json accepted_edges"
    )


def write_candidate_edges(output_file: Path, edges: Sequence[Mapping[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix.lower()
    if suffix == ".json":
        output_file.write_text(json.dumps(list(edges), indent=2, sort_keys=True, default=str), encoding="utf-8")
        return
    if suffix == ".csv":
        fieldnames = list(dict.fromkeys([key for edge in edges for key in edge.keys()]))
        with output_file.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for edge in edges:
                writer.writerow({
                    key: _csv_cell(edge.get(key, ""))
                    for key in fieldnames
                })
        return
    raise ValueError("output_file_must_end_with_json_or_csv")


def _complete_edge_definitions(edges: Sequence[Mapping[str, Any]]) -> bool:
    return bool(edges) and all(
        all(str(edge.get(field, "")).strip() for field in REQUIRED_EDGE_FIELDS)
        for edge in edges
    )


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
