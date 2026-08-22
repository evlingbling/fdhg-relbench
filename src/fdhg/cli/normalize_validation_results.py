from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.compiler.validation_results import (
    audit_validation_sources,
    iter_default_audit_paths,
    normalize_validation_artifact,
    write_canonical_validation_csv,
)


class CliError(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run(args)
    except (
        CliError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize explicit validation artifacts into the "
            "canonical candidate-selection schema without running "
            "experiments."
        )
    )
    parser.add_argument("--dataset")
    parser.add_argument("--task")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="CSV source artifact to inspect or normalize.",
    )
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "Print deterministic source audit instead of canonical "
            "records."
        ),
    )
    parser.add_argument(
        "--audit-default-results",
        action="store_true",
        help=(
            "Audit results/compiler and results/paper_tables CSVs "
            "read-only."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    sources = [Path(source) for source in args.source]

    if args.audit_default_results:
        sources.extend(iter_default_audit_paths())

    if not sources:
        raise CliError("at least one --source is required")

    if args.audit:
        _print_audit(sources)
        return 0

    records = []
    for source in sources:
        report = normalize_validation_artifact(
            source,
            dataset=args.dataset,
            task=args.task,
        )
        records.extend(report.normalized_records)

    if args.output:
        output = Path(args.output)
        _validate_output_path(output, overwrite=args.overwrite)
        with output.open("w", encoding="utf-8", newline="") as handle:
            write_canonical_validation_csv(records, handle)
    else:
        write_canonical_validation_csv(records, sys.stdout)

    return 0


def _print_audit(sources: Sequence[Path]) -> None:
    rows = audit_validation_sources(sources)
    print(
        "\t".join([
            "source_path",
            "schema",
            "validation_only_status",
            "task_identity_explicit",
            "program_identity_explicit",
            "safety_evidence_available",
            "adapter_supported",
            "reason",
        ])
    )
    for row in rows:
        print(
            "\t".join([
                row.source_path,
                row.schema,
                row.validation_only_status,
                str(row.task_identity_explicit),
                str(row.program_identity_explicit),
                str(row.safety_evidence_available),
                str(row.adapter_supported),
                row.reason,
            ])
        )


def _validate_output_path(
    output: Path,
    *,
    overwrite: bool,
) -> None:
    paper_tables = Path("results/paper_tables").resolve()
    resolved = output.resolve()
    if (
        resolved == paper_tables
        or paper_tables in resolved.parents
    ):
        raise CliError(
            "refusing to write normalized selector input under "
            "results/paper_tables"
        )

    if output.exists() and not overwrite:
        raise FileExistsError(output)

    if output.parent.exists() and not output.parent.is_dir():
        raise NotADirectoryError(output.parent)


if __name__ == "__main__":
    raise SystemExit(main())
