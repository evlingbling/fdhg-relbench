from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.onboarding.relbench_v1 import export_relbench_v1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export official RelBench train/validation data "
            "for automatic onboarding."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument("--task-metadata-config")

    download = parser.add_mutually_exclusive_group()
    download.add_argument("--download", action="store_true")
    download.add_argument("--no-download", action="store_true")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")

    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = export_relbench_v1(
            dataset_name=args.dataset,
            task_name=args.task,
            output_root=Path(args.output_root),
            config_output=Path(args.config_output),
            task_metadata_config=(
                Path(args.task_metadata_config)
                if args.task_metadata_config
                else None
            ),
            download=bool(args.download),
            write=bool(args.write),
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("DATASET", report.dataset)
    print("TASK", report.task)
    print("STATUS", report.status)
    print("DRY_RUN", report.dry_run)
    print("REUSED", report.reused)
    print("OUTPUT_DIR", report.output_dir)
    print("CONFIG_PATH", report.config_path)
    print("RELATION_COUNT", report.relation_count)
    print("TABLE_COUNT", report.table_count)
    print("TRAIN_ROWS", report.train_rows)
    print("VALIDATION_ROWS", report.validation_rows)
    print("BLOCKERS", "|".join(report.blockers))

    return (
        0
        if report.status in {"completed", "reused", "dry_run_ready"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
