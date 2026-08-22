from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from fdhg.onboarding import onboard_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile and onboard an explicit relational dataset config."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = onboard_dataset(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            write=bool(args.write),
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("DATASET", report.dataset)
    print("TASK", report.task)
    print("STATUS", report.status)
    print("DRY_RUN", report.dry_run)
    print("REUSED", report.reused)
    print("OUTPUT_DIR", report.output_dir)
    print("BLOCKERS", "|".join(report.blockers))
    return 0 if report.status in {"completed", "reused", "dry_run_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
