#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/experiments/paper_artifact_manifest.yaml"


@dataclass
class Check:
    experiment: str
    path: str
    present: bool
    kind: str


def has_glob(path_text: str) -> bool:
    if "*" not in path_text and "?" not in path_text and "[" not in path_text:
        return (ROOT / path_text).exists()
    return any(ROOT.glob(path_text))


def main() -> int:
    with MANIFEST.open() as f:
        manifest = yaml.safe_load(f)

    checks: list[Check] = []

    for experiment, spec in manifest["experiments"].items():
        for field, kind in (
            ("required_inputs", "required input"),
            ("curated_outputs", "curated output"),
            ("outputs", "output"),
        ):
            for path_text in spec.get(field, []):
                checks.append(
                    Check(
                        experiment=experiment,
                        path=path_text,
                        present=has_glob(path_text),
                        kind=kind,
                    )
                )

    width = max((len(c.experiment) for c in checks), default=10)
    missing_required = 0

    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print()
    for check in checks:
        marker = "OK" if check.present else "MISSING"
        print(
            f"{marker:7}  {check.experiment:<{width}}  "
            f"{check.kind:<14}  {check.path}"
        )
        if not check.present and check.kind == "required input":
            missing_required += 1

    print()
    print(f"Missing required inputs: {missing_required}")

    if missing_required:
        print(
            "Curated outputs may be inspectable, but the affected tables "
            "cannot be rebuilt from this repository snapshot."
        )
        return 1

    print("All declared raw inputs are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
