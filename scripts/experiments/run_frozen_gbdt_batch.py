#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


SEEDS = [41, 42, 43, 44]
MODELS = ["xgboost", "catboost"]


def terminal_status(
    output_dir: Path,
    *,
    dataset: str,
    task: str,
    model: str,
    seed: int,
    variant: str,
) -> str | None:
    path = output_dir / "metrics.json"

    if not path.exists():
        return None

    try:
        result = json.loads(path.read_text())
    except Exception:
        return None

    valid_identity = (
        result.get("dataset") == dataset
        and result.get("task") == task
        and result.get("predictor") == model
        and int(result.get("seed", -1)) == seed
        and result.get("selected_variant") == variant
    )

    if not valid_identity:
        return None

    # Historical completed artifacts predate the
    # explicit status field.
    status = str(
        result.get("status", "completed")
    )

    if status not in {
        "completed",
        "skipped",
    }:
        return None

    return status


def run_one(job):
    (
        evaluator,
        matrix_dir,
        output_dir,
        model,
        seed,
        threads,
    ) = job

    cmd = [
        sys.executable,
        str(evaluator),
        "--matrix-dir",
        str(matrix_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--seed",
        str(seed),
        "--threads",
        str(threads),
    ]

    env = dict(os.environ)

    # Avoid nested BLAS/OpenMP oversubscription.
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    proc = subprocess.run(
        cmd,
        env=env,
    )

    return (
        matrix_dir.name,
        model,
        seed,
        proc.returncode,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("outputs/predictor-generalization/frozen-matrices"),
    )

    ap.add_argument(
        "--result-root",
        type=Path,
        default=Path("outputs/predictor-generalization/frozen-gbdt"),
    )

    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--threads-per-model",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--write",
        action="store_true",
    )

    args = ap.parse_args()

    evaluator = Path(
        "scripts/evaluate/evaluate_frozen_gbdt.py"
    )

    if not evaluator.exists():
        raise FileNotFoundError(evaluator)

    manifests = sorted(
        args.matrix_root.glob(
            "*/manifest.json"
        )
    )

    print("TASK_MANIFESTS =", len(manifests))

    jobs = []
    existing_completed = 0
    existing_skipped = 0

    for manifest_path in manifests:
        manifest = json.loads(
            manifest_path.read_text()
        )

        matrix_dir = manifest_path.parent

        dataset = str(
            manifest["dataset"]
        )
        task = str(
            manifest["task"]
        )
        variant = str(
            manifest["selected_variant"]
        )

        for model in MODELS:
            for seed in SEEDS:
                output_dir = (
                    args.result_root
                    / matrix_dir.name
                    / model
                    / f"seed_{seed}"
                )

                status = terminal_status(
                    output_dir,
                    dataset=dataset,
                    task=task,
                    model=model,
                    seed=seed,
                    variant=variant,
                )

                if status is not None:
                    if status == "completed":
                        existing_completed += 1
                        label = "READY_EXISTING"
                    else:
                        existing_skipped += 1
                        label = "READY_SKIPPED"

                    print(
                        label,
                        dataset,
                        task,
                        model,
                        seed,
                    )

                    continue

                jobs.append(
                    (
                        evaluator,
                        matrix_dir,
                        output_dir,
                        model,
                        seed,
                        args.threads_per_model,
                    )
                )

    expected = (
        len(manifests)
        * len(MODELS)
        * len(SEEDS)
    )

    print()
    print("EXPECTED_RUNS =", expected)
    print(
        "EXISTING_COMPLETED =",
        existing_completed,
    )
    print(
        "EXISTING_SKIPPED =",
        existing_skipped,
    )
    print("NEEDS_RUN =", len(jobs))

    terminal_existing = (
        existing_completed
        + existing_skipped
    )

    if terminal_existing + len(jobs) != expected:
        raise RuntimeError(
            "Run accounting mismatch"
        )

    if not args.write:
        print(
            "WRITE_ENABLED=0 "
            "(batch preflight only)"
        )
        return

    failures = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.jobs
    ) as executor:

        futures = [
            executor.submit(
                run_one,
                job,
            )
            for job in jobs
        ]

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):
            slug, model, seed, rc = (
                future.result()
            )

            if rc == 0:
                print(
                    "DONE",
                    slug,
                    model,
                    seed,
                )
            else:
                print(
                    "FAILED",
                    slug,
                    model,
                    seed,
                    rc,
                )

                failures.append(
                    (
                        slug,
                        model,
                        seed,
                        rc,
                    )
                )

    print()
    print("BATCH FINISHED")
    print("FAILURES =", len(failures))

    for failure in failures:
        print(failure)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
