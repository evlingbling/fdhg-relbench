from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/reproduction/tasks.yaml"

AUTHOR_CATEGORY_RESIDUALS = [
    "fdhg::author_category::majority_confidence",
    "fdhg::author_category::entropy",
    "fdhg::author_category::last_primary_category",
]


def run(command: list[str]) -> None:
    print()
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def load_config(dataset: str, task: str) -> dict[str, Any]:
    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)

    key = f"{dataset}/{task}"

    try:
        return config["tasks"][key]
    except KeyError as exc:
        available = "\n".join(
            f"  - {name}" for name in sorted(config.get("tasks", {}))
        )
        raise SystemExit(
            f"No reproduction configuration for {key!r}.\n"
            f"Available tasks:\n{available}"
        ) from exc


def apply_table_aliases(
    inspect_dir: Path,
    aliases: dict[str, dict[str, str]],
) -> None:
    """Add configured column aliases to standardized inspect tables."""
    for table_name, column_map in aliases.items():
        table_path = inspect_dir / f"table_{table_name}.parquet"

        if not table_path.exists():
            raise FileNotFoundError(
                f"Alias source table does not exist: {table_path}"
            )

        df = pd.read_parquet(table_path)
        changed = False

        for alias, source in column_map.items():
            if alias in df.columns:
                continue
            if source not in df.columns:
                raise KeyError(
                    f"Cannot create alias {alias!r}: "
                    f"source column {source!r} is missing from {table_path}"
                )

            df[alias] = df[source]
            changed = True
            print(
                f"[alias] {table_name}.{alias} "
                f"<- {table_name}.{source}"
            )

        if changed:
            df.to_parquet(table_path, index=False)


def add_entity_alias(
    inspect_dir: Path,
    *,
    entity_table: str,
    source_key: str,
    target_key: str,
) -> None:
    path = inspect_dir / f"table_{entity_table}.parquet"
    df = pd.read_parquet(path)

    if target_key not in df.columns:
        if source_key not in df.columns:
            raise KeyError(
                f"Neither {target_key!r} nor {source_key!r} exists in {path}"
            )
        df[target_key] = df[source_key]

    df.to_parquet(path, index=False)


def standard_artifact_exists(directory: Path) -> bool:
    return (
        directory / "target_with_dfs_agg_train.parquet"
    ).exists() and (
        directory / "target_with_dfs_agg_val.parquet"
    ).exists()


def split_author_category_artifacts(
    combined_dir: Path,
    dfs_dir: Path,
    fdhg_dir: Path,
) -> None:
    dfs_dir.mkdir(parents=True, exist_ok=True)
    fdhg_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        source = (
            combined_dir
            / f"target_with_dfs_agg_{split}.parquet"
        )
        frame = pd.read_parquet(source)

        missing = [
            column
            for column in AUTHOR_CATEGORY_RESIDUALS
            if column not in frame.columns
        ]
        if missing:
            raise KeyError(
                "Author-category custom artifact is missing "
                f"residual columns: {missing}"
            )

        dfs_frame = frame.drop(
            columns=AUTHOR_CATEGORY_RESIDUALS
        )

        dfs_frame.to_parquet(
            dfs_dir
            / f"target_with_dfs_agg_{split}.parquet",
            index=False,
        )
        frame.to_parquet(
            fdhg_dir
            / f"target_with_dfs_agg_{split}.parquet",
            index=False,
        )

        print(
            f"[author-category] {split}: "
            f"DFS={dfs_frame.shape}, FDHG={frame.shape}"
        )


def normalize_pairwise_artifacts(
    pairwise_dir: Path,
    dfs_dir: Path,
    fdhg_dir: Path,
) -> None:
    dfs_dir.mkdir(parents=True, exist_ok=True)
    fdhg_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        mappings = [
            (
                pairwise_dir / f"dfs_{split}_pairwise.parquet",
                dfs_dir
                / f"target_with_dfs_agg_{split}.parquet",
            ),
            (
                pairwise_dir / f"fdhg_{split}_pairwise.parquet",
                fdhg_dir
                / f"target_with_dfs_agg_{split}.parquet",
            ),
        ]

        for source, destination in mappings:
            if not source.exists():
                raise FileNotFoundError(
                    f"Pairwise adapter output is missing: {source}"
                )
            shutil.copy2(source, destination)


def build_generic_dfs(
    *,
    inspect_dir: Path,
    dfs_dir: Path,
    target_cfg: dict[str, Any],
    dfs_cfg: dict[str, Any],
    seed: int,
) -> None:
    if standard_artifact_exists(dfs_dir):
        return

    run([
        sys.executable,
        "scripts/prepare/build_relbench_features.py",
        "--inspect-dir", str(inspect_dir),
        "--out-dir", str(dfs_dir),
        "--mode", "dfs",
        "--max-train", str(dfs_cfg["max_train"]),
        "--max-val", str(dfs_cfg["max_val"]),
        "--seed", str(seed),
        "--target-key", target_cfg["entity_key"],
        "--target-time-col", target_cfg["time_col"],
        "--child-table", dfs_cfg["child_table"],
        "--numeric-col", dfs_cfg["numeric_col"],
        "--splits", "train", "val",
    ])



def afd_fit_artifact_exists(afd_dir: Path) -> bool:
    """Return whether all train-only AFD fit artifacts are complete."""
    return all(
        path.exists()
        for path in (
            afd_dir / "afd_fit_table.parquet",
            afd_dir / "afd_fit_manifest.json",
            afd_dir / "_SUCCESS",
        )
    )


def build_train_only_afd_fit_table(
    *,
    inspect_dir: Path,
    afd_dir: Path,
    afd_cfg: dict,
    target_cfg: dict,
) -> Path:
    """Build and return the entity rows eligible for AFD fitting."""
    afd_dir.mkdir(parents=True, exist_ok=True)

    fit_path = afd_dir / "afd_fit_table.parquet"
    manifest_path = afd_dir / "afd_fit_manifest.json"

    if afd_fit_artifact_exists(afd_dir):
        print(
            "[afd-fit] Reusing train-only AFD fit table:",
            fit_path,
        )
        return fit_path

    entity_table = afd_cfg["entity_table"]
    entity_path = (
        inspect_dir / f"table_{entity_table}.parquet"
    )
    target_train_path = inspect_dir / "target_train.parquet"

    entity_key = afd_cfg.get(
        "fit_entity_key",
        afd_cfg.get(
            "entity_table_key",
            target_cfg["entity_key"],
        ),
    )
    target_entity_key = target_cfg["entity_key"]

    command = [
        sys.executable,
        "scripts/prepare/build_afd_fit_table.py",
        "--entity-table",
        str(entity_path),
        "--target-train",
        str(target_train_path),
        "--out",
        str(fit_path),
        "--manifest",
        str(manifest_path),
        "--entity-key",
        entity_key,
        "--target-entity-key",
        target_entity_key,
    ]

    fit_entity_time_col = afd_cfg.get(
        "fit_entity_time_col"
    )
    fit_target_time_col = afd_cfg.get(
        "fit_target_time_col"
    )

    if bool(fit_entity_time_col) != bool(
        fit_target_time_col
    ):
        raise ValueError(
            "AFD configuration must provide both "
            "`fit_entity_time_col` and "
            "`fit_target_time_col`, or neither."
        )

    if fit_entity_time_col:
        command.extend(
            [
                "--entity-time-col",
                fit_entity_time_col,
                "--target-time-col",
                fit_target_time_col,
            ]
        )

    run(command)

    if not afd_fit_artifact_exists(afd_dir):
        raise FileNotFoundError(
            "Train-only AFD fit-table generation did not "
            f"complete successfully under {afd_dir}"
        )

    return fit_path


def copytree_hardlink(
    source: Path,
    destination: Path,
) -> None:
    """Copy a tree using hard links where possible."""
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source}"
        )

    if destination.exists():
        shutil.rmtree(destination)

    def link_or_copy(src: str, dst: str) -> str:
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(
        source,
        destination,
        copy_function=link_or_copy,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end FDHG reproduction task."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build DFS/FDHG artifacts without model evaluation.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Reuse existing DFS/FDHG artifacts and run "
            "model evaluation only."
        ),
    )
    args = parser.parse_args()

    if args.prepare_only and args.evaluate_only:
        parser.error(
            "--prepare-only and --evaluate-only "
            "cannot be used together"
        )

    if args.force and args.evaluate_only:
        parser.error(
            "--force and --evaluate-only cannot be used together"
        )

    cfg = load_config(args.dataset, args.task)

    task_key = f"{args.dataset}_{args.task}"
    work_root = ROOT / "outputs" / "e2e" / task_key

    inspect_task = cfg.get("inspect_task", args.task)
    inspect_task_key = f"{args.dataset}_{inspect_task}"

    inspect_parent = work_root / "inspect"
    raw_inspect = inspect_parent / inspect_task_key
    enriched_inspect = work_root / "inspect_enriched"
    fdhg_inspect = work_root / "inspect_fdhg"

    dfs_dir = work_root / "dfs"
    afd_dir = work_root / "afd"
    fdhg_dir = work_root / "fdhg"

    combined_dir = work_root / "combined"
    base_dfs_dir = work_root / "base_dfs"
    base_fdhg_dir = work_root / "base_fdhg"
    pairwise_dir = work_root / "pairwise_raw"

    if not args.evaluate_only:
        if args.force and work_root.exists():
            shutil.rmtree(work_root)

        # 1. RelBench inspection
        if not (raw_inspect / "target_train.parquet").exists():
            run([
                "fdhg-inspect",
                "--dataset", args.dataset,
                "--task", inspect_task,
                "--out-dir", str(inspect_parent),
            ])

        target_cfg = cfg.get("target", {})

        # 2. Optional row-key enrichment
        if target_cfg.get("primary_key_col"):
            if not (
                enriched_inspect / "target_train.parquet"
            ).exists():
                command = [
                    sys.executable,
                    "scripts/prepare/enrich_target_from_primary_key.py",
                    "--inspect-dir", str(raw_inspect),
                    "--out-dir", str(enriched_inspect),
                    "--source-table", target_cfg["source_table"],
                    "--primary-key-col",
                    target_cfg["primary_key_col"],
                    "--entity-key", target_cfg["entity_key"],
                    "--source-entity-col",
                    target_cfg["source_entity_col"],
                    "--verify-time-col", target_cfg["time_col"],
                    "--splits", "train", "val",
                ]

                verify_label = target_cfg.get("verify_label_col")
                if verify_label:
                    command.extend([
                        "--verify-label-col",
                        verify_label,
                    ])

                run(command)
        else:
            enriched_inspect = raw_inspect

        table_aliases = cfg.get("table_aliases", {})
        if table_aliases:
            apply_table_aliases(
                enriched_inspect,
                table_aliases,
            )

        dfs_cfg = cfg["dfs"]
        builder = cfg.get("builder", "generic_dfs")

        # 3. Feature builder dispatch
        if builder == "generic_dfs":
            build_generic_dfs(
                inspect_dir=enriched_inspect,
                dfs_dir=dfs_dir,
                target_cfg=target_cfg,
                dfs_cfg=dfs_cfg,
                seed=args.seed,
            )

        elif builder == "arxiv_paper_citation":
            if not standard_artifact_exists(dfs_dir):
                run([
                    sys.executable,
                    "scripts/prepare/custom/"
                    "build_arxiv_paper_citation_features.py",
                    "--inspect-dir", str(enriched_inspect),
                    "--out-dir", str(dfs_dir),
                    "--max-train", str(dfs_cfg["max_train"]),
                    "--max-val", str(dfs_cfg["max_val"]),
                    "--seed", str(args.seed),
                    "--splits", "train", "val",
                ])

        elif builder == "arxiv_author_category":
            if not standard_artifact_exists(combined_dir):
                run([
                    sys.executable,
                    "scripts/prepare/custom/"
                    "build_arxiv_author_category_features.py",
                    "--inspect-dir", str(enriched_inspect),
                    "--out-dir", str(combined_dir),
                    "--max-train", str(dfs_cfg["max_train"]),
                    "--max-val", str(dfs_cfg["max_val"]),
                    "--seed", str(args.seed),
                ])

            if (
                not standard_artifact_exists(dfs_dir)
                or not standard_artifact_exists(fdhg_dir)
            ):
                split_author_category_artifacts(
                    combined_dir,
                    dfs_dir,
                    fdhg_dir,
                )

        elif builder == "ratebeer_user_place_pairwise":
            build_generic_dfs(
                inspect_dir=enriched_inspect,
                dfs_dir=base_dfs_dir,
                target_cfg=target_cfg,
                dfs_cfg=dfs_cfg,
                seed=args.seed,
            )

            # Build the real FDHG candidate before pairwise adaptation.
            # The validation gate may later reject it and select exact DFS
            # fallback, but candidate construction must remain independent.
            afd_cfg = cfg.get("afd")
            if not afd_cfg:
                raise ValueError(
                    "ratebeer_user_place_pairwise requires an AFD "
                    "configuration."
                )

            afd_dir.mkdir(parents=True, exist_ok=True)
            afd_path = afd_dir / "afd_dmax1.csv"

            if not afd_path.exists():
                afd_cmd = [
                    sys.executable,
                    "scripts/prepare/compute_afd_dmax1.py",
                    "--table",
                    str(afd_fit_path),
                    "--out",
                    str(afd_path),
                    "--seed",
                    str(args.seed),
                    "--columns",
                    *afd_cfg["columns"],
                ]

                if afd_cfg.get("max_rows") is not None:
                    afd_cmd.extend([
                        "--max-rows",
                        str(afd_cfg["max_rows"]),
                    ])

                exclude_columns = afd_cfg.get(
                    "exclude_columns",
                    [],
                )
                if exclude_columns:
                    afd_cmd.extend([
                        "--exclude-columns",
                        *exclude_columns,
                    ])

                run(afd_cmd)

            if not standard_artifact_exists(base_fdhg_dir):
                run([
                    sys.executable,
                    "scripts/prepare/build_fdhg_ambiguity.py",
                    "--inspect-dir",
                    str(enriched_inspect),
                    "--dfs-dir",
                    str(base_dfs_dir),
                    "--afd-stats",
                    str(afd_path),
                    "--out-dir",
                    str(base_fdhg_dir),
                    "--entity-key",
                    target_cfg["entity_key"],
                    "--target-table",
                    afd_cfg["entity_table"],
                    "--fit-table",
                    str(afd_fit_path),
                ])

            required_base_fdhg = (
                base_fdhg_dir
                / "target_with_dfs_agg_train.parquet"
            )
            if not required_base_fdhg.exists():
                raise FileNotFoundError(
                    f"Missing base FDHG artifact: "
                    f"{required_base_fdhg}"
                )

            expected_pairwise = (
                pairwise_dir / "dfs_train_pairwise.parquet"
            )
            if not expected_pairwise.exists():
                run([
                    sys.executable,
                    "scripts/prepare/custom/"
                    "build_ratebeer_user_place_pairwise.py",
                    "--dfs-dir", str(base_dfs_dir),
                    "--fdhg-dir", str(base_fdhg_dir),
                    "--inspect-dir", str(enriched_inspect),
                    "--out-dir", str(pairwise_dir),
                    "--seed", str(args.seed),
                    "--neg-per-pos",
                    str(cfg.get("pairwise", {}).get(
                        "neg_per_pos",
                        1,
                    )),
                    "--max-train-rows",
                    str(dfs_cfg["max_train"]),
                    "--max-val-rows",
                    str(dfs_cfg["max_val"]),
                ])

            if (
                not standard_artifact_exists(dfs_dir)
                or not standard_artifact_exists(fdhg_dir)
            ):
                normalize_pairwise_artifacts(
                    pairwise_dir,
                    dfs_dir,
                    fdhg_dir,
                )

        else:
            raise ValueError(
                f"Unsupported feature builder: {builder!r}"
            )

        # 4. Optional AFD/FDHG block
        # Custom author-category and pairwise builders already produce
        # both DFS and FDHG artifacts.
        custom_has_fdhg = builder in {
            "arxiv_author_category",
            "ratebeer_user_place_pairwise",
        }

        if not custom_has_fdhg:
            afd_cfg = cfg.get("afd")

            if afd_cfg:
                if not fdhg_inspect.exists():
                    copytree_hardlink(
                        enriched_inspect,
                        fdhg_inspect,
                    )

                if afd_cfg.get("add_entity_alias", True):
                    add_entity_alias(
                        fdhg_inspect,
                        entity_table=afd_cfg["entity_table"],
                        source_key=afd_cfg["entity_table_key"],
                        target_key=target_cfg["entity_key"],
                    )

                afd_dir.mkdir(parents=True, exist_ok=True)

                afd_fit_path = build_train_only_afd_fit_table(
                    inspect_dir=fdhg_inspect,
                    afd_dir=afd_dir,
                    afd_cfg=afd_cfg,
                    target_cfg=target_cfg,
                )

                afd_path = afd_dir / "afd_dmax1.csv"

                if not afd_path.exists():
                    afd_cmd = [
                        sys.executable,
                        "scripts/prepare/compute_afd_dmax1.py",
                        "--table",
                        str(afd_fit_path),
                        "--out", str(afd_path),
                        "--seed", str(args.seed),
                        "--columns", *afd_cfg["columns"],
                    ]

                    if afd_cfg.get("max_rows") is not None:
                        afd_cmd.extend([
                            "--max-rows",
                            str(afd_cfg["max_rows"]),
                        ])

                    exclude_columns = afd_cfg.get(
                        "exclude_columns",
                        [
                            target_cfg["entity_key"],
                            afd_cfg["entity_table_key"],
                        ],
                    )
                    if exclude_columns:
                        afd_cmd.extend([
                            "--exclude-columns",
                            *exclude_columns,
                        ])

                    run(afd_cmd)

                if not standard_artifact_exists(fdhg_dir):
                    run([
                        sys.executable,
                        "scripts/prepare/build_fdhg_ambiguity.py",
                        "--inspect-dir", str(fdhg_inspect),
                        "--dfs-dir", str(dfs_dir),
                        "--afd-stats", str(afd_path),
                        "--out-dir", str(fdhg_dir),
                        "--entity-key", target_cfg["entity_key"],
                        "--target-table", afd_cfg["entity_table"],
                        "--fit-table", str(afd_fit_path),
                    ])
            else:
                if not cfg.get("fallback_to_dfs", False):
                    raise ValueError(
                        "No AFD configuration was provided. "
                        "Set fallback_to_dfs: true for an "
                        "intentional DFS fallback."
                    )

                print(
                    "[fallback] No FDHG candidate selected; "
                    "using the exact DFS artifact."
                )
                fdhg_dir = dfs_dir

        if args.prepare_only:
            print(
                f"\nPrepared task artifacts under: {work_root}"
            )
            return


    else:
        required_dfs = [
            dfs_dir / "target_with_dfs_agg_train.parquet",
            dfs_dir / "target_with_dfs_agg_val.parquet",
        ]
        required_fdhg = [
            fdhg_dir / "target_with_dfs_agg_train.parquet",
            fdhg_dir / "target_with_dfs_agg_val.parquet",
        ]

        missing_dfs = [
            path for path in required_dfs
            if not path.exists()
        ]

        if missing_dfs:
            missing_text = "\n".join(
                f" - {path}" for path in missing_dfs
            )
            raise FileNotFoundError(
                "Cannot run --evaluate-only because DFS "
                "artifacts are missing:\n"
                f"{missing_text}\n"
                "Run once with --prepare-only first."
            )

        missing_fdhg = [
            path for path in required_fdhg
            if not path.exists()
        ]

        if missing_fdhg:
            if cfg.get("fallback_to_dfs", False):
                print(
                    "[evaluate-only] Reusing DFS as the "
                    "intentional FDHG fallback."
                )
                fdhg_dir = dfs_dir
            else:
                missing_text = "\n".join(
                    f" - {path}" for path in missing_fdhg
                )
                raise FileNotFoundError(
                    "Cannot run --evaluate-only because FDHG "
                    "artifacts are missing:\n"
                    f"{missing_text}\n"
                    "Run once with --prepare-only first."
                )

        print(
            "[evaluate-only] Reusing prepared artifacts under: "
            f"{work_root}"
        )

    os.environ.setdefault(
        "TABPFN_ALLOW_CPU_LARGE_DATASET",
        "1",
    )

    evaluation_cfg = cfg.get("evaluation", {})
    evaluation_backend = evaluation_cfg.get(
        "backend",
        "tabpfn",
    )

    evaluator = {
        "binary": (
            "scripts/evaluate/"
            "evaluate_binary_tabpfn.py"
        ),
        "regression": (
            "scripts/evaluate/"
            "evaluate_regression_tabpfn.py"
        ),
        "multiclass": (
            "scripts/evaluate/"
            + (
                "evaluate_multiclass_catboost.py"
                if evaluation_backend == "catboost"
                else "evaluate_multiclass_tabpfn.py"
            )
        ),
    }[cfg["problem_type"]]

    drop_cols = ",".join(
        cfg["evaluation"]["drop_cols"]
    )
    result_root = (
        ROOT
        / "results"
        / f"{task_key}_{evaluation_backend}"
    )

    for variant, feature_root in [
        ("dfs", dfs_dir),
        ("fdhg_dmax1", fdhg_dir),
    ]:
        out_dir = (
            result_root
            / variant
            / f"seed{args.seed}"
        )

        evaluator_variant = variant
        if (
            evaluation_backend == "catboost"
            and variant == "fdhg_dmax1"
        ):
            # Match legacy CatBoost metric metadata while
            # retaining the descriptive output directory.
            evaluator_variant = "fdhg"

        evaluation_command = [
            sys.executable,
            evaluator,
            "--train-parquet",
            str(
                feature_root
                / "target_with_dfs_agg_train.parquet"
            ),
            "--val-parquet",
            str(
                feature_root
                / "target_with_dfs_agg_val.parquet"
            ),
            "--output-dir", str(out_dir),
            "--dataset", args.dataset,
            "--task", args.task,
            "--variant", evaluator_variant,
            "--label-col", cfg["label_col"],
            "--drop-cols", drop_cols,
            "--seed", str(args.seed),
        ]

        if evaluation_backend != "catboost":
            evaluation_command.extend([
                "--device",
                args.device,
            ])

        run(evaluation_command)

    print(
        "\nCompleted end-to-end reproduction: "
        f"{args.dataset}/{args.task}"
    )


if __name__ == "__main__":
    main()
