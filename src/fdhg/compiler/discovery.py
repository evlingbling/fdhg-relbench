from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from .existing_backend import ExistingProgramArtifact


TARGET_OR_METADATA_NAMES = {
    "__row_id",
    "date",
    "CREATIONTIMESTAMP",
    "ID",
    "Author_ID",
    "SALESDOCUMENT",
    "SHIPPINGPOINT",
    "SALESGROUP",
    "primary_category",
}


STRUCTURAL_SUFFIXES = {
    "majconf": "majority_confidence",
    "majority_confidence": "majority_confidence",
    "entropy": "entropy",
    "conflict_count": "conflict_count",
    "support_count": "support_count",
    "top1_margin": "top1_margin",
    "unique_count": "unique_count",
    "last_primary_category": "last_observed_value",
}


@dataclass(frozen=True)
class DiscoveredTaskLayout:
    dataset: str
    task: str
    result_root: Path
    artifact_root: Path


def task_slug(
    dataset: str,
    task: str,
) -> str:
    return f"{dataset}_{task}"


def discover_result_root(
    *,
    dataset: str,
    task: str,
    results_dir: Path = Path("results"),
) -> Path:
    """
    Locate a result root containing seed-level metrics.

    Prefer a single current root. Archived roots containing
    '.before_' are excluded from automatic onboarding.
    """
    slug = task_slug(dataset, task)

    candidates = []

    for path in results_dir.glob(f"{slug}_*"):
        if not path.is_dir():
            continue

        if ".before_" in path.name:
            continue

        metric_files = list(
            path.glob("*/seed*/metrics.csv")
        )

        if metric_files:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No active result root found for "
            f"{dataset}/{task} under {results_dir}"
        )

    if len(candidates) > 1:
        raise ValueError(
            "Ambiguous active result roots for "
            f"{dataset}/{task}: "
            + ", ".join(str(path) for path in candidates)
        )

    return candidates[0]


def discover_artifact_root(
    *,
    dataset: str,
    task: str,
    outputs_dir: Path = Path("outputs/e2e"),
) -> Path:
    slug = task_slug(dataset, task)
    root = outputs_dir / slug

    if not root.exists():
        raise FileNotFoundError(root)

    return root


def discover_task_layout(
    *,
    dataset: str,
    task: str,
) -> DiscoveredTaskLayout:
    return DiscoveredTaskLayout(
        dataset=dataset,
        task=task,
        result_root=discover_result_root(
            dataset=dataset,
            task=task,
        ),
        artifact_root=discover_artifact_root(
            dataset=dataset,
            task=task,
        ),
    )


def discover_result_variants(
    *,
    result_root: Path,
    seeds: list[int],
) -> list[str]:
    variants = []

    for variant_dir in sorted(result_root.iterdir()):
        if not variant_dir.is_dir():
            continue

        complete = all(
            (
                variant_dir
                / f"seed{seed}"
                / "metrics.csv"
            ).exists()
            for seed in seeds
        )

        if complete:
            variants.append(variant_dir.name)

    if not variants:
        raise ValueError(
            f"No complete variants found under {result_root}"
        )

    return variants


def read_variant_feature_columns(
    *,
    result_root: Path,
    result_variant: str,
    seeds: list[int],
) -> list[str]:
    feature_sets = []

    for seed in seeds:
        path = (
            result_root
            / result_variant
            / f"seed{seed}"
            / "metrics.csv"
        )

        frame = pd.read_csv(path)

        if len(frame) != 1:
            raise ValueError(
                f"Expected one row in {path}, "
                f"found {len(frame)}"
            )

        feature_cols = frame.iloc[0].get(
            "feature_cols"
        )

        if not isinstance(feature_cols, str):
            raise ValueError(
                f"feature_cols missing in {path}"
            )

        columns = feature_cols.split("|")
        feature_sets.append(columns)

    first = feature_sets[0]

    for columns in feature_sets[1:]:
        if columns != first:
            raise ValueError(
                f"Inconsistent feature columns for "
                f"{result_variant}"
            )

    return first


def resolve_artifact_variant(
    *,
    artifact_root: Path,
    result_variant: str,
) -> Path:
    """
    Match result variants to artifact directories without
    task-specific Python wiring.

    Exact match is preferred. A single fdhg* result may map to
    artifact directory 'fdhg'.
    """
    exact = artifact_root / result_variant

    if (
        exact
        / "target_with_dfs_agg_train.parquet"
    ).exists():
        return exact

    aliases = []

    if result_variant.startswith("fdhg"):
        aliases.append("fdhg")

    if result_variant == "dfs":
        aliases.append("dfs")

    for alias in aliases:
        candidate = artifact_root / alias

        if (
            candidate
            / "target_with_dfs_agg_train.parquet"
        ).exists():
            return candidate

    raise FileNotFoundError(
        "No artifact directory for variant "
        f"{result_variant!r} under {artifact_root}"
    )


def remove_missing_suffix(
    column: str,
) -> tuple[str, bool]:
    suffix = "__is_missing"

    if column.endswith(suffix):
        return column[:-len(suffix)], True

    return column, False


def normalize_baseline_column(
    column: str,
) -> str | None:
    base, _ = remove_missing_suffix(column)

    if base.startswith("dfs::"):
        if "days_since_last" in base:
            return "baseline::days_since_last"

        if re.search(
            r"(?:^|::)past_.*_count_30d$",
            base,
        ):
            return (
                "baseline::history::"
                "window_count_short"
            )

        if re.search(
            r"(?:^|::)past_.*_count_90d$",
            base,
        ):
            return (
                "baseline::history::"
                "window_count_aligned"
            )

        if re.search(
            r"(?:^|::)past_.*_count_365d$",
            base,
        ):
            if "incoming" in base:
                return (
                    "baseline::history::"
                    "incoming_event_count_long"
                )

            return (
                "baseline::history::"
                "window_count_long"
            )

        if "past_unique_primary" in base:
            return (
                "baseline::history::"
                "past_unique_values"
            )

        if "past_unique_collaborator" in base:
            return (
                "baseline::history::"
                "past_unique_neighbors"
            )

        if "mean_authors_per" in base:
            return (
                "baseline::history::mean_group_size"
            )

        if "max_authors_per" in base:
            return (
                "baseline::history::max_group_size"
            )

        if "past_incoming_citation_count" in base:
            return (
                "baseline::history::incoming_event_count"
            )

        if "past_unique_citing" in base:
            return (
                "baseline::history::past_unique_sources"
            )

        if base.endswith("_count"):
            return "baseline::count"

        return None

    if not base.startswith("f_"):
        return None

    if base.endswith("_count"):
        return "baseline::count"

    if "days_since_last" in base:
        return "baseline::days_since_last"

    if base.endswith("_mean"):
        return "baseline::numeric_mean"

    if base.endswith("_std"):
        return "baseline::numeric_std"

    if base.endswith("_max"):
        return "baseline::numeric_max"

    return None


def normalize_structural_column(
    column: str,
) -> str | None:
    base, _ = remove_missing_suffix(column)

    structural_prefixes = (
        "f_amb__",
        "fdhg::",
    )

    if not base.startswith(structural_prefixes):
        return None

    for suffix, operation in STRUCTURAL_SUFFIXES.items():
        if base.endswith(suffix):
            return f"structural::afd::{operation}"

    return None


def normalize_feature_column(
    column: str,
) -> str | None:
    structural = normalize_structural_column(column)

    if structural is not None:
        return structural

    return normalize_baseline_column(column)


def build_logical_bindings(
    feature_columns: list[str],
) -> dict[str, tuple[str, ...]]:
    bindings: dict[str, list[str]] = {}

    for column in feature_columns:
        primitive_id = normalize_feature_column(column)

        if primitive_id is None:
            raise ValueError(
                "Cannot normalize feature column "
                f"{column!r}"
            )

        bindings.setdefault(
            primitive_id,
            [],
        ).append(column)

    return {
        primitive_id: tuple(columns)
        for primitive_id, columns in bindings.items()
    }


def infer_program_id(
    *,
    result_variant: str,
    primitive_ids: tuple[str, ...],
) -> str:
    if result_variant == "dfs":
        return "baseline"

    structural = sorted(
        primitive_id
        for primitive_id in primitive_ids
        if primitive_id.startswith("structural::")
    )

    temporal = sorted(
        primitive_id
        for primitive_id in primitive_ids
        if primitive_id.startswith("temporal::")
    )

    if structural and temporal:
        return "baseline_plus_structural_temporal"

    if structural:
        compact = {
            "structural::afd::majority_confidence",
            "structural::afd::entropy",
            "structural::afd::conflict_count",
            "structural::afd::support_count",
        }

        if set(structural) == compact:
            return "baseline_plus_structural_compact"

        suffix = result_variant.removeprefix(
            "fdhg_"
        ).removeprefix("plus_")

        return f"baseline_plus_{suffix}"

    if temporal:
        return "baseline_plus_temporal"

    return f"program_{result_variant}"


def discover_existing_artifacts(
    *,
    dataset: str,
    task: str,
    seeds: list[int],
) -> dict[str, ExistingProgramArtifact]:
    layout = discover_task_layout(
        dataset=dataset,
        task=task,
    )

    artifacts = {}

    for result_variant in discover_result_variants(
        result_root=layout.result_root,
        seeds=seeds,
    ):
        feature_columns = read_variant_feature_columns(
            result_root=layout.result_root,
            result_variant=result_variant,
            seeds=seeds,
        )

        bindings = build_logical_bindings(
            feature_columns
        )

        primitive_ids = tuple(bindings)

        program_id = infer_program_id(
            result_variant=result_variant,
            primitive_ids=primitive_ids,
        )

        if program_id in artifacts:
            raise ValueError(
                f"Duplicate discovered program ID "
                f"{program_id!r}"
            )

        artifacts[program_id] = (
            ExistingProgramArtifact(
                program_id=program_id,
                artifact_dir=resolve_artifact_variant(
                    artifact_root=layout.artifact_root,
                    result_variant=result_variant,
                ),
                result_variant=result_variant,
                result_root=layout.result_root,
                realized_primitive_ids=primitive_ids,
                primitive_column_bindings=bindings,
            )
        )

    return artifacts
