from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExistingProgramArtifact:
    program_id: str
    artifact_dir: Path
    result_variant: str
    result_root: Path
    realized_primitive_ids: tuple[str, ...]
    primitive_column_bindings: dict[str, tuple[str, ...]]


# ---------------------------------------------------------------------
# Shared logical primitive IDs
# ---------------------------------------------------------------------

BASELINE_PRIMITIVES = (
    "baseline::count",
    "baseline::numeric_mean",
    "baseline::numeric_std",
    "baseline::numeric_max",
    "baseline::days_since_last",
)

STRUCTURAL_COMPACT_PRIMITIVES = (
    "structural::afd::majority_confidence",
    "structural::afd::entropy",
    "structural::afd::conflict_count",
    "structural::afd::support_count",
)

STRUCTURAL_MARGIN_PRIMITIVES = (
    *STRUCTURAL_COMPACT_PRIMITIVES,
    "structural::afd::top1_margin",
)

USER_COUNT_TEMPORAL_PRIMITIVES = (
    "temporal::count::30d",
    "temporal::count::90d",
    "temporal::count::365d",
)


def merge_bindings(
    *parts: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    merged: dict[str, tuple[str, ...]] = {}

    for part in parts:
        merged.update(part)

    return merged


# ---------------------------------------------------------------------
# rel-ratebeer / user-count
# ---------------------------------------------------------------------

RATEBEER_BASELINE_BINDINGS = {
    "baseline::count": (
        "f_beer_ratings_count",
    ),
    "baseline::numeric_mean": (
        "f_beer_ratings_aroma_mean",
    ),
    "baseline::numeric_std": (
        "f_beer_ratings_aroma_std",
    ),
    "baseline::numeric_max": (
        "f_beer_ratings_aroma_max",
    ),
    "baseline::days_since_last": (
        "f_beer_ratings_days_since_last",
    ),
}

RATEBEER_STRUCTURAL_BINDINGS = {
    "structural::afd::majority_confidence": (
        "f_amb__place_id_to_postal_code__majconf",
    ),
    "structural::afd::entropy": (
        "f_amb__place_id_to_postal_code__entropy",
    ),
    "structural::afd::conflict_count": (
        "f_amb__place_id_to_postal_code__conflict_count",
    ),
    "structural::afd::support_count": (
        "f_amb__place_id_to_postal_code__support_count",
    ),
}

RATEBEER_TEMPORAL_BINDINGS = {
    "temporal::count::30d": (
        "fdhg::temporal_activity::count_30d",
    ),
    "temporal::count::90d": (
        "fdhg::temporal_activity::count_90d",
    ),
    "temporal::count::365d": (
        "fdhg::temporal_activity::count_365d",
    ),
}

RATEBEER_RESULT_ROOT = Path(
    "results/rel-ratebeer_user-count_"
    "canonical_temporal_tabpfn"
)

RATEBEER_ARTIFACT_ROOT = Path(
    "outputs/e2e/rel-ratebeer_user-count"
)

USER_COUNT_ARTIFACTS = {
    "baseline": ExistingProgramArtifact(
        program_id="baseline",
        artifact_dir=(
            RATEBEER_ARTIFACT_ROOT
            / "dfs_corrected_canonical"
        ),
        result_variant="corrected_dfs",
        result_root=RATEBEER_RESULT_ROOT,
        realized_primitive_ids=BASELINE_PRIMITIVES,
        primitive_column_bindings=(
            RATEBEER_BASELINE_BINDINGS
        ),
    ),
    "baseline_plus_structural": ExistingProgramArtifact(
        program_id="baseline_plus_structural",
        artifact_dir=(
            RATEBEER_ARTIFACT_ROOT
            / "fdhg_corrected_canonical"
        ),
        result_variant="corrected_dfs_plus_ambiguity",
        result_root=RATEBEER_RESULT_ROOT,
        realized_primitive_ids=(
            BASELINE_PRIMITIVES
            + STRUCTURAL_COMPACT_PRIMITIVES
        ),
        primitive_column_bindings=merge_bindings(
            RATEBEER_BASELINE_BINDINGS,
            RATEBEER_STRUCTURAL_BINDINGS,
        ),
    ),
    "baseline_plus_temporal": ExistingProgramArtifact(
        program_id="baseline_plus_temporal",
        artifact_dir=(
            RATEBEER_ARTIFACT_ROOT
            / "corrected_dfs_plus_counts"
        ),
        result_variant="corrected_dfs_plus_counts",
        result_root=RATEBEER_RESULT_ROOT,
        realized_primitive_ids=(
            BASELINE_PRIMITIVES
            + USER_COUNT_TEMPORAL_PRIMITIVES
        ),
        primitive_column_bindings=merge_bindings(
            RATEBEER_BASELINE_BINDINGS,
            RATEBEER_TEMPORAL_BINDINGS,
        ),
    ),
    "baseline_plus_structural_temporal":
        ExistingProgramArtifact(
            program_id=(
                "baseline_plus_structural_temporal"
            ),
            artifact_dir=(
                RATEBEER_ARTIFACT_ROOT
                / "corrected_dfs_plus_ambiguity_counts"
            ),
            result_variant=(
                "corrected_dfs_plus_ambiguity_counts"
            ),
            result_root=RATEBEER_RESULT_ROOT,
            realized_primitive_ids=(
                BASELINE_PRIMITIVES
                + STRUCTURAL_COMPACT_PRIMITIVES
                + USER_COUNT_TEMPORAL_PRIMITIVES
            ),
            primitive_column_bindings=merge_bindings(
                RATEBEER_BASELINE_BINDINGS,
                RATEBEER_STRUCTURAL_BINDINGS,
                RATEBEER_TEMPORAL_BINDINGS,
            ),
        ),
}


# ---------------------------------------------------------------------
# rel-salt / item-shippoint
# ---------------------------------------------------------------------

ITEM_SHIPPOINT_BASELINE_BINDINGS = {
    "baseline::count": (
        "f_salesdocumentitem_count",
    ),
    "baseline::numeric_mean": (
        "f_salesdocumentitem_PLANT_mean",
    ),
    "baseline::numeric_std": (
        "f_salesdocumentitem_PLANT_std",
    ),
    "baseline::numeric_max": (
        "f_salesdocumentitem_PLANT_max",
    ),
    "baseline::days_since_last": (
        "f_salesdocumentitem_days_since_last",
    ),
}

ITEM_SHIPPOINT_STRUCTURAL_BINDINGS = {
    "structural::afd::majority_confidence": (
        "f_amb__PRODUCT_to_"
        "SALESDOCUMENTITEMCATEGORY__majconf",
    ),
    "structural::afd::entropy": (
        "f_amb__PRODUCT_to_"
        "SALESDOCUMENTITEMCATEGORY__entropy",
    ),
    "structural::afd::conflict_count": (
        "f_amb__PRODUCT_to_"
        "SALESDOCUMENTITEMCATEGORY__conflict_count",
    ),
    "structural::afd::support_count": (
        "f_amb__PRODUCT_to_"
        "SALESDOCUMENTITEMCATEGORY__support_count",
    ),
    "structural::afd::top1_margin": (
        "f_amb__PRODUCT_to_"
        "SALESDOCUMENTITEMCATEGORY__top1_margin",
    ),
}

ITEM_SHIPPOINT_BASELINE_ROOT = Path(
    "outputs/e2e/rel-salt_item-shippoint"
)

ITEM_SHIPPOINT_COMPACT_ROOT = Path(
    "outputs/e2e/"
    "rel-salt_item-shippoint."
    "before_margin_unique_20260715_184218"
)

ITEM_SHIPPOINT_CURRENT_RESULTS = Path(
    "results/rel-salt_item-shippoint_tabpfn"
)

ITEM_SHIPPOINT_COMPACT_RESULTS = Path(
    "results/"
    "rel-salt_item-shippoint_tabpfn."
    "before_margin_unique_20260715_184219"
)

ITEM_SHIPPOINT_ARTIFACTS = {
    "baseline": ExistingProgramArtifact(
        program_id="baseline",
        artifact_dir=Path(
            "outputs/e2e/rel-salt_item-shippoint/dfs"
        ),
        result_variant="dfs",
        result_root=Path(
            "results/rel-salt_item-shippoint_tabpfn"
        ),
        realized_primitive_ids=BASELINE_PRIMITIVES,
        primitive_column_bindings=(
            ITEM_SHIPPOINT_BASELINE_BINDINGS
        ),
    ),

    "baseline_plus_structural_compact":
        ExistingProgramArtifact(
            program_id=(
                "baseline_plus_structural_compact"
            ),
            artifact_dir=Path(
                "outputs/e2e/"
                "rel-salt_item-shippoint."
                "before_margin_unique_20260715_184218/"
                "fdhg"
            ),
            result_variant="fdhg_dmax1",
            result_root=Path(
                "results/"
                "rel-salt_item-shippoint_tabpfn."
                "before_margin_unique_20260715_184219"
            ),
            realized_primitive_ids=(
                BASELINE_PRIMITIVES
                + STRUCTURAL_COMPACT_PRIMITIVES
            ),
            primitive_column_bindings=merge_bindings(
                ITEM_SHIPPOINT_BASELINE_BINDINGS,
                {
                    key: value
                    for key, value
                    in ITEM_SHIPPOINT_STRUCTURAL_BINDINGS.items()
                    if key
                    in STRUCTURAL_COMPACT_PRIMITIVES
                },
            ),
        ),

    "baseline_plus_structural_margin":
        ExistingProgramArtifact(
            program_id=(
                "baseline_plus_structural_margin"
            ),
            artifact_dir=Path(
                "outputs/e2e/"
                "rel-salt_item-shippoint/fdhg"
            ),
            result_variant="fdhg_dmax1",
            result_root=Path(
                "results/rel-salt_item-shippoint_tabpfn"
            ),
            realized_primitive_ids=(
                BASELINE_PRIMITIVES
                + STRUCTURAL_MARGIN_PRIMITIVES
            ),
            primitive_column_bindings=merge_bindings(
                ITEM_SHIPPOINT_BASELINE_BINDINGS,
                ITEM_SHIPPOINT_STRUCTURAL_BINDINGS,
            ),
        ),
}



# ---------------------------------------------------------------------
# rel-arxiv / author-category
# ---------------------------------------------------------------------

AUTHOR_BASELINE_PRIMITIVES = (
    "baseline::count",
    "baseline::history::window_count_short",
    "baseline::history::window_count_aligned",
    "baseline::history::window_count_long",
    "baseline::days_since_last",
    "baseline::history::past_unique_values",
    "baseline::history::past_unique_neighbors",
    "baseline::history::mean_group_size",
    "baseline::history::max_group_size",
    "baseline::history::incoming_event_count",
    "baseline::history::past_unique_sources",
    "baseline::history::incoming_event_count_long",
)

AUTHOR_ORIGINAL3_PRIMITIVES = (
    "structural::afd::majority_confidence",
    "structural::afd::entropy",
    "structural::afd::last_observed_value",
)

AUTHOR_PLUS_SUPPORT_PRIMITIVES = (
    *AUTHOR_ORIGINAL3_PRIMITIVES,
    "structural::afd::support_count",
)

AUTHOR_PLUS_CONFLICT_PRIMITIVES = (
    *AUTHOR_PLUS_SUPPORT_PRIMITIVES,
    "structural::afd::conflict_count",
)

AUTHOR_PLUS_MARGIN_PRIMITIVES = (
    *AUTHOR_PLUS_CONFLICT_PRIMITIVES,
    "structural::afd::top1_margin",
)

AUTHOR_PLUS_UNIQUE_PRIMITIVES = (
    *AUTHOR_PLUS_MARGIN_PRIMITIVES,
    "structural::afd::unique_count",
)

AUTHOR_ALL8_PRIMITIVES = (
    "structural::afd::majority_confidence",
    "structural::afd::entropy",
    "structural::afd::support_count",
    "structural::afd::conflict_count",
    "structural::afd::top1_margin",
    "structural::afd::unique_count",
    "structural::afd::last_observed_value",
)


AUTHOR_BASELINE_BINDINGS = {
    "baseline::count": (
        "dfs::author::past_paper_count",
    ),
    "baseline::history::window_count_short": (
        "dfs::author::past_paper_count_30d",
    ),
    "baseline::history::window_count_aligned": (
        "dfs::author::past_paper_count_90d",
    ),
    "baseline::history::window_count_long": (
        "dfs::author::past_paper_count_365d",
    ),
    "baseline::days_since_last": (
        "dfs::author::days_since_last_paper",
    ),
    "baseline::history::past_unique_values": (
        "dfs::author::past_unique_primary_categories",
    ),
    "baseline::history::past_unique_neighbors": (
        "dfs::author::past_unique_collaborators",
    ),
    "baseline::history::mean_group_size": (
        "dfs::author::mean_authors_per_paper",
    ),
    "baseline::history::max_group_size": (
        "dfs::author::max_authors_per_paper",
    ),
    "baseline::history::incoming_event_count": (
        "dfs::author::past_incoming_citation_count",
    ),
    "baseline::history::past_unique_sources": (
        "dfs::author::past_unique_citing_papers",
    ),
    "baseline::history::incoming_event_count_long": (
        "dfs::author::past_incoming_citation_count_365d",
    ),
}

AUTHOR_STRUCTURAL_BINDINGS = {
    "structural::afd::majority_confidence": (
        "fdhg::author_category::majority_confidence",
    ),
    "structural::afd::entropy": (
        "fdhg::author_category::entropy",
    ),
    "structural::afd::support_count": (
        "fdhg::author_category::support_count",
    ),
    "structural::afd::conflict_count": (
        "fdhg::author_category::conflict_count",
    ),
    "structural::afd::top1_margin": (
        "fdhg::author_category::top1_margin",
    ),
    "structural::afd::unique_count": (
        "fdhg::author_category::unique_count",
    ),
    "structural::afd::last_observed_value": (
        "fdhg::author_category::last_primary_category",
    ),
}

AUTHOR_RESULT_ROOT = Path(
    "results/rel-arxiv_author-category_ablation_tabpfn"
)

AUTHOR_ARTIFACT_ROOT = Path(
    "outputs/e2e/rel-arxiv_author-category_ablation"
)


def author_artifact(
    program_id: str,
    variant: str,
    structural_primitives: tuple[str, ...],
) -> ExistingProgramArtifact:
    primitive_ids = (
        AUTHOR_BASELINE_PRIMITIVES
        + structural_primitives
    )

    structural_bindings = {
        primitive_id: AUTHOR_STRUCTURAL_BINDINGS[
            primitive_id
        ]
        for primitive_id in structural_primitives
    }

    return ExistingProgramArtifact(
        program_id=program_id,
        artifact_dir=AUTHOR_ARTIFACT_ROOT / variant,
        result_variant=variant,
        result_root=AUTHOR_RESULT_ROOT,
        realized_primitive_ids=primitive_ids,
        primitive_column_bindings=merge_bindings(
            AUTHOR_BASELINE_BINDINGS,
            structural_bindings,
        ),
    )


AUTHOR_CATEGORY_ARTIFACTS = {
    "baseline": ExistingProgramArtifact(
        program_id="baseline",
        artifact_dir=AUTHOR_ARTIFACT_ROOT / "dfs",
        result_variant="dfs",
        result_root=AUTHOR_RESULT_ROOT,
        realized_primitive_ids=AUTHOR_BASELINE_PRIMITIVES,
        primitive_column_bindings=AUTHOR_BASELINE_BINDINGS,
    ),
    "baseline_plus_original3": author_artifact(
        "baseline_plus_original3",
        "original3",
        AUTHOR_ORIGINAL3_PRIMITIVES,
    ),
    "baseline_plus_support": author_artifact(
        "baseline_plus_support",
        "plus_support",
        AUTHOR_PLUS_SUPPORT_PRIMITIVES,
    ),
    "baseline_plus_conflict": author_artifact(
        "baseline_plus_conflict",
        "plus_conflict",
        AUTHOR_PLUS_CONFLICT_PRIMITIVES,
    ),
    "baseline_plus_margin": author_artifact(
        "baseline_plus_margin",
        "plus_margin",
        AUTHOR_PLUS_MARGIN_PRIMITIVES,
    ),
    "baseline_plus_unique": author_artifact(
        "baseline_plus_unique",
        "plus_unique",
        AUTHOR_PLUS_UNIQUE_PRIMITIVES,
    ),
    "baseline_plus_all8": author_artifact(
        "baseline_plus_all8",
        "all8",
        AUTHOR_ALL8_PRIMITIVES,
    ),
}


def resolve_existing_artifacts(
    dataset: str,
    task: str,
) -> dict[str, ExistingProgramArtifact]:
    key = f"{dataset}/{task}"

    if key == "rel-ratebeer/user-count":
        return USER_COUNT_ARTIFACTS

    if key == "rel-salt/item-shippoint":
        return ITEM_SHIPPOINT_ARTIFACTS

    if key == "rel-arxiv/author-category":
        return AUTHOR_CATEGORY_ARTIFACTS

    raise KeyError(
        f"No existing compiler backend mapping for {key}"
    )
