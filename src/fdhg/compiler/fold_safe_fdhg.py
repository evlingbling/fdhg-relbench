from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fdhg.onboarding.relbench_v1 import _table_df

from .ambiguity import (
    NULL_TOKEN,
    FittedAmbiguityEdge,
    edge_from_mapping,
    entropy_from_counts,
    fit_ambiguity_map,
    fitted_edge_to_audit_row,
    materialize_ambiguity_from_map,
    normalize_join_key_pair,
    normalize_lhs_frame,
    normalize_series,
)

ID_RE = re.compile(r"(^id$|_id$|id$|^key$|_key$|uuid|hash)", re.IGNORECASE)
UNSAFE_NAME_RE = re.compile(r"(url|email|phone|postal|zip)", re.IGNORECASE)
LONG_TEXT_NAME_RE = re.compile(
    r"(^|_)(text|body|title|comment|comments|description|summary|content)$",
    re.IGNORECASE,
)
UUID_VALUE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
CONTINUOUS_MISSING_CATEGORY = "__FDHG_CONTINUOUS_MISSING__"
CONTINUOUS_TRANSFORM_PREFIX = "__fdhg_qbin__"


CONTINUOUS_DISCRETIZATION_AUDIT_COLUMNS = (
    "dataset",
    "task",
    "source_table",
    "original_column",
    "transformed_column",
    "fold",
    "fit_split",
    "requested_bins",
    "effective_bins",
    "non_null_count",
    "missing_count",
    "missing_policy",
    "accepted",
    "rejection_reason",
)


def is_identifier_like(name: str) -> bool:
    return bool(ID_RE.search(str(name)))


def column_eligibility_audit(
    name: str,
    series: pd.Series,
    *,
    metadata: Mapping[str, Any],
    table: Any,
    source_entity_column: str | None = None,
    min_support: int = 2,
    max_cardinality: int = 256,
    max_unique_ratio: float = 0.95,
    continuous_fdhg_mode: str = "exclude",
    continuous_fdhg_bins: int = 8,
    continuous_fdhg_min_effective_bins: int = 2,
) -> dict[str, Any]:
    col = str(name)
    non_null_count = int(series.notna().sum())
    cardinality = int(series.nunique(dropna=True))
    unique_ratio = float(cardinality / max(1, non_null_count))
    actual_primary_key = col == getattr(table, "pkey_col", None)
    actual_foreign_key = col in (getattr(table, "fkey_col_to_pkey_table", {}) or {})
    is_source_entity = bool(source_entity_column) and col == str(source_entity_column)
    audit = {
        "source_table": "",
        "column": col,
        "dtype": str(series.dtype),
        "non_null_count": non_null_count,
        "cardinality": cardinality,
        "unique_ratio": unique_ratio,
        "actual_primary_key": actual_primary_key,
        "actual_foreign_key": actual_foreign_key,
        "source_entity_column": is_source_entity,
        "determinant_eligible": False,
        "dependent_eligible": False,
        "exclusion_reason": "",
        "eligibility_status": "",
        "transformed_column": "",
        "original_column": col,
        "effective_bins": "",
        "requested_bins": "",
    }
    reason = _categorical_exclusion_reason(
        col,
        series,
        metadata=metadata,
        table=table,
        source_entity_column=source_entity_column,
        non_null_count=non_null_count,
        cardinality=cardinality,
        unique_ratio=unique_ratio,
        max_cardinality=max_cardinality,
        max_unique_ratio=max_unique_ratio,
        continuous_fdhg_mode=continuous_fdhg_mode,
    )
    if reason:
        if reason == "continuous_numeric_discretization_pending":
            spec = fit_quantile_discretizer(
                series,
                requested_bins=continuous_fdhg_bins,
                min_effective_bins=continuous_fdhg_min_effective_bins,
            )
            audit["requested_bins"] = continuous_fdhg_bins
            audit["effective_bins"] = spec["effective_bins"]
            audit["transformed_column"] = transformed_continuous_column_name(col)
            if spec["accepted"]:
                transformed = apply_quantile_discretizer(series, boundaries=spec["boundaries"])
                determinant_reason = determinant_exclusion_reason(
                    transformed,
                    min_support=min_support,
                )
                audit["dependent_eligible"] = True
                audit["determinant_eligible"] = determinant_reason == ""
                audit["exclusion_reason"] = ""
                audit["eligibility_status"] = (
                    "continuous_discretization_eligible"
                    if determinant_reason == ""
                    else f"continuous_discretization_dependent_only:{determinant_reason}"
                )
                return audit
            audit["exclusion_reason"] = spec["rejection_reason"]
            audit["eligibility_status"] = spec["rejection_reason"]
            return audit
        audit["exclusion_reason"] = reason
        audit["eligibility_status"] = reason
        return audit
    determinant_reason = determinant_exclusion_reason(
        series,
        min_support=min_support,
    )
    audit["dependent_eligible"] = True
    audit["determinant_eligible"] = determinant_reason == ""
    audit["exclusion_reason"] = determinant_reason
    audit["eligibility_status"] = "safe_categorical" if determinant_reason == "" else determinant_reason
    return audit


def dependent_eligible_column(
    name: str,
    series: pd.Series,
    *,
    metadata: Mapping[str, Any],
    table: Any,
    source_entity_column: str | None = None,
) -> tuple[bool, str]:
    audit = column_eligibility_audit(
        name,
        series,
        metadata=metadata,
        table=table,
        source_entity_column=source_entity_column,
    )
    return bool(audit["dependent_eligible"]), str(audit["exclusion_reason"])


def determinant_eligible_column(
    name: str,
    series: pd.Series,
    *,
    metadata: Mapping[str, Any],
    table: Any,
    source_entity_column: str | None = None,
    min_support: int = 2,
) -> tuple[bool, str]:
    audit = column_eligibility_audit(
        name,
        series,
        metadata=metadata,
        table=table,
        source_entity_column=source_entity_column,
        min_support=min_support,
    )
    return bool(audit["determinant_eligible"]), str(audit["exclusion_reason"])


def is_safe_fdhg_column(
    name: str,
    series: pd.Series,
    *,
    metadata: Mapping[str, Any],
    table: Any,
) -> tuple[bool, str]:
    ok, reason = dependent_eligible_column(name, series, metadata=metadata, table=table)
    return ok, "safe_categorical" if ok else reason


def _categorical_exclusion_reason(
    col: str,
    series: pd.Series,
    *,
    metadata: Mapping[str, Any],
    table: Any,
    source_entity_column: str | None,
    non_null_count: int,
    cardinality: int,
    unique_ratio: float,
    max_cardinality: int,
    max_unique_ratio: float,
    continuous_fdhg_mode: str = "exclude",
) -> str:
    if col == metadata.get("label_col"):
        return "target_named_column_excluded"
    if col == metadata.get("entity_key"):
        return "entity_key_excluded"
    if col == getattr(table, "pkey_col", None):
        return "primary_key_excluded"
    if col in (getattr(table, "fkey_col_to_pkey_table", {}) or {}):
        return "foreign_key_excluded"
    if source_entity_column and col == str(source_entity_column):
        return "source_entity_column_excluded"
    if col == getattr(table, "time_col", None):
        return "timestamp_excluded"
    if UNSAFE_NAME_RE.search(col):
        return "unsafe_name_excluded"
    if is_identifier_like(col) and re.search(r"(uuid|guid|hash)", col, re.IGNORECASE):
        return "guid_like_excluded"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp_excluded"
    if non_null_count <= 0 or cardinality <= 1:
        return "constant_or_empty_excluded"
    if _is_guid_like_values(series):
        return "guid_like_excluded"
    if pd.api.types.is_bool_dtype(series):
        return ""
    if pd.api.types.is_numeric_dtype(series):
        if unique_ratio > max_unique_ratio:
            return "surrogate_key_identity_excluded"
        if cardinality > max_cardinality:
            if continuous_fdhg_mode == "quantile":
                return "continuous_numeric_discretization_pending"
            return "high_cardinality_numeric_excluded"
        return ""
    if series.dtype == "object" or str(series.dtype).startswith(("string", "category")):
        strings = series.dropna().astype(str)
        mean_len = float(strings.str.len().mean()) if len(strings) else 0.0
        max_len = int(strings.str.len().max()) if len(strings) else 0
        mean_tokens = float(strings.str.split().str.len().mean()) if len(strings) else 0.0
        if LONG_TEXT_NAME_RE.search(col) and cardinality > 16:
            return "free_text_name_excluded"
        if cardinality > max_cardinality or mean_len > 80.0 or max_len > 256 or mean_tokens > 12.0:
            return "free_text_or_high_cardinality_excluded"
        if unique_ratio > max_unique_ratio:
            return "surrogate_key_identity_excluded"
        return ""
    return "unsafe_or_unknown"


def determinant_exclusion_reason(series: pd.Series, *, min_support: int = 2) -> str:
    counts = series.dropna().value_counts(sort=False)
    non_singleton_rows = int(counts[counts > 1].sum()) if len(counts) else 0
    repeated_group_count = int((counts > 1).sum()) if len(counts) else 0
    if repeated_group_count <= 0 or non_singleton_rows < min_support:
        return "determinant_repeated_support_below_minimum"
    return ""


def _is_guid_like_values(series: pd.Series) -> bool:
    values = series.dropna().astype(str)
    if values.empty:
        return False
    sample = values.head(1000)
    return bool(sample.str.fullmatch(UUID_VALUE_RE).mean() >= 0.8)


def transformed_continuous_column_name(column: str) -> str:
    return f"{CONTINUOUS_TRANSFORM_PREFIX}{column}"


def fit_quantile_discretizer(
    series: pd.Series,
    *,
    requested_bins: int,
    min_effective_bins: int,
) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric[np.isfinite(numeric)]
    missing_count = int(len(series) - len(non_null))
    requested = max(1, int(requested_bins))
    if len(non_null) <= 0:
        return {
            "boundaries": [],
            "requested_bins": requested,
            "effective_bins": 0,
            "non_null_count": 0,
            "missing_count": missing_count,
            "accepted": False,
            "rejection_reason": "continuous_discretization_no_non_null_values",
        }
    quantiles = np.linspace(0.0, 1.0, requested + 1)[1:-1]
    raw = np.quantile(non_null.to_numpy(dtype=float), quantiles) if len(quantiles) else np.array([], dtype=float)
    min_value = float(non_null.min())
    max_value = float(non_null.max())
    boundaries = sorted({
        float(value)
        for value in raw
        if np.isfinite(value) and min_value < float(value) < max_value
    })
    effective_bins = len(boundaries) + 1
    rejection_reason = ""
    accepted = effective_bins >= int(min_effective_bins)
    if not accepted:
        rejection_reason = "continuous_discretization_effective_bins_below_minimum"
    return {
        "boundaries": boundaries,
        "requested_bins": requested,
        "effective_bins": effective_bins,
        "non_null_count": int(len(non_null)),
        "missing_count": missing_count,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
    }


def apply_quantile_discretizer(series: pd.Series, *, boundaries: Sequence[float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    out = np.full(len(series), CONTINUOUS_MISSING_CATEGORY, dtype=object)
    if finite.any():
        bins = np.searchsorted(np.asarray(list(boundaries), dtype=float), values[finite], side="right")
        out[finite] = np.char.add("bin_", np.char.mod("%d", bins.astype(np.int64)))
    return pd.Series(out, index=series.index, dtype="string")


def _continuous_columns_from_edges(edges: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    for edge in edges:
        for item in edge.get("continuous_columns", []) or []:
            if isinstance(item, Mapping):
                transformed = str(item.get("transformed_column", ""))
                if transformed:
                    columns[transformed] = dict(item)
    return columns


def fit_apply_continuous_discretizers(
    *,
    fit_df: pd.DataFrame,
    apply_frames: Sequence[pd.DataFrame],
    source_table: str,
    continuous_columns: Mapping[str, Mapping[str, Any]],
    requested_bins: int,
    min_effective_bins: int,
    dataset: str = "",
    task: str = "",
    fold: int | str = "",
    fit_split: str = "fold_train",
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    if not continuous_columns:
        return fit_df, [frame for frame in apply_frames], [], {}
    fit_out = fit_df.copy()
    apply_out = [frame.copy() for frame in apply_frames]
    audit_rows: list[dict[str, Any]] = []
    boundary_rows: dict[str, Any] = {}
    for transformed, spec in sorted(continuous_columns.items()):
        original = str(spec.get("original_column", ""))
        if not original or original not in fit_df.columns:
            continue
        fitted = fit_quantile_discretizer(
            fit_df[original],
            requested_bins=requested_bins,
            min_effective_bins=min_effective_bins,
        )
        accepted = bool(fitted["accepted"])
        row = {
            "dataset": dataset,
            "task": task,
            "source_table": source_table,
            "original_column": original,
            "transformed_column": transformed,
            "fold": fold,
            "fit_split": fit_split,
            "requested_bins": fitted["requested_bins"],
            "effective_bins": fitted["effective_bins"],
            "non_null_count": fitted["non_null_count"],
            "missing_count": fitted["missing_count"],
            "missing_policy": f"explicit_category:{CONTINUOUS_MISSING_CATEGORY}",
            "accepted": accepted,
            "rejection_reason": fitted["rejection_reason"],
        }
        audit_rows.append(row)
        boundary_rows[transformed] = {
            "source_table": source_table,
            "original_column": original,
            "transformed_column": transformed,
            "fold": str(fold),
            "fit_split": fit_split,
            "requested_bins": fitted["requested_bins"],
            "effective_bins": fitted["effective_bins"],
            "boundaries": fitted["boundaries"],
            "missing_policy": row["missing_policy"],
            "accepted": accepted,
            "rejection_reason": fitted["rejection_reason"],
        }
        if accepted:
            fit_out[transformed] = apply_quantile_discretizer(fit_out[original], boundaries=fitted["boundaries"])
            for frame in apply_out:
                if original in frame.columns:
                    frame[transformed] = apply_quantile_discretizer(frame[original], boundaries=fitted["boundaries"])
    return fit_out, apply_out, audit_rows, boundary_rows


def _edge_lhs_lookup_columns(edge: FittedAmbiguityEdge | Mapping[str, Any]) -> list[str]:
    lhs_columns = list(edge.lhs_columns if isinstance(edge, FittedAmbiguityEdge) else edge.get("lhs_columns", ()))
    continuous = (
        edge.continuous_discretization
        if isinstance(edge, FittedAmbiguityEdge)
        else {"boundaries": _continuous_columns_from_edges([edge])}
    ) or {}
    boundaries = continuous.get("boundaries", {}) if isinstance(continuous, Mapping) else {}
    out: list[str] = []
    for col in lhs_columns:
        spec = boundaries.get(str(col), {}) if isinstance(boundaries, Mapping) else {}
        original = str(spec.get("original_column", "")) if isinstance(spec, Mapping) else ""
        out.append(original or str(col))
    return list(dict.fromkeys(out))


def _apply_fitted_edge_discretization_to_lookup(source_view: pd.DataFrame, edge: FittedAmbiguityEdge) -> pd.DataFrame:
    continuous = edge.continuous_discretization or {}
    boundaries = continuous.get("boundaries", {}) if isinstance(continuous, Mapping) else {}
    if not boundaries:
        return source_view
    out = source_view.copy()
    for transformed, spec in boundaries.items():
        if not isinstance(spec, Mapping) or not spec.get("accepted", False):
            continue
        original = str(spec.get("original_column", ""))
        if original in out.columns:
            out[str(transformed)] = apply_quantile_discretizer(
                out[original],
                boundaries=spec.get("boundaries", []),
            )
    return out


def discover_dmax1_edges(
    *,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
    max_edges: int = 4,
    min_support: int = 2,
    min_coverage: float = 0.1,
    continuous_fdhg_mode: str = "exclude",
    continuous_fdhg_bins: int = 8,
    continuous_fdhg_min_effective_bins: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, rejected, _audit = discover_dmax1_edges_with_audit(
        table_dict=table_dict,
        metadata=metadata,
        relations=relations,
        max_edges=max_edges,
        min_support=min_support,
        min_coverage=min_coverage,
        continuous_fdhg_mode=continuous_fdhg_mode,
        continuous_fdhg_bins=continuous_fdhg_bins,
        continuous_fdhg_min_effective_bins=continuous_fdhg_min_effective_bins,
    )
    return accepted, rejected


def discover_dmax1_edges_with_audit(
    *,
    table_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    relations: Sequence[Mapping[str, Any]],
    max_edges: int | None = 4,
    min_support: int = 2,
    min_coverage: float = 0.1,
    continuous_fdhg_mode: str = "exclude",
    continuous_fdhg_bins: int = 8,
    continuous_fdhg_min_effective_bins: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    column_audit: list[dict[str, Any]] = []
    for relation in sorted(relations, key=lambda r: (str(r.get("child_table", "")), str(r.get("child_fk", "")))):
        if relation.get("status") != "accepted":
            continue
        table_name = str(relation["child_table"])
        table = table_dict[table_name]
        df = _table_df(table)
        determinant_cols: list[str] = []
        dependent_cols: list[str] = []
        column_kinds: dict[str, str] = {}
        column_statuses: dict[str, str] = {}
        continuous_specs: dict[str, dict[str, Any]] = {}
        unique_counts: dict[str, int] = {}
        for col in sorted(df.columns):
            audit = column_eligibility_audit(
                col,
                df[col],
                metadata=metadata,
                table=table,
                source_entity_column=str(relation.get("child_fk", "")),
                min_support=min_support,
                continuous_fdhg_mode=continuous_fdhg_mode,
                continuous_fdhg_bins=continuous_fdhg_bins,
                continuous_fdhg_min_effective_bins=continuous_fdhg_min_effective_bins,
            )
            audit["source_table"] = table_name
            column_audit.append(audit)
            candidate_col = str(audit.get("transformed_column") or col)
            column_statuses[candidate_col] = str(audit.get("eligibility_status", ""))
            if audit["determinant_eligible"]:
                determinant_cols.append(candidate_col)
            if audit["dependent_eligible"]:
                dependent_cols.append(candidate_col)
            if str(audit.get("transformed_column", "")).startswith(CONTINUOUS_TRANSFORM_PREFIX):
                column_kinds[candidate_col] = "discretized_continuous"
                continuous_specs[candidate_col] = {
                    "original_column": col,
                    "transformed_column": candidate_col,
                    "requested_bins": continuous_fdhg_bins,
                    "effective_bins": audit.get("effective_bins", ""),
                }
            else:
                column_kinds[candidate_col] = "categorical"
            if not audit["dependent_eligible"]:
                rejected.append({
                    "source_table": table_name,
                    "lhs_columns": col,
                    "rhs_column": "",
                    "status": "rejected",
                    "rejection_reason": audit["exclusion_reason"],
                })
        if continuous_specs:
            df = df.copy()
            for transformed, spec in continuous_specs.items():
                original = str(spec["original_column"])
                fitted = fit_quantile_discretizer(
                    df[original],
                    requested_bins=continuous_fdhg_bins,
                    min_effective_bins=continuous_fdhg_min_effective_bins,
                )
                if fitted["accepted"]:
                    df[transformed] = apply_quantile_discretizer(df[original], boundaries=fitted["boundaries"])
        for lhs in determinant_cols:
            unique_counts.setdefault(lhs, int(df[lhs].nunique(dropna=True)))
            lhs_unique_ratio = float(unique_counts[lhs] / max(1, len(df)))
            if lhs_unique_ratio > 0.95:
                rejected.append({
                    "source_table": table_name,
                    "lhs_columns": lhs,
                    "rhs_column": "",
                    "status": "rejected",
                    "rejection_reason": "surrogate_key_identity_excluded",
                })
                continue
            for rhs in dependent_cols:
                if lhs == rhs:
                    continue
                if column_kinds.get(lhs) == "discretized_continuous" and column_kinds.get(rhs) == "discretized_continuous":
                    rejected.append({
                        "source_table": table_name,
                        "lhs_columns": lhs,
                        "rhs_column": rhs,
                        "status": "rejected",
                        "rejection_reason": "continuous_to_continuous_discretized_pair_excluded",
                    })
                    continue
                if _semantically_excluded_continuous_pair(
                    lhs_status=column_statuses.get(lhs, ""),
                    rhs_status=column_statuses.get(rhs, ""),
                ):
                    rejected.append({
                        "source_table": table_name,
                        "lhs_columns": lhs,
                        "rhs_column": rhs,
                        "status": "rejected",
                        "rejection_reason": "continuous_numeric_pair_excluded",
                    })
                    continue
                grouped = df[[lhs, rhs]].dropna().groupby(lhs, sort=True)[rhs].agg(["count", "nunique"])
                support = int(grouped["count"].sum()) if not grouped.empty else 0
                coverage = float(df[lhs].notna().mean()) if len(df) else 0.0
                if support <= 0:
                    reason = "zero_support"
                elif support < min_support:
                    reason = "support_below_minimum"
                elif coverage <= min_coverage:
                    reason = "coverage_below_threshold"
                elif grouped.empty or int((grouped["nunique"] > 1).sum()) == 0:
                    reason = "constant_or_non_informative_edge"
                else:
                    edge_id = f"{table_name}:{lhs}->{rhs}"
                    accepted.append({
                        "edge_id": edge_id,
                        "source_table": table_name,
                        "lhs_columns": (lhs,),
                        "rhs_column": rhs,
                        "lhs_original_columns": (
                            continuous_specs[lhs]["original_column"],
                        ) if lhs in continuous_specs else (lhs,),
                        "rhs_original_column": continuous_specs[rhs]["original_column"] if rhs in continuous_specs else rhs,
                        "lhs_column_kind": column_kinds.get(lhs, "categorical"),
                        "rhs_column_kind": column_kinds.get(rhs, "categorical"),
                        "continuous_columns": [
                            spec for col, spec in continuous_specs.items() if col in {lhs, rhs}
                        ],
                        "support": support,
                        "coverage": coverage,
                        "confidence": float((grouped["count"] / grouped["count"].sum()).max()) if support else 0.0,
                        "conflict_rate": float((grouped["nunique"] > 1).mean()) if len(grouped) else 0.0,
                        "edge_quality": _edge_quality(
                            confidence=float((grouped["count"] / grouped["count"].sum()).max()) if support else 0.0,
                            conflict_rate=float((grouped["nunique"] > 1).mean()) if len(grouped) else 0.0,
                        ),
                        "selection_status": "candidate",
                        "rejection_reason": "",
                    })
                    continue
                rejected.append({
                    "source_table": table_name,
                    "lhs_columns": lhs,
                    "rhs_column": rhs,
                    "status": "rejected",
                    "rejection_reason": reason,
                })
    accepted = sorted(
        accepted,
        key=lambda r: (
            -float(r["conflict_rate"]),
            -int(r["support"]),
            str(r["source_table"]),
            "|".join(r["lhs_columns"]),
            str(r["rhs_column"]),
        ),
    )
    if max_edges is not None:
        accepted = accepted[:max_edges]
    for row in accepted:
        row["selection_status"] = "accepted"
    return accepted, rejected, column_audit


class _FilteredTable:
    def __init__(self, original: Any, df: pd.DataFrame) -> None:
        self.df = df
        self.pkey_col = getattr(original, "pkey_col", None)
        self.fkey_col_to_pkey_table = getattr(original, "fkey_col_to_pkey_table", {}) or {}
        self.time_col = getattr(original, "time_col", None)


def discover_earliest_fold_candidate_edges(
    *,
    prepared: Mapping[str, Any],
    edge_budget: int,
    relations_key: str = "accepted_relations",
    continuous_fdhg_mode: str = "exclude",
    continuous_fdhg_bins: int = 8,
    continuous_fdhg_min_effective_bins: int = 2,
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    metadata = prepared["metadata"]
    discovery_fold = min(prepared["split_plan"]["folds"], key=lambda row: int(row["fold"]))
    train_targets = prepared["train_df"].loc[discovery_fold["train_indices"]].reset_index(drop=True)
    table_dict = prepared["table_dict"]
    filtered_tables: dict[str, Any] = {}
    source_counts: dict[str, dict[str, Any]] = {}
    rejected_static: list[dict[str, Any]] = []
    relations = prepared.get(relations_key, prepared["accepted_relations"])
    relation_count_by_table: dict[str, int] = {}
    for relation in relations:
        if relation.get("status") == "accepted":
            table_name = str(relation.get("child_table", ""))
            relation_count_by_table[table_name] = relation_count_by_table.get(table_name, 0) + 1
    for relation in relations:
        if relation.get("status") != "accepted":
            continue
        table_name = str(relation["child_table"])
        if relation_count_by_table.get(table_name, 0) != 1:
            rejected_static.append({
                "source_table": table_name,
                "lhs_columns": "",
                "rhs_column": "",
                "status": "rejected",
                "rejection_reason": "ambiguous_source_relation_for_table",
            })
            continue
        view = train_source_view(
            table=table_dict[table_name],
            metadata=metadata,
            train_targets=train_targets,
            validation_targets=pd.DataFrame(columns=train_targets.columns),
            relation=relation,
            target_lookup_value_mapping=(
                target_lookup_value_mapping
            ),
        )
        source_counts[table_name] = view["audit"]
        if view["blocked_reason"]:
            rejected_static.append({
                "source_table": table_name,
                "lhs_columns": "",
                "rhs_column": "",
                "status": "rejected",
                "rejection_reason": view["blocked_reason"],
            })
            continue
        filtered_tables[table_name] = _FilteredTable(table_dict[table_name], view["fit_rows"])
    all_accepted, rejected, column_audit = discover_dmax1_edges_with_audit(
        table_dict={**table_dict, **filtered_tables},
        metadata=metadata,
        relations=[
            relation
            for relation in relations
            if str(relation.get("child_table", "")) in filtered_tables
        ],
        max_edges=None,
        continuous_fdhg_mode=continuous_fdhg_mode,
        continuous_fdhg_bins=continuous_fdhg_bins,
        continuous_fdhg_min_effective_bins=continuous_fdhg_min_effective_bins,
    )
    accepted = [dict(edge) for edge in all_accepted[: int(edge_budget)]]
    for idx, edge in enumerate(accepted, start=1):
        edge["edge_rank"] = idx
        relation = _relation_for_edge(relations, edge)
        if relation:
            edge["source_entity_column"] = relation.get(
                "child_fk",
                "",
            )
            edge["source_relation_id"] = _relation_id(
                relation
            )
            edge[
                "source_entity_column_resolution"
            ] = "accepted_relation_child_fk"

            # Event-row tasks separate prediction-row identity from
            # relational lookup.  Preserve this per relation/edge;
            # do not apply task-level lookup semantics to unrelated
            # relations.
            relation_matches_task_relation = (
                str(
                    relation.get(
                        "child_table",
                        "",
                    )
                )
                == str(
                    metadata.get(
                        "child_table",
                        "",
                    )
                )
                and str(
                    relation.get(
                        "child_fk",
                        "",
                    )
                )
                == str(
                    metadata.get(
                        "child_fk",
                        "",
                    )
                )
            )

            edge["target_lookup_column"] = str(
                relation.get(
                    "target_lookup_column",
                    (
                        metadata.get(
                            "relation_entity_key",
                            metadata["entity_key"],
                        )
                        if relation_matches_task_relation
                        else metadata["entity_key"]
                    ),
                )
            )

            if relation.get(
                "target_lookup_value_transform"
            ):
                edge[
                    "target_lookup_value_transform"
                ] = str(
                    relation[
                        "target_lookup_value_transform"
                    ]
                )

            edge["strict_before"] = bool(
                relation.get(
                    "strict_before",
                    (
                        metadata.get(
                            "strict_before",
                            False,
                        )
                        if relation_matches_task_relation
                        else False
                    ),
                )
            )

            edge["allow_exact_matches"] = (
                not edge["strict_before"]
            )
    max_timestamp = ""
    if source_counts:
        horizons = [
            row.get("reliability_fit_horizon")
            for row in source_counts.values()
            if row.get("reliability_fit_horizon")
        ]
        max_timestamp = str(max(horizons)) if horizons else ""
    rejected_all = [*rejected_static, *rejected]
    provenance = {
        "candidate_discovery_protocol": "fixed_from_earliest_inner_train_fold",
        "candidate_discovery_fold": int(discovery_fold["fold"]),
        "candidate_discovery_target_row_count": len(train_targets),
        "candidate_discovery_fit_horizon": _target_horizon(train_targets, metadata),
        "candidate_discovery_entity_count": int(train_targets[metadata["entity_key"]].nunique(dropna=True)),
        "candidate_discovery_scope": "earliest_inner_train_fold_source_snapshot",
        "candidate_discovery_max_timestamp": max_timestamp,
        "candidate_discovery_source_row_counts": source_counts,
        "candidate_column_audit": column_audit,
        "continuous_fdhg_mode": continuous_fdhg_mode,
        "continuous_fdhg_bins": int(continuous_fdhg_bins),
        "continuous_fdhg_min_effective_bins": int(continuous_fdhg_min_effective_bins),
        "candidate_pair_count_before_edge_validation": _candidate_pair_count_before_edge_validation(column_audit),
        "candidate_count_before_budget": len(all_accepted),
        "candidate_count_after_budget": len(accepted),
        "accepted_candidate_edge_count": len(accepted),
        "rejected_candidate_edge_count": len(rejected_all),
        "rejection_reason_counts": _rejection_reason_counts(rejected_all),
        "ordered_candidate_edge_ids": [str(edge.get("edge_id", "")) for edge in accepted],
        "inner_validation_rows_used_for_candidate_discovery": 0,
        "official_validation_rows_used_for_candidate_discovery": 0,
        "test_rows_used_for_candidate_discovery": 0,
    }
    return {
        "accepted_edges": accepted,
        "rejected_edges": rejected_all,
        "provenance": provenance,
    }


def train_source_view(
    *,
    table: Any,
    metadata: Mapping[str, Any],
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    relation: Mapping[str, Any] | None = None,
    edge: Mapping[str, Any] | None = None,
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    entity_key = str(metadata["entity_key"])
    target_time_col = str(metadata["target_time_col"])

    # Prediction-row identity and relational lookup identity can differ.
    # Resolve the target-side lookup per relation/edge so that source keys
    # such as itemid/visitorid are matched against the corresponding target
    # lookup column rather than against a synthetic prediction-row ID.
    target_lookup_column = str(
        (
            edge.get("target_lookup_column")
            if edge is not None
            and edge.get("target_lookup_column") is not None
            else relation.get("target_lookup_column")
            if relation is not None
            and relation.get("target_lookup_column") is not None
            else metadata.get(
                "relation_entity_key",
                entity_key,
            )
        )
    )

    df = _table_df(table)
    time_col = getattr(table, "time_col", None)
    source_entity, resolution, relation_id = _resolve_source_entity_column(
        table=table,
        df=df,
        metadata=metadata,
        relation=relation,
        edge=edge,
    )
    base_audit = {
        "reliability_fit_scope": "fold_train_temporal_entity_snapshot",
        "reliability_fit_horizon": "",
        "target_entity_key": entity_key,
        "target_lookup_column": target_lookup_column,
        "source_entity_column": source_entity,
        "source_entity_column_resolution": resolution,
        "source_relation_id": relation_id,
        "source_row_count_before_filtering": len(df),
        "source_row_count_after_filtering": 0,
        "validation_target_entity_overlap_in_reliability_rows": 0,
        "future_row_violation_count": 0,
        "official_validation_row_usage_count": 0,
        "test_row_usage_count": 0,
    }
    if not source_entity or source_entity not in df.columns:
        return {
            "fit_rows": df.iloc[0:0].copy(),
            "blocked_reason": "blocked_unresolvable_entity_linkage",
            "audit": {
                **base_audit,
                "reliability_fit_scope": "blocked_unresolvable_entity_linkage",
            },
        }
    if target_lookup_column not in train_targets.columns:
        return {
            "fit_rows": df.iloc[0:0].copy(),
            "blocked_reason": (
                "blocked_missing_target_lookup_column:"
                + target_lookup_column
            ),
            "audit": {
                **base_audit,
                "reliability_fit_scope":
                    "blocked_missing_target_lookup_column",
            },
        }

    train = train_targets[
        [target_lookup_column, target_time_col]
    ].copy()
    train["__target_time"] = pd.to_datetime(
        train[target_time_col],
        errors="coerce",
    )

    lookup_transform = (
        edge.get("target_lookup_value_transform")
        if edge is not None
        else relation.get("target_lookup_value_transform")
        if relation is not None
        else None
    )

    train["__lookup_entity"] = train[
        target_lookup_column
    ]

    if (
        lookup_transform
        == "dbinfer_inverse_entity_mapping"
    ):
        if target_lookup_value_mapping is None:
            return {
                "fit_rows": df.iloc[0:0].copy(),
                "blocked_reason":
                    "blocked_missing_dbinfer_inverse_entity_mapping",
                "audit": {
                    **base_audit,
                    "reliability_fit_scope":
                        "blocked_missing_dbinfer_inverse_entity_mapping",
                },
            }

        train["__lookup_entity"] = (
            train["__lookup_entity"].map(
                target_lookup_value_mapping
            )
        )

    entity_horizon = train.groupby(
        "__lookup_entity",
        sort=True,
    )["__target_time"].max()

    # Filter the source snapshot chunk-wise.  This is semantically
    # identical to constructing a full-length per-row horizon Series,
    # but avoids materializing that Series for very large source tables.
    is_temporal = bool(
        time_col
        and str(time_col) in df.columns
    )
    source_chunk_rows = 1_000_000
    fit_chunks: list[pd.DataFrame] = []
    future = 0

    for start in range(0, len(df), source_chunk_rows):
        stop = min(
            start + source_chunk_rows,
            len(df),
        )
        chunk = df.iloc[start:stop]
        chunk_horizons = chunk[source_entity].map(
            entity_horizon
        )

        if is_temporal:
            chunk_times = pd.to_datetime(
                chunk[str(time_col)],
                errors="coerce",
            )
            chunk_mask = (
                chunk_times.notna()
                & chunk_horizons.notna()
                & (chunk_times <= chunk_horizons)
            )
        else:
            chunk_mask = chunk_horizons.notna()

        if bool(chunk_mask.any()):
            fit_chunks.append(
                chunk.loc[chunk_mask].copy()
            )

    if fit_chunks:
        fit_rows = pd.concat(
            fit_chunks,
            axis=0,
            copy=False,
        )
    else:
        fit_rows = df.iloc[0:0].copy()

    if is_temporal:
        scope = "fold_train_temporal_entity_snapshot"
    else:
        scope = "fold_train_static_entity_snapshot"
    if source_entity != entity_key and entity_key not in fit_rows.columns:
        fit_rows[entity_key] = fit_rows[source_entity].to_numpy()
    validation_entities = set(validation_targets.get(entity_key, pd.Series(dtype=object)).dropna())
    overlap = int(fit_rows[source_entity].isin(validation_entities).sum()) if validation_entities and source_entity in fit_rows else 0
    max_horizon = entity_horizon.max()
    audit = {
        **base_audit,
        "reliability_fit_scope": scope,
        "reliability_fit_horizon": str(max_horizon) if pd.notna(max_horizon) else "",
        "source_row_count_after_filtering": len(fit_rows),
        "validation_target_entity_overlap_in_reliability_rows": overlap,
        "future_row_violation_count": future,
    }
    return {"fit_rows": fit_rows, "blocked_reason": "", "audit": audit}


def _candidate_pair_count_before_edge_validation(column_audit: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    by_table: dict[str, list[Mapping[str, Any]]] = {}
    for row in column_audit:
        by_table.setdefault(str(row.get("source_table", "")), []).append(row)
    for rows in by_table.values():
        determinants = [row for row in rows if row.get("determinant_eligible")]
        dependents = [row for row in rows if row.get("dependent_eligible")]
        for lhs in determinants:
            for rhs in dependents:
                if lhs.get("column") != rhs.get("column"):
                    count += 1
    return count


def _rejection_reason_counts(rejected: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rejected:
        reason = str(row.get("rejection_reason", ""))
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _relation_for_edge(
    relations: Sequence[Mapping[str, Any]] | None,
    edge: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if relations is None:
        return None
    source_table = str(edge.get("source_table", ""))
    matches = [
        relation
        for relation in relations
        if str(relation.get("child_table", "")) == source_table
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _relation_id(relation: Mapping[str, Any]) -> str:
    return f"{relation.get('child_table', '')}:{relation.get('child_fk', '')}->{relation.get('parent_table', '')}:{relation.get('parent_key', '')}"


def _resolve_source_entity_column(
    *,
    table: Any,
    df: pd.DataFrame,
    metadata: Mapping[str, Any],
    relation: Mapping[str, Any] | None,
    edge: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    if edge and edge.get("source_entity_column"):
        return (
            str(edge["source_entity_column"]),
            str(edge.get("source_entity_column_resolution") or "edge_metadata"),
            str(edge.get("source_relation_id") or ""),
        )
    if relation and relation.get("child_fk"):
        return str(relation["child_fk"]), "accepted_relation_child_fk", _relation_id(relation)
    entity_key = str(metadata["entity_key"])
    if entity_key in df.columns:
        return entity_key, "target_entity_key_present_in_source_table", ""
    fkeys = getattr(table, "fkey_col_to_pkey_table", {}) or {}
    if len(fkeys) == 1:
        col = str(next(iter(fkeys.keys())))
        return col, "single_source_foreign_key_fallback", ""
    return "", "unresolved", ""


def _target_horizon(targets: pd.DataFrame, metadata: Mapping[str, Any]) -> str:
    value = pd.to_datetime(targets[str(metadata["target_time_col"])], errors="coerce").max()
    return str(value) if pd.notna(value) else ""


def _continuous_numeric_pair(lhs: pd.Series, rhs: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(lhs) or pd.api.types.is_bool_dtype(rhs):
        return False
    if not pd.api.types.is_numeric_dtype(lhs) or not pd.api.types.is_numeric_dtype(rhs):
        return False
    return not (pd.api.types.is_integer_dtype(lhs) and pd.api.types.is_integer_dtype(rhs))


def _semantically_excluded_continuous_pair(*, lhs_status: str, rhs_status: str) -> bool:
    excluded_statuses = {
        "high_cardinality_numeric_excluded",
        "surrogate_key_identity_excluded",
        "continuous_discretization_no_non_null_values",
        "continuous_discretization_effective_bins_below_minimum",
    }
    return str(lhs_status) in excluded_statuses or str(rhs_status) in excluded_statuses


def _edge_quality(*, confidence: float, conflict_rate: float) -> str:
    if confidence >= 0.8 and conflict_rate < 1.0:
        return "accepted_dependency"
    return "accepted_ambiguity_probe"


def _fit_afd_edge_chunked_exclude(
    *,
    full_df: pd.DataFrame,
    edge: Mapping[str, Any],
    table_name: str,
    time_col: str,
    fit_horizon: Any,
    horizon: pd.Timestamp | None,
    fold: int | str | None,
    chunk_rows: int = 1_000_000,
) -> FittedAmbiguityEdge:
    """Exact bounded-memory equivalent of fit_ambiguity_map + edge_from_mapping.

    This path is used only when continuous FDHG discretization is disabled.
    It preserves the full source population and the original temporal horizon.
    """
    lhs_columns = [
        str(col)
        for col in edge["lhs_columns"]
    ]
    rhs_column = str(edge["rhs_column"])

    # Accumulate exact normalized (lhs, rhs) frequencies.
    pair_counts: dict[tuple[str, str], int] = {}

    # edge_from_mapping computes coverage using the number of distinct
    # raw, fully non-null LHS tuples. Candidate determinants are normally
    # low-cardinality, so retaining only per-chunk uniques is bounded in
    # the intended FDHG candidate space.
    lhs_unique_parts: list[pd.DataFrame] = []

    fit_start: pd.Timestamp | None = None
    max_time: pd.Timestamp | None = None

    projection = [
        *lhs_columns,
        rhs_column,
    ]

    for start in range(
        0,
        len(full_df),
        int(chunk_rows),
    ):
        stop = min(
            start + int(chunk_rows),
            len(full_df),
        )
        chunk = full_df.iloc[start:stop]

        chunk_times = pd.to_datetime(
            chunk[time_col],
            errors="coerce",
        )

        if horizon is not None:
            chunk_mask = chunk_times <= horizon
            if not bool(chunk_mask.any()):
                continue

            work = chunk.loc[
                chunk_mask,
                projection,
            ].copy()
            selected_times = chunk_times.loc[
                chunk_mask
            ]
        else:
            # Legacy fit_afd_edges includes every source row when there
            # is no horizon, including rows whose source time is NaT.
            work = chunk[
                projection
            ].copy()
            selected_times = chunk_times

        valid_times = selected_times.dropna()
        if len(valid_times):
            chunk_min = valid_times.min()
            chunk_max = valid_times.max()

            if (
                fit_start is None
                or chunk_min < fit_start
            ):
                fit_start = chunk_min

            if (
                max_time is None
                or chunk_max > max_time
            ):
                max_time = chunk_max

        if lhs_columns:
            raw_unique = (
                work[lhs_columns]
                .dropna()
                .drop_duplicates()
            )
            if len(raw_unique):
                lhs_unique_parts.append(
                    raw_unique
                )

        normalized = pd.DataFrame(
            index=work.index
        )
        normalized["lhs_norm"] = (
            normalize_lhs_frame(
                work,
                lhs_columns,
            )
        )
        normalized["rhs_norm"] = (
            normalize_series(
                work[rhs_column]
            )
        )

        normalized = normalized[
            (normalized["lhs_norm"] != NULL_TOKEN)
            & (normalized["rhs_norm"] != NULL_TOKEN)
        ]

        if normalized.empty:
            continue

        chunk_counts = (
            normalized
            .groupby(
                ["lhs_norm", "rhs_norm"],
                sort=True,
            )
            .size()
        )

        for key, count in chunk_counts.items():
            lhs_norm = str(key[0])
            rhs_norm = str(key[1])
            pair_key = (
                lhs_norm,
                rhs_norm,
            )
            pair_counts[pair_key] = (
                pair_counts.get(
                    pair_key,
                    0,
                )
                + int(count)
            )

    if lhs_unique_parts:
        lhs_unique_count = int(
            pd.concat(
                lhs_unique_parts,
                axis=0,
                ignore_index=True,
                copy=False,
            )
            .drop_duplicates()
            .shape[0]
        )
    else:
        lhs_unique_count = 0

    if pair_counts:
        counts = pd.DataFrame(
            [
                {
                    "lhs_norm": lhs_norm,
                    "rhs_norm": rhs_norm,
                    "n": int(count),
                }
                for (
                    lhs_norm,
                    rhs_norm,
                ), count in pair_counts.items()
            ]
        ).sort_values(
            ["lhs_norm", "rhs_norm"],
            kind="mergesort",
        ).reset_index(drop=True)

        mapping_rows: list[
            dict[str, Any]
        ] = []

        for lhs_value, group in counts.groupby(
            "lhs_norm",
            sort=True,
        ):
            ordered = (
                group["n"]
                .sort_values(
                    ascending=False
                )
                .to_numpy(dtype=float)
            )
            total = float(
                ordered.sum()
            )
            top = (
                float(ordered[0])
                if len(ordered)
                else 0.0
            )
            second = (
                float(ordered[1])
                if len(ordered) > 1
                else 0.0
            )

            mapping_rows.append({
                "lhs_norm": lhs_value,
                "majority_confidence":
                    top / total
                    if total
                    else np.nan,
                "entropy":
                    entropy_from_counts(
                        group["n"]
                    ),
                "conflict_count":
                    int(
                        group[
                            "rhs_norm"
                        ].nunique()
                    ),
                "support_count":
                    int(total),
                "top1_margin":
                    (top - second) / total
                    if total
                    else np.nan,
            })

        mapping = (
            pd.DataFrame(mapping_rows)
            .sort_values(
                "lhs_norm",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
    else:
        mapping = pd.DataFrame(
            columns=[
                "lhs_norm",
                "majority_confidence",
                "entropy",
                "conflict_count",
                "support_count",
                "top1_margin",
            ]
        )

    support = (
        int(
            mapping[
                "support_count"
            ].sum()
        )
        if "support_count" in mapping
        else 0
    )

    confidence = (
        float(
            mapping[
                "majority_confidence"
            ].mean()
        )
        if (
            "majority_confidence"
            in mapping
            and len(mapping)
        )
        else 0.0
    )

    conflict_rate = (
        float(
            (
                mapping[
                    "conflict_count"
                ].fillna(0)
                > 1
            ).mean()
        )
        if (
            "conflict_count"
            in mapping
            and len(mapping)
        )
        else 0.0
    )

    coverage = (
        float(
            len(mapping)
            / max(
                1,
                lhs_unique_count,
            )
        )
        if lhs_columns
        else 0.0
    )

    return FittedAmbiguityEdge(
        edge_id=str(edge["edge_id"]),
        source_table=table_name,
        lhs_columns=tuple(lhs_columns),
        rhs_column=rhs_column,
        mapping=mapping,
        fit_start_time=(
            str(fit_start)
            if fit_start is not None
            else None
        ),
        fit_end_time=(
            str(fit_horizon)
            if fit_horizon is not None
            else None
        ),
        maximum_source_time_used=(
            str(max_time)
            if max_time is not None
            else None
        ),
        support=support,
        coverage=coverage,
        confidence=confidence,
        conflict_rate=conflict_rate,
        selection_status="accepted",
        rejection_reason="",
        fold=fold,
        continuous_discretization=None,
    )


def fit_afd_edges(
    *,
    inner_train_rows: pd.DataFrame,
    source_tables: Mapping[str, Any],
    schema: Any,
    task_metadata: Mapping[str, Any],
    candidate_edges: Sequence[Mapping[str, Any]],
    max_edges: int,
    fold: int | str | None = None,
    fit_horizon: Any = None,
    min_coverage: float = 0.1,
    continuous_fdhg_mode: str = "exclude",
    continuous_fdhg_bins: int = 8,
    continuous_fdhg_min_effective_bins: int = 2,
    dataset: str = "",
    task: str = "",
    fit_split: str = "fold_train",
) -> list[FittedAmbiguityEdge]:
    del inner_train_rows, schema
    fitted: list[FittedAmbiguityEdge] = []
    horizon = pd.to_datetime(fit_horizon) if fit_horizon is not None else None
    for edge in list(candidate_edges)[:max_edges]:
        table_name = str(edge["source_table"])
        table = source_tables[table_name]
        full_df = _table_df(table)
        time_col = getattr(table, "time_col", None)
        continuous_specs = _continuous_columns_from_edges([edge]) if continuous_fdhg_mode == "quantile" else {}
        if not time_col or str(time_col) not in full_df.columns:
            fitted.append(FittedAmbiguityEdge(
                edge_id=str(edge["edge_id"]),
                source_table=table_name,
                lhs_columns=tuple(edge["lhs_columns"]),
                rhs_column=str(edge["rhs_column"]),
                mapping=pd.DataFrame(),
                fit_start_time=None,
                fit_end_time=str(fit_horizon) if fit_horizon is not None else None,
                maximum_source_time_used=None,
                support=0,
                coverage=0.0,
                confidence=0.0,
                conflict_rate=0.0,
                selection_status="rejected",
                rejection_reason="missing_source_time_for_point_in_time_lookup",
                fold=fold,
                continuous_discretization=None,
            ))
            continue
        required_cols = [str(time_col), *edge["lhs_columns"], str(edge["rhs_column"])]
        for spec in continuous_specs.values():
            required_cols.append(str(spec.get("original_column", "")))
        required_cols = [col for col in dict.fromkeys(required_cols) if col]
        missing = [col for col in required_cols if col not in full_df.columns]
        transformed_missing = [col for col in missing if col in continuous_specs]
        missing = [col for col in missing if col not in transformed_missing]
        if missing:
            raise ValueError(f"missing_ambiguity_columns:{','.join(sorted(missing))}")

        # The paper's frozen default uses continuous_fdhg_mode="exclude".
        # For that mode, fit the same ambiguity statistics in bounded
        # chunks instead of materializing the entire horizon-filtered
        # source table. Quantile mode intentionally retains the original
        # implementation below.
        if not continuous_specs:
            fitted_edge = _fit_afd_edge_chunked_exclude(
                full_df=full_df,
                edge=edge,
                table_name=table_name,
                time_col=str(time_col),
                fit_horizon=fit_horizon,
                horizon=horizon,
                fold=fold,
            )

            reason = ""
            if fitted_edge.support <= 0:
                reason = "zero_support"
            elif fitted_edge.coverage <= min_coverage:
                reason = "coverage_below_threshold"

            if reason:
                fitted_edge = FittedAmbiguityEdge(
                    edge_id=fitted_edge.edge_id,
                    source_table=fitted_edge.source_table,
                    lhs_columns=fitted_edge.lhs_columns,
                    rhs_column=fitted_edge.rhs_column,
                    mapping=fitted_edge.mapping,
                    fit_start_time=fitted_edge.fit_start_time,
                    fit_end_time=fitted_edge.fit_end_time,
                    maximum_source_time_used=(
                        fitted_edge.maximum_source_time_used
                    ),
                    support=fitted_edge.support,
                    coverage=fitted_edge.coverage,
                    confidence=fitted_edge.confidence,
                    conflict_rate=fitted_edge.conflict_rate,
                    selection_status="rejected",
                    rejection_reason=reason,
                    fold=fitted_edge.fold,
                    continuous_discretization=None,
                )

            if (
                fitted_edge.maximum_source_time_used
                and fit_horizon is not None
                and pd.Timestamp(
                    fitted_edge.maximum_source_time_used
                )
                > pd.Timestamp(fit_horizon)
            ):
                raise AssertionError(
                    "maximum_source_time_used_exceeds_fit_horizon"
                )

            fitted.append(fitted_edge)
            continue

        if horizon is not None and time_col and time_col in full_df.columns:
            times = pd.to_datetime(full_df[time_col], errors="coerce")
            mask = times <= horizon
            df = full_df.loc[mask, [col for col in required_cols if col in full_df.columns]].copy()
            if len(df) and times.loc[df.index].max() > horizon:
                raise AssertionError("fdhg_fit_uses_future_source_rows")
        else:
            df = full_df[[col for col in required_cols if col in full_df.columns]].copy()
        continuous_audit: list[dict[str, Any]] = []
        continuous_boundaries: dict[str, Any] = {}
        if continuous_specs:
            df, _unused, continuous_audit, continuous_boundaries = fit_apply_continuous_discretizers(
                fit_df=df,
                apply_frames=[],
                source_table=table_name,
                continuous_columns=continuous_specs,
                requested_bins=continuous_fdhg_bins,
                min_effective_bins=continuous_fdhg_min_effective_bins,
                dataset=dataset,
                task=task,
                fold=fold if fold is not None else "",
                fit_split=fit_split,
            )
            unavailable = [
                col
                for col in [*edge["lhs_columns"], str(edge["rhs_column"])]
                if col in continuous_specs and col not in df.columns
            ]
            if unavailable:
                fitted.append(FittedAmbiguityEdge(
                    edge_id=str(edge["edge_id"]),
                    source_table=table_name,
                    lhs_columns=tuple(edge["lhs_columns"]),
                    rhs_column=str(edge["rhs_column"]),
                    mapping=pd.DataFrame(),
                    fit_start_time=None,
                    fit_end_time=str(fit_horizon) if fit_horizon is not None else None,
                    maximum_source_time_used=None,
                    support=0,
                    coverage=0.0,
                    confidence=0.0,
                    conflict_rate=0.0,
                    selection_status="rejected",
                    rejection_reason="continuous_discretization_effective_bins_below_minimum",
                    fold=fold,
                    continuous_discretization={
                        "audit": continuous_audit,
                        "boundaries": continuous_boundaries,
                    },
                ))
                continue
        mapping = fit_ambiguity_map(
            df,
            lhs_columns=edge["lhs_columns"],
            rhs_column=str(edge["rhs_column"]),
        )
        fitted_edge = edge_from_mapping(
            edge_id=str(edge["edge_id"]),
            source_table=table_name,
            lhs_columns=edge["lhs_columns"],
            rhs_column=str(edge["rhs_column"]),
            mapping=mapping,
            fit_df=df,
            time_col=time_col,
            fit_horizon=fit_horizon,
            fold=fold,
            continuous_discretization={
                "audit": continuous_audit,
                "boundaries": continuous_boundaries,
            } if continuous_specs else None,
        )
        reason = ""
        if fitted_edge.support <= 0:
            reason = "zero_support"
        elif fitted_edge.coverage <= min_coverage:
            reason = "coverage_below_threshold"
        if reason:
            fitted_edge = FittedAmbiguityEdge(
                edge_id=fitted_edge.edge_id,
                source_table=fitted_edge.source_table,
                lhs_columns=fitted_edge.lhs_columns,
                rhs_column=fitted_edge.rhs_column,
                mapping=fitted_edge.mapping,
                fit_start_time=fitted_edge.fit_start_time,
                fit_end_time=fitted_edge.fit_end_time,
                maximum_source_time_used=fitted_edge.maximum_source_time_used,
                support=fitted_edge.support,
                coverage=fitted_edge.coverage,
                confidence=fitted_edge.confidence,
                conflict_rate=fitted_edge.conflict_rate,
                selection_status="rejected",
                rejection_reason=reason,
                fold=fitted_edge.fold,
                continuous_discretization=fitted_edge.continuous_discretization,
            )
        if (
            fitted_edge.maximum_source_time_used
            and fit_horizon is not None
            and pd.Timestamp(fitted_edge.maximum_source_time_used) > pd.Timestamp(fit_horizon)
        ):
            raise AssertionError("maximum_source_time_used_exceeds_fit_horizon")
        fitted.append(fitted_edge)
    return fitted


def point_in_time_asof_join(
    *,
    target_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    entity_key: str,
    target_time_col: str,
    source_time_col: str,
    source_columns: Sequence[str],
    source_entity_key: str | None = None,
    target_lookup_entity_key: str | None = None,
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
    allow_exact_matches: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # ``entity_key`` is prediction-row identity and must be preserved
    # in the returned row order.  Relational lookup may use a
    # different target-side column for event-row tasks.
    target_lookup_entity = str(
        target_lookup_entity_key
        or entity_key
    )
    source_entity = str(
        source_entity_key
        or target_lookup_entity
    )

    required_target = list(
        dict.fromkeys(
            [
                entity_key,
                target_lookup_entity,
                target_time_col,
            ]
        )
    )
    required_source = list(
        dict.fromkeys(
            [
                source_entity,
                source_time_col,
                *source_columns,
            ]
        )
    )

    missing_target = [
        col
        for col in required_target
        if col not in target_rows.columns
    ]
    missing_source = [
        col
        for col in required_source
        if col not in source_rows.columns
    ]

    if missing_target:
        raise ValueError(
            "missing_target_lookup_columns:"
            + ",".join(
                sorted(missing_target)
            )
        )

    if missing_source:
        raise ValueError(
            "missing_source_lookup_columns:"
            + ",".join(
                sorted(missing_source)
            )
        )

    target = (
        target_rows[
            required_target
        ]
        .reset_index(drop=True)
        .copy()
    )
    target["__target_row_id"] = np.arange(
        len(target),
        dtype=np.int64,
    )
    target["__target_time"] = pd.to_datetime(
        target[target_time_col],
        errors="coerce",
    )

    source = (
        source_rows[
            required_source
        ]
        .reset_index(drop=True)
        .copy()
    )
    source["__source_pos"] = np.arange(
        len(source),
        dtype=np.int64,
    )
    source["__source_time"] = pd.to_datetime(
        source[source_time_col],
        errors="coerce",
    )

    target_lookup_values = target[
        target_lookup_entity
    ]

    if target_lookup_value_mapping is not None:
        target_lookup_values = (
            target_lookup_values.map(
                target_lookup_value_mapping
            )
        )

    (
        target["__lookup_entity"],
        source["__lookup_entity"],
    ) = normalize_join_key_pair(
        target_lookup_values,
        source[source_entity],
    )

    source = source[
        source["__source_time"].notna()
    ].copy()

    empty_cols = [
        entity_key,
        target_time_col,
        "__target_row_id",
        "__matched_source_time",
        *source_columns,
    ]

    out = target[
        [
            entity_key,
            target_time_col,
            "__target_row_id",
        ]
    ].copy()

    out["__matched_source_time"] = pd.NaT

    value_arrays = {
        str(col): np.full(
            len(target),
            np.nan,
            dtype=object,
        )
        for col in source_columns
    }

    valid_target = target[
        target["__target_time"].notna()
    ].copy()

    if (
        not valid_target.empty
        and not source.empty
    ):
        base = valid_target.sort_values(
            [
                "__target_time",
                "__lookup_entity",
                "__target_row_id",
            ],
            kind="mergesort",
        )

        child = source.sort_values(
            [
                "__source_time",
                "__lookup_entity",
                "__source_pos",
            ],
            kind="mergesort",
        )

        merged = pd.merge_asof(
            base,
            child[
                [
                    "__lookup_entity",
                    "__source_time",
                    *source_columns,
                ]
            ],
            left_on="__target_time",
            right_on="__source_time",
            by="__lookup_entity",
            direction="backward",
            allow_exact_matches=bool(
                allow_exact_matches
            ),
        )

        positions = merged[
            "__target_row_id"
        ].to_numpy(
            dtype=np.int64
        )

        out.loc[
            positions,
            "__matched_source_time",
        ] = merged[
            "__source_time"
        ].to_numpy()

        for col in source_columns:
            value_arrays[
                str(col)
            ][positions] = merged[
                col
            ].to_numpy()

    for col in source_columns:
        out[col] = value_arrays[
            str(col)
        ]

    out = (
        out[
            empty_cols
        ]
        .sort_values(
            "__target_row_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(out) != len(target_rows):
        raise AssertionError(
            "point_in_time_lookup_row_count_changed"
        )

    if out[
        "__target_row_id"
    ].duplicated().any():
        raise AssertionError(
            "point_in_time_lookup_duplicated_target_rows"
        )

    expected_row_ids = pd.Series(
        np.arange(
            len(target_rows),
            dtype=np.int64,
        )
    )

    if not out[
        "__target_row_id"
    ].equals(expected_row_ids):
        raise AssertionError(
            "point_in_time_lookup_lost_target_rows"
        )

    if not (
        out[entity_key]
        .reset_index(drop=True)
        .equals(
            target_rows[
                entity_key
            ].reset_index(drop=True)
        )
    ):
        raise AssertionError(
            "point_in_time_lookup_entity_keys_changed"
        )

    target_times = (
        pd.to_datetime(
            target_rows[
                target_time_col
            ],
            errors="coerce",
        )
        .reset_index(drop=True)
    )

    if not (
        pd.Series(
            pd.to_datetime(
                out[target_time_col],
                errors="coerce",
            )
        )
        .reset_index(drop=True)
        .equals(target_times)
    ):
        raise AssertionError(
            "point_in_time_lookup_target_times_changed"
        )

    matched = pd.to_datetime(
        out["__matched_source_time"],
        errors="coerce",
    )

    future = (
        matched.notna()
        & target_times.notna()
        & (matched > target_times)
    )

    exact = (
        matched.notna()
        & target_times.notna()
        & (matched == target_times)
    )

    exact_violation = (
        exact
        if not allow_exact_matches
        else pd.Series(
            False,
            index=matched.index,
        )
    )

    temporal_violation = (
        future
        | exact_violation
    )

    audit = {
        "target_row_count":
            len(target_rows),
        "matched_target_rows":
            int(matched.notna().sum()),
        "unmatched_target_rows":
            int(matched.isna().sum()),
        "target_lookup_coverage": (
            float(
                matched.notna().mean()
            )
            if len(matched)
            else 0.0
        ),
        "maximum_lookup_source_time": (
            str(matched.max())
            if matched.notna().any()
            else None
        ),
        "maximum_target_time": (
            str(target_times.max())
            if target_times.notna().any()
            else None
        ),
        "future_lookup_violation_count":
            int(future.sum()),
        "exact_match_violation_count":
            int(exact_violation.sum()),
        "temporal_lookup_violation_count":
            int(temporal_violation.sum()),

        # Prediction-row identity.
        "target_entity_key":
            entity_key,

        # Relational lookup columns.
        "target_lookup_entity_column":
            target_lookup_entity,
        "source_entity_column":
            source_entity,

        "allow_exact_matches":
            bool(allow_exact_matches),
        "temporal_predicate": (
            "<="
            if allow_exact_matches
            else "<"
        ),
    }

    if (
        audit[
            "future_lookup_violation_count"
        ]
        != 0
    ):
        raise AssertionError(
            "point_in_time_lookup_future_violation"
        )

    if (
        audit[
            "exact_match_violation_count"
        ]
        != 0
    ):
        raise AssertionError(
            "point_in_time_lookup_exact_match_violation"
        )

    return (
        out[
            [
                *source_columns,
                "__matched_source_time",
            ]
        ].copy(),
        audit,
    )


def resolve_source_lookup_entity_key(
    *,
    table: Any,
    source_rows: pd.DataFrame,
    target_entity_key: str,
    edge_source_entity_column: str | None = None,
) -> str:
    if edge_source_entity_column and str(edge_source_entity_column) in source_rows.columns:
        return str(edge_source_entity_column)
    fkeys = getattr(table, "fkey_col_to_pkey_table", {}) or {}
    if len(fkeys) == 1:
        child_fk = str(next(iter(fkeys.keys())))
        if child_fk in source_rows.columns:
            return child_fk
    if str(target_entity_key) in source_rows.columns:
        return str(target_entity_key)
    requested = str(edge_source_entity_column or target_entity_key)
    raise ValueError(f"missing_source_lookup_columns:{requested}")


def materialize_ambiguity_features(
    *,
    fitted_edges: Sequence[FittedAmbiguityEdge],
    target_rows: pd.DataFrame,
    source_tables: Mapping[str, Any],
    task_metadata: Mapping[str, Any],
    source_entity_columns_by_edge: Mapping[str, str] | None = None,
    target_lookup_columns_by_edge: Mapping[str, str] | None = None,
    target_lookup_value_mappings_by_edge: Mapping[
        str,
        Mapping[Any, Any],
    ] | None = None,
    strict_before_by_edge: Mapping[str, bool] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    row_entity_key = str(
        task_metadata["entity_key"]
    )
    target_time_col = str(
        task_metadata["target_time_col"]
    )

    result = (
        target_rows[
            [
                row_entity_key,
                target_time_col,
            ]
        ]
        .reset_index(drop=True)
        .copy()
    )
    provenance: list[dict[str, Any]] = []
    lookup_audit: list[dict[str, Any]] = []
    for edge in fitted_edges:
        if edge.selection_status != "accepted":
            continue
        table = source_tables[edge.source_table]
        df = _table_df(table)
        edge_id = str(
            edge.edge_id
        )

        # Backward-compatible fallback is the prediction-row entity
        # key.  Event-row lookup is enabled only when the candidate
        # edge explicitly carries target_lookup_column metadata.
        target_lookup_entity_key = str(
            (
                None
                if target_lookup_columns_by_edge is None
                else target_lookup_columns_by_edge.get(
                    edge_id
                )
            )
            or row_entity_key
        )

        target_lookup_value_mapping = (
            None
            if target_lookup_value_mappings_by_edge is None
            else target_lookup_value_mappings_by_edge.get(
                edge_id
            )
        )

        strict_before = bool(
            False
            if strict_before_by_edge is None
            else strict_before_by_edge.get(
                edge_id,
                False,
            )
        )

        source_entity_key = (
            resolve_source_lookup_entity_key(
                table=table,
                source_rows=df,
                target_entity_key=(
                    target_lookup_entity_key
                ),
                edge_source_entity_column=(
                    None
                    if source_entity_columns_by_edge is None
                    else source_entity_columns_by_edge.get(
                        edge_id
                    )
                ),
            )
        )
        time_col = getattr(table, "time_col", None)
        if not time_col or str(time_col) not in df.columns:
            lookup_audit.append({
                "fold": edge.fold,
                "edge_id": edge.edge_id,
                "target_row_count": len(target_rows),
                "matched_target_rows": 0,
                "unmatched_target_rows": len(target_rows),
                "target_lookup_coverage": 0.0,
                "maximum_lookup_source_time": None,
                "maximum_target_time": str(
                    pd.to_datetime(
                        target_rows[target_time_col],
                        errors="coerce",
                    ).max()
                ),
                "future_lookup_violation_count": 0,
                "mapping_fit_horizon": edge.fit_end_time,
                "maximum_mapping_source_time": edge.maximum_source_time_used,
                "rejection_reason": "missing_source_time_for_point_in_time_lookup",
            })
            continue
        source_view, audit = point_in_time_asof_join(
            target_rows=target_rows,
            source_rows=df,

            # Prediction-row identity.
            entity_key=row_entity_key,

            # Relational lookup identity.
            target_lookup_entity_key=(
                target_lookup_entity_key
            ),
            target_lookup_value_mapping=(
                target_lookup_value_mapping
            ),
            source_entity_key=source_entity_key,

            target_time_col=target_time_col,
            source_time_col=str(time_col),
            source_columns=_edge_lhs_lookup_columns(
                edge
            ),

            # Only explicitly strict edges reject equal timestamps.
            allow_exact_matches=(
                not strict_before
            ),
        )
        source_view = _apply_fitted_edge_discretization_to_lookup(source_view, edge)
        audit.update({
            "fold": edge.fold,
            "edge_id": edge.edge_id,
            "mapping_fit_horizon": edge.fit_end_time,
            "maximum_mapping_source_time": edge.maximum_source_time_used,
            "rejection_reason": "",
        })
        if (
            edge.maximum_source_time_used
            and edge.fit_end_time
            and pd.Timestamp(edge.maximum_source_time_used) > pd.Timestamp(edge.fit_end_time)
        ):
            raise AssertionError("maximum_mapping_source_time_exceeds_fit_horizon")
        lookup_audit.append(audit)
        frame, rows = materialize_ambiguity_from_map(source_view, fitted_edge=edge)
        for col in frame.columns:
            result[col] = frame[col].to_numpy()
        provenance.extend(rows)
    return result, provenance, lookup_audit


def fit_transform_fdhg_fold(
    *,
    inner_train_rows: pd.DataFrame,
    inner_validation_rows: pd.DataFrame,
    source_tables: Mapping[str, Any],
    schema: Any,
    task_metadata: Mapping[str, Any],
    candidate_edges: Sequence[Mapping[str, Any]],
    max_edges: int,
    fold: int | str,
    continuous_fdhg_mode: str = "exclude",
    continuous_fdhg_bins: int = 8,
    continuous_fdhg_min_effective_bins: int = 2,
    dataset: str = "",
    task: str = "",
    fit_split: str = "fold_train",
    target_lookup_value_mapping: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    fit_horizon = inner_train_rows[task_metadata["target_time_col"]].max()
    fitted = fit_afd_edges(
        inner_train_rows=inner_train_rows,
        source_tables=source_tables,
        schema=schema,
        task_metadata=task_metadata,
        candidate_edges=candidate_edges,
        max_edges=max_edges,
        fold=fold,
        fit_horizon=fit_horizon,
        continuous_fdhg_mode=continuous_fdhg_mode,
        continuous_fdhg_bins=continuous_fdhg_bins,
        continuous_fdhg_min_effective_bins=continuous_fdhg_min_effective_bins,
        dataset=dataset,
        task=task,
        fit_split=fit_split,
    )
    accepted_fitted = [edge for edge in fitted if edge.selection_status == "accepted"]
    source_entity_columns_by_edge = {
        str(
            edge.get(
                "edge_id",
                "",
            )
        ): str(
            edge.get(
                "source_entity_column",
                "",
            )
        )
        for edge in candidate_edges
        if edge.get(
            "source_entity_column"
        )
    }

    target_lookup_columns_by_edge = {
        str(
            edge.get(
                "edge_id",
                "",
            )
        ): str(
            edge.get(
                "target_lookup_column",
                "",
            )
        )
        for edge in candidate_edges
        if edge.get(
            "target_lookup_column"
        )
    }

    target_lookup_value_mappings_by_edge = {
        str(edge.get("edge_id", "")):
            target_lookup_value_mapping
        for edge in candidate_edges
        if (
            target_lookup_value_mapping is not None
            and edge.get(
                "target_lookup_value_transform"
            )
            == "dbinfer_inverse_entity_mapping"
        )
    }

    strict_before_by_edge = {
        str(
            edge.get(
                "edge_id",
                "",
            )
        ): bool(
            edge.get(
                "strict_before",
                False,
            )
        )
        for edge in candidate_edges
        if "strict_before" in edge
    }
    train_x, train_prov, train_lookup_audit = materialize_ambiguity_features(
        fitted_edges=accepted_fitted,
        target_rows=inner_train_rows,
        source_tables=source_tables,
        task_metadata=task_metadata,
        source_entity_columns_by_edge=source_entity_columns_by_edge,
        target_lookup_columns_by_edge=target_lookup_columns_by_edge,
        target_lookup_value_mappings_by_edge=(
            target_lookup_value_mappings_by_edge
        ),
        strict_before_by_edge=strict_before_by_edge,
    )
    val_x, val_prov, val_lookup_audit = materialize_ambiguity_features(
        fitted_edges=accepted_fitted,
        target_rows=inner_validation_rows,
        source_tables=source_tables,
        task_metadata=task_metadata,
        source_entity_columns_by_edge=source_entity_columns_by_edge,
        target_lookup_columns_by_edge=target_lookup_columns_by_edge,
        target_lookup_value_mappings_by_edge=(
            target_lookup_value_mappings_by_edge
        ),
        strict_before_by_edge=strict_before_by_edge,
    )
    return {
        "fitted_edges": fitted,
        "edge_audit": [fitted_edge_to_audit_row(edge) for edge in fitted],
        "train_x": train_x,
        "validation_x": val_x,
        "feature_provenance": train_prov + val_prov,
        "target_lookup_audit": train_lookup_audit + val_lookup_audit,
        "continuous_discretization_audit": [
            row
            for edge in fitted
            for row in ((edge.continuous_discretization or {}).get("audit", []) if isinstance(edge.continuous_discretization, Mapping) else [])
        ],
        "continuous_discretization_boundaries": {
            str(edge.edge_id): (edge.continuous_discretization or {}).get("boundaries", {})
            for edge in fitted
            if isinstance(edge.continuous_discretization, Mapping)
        },
    }
