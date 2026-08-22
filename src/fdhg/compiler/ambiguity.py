from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


NULL_TOKEN = "__NULL__"
AMBIGUITY_STATS = (
    "majority_confidence",
    "entropy",
    "conflict_count",
    "support_count",
    "top1_margin",
)


@dataclass(frozen=True)
class FittedAmbiguityEdge:
    edge_id: str
    source_table: str
    lhs_columns: tuple[str, ...]
    rhs_column: str
    mapping: pd.DataFrame
    fit_start_time: str | None
    fit_end_time: str | None
    maximum_source_time_used: str | None
    support: int
    coverage: float
    confidence: float
    conflict_rate: float
    selection_status: str
    rejection_reason: str
    fold: int | str | None = None
    continuous_discretization: Mapping[str, Any] | None = None


def normalize_value(x: Any) -> str:
    if x is None:
        return NULL_TOKEN
    if isinstance(x, (list, tuple, np.ndarray)):
        return "|".join(normalize_value(value) for value in list(x))
    try:
        if pd.isna(x):
            return NULL_TOKEN
    except Exception:
        pass
    if isinstance(x, (bool, np.bool_)):
        return f"b:{int(bool(x))}"
    if isinstance(x, str):
        return f"s:{x}"
    numeric = _canonical_numeric_token(x)
    if numeric is not None:
        return numeric
    return f"s:{x}"


def _canonical_numeric_token(x: Any) -> str | None:
    if isinstance(x, Decimal):
        return _decimal_numeric_token(x)
    if isinstance(x, Integral) and not isinstance(x, (bool, np.bool_)):
        return f"n:{int(x)}"
    if isinstance(x, Real) and not isinstance(x, (bool, np.bool_)):
        value = float(x)
        if np.isposinf(value):
            return "n:+inf"
        if np.isneginf(value):
            return "n:-inf"
        if np.isnan(value):
            return NULL_TOKEN
        return _decimal_numeric_token(Decimal(str(value)))
    return None


def _decimal_numeric_token(value: Decimal) -> str:
    if value.is_nan():
        return NULL_TOKEN
    if value == Decimal("Infinity"):
        return "n:+inf"
    if value == Decimal("-Infinity"):
        return "n:-inf"
    try:
        normalized = value.normalize()
    except InvalidOperation:
        normalized = value
    if normalized == normalized.to_integral_value():
        return f"n:{int(normalized)}"
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"n:{text}"


def normalize_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(s):
        out = np.full(len(s), NULL_TOKEN, dtype=object)
        mask = s.notna().to_numpy()
        values = s.to_numpy()[mask].astype(np.int64)
        out[mask] = np.char.add("n:", np.char.mod("%d", values))
        return pd.Series(out, index=s.index, dtype="string")
    if pd.api.types.is_float_dtype(s):
        values = s.to_numpy(dtype=float, na_value=np.nan)
        out = np.full(len(s), NULL_TOKEN, dtype=object)
        finite = np.isfinite(values)
        finite_values = values[finite]
        rounded = np.rint(finite_values)
        if len(finite_values) and np.all(finite_values == rounded):
            text = np.char.mod("%d", rounded.astype(np.int64))
        else:
            text = np.char.mod("%.15g", finite_values)
        out[finite] = np.char.add("n:", text)
        out[np.isposinf(values)] = "n:+inf"
        out[np.isneginf(values)] = "n:-inf"
        return pd.Series(out, index=s.index, dtype="string")
    if pd.api.types.is_string_dtype(s) or str(s.dtype).startswith("category"):
        out = "s:" + s.astype("string")
        return out.fillna(NULL_TOKEN).astype("string")
    return s.map(normalize_value).astype("string")


def normalize_join_key_pair(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        if left_num.notna().all() and right_num.notna().all():
            return (
                pd.Series(left_num.to_numpy(dtype=np.int64), index=left.index),
                pd.Series(right_num.to_numpy(dtype=np.int64), index=right.index),
            )
    if str(left.dtype) == str(right.dtype):
        return left, right
    return left.map(normalize_value), right.map(normalize_value)


def normalize_lhs_frame(df: pd.DataFrame, lhs_columns: Sequence[str]) -> pd.Series:
    if not lhs_columns:
        return pd.Series([NULL_TOKEN] * len(df), index=df.index, dtype="string")
    parts = [normalize_series(df[col]) for col in lhs_columns]
    out = parts[0]
    for part in parts[1:]:
        out = out.str.cat(part, sep="|")
    return out.astype("string")


def entropy_from_counts(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum())


def fit_ambiguity_map(
    fit_df: pd.DataFrame,
    *,
    lhs_columns: Sequence[str],
    rhs_column: str,
) -> pd.DataFrame:
    columns = [*lhs_columns, rhs_column]
    missing = [col for col in columns if col not in fit_df.columns]
    if missing:
        raise ValueError(f"missing_ambiguity_columns:{','.join(sorted(missing))}")
    tmp = fit_df[columns].copy()
    tmp["lhs_norm"] = normalize_lhs_frame(tmp, lhs_columns)
    tmp["rhs_norm"] = normalize_series(tmp[rhs_column])
    tmp = tmp[(tmp["lhs_norm"] != NULL_TOKEN) & (tmp["rhs_norm"] != NULL_TOKEN)]
    if tmp.empty:
        return pd.DataFrame(
            columns=[
                "lhs_norm",
                "majority_confidence",
                "entropy",
                "conflict_count",
                "support_count",
                "top1_margin",
            ]
        )
    counts = tmp.groupby(["lhs_norm", "rhs_norm"], sort=True).size().rename("n").reset_index()
    rows: list[dict[str, Any]] = []
    for lhs_value, group in counts.groupby("lhs_norm", sort=True):
        ordered = group["n"].sort_values(ascending=False).to_numpy(dtype=float)
        total = float(ordered.sum())
        top = float(ordered[0]) if len(ordered) else 0.0
        second = float(ordered[1]) if len(ordered) > 1 else 0.0
        rows.append({
            "lhs_norm": lhs_value,
            "majority_confidence": top / total if total else np.nan,
            "entropy": entropy_from_counts(group["n"]),
            "conflict_count": int(group["rhs_norm"].nunique()),
            "support_count": int(total),
            "top1_margin": (top - second) / total if total else np.nan,
        })
    return pd.DataFrame(rows).sort_values("lhs_norm", kind="mergesort").reset_index(drop=True)


def materialize_ambiguity_from_map(
    target_source_df: pd.DataFrame,
    *,
    fitted_edge: FittedAmbiguityEdge,
    prefix: str = "f_fdhg",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    lhs_columns = list(fitted_edge.lhs_columns)
    missing = [col for col in lhs_columns if col not in target_source_df.columns]
    if missing:
        raise ValueError(f"missing_lhs_columns:{','.join(sorted(missing))}")
    edge_token = fitted_edge.edge_id.replace("->", "__to__").replace("|", "_")
    lhs_norm = normalize_lhs_frame(target_source_df, lhs_columns)
    tmp = pd.DataFrame({"__row": range(len(target_source_df)), "lhs_norm": lhs_norm})
    tmp = tmp.merge(fitted_edge.mapping, on="lhs_norm", how="left", sort=False)
    out = pd.DataFrame(index=target_source_df.index)
    provenance = []
    for stat in AMBIGUITY_STATS:
        col = f"{prefix}__{edge_token}__{stat}"
        out[col] = tmp[stat].to_numpy() if stat in tmp.columns else np.nan
        miss_col = f"{col}__is_missing"
        out[miss_col] = out[col].isna().astype("int8")
        for feature_name, aggregation in ((col, stat), (miss_col, f"{stat}_missing")):
            provenance.append({
                "feature_name": feature_name,
                "origin": "fdhg_residual",
                "edge_id": fitted_edge.edge_id,
                "source_table": fitted_edge.source_table,
                "lhs_columns": "|".join(lhs_columns),
                "rhs_column": fitted_edge.rhs_column,
                "aggregation_or_statistic": aggregation,
                "fold": fitted_edge.fold,
                "fit_horizon": fitted_edge.fit_end_time,
                "support": fitted_edge.support,
                "coverage": fitted_edge.coverage,
                "missing_rate": float(out[feature_name].isna().mean())
                if feature_name == col
                else 0.0,
            })
    return out, provenance


def fitted_edge_to_audit_row(edge: FittedAmbiguityEdge) -> dict[str, Any]:
    edge_quality = "rejected"
    if edge.selection_status == "accepted":
        edge_quality = (
            "accepted_dependency"
            if edge.confidence >= 0.8 and edge.conflict_rate < 1.0
            else "accepted_ambiguity_probe"
        )
    return {
        "fold": edge.fold,
        "edge_id": edge.edge_id,
        "source_table": edge.source_table,
        "lhs_columns": "|".join(edge.lhs_columns),
        "rhs_column": edge.rhs_column,
        "fit_start_time": edge.fit_start_time,
        "fit_end_time": edge.fit_end_time,
        "maximum_source_time_used": edge.maximum_source_time_used,
        "support": edge.support,
        "coverage": edge.coverage,
        "confidence": edge.confidence,
        "conflict_rate": edge.conflict_rate,
        "selection_status": edge.selection_status,
        "edge_quality": edge_quality,
        "rejection_reason": edge.rejection_reason,
    }


def edge_from_mapping(
    *,
    edge_id: str,
    source_table: str,
    lhs_columns: Sequence[str],
    rhs_column: str,
    mapping: pd.DataFrame,
    fit_df: pd.DataFrame,
    time_col: str | None,
    fit_horizon: Any,
    fold: int | str | None,
    status: str = "accepted",
    rejection_reason: str = "",
    continuous_discretization: Mapping[str, Any] | None = None,
) -> FittedAmbiguityEdge:
    times = pd.to_datetime(fit_df[time_col], errors="coerce") if time_col and time_col in fit_df else pd.Series(dtype="datetime64[ns]")
    fit_start = str(times.min()) if len(times.dropna()) else None
    max_time = str(times.max()) if len(times.dropna()) else None
    support = int(mapping["support_count"].sum()) if "support_count" in mapping else 0
    confidence = float(mapping["majority_confidence"].mean()) if "majority_confidence" in mapping and len(mapping) else 0.0
    conflict_rate = float((mapping["conflict_count"].fillna(0) > 1).mean()) if "conflict_count" in mapping and len(mapping) else 0.0
    coverage = float(len(mapping) / max(1, fit_df[list(lhs_columns)].dropna().drop_duplicates().shape[0])) if lhs_columns else 0.0
    return FittedAmbiguityEdge(
        edge_id=edge_id,
        source_table=source_table,
        lhs_columns=tuple(lhs_columns),
        rhs_column=rhs_column,
        mapping=mapping,
        fit_start_time=fit_start,
        fit_end_time=str(fit_horizon) if fit_horizon is not None else None,
        maximum_source_time_used=max_time,
        support=support,
        coverage=coverage,
        confidence=confidence,
        conflict_rate=conflict_rate,
        selection_status=status,
        rejection_reason=rejection_reason,
        fold=fold,
        continuous_discretization=continuous_discretization,
    )
