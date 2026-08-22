from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fdhg.compiler.ambiguity import NULL_TOKEN, normalize_lhs_frame, normalize_series


def compute_edge_reliability(
    rows: pd.DataFrame,
    *,
    lhs_columns: Sequence[str],
    rhs_column: str,
    edge_rank: int | None = None,
) -> dict[str, Any]:
    """Compute label-free dependency reliability for one FDHG edge.

    The convention matches the FDHG ambiguity feature fitter: rows with missing
    determinant or dependent values normalize to ``NULL_TOKEN`` and are excluded.
    When the dependent cardinality is 0, entropy-derived fields are NaN. When it
    is 1, normalized conditional entropy is 0 and reliability_entropy is 1.
    """
    required = [*lhs_columns, rhs_column]
    missing = [col for col in required if col not in rows.columns]
    if missing:
        raise ValueError(f"missing_reliability_columns:{','.join(sorted(missing))}")

    if not lhs_columns:
        lhs_norm = pd.Series([NULL_TOKEN] * len(rows), index=rows.index, dtype="string")
    else:
        lhs_norm = normalize_lhs_frame(rows, lhs_columns)
    rhs_norm = normalize_series(rows[rhs_column])
    tmp = pd.DataFrame({"lhs_norm": lhs_norm, "rhs_norm": rhs_norm})
    tmp = tmp[(tmp["lhs_norm"] != NULL_TOKEN) & (tmp["rhs_norm"] != NULL_TOKEN)]

    total_support = len(tmp)
    if total_support == 0:
        return {
            "reliability_raw": math.nan,
            "non_singleton_coverage": math.nan,
            "reliability_non_singleton": math.nan,
            "reliability_loo": math.nan,
            "conditional_entropy_normalized": math.nan,
            "reliability_entropy": math.nan,
            "total_support": 0,
            "determinant_group_count": 0,
            "determinant_cardinality": 0,
            "dependent_cardinality": 0,
            "singleton_group_count": 0,
            "singleton_group_ratio": math.nan,
            "singleton_row_ratio": math.nan,
            "mean_group_size": math.nan,
            "median_group_size": math.nan,
            "max_group_size": 0,
            "non_singleton_row_count": 0,
            "edge_rank": edge_rank if edge_rank is not None else "",
        }

    counts = (
        tmp.groupby(["lhs_norm", "rhs_norm"], sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    group_sizes = counts.groupby("lhs_norm", sort=True)["n"].sum()
    group_modes = counts.groupby("lhs_norm", sort=True)["n"].max()
    raw = float(group_modes.sum() / total_support)

    non_singleton_mask = group_sizes >= 2
    ns_rows = int(group_sizes.loc[non_singleton_mask].sum())
    ns_coverage = float(ns_rows / total_support)
    if ns_rows:
        ns_mode_sum = float(group_modes.loc[non_singleton_mask].sum())
        ns_reliability = float(ns_mode_sum / ns_rows)
        loo = _leave_one_out_mode_predictability(counts, group_sizes)
    else:
        ns_reliability = math.nan
        loo = math.nan

    dependent_cardinality = int(tmp["rhs_norm"].nunique(dropna=True))
    if dependent_cardinality == 0:
        h_norm = math.nan
        r_entropy = math.nan
    elif dependent_cardinality == 1:
        h_norm = 0.0
        r_entropy = 1.0
    else:
        h_cond = 0.0
        for _, group in counts.groupby("lhs_norm", sort=True):
            n = group["n"].to_numpy(dtype=float)
            total = float(n.sum())
            p = n / total
            entropy = float(-(p * np.log(p)).sum())
            h_cond += (total / total_support) * entropy
        h_norm = float(h_cond / math.log(dependent_cardinality))
        r_entropy = float(1.0 - h_norm)

    singleton_groups = int((group_sizes == 1).sum())
    group_count = len(group_sizes)
    return {
        "reliability_raw": raw,
        "non_singleton_coverage": ns_coverage,
        "reliability_non_singleton": ns_reliability,
        "reliability_loo": loo,
        "conditional_entropy_normalized": h_norm,
        "reliability_entropy": r_entropy,
        "total_support": total_support,
        "determinant_group_count": group_count,
        "determinant_cardinality": group_count,
        "dependent_cardinality": dependent_cardinality,
        "singleton_group_count": singleton_groups,
        "singleton_group_ratio": float(singleton_groups / group_count) if group_count else math.nan,
        "singleton_row_ratio": float(singleton_groups / total_support),
        "mean_group_size": float(group_sizes.mean()),
        "median_group_size": float(group_sizes.median()),
        "max_group_size": int(group_sizes.max()),
        "non_singleton_row_count": ns_rows,
        "edge_rank": edge_rank if edge_rank is not None else "",
    }


def _leave_one_out_mode_predictability(counts: pd.DataFrame, group_sizes: pd.Series) -> float:
    correct = 0
    evaluated = 0
    non_singleton_lhs = set(group_sizes[group_sizes >= 2].index)
    for lhs_value, group in counts.groupby("lhs_norm", sort=True):
        if lhs_value not in non_singleton_lhs:
            continue
        values = group.sort_values("rhs_norm", kind="mergesort").reset_index(drop=True)
        rhs_values = values["rhs_norm"].astype(str).tolist()
        n_values = values["n"].astype(int).tolist()
        total = int(sum(n_values))
        evaluated += total
        for idx, rhs in enumerate(rhs_values):
            removed_counts = [
                (n - 1 if pos == idx else n, value)
                for pos, (n, value) in enumerate(zip(n_values, rhs_values))
            ]
            mode_count, mode_value = min(
                removed_counts,
                key=lambda item: (-item[0], item[1]),
            )
            if mode_count > 0 and mode_value == rhs:
                correct += int(n_values[idx])
    return float(correct / evaluated) if evaluated else math.nan


def reliability_source_frame_for_edge(
    *,
    source_table: Any,
    edge: Mapping[str, Any],
    fit_horizon: Any,
) -> pd.DataFrame:
    from fdhg.onboarding.relbench_v1 import _table_df

    df = _table_df(source_table)
    time_col = getattr(source_table, "time_col", None)
    cols = list(dict.fromkeys([*edge["lhs_columns"], str(edge["rhs_column"])]))
    if time_col and str(time_col) in df.columns:
        cols = [str(time_col), *cols]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing_reliability_columns:{','.join(sorted(missing))}")
    out = df[cols].copy()
    if time_col and str(time_col) in out.columns and fit_horizon is not None:
        times = pd.to_datetime(out[str(time_col)], errors="coerce")
        out = out.loc[times <= pd.Timestamp(fit_horizon)].copy()
    return out
