from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot dependency reliability versus single-edge validation utility."
    )
    parser.add_argument("aggregate_csv", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="reliability_utility")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(path) for path in args.aggregate_csv], ignore_index=True)
    df["mean_reliability_loo"] = pd.to_numeric(df["mean_reliability_loo"], errors="coerce")
    df["mean_delta"] = pd.to_numeric(df["mean_delta"], errors="coerce")
    df["mean_coverage"] = pd.to_numeric(df["mean_coverage"], errors="coerce")
    structural = _structural_edge_frame(df)
    valid = structural.dropna(subset=["mean_reliability_loo", "mean_delta"]).copy()

    rho, p_value = _spearman(valid["mean_reliability_loo"], valid["mean_delta"])
    _write_scatter(
        valid,
        output_dir / f"{args.prefix}_scatter.png",
        rho=rho,
        p_value=p_value,
    )
    quartile = _quartile_summary(valid)
    quartile_path = output_dir / f"{args.prefix}_quartiles.csv"
    quartile.to_csv(quartile_path, index=False)
    _write_quartile_plot(quartile, output_dir / f"{args.prefix}_quartiles.png")
    summary = {
        "number_of_tasks": int(valid[["dataset", "task"]].drop_duplicates().shape[0]) if not valid.empty else 0,
        "number_of_candidate_edges": _candidate_observation_count(df),
        "number_of_unique_structural_edges": int(structural[["dataset", "task", "edge_id"]].drop_duplicates().shape[0]) if not structural.empty else 0,
        "number_of_valid_edge_observations": len(valid),
        "pooled_spearman_rho": rho,
        "pooled_spearman_p_value": p_value,
        "task_wise_spearman_correlations": _taskwise_spearman(valid),
        "mean_task_wise_correlation": math.nan,
        "median_task_wise_correlation": math.nan,
        "positive_edge_fraction_by_reliability_quartile": {},
    }
    task_rhos = [
        item["spearman_rho"]
        for item in summary["task_wise_spearman_correlations"]
        if np.isfinite(item["spearman_rho"])
    ]
    if task_rhos:
        summary["mean_task_wise_correlation"] = float(np.mean(task_rhos))
        summary["median_task_wise_correlation"] = float(np.median(task_rhos))
    if not quartile.empty:
        summary["positive_edge_fraction_by_reliability_quartile"] = {
            str(row["reliability_quartile"]): float(row["positive_edge_fraction"])
            for _, row in quartile.iterrows()
        }
    (output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(_json_clean(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("SCATTER", output_dir / f"{args.prefix}_scatter.png")
    print("QUARTILE_PLOT", output_dir / f"{args.prefix}_quartiles.png")
    print("QUARTILE_CSV", quartile_path)
    print("SUMMARY_JSON", output_dir / f"{args.prefix}_summary.json")
    return 0


def _structural_edge_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "seed" not in df.columns:
        return df.copy()
    aggregations = {
        "mean_reliability_loo": ("mean_reliability_loo", "mean"),
        "mean_delta": ("mean_delta", "mean"),
        "mean_coverage": ("mean_coverage", "mean"),
    }
    for optional in ("edge_rank", "determinant", "dependent", "source_table", "relational_path"):
        if optional in df.columns:
            aggregations[optional] = (optional, "first")
    return (
        df.groupby(["dataset", "task", "edge_id"], sort=True)
        .agg(**aggregations)
        .reset_index()
    )


def _candidate_observation_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    keys = ["dataset", "task", "edge_id"]
    if "seed" in df.columns:
        keys.insert(2, "seed")
    return int(df[keys].drop_duplicates().shape[0])


def _write_scatter(df: pd.DataFrame, path: Path, *, rho: float, p_value: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if not df.empty:
        sizes = 30.0 + 220.0 * df["mean_coverage"].fillna(0.0).clip(0, 1)
        ax.scatter(df["mean_reliability_loo"], df["mean_delta"], s=sizes, alpha=0.7, edgecolor="black", linewidth=0.4)
        if df["mean_reliability_loo"].nunique(dropna=True) >= 2:
            coef = np.polyfit(df["mean_reliability_loo"], df["mean_delta"], deg=1)
            xs = np.linspace(float(df["mean_reliability_loo"].min()), float(df["mean_reliability_loo"].max()), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#b23a48", linewidth=2)
    ax.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("Mean leave-one-out reliability")
    ax.set_ylabel("Mean validation delta")
    ax.set_title(f"Reliability vs utility (Spearman rho={rho:.3g}, p={p_value:.3g}, n={len(df)})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _quartile_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(columns=[
            "dataset",
            "task",
            "reliability_quartile",
            "edge_count",
            "mean_delta",
            "median_delta",
            "positive_edge_fraction",
        ])
    for (dataset, task), group in df.groupby(["dataset", "task"], sort=True):
        group = group.copy()
        if group["mean_reliability_loo"].nunique(dropna=True) < 4 or len(group) < 4:
            group["reliability_quartile"] = "all"
        else:
            group["reliability_quartile"] = pd.qcut(
                group["mean_reliability_loo"],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            ).astype(str)
        for quartile, sub in group.groupby("reliability_quartile", sort=True):
            rows.append({
                "dataset": dataset,
                "task": task,
                "reliability_quartile": quartile,
                "edge_count": len(sub),
                "mean_delta": float(sub["mean_delta"].mean()),
                "median_delta": float(sub["mean_delta"].median()),
                "positive_edge_fraction": float((sub["mean_delta"] > 0.0).mean()),
            })
    return pd.DataFrame(rows)


def _write_quartile_plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if not summary.empty:
        pooled = (
            summary.groupby("reliability_quartile", sort=True)
            .agg(mean_delta=("mean_delta", "mean"), positive_edge_fraction=("positive_edge_fraction", "mean"))
            .reset_index()
        )
        ax.bar(pooled["reliability_quartile"], pooled["mean_delta"], color="#4c78a8")
        ax2 = ax.twinx()
        ax2.plot(pooled["reliability_quartile"], pooled["positive_edge_fraction"], color="#f58518", marker="o")
        ax2.set_ylabel("Positive-edge fraction")
    ax.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlabel("Within-task reliability quartile")
    ax.set_ylabel("Mean validation delta")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _taskwise_spearman(df: pd.DataFrame) -> list[dict[str, float | str | int]]:
    rows = []
    for (dataset, task), group in df.groupby(["dataset", "task"], sort=True):
        rho, p_value = _spearman(group["mean_reliability_loo"], group["mean_delta"])
        rows.append({
            "dataset": dataset,
            "task": task,
            "n": len(group.dropna(subset=["mean_reliability_loo", "mean_delta"])),
            "spearman_rho": rho,
            "spearman_p_value": p_value,
        })
    return rows


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 2 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return math.nan, math.nan
    try:
        from scipy.stats import spearmanr

        result = spearmanr(tmp["x"], tmp["y"])
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        return float(tmp["x"].rank().corr(tmp["y"].rank())), math.nan


def _json_clean(value):
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
