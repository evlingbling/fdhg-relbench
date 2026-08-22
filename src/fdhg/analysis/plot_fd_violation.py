from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_fd_violation(figure_aggregate_csv: Path, output_dir: Path) -> list[Path]:
    df = pd.read_csv(figure_aggregate_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _figure_a(df, output_dir / "figure_a_reliability_by_corruption.png"),
        _figure_b(df, output_dir / "figure_b_delta_by_reliability.png"),
        _figure_c(df, output_dir / "figure_c_residual_by_corruption.png"),
    ]
    return paths


def _figure_a(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for edge_id, group in df.groupby("edge_id", sort=True):
        group = group.sort_values("requested_corruption_rate")
        ax.errorbar(
            group["mean_effective_changed_row_rate"],
            group["mean_reliability_loo"],
            yerr=group.get("sem_reliability_loo"),
            marker="o",
            linewidth=1.5,
            label=edge_id,
        )
    ax.set_xlabel("Effective changed-row rate")
    ax.set_ylabel("Measured reliability (leave-one-out)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _figure_b(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for edge_id, group in df.groupby("edge_id", sort=True):
        group = group.sort_values("mean_reliability_loo", ascending=False)
        ax.errorbar(
            group["mean_reliability_loo"],
            group["mean_delta_relative_to_uncorrupted"],
            yerr=group.get("sem_delta_relative_to_uncorrupted"),
            marker="o",
            linewidth=1.5,
            label=edge_id,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Measured reliability (leave-one-out)")
    ax.set_ylabel("Delta relative to uncorrupted edge")
    ax.set_xlim(1.05, -0.05)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _figure_c(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for edge_id, group in df.groupby("edge_id", sort=True):
        group = group.sort_values("requested_corruption_rate")
        ax.errorbar(
            group["mean_effective_changed_row_rate"],
            group["mean_residual_variance"],
            yerr=group.get("sem_residual_variance"),
            marker="o",
            linewidth=1.5,
            label=edge_id,
        )
    ax.set_xlabel("Effective changed-row rate")
    ax.set_ylabel("Residual feature variance")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot FD violation motivation figures.")
    parser.add_argument("figure_aggregate_csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    for path in plot_fd_violation(Path(args.figure_aggregate_csv), Path(args.output_dir)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
