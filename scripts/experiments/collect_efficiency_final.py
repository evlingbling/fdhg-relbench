from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/efficiency-final")

TASKS = [
    ("rel-event", "user-attendance"),
    ("rel-f1", "driver-position"),
    ("rel-trial", "studies-enrollment"),
    ("rel-trial", "study-outcome"),
]

METHODS = [
    ("dfs", "Canonical DFS"),
    ("auto", "Auto"),
    ("all", "Auto + All FDHG"),
    ("independent", "Auto + Independent"),
    ("greedy", "Auto + Greedy"),
]


def elapsed_to_seconds(value: str) -> float:
    value = value.strip()
    parts = value.split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (
            float(hours) * 3600
            + float(minutes) * 60
            + float(seconds)
        )

    raise ValueError(value)


def parse_time(path: Path) -> dict:
    if not path.exists():
        return {}

    text = path.read_text(errors="replace")

    patterns = {
        "user_seconds": r"User time \(seconds\):\s*([0-9.]+)",
        "system_seconds": r"System time \(seconds\):\s*([0-9.]+)",
        "elapsed": (
            r"Elapsed \(wall clock\) time "
            r"\(h:mm:ss or m:ss\):\s*(\S+)"
        ),
        "max_rss_kb": (
            r"Maximum resident set size \(kbytes\):\s*(\d+)"
        ),
    }

    out = {}

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1)

    if "elapsed" in out:
        out["wall_seconds"] = elapsed_to_seconds(out["elapsed"])

    if "max_rss_kb" in out:
        out["peak_memory_gib"] = (
            float(out["max_rss_kb"]) / 1024.0 / 1024.0
        )

    return out


rows = []

for method, label in METHODS:
    for dataset, task in TASKS:
        slug = f"{dataset}_{task}"

        manifest = (
            ROOT
            / "raw"
            / method
            / slug
            / "manifest.json"
        )

        timefile = (
            ROOT
            / "time"
            / f"{method}__{slug}.time.txt"
        )

        row = {
            "method": method,
            "method_label": label,
            "dataset": dataset,
            "task": task,
            "manifest_path": str(manifest),
            "time_path": str(timefile),
        }

        row.update(parse_time(timefile))

        if not manifest.exists():
            row["status"] = "missing_manifest"
            rows.append(row)
            continue

        d = json.loads(manifest.read_text())

        metrics = d.get("official_validation_metrics", {}) or {}

        # Some manifests nest the actual metric payload.
        if (
            "n_features" not in metrics
            and isinstance(metrics.get("metrics"), dict)
        ):
            metrics = metrics["metrics"]

        if method in {"dfs", "auto"}:
            selected_edges = 0
        else:
            selected_edges = int(
                d.get(
                    "strategy_selected_edge_count",
                    d.get(
                        "selected_screened_edge_count",
                        d.get("screened_in_fdhg_edge_count", 0),
                    ),
                )
                or 0
            )

        feature_count = metrics.get("n_features")

        # Fallbacks only if n_features is absent.
        if feature_count is None:
            if method == "dfs":
                feature_count = d.get("dfs_model_column_count")
            elif method == "auto":
                feature_count = d.get("auto_feature_count")
            else:
                auto_n = int(d.get("auto_feature_count", 0) or 0)
                residual_n = int(
                    d.get("fdhg_final_refit_usable_features", 0)
                    or 0
                )
                feature_count = auto_n + residual_n

        row.update({
            "status": d.get("status", "completed"),
            "feature_count": feature_count,
            "selected_edges": selected_edges,
            "candidate_edges": d.get("candidate_fdhg_edge_count", 0),
            "gate_selected_variant": d.get(
                "gate_selected_variant",
                d.get("selected_variant"),
            ),
            "final_evaluated_variant": d.get(
                "final_evaluated_variant",
                d.get("selected_variant"),
            ),
            "forced_final_evaluation": d.get(
                "forced_final_evaluation",
                False,
            ),
            "official_validation_score": d.get(
                "official_validation_score"
            ),
            "test_split_accessed": d.get(
                "test_split_accessed"
            ),
            "official_validation_used_for_selection": d.get(
                "official_validation_was_used_for_selection"
            ),
        })

        rows.append(row)


df = pd.DataFrame(rows)

per_task_path = ROOT / "efficiency_per_task.csv"
df.to_csv(per_task_path, index=False)


def mean_std(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return math.nan, math.nan

    return float(values.mean()), float(values.std(ddof=1))


summary_rows = []

for method, label in METHODS:
    g = df[df["method"] == method].copy()

    fc_m, fc_s = mean_std(g["feature_count"])
    edge_m, edge_s = mean_std(g["selected_edges"])
    wall_m, wall_s = mean_std(g["wall_seconds"])
    mem_m, mem_s = mean_std(g["peak_memory_gib"])

    summary_rows.append({
        "method": method,
        "method_label": label,
        "n_tasks": int(len(g)),
        "n_completed": int(
            g["manifest_path"].map(lambda p: Path(p).exists()).sum()
        ),
        "feature_count_mean": fc_m,
        "feature_count_std": fc_s,
        "selected_edges_mean": edge_m,
        "selected_edges_std": edge_s,
        "wall_seconds_mean": wall_m,
        "wall_seconds_std": wall_s,
        "peak_memory_gib_mean": mem_m,
        "peak_memory_gib_std": mem_s,
    })


summary = pd.DataFrame(summary_rows)
summary_path = ROOT / "efficiency_summary.csv"
summary.to_csv(summary_path, index=False)


def pm(mean, std, digits=1):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


lines = [
    r"\begin{table*}[ht]",
    r"\centering",
    r"\caption{",
    r"Feature complexity and end-to-end pipeline cost averaged over four representative tasks.",
    r"All runs use the same current implementation, hardware configuration, frozen Auto declarations,",
    r"canonical DFS provenance, and frozen FDHG candidate pools.",
    r"FDHG variants share the same candidate pool; All FDHG retains every candidate.",
    r"}",
    r"\label{tab:efficiency}",
    r"\small",
    r"\begin{tabular}{lcccc}",
    r"\hline",
    r"Method & Feature Count & Selected Edges & End-to-End Time (s) & Peak Memory (GiB) \\",
    r"\hline",
]

for _, r in summary.iterrows():
    lines.append(
        f"{r['method_label']} & "
        f"{pm(r['feature_count_mean'], r['feature_count_std'], 1)} & "
        f"{pm(r['selected_edges_mean'], r['selected_edges_std'], 1)} & "
        f"{pm(r['wall_seconds_mean'], r['wall_seconds_std'], 1)} & "
        f"{pm(r['peak_memory_gib_mean'], r['peak_memory_gib_std'], 2)} "
        r"\\"
    )

lines += [
    r"\hline",
    r"\end{tabular}",
    r"\end{table*}",
]

tex_path = ROOT / "efficiency_table.tex"
tex_path.write_text("\n".join(lines) + "\n")


print("\n===== PER-TASK =====")
print(
    df[
        [
            "method_label",
            "dataset",
            "task",
            "feature_count",
            "selected_edges",
            "wall_seconds",
            "peak_memory_gib",
            "test_split_accessed",
        ]
    ].to_string(index=False)
)

print("\n===== SUMMARY =====")
print(summary.to_string(index=False))

print("\nWROTE", per_task_path)
print("WROTE", summary_path)
print("WROTE", tex_path)

unsafe = df[
    df["test_split_accessed"].fillna(False).astype(bool)
]

if not unsafe.empty:
    raise SystemExit(
        "ERROR: test split was accessed in at least one run"
    )

validation_used = df[
    df["official_validation_used_for_selection"]
    .fillna(False)
    .astype(bool)
]

if not validation_used.empty:
    raise SystemExit(
        "ERROR: official validation was used for selection "
        "in at least one run"
    )

missing = df[df["status"] == "missing_manifest"]

if not missing.empty:
    raise SystemExit(
        f"ERROR: {len(missing)} efficiency runs are missing"
    )
