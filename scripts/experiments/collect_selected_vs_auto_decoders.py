#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PG = Path("outputs/predictor-generalization")
MATRIX_ROOT = PG / "frozen-matrices"

OUT_DIR = PG / "selected-vs-auto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [41, 42, 43, 44]

SUPPORTED_PREDICTORS = [
    "tabpfn",
    "xgboost",
    "catboost",
    "hgb",
    "tabicl",
    "realmlp",
]

DEFAULT_PREDICTORS = [
    "xgboost",
    "catboost",
    "hgb",
]

AUTO_ROOTS = {
    "tabpfn": PG / "auto-tabpfn",
    "xgboost": PG / "auto-frozen-gbdt",
    "catboost": PG / "auto-frozen-gbdt",
    "hgb": PG / "auto-hgb-frozen",
    "tabicl": PG / "auto-modern" / "tabicl",
    "realmlp": PG / "auto-modern" / "realmlp",
}

# Known preferred selected-result roots.
SELECTED_ROOTS = {
    "tabpfn": [
        PG / "tabpfn",
        PG / "tabpfn-canonical-fix",
    ],
    "xgboost": [
        PG / "frozen-gbdt",
    ],
    "catboost": [
        PG / "frozen-gbdt",
    ],
    "hgb": [
        PG / "hgb-frozen",
    ],
    "tabicl": [
        PG / "tabicl",
    ],
    "realmlp": [
        PG / "realmlp",
    ],
}

HIGHER_BETTER = {
    "roc_auc",
    "auroc",
    "average_precision",
    "ap",
    "accuracy",
    "acc",
    "macro_f1",
    "f1_macro",
}

LOWER_BETTER = {
    "rmse",
    "mae",
    "mse",
    "mean_squared_error",
}

AUTO_VARIANTS = {
    "auto_only",
    "auto",
}


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def normalize_problem_metric(metric: str) -> str:
    return str(metric).strip().lower()


def direction_delta(metric: str, selected: float, auto: float) -> float:
    """
    Positive always means Selected is better than Auto.
    """
    metric = normalize_problem_metric(metric)

    if metric in HIGHER_BETTER:
        return float(selected - auto)

    if metric in LOWER_BETTER:
        return float(auto - selected)

    raise RuntimeError(
        f"Unknown metric direction: {metric}"
    )


def selected_is_auto(variant: str) -> bool:
    return str(variant).strip().lower() in AUTO_VARIANTS


def manifest_inventory():
    rows = []

    manifests = sorted(
        MATRIX_ROOT.glob("*/manifest.json")
    )

    if len(manifests) != 51:
        raise RuntimeError(
            f"Expected 51 selected frozen manifests, found {len(manifests)}"
        )

    for mp in manifests:
        m = load_json(mp)

        if m is None:
            raise RuntimeError(f"Unreadable manifest: {mp}")

        rows.append({
            "slug": mp.parent.name,
            "dataset": str(m["dataset"]),
            "task": str(m["task"]),
            "selected_variant": str(m["selected_variant"]),
            "metric": str(m["primary_metric"]),
        })

    return pd.DataFrame(rows)


def obj_score(obj):
    if not isinstance(obj, dict):
        return None

    # GBDT artifacts store the score under the name
    # specified by primary_metric, e.g.
    # primary_metric="accuracy" -> obj["accuracy"].
    metric = (
        obj.get("primary_metric")
        or obj.get("metric")
        or obj.get("metric_name")
    )

    if metric is not None:
        value = obj.get(str(metric))

        if value is not None:
            try:
                return float(value)
            except Exception:
                pass

    # Modern / TabPFN / HGB artifacts.
    if obj.get("score") is not None:
        try:
            return float(obj["score"])
        except Exception:
            pass

    # Historical fallback field names.
    for key in [
        "official_validation_score",
        "validation_score",
        "metric_value",
        "val_score",
    ]:
        if obj.get(key) is not None:
            try:
                return float(obj[key])
            except Exception:
                pass

    return None


def obj_predictor(obj):
    if not isinstance(obj, dict):
        return ""

    return str(
        obj.get("predictor")
        or obj.get("model")
        or obj.get("decoder")
        or ""
    ).lower()


def obj_seed(obj):
    if not isinstance(obj, dict):
        return None

    try:
        return int(obj.get("seed"))
    except Exception:
        return None


def identity_matches(
    obj,
    predictor: str,
    dataset: str,
    task: str,
    seed: int,
):
    if not isinstance(obj, dict):
        return False

    if str(obj.get("dataset", "")) != dataset:
        return False

    if str(obj.get("task", "")) != task:
        return False

    p = obj_predictor(obj)

    if predictor not in p:
        return False

    s = obj_seed(obj)

    if s != seed:
        return False

    if obj_score(obj) is None:
        return False

    return True


def json_candidates_under(root: Path):
    if not root.exists():
        return []

    return list(root.rglob("*.json"))


def selected_candidates(
    predictor: str,
    dataset: str,
    task: str,
    seed: int,
):
    """
    Search preferred selected roots first.
    Never search Auto counterfactual roots.
    """
    matches = []

    seen = set()

    for root in SELECTED_ROOTS[predictor]:
        for p in json_candidates_under(root):
            if p in seen:
                continue

            seen.add(p)

            path_text = str(p)

            if "/auto-" in path_text:
                continue

            obj = load_json(p)

            if identity_matches(
                obj,
                predictor,
                dataset,
                task,
                seed,
            ):
                matches.append((p, obj))

    return matches


def choose_selected_result(
    predictor: str,
    dataset: str,
    task: str,
    seed: int,
):
    matches = selected_candidates(
        predictor,
        dataset,
        task,
        seed,
    )

    if not matches:
        return None

    # Prefer the canonical roots in their declared order.
    for root in SELECTED_ROOTS[predictor]:
        root_s = str(root)

        subset = [
            x for x in matches
            if str(x[0]).startswith(root_s + "/")
        ]

        if subset:
            # If duplicate valid artifacts exist within one preferred root,
            # require numerical agreement.
            vals = [
                obj_score(obj)
                for _, obj in subset
            ]

            rounded = {
                round(float(v), 12)
                for v in vals
            }

            if len(rounded) > 1:
                print(
                    "WARNING SELECTED DUPLICATE SCORE CONFLICT:",
                    predictor,
                    dataset,
                    task,
                    seed,
                )

                for p, obj in subset:
                    print(
                        " ",
                        p,
                        obj_score(obj),
                    )

            return subset[0]

    return matches[0]


def auto_result_path_candidates(
    predictor: str,
    slug: str,
    dataset: str,
    task: str,
    seed: int,
):
    root = AUTO_ROOTS[predictor]

    if predictor == "tabpfn":
        return [
            root / slug / f"seed-{seed}.json",
        ]

    if predictor in {"tabicl", "realmlp"}:
        return [
            root / slug / f"seed-{seed}.json",
        ]

    if predictor in {"xgboost", "catboost"}:
        return [
            root
            / slug
            / predictor
            / f"seed_{seed}"
            / "metrics.json"
        ]

    if predictor == "hgb":
        return [
            root
            / dataset
            / task
            / "seed_0.json"
        ]

    raise RuntimeError(predictor)


def load_auto_result(
    predictor: str,
    slug: str,
    dataset: str,
    task: str,
    seed: int,
):
    for p in auto_result_path_candidates(
        predictor,
        slug,
        dataset,
        task,
        seed,
    ):
        if not p.exists():
            continue

        obj = load_json(p)

        if obj is None:
            continue

        score = obj_score(obj)

        if score is None:
            continue

        return p, obj

    return None


def mean_selected_score(
    predictor: str,
    dataset: str,
    task: str,
):
    seeds = [0] if predictor == "hgb" else SEEDS

    scores = []
    paths = []

    for seed in seeds:
        hit = choose_selected_result(
            predictor,
            dataset,
            task,
            seed,
        )

        if hit is None:
            continue

        p, obj = hit

        scores.append(
            obj_score(obj)
        )
        paths.append(str(p))

    if len(scores) != len(seeds):
        return None, scores, paths

    return float(np.mean(scores)), scores, paths


def mean_auto_score(
    predictor: str,
    slug: str,
    dataset: str,
    task: str,
):
    seeds = [0] if predictor == "hgb" else SEEDS

    scores = []
    paths = []

    for seed in seeds:
        hit = load_auto_result(
            predictor,
            slug,
            dataset,
            task,
            seed,
        )

        if hit is None:
            continue

        p, obj = hit

        scores.append(
            obj_score(obj)
        )
        paths.append(str(p))

    if len(scores) != len(seeds):
        return None, scores, paths

    return float(np.mean(scores)), scores, paths


ap = argparse.ArgumentParser()

ap.add_argument(
    "--predictors",
    nargs="+",
    choices=SUPPORTED_PREDICTORS,
    default=DEFAULT_PREDICTORS,
)

ap.add_argument(
    "--selected-gbdt-root",
    type=Path,
    default=None,
    help=(
        "Optional override for selected XGBoost/CatBoost "
        "result root. By default, uses "
        "outputs/predictor-generalization/frozen-gbdt."
    ),
)

args = ap.parse_args()
predictors = args.predictors

if args.selected_gbdt_root is not None:
    SELECTED_ROOTS["xgboost"] = [
        args.selected_gbdt_root
    ]
    SELECTED_ROOTS["catboost"] = [
        args.selected_gbdt_root
    ]

inventory = manifest_inventory()

print("=" * 100)
print("SELECTED VS AUTO DECODER COMPARISON")
print("=" * 100)

print()
print("SELECTION INVENTORY")
print(
    inventory["selected_variant"]
    .value_counts()
    .to_string()
)

all_task_rows = []
summary_rows = []

for predictor in predictors:
    print()
    print("=" * 100)
    print("PREDICTOR:", predictor)
    print("=" * 100)

    rows = []

    for r in inventory.itertuples(index=False):
        sel_mean, sel_seed_scores, sel_paths = (
            mean_selected_score(
                predictor,
                r.dataset,
                r.task,
            )
        )

        # Coverage requires an actually available selected-side
        # downstream decoder result.
        if sel_mean is None:
            rows.append({
                "predictor": predictor,
                "slug": r.slug,
                "dataset": r.dataset,
                "task": r.task,
                "metric": r.metric,
                "selected_variant": r.selected_variant,
                "selected_score": np.nan,
                "auto_score": np.nan,
                "delta": np.nan,
                "outcome": "MISSING_SELECTED",
                "structural_tie": False,
                "selected_seed_count": len(sel_seed_scores),
                "auto_seed_count": 0,
                "selected_paths": " | ".join(sel_paths),
                "auto_paths": "",
            })
            continue

        if selected_is_auto(r.selected_variant):
            auto_mean = sel_mean
            auto_seed_scores = list(sel_seed_scores)
            auto_paths = list(sel_paths)

            delta = 0.0
            outcome = "T"
            structural_tie = True

        else:
            auto_mean, auto_seed_scores, auto_paths = (
                mean_auto_score(
                    predictor,
                    r.slug,
                    r.dataset,
                    r.task,
                )
            )

            if auto_mean is None:
                rows.append({
                    "predictor": predictor,
                    "slug": r.slug,
                    "dataset": r.dataset,
                    "task": r.task,
                    "metric": r.metric,
                    "selected_variant": r.selected_variant,
                    "selected_score": sel_mean,
                    "auto_score": np.nan,
                    "delta": np.nan,
                    "outcome": "MISSING_AUTO",
                    "structural_tie": False,
                    "selected_seed_count": len(sel_seed_scores),
                    "auto_seed_count": len(auto_seed_scores),
                    "selected_paths": " | ".join(sel_paths),
                    "auto_paths": " | ".join(auto_paths),
                })
                continue

            delta = direction_delta(
                r.metric,
                sel_mean,
                auto_mean,
            )

            # Reporting uses exact numerical sign.
            # No strategy-gate epsilon is reused here.
            if delta > 0:
                outcome = "W"
            elif delta < 0:
                outcome = "L"
            else:
                outcome = "T"

            structural_tie = False

        rows.append({
            "predictor": predictor,
            "slug": r.slug,
            "dataset": r.dataset,
            "task": r.task,
            "metric": r.metric,
            "selected_variant": r.selected_variant,
            "selected_score": sel_mean,
            "auto_score": auto_mean,
            "delta": delta,
            "outcome": outcome,
            "structural_tie": structural_tie,
            "selected_seed_count": len(sel_seed_scores),
            "auto_seed_count": len(auto_seed_scores),
            "selected_paths": " | ".join(sel_paths),
            "auto_paths": " | ".join(auto_paths),
        })

    df = pd.DataFrame(rows)

    valid = df[df["outcome"].isin(["W", "T", "L"])].copy()

    w = int((valid["outcome"] == "W").sum())
    t = int((valid["outcome"] == "T").sum())
    l = int((valid["outcome"] == "L").sum())

    coverage = len(valid)

    mean_delta = (
        float(valid["delta"].mean())
        if coverage
        else np.nan
    )

    median_delta = (
        float(valid["delta"].median())
        if coverage
        else np.nan
    )

    summary_rows.append({
        "predictor": predictor,
        "coverage": coverage,
        "total_tasks": 51,
        "wins": w,
        "ties": t,
        "losses": l,
        "wtl": f"{w}/{t}/{l}",
        "mean_direction_normalized_delta": mean_delta,
        "median_direction_normalized_delta": median_delta,
        "missing_selected": int(
            (df["outcome"] == "MISSING_SELECTED").sum()
        ),
        "missing_auto": int(
            (df["outcome"] == "MISSING_AUTO").sum()
        ),
    })

    print(
        f"COVERAGE: {coverage}/51"
    )
    print(
        f"W/T/L: {w}/{t}/{l}"
    )
    print(
        "MEAN DELTA:",
        mean_delta,
    )
    print(
        "MEDIAN DELTA:",
        median_delta,
    )

    missing = df[
        ~df["outcome"].isin(["W", "T", "L"])
    ][
        [
            "dataset",
            "task",
            "selected_variant",
            "outcome",
            "selected_seed_count",
            "auto_seed_count",
        ]
    ]

    if len(missing):
        print()
        print("MISSING:")
        print(
            missing.to_string(index=False)
        )

    all_task_rows.append(df)


detail = pd.concat(
    all_task_rows,
    ignore_index=True,
)

summary = pd.DataFrame(
    summary_rows
)

detail_path = (
    OUT_DIR
    / "selected_vs_auto_task_level.csv"
)

summary_path = (
    OUT_DIR
    / "selected_vs_auto_summary.csv"
)

nonauto_path = (
    OUT_DIR
    / "selected_vs_auto_nonauto18.csv"
)

detail.to_csv(
    detail_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

detail[
    ~detail["selected_variant"].isin(
        AUTO_VARIANTS
    )
].to_csv(
    nonauto_path,
    index=False,
)

print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

display_cols = [
    "predictor",
    "coverage",
    "wtl",
    "mean_direction_normalized_delta",
    "median_direction_normalized_delta",
    "missing_selected",
    "missing_auto",
]

print(
    summary[display_cols]
    .to_string(index=False)
)

print()
print("SAVED:")
print(" ", summary_path)
print(" ", detail_path)
print(" ", nonauto_path)
