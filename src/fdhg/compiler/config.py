from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .ir import (
    PairwiseHistorySpec,
    PairwiseSpec,
    TaskSpec,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")

    return data


def load_pairwise_history(
    value: Any,
) -> PairwiseHistorySpec | None:
    if value is None:
        return None

    if not isinstance(value, dict):
        raise ValueError(
            "Pairwise history configuration must be a mapping"
        )

    if "table" not in value:
        raise ValueError(
            "Pairwise history configuration requires 'table'"
        )

    return PairwiseHistorySpec(
        table=str(value["table"]),
        key=(
            str(value["key"])
            if value.get("key") is not None
            else None
        ),
        left_key=(
            str(value["left_key"])
            if value.get("left_key") is not None
            else None
        ),
        right_key=(
            str(value["right_key"])
            if value.get("right_key") is not None
            else None
        ),
        related_col=(
            str(value["related_col"])
            if value.get("related_col") is not None
            else None
        ),
        time_col=(
            str(value["time_col"])
            if value.get("time_col") is not None
            else None
        ),
    )


def load_pairwise_spec(
    value: Any,
) -> PairwiseSpec | None:
    if value is None:
        return None

    if not isinstance(value, dict):
        raise ValueError(
            "Pairwise task semantics must be a mapping"
        )

    required = [
        "left_key",
        "right_key",
        "target_right_key",
    ]

    missing = [
        key
        for key in required
        if value.get(key) is None
    ]

    if missing:
        raise ValueError(
            "Pairwise task semantics are missing: "
            + ", ".join(missing)
        )

    return PairwiseSpec(
        left_key=str(value["left_key"]),
        right_key=str(value["right_key"]),
        target_right_key=str(
            value["target_right_key"]
        ),
        left_history=load_pairwise_history(
            value.get("left_history")
        ),
        right_history=load_pairwise_history(
            value.get("right_history")
        ),
        pair_history=load_pairwise_history(
            value.get("pair_history")
        ),
    )


def load_task_spec(
    *,
    dataset: str,
    task: str,
    reproduction_config: Path,
    semantics_config: Path | None = None,
) -> TaskSpec:
    key = f"{dataset}/{task}"
    config = load_yaml(reproduction_config)

    if "tasks" in config:
        tasks = config["tasks"]
    else:
        tasks = config

    if not isinstance(tasks, dict):
        raise ValueError(
            f"Expected task mapping in {reproduction_config}, "
            f"found {type(tasks).__name__}"
        )

    if key not in tasks:
        available = sorted(str(k) for k in tasks.keys())
        preview = ", ".join(available[:10])
        raise KeyError(
            f"Task {key!r} is missing from {reproduction_config}. "
            f"First available tasks: {preview}"
        )

    raw = tasks[key]
    target = raw.get("target", {})
    dfs = raw.get("dfs", {})

    semantics: dict[str, Any] = {}

    if semantics_config is not None and semantics_config.exists():
        semantics = load_yaml(semantics_config).get(key, {})

    return TaskSpec(
        dataset=dataset,
        task=task,
        problem_type=str(raw["problem_type"]),
        label_col=str(raw["label_col"]),
        entity_key=str(target["entity_key"]),
        target_time_col=str(target["time_col"]),
        child_table=dfs.get("child_table"),
        child_time_col=dfs.get("child_time_col"),
        numeric_col=dfs.get("numeric_col"),
        baseline_operations=(
            None
            if dfs.get("baseline_operations") is None
            else tuple(str(item) for item in dfs["baseline_operations"])
        ),
        horizon_days=semantics.get("horizon_days"),
        feature_budget=int(
            semantics.get("feature_budget", 32)
        ),
        primary_metric=semantics.get(
            "primary_metric",
            raw.get("primary_metric"),
        ),
        secondary_metric=semantics.get(
            "secondary_metric",
            raw.get("secondary_metric"),
        ),
        metric_direction=semantics.get(
            "metric_direction",
            raw.get("metric_direction"),
        ),
        seeds=tuple(
            int(seed)
            for seed in semantics.get(
                "seeds",
                [41, 42, 43, 44],
            )
        ),
        pairwise=load_pairwise_spec(
            semantics.get("pairwise")
        ),
    )
