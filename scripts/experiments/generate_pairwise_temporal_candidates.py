from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd


TASK_ROOT = Path(
    "outputs/e2e/"
    "rel-ratebeer_user-place-liked_pairwise"
)

INSPECT_ROOT = (
    TASK_ROOT
    / "inspect"
    / "rel-ratebeer_user-place-liked"
)

SPLITS = ("train", "val")
DAY_NS = int(pd.Timedelta(days=1).value)


def prepare_single_key_groups(
    frame: pd.DataFrame,
    *,
    key_col: str,
    time_col: str,
    related_col: str | None,
) -> dict[Any, tuple[np.ndarray, np.ndarray | None]]:
    columns = [key_col, time_col]

    if related_col is not None:
        columns.append(related_col)

    data = frame[columns].copy()

    data[time_col] = pd.to_datetime(
        data[time_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[key_col, time_col]
    )

    data = data.sort_values(
        [key_col, time_col],
        kind="mergesort",
    )

    groups = {}

    for key, group in data.groupby(
        key_col,
        sort=False,
    ):
        times = (
            group[time_col]
            .astype("int64")
            .to_numpy()
        )

        related = None

        if related_col is not None:
            related = (
                group[related_col]
                .astype("string")
                .fillna("__NULL__")
                .to_numpy()
            )

        groups[key] = (
            times,
            related,
        )

    return groups


def prepare_pair_groups(
    frame: pd.DataFrame,
    *,
    user_col: str,
    place_col: str,
    time_col: str,
) -> dict[tuple[Any, Any], np.ndarray]:
    data = frame[
        [user_col, place_col, time_col]
    ].copy()

    data[time_col] = pd.to_datetime(
        data[time_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            user_col,
            place_col,
            time_col,
        ]
    )

    data = data.sort_values(
        [user_col, place_col, time_col],
        kind="mergesort",
    )

    groups = {}

    for key, group in data.groupby(
        [user_col, place_col],
        sort=False,
    ):
        groups[key] = (
            group[time_col]
            .astype("int64")
            .to_numpy()
        )

    return groups


def window_count(
    times: np.ndarray,
    *,
    target_ns: int,
    days: int,
) -> tuple[int, int, int]:
    right = int(
        np.searchsorted(
            times,
            target_ns,
            side="left",
        )
    )

    left = int(
        np.searchsorted(
            times,
            target_ns - days * DAY_NS,
            side="left",
        )
    )

    return right - left, left, right


def build_features(
    target: pd.DataFrame,
    *,
    user_groups: dict,
    place_groups: dict,
    pair_groups: dict,
) -> pd.DataFrame:
    n_rows = len(target)

    values = {
        "f_pairtmp__user_count_30d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__user_count_90d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__user_count_365d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__user_unique_beers_30d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__user_unique_beers_90d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__user_days_since_last":
            np.full(n_rows, np.nan, dtype=np.float32),
        "f_pairtmp__place_count_30d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__place_count_90d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__place_count_365d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__place_unique_users_30d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__place_unique_users_90d":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__place_days_since_last":
            np.full(n_rows, np.nan, dtype=np.float32),
        "f_pairtmp__prior_pair_count":
            np.zeros(n_rows, dtype=np.float32),
        "f_pairtmp__pair_days_since_last":
            np.full(n_rows, np.nan, dtype=np.float32),
    }

    timestamps = pd.to_datetime(
        target["timestamp"],
        errors="coerce",
    )

    users = target["user_id"].to_numpy()
    places = target[
        "candidate_place_id"
    ].to_numpy()

    for i, (user_id, place_id, timestamp) in enumerate(
        zip(users, places, timestamps)
    ):
        if pd.isna(timestamp):
            continue

        target_ns = int(timestamp.value)

        user_group = user_groups.get(user_id)

        if user_group is not None:
            user_times, user_related = user_group

            for days in (30, 90, 365):
                count, left, right = window_count(
                    user_times,
                    target_ns=target_ns,
                    days=days,
                )

                values[
                    f"f_pairtmp__user_count_{days}d"
                ][i] = count

                if (
                    days in {30, 90}
                    and user_related is not None
                ):
                    values[
                        f"f_pairtmp__user_unique_beers_{days}d"
                    ][i] = len(
                        np.unique(
                            user_related[left:right]
                        )
                    )

            right = int(
                np.searchsorted(
                    user_times,
                    target_ns,
                    side="left",
                )
            )

            if right > 0:
                values[
                    "f_pairtmp__user_days_since_last"
                ][i] = (
                    target_ns
                    - int(user_times[right - 1])
                ) / DAY_NS

        place_group = place_groups.get(place_id)

        if place_group is not None:
            place_times, place_related = place_group

            for days in (30, 90, 365):
                count, left, right = window_count(
                    place_times,
                    target_ns=target_ns,
                    days=days,
                )

                values[
                    f"f_pairtmp__place_count_{days}d"
                ][i] = count

                if (
                    days in {30, 90}
                    and place_related is not None
                ):
                    values[
                        f"f_pairtmp__place_unique_users_{days}d"
                    ][i] = len(
                        np.unique(
                            place_related[left:right]
                        )
                    )

            right = int(
                np.searchsorted(
                    place_times,
                    target_ns,
                    side="left",
                )
            )

            if right > 0:
                values[
                    "f_pairtmp__place_days_since_last"
                ][i] = (
                    target_ns
                    - int(place_times[right - 1])
                ) / DAY_NS

        pair_times = pair_groups.get(
            (user_id, place_id)
        )

        if pair_times is not None:
            right = int(
                np.searchsorted(
                    pair_times,
                    target_ns,
                    side="left",
                )
            )

            values[
                "f_pairtmp__prior_pair_count"
            ][i] = right

            if right > 0:
                values[
                    "f_pairtmp__pair_days_since_last"
                ][i] = (
                    target_ns
                    - int(pair_times[right - 1])
                ) / DAY_NS

    output = pd.DataFrame(values)

    for column in [
        "f_pairtmp__user_days_since_last",
        "f_pairtmp__place_days_since_last",
        "f_pairtmp__pair_days_since_last",
    ]:
        output[
            f"{column}__is_missing"
        ] = (
            output[column]
            .isna()
            .astype("int8")
        )

    output[
        "f_pairtmp__user_place_activity_product"
    ] = (
        output["f_pairtmp__user_count_90d"]
        * output["f_pairtmp__place_count_90d"]
    )

    output[
        "f_pairtmp__user_place_activity_ratio"
    ] = (
        output["f_pairtmp__user_count_90d"]
        / np.maximum(
            output["f_pairtmp__place_count_90d"],
            1.0,
        )
    )

    return output


def write_candidate(
    *,
    base_root: Path,
    output_root: Path,
    features_by_split: dict[str, pd.DataFrame],
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:
        base = pd.read_parquet(
            base_root
            / f"target_with_dfs_agg_{split}.parquet"
        ).reset_index(drop=True)

        features = features_by_split[
            split
        ].reset_index(drop=True)

        if len(base) != len(features):
            raise ValueError(
                f"Row mismatch for {split}: "
                f"{len(base)} != {len(features)}"
            )

        combined = pd.concat(
            [base, features],
            axis=1,
        )

        combined.to_parquet(
            output_root
            / f"target_with_dfs_agg_{split}.parquet",
            index=False,
        )


def main() -> None:
    beer_ratings = pd.read_parquet(
        INSPECT_ROOT
        / "table_beer_ratings.parquet"
    )

    place_ratings = pd.read_parquet(
        INSPECT_ROOT
        / "table_place_ratings.parquet"
    )

    print(
        "[LOAD]",
        "beer_ratings",
        beer_ratings.shape,
    )
    print(
        "[LOAD]",
        "place_ratings",
        place_ratings.shape,
    )

    user_groups = prepare_single_key_groups(
        beer_ratings,
        key_col="user_id",
        time_col="updated_at",
        related_col="beer_id",
    )

    place_groups = prepare_single_key_groups(
        place_ratings,
        key_col="place_id",
        time_col="created_at",
        related_col="user_id",
    )

    pair_groups = prepare_pair_groups(
        place_ratings,
        user_col="user_id",
        place_col="place_id",
        time_col="created_at",
    )

    features_by_split = {}

    for split in SPLITS:
        target = pd.read_parquet(
            TASK_ROOT
            / "dfs"
            / f"target_with_dfs_agg_{split}.parquet"
        )

        features = build_features(
            target,
            user_groups=user_groups,
            place_groups=place_groups,
            pair_groups=pair_groups,
        )

        features_by_split[split] = features

        print(
            f"[{split}]",
            "rows=",
            len(features),
            "features=",
            features.shape[1],
        )

    candidates = {
        "temporal_only":
            TASK_ROOT / "dfs",
        "structural_compact_temporal":
            TASK_ROOT / "fdhg",
    }

    margin_root = (
        TASK_ROOT
        / "candidates"
        / "structural_margin"
    )

    if all(
        (
            margin_root
            / f"target_with_dfs_agg_{split}.parquet"
        ).exists()
        for split in SPLITS
    ):
        candidates[
            "structural_margin_temporal"
        ] = margin_root

    for candidate, base_root in candidates.items():
        output_root = (
            TASK_ROOT
            / "candidates"
            / candidate
        )

        write_candidate(
            base_root=base_root,
            output_root=output_root,
            features_by_split=features_by_split,
        )

        print(
            "[WROTE]",
            output_root,
        )


if __name__ == "__main__":
    main()
