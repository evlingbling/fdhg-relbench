from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tabpfn import TabPFNRegressor


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    label_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    y_train = pd.to_numeric(
        train_df[label_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    y_val = pd.to_numeric(
        val_df[label_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    excluded = set(drop_cols) | {label_col}

    x_train = train_df.drop(
        columns=[
            c for c in excluded
            if c in train_df.columns
        ],
        errors="ignore",
    ).copy()

    x_val = val_df.drop(
        columns=[
            c for c in excluded
            if c in val_df.columns
        ],
        errors="ignore",
    ).copy()

    # Align validation columns exactly to train.
    x_val = x_val.reindex(
        columns=x_train.columns,
    )

    for col in x_train.columns:
        train_col = x_train[col]
        val_col = x_val[col]

        if pd.api.types.is_datetime64_any_dtype(train_col):
            x_train[col] = (
                pd.to_datetime(
                    train_col,
                    errors="coerce",
                )
                .astype("int64")
                .replace(
                    np.iinfo("int64").min,
                    np.nan,
                )
            )
            x_val[col] = (
                pd.to_datetime(
                    val_col,
                    errors="coerce",
                )
                .astype("int64")
                .replace(
                    np.iinfo("int64").min,
                    np.nan,
                )
            )

        elif (
            pd.api.types.is_object_dtype(train_col)
            or isinstance(
                train_col.dtype,
                pd.CategoricalDtype,
            )
            or pd.api.types.is_string_dtype(train_col)
        ):
            combined = pd.concat(
                [
                    train_col.astype("string"),
                    val_col.astype("string"),
                ],
                ignore_index=True,
            )

            codes, _ = pd.factorize(
                combined,
                sort=True,
            )

            x_train[col] = codes[:len(x_train)]
            x_val[col] = codes[len(x_train):]

        elif pd.api.types.is_bool_dtype(train_col):
            x_train[col] = train_col.astype("int8")
            x_val[col] = val_col.astype("int8")

        else:
            x_train[col] = pd.to_numeric(
                train_col,
                errors="coerce",
            )
            x_val[col] = pd.to_numeric(
                val_col,
                errors="coerce",
            )

    # Remove columns constant in the training split.
    nonconstant_cols = [
        col
        for col in x_train.columns
        if x_train[col].nunique(dropna=False) > 1
    ]

    x_train = x_train[nonconstant_cols]
    x_val = x_val[nonconstant_cols]

    # Match the existing parquet evaluator convention.
    x_train = x_train.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    x_val = x_val.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    valid_train = np.isfinite(y_train)
    valid_val = np.isfinite(y_val)

    x_train = x_train.loc[valid_train].reset_index(drop=True)
    y_train = y_train[valid_train]

    x_val = x_val.loc[valid_val].reset_index(drop=True)
    y_val = y_val[valid_val]

    return x_train, x_val, y_train, y_val


def sample_rows(
    df: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)

    return (
        df.sample(
            n=max_rows,
            random_state=seed,
        )
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-parquet",
        required=True,
    )
    parser.add_argument(
        "--val-parquet",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--task",
        required=True,
    )
    parser.add_argument(
        "--variant",
        required=True,
    )
    parser.add_argument(
        "--label-col",
        required=True,
    )
    parser.add_argument(
        "--drop-cols",
        default="",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = pd.read_parquet(
        args.train_parquet
    )
    val_df = pd.read_parquet(
        args.val_parquet
    )

    train_df = sample_rows(
        train_df,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )

    val_df = sample_rows(
        val_df,
        max_rows=args.max_val_rows,
        seed=args.seed,
    )

    drop_cols = [
        c.strip()
        for c in args.drop_cols.split(",")
        if c.strip()
    ]

    x_train, x_val, y_train, y_val = (
        prepare_features(
            train_df,
            val_df,
            label_col=args.label_col,
            drop_cols=drop_cols,
        )
    )

    print("dataset/task:", args.dataset, args.task)
    print("variant:", args.variant)
    print("seed:", args.seed)
    print("X_train:", x_train.shape)
    print("X_val:", x_val.shape)
    print("features:", list(x_train.columns))

    model = TabPFNRegressor(
        device=args.device,
        random_state=args.seed,
    )

    model.fit(
        x_train.to_numpy(dtype=np.float32),
        y_train,
    )

    pred = model.predict(
        x_val.to_numpy(dtype=np.float32)
    )

    rmse = float(
        mean_squared_error(
            y_val,
            pred,
        ) ** 0.5
    )

    mae = float(
        mean_absolute_error(
            y_val,
            pred,
        )
    )

    r2 = float(
        r2_score(
            y_val,
            pred,
        )
    )

    metrics = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "task": args.task,
                "variant": args.variant,
                "seed": args.seed,
                "n_train_used": len(y_train),
                "n_val_used": len(y_val),
                "n_features": x_train.shape[1],
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "train_parquet": args.train_parquet,
                "val_parquet": args.val_parquet,
            }
        ]
    )

    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(
        metrics_path,
        index=False,
    )

    pred_path = output_dir / "predictions.parquet"
    pd.DataFrame(
        {
            "y_true": y_val,
            "y_pred": pred,
        }
    ).to_parquet(
        pred_path,
        index=False,
    )

    print("\n=== METRICS ===")
    print(metrics.to_string(index=False))
    print("\nsaved:", metrics_path)
    print("saved:", pred_path)


if __name__ == "__main__":
    main()
