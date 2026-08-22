import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder


def canonical_label_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        values = numeric.to_numpy(dtype=float)

        if np.all(np.isclose(values, np.round(values))):
            return pd.Series(
                np.round(values).astype(np.int64),
                index=series.index,
            ).astype(str)

        return numeric.astype(str)

    return series.astype(str)


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    for column in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(X_train[column]):
            X_train[column] = X_train[column].astype("int64") / 1e9
            X_val[column] = pd.to_datetime(
                X_val[column], errors="coerce"
            ).astype("int64") / 1e9
        else:
            X_train[column] = pd.to_numeric(
                X_train[column], errors="coerce"
            )
            X_val[column] = pd.to_numeric(
                X_val[column], errors="coerce"
            )

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians).fillna(0)
    X_val = X_val.fillna(train_medians).fillna(0)

    return X_train, X_val


def calculate_mrr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    ranked_classes = np.argsort(-probabilities, axis=1)

    reciprocal_ranks = []
    for row_index, true_class in enumerate(y_true):
        positions = np.where(
            ranked_classes[row_index] == true_class
        )[0]

        if len(positions) == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(
                1.0 / float(positions[0] + 1)
            )

    return float(np.mean(reciprocal_ranks))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--drop-cols", default="")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-train-rows", type=int, default=10000)
    parser.add_argument("--max-val-rows", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)

    args = parser.parse_args()

    train_df = pd.read_parquet(args.train_parquet)
    val_df = pd.read_parquet(args.val_parquet)

    if args.label_col not in train_df.columns:
        raise KeyError(
            f"Label column missing from train: {args.label_col}"
        )

    if args.label_col not in val_df.columns:
        raise KeyError(
            f"Label column missing from val: {args.label_col}"
        )

    train_df = train_df[
        train_df[args.label_col].notna()
    ].copy()

    val_df = val_df[
        val_df[args.label_col].notna()
    ].copy()

    if (
        args.max_train_rows > 0
        and len(train_df) > args.max_train_rows
    ):
        train_df = train_df.sample(
            n=args.max_train_rows,
            random_state=args.seed,
        )

    if (
        args.max_val_rows > 0
        and len(val_df) > args.max_val_rows
    ):
        val_df = val_df.sample(
            n=args.max_val_rows,
            random_state=args.seed,
        )

    train_labels = canonical_label_series(
        train_df[args.label_col]
    )
    val_labels = canonical_label_series(
        val_df[args.label_col]
    )

    train_classes = set(train_labels.unique())
    seen_mask = val_labels.isin(train_classes)

    unseen_rows = int((~seen_mask).sum())

    if unseen_rows > 0:
        print(
            f"[warn] dropping {unseen_rows} val rows "
            "with unseen labels"
        )

    val_df = val_df.loc[seen_mask].copy()
    val_labels = val_labels.loc[seen_mask]

    if len(val_df) == 0:
        raise ValueError(
            "No validation rows remain after filtering "
            "unseen labels."
        )

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)

    user_drop_cols = {
        column.strip()
        for column in args.drop_cols.split(",")
        if column.strip()
    }

    excluded_columns = user_drop_cols | {args.label_col}

    feature_cols = [
        column
        for column in train_df.columns
        if column in val_df.columns
        and column not in excluded_columns
    ]

    if not feature_cols:
        raise ValueError("No feature columns remain.")

    X_train, X_val = prepare_features(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
    )

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        X_train.to_numpy(),
        y_train,
    )

    probabilities = model.predict_proba(
        X_val.to_numpy()
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    metrics = {
        "dataset": args.dataset,
        "task": args.task,
        "variant": args.variant,
        "decoder": "catboost_multiclass",
        "seed": args.seed,
        "n_train_used": int(len(train_df)),
        "n_val_used": int(len(val_df)),
        "n_val_unseen_dropped": unseen_rows,
        "n_features": int(len(feature_cols)),
        "n_classes_train": int(len(label_encoder.classes_)),
        "accuracy": float(
            accuracy_score(y_val, predictions)
        ),
        "micro_f1": float(
            f1_score(
                y_val,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_val,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "mrr": calculate_mrr(
            y_true=y_val,
            probabilities=probabilities,
        ),
        "log_loss": float(
            log_loss(
                y_val,
                probabilities,
                labels=np.arange(
                    len(label_encoder.classes_)
                ),
            )
        ),
        "feature_cols": "|".join(feature_cols),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "metrics.csv"

    pd.DataFrame([metrics]).to_csv(
        output_path,
        index=False,
    )

    print("\n=== METRICS ===")
    print(
        pd.DataFrame([metrics])
        .drop(columns=["feature_cols"])
        .to_string(index=False)
    )

    print("\nfeatures:")
    for feature in feature_cols:
        print(" -", feature)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
