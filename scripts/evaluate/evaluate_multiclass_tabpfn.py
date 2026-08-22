import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier


def canonical_label_series(
    series: pd.Series,
) -> pd.Series:
    """Return stable numeric labels when possible, otherwise strings."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    original_non_missing = series.notna()
    numeric_conversion_ok = bool(
        numeric[original_non_missing].notna().all()
    )

    if numeric_conversion_ok:
        numeric = numeric.astype("float64")

        finite = numeric.dropna()
        integer_valued = bool(
            ((finite % 1) == 0).all()
        )

        if integer_valued:
            return numeric.astype("Int64")

        return numeric

    return (
        series.astype("string")
        .fillna("__MISSING_LABEL__")
    )

def preprocess_X(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on train and apply it consistently to val."""
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()

    for column in feature_cols:
        train_series = X_train[column]
        val_series = X_val[column]

        train_numeric = pd.to_numeric(
            train_series,
            errors="coerce",
        )
        val_numeric = pd.to_numeric(
            val_series,
            errors="coerce",
        )

        # Treat a column as numeric when all non-missing training
        # values can be represented numerically.
        original_non_missing = train_series.notna()
        numeric_non_missing = train_numeric.notna()

        is_numeric = bool(
            numeric_non_missing[original_non_missing].all()
        )

        if is_numeric:
            train_numeric = train_numeric.astype("float64")
            val_numeric = val_numeric.astype("float64")

            median = train_numeric.median()
            if pd.isna(median):
                median = 0.0

            X_train[column] = train_numeric.fillna(
                float(median)
            )
            X_val[column] = val_numeric.fillna(
                float(median)
            )
        else:
            train_text = train_series.astype("string")
            val_text = val_series.astype("string")

            mode = train_text.mode(dropna=True)
            fill_value = (
                mode.iloc[0]
                if not mode.empty
                else "__MISSING__"
            )

            combined = pd.concat(
                [train_text, val_text],
                ignore_index=True,
            ).fillna(fill_value)

            codes, _ = pd.factorize(
                combined,
                sort=True,
            )

            X_train[column] = codes[
                : len(X_train)
            ].astype("float64")
            X_val[column] = codes[
                len(X_train) :
            ].astype("float64")

    return X_train, X_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--drop-cols", default="")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max-train-rows", type=int, default=10000)
    ap.add_argument("--max-val-rows", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    train = pd.read_parquet(args.train_parquet)
    val = pd.read_parquet(args.val_parquet)

    # Drop rows with missing labels.
    train = train[train[args.label_col].notna()].copy()
    val = val[val[args.label_col].notna()].copy()

    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(args.max_train_rows, random_state=args.seed)
    if args.max_val_rows and len(val) > args.max_val_rows:
        val = val.sample(args.max_val_rows, random_state=args.seed)

    # Keep only validation classes seen in train.
    # Important for SALT: train labels may be int-like (6) while val labels may be float-like (6.0).
    train_labels_raw = canonical_label_series(train[args.label_col])
    val_labels_raw = canonical_label_series(val[args.label_col])

    seen = set(train_labels_raw.unique())
    keep = val_labels_raw.isin(seen)
    if keep.sum() < len(val):
        print(f"[warn] dropping {len(val) - int(keep.sum())} val rows with unseen labels")
        val = val[keep].copy()
        val_labels_raw = val_labels_raw[keep]

    if len(val) == 0:
        raise ValueError(
            "All validation rows were dropped as unseen labels. "
            f"train label examples={sorted(list(seen))[:20]}"
        )

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels_raw)
    y_val = le.transform(val_labels_raw)

    drop_cols = {x.strip() for x in args.drop_cols.split(",") if x.strip()}
    drop_cols.add(args.label_col)

    feature_cols = [c for c in train.columns if c not in drop_cols and c in val.columns]
    X_train, X_val = preprocess_X(train, val, feature_cols)

    kwargs = {}
    for k, v in [("device", args.device), ("random_state", args.seed)]:
        try:
            test_kwargs = dict(kwargs)
            test_kwargs[k] = v
            TabPFNClassifier(**test_kwargs)
            kwargs[k] = v
        except TypeError:
            pass

    model = TabPFNClassifier(**kwargs)
    model.fit(X_train.to_numpy(), y_train)

    pred = model.predict(X_val.to_numpy())

    row = {
        "dataset": args.dataset,
        "task": args.task,
        "variant": args.variant,
        "decoder": "tabpfn_classifier_multiclass",
        "seed": args.seed,
        "n_train_used": int(len(train)),
        "n_val_used": int(len(val)),
        "n_features": int(len(feature_cols)),
        "n_classes_train": int(len(le.classes_)),
        "accuracy": float(accuracy_score(y_val, pred)),
        "micro_f1": float(
            f1_score(y_val, pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_val, pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_val, pred, average="weighted", zero_division=0)
        ),
        "feature_cols": "|".join(feature_cols),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val.to_numpy())

        ranked_classes = np.argsort(-proba, axis=1)
        reciprocal_ranks = []

        for row_index, true_class in enumerate(y_val):
            positions = np.where(
                ranked_classes[row_index] == true_class
            )[0]

            if len(positions) == 0:
                reciprocal_ranks.append(0.0)
            else:
                reciprocal_ranks.append(
                    1.0 / float(positions[0] + 1)
                )

        row["mrr"] = float(np.mean(reciprocal_ranks))

        try:
            row["log_loss"] = float(
                log_loss(
                    y_val,
                    proba,
                    labels=np.arange(len(le.classes_)),
                )
            )
        except Exception as e:
            row["log_loss"] = np.nan
            row["log_loss_error"] = f"{type(e).__name__}: {e}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "metrics.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)

    print("\n=== METRICS ===")
    print(pd.DataFrame([row]).drop(columns=["feature_cols"], errors="ignore").to_string(index=False))
    print("\nfeatures:")
    for c in feature_cols:
        print(" -", c)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
