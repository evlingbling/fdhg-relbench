import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier


def canonical_label_series(s):
    """Normalize labels so 6 and 6.0 are treated as the same class."""
    x = pd.to_numeric(s, errors="ignore")
    if pd.api.types.is_numeric_dtype(x):
        xf = pd.to_numeric(x, errors="coerce")
        if xf.notna().all():
            # If all numeric labels are integer-valued, cast to int strings.
            if np.all(np.isclose(xf.to_numpy(), np.round(xf.to_numpy()))):
                return pd.Series(np.round(xf.to_numpy()).astype(int), index=s.index).astype(str)
            return xf.astype(str)
    return s.astype(str)


def preprocess_X(train_df, val_df, feature_cols):
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    for X in (X_train, X_val):
        for c in X.columns:
            if str(X[c].dtype).startswith("datetime"):
                X[c] = X[c].astype("int64") / 1e9
            elif X[c].dtype == "bool":
                X[c] = X[c].astype(int)
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        for c in X.columns:
            med = X[c].median()
            if pd.isna(med):
                med = 0.0
            X[c] = X[c].fillna(med)

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

    proba = None

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

    # Save row-aligned validation predictions for subgroup analysis.
    val_out = val.reset_index(drop=True).copy()

    metadata_cols = [
        c for c in [
            "__row_id",
            "date",
            "Author_ID",
            args.label_col,
            "dfs::author::past_paper_count",
        ]
        if c in val_out.columns
    ]

    prediction_df = val_out[metadata_cols].copy()
    prediction_df["y_true_encoded"] = y_val
    prediction_df["y_pred_encoded"] = pred
    prediction_df["y_true_label"] = le.inverse_transform(y_val)
    prediction_df["y_pred_label"] = le.inverse_transform(pred)

    if proba is not None:
        ranked_classes = np.argsort(-proba, axis=1)
        reciprocal_rank = np.zeros(len(y_val), dtype=float)

        for i, true_class in enumerate(y_val):
            positions = np.where(
                ranked_classes[i] == true_class
            )[0]

            if len(positions):
                reciprocal_rank[i] = (
                    1.0 / float(positions[0] + 1)
                )

        prediction_df["reciprocal_rank"] = reciprocal_rank
        prediction_df["true_class_probability"] = proba[
            np.arange(len(y_val)),
            y_val,
        ]

        np.save(
            out_dir / "val_probabilities.npy",
            proba,
        )

        pd.DataFrame(
            {
                "encoded_class": np.arange(len(le.classes_)),
                "original_label": le.classes_,
            }
        ).to_csv(
            out_dir / "class_mapping.csv",
            index=False,
        )

    prediction_path = out_dir / "val_predictions.parquet"
    prediction_df.to_parquet(
        prediction_path,
        index=False,
    )

    print("\n=== METRICS ===")
    print(pd.DataFrame([row]).drop(columns=["feature_cols"], errors="ignore").to_string(index=False))
    print("\nfeatures:")
    for c in feature_cols:
        print(" -", c)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
