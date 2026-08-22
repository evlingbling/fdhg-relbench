import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score


def get_tabpfn_classifier(device: str, seed: int):
    from tabpfn import TabPFNClassifier

    kwargs = {}
    if device and device != "None":
        kwargs["device"] = device
    try:
        return TabPFNClassifier(random_state=seed, **kwargs)
    except TypeError:
        return TabPFNClassifier(**kwargs)


def sample_df(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def prepare_xy(df: pd.DataFrame, label_col: str, drop_cols: list[str]):
    if label_col not in df.columns:
        raise KeyError(f"label_col={label_col!r} not found. columns={list(df.columns)}")

    y = df[label_col].copy()

    # Binary screening only.
    if y.dtype == "bool":
        y = y.astype(int)
    y = pd.to_numeric(y, errors="coerce")

    keep = y.notna()
    df = df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)

    if y.nunique(dropna=True) != 2:
        raise ValueError(
            f"Expected binary label for screening, got nunique={y.nunique(dropna=True)}, "
            f"value_counts={y.value_counts(dropna=False).head(20).to_dict()}"
        )

    drop = set(drop_cols + [label_col])
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    # Keep simple numeric/bool columns only for TabPFN screening.
    for c in list(X.columns):
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif pd.api.types.is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype("int64") / 1e9
        elif not pd.api.types.is_numeric_dtype(X[c]):
            # Try categorical codes for low-risk screening.
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)

    # Clean infinities / missing.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Drop constant columns.
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

    if X.shape[1] == 0:
        raise ValueError("No nonconstant features left after preprocessing.")

    return X, y.to_numpy()


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
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train_parquet)
    val = pd.read_parquet(args.val_parquet)

    train = sample_df(train, args.max_train_rows, args.seed)
    val = sample_df(val, args.max_val_rows, args.seed)

    drop_cols = [x for x in args.drop_cols.split(",") if x]

    X_train, y_train = prepare_xy(train, args.label_col, drop_cols)
    X_val, y_val = prepare_xy(val, args.label_col, drop_cols)

    # Align validation features to training features.
    # This prevents sklearn/TabPFN feature-name mismatch when a baseline
    # matrix has split-specific missing/extra columns.
    missing_in_val = [c for c in X_train.columns if c not in X_val.columns]
    extra_in_val = [c for c in X_val.columns if c not in X_train.columns]
    if missing_in_val or extra_in_val:
        print("[ALIGN] missing_in_val=", missing_in_val)
        print("[ALIGN] extra_in_val=", extra_in_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    print("dataset/task:", args.dataset, args.task)
    print("variant:", args.variant)
    print("train shape:", X_train.shape, "label:", pd.Series(y_train).value_counts().to_dict())
    print("val shape:", X_val.shape, "label:", pd.Series(y_val).value_counts().to_dict())
    print("features:")
    for c in X_train.columns:
        print(" -", c)

    clf = get_tabpfn_classifier(args.device, args.seed)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_val)
    proba = clf.predict_proba(X_val)
    score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "roc_auc": float(roc_auc_score(y_val, score)),
        "average_precision": float(average_precision_score(y_val, score)),
        "log_loss": float(log_loss(y_val, score, labels=[0, 1])),
    }

    result = {
        "dataset": args.dataset,
        "task": args.task,
        "variant": args.variant,
        "seed": args.seed,
        "n_train_used": int(len(X_train)),
        "n_val_used": int(len(X_val)),
        "n_features": int(X_train.shape[1]),
        **metrics,
        "train_parquet": args.train_parquet,
        "val_parquet": args.val_parquet,
    }

    out = pd.DataFrame([result])
    out_path = out_dir / "metrics.csv"
    out.to_csv(out_path, index=False)

    # Save row-aligned validation predictions for subgroup/bootstrap analysis.
    val_out = val.reset_index(drop=True).copy()

    # Preserve identifiers and useful subgroup-analysis metadata when present.
    exact_metadata_cols = {
        "__row_id",
        "row_id",
        "timestamp",
        "date",
        "user_id",
        "UserId",
        "OwnerUserId",
        "product_id",
        "customer_id",
        "Author_ID",
        args.label_col,
    }

    metadata_cols = []
    for col in val_out.columns:
        col_lower = str(col).lower()

        keep = (
            col in exact_metadata_cols
            or "history_count" in col_lower
            or "past_" in col_lower
            or col_lower.endswith("_count")
            or "cold_start" in col_lower
            or "warm_start" in col_lower
        )

        if keep and col not in metadata_cols:
            metadata_cols.append(col)

    prediction_df = val_out[metadata_cols].copy()
    prediction_df["dataset"] = args.dataset
    prediction_df["task"] = args.task
    prediction_df["variant"] = args.variant
    prediction_df["seed"] = args.seed
    prediction_df["split"] = "val"
    prediction_df["y_true"] = pd.Series(y_val).reset_index(drop=True)
    prediction_df["y_pred"] = pd.Series(pred).reset_index(drop=True)

    # Binary positive-class probability.
    if proba is not None:
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            prediction_df["y_score"] = proba[:, 1]
            prediction_df["prob_class_0"] = proba[:, 0]
            prediction_df["prob_class_1"] = proba[:, 1]
        else:
            prediction_df["y_score"] = proba.reshape(-1)

    # Store the model class ordering used by predict_proba.
    if hasattr(clf, "classes_"):
        pd.DataFrame(
            {
                "probability_column": range(len(clf.classes_)),
                "class_label": clf.classes_,
            }
        ).to_csv(
            out_dir / "class_mapping.csv",
            index=False,
        )

    prediction_path = out_dir / "val_predictions.parquet"
    prediction_df.to_parquet(prediction_path, index=False)

    print("\n=== METRICS ===")
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
