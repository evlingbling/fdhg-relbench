from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocess(train_df, val_df, label_col, drop_cols):
    y_train = pd.to_numeric(train_df[label_col], errors="coerce").fillna(0).to_numpy()
    y_val = pd.to_numeric(val_df[label_col], errors="coerce").fillna(0).to_numpy()

    drop = set(drop_cols + [label_col])
    X_train = train_df.drop(columns=[c for c in drop if c in train_df.columns], errors="ignore").copy()
    X_val = val_df.drop(columns=[c for c in drop if c in val_df.columns], errors="ignore").copy()

    # Convert datetimes to int64 seconds.
    for df in [X_train, X_val]:
        for c in list(df.columns):
            if is_datetime64_any_dtype(df[c]):
                df[c] = df[c].view("int64") // 10**9

    # Simple categorical handling.
    all_df = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    for c in all_df.columns:
        if all_df[c].dtype == "object" or str(all_df[c].dtype).startswith("category"):
            all_df[c] = all_df[c].astype("string").fillna("__MISSING__")
            codes, _ = pd.factorize(all_df[c], sort=True)
            all_df[c] = codes
        else:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    all_df = all_df.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

    X_train2 = all_df.iloc[:len(X_train)].to_numpy(dtype=np.float32)
    X_val2 = all_df.iloc[len(X_train):].to_numpy(dtype=np.float32)

    return X_train2, y_train, X_val2, y_val, list(all_df.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--label-col", default="ltv")
    ap.add_argument("--drop-cols", default="")
    ap.add_argument("--model", choices=["catboost", "xgboost"], default="catboost")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max-train-rows", type=int, default=10000)
    ap.add_argument("--max-val-rows", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_parquet).head(args.max_train_rows)
    val_df = pd.read_parquet(args.val_parquet).head(args.max_val_rows)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    X_train, y_train, X_val, y_val, feature_cols = preprocess(
        train_df, val_df, args.label_col, drop_cols
    )

    if args.model == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=args.seed,
            tree_method="hist",
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    mae = float(mean_absolute_error(y_val, pred))
    r2 = float(r2_score(y_val, pred))

    # Also evaluate log1p target scale because LTV is very heavy-tailed.
    y_val_log = np.log1p(np.maximum(y_val, 0))
    pred_log = np.log1p(np.maximum(pred, 0))
    rmse_log = float(np.sqrt(mean_squared_error(y_val_log, pred_log)))
    mae_log = float(mean_absolute_error(y_val_log, pred_log))
    r2_log = float(r2_score(y_val_log, pred_log))

    row = {
        "dataset": args.dataset,
        "task": args.task,
        "variant": args.variant,
        "model": args.model,
        "seed": args.seed,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_features": len(feature_cols),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmse_log1p": rmse_log,
        "mae_log1p": mae_log,
        "r2_log1p": r2_log,
    }

    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(row, f, indent=2)

    print(pd.DataFrame([row]).to_string(index=False))
    print("Saved:", out_dir / "metrics.csv")


if __name__ == "__main__":
    main()
