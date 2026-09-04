#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def feature_hash(columns: list[str]) -> str:
    payload = "\n".join(columns).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_label_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        arr = numeric.to_numpy(dtype=float)

        if np.all(
            np.isclose(
                arr,
                np.round(arr),
            )
        ):
            return pd.Series(
                np.round(arr).astype(np.int64),
                index=series.index,
            ).astype(str)

        return numeric.astype(str)

    return series.astype(str)


def preprocess_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
):
    X_train = train_df[
        feature_cols
    ].copy()

    X_val = val_df[
        feature_cols
    ].copy()

    categorical_mappings = {}

    for col in feature_cols:
        tr = X_train[col]
        va = X_val[col]

        if pd.api.types.is_datetime64_any_dtype(tr):
            X_train[col] = (
                pd.to_datetime(
                    tr,
                    errors="coerce",
                ).astype("int64") / 1e9
            )

            X_val[col] = (
                pd.to_datetime(
                    va,
                    errors="coerce",
                ).astype("int64") / 1e9
            )

        elif pd.api.types.is_bool_dtype(tr):
            X_train[col] = (
                tr.astype("float64")
            )

            X_val[col] = (
                va.astype("float64")
            )

        elif (
            pd.api.types.is_numeric_dtype(tr)
        ):
            X_train[col] = pd.to_numeric(
                tr,
                errors="coerce",
            )

            X_val[col] = pd.to_numeric(
                va,
                errors="coerce",
            )

        else:
            # TRAIN-ONLY categorical encoding.
            train_str = (
                tr.astype("string")
                .fillna("__MISSING__")
            )

            val_str = (
                va.astype("string")
                .fillna("__MISSING__")
            )

            categories = sorted(
                train_str.unique().tolist()
            )

            mapping = {
                value: index
                for index, value
                in enumerate(categories)
            }

            categorical_mappings[col] = {
                "n_train_categories":
                    len(mapping),
            }

            X_train[col] = (
                train_str
                .map(mapping)
                .astype("float64")
            )

            # Unseen validation category = -1.
            X_val[col] = (
                val_str
                .map(mapping)
                .fillna(-1)
                .astype("float64")
            )

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_val = X_val.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Imputation learned from TRAIN ONLY.
    medians = X_train.median(
        numeric_only=True
    )

    X_train = (
        X_train
        .fillna(medians)
        .fillna(0.0)
    )

    X_val = (
        X_val
        .fillna(medians)
        .fillna(0.0)
    )

    return (
        X_train,
        X_val,
        categorical_mappings,
    )


def calculate_mrr(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    ranked = np.argsort(
        -probabilities,
        axis=1,
    )

    rr = np.zeros(
        len(y_true),
        dtype=float,
    )

    for i, true_class in enumerate(y_true):
        positions = np.where(
            ranked[i] == true_class
        )[0]

        if len(positions):
            rr[i] = (
                1.0
                / float(positions[0] + 1)
            )

    return float(rr.mean())


def make_model(
    *,
    model_name: str,
    problem_type: str,
    seed: int,
    threads: int,
):
    if problem_type in {
        "binary",
        "multiclass",
    }:
        if model_name == "catboost":
            from catboost import (
                CatBoostClassifier,
            )

            kwargs = dict(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=threads,
            )

            if problem_type == "binary":
                kwargs[
                    "loss_function"
                ] = "Logloss"
            else:
                kwargs[
                    "loss_function"
                ] = "MultiClass"

            return CatBoostClassifier(
                **kwargs
            )

        if model_name == "xgboost":
            from xgboost import (
                XGBClassifier,
            )

            kwargs = dict(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                tree_method="hist",
                n_jobs=threads,
            )

            if problem_type == "binary":
                kwargs.update(
                    objective=(
                        "binary:logistic"
                    ),
                    eval_metric="logloss",
                )
            else:
                kwargs.update(
                    objective=(
                        "multi:softprob"
                    ),
                    eval_metric="mlogloss",
                )

            return XGBClassifier(
                **kwargs
            )

    elif problem_type == "regression":
        if model_name == "catboost":
            from catboost import (
                CatBoostRegressor,
            )

            return CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=threads,
            )

        if model_name == "xgboost":
            from xgboost import (
                XGBRegressor,
            )

            return XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective=(
                    "reg:squarederror"
                ),
                random_state=seed,
                tree_method="hist",
                n_jobs=threads,
            )

    raise ValueError(
        "Unsupported model/problem_type: "
        f"{model_name}/{problem_type}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--model",
        choices=[
            "xgboost",
            "catboost",
        ],
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--threads",
        type=int,
        default=2,
    )

    args = ap.parse_args()

    matrix_dir = (
        args.matrix_dir.resolve()
    )

    manifest_path = (
        matrix_dir / "manifest.json"
    )

    train_path = (
        matrix_dir / "train.parquet"
    )

    val_path = (
        matrix_dir / "val.parquet"
    )

    for path in [
        manifest_path,
        train_path,
        val_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = load_json(
        manifest_path
    )

    feature_cols = list(
        manifest[
            "model_feature_columns"
        ]
    )

    if not feature_cols:
        raise ValueError(
            "Zero model_feature_columns"
        )

    if len(feature_cols) != int(
        manifest[
            "model_feature_count"
        ]
    ):
        raise ValueError(
            "Manifest feature count mismatch"
        )

    label_col = str(
        manifest["label_col"]
    )

    raw_problem_type = str(
        manifest["problem_type"]
    ).lower()

    problem_type_aliases = {
        "binary": "binary",
        "binary_classification": "binary",
        "multiclass": "multiclass",
        "multi_class": "multiclass",
        "multiclass_classification": "multiclass",
        "regression": "regression",
    }

    if raw_problem_type not in problem_type_aliases:
        raise ValueError(
            "Unsupported manifest problem_type: "
            f"{raw_problem_type}"
        )

    problem_type = problem_type_aliases[
        raw_problem_type
    ]

    dataset = str(
        manifest["dataset"]
    )

    task = str(
        manifest["task"]
    )

    selected_variant = str(
        manifest["selected_variant"]
    )

    if (
        manifest.get(
            "test_split_accessed"
        )
        is not False
    ):
        raise RuntimeError(
            "REFUSING: test access flag != False"
        )

    if (
        manifest.get(
            "official_validation_used_for_selection"
        )
        is not False
    ):
        raise RuntimeError(
            "REFUSING: official validation "
            "was used for selection"
        )

    train = pd.read_parquet(
        train_path
    )

    val = pd.read_parquet(
        val_path
    )

    required_cols = {
        *feature_cols,
        label_col,
    }

    missing_train = sorted(
        required_cols
        - set(train.columns)
    )

    missing_val = sorted(
        required_cols
        - set(val.columns)
    )

    if missing_train:
        raise KeyError(
            f"Missing train columns: "
            f"{missing_train}"
        )

    if missing_val:
        raise KeyError(
            f"Missing val columns: "
            f"{missing_val}"
        )

    # Use FULL frozen official train/val.
    train = train.loc[
        train[label_col].notna()
    ].reset_index(drop=True)

    val = val.loc[
        val[label_col].notna()
    ].reset_index(drop=True)

    X_train, X_val, cat_info = (
        preprocess_features(
            train,
            val,
            feature_cols,
        )
    )

    result = {
        "dataset": dataset,
        "task": task,
        "selected_variant":
            selected_variant,
        "predictor":
            args.model,
        "seed":
            args.seed,
        "threads":
            args.threads,
        "problem_type":
            problem_type,
        "primary_metric":
            manifest[
                "primary_metric"
            ],
        "n_train":
            int(len(train)),
        "n_val":
            int(len(val)),
        "n_features":
            int(len(feature_cols)),
        "feature_hash":
            feature_hash(
                feature_cols
            ),
        "matrix_dir":
            str(matrix_dir),
        "test_split_accessed":
            bool(manifest["test_split_accessed"]),
        "official_validation_was_used_for_selection":
            bool(
                manifest[
                    "official_validation_used_for_selection"
                ]
            ),
    }

    nonconstant_feature_count = sum(
        X_train[col].nunique(dropna=False) > 1
        for col in X_train.columns
    )

    result["n_nonconstant_features"] = int(
        nonconstant_feature_count
    )
    result["status"] = "completed"

    # CatBoost refuses a design matrix in which every
    # feature is constant. Record this deterministic,
    # representation-level limitation as a structural
    # skip rather than an execution failure.
    if (
        args.model == "catboost"
        and nonconstant_feature_count == 0
    ):
        result["status"] = "skipped"
        result["skip_reason"] = (
            "all_features_constant_after_preprocessing"
        )

        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_path = (
            args.output_dir / "metrics.json"
        )
        metrics_path.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        print()
        print("EVALUATION SKIPPED")
        print(f"{dataset}/{task}")
        print(f"predictor={args.model}")
        print(f"seed={args.seed}")
        print(
            "reason="
            + result["skip_reason"]
        )
        print(f"saved: {metrics_path}")
        return

    model = make_model(
        model_name=args.model,
        problem_type=problem_type,
        seed=args.seed,
        threads=args.threads,
    )

    if problem_type == "binary":
        y_train_raw = canonical_label_series(
            train[label_col]
        )

        y_val_raw = canonical_label_series(
            val[label_col]
        )

        le = LabelEncoder()

        y_train = le.fit_transform(
            y_train_raw
        )

        if len(le.classes_) != 2:
            raise ValueError(
                "Binary task does not have "
                f"2 train classes: "
                f"{list(le.classes_)}"
            )

        seen = set(le.classes_)

        unseen_mask = (
            ~y_val_raw.isin(seen)
        )

        if unseen_mask.any():
            raise ValueError(
                "Binary validation contains "
                "unseen labels"
            )

        y_val = le.transform(
            y_val_raw
        )

        model.fit(
            X_train,
            y_train,
        )

        proba = model.predict_proba(
            X_val
        )

        pred = np.argmax(
            proba,
            axis=1,
        )

        score = proba[:, 1]

        result.update(
            accuracy=float(
                accuracy_score(
                    y_val,
                    pred,
                )
            ),
            roc_auc=float(
                roc_auc_score(
                    y_val,
                    score,
                )
            ),
            average_precision=float(
                average_precision_score(
                    y_val,
                    score,
                )
            ),
            log_loss=float(
                log_loss(
                    y_val,
                    proba,
                    labels=[0, 1],
                )
            ),
        )

    elif problem_type == "multiclass":
        y_train_raw = canonical_label_series(
            train[label_col]
        )

        y_val_raw = canonical_label_series(
            val[label_col]
        )

        le = LabelEncoder()

        y_train = le.fit_transform(
            y_train_raw
        )

        seen = set(le.classes_)

        seen_mask = (
            y_val_raw.isin(seen)
        )

        unseen_dropped = int(
            (~seen_mask).sum()
        )

        if unseen_dropped:
            X_val = X_val.loc[
                seen_mask.to_numpy()
            ].reset_index(drop=True)

            y_val_raw = (
                y_val_raw.loc[
                    seen_mask
                ]
                .reset_index(drop=True)
            )

        if len(y_val_raw) == 0:
            raise ValueError(
                "No validation rows after "
                "unseen-class filtering"
            )

        y_val = le.transform(
            y_val_raw
        )

        model.fit(
            X_train,
            y_train,
        )

        proba = model.predict_proba(
            X_val
        )

        pred = np.argmax(
            proba,
            axis=1,
        )

        result.update(
            n_val_unseen_dropped=
                unseen_dropped,
            n_val=int(
                len(y_val)
            ),
            n_classes_train=int(
                len(le.classes_)
            ),
            accuracy=float(
                accuracy_score(
                    y_val,
                    pred,
                )
            ),
            micro_f1=float(
                f1_score(
                    y_val,
                    pred,
                    average="micro",
                    zero_division=0,
                )
            ),
            macro_f1=float(
                f1_score(
                    y_val,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            weighted_f1=float(
                f1_score(
                    y_val,
                    pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            mrr=calculate_mrr(
                y_val,
                proba,
            ),
            log_loss=float(
                log_loss(
                    y_val,
                    proba,
                    labels=np.arange(
                        len(le.classes_)
                    ),
                )
            ),
        )

    elif problem_type == "regression":
        y_train = pd.to_numeric(
            train[label_col],
            errors="coerce",
        )

        y_val = pd.to_numeric(
            val[label_col],
            errors="coerce",
        )

        train_keep = y_train.notna()
        val_keep = y_val.notna()

        X_train = (
            X_train.loc[
                train_keep
            ]
            .reset_index(drop=True)
        )

        X_val = (
            X_val.loc[
                val_keep
            ]
            .reset_index(drop=True)
        )

        y_train = (
            y_train.loc[
                train_keep
            ]
            .to_numpy(dtype=float)
        )

        y_val = (
            y_val.loc[
                val_keep
            ]
            .to_numpy(dtype=float)
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_val
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_val,
                    pred,
                )
            )
        )

        result.update(
            n_train=int(
                len(y_train)
            ),
            n_val=int(
                len(y_val)
            ),
            rmse=rmse,
            mae=float(
                mean_absolute_error(
                    y_val,
                    pred,
                )
            ),
            r2=float(
                r2_score(
                    y_val,
                    pred,
                )
            ),
        )

    else:
        raise ValueError(
            f"Unknown problem_type="
            f"{problem_type}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result[
        "categorical_encoding"
    ] = (
        "train_only_mapping_"
        "unseen_val_minus1"
    )

    result[
        "categorical_feature_count"
    ] = len(cat_info)

    with (
        args.output_dir
        / "metrics.json"
    ).open("w") as f:
        json.dump(
            result,
            f,
            indent=2,
            default=str,
        )

    flat_result = {
        k: v
        for k, v in result.items()
        if not isinstance(
            v,
            (dict, list),
        )
    }

    pd.DataFrame(
        [flat_result]
    ).to_csv(
        args.output_dir
        / "metrics.csv",
        index=False,
    )

    with (
        args.output_dir
        / "feature_columns.json"
    ).open("w") as f:
        json.dump(
            feature_cols,
            f,
            indent=2,
        )

    print()
    print(
        "EVALUATION COMPLETED"
    )
    print(
        f"{dataset}/{task}"
    )
    print(
        f"variant="
        f"{selected_variant}"
    )
    print(
        f"predictor="
        f"{args.model}"
    )
    print(
        f"seed={args.seed}"
    )
    print(
        f"problem_type="
        f"{problem_type}"
    )
    print(
        f"n_train="
        f"{result['n_train']}"
    )
    print(
        f"n_val="
        f"{result['n_val']}"
    )
    print(
        f"n_features="
        f"{result['n_features']}"
    )

    primary = str(
        result["primary_metric"]
    )

    if primary in result:
        print(
            f"primary_metric "
            f"{primary}="
            f"{result[primary]}"
        )

    print(
        "saved:",
        args.output_dir
        / "metrics.json",
    )


if __name__ == "__main__":
    main()
