from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_classification_parquets(tmp_path: Path, *, multiclass: bool = False):
    labels = [0, 1, 2, 0] if multiclass else [0, 1, 0, 1]
    train = pd.DataFrame({"label": labels, "f": [0.0, 1.0, 2.0, 3.0]})
    val_labels = [0, 1, 2] if multiclass else [0, 1]
    val = pd.DataFrame({"label": val_labels, "f": list(range(len(val_labels)))})
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    return train_path, val_path


def write_regression_parquets(tmp_path: Path):
    train = pd.DataFrame({"label": [0.0, 1.0, 2.0, 3.0], "f": [0.0, 1.0, 2.0, 3.0]})
    val = pd.DataFrame({"label": [0.0, 1.0], "f": [0.0, 1.0]})
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    return train_path, val_path


def run_script_main(monkeypatch, module, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", [str(module.__file__), *argv])
    module.main()


def base_args(train_path: Path, val_path: Path, out_dir: Path) -> list[str]:
    return [
        "--train-parquet",
        str(train_path),
        "--val-parquet",
        str(val_path),
        "--output-dir",
        str(out_dir),
        "--dataset",
        "rel-example",
        "--task",
        "pairwise",
        "--variant",
        "baseline_plus_pair_left_temporal",
        "--label-col",
        "label",
        "--seed",
        "41",
    ]


def test_binary_evaluator_parser_and_metrics_contract(tmp_path: Path, monkeypatch) -> None:
    module = load_script(
        "scripts/evaluate/evaluate_binary_tabpfn.py",
        "test_binary_eval_script",
    )
    train_path, val_path = write_classification_parquets(tmp_path)
    out_dir = tmp_path / "out"

    class FakeBinary:
        classes_ = np.array([0, 1])

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.array([0, 1])

        def predict_proba(self, x):
            return np.array([[0.8, 0.2], [0.2, 0.8]])

    monkeypatch.setattr(
        module,
        "get_tabpfn_classifier",
        lambda device, seed: FakeBinary(),
    )

    run_script_main(
        monkeypatch,
        module,
        [*base_args(train_path, val_path, out_dir), "--drop-cols", "", "--device", "cpu"],
    )

    row = pd.read_csv(out_dir / "metrics.csv").iloc[0].to_dict()
    assert row["dataset"] == "rel-example"
    assert row["task"] == "pairwise"
    assert row["variant"] == "baseline_plus_pair_left_temporal"
    assert int(row["seed"]) == 41
    assert "roc_auc" in row
    assert int(row["n_features"]) == 1
    predictions = pd.read_parquet(out_dir / "val_predictions.parquet")
    assert set(predictions["split"]) == {"val"}


def test_regression_evaluator_parser_and_metrics_contract(tmp_path: Path, monkeypatch) -> None:
    fake_tabpfn = types.ModuleType("tabpfn")

    class FakeRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.zeros(len(x), dtype=float)

    fake_tabpfn.TabPFNRegressor = FakeRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    module = load_script(
        "scripts/evaluate/evaluate_regression_tabpfn.py",
        "test_regression_eval_script",
    )
    train_path, val_path = write_regression_parquets(tmp_path)
    out_dir = tmp_path / "out"

    run_script_main(
        monkeypatch,
        module,
        [*base_args(train_path, val_path, out_dir), "--device", "cpu"],
    )

    row = pd.read_csv(out_dir / "metrics.csv").iloc[0].to_dict()
    assert row["variant"] == "baseline_plus_pair_left_temporal"
    assert "rmse" in row
    assert int(row["n_features"]) == 1


def test_multiclass_tabpfn_parser_and_metrics_contract(tmp_path: Path, monkeypatch) -> None:
    fake_tabpfn = types.ModuleType("tabpfn")

    class FakeClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.array([0, 1, 2])

        def predict_proba(self, x):
            return np.array([
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ])

    fake_tabpfn.TabPFNClassifier = FakeClassifier
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)
    module = load_script(
        "scripts/evaluate/evaluate_multiclass_tabpfn.py",
        "test_multiclass_tabpfn_eval_script",
    )
    train_path, val_path = write_classification_parquets(tmp_path, multiclass=True)
    out_dir = tmp_path / "out"

    run_script_main(
        monkeypatch,
        module,
        [*base_args(train_path, val_path, out_dir), "--device", "cpu"],
    )

    row = pd.read_csv(out_dir / "metrics.csv").iloc[0].to_dict()
    assert row["decoder"] == "tabpfn_classifier_multiclass"
    assert "macro_f1" in row
    assert int(row["n_features"]) == 1


def test_multiclass_catboost_parser_and_metrics_contract(tmp_path: Path, monkeypatch) -> None:
    fake_catboost = types.ModuleType("catboost")

    class FakeCatBoostClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            return np.array([
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ])

    fake_catboost.CatBoostClassifier = FakeCatBoostClassifier
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)
    module = load_script(
        "scripts/evaluate/evaluate_multiclass_catboost.py",
        "test_multiclass_catboost_eval_script",
    )
    train_path, val_path = write_classification_parquets(tmp_path, multiclass=True)
    out_dir = tmp_path / "out"

    run_script_main(
        monkeypatch,
        module,
        [
            *base_args(train_path, val_path, out_dir),
            "--iterations",
            "1",
            "--depth",
            "1",
        ],
    )

    row = pd.read_csv(out_dir / "metrics.csv").iloc[0].to_dict()
    assert row["decoder"] == "catboost_multiclass"
    assert "log_loss" in row
    assert int(row["n_features"]) == 1
