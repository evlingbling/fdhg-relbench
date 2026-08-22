from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_sweep_module():
    path = Path("scripts/experiments/run_candidate_program_sweep.py")
    spec = importlib.util.spec_from_file_location(
        "run_candidate_program_sweep",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_metrics(
    root: Path,
    *,
    candidate: str,
    seed: int,
    score: float,
) -> None:
    path = root / candidate / f"seed{seed}" / "metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([
        {
            "roc_auc": score,
            "n_features": 4,
        }
    ]).to_csv(path, index=False)


def test_load_seed_metric_rows_for_export(tmp_path: Path) -> None:
    sweep = load_sweep_module()
    write_metrics(tmp_path, candidate="dfs", seed=41, score=0.70)
    write_metrics(tmp_path, candidate="fdhg_a", seed=41, score=0.75)

    rows = sweep.load_seed_metric_rows(
        result_root=tmp_path,
        candidates=["dfs", "fdhg_a"],
        seeds=[41, 42],
        primary_metric="roc_auc",
    )

    assert list(rows["candidate"]) == ["dfs", "fdhg_a"]
    assert list(rows["seed"]) == [41, 41]
    assert all("metrics.csv" in value for value in rows["evidence_location"])


def test_no_output_by_default_is_parser_default() -> None:
    source = Path(
        "scripts/experiments/run_candidate_program_sweep.py"
    ).read_text(encoding="utf-8")

    assert "--canonical-validation-output" in source
    assert "default=" not in source.split(
        "--canonical-validation-output",
        1,
    )[1].split(")", 1)[0]


def test_overwrite_refusal(tmp_path: Path) -> None:
    sweep = load_sweep_module()
    output = tmp_path / "validation.csv"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sweep._validate_validation_output_path(
            output,
            overwrite=False,
        )


def test_explicit_tmp_output_allowed() -> None:
    sweep = load_sweep_module()
    output = Path("/tmp/fdhg_candidate_sweep_validation.csv")
    if output.exists():
        output.unlink()

    sweep._validate_validation_output_path(output, overwrite=False)


def test_results_paper_tables_output_rejected() -> None:
    sweep = load_sweep_module()

    with pytest.raises(ValueError, match="results/paper_tables"):
        sweep._validate_validation_output_path(
            Path("results/paper_tables/validation.csv"),
            overwrite=True,
        )
