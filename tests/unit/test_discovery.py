from __future__ import annotations

import importlib.util
from pathlib import Path


def load_sweep_module():
    path = Path("scripts/experiments/run_candidate_program_sweep.py")
    spec = importlib.util.spec_from_file_location(
        "run_candidate_program_sweep",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_materialized_candidates_ignores_private_dirs(
    tmp_path: Path,
) -> None:
    sweep = load_sweep_module()
    root = tmp_path / "rel-example_task"
    good = root / "candidates" / "candidate_a"
    private = root / "candidates" / "_candidate_b"
    incomplete = root / "candidates" / "candidate_c"
    for path in (good, private, incomplete):
        path.mkdir(parents=True)
    for path in (good, private):
        (path / "target_with_dfs_agg_train.parquet").touch()
        (path / "target_with_dfs_agg_val.parquet").touch()
    (incomplete / "target_with_dfs_agg_train.parquet").touch()

    discovered = sweep.discover_materialized_candidates(
        task_output_root=root,
        configured_candidates=["dfs"],
    )

    assert discovered == ["dfs", "candidate_a"]
