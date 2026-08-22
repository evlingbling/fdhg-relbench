from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fdhg.onboarding.pipeline import _load_tables, onboard_dataset, split_targets
from tests.unit.onboarding_fixtures import write_onboarding_fixture


def test_explicit_task_spec_resolves(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path)

    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=False,
    )

    assert report.status == "dry_run_ready"


def test_missing_label_blocks(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path, missing_label=True)

    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=False,
    )

    assert report.status == "blocked"
    assert "missing_label" in report.blockers


def test_temporal_split_has_no_overlap(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tables = _load_tables(config_path, config)

    split = split_targets(
        tables["users"].frame,
        target_time_col="timestamp",
        train_end="2026-01-31",
        validation_end="2026-02-28",
    )

    assert len(split["train"]) == 2
    assert len(split["validation"]) == 2
    assert set(split["train"]["user_id"]).isdisjoint(
        set(split["validation"]["user_id"])
    )


def test_invalid_temporal_boundaries_block(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path, invalid_split=True)

    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=False,
    )

    assert report.status == "blocked"
    assert "invalid_temporal_split" in report.blockers


def test_test_artifact_is_never_created_or_substituted(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path)
    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=True,
    )

    assert report.status == "completed"
    assert not (report.output_dir / "target_with_dfs_agg_test.parquet").exists()
    manifest = yaml.safe_load(
        (report.output_dir / "resolved_task_spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    prepared = manifest["tasks"]["example-commerce/user-spend"][
        "prepared_artifacts"
    ]
    assert "test" not in str(prepared).lower()


def test_missing_target_time_blocks() -> None:
    import pandas as pd

    with pytest.raises(ValueError, match="missing_target_time"):
        split_targets(
            pd.DataFrame({"timestamp": [None, None]}),
            target_time_col="timestamp",
            train_end="2026-01-31",
            validation_end="2026-02-28",
        )
