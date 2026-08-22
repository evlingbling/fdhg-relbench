from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materialization_inputs import resolve_materialization_inputs
from fdhg.onboarding.pipeline import onboard_dataset
from tests.unit.onboarding_fixtures import write_onboarding_fixture


def _completed(tmp_path: Path):
    config_path = write_onboarding_fixture(tmp_path)
    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=True,
    )
    assert report.status == "completed"
    return report


def test_every_feature_has_temporal_predicate_and_lowering_provenance(
    tmp_path: Path,
) -> None:
    report = _completed(tmp_path)
    features = pd.read_csv(report.output_dir / "baseline_feature_manifest.csv")
    lowering = pd.read_csv(report.output_dir / "lowering_provenance.csv")

    assert features["temporal_predicate"].eq(
        "events.created_at <= users.timestamp"
    ).all()
    assert set(features["primitive_id"]) == set(lowering["primitive_id"])
    assert lowering["status"].eq("proven").all()


def test_safety_audits_are_task_scoped_and_passing(tmp_path: Path) -> None:
    report = _completed(tmp_path)

    for name in ("temporal_safety_audit.csv", "leakage_safety_audit.csv"):
        rows = pd.read_csv(report.output_dir / name)
        assert rows["dataset"].eq("example-commerce").all()
        assert rows["task"].eq("user-spend").all()
        assert rows["program_id"].eq("baseline_auto").all()
        assert rows["passed"].eq(True).all()


def test_failed_onboarding_remains_staging_only(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path, orphan=True)
    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=False,
    )

    assert report.status == "blocked"
    assert not report.output_dir.exists()
    assert not (tmp_path / "out" / "_example-commerce_user-spend.staging").exists()


def test_onboarding_manifest_resolves_through_materialization_inputs(
    tmp_path: Path,
) -> None:
    report = _completed(tmp_path)
    reproduction = report.output_dir / "resolved_task_spec.yaml"
    spec = load_task_spec(
        dataset="example-commerce",
        task="user-spend",
        reproduction_config=reproduction,
        semantics_config=tmp_path / "missing.yaml",
    )

    resolved = resolve_materialization_inputs(
        spec,
        reproduction_config=reproduction,
        semantics_config=tmp_path / "missing.yaml",
    )

    assert resolved.resolved is True
    assert resolved.inputs is not None
    assert resolved.inputs.train_target.path.name == "target_with_dfs_agg_train.parquet"
    assert resolved.inputs.validation_target.split == "validation"
    assert {row.program_id for row in resolved.inputs.explicit_lowering_evidence} == {
        "baseline_auto"
    }


def test_source_hash_change_invalidates_reuse(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path)
    first = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=True,
    )
    assert first.status == "completed"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    users_path = config_path.parent / raw["tables"]["users"]["path"]
    users = pd.read_parquet(users_path)
    users.loc[0, "future_spend"] = 101.0
    users.to_parquet(users_path, index=False)

    try:
        onboard_dataset(
            config_path=config_path,
            output_root=tmp_path / "out",
            write=True,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected stale onboarding reuse to be refused")


def test_config_change_invalidates_reuse(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path)
    assert onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=True,
    ).status == "completed"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["split"]["train_end"] = "2026-01-30"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    try:
        onboard_dataset(
            config_path=config_path,
            output_root=tmp_path / "out",
            write=True,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected changed config reuse to be refused")


def test_manifest_contains_output_hashes(tmp_path: Path) -> None:
    report = _completed(tmp_path)
    manifest = json.loads(
        (report.output_dir / "onboarding_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["status"] == "completed"
    assert "target_with_dfs_agg_train.parquet" in manifest["file_hashes"]
    assert "candidate_programs" in manifest
    workload = manifest["baseline_feature_workload"]
    assert workload["materialization_strategy"] == "grouped_temporal_sweep"
    assert workload["target_train_rows"] == 2
    assert workload["target_validation_rows"] == 2
    assert workload["child_rows"] == 7
    assert workload["materialization_executed"] is True
    assert workload["implementation_version"] == "onboarding-v1"
