from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fdhg.onboarding.pipeline import (
    _load_tables,
    discover_relations,
    onboard_dataset,
    profile_tables,
)
from tests.unit.onboarding_fixtures import write_onboarding_fixture


def test_foreign_key_discovery_with_referential_coverage(
    tmp_path: Path,
) -> None:
    config_path = write_onboarding_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tables = _load_tables(config_path, config)

    relations = discover_relations(
        tables=tables,
        profiles=profile_tables(tables),
        configured=config["tables"],
        threshold=0.98,
    )

    accepted = relations["accepted"]
    assert accepted["child_table"] == "events"
    assert accepted["parent_table"] == "users"
    assert accepted["referential_coverage"] == 1.0
    assert accepted["parent_primary_key_proven"] is True


def test_ambiguous_foreign_key_blocks(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tables"]["events"]["foreign_keys"].append({
        "column": "user_id",
        "references": {"table": "users", "column": "user_id"},
    })
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=False,
    )

    assert report.status == "blocked"
    assert "ambiguous_foreign_key" in report.blockers


def test_orphan_threshold_failure_blocks(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path, orphan=True)

    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=False,
    )

    assert report.status == "blocked"
    assert "referential_integrity_failure" in report.blockers


def test_ambiguous_timestamp_blocks(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path, ambiguous_time=True)

    report = onboard_dataset(
        config_path=config_path,
        output_root=tmp_path / "out",
        write=False,
    )

    assert report.status == "blocked"
    assert "ambiguous_event_time" in report.blockers


def test_relation_name_match_alone_is_not_accepted(tmp_path: Path) -> None:
    config_path = write_onboarding_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tables"]["users"]["primary_key"] = "signup_time"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous_foreign_key"):
        tables = _load_tables(config_path, config)
        discover_relations(
            tables=tables,
            profiles=profile_tables(tables),
            configured=config["tables"],
            threshold=0.98,
        )
