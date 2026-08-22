from __future__ import annotations

from pathlib import Path

from fdhg.onboarding.pipeline import _load_tables, profile_tables
from tests.unit.onboarding_fixtures import write_onboarding_fixture


def test_schema_profiling_for_csv(tmp_path: Path) -> None:
    config = write_onboarding_fixture(tmp_path, table_format="csv")

    profiles = profile_tables(_load_tables(config, _read_config(config)))

    assert profiles["users"]["row_count"] == 4
    user_cols = {row["column"]: row for row in profiles["users"]["columns"]}
    assert user_cols["user_id"]["is_unique_non_null"] is True
    assert user_cols["future_spend"]["is_numeric"] is True
    assert profiles["events"]["duplicate_rows"] == 0


def test_schema_profiling_for_parquet(tmp_path: Path) -> None:
    config = write_onboarding_fixture(tmp_path, table_format="parquet")

    profiles = profile_tables(_load_tables(config, _read_config(config)))

    assert profiles["events"]["row_count"] == 7
    event_cols = {row["column"]: row for row in profiles["events"]["columns"]}
    assert event_cols["event_id"]["is_unique_non_null"] is True
    assert event_cols["created_at"]["timestamp_parse_success_rate"] == 1.0


def test_exact_primary_key_discovery(tmp_path: Path) -> None:
    config = write_onboarding_fixture(tmp_path)

    profiles = profile_tables(_load_tables(config, _read_config(config)))

    assert "user_id" in profiles["users"]["candidate_primary_keys"]
    assert "event_id" in profiles["events"]["candidate_primary_keys"]


def _read_config(path: Path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))
