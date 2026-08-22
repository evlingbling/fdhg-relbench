from __future__ import annotations

from pathlib import Path

from fdhg.cli import export_relbench_v1 as cli
from fdhg.onboarding.relbench_v1 import RelBenchV1ExportReport


def test_export_relbench_v1_cli_prints_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    calls = {}

    def fake_export(**kwargs):
        calls.update(kwargs)
        return RelBenchV1ExportReport(
            dataset=kwargs["dataset_name"],
            task=kwargs["task_name"],
            status="dry_run_ready",
            output_dir=tmp_path / "data" / "rel-ratebeer" / "user-count",
            config_path=kwargs["config_output"],
            blockers=(),
            dry_run=True,
            reused=False,
            relation_count=1,
            table_count=2,
            train_rows=2,
            validation_rows=1,
            relbench_version="0",
            dataset_class="FakeDataset",
            task_class="FakeTask",
            table_names=("beer_ratings", "users"),
            entity_key="user_id",
            target_time_col="timestamp",
            label_col="num_ratings",
            child_relation="beer_ratings.user_id->users.user_id",
            child_event_time_col="created_at",
        )

    monkeypatch.setattr(cli, "export_relbench_v1", fake_export)

    code = cli.main([
        "--dataset", "rel-ratebeer",
        "--task", "user-count",
        "--output-root", str(tmp_path / "data"),
        "--config-output", str(tmp_path / "config.yaml"),
        "--download",
        "--dry-run",
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert calls["download"] is True
    assert calls["write"] is False
    assert "STATUS dry_run_ready" in out
    assert "RELATION_COUNT 1" in out
    assert "TRAIN_ROWS 2" in out


def test_export_relbench_v1_cli_blocked_returns_two(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def fake_export(**kwargs):
        return RelBenchV1ExportReport(
            dataset=kwargs["dataset_name"],
            task=kwargs["task_name"],
            status="blocked",
            output_dir=tmp_path / "data",
            config_path=kwargs["config_output"],
            blockers=("missing_label",),
            dry_run=False,
            reused=False,
            relation_count=0,
            table_count=0,
            train_rows=0,
            validation_rows=0,
            relbench_version="",
            dataset_class="",
            task_class="",
            table_names=(),
            entity_key=None,
            target_time_col=None,
            label_col=None,
            child_relation=None,
            child_event_time_col=None,
        )

    monkeypatch.setattr(cli, "export_relbench_v1", fake_export)

    code = cli.main([
        "--dataset", "rel-ratebeer",
        "--task", "user-count",
        "--output-root", str(tmp_path / "data"),
        "--config-output", str(tmp_path / "config.yaml"),
        "--no-download",
        "--write",
    ])

    out = capsys.readouterr().out
    assert code == 2
    assert "STATUS blocked" in out
    assert "BLOCKERS missing_label" in out
