from __future__ import annotations

import json
from pathlib import Path

from fdhg.cli.materialize_candidate import main


def write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_print_only_mode(tmp_path: Path, capsys) -> None:
    exit_code = main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--program-id",
        "baseline",
        "--output-root",
        str(tmp_path),
        "--dry-run",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "DRY_RUN True" in out
    assert "SELECTOR_READY False" in out
    assert not list(tmp_path.rglob("*"))


def test_cli_write_requires_input_rows(tmp_path: Path, capsys) -> None:
    exit_code = main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--program-id",
        "baseline",
        "--output-root",
        str(tmp_path),
        "--write",
    ])

    assert exit_code == 1
    assert "--source-rows-json" in capsys.readouterr().err
    assert not list(tmp_path.rglob("*"))


def test_cli_write_fails_closed_without_provenance(
    tmp_path: Path,
    capsys,
) -> None:
    source = write_json(tmp_path / "source.json", {"beer_ratings": []})
    train = write_json(
        tmp_path / "train.json",
        [
            {
                "user_id": "u1",
                "timestamp": "2026-01-01T00:00:00",
                "label": 1,
            }
        ],
    )
    validation = write_json(
        tmp_path / "validation.json",
        [
            {
                "user_id": "u1",
                "timestamp": "2026-01-02T00:00:00",
                "label": 0,
            }
        ],
    )

    exit_code = main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--program-id",
        "baseline",
        "--output-root",
        str(tmp_path / "out"),
        "--write",
        "--source-rows-json",
        str(source),
        "--train-target-json",
        str(train),
        "--validation-target-json",
        str(validation),
    ])

    assert exit_code == 1
    assert "provenance" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_overwrite_refusal_on_existing_output(
    tmp_path: Path,
    capsys,
) -> None:
    existing = (
        tmp_path
        / "rel-ratebeer_user-place-liked_pairwise"
        / "candidates"
        / "baseline"
    )
    existing.mkdir(parents=True)
    source = write_json(tmp_path / "source.json", {"beer_ratings": []})
    train = write_json(tmp_path / "train.json", [])
    validation = write_json(tmp_path / "validation.json", [])

    exit_code = main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--program-id",
        "baseline",
        "--output-root",
        str(tmp_path),
        "--write",
        "--source-rows-json",
        str(source),
        "--train-target-json",
        str(train),
        "--validation-target-json",
        str(validation),
    ])

    assert exit_code == 1
    assert "baseline" in capsys.readouterr().err


def test_cli_explicit_tmp_output_dry_run(capsys) -> None:
    exit_code = main([
        "--dataset",
        "rel-ratebeer",
        "--task",
        "user-place-liked_pairwise",
        "--program-id",
        "baseline",
        "--output-root",
        "/tmp/fdhg-materialize-candidate-dry-run",
        "--dry-run",
    ])

    assert exit_code == 0
    assert "OUTPUT_DIR /tmp/fdhg-materialize-candidate-dry-run" in (
        capsys.readouterr().out
    )
