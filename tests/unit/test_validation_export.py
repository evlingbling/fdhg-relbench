from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from fdhg.compiler.selection import (
    CandidateSelectionPolicy,
    load_candidate_validation_results,
    select_candidate_program,
)
from fdhg.compiler.programs import CandidateProgram
from fdhg.compiler.validation_export import (
    CandidateSafetyEvidence,
    build_validation_export_records,
    inspect_candidate_safety_evidence,
    write_validation_export_csv,
)


SEEDS = (41, 42, 43, 44)


def aggregate(
    candidate: str,
    score: float,
    *,
    n_features: float = 4.0,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "primary_mean": score,
        "n_features_mean": n_features,
        "evidence_location": f"aggregate:{candidate}",
    }


def seed_rows(
    candidate: str,
    scores: tuple[float, ...],
    *,
    seeds: tuple[int, ...] = SEEDS,
    n_features: int = 4,
    metric: str = "roc_auc",
) -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate,
            "seed": seed,
            metric: score,
            "n_features": n_features,
            "evidence_location": f"seed:{candidate}:{seed}",
        }
        for seed, score in zip(seeds, scores)
    ]


def complete_safety(program_id: str) -> CandidateSafetyEvidence:
    return CandidateSafetyEvidence(
        program_id=program_id,
        materializable=True,
        leakage_safe=True,
        temporally_safe=True,
        provenance_complete=True,
        evidence_locations=(f"safety:{program_id}",),
    )


def report(
    *,
    metric: str = "roc_auc",
    direction: str = "higher",
    candidate_score: float = 0.75,
    baseline_score: float = 0.70,
    selected: str = "fdhg_a",
    candidate_seed_scores: tuple[float, ...] = (
        0.75,
        0.76,
        0.74,
        0.75,
    ),
    baseline_seed_scores: tuple[float, ...] = (
        0.70,
        0.70,
        0.70,
        0.70,
    ),
    candidate_program_ids: tuple[str, ...] = ("dfs", "fdhg_a"),
    safety: dict[str, CandidateSafetyEvidence] | None = None,
):
    return build_validation_export_records(
        dataset="rel-example",
        task="example-task",
        split="validation",
        primary_metric=metric,
        metric_direction=direction,
        candidate_program_ids=candidate_program_ids,
        expected_seeds=SEEDS,
        aggregate_rows=[
            aggregate("dfs", baseline_score, n_features=2),
            aggregate("fdhg_a", candidate_score),
        ],
        seed_rows=(
            seed_rows(
                "dfs",
                baseline_seed_scores,
                n_features=2,
                metric=metric,
            )
            + seed_rows(
                "fdhg_a",
                candidate_seed_scores,
                metric=metric,
            )
        ),
        selected_program_id=selected,
        baseline_program_id="dfs",
        safety_evidence_by_program=safety
        or {
            "dfs": complete_safety("dfs"),
            "fdhg_a": complete_safety("fdhg_a"),
        },
    )


def record_by_id(records, program_id: str):
    return next(
        record for record in records if record.program_id == program_id
    )


def test_aggregate_export_schema(tmp_path: Path) -> None:
    output = tmp_path / "export.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        write_validation_export_csv(report().aggregate_records, handle)

    header = output.read_text(encoding="utf-8").splitlines()[0]

    assert header == (
        "dataset,task,program_id,split,primary_metric,"
        "metric_direction,score,n_features,eligible,"
        "rejection_reason,evidence_location,materializable,"
        "leakage_safe,temporally_safe,provenance_complete,"
        "baseline_program_id,baseline_score"
    )


def test_seed_level_export_schema(tmp_path: Path) -> None:
    output = tmp_path / "seed_export.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        write_validation_export_csv(
            report().seed_records,
            handle,
            include_seed=True,
        )

    assert output.read_text(encoding="utf-8").splitlines()[0].endswith(
        ",seed"
    )


def test_validation_split_only() -> None:
    with pytest.raises(ValueError, match="validation split"):
        build_validation_export_records(
            dataset="rel-example",
            task="example-task",
            split="test",
            primary_metric="roc_auc",
            metric_direction="higher",
            candidate_program_ids=("dfs",),
            expected_seeds=SEEDS,
            aggregate_rows=[],
            seed_rows=[],
            selected_program_id="dfs",
        )


def test_test_final_metrics_excluded() -> None:
    with pytest.raises(ValueError, match="validation split"):
        build_validation_export_records(
            dataset="rel-example",
            task="example-task",
            split="paper-final",
            primary_metric="roc_auc",
            metric_direction="higher",
            candidate_program_ids=("dfs",),
            expected_seeds=SEEDS,
            aggregate_rows=[],
            seed_rows=[],
            selected_program_id="dfs",
        )


def test_deterministic_seed_ordering() -> None:
    seed_records = list(reversed(report().seed_records))

    assert [
        record.seed
        for record in sorted(
            seed_records,
            key=lambda item: (
                item.program_id,
                -1 if item.seed is None else item.seed,
            ),
        )
        if record_by_id(seed_records, record.program_id)
    ][:4] == [41, 42, 43, 44]


def test_duplicate_seed_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate seed"):
        build_validation_export_records(
            dataset="rel-example",
            task="example-task",
            split="validation",
            primary_metric="roc_auc",
            metric_direction="higher",
            candidate_program_ids=("dfs", "fdhg_a"),
            expected_seeds=SEEDS,
            aggregate_rows=[
                aggregate("dfs", 0.70),
                aggregate("fdhg_a", 0.75),
            ],
            seed_rows=(
                seed_rows("dfs", (0.70, 0.70, 0.70, 0.70))
                + seed_rows(
                    "fdhg_a",
                    (0.75, 0.76),
                    seeds=(41, 41),
                )
            ),
            selected_program_id="fdhg_a",
            safety_evidence_by_program={
                "dfs": complete_safety("dfs"),
                "fdhg_a": complete_safety("fdhg_a"),
            },
        )


def test_missing_seed_rejection() -> None:
    export = report(
        candidate_seed_scores=(0.75, 0.76, 0.74),
        candidate_program_ids=("dfs", "fdhg_a"),
    )

    fd = record_by_id(export.aggregate_records, "fdhg_a")
    assert "missing_seeds" in fd.rejection_reasons
    assert not fd.eligible


def test_stability_gate_failure_reason() -> None:
    export = report(
        candidate_score=0.69,
        selected="dfs",
        candidate_seed_scores=(0.69, 0.68, 0.70, 0.69),
    )

    fd = record_by_id(export.aggregate_records, "fdhg_a")
    assert "failed_stability_gate" in fd.rejection_reasons
    assert "failed_minimum_paired_delta" in fd.rejection_reasons


def test_higher_is_better_metric() -> None:
    fd = record_by_id(report().aggregate_records, "fdhg_a")

    assert fd.eligible
    assert fd.score == pytest.approx(0.75)


def test_lower_is_better_metric() -> None:
    export = report(
        metric="log_loss",
        direction="lower",
        baseline_score=0.40,
        candidate_score=0.30,
        baseline_seed_scores=(0.40, 0.41, 0.39, 0.40),
        candidate_seed_scores=(0.30, 0.31, 0.29, 0.30),
    )

    fd = record_by_id(export.aggregate_records, "fdhg_a")
    assert fd.eligible


def test_baseline_row_export() -> None:
    baseline = record_by_id(report().aggregate_records, "dfs")

    assert baseline.baseline_program_id == "dfs"
    assert baseline.baseline_score == pytest.approx(0.70)


def test_candidate_feature_counts() -> None:
    fd = record_by_id(report().aggregate_records, "fdhg_a")

    assert fd.n_features == 4


def test_missing_materialization_evidence() -> None:
    export = report(safety={"dfs": complete_safety("dfs")})

    fd = record_by_id(export.aggregate_records, "fdhg_a")
    assert "missing_materialization_evidence" in fd.rejection_reasons
    assert not fd.eligible


def test_leakage_audit_failure() -> None:
    bad = complete_safety("fdhg_a")
    export = report(
        safety={
            "dfs": complete_safety("dfs"),
            "fdhg_a": CandidateSafetyEvidence(
                program_id="fdhg_a",
                materializable=True,
                leakage_safe=False,
                temporally_safe=True,
                provenance_complete=True,
            ),
        }
    )

    fd = record_by_id(export.aggregate_records, bad.program_id)
    assert "leakage_audit_failure" in fd.rejection_reasons


def test_temporal_audit_failure() -> None:
    export = report(
        safety={
            "dfs": complete_safety("dfs"),
            "fdhg_a": CandidateSafetyEvidence(
                program_id="fdhg_a",
                materializable=True,
                leakage_safe=True,
                temporally_safe=False,
                provenance_complete=True,
            ),
        }
    )

    fd = record_by_id(export.aggregate_records, "fdhg_a")
    assert "temporal_audit_failure" in fd.rejection_reasons


def test_incomplete_provenance() -> None:
    export = report(
        safety={
            "dfs": complete_safety("dfs"),
            "fdhg_a": CandidateSafetyEvidence(
                program_id="fdhg_a",
                materializable=True,
                leakage_safe=True,
                temporally_safe=True,
                provenance_complete=False,
            ),
        }
    )

    fd = record_by_id(export.aggregate_records, "fdhg_a")
    assert "incomplete_provenance" in fd.rejection_reasons


def test_complete_safety_evidence() -> None:
    fd = record_by_id(report().aggregate_records, "fdhg_a")

    assert fd.materializable is True
    assert fd.leakage_safe is True
    assert fd.temporally_safe is True
    assert fd.provenance_complete is True


def test_ratebeer_incomplete_provenance_remains_false(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    (artifact / "target_with_dfs_agg_train.parquet").write_text("")
    (artifact / "target_with_dfs_agg_val.parquet").write_text("")

    safety = inspect_candidate_safety_evidence(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        program_id="baseline_plus_pairwise_temporal",
        artifact_dir=artifact,
        baseline_program_id="dfs",
    )

    assert safety.provenance_complete is False
    assert "incomplete_passthrough_provenance" in safety.rejection_reasons


def test_no_mutation() -> None:
    aggregates = [aggregate("dfs", 0.70), aggregate("fdhg_a", 0.75)]
    seeds = seed_rows("dfs", (0.70, 0.70, 0.70, 0.70)) + seed_rows(
        "fdhg_a",
        (0.75, 0.76, 0.74, 0.75),
    )
    aggregates_before = tuple(dict(row) for row in aggregates)
    seeds_before = tuple(dict(row) for row in seeds)

    build_validation_export_records(
        dataset="rel-example",
        task="example-task",
        split="validation",
        primary_metric="roc_auc",
        metric_direction="higher",
        candidate_program_ids=("dfs", "fdhg_a"),
        expected_seeds=SEEDS,
        aggregate_rows=aggregates,
        seed_rows=seeds,
        selected_program_id="fdhg_a",
        safety_evidence_by_program={
            "dfs": complete_safety("dfs"),
            "fdhg_a": complete_safety("fdhg_a"),
        },
    )

    assert tuple(dict(row) for row in aggregates) == aggregates_before
    assert tuple(dict(row) for row in seeds) == seeds_before


def test_no_filesystem_writes_in_pure_api(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("unexpected write")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "mkdir", fail)

    assert report().aggregate_records


def test_normalization_adapter_compatibility(tmp_path: Path) -> None:
    output = tmp_path / "canonical.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        write_validation_export_csv(report().aggregate_records, handle)

    records = load_candidate_validation_results(output)

    assert {record.program_id for record in records} == {"dfs", "fdhg_a"}


def test_selector_compatibility(tmp_path: Path) -> None:
    output = tmp_path / "canonical.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        write_validation_export_csv(report().aggregate_records, handle)
    records = load_candidate_validation_results(output)

    decision = select_candidate_program(
        (
            CandidateProgram("dfs", ["b"], ["baseline"], "baseline"),
            CandidateProgram(
                "fdhg_a",
                ["b", "a"],
                ["baseline", "temporal"],
                "fdhg",
            ),
        ),
        records,
        CandidateSelectionPolicy(
            dataset="rel-example",
            task="example-task",
            primary_metric="roc_auc",
            metric_direction="higher",
            baseline_program_id="dfs",
        ),
    )

    assert decision.selected_program_id == "fdhg_a"
