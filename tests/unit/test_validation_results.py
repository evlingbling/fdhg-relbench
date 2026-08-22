from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from fdhg.compiler.programs import CandidateProgram
from fdhg.compiler.selection import CandidateSelectionPolicy
from fdhg.compiler.selection import select_candidate_program
from fdhg.compiler.validation_results import (
    audit_validation_sources,
    normalize_validation_artifact,
    records_to_candidate_validation_results,
    write_canonical_validation_csv,
)


def aggregate_header() -> str:
    return (
        "dataset,task,program_id,split,primary_metric,"
        "metric_direction,validation_score,n_features,eligible,"
        "rejection_reason,evidence_location,materializable,"
        "leakage_safe,temporally_safe,provenance_complete,"
        "baseline_program_id,baseline_score"
    )


def aggregate_row(
    program_id: str = "fdhg_a",
    *,
    dataset: str = "rel-example",
    task: str = "example-task",
    split: str = "validation",
    score: str = "0.75",
    n_features: str = "4",
    eligible: str = "true",
    rejection_reason: str = "",
    evidence_location: str = "source.csv:2",
    materializable: str = "true",
    leakage_safe: str = "true",
    temporally_safe: str = "true",
    provenance_complete: str = "true",
    baseline_program_id: str = "baseline",
    baseline_score: str = "0.70",
) -> str:
    return (
        f"{dataset},{task},{program_id},{split},roc_auc,higher,"
        f"{score},{n_features},{eligible},{rejection_reason},"
        f"{evidence_location},{materializable},{leakage_safe},"
        f"{temporally_safe},{provenance_complete},"
        f"{baseline_program_id},{baseline_score}"
    )


def write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "source.csv"
    path.write_text(text, encoding="utf-8")
    return path


def aggregate_csv(*rows: str) -> str:
    return "\n".join([aggregate_header(), *rows])


def test_supported_source_schema_normalization(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(
            aggregate_row("baseline", score="0.70", n_features="2"),
            aggregate_row("fdhg_a", score="0.75", n_features="4"),
        ),
    )

    report = normalize_validation_artifact(
        path,
        dataset="rel-example",
        task="example-task",
    )

    assert report.supported
    assert report.adapter_name == "explicit_aggregate_validation_v1"
    assert [
        record.program_id for record in report.normalized_records
    ] == ["baseline", "fdhg_a"]
    assert not report.rejected_records


def test_missing_required_source_columns(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "dataset,task,program_id\nrel-example,example-task,fdhg_a\n",
    )

    report = normalize_validation_artifact(path)

    assert not report.supported
    assert report.rejected_records[0].reason.startswith(
        "unsupported_schema_missing_columns"
    )


def test_validation_split_accepted(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(aggregate_row(split="validation")),
    )

    report = normalize_validation_artifact(path)

    assert report.normalized_records[0].eligible
    assert report.normalized_records[0].rejection_reasons == ()


def test_test_final_split_rejected(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(aggregate_row(split="paper-final-test")),
    )

    report = normalize_validation_artifact(path)

    record = report.normalized_records[0]
    assert not record.eligible
    assert "test_or_final_split_evidence" in record.rejection_reasons


def test_missing_safety_evidence_remains_rejected(
    tmp_path: Path,
) -> None:
    path = write_csv(
        tmp_path,
        (
            "dataset,task,program_id,split,primary_metric,"
            "metric_direction,validation_score,n_features,eligible,"
            "rejection_reason,evidence_location\n"
            "rel-example,example-task,fdhg_a,validation,"
            "roc_auc,higher,0.75,4,true,,source.csv:2\n"
        ),
    )

    report = normalize_validation_artifact(path)

    assert not report.supported
    assert "materializable" in report.rejected_records[0].reason


def test_explicit_complete_safety_evidence_accepted(
    tmp_path: Path,
) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(
            aggregate_row(
                materializable="true",
                leakage_safe="true",
                temporally_safe="true",
                provenance_complete="true",
            )
        ),
    )

    report = normalize_validation_artifact(path)

    record = report.normalized_records[0]
    assert record.materializable is True
    assert record.leakage_safe is True
    assert record.temporally_safe is True
    assert record.provenance_complete is True


def test_task_mismatch_rejected(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(aggregate_row(task="other-task")),
    )

    report = normalize_validation_artifact(
        path,
        dataset="rel-example",
        task="example-task",
    )

    assert not report.normalized_records
    assert report.rejected_records[0].reason == "task_mismatch"


def test_program_mismatch_becomes_unknown_in_selector(
    tmp_path: Path,
) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(
            aggregate_row("baseline", score="0.70", n_features="2"),
            aggregate_row("unknown_program", score="0.99"),
        ),
    )
    report = normalize_validation_artifact(path)

    decision = select_candidate_program(
        (
            CandidateProgram(
                "baseline",
                ["b"],
                ["baseline"],
                "baseline",
            ),
            CandidateProgram(
                "fdhg_a",
                ["b", "a"],
                ["baseline", "temporal"],
                "fdhg",
            ),
        ),
        records_to_candidate_validation_results(
            report.normalized_records
        ),
        CandidateSelectionPolicy(
            dataset="rel-example",
            task="example-task",
            primary_metric="roc_auc",
            metric_direction="higher",
        ),
    )

    assert any(
        rejected.program_id == "unknown_program"
        and "unknown_program_id" in rejected.rejection_reasons
        for rejected in decision.rejected_candidates
    )


def test_conflicting_metric_rows(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(
            aggregate_row("fdhg_a", score="0.75"),
            aggregate_row("fdhg_a", score="0.76"),
        ),
    )

    with pytest.raises(ValueError, match="conflicting metric rows"):
        normalize_validation_artifact(path)


def test_conflicting_safety_evidence(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        seed_csv(
            seed_row("fdhg_a", seed="41", materializable="true"),
            seed_row("fdhg_a", seed="42", materializable="false"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="conflicting safety evidence",
    ):
        normalize_validation_artifact(path)


def seed_header() -> str:
    return (
        "dataset,task,program_id,split,seed,primary_metric,"
        "metric_direction,validation_score,n_features,eligible,"
        "rejection_reason,evidence_location,materializable,"
        "leakage_safe,temporally_safe,provenance_complete"
    )


def seed_row(
    program_id: str,
    *,
    seed: str,
    split: str = "validation",
    score: str = "0.75",
    materializable: str = "true",
) -> str:
    return (
        "rel-example,example-task,"
        f"{program_id},{split},{seed},roc_auc,higher,{score},"
        f"4,true,,seed-{seed}.csv,{materializable},true,true,true"
    )


def seed_csv(*rows: str) -> str:
    return "\n".join([seed_header(), *rows])


def test_duplicate_seed(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        seed_csv(
            seed_row("fdhg_a", seed="41"),
            seed_row("fdhg_a", seed="41"),
        ),
    )

    with pytest.raises(ValueError, match="duplicate seed"):
        normalize_validation_artifact(path)


def test_validation_test_seed_mixing(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        seed_csv(
            seed_row("fdhg_a", seed="41", split="validation"),
            seed_row("fdhg_a", seed="42", split="test"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="mixed validation and test seed evidence",
    ):
        normalize_validation_artifact(path)


def test_seed_aggregation_is_deterministic(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        seed_csv(
            seed_row("fdhg_a", seed="42", score="0.80"),
            seed_row("fdhg_a", seed="41", score="0.70"),
        ),
    )

    report = normalize_validation_artifact(path)

    record = report.normalized_records[0]
    assert record.score == pytest.approx(0.75)
    assert record.evidence_location == "seed-41.csv|seed-42.csv"


def test_no_mutation(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(aggregate_row()),
    )
    before = path.read_text(encoding="utf-8")

    normalize_validation_artifact(path)

    assert path.read_text(encoding="utf-8") == before


def test_no_filesystem_writes_in_pure_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(aggregate_row()),
    )

    def fail(*args, **kwargs):
        raise AssertionError("unexpected write")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "mkdir", fail)

    report = normalize_validation_artifact(path)

    assert report.normalized_records


def test_selector_integration(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(
            aggregate_row("baseline", score="0.70", n_features="2"),
            aggregate_row("fdhg_a", score="0.75", n_features="4"),
        ),
    )
    report = normalize_validation_artifact(path)

    decision = select_candidate_program(
        (
            CandidateProgram(
                "baseline",
                ["b"],
                ["baseline"],
                "baseline",
            ),
            CandidateProgram(
                "fdhg_a",
                ["b", "a"],
                ["baseline", "temporal"],
                "fdhg",
            ),
        ),
        records_to_candidate_validation_results(
            report.normalized_records
        ),
        CandidateSelectionPolicy(
            dataset="rel-example",
            task="example-task",
            primary_metric="roc_auc",
            metric_direction="higher",
        ),
    )

    assert decision.selected_program_id == "fdhg_a"


def test_ratebeer_incomplete_provenance_remains_unselectable(
    tmp_path: Path,
) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(
            aggregate_row(
                "baseline",
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                score="0.70",
                n_features="15",
            ),
            aggregate_row(
                "baseline_plus_pairwise_temporal",
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                score="0.99",
                n_features="30",
                provenance_complete="false",
            ),
        ),
    )
    report = normalize_validation_artifact(path)

    decision = select_candidate_program(
        (
            CandidateProgram(
                "baseline",
                ["b"],
                ["baseline"],
                "baseline",
            ),
            CandidateProgram(
                "baseline_plus_pairwise_temporal",
                ["b", "t"],
                ["baseline", "temporal"],
                "pairwise",
            ),
        ),
        records_to_candidate_validation_results(
            report.normalized_records
        ),
        CandidateSelectionPolicy(
            dataset="rel-ratebeer",
            task="user-place-liked_pairwise",
            primary_metric="roc_auc",
            metric_direction="higher",
        ),
    )

    assert decision.selected_program_id == "baseline"
    assert any(
        "incomplete_provenance_contract"
        in rejected.rejection_reasons
        for rejected in decision.rejected_candidates
    )


def test_write_canonical_validation_csv(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        aggregate_csv(aggregate_row()),
    )
    report = normalize_validation_artifact(path)
    output = tmp_path / "canonical.csv"

    with output.open("w", encoding="utf-8", newline="") as handle:
        write_canonical_validation_csv(
            report.normalized_records,
            handle,
        )

    assert output.read_text(encoding="utf-8").startswith(
        "dataset,task,program_id,split,primary_metric"
    )


def test_audit_reports_unsupported_realistic_schema(
    tmp_path: Path,
) -> None:
    path = write_csv(
        tmp_path,
        (
            "task,selected_candidate,primary_metric,selected_score\n"
            "rel-ratebeer_user-place-liked_pairwise,fdhg,roc_auc,0.7\n"
        ),
    )

    audit = audit_validation_sources((path,))

    assert audit[0].adapter_supported is False
    assert audit[0].program_identity_explicit is False
