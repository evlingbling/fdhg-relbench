from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path

import pytest

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.programs import CandidateProgram
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates
from fdhg.compiler.selection import (
    CandidateSelectionPolicy,
    CandidateValidationResult,
    ProgramScore,
    load_candidate_validation_results,
    select_program,
    select_candidate_program,
)


DATASET = "rel-example"
TASK = "example-task"


def program(
    program_id: str,
    *,
    families: tuple[str, ...] = ("baseline", "temporal"),
    n_primitives: int = 3,
) -> CandidateProgram:
    return CandidateProgram(
        program_id=program_id,
        primitive_ids=[
            f"{program_id}::primitive::{index}"
            for index in range(n_primitives)
        ],
        families=list(families),
        description=program_id,
    )


def programs() -> tuple[CandidateProgram, ...]:
    return (
        program(
            "baseline",
            families=("baseline",),
            n_primitives=2,
        ),
        program("fdhg_a", n_primitives=4),
        program("fdhg_b", n_primitives=5),
    )


def policy(
    *,
    metric_direction: str = "higher",
    min_improvement: float = 0.0,
) -> CandidateSelectionPolicy:
    return CandidateSelectionPolicy(
        dataset=DATASET,
        task=TASK,
        primary_metric="roc_auc",
        metric_direction=metric_direction,
        min_improvement=min_improvement,
    )


def result(
    program_id: str,
    score: float | None,
    *,
    baseline_score: float | None = 0.70,
    baseline_program_id: str | None = "baseline",
    split: str = "validation",
    dataset: str = DATASET,
    task: str = TASK,
    metric: str = "roc_auc",
    direction: str = "higher",
    n_features: int | None = None,
    eligible: bool = True,
    reasons: tuple[str, ...] = (),
    location: str | None = None,
    materializable: bool = True,
    leakage_safe: bool = True,
    temporally_safe: bool = True,
    provenance_complete: bool = True,
) -> CandidateValidationResult:
    return CandidateValidationResult(
        dataset=dataset,
        task=task,
        program_id=program_id,
        primary_metric=metric,
        metric_direction=direction,
        validation_score=score,
        baseline_program_id=baseline_program_id,
        baseline_score=baseline_score,
        split=split,
        n_features=n_features,
        eligible=eligible,
        rejection_reasons=reasons,
        evidence_location=location or f"validation:{program_id}",
        materializable=materializable,
        leakage_safe=leakage_safe,
        temporally_safe=temporally_safe,
        provenance_complete=provenance_complete,
    )


def baseline(
    score: float = 0.70,
    *,
    direction: str = "higher",
    metric: str = "roc_auc",
) -> CandidateValidationResult:
    return result(
        "baseline",
        score,
        baseline_score=score,
        direction=direction,
        metric=metric,
        n_features=2,
    )


def rejected_reasons(decision, program_id: str) -> tuple[str, ...]:
    for rejected in decision.rejected_candidates:
        if rejected.program_id == program_id:
            return rejected.rejection_reasons
    raise AssertionError(f"no rejected candidate {program_id}")


def test_higher_is_better_selection() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.72, n_features=4),
        ],
        policy(),
    )

    assert decision.selected_program_id == "fdhg_a"
    assert decision.improvement_over_baseline == pytest.approx(0.02)
    assert not decision.fallback_occurred


def test_lower_is_better_selection() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.40, direction="lower", metric="log_loss"),
            result(
                "fdhg_a",
                0.35,
                baseline_score=0.40,
                direction="lower",
                metric="log_loss",
                n_features=4,
            ),
        ],
        CandidateSelectionPolicy(
            dataset=DATASET,
            task=TASK,
            primary_metric="log_loss",
            metric_direction="lower",
        ),
    )

    assert decision.selected_program_id == "fdhg_a"
    assert decision.improvement_over_baseline == pytest.approx(0.05)


def test_best_of_multiple_fdhg_candidates() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.72, n_features=4),
            result("fdhg_b", 0.75, n_features=5),
        ],
        policy(),
    )

    assert decision.selected_program_id == "fdhg_b"
    assert [
        candidate.program_id
        for candidate in decision.ranked_candidates
    ] == ["fdhg_b", "fdhg_a"]


def test_dfs_fallback_when_every_fdhg_candidate_is_worse() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.69, n_features=4),
            result("fdhg_b", 0.68, n_features=5),
        ],
        policy(),
    )

    assert decision.selected_program_id == "baseline"
    assert decision.fallback_occurred
    assert (
        decision.fallback_reason
        == "best_fdhg_candidate_does_not_improve_baseline"
    )


def test_min_improvement_threshold() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.705, n_features=4),
        ],
        policy(min_improvement=0.01),
    )

    assert decision.selected_program_id == "baseline"
    assert decision.fallback_occurred


def test_exact_tie_falls_back_to_baseline() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.70, n_features=4),
        ],
        policy(),
    )

    assert decision.selected_program_id == "baseline"
    assert decision.fallback_occurred


def test_positive_min_improvement_allows_exact_threshold() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.72, n_features=4),
        ],
        policy(min_improvement=0.02),
    )

    assert decision.selected_program_id == "fdhg_a"


def test_slightly_below_threshold_falls_back() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.719999, n_features=4),
        ],
        policy(min_improvement=0.02),
    )

    assert decision.selected_program_id == "baseline"


def test_slightly_above_threshold_selects_candidate() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.720001, n_features=4),
        ],
        policy(min_improvement=0.02),
    )

    assert decision.selected_program_id == "fdhg_a"


def test_lower_is_better_exact_tie_falls_back() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.40, direction="lower", metric="log_loss"),
            result(
                "fdhg_a",
                0.40,
                baseline_score=0.40,
                direction="lower",
                metric="log_loss",
                n_features=4,
            ),
        ],
        CandidateSelectionPolicy(
            dataset=DATASET,
            task=TASK,
            primary_metric="log_loss",
            metric_direction="lower",
        ),
    )

    assert decision.selected_program_id == "baseline"


def test_lower_is_better_threshold_semantics() -> None:
    lower_policy = CandidateSelectionPolicy(
        dataset=DATASET,
        task=TASK,
        primary_metric="log_loss",
        metric_direction="lower",
        min_improvement=0.02,
    )
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.40, direction="lower", metric="log_loss"),
            result(
                "fdhg_a",
                0.38,
                baseline_score=0.40,
                direction="lower",
                metric="log_loss",
                n_features=4,
            ),
        ],
        lower_policy,
    )

    assert decision.selected_program_id == "fdhg_a"


def test_deterministic_tie_breaking() -> None:
    tied_programs = (
        program(
            "baseline",
            families=("baseline",),
            n_primitives=2,
        ),
        program("fdhg_wide", n_primitives=5),
        program("fdhg_narrow_b", n_primitives=4),
        program("fdhg_narrow_a", n_primitives=4),
    )
    decision = select_candidate_program(
        tied_programs,
        [
            baseline(0.70),
            result("fdhg_wide", 0.75, n_features=5),
            result("fdhg_narrow_b", 0.75, n_features=4),
            result("fdhg_narrow_a", 0.75, n_features=4),
        ],
        policy(),
    )

    assert decision.selected_program_id == "fdhg_narrow_a"
    assert [
        candidate.program_id
        for candidate in decision.ranked_candidates
    ] == [
        "fdhg_narrow_a",
        "fdhg_narrow_b",
        "fdhg_wide",
    ]


def test_missing_validation_score_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [baseline(0.70), result("fdhg_a", None)],
        policy(),
    )

    assert "missing_validation_score" in rejected_reasons(
        decision,
        "fdhg_a",
    )


def test_infeasible_candidate_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.75, materializable=False),
        ],
        policy(),
    )

    assert "infeasible" in rejected_reasons(decision, "fdhg_a")


def test_leakage_safety_rejection() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.75, leakage_safe=False),
        ],
        policy(),
    )

    assert "leakage_violation" in rejected_reasons(
        decision,
        "fdhg_a",
    )


def test_incomplete_provenance_rejection() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.75, provenance_complete=False),
        ],
        policy(),
    )

    assert "incomplete_provenance_contract" in rejected_reasons(
        decision,
        "fdhg_a",
    )


def test_missing_safety_evidence_rejected() -> None:
    unsafe_unknown = CandidateValidationResult(
        dataset=DATASET,
        task=TASK,
        program_id="fdhg_a",
        primary_metric="roc_auc",
        metric_direction="higher",
        validation_score=0.75,
        split="validation",
        n_features=4,
        evidence_location="validation:fdhg_a",
    )
    decision = select_candidate_program(
        programs(),
        [baseline(0.70), unsafe_unknown],
        policy(),
    )

    reasons = rejected_reasons(decision, "fdhg_a")
    assert "unknown_materialization_feasibility" in reasons
    assert "unknown_leakage_safety" in reasons
    assert "unknown_temporal_safety" in reasons
    assert "unknown_provenance_completeness" in reasons


def test_test_split_evidence_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.99, split="held-out test"),
        ],
        policy(),
    )

    assert "test_or_final_split_evidence" in rejected_reasons(
        decision,
        "fdhg_a",
    )
    assert decision.selected_program_id == "baseline"


def test_task_mismatched_evidence_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.75, task="other-task"),
        ],
        policy(),
    )

    assert "task_mismatch" in rejected_reasons(decision, "fdhg_a")


def test_mismatched_records_are_order_independent() -> None:
    mismatched = [
        result(
            "fdhg_a",
            0.75,
            task="other-task",
            location="z-location",
        ),
        result(
            "fdhg_a",
            0.75,
            metric="accuracy",
            location="a-location",
        ),
    ]

    first = select_candidate_program(
        programs(),
        [baseline(0.70)] + mismatched,
        policy(),
    )
    second = select_candidate_program(
        programs(),
        [baseline(0.70)] + list(reversed(mismatched)),
        policy(),
    )

    first_rejection = next(
        candidate
        for candidate in first.rejected_candidates
        if candidate.program_id == "fdhg_a"
    )
    second_rejection = next(
        candidate
        for candidate in second.rejected_candidates
        if candidate.program_id == "fdhg_a"
    )
    assert first_rejection == second_rejection
    assert first_rejection.evidence_location == (
        "a-location|z-location"
    )


def test_duplicate_program_result_conflict() -> None:
    with pytest.raises(
        ValueError,
        match="conflicting duplicate validation results",
    ):
        select_candidate_program(
            programs(),
            [
                baseline(0.70),
                result("fdhg_a", 0.75),
                result("fdhg_a", 0.76),
            ],
            policy(),
        )


def test_conflicting_baseline_program_id_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result(
                "fdhg_a",
                0.75,
                baseline_program_id="other_baseline",
            ),
        ],
        policy(),
    )

    assert "baseline_program_id_conflict" in rejected_reasons(
        decision,
        "fdhg_a",
    )


def test_conflicting_baseline_score_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.75, baseline_score=0.71),
        ],
        policy(),
    )

    assert "baseline_score_conflict" in rejected_reasons(
        decision,
        "fdhg_a",
    )


def test_consistent_baseline_metadata_is_accepted() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result(
                "fdhg_a",
                0.75,
                baseline_program_id="baseline",
                baseline_score=0.70,
            ),
        ],
        policy(),
    )

    assert decision.selected_program_id == "fdhg_a"


def test_blank_baseline_metadata_is_supported() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result(
                "fdhg_a",
                0.75,
                baseline_program_id=None,
                baseline_score=None,
            ),
        ],
        policy(),
    )

    assert decision.selected_program_id == "fdhg_a"


def test_eligible_unknown_program_is_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("unknown_candidate", 0.99),
        ],
        policy(),
    )

    assert "unknown_program_id" in rejected_reasons(
        decision,
        "unknown_candidate",
    )


def test_ineligible_unknown_program_is_rejected() -> None:
    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result(
                "unknown_candidate",
                0.99,
                eligible=False,
                reasons=("manual_rejection",),
            ),
        ],
        policy(),
    )

    reasons = rejected_reasons(decision, "unknown_candidate")
    assert "unknown_program_id" in reasons
    assert "ineligible" in reasons
    assert "manual_rejection" in reasons


def test_no_mutation_of_inputs() -> None:
    input_programs = list(programs())
    input_results = [
        baseline(0.70),
        result("fdhg_a", 0.75),
    ]
    original_programs = tuple(input_programs)
    original_results = tuple(input_results)

    select_candidate_program(
        input_programs,
        input_results,
        policy(),
    )

    assert tuple(input_programs) == original_programs
    assert tuple(input_results) == original_results


def test_no_filesystem_writes_in_pure_api(monkeypatch) -> None:
    def fail_open(*args, **kwargs):
        raise AssertionError("pure selector opened a file")

    monkeypatch.setattr(builtins, "open", fail_open)
    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr(Path, "write_text", fail_open)
    monkeypatch.setattr(Path, "mkdir", fail_open)

    decision = select_candidate_program(
        programs(),
        [
            baseline(0.70),
            result("fdhg_a", 0.75),
        ],
        policy(),
    )

    assert decision.selected_program_id == "fdhg_a"


def test_ratebeer_regression_fixture_falls_back_to_baseline() -> None:
    ratebeer_policy = CandidateSelectionPolicy(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        primary_metric="roc_auc",
        metric_direction="higher",
    )
    ratebeer_programs = (
        program(
            "baseline",
            families=("baseline",),
            n_primitives=15,
        ),
        program(
            "baseline_plus_pairwise_temporal",
            n_primitives=30,
        ),
    )

    decision = select_candidate_program(
        ratebeer_programs,
        [
            replace(
                baseline(0.72),
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                n_features=15,
            ),
            result(
                "baseline_plus_pairwise_temporal",
                0.71,
                baseline_score=0.72,
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                n_features=30,
            ),
        ],
        ratebeer_policy,
    )

    assert decision.selected_program_id == "baseline"
    assert decision.fallback_occurred


def test_ratebeer_regression_fixture_selects_better_second_candidate() -> None:
    ratebeer_policy = CandidateSelectionPolicy(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        primary_metric="roc_auc",
        metric_direction="higher",
    )
    ratebeer_programs = (
        program(
            "baseline",
            families=("baseline",),
            n_primitives=15,
        ),
        program(
            "baseline_plus_pairwise_temporal",
            n_primitives=30,
        ),
        program(
            "baseline_plus_pair_left_temporal",
            n_primitives=22,
        ),
    )

    decision = select_candidate_program(
        ratebeer_programs,
        [
            replace(
                baseline(0.72),
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                n_features=15,
            ),
            result(
                "baseline_plus_pairwise_temporal",
                0.71,
                baseline_score=0.72,
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                n_features=30,
            ),
            result(
                "baseline_plus_pair_left_temporal",
                0.74,
                baseline_score=0.72,
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                n_features=22,
            ),
        ],
        ratebeer_policy,
    )

    assert decision.selected_program_id == (
        "baseline_plus_pair_left_temporal"
    )
    assert not decision.fallback_occurred


def test_real_compiler_candidate_family_complexity_is_stable() -> None:
    task_spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=Path("configs/reproduction/tasks.yaml"),
        semantics_config=Path(
            "configs/reproduction/task_semantics.yaml"
        ),
    )
    compiled = build_candidate_program(task_spec)
    real_programs = tuple(build_default_candidates(compiled))
    candidate_ids = {
        candidate.program_id for candidate in real_programs
    }

    assert "baseline" in candidate_ids
    assert "baseline_plus_pairwise_temporal" in candidate_ids

    decision = select_candidate_program(
        real_programs,
        [
            result(
                "baseline",
                0.70,
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                n_features=15,
            ),
            result(
                "baseline_plus_pairwise_temporal",
                0.75,
                baseline_score=0.70,
                dataset="rel-ratebeer",
                task="user-place-liked_pairwise",
                n_features=30,
            ),
        ],
        CandidateSelectionPolicy(
            dataset="rel-ratebeer",
            task="user-place-liked_pairwise",
            primary_metric="roc_auc",
            metric_direction="higher",
        ),
    )

    assert decision.selected_program_id == (
        "baseline_plus_pairwise_temporal"
    )


def test_program_score_selector_backward_compatibility() -> None:
    scores = [
        ProgramScore(
            program_id="wide",
            result_variant="wide",
            n_runs=4,
            n_features_mean=10.0,
            primary_metric="roc_auc",
            primary_mean=0.80,
            primary_std=0.01,
            secondary_metric=None,
            secondary_mean=None,
        ),
        ProgramScore(
            program_id="narrow",
            result_variant="narrow",
            n_runs=4,
            n_features_mean=5.0,
            primary_metric="roc_auc",
            primary_mean=0.80,
            primary_std=0.01,
            secondary_metric=None,
            secondary_mean=None,
        ),
    ]

    selected = select_program(scores, metric_direction="higher")

    assert selected.program_id == "narrow"


def test_canonical_validation_loader_requires_explicit_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.csv"
    path.write_text(
        "task,program_id,score\n"
        "example-task,fdhg_a,0.75\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing canonical validation columns",
    ):
        load_candidate_validation_results(path)


def test_canonical_validation_loader_reads_task_scoped_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.csv"
    path.write_text(
        "\n".join([
            (
                "dataset,task,program_id,split,primary_metric,"
                "metric_direction,score,n_features,eligible,"
                "rejection_reason,evidence_location,"
                "materializable,leakage_safe,temporally_safe,"
                "provenance_complete,baseline_program_id,"
                "baseline_score"
            ),
            (
                "rel-example,example-task,fdhg_a,validation,"
                "roc_auc,higher,0.75,4,true,,validation.csv,"
                "true,true,true,true,baseline,0.70"
            ),
        ]),
        encoding="utf-8",
    )

    records = load_candidate_validation_results(path)

    assert records == (
        CandidateValidationResult(
            dataset="rel-example",
            task="example-task",
            program_id="fdhg_a",
            split="validation",
            primary_metric="roc_auc",
            metric_direction="higher",
            validation_score=0.75,
            baseline_program_id="baseline",
            baseline_score=0.70,
            n_features=4,
            eligible=True,
            rejection_reasons=(),
            evidence_location="validation.csv",
            materializable=True,
            leakage_safe=True,
            temporally_safe=True,
            provenance_complete=True,
        ),
    )


def canonical_csv(
    *,
    score: str = "0.75",
    n_features: str = "4",
    eligible: str = "true",
    evidence_location: str = "validation.csv",
    split: str = "validation",
    materializable: str = "true",
    leakage_safe: str = "true",
    temporally_safe: str = "true",
    provenance_complete: str = "true",
    extra_row: str | None = None,
) -> str:
    rows = [
        (
            "dataset,task,program_id,split,primary_metric,"
            "metric_direction,score,n_features,eligible,"
            "rejection_reason,evidence_location,"
            "materializable,leakage_safe,temporally_safe,"
            "provenance_complete,baseline_program_id,"
            "baseline_score"
        ),
        (
            "rel-example,example-task,fdhg_a,"
            f"{split},roc_auc,higher,{score},{n_features},"
            f"{eligible},,{evidence_location},{materializable},"
            f"{leakage_safe},{temporally_safe},"
            f"{provenance_complete},baseline,0.70"
        ),
    ]
    if extra_row is not None:
        rows.append(extra_row)
    return "\n".join(rows)


def write_canonical(
    tmp_path: Path,
    content: str,
) -> Path:
    path = tmp_path / "validation.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_canonical_loader_rejects_missing_safety_columns(
    tmp_path: Path,
) -> None:
    path = write_canonical(
        tmp_path,
        (
            "dataset,task,program_id,split,primary_metric,"
            "metric_direction,score,n_features,eligible,"
            "rejection_reason,evidence_location\n"
            "rel-example,example-task,fdhg_a,validation,"
            "roc_auc,higher,0.75,4,true,,validation.csv\n"
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing canonical validation columns",
    ):
        load_candidate_validation_results(path)


def test_canonical_loader_rejects_malformed_boolean(
    tmp_path: Path,
) -> None:
    path = write_canonical(
        tmp_path,
        canonical_csv(materializable="maybe"),
    )

    with pytest.raises(ValueError, match="Expected boolean"):
        load_candidate_validation_results(path)


@pytest.mark.parametrize("score", ["nan", "inf", "-inf"])
def test_canonical_loader_rejects_nan_and_infinity(
    tmp_path: Path,
    score: str,
) -> None:
    path = write_canonical(tmp_path, canonical_csv(score=score))

    with pytest.raises(ValueError, match="Expected finite float"):
        load_candidate_validation_results(path)


def test_canonical_loader_rejects_negative_feature_count(
    tmp_path: Path,
) -> None:
    path = write_canonical(
        tmp_path,
        canonical_csv(n_features="-1"),
    )

    with pytest.raises(
        ValueError,
        match="Expected non-negative integer",
    ):
        load_candidate_validation_results(path)


def test_canonical_loader_rejects_duplicate_rows(
    tmp_path: Path,
) -> None:
    duplicate = (
        "rel-example,example-task,fdhg_a,validation,"
        "roc_auc,higher,0.75,4,true,,validation.csv,"
        "true,true,true,true,baseline,0.70"
    )
    path = write_canonical(
        tmp_path,
        canonical_csv(extra_row=duplicate),
    )

    with pytest.raises(ValueError, match="duplicate validation rows"):
        load_candidate_validation_results(path)


def test_canonical_loader_rejects_missing_evidence_location(
    tmp_path: Path,
) -> None:
    path = write_canonical(
        tmp_path,
        canonical_csv(evidence_location=""),
    )

    with pytest.raises(ValueError, match="Expected non-empty value"):
        load_candidate_validation_results(path)


def test_test_split_loaded_but_rejected_by_selector(
    tmp_path: Path,
) -> None:
    path = write_canonical(
        tmp_path,
        canonical_csv(split="paper-final-test"),
    )
    records = load_candidate_validation_results(path)

    decision = select_candidate_program(
        programs(),
        [baseline(0.70), *records],
        policy(),
    )

    assert "test_or_final_split_evidence" in rejected_reasons(
        decision,
        "fdhg_a",
    )
