from __future__ import annotations

import builtins
from pathlib import Path

from fdhg.compiler.candidate_safety import (
    ExplicitLoweringEvidence,
    audit_leakage_safety,
    audit_lowering_provenance,
    audit_temporal_safety,
    build_candidate_safety_audit_report,
    write_audit_csv,
)
from fdhg.compiler.materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    MaterializationAuditRow,
    PrimitiveMaterializationStep,
)
from fdhg.compiler.passthrough_provenance import (
    PassthroughBindingEvidence,
    PassthroughProvenanceReport,
)
from fdhg.compiler.validation_export import (
    inspect_candidate_safety_evidence,
)


def step(
    primitive_id: str = "temporal::count",
    *,
    lowering_mode: LoweringMode = LoweringMode.GENERATE,
    source_event_time_col: str | None = "event_time",
    target_time_col: str = "timestamp",
    cutoff_operator: str = "<",
    window_days: int | None = 30,
    temporally_safe: bool = True,
    materializable: bool = True,
    output_columns: tuple[str, ...] = ("f_count",),
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation="window_count",
        lowering_mode=lowering_mode,
        pairwise_role="left",
        source_table="events",
        source_group_key="user_id",
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=source_event_time_col,
        target_key="user_id",
        target_left_key=None,
        target_right_key=None,
        target_time_col=target_time_col,
        related_col="item_id",
        window_days=window_days,
        cutoff_operator=cutoff_operator,
        output_columns=output_columns,
        materializable=materializable,
        temporally_safe=temporally_safe,
        requires_external_provider=False,
    )


def plan(steps: tuple[PrimitiveMaterializationStep, ...]):
    return CandidateMaterializationPlan(
        program_id="program",
        steps=steps,
        audit_rows=tuple(
            MaterializationAuditRow(
                program_id="program",
                primitive_id=item.primitive_id,
                lowering_mode=item.lowering_mode,
                pairwise_role=item.pairwise_role,
                source_table=item.source_table,
                source_event_time_col=item.source_event_time_col,
                logical_temporal_predicate=None,
                required_cutoff_operator="<",
                configured_cutoff_operator=item.cutoff_operator,
                temporally_safe=item.temporally_safe,
                materializable=item.materializable,
                requires_external_provider=False,
                errors=(),
                warnings=(),
            )
            for item in steps
        ),
        materializable=all(item.materializable for item in steps),
        temporally_safe=all(item.temporally_safe for item in steps),
        requires_external_provider=False,
    )


def test_temporally_safe_native_primitive() -> None:
    audit = audit_temporal_safety(
        dataset="rel-example",
        task="task",
        plan=plan((step(),)),
    )

    assert audit.passed


def test_missing_timestamp_requirement() -> None:
    audit = audit_temporal_safety(
        dataset="rel-example",
        task="task",
        plan=plan((step(source_event_time_col=None),)),
    )

    assert "missing_timestamp_requirement" in audit.rows[0].rejection_reason


def test_future_event_access() -> None:
    audit = audit_temporal_safety(
        dataset="rel-example",
        task="task",
        plan=plan((step(cutoff_operator="<="),)),
    )

    assert "future_event_access" in audit.rows[0].rejection_reason


def test_invalid_as_of_direction() -> None:
    audit = audit_temporal_safety(
        dataset="rel-example",
        task="task",
        plan=plan((step(cutoff_operator=">"),)),
    )

    assert "invalid_as_of_direction" in audit.rows[0].rejection_reason


def test_leakage_through_label_column() -> None:
    audit = audit_leakage_safety(
        dataset="rel-example",
        task="task",
        program_id="program",
        feature_columns=("label", "f_count"),
        label_col="label",
    )

    assert not audit.passed
    assert "label_column_in_features" in audit.rows[0].rejection_reason


def test_leakage_through_surrogate_key() -> None:
    audit = audit_leakage_safety(
        dataset="rel-example",
        task="task",
        program_id="program",
        feature_columns=("__row_id", "f_count"),
        label_col="label",
        surrogate_key_columns=("__row_id",),
    )

    assert any("surrogate_key_in_features" in row.rejection_reason for row in audit.rows)


def test_target_aware_aggregate_without_cross_fitting() -> None:
    audit = audit_leakage_safety(
        dataset="rel-example",
        task="task",
        program_id="program",
        feature_columns=("label_mean",),
        label_col="label",
        target_aggregate_columns=("label_mean",),
        cross_fitted_target_aggregates=False,
    )

    assert not audit.passed


def test_cross_fitted_target_aggregate_accepted() -> None:
    audit = audit_leakage_safety(
        dataset="rel-example",
        task="task",
        program_id="program",
        feature_columns=("label_mean",),
        label_col="label",
        target_aggregate_columns=("label_mean",),
        cross_fitted_target_aggregates=True,
    )

    assert audit.passed


def test_task_mismatched_evidence_rejected() -> None:
    evidence = ExplicitLoweringEvidence(
        dataset="rel-ratebeer",
        task="user-count",
        program_id="program",
        primitive_id="external::x",
        source_table="events",
        source_column="a",
        output_column="b",
        status="proven",
        evidence_location="manifest.csv",
    )
    audit = audit_lowering_provenance(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        plan=plan((step("external::x", lowering_mode=LoweringMode.EXTERNAL),)),
        explicit_evidence=(evidence,),
    )

    assert not audit.passed


def test_complete_native_lowering_provenance() -> None:
    audit = audit_lowering_provenance(
        dataset="rel-example",
        task="task",
        plan=plan((step(),)),
    )

    assert audit.passed


def test_complete_passthrough_provenance() -> None:
    report = PassthroughProvenanceReport(
        program_id="program",
        passthrough_step_count=1,
        binding_evidence=(
            PassthroughBindingEvidence(
                program_id="program",
                primitive_id="baseline::count",
                source_column="src",
                output_column="out",
                evidence_kind="manifest",
                evidence_location="manifest.csv",
                status="proven",
            ),
        ),
        proven_bindings=(),
        unresolved_primitive_ids=(),
        conflicting_primitive_ids=(),
        complete=True,
    )
    audit = audit_lowering_provenance(
        dataset="rel-example",
        task="task",
        plan=plan((step("baseline::count", lowering_mode=LoweringMode.PASSTHROUGH),)),
        passthrough_report=report,
    )

    assert audit.passed


def test_partial_passthrough_provenance_rejected() -> None:
    report = PassthroughProvenanceReport(
        program_id="program",
        passthrough_step_count=1,
        binding_evidence=(
            PassthroughBindingEvidence(
                program_id="program",
                primitive_id="baseline::count",
                source_column="src",
                output_column=None,
                evidence_kind="schema",
                evidence_location="schema",
                status="partial",
            ),
        ),
        proven_bindings=(),
        unresolved_primitive_ids=("baseline::count",),
        conflicting_primitive_ids=(),
        complete=False,
    )
    audit = audit_lowering_provenance(
        dataset="rel-example",
        task="task",
        plan=plan((step("baseline::count", lowering_mode=LoweringMode.PASSTHROUGH),)),
        passthrough_report=report,
    )

    assert not audit.passed


def test_conflicting_provenance_rejected() -> None:
    evidence = (
        ExplicitLoweringEvidence("d", "t", "program", "x", "s", "a", "o", "proven", "a"),
        ExplicitLoweringEvidence("d", "t", "program", "x", "s", "b", "o", "proven", "b"),
    )
    try:
        audit_lowering_provenance(
            dataset="d",
            task="t",
            plan=plan((step("x", lowering_mode=LoweringMode.EXTERNAL),)),
            explicit_evidence=evidence,
        )
    except ValueError as exc:
        assert "conflicting provenance evidence" in str(exc)
    else:
        raise AssertionError("expected conflict")


def test_unknown_primitive_rejected() -> None:
    audit = audit_lowering_provenance(
        dataset="rel-example",
        task="task",
        plan=plan((step("unknown", lowering_mode=LoweringMode.UNSUPPORTED),)),
    )

    assert not audit.passed


def test_deterministic_row_ordering() -> None:
    audit = audit_temporal_safety(
        dataset="rel-example",
        task="task",
        plan=plan((step("z"), step("a"))),
    )

    assert [row.primitive_id for row in audit.rows] == ["a", "z"]


def test_no_mutation() -> None:
    original = plan((step(),))
    before = original.steps

    audit_temporal_safety(dataset="d", task="t", plan=original)

    assert original.steps == before


def test_no_filesystem_writes_in_pure_apis(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("unexpected write")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "mkdir", fail)

    assert audit_temporal_safety(dataset="d", task="t", plan=plan((step(),))).passed


def test_ratebeer_pairwise_remains_provenance_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "target_with_dfs_agg_train.parquet").write_text("")
    (root / "target_with_dfs_agg_val.parquet").write_text("")

    safety = inspect_candidate_safety_evidence(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        program_id="baseline_plus_pairwise_temporal",
        artifact_dir=root,
    )

    assert safety.provenance_complete is False


def test_validation_export_compatibility(tmp_path: Path) -> None:
    audit = audit_temporal_safety(dataset="d", task="t", plan=plan((step(),)))
    path = tmp_path / "temporal_safety_audit.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        write_audit_csv(audit.rows, handle)

    assert "audit_type" in path.read_text(encoding="utf-8").splitlines()[0]
