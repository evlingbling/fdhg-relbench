from __future__ import annotations

import builtins
from copy import deepcopy
import importlib.util
from pathlib import Path
import pathlib

import pytest

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    MaterializationAuditRow,
    PrimitiveMaterializationStep,
    plan_candidate_materialization,
)
from fdhg.compiler.passthrough_provenance import (
    CONFLICTING,
    MISSING,
    PARTIAL,
    PROVEN,
    PassthroughBindingEvidence,
    build_passthrough_provenance_report,
    provenance_report_to_contract,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates


ACTIVITY_PRODUCT = "f_pairtmp__user_place_activity_product"
ACTIVITY_RATIO = "f_pairtmp__user_place_activity_ratio"
UNSET = object()


def step(
    primitive_id: str,
    mode: LoweringMode = LoweringMode.PASSTHROUGH,
) -> PrimitiveMaterializationStep:
    return PrimitiveMaterializationStep(
        program_id="program",
        primitive_id=primitive_id,
        operation="count",
        lowering_mode=mode,
        pairwise_role=None,
        source_table=None,
        source_group_key=None,
        source_left_key=None,
        source_right_key=None,
        source_event_time_col=None,
        target_key=None,
        target_left_key=None,
        target_right_key=None,
        target_time_col="timestamp",
        related_col=None,
        window_days=None,
        cutoff_operator="<",
        output_columns=(),
        materializable=True,
        temporally_safe=True,
        requires_external_provider=False,
    )


def plan_with(
    steps: tuple[PrimitiveMaterializationStep, ...],
) -> CandidateMaterializationPlan:
    return CandidateMaterializationPlan(
        program_id="program",
        steps=steps,
        audit_rows=tuple(
            MaterializationAuditRow(
                program_id="program",
                primitive_id=item.primitive_id,
                lowering_mode=item.lowering_mode,
                pairwise_role=None,
                source_table=None,
                source_event_time_col=None,
                logical_temporal_predicate=None,
                required_cutoff_operator="<",
                configured_cutoff_operator="<",
                temporally_safe=True,
                materializable=True,
                requires_external_provider=False,
                errors=(),
                warnings=(),
            )
            for item in steps
        ),
        materializable=True,
        temporally_safe=True,
        requires_external_provider=False,
    )


def evidence(
    primitive_id: str,
    *,
    source_column=UNSET,
    output_column=UNSET,
    status: str = PROVEN,
    location: str = "manifest.csv:1",
    kind: str = "manifest",
) -> PassthroughBindingEvidence:
    return PassthroughBindingEvidence(
        program_id="program",
        primitive_id=primitive_id,
        source_column=(
            source_column
            if source_column is not UNSET
            else f"{primitive_id}_source"
        ),
        output_column=(
            output_column
            if output_column is not UNSET
            else f"{primitive_id}_output"
        ),
        evidence_kind=kind,
        evidence_location=location,
        status=status,
        notes=("stable",),
    )


def report(plan, records):
    return build_passthrough_provenance_report(
        plan,
        evidence_records=records,
    )


def test_all_proven_report_becomes_complete() -> None:
    plan = plan_with((step("p1"), step("p2")))
    result = report(plan, (evidence("p1"), evidence("p2")))
    assert result.complete is True
    assert result.unresolved_primitive_ids == ()
    assert result.conflicting_primitive_ids == ()


def test_one_missing_primitive_makes_report_incomplete() -> None:
    result = report(
        plan_with((step("p1"), step("p2"))),
        (evidence("p1"),),
    )
    assert result.complete is False
    assert result.unresolved_primitive_ids == ("p2",)
    assert result.binding_evidence[1].status == MISSING
    assert [row.primitive_id for row in result.proven_bindings] == [
        "p1",
    ]


def test_one_partial_primitive_makes_report_incomplete() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence(
                "p1",
                source_column=None,
                output_column="out",
                status=PARTIAL,
            ),
        ),
    )
    assert result.complete is False
    assert result.unresolved_primitive_ids == ("p1",)
    assert result.proven_bindings == ()


def test_conflicting_evidence_makes_report_incomplete() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence("p1", source_column="a", output_column="x"),
            evidence(
                "p1",
                source_column="b",
                output_column="x",
                location="manifest.csv:2",
            ),
        ),
    )
    assert result.complete is False
    assert result.conflicting_primitive_ids == ("p1",)
    assert result.binding_evidence[0].status == CONFLICTING
    assert result.proven_bindings == ()


def test_duplicate_identical_evidence_is_deduplicated() -> None:
    record = evidence("p1")
    result = report(plan_with((step("p1"),)), (record, record))
    assert result.complete is True
    assert result.proven_bindings == (record,)


def test_conflicting_duplicate_evidence_is_detected() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence("p1", source_column="a", output_column="x"),
            evidence(
                "p1",
                source_column="a",
                output_column="y",
                location="manifest.csv:2",
            ),
        ),
    )
    assert result.conflicting_primitive_ids == ("p1",)


def test_source_only_partial_and_output_only_partial_are_merged() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence(
                "p1",
                source_column="source",
                output_column=None,
                status=PARTIAL,
                location="manifest.csv:1",
            ),
            evidence(
                "p1",
                source_column=None,
                output_column="output",
                status=PARTIAL,
                location="manifest.csv:2",
            ),
        ),
    )
    row = result.binding_evidence[0]
    assert row.status == PARTIAL
    assert row.source_column == "source"
    assert row.output_column == "output"
    assert result.complete is False
    assert result.conflicting_primitive_ids == ()


def test_partial_and_compatible_proven_are_merged_as_proven() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence(
                "p1",
                source_column="source",
                output_column=None,
                status=PARTIAL,
                location="manifest.csv:1",
            ),
            evidence(
                "p1",
                source_column="source",
                output_column="output",
                status=PROVEN,
                location="manifest.csv:2",
            ),
        ),
    )
    row = result.binding_evidence[0]
    assert row.status == PROVEN
    assert row.source_column == "source"
    assert row.output_column == "output"
    assert result.proven_bindings == (row,)
    assert result.complete is True


def test_partial_and_incompatible_proven_conflict() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence(
                "p1",
                source_column="other",
                output_column=None,
                status=PARTIAL,
                location="manifest.csv:1",
            ),
            evidence(
                "p1",
                source_column="source",
                output_column="output",
                status=PROVEN,
                location="manifest.csv:2",
            ),
        ),
    )
    assert result.binding_evidence[0].status == CONFLICTING
    assert result.conflicting_primitive_ids == ("p1",)
    assert result.proven_bindings == ()


def test_multiple_compatible_evidence_locations_are_stable() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence(
                "p1",
                source_column="source",
                output_column="output",
                status=PROVEN,
                location="z.csv:9",
                kind="manifest",
            ),
            evidence(
                "p1",
                source_column="source",
                output_column=None,
                status=PARTIAL,
                location="a.json:1",
                kind="json",
            ),
        ),
    )
    row = result.binding_evidence[0]
    assert row.evidence_kind == "json + manifest"
    assert row.evidence_location == "a.json:1 | z.csv:9"


def test_unknown_primitive_evidence_rejected() -> None:
    with pytest.raises(ValueError, match="unknown primitive_id"):
        report(plan_with((step("p1"),)), (evidence("missing"),))


def test_non_passthrough_primitive_evidence_rejected() -> None:
    with pytest.raises(ValueError, match="non-passthrough"):
        report(
            plan_with((step("p1", LoweringMode.GENERATE),)),
            (evidence("p1"),),
        )


def test_missing_source_column_cannot_be_proven() -> None:
    with pytest.raises(ValueError, match="requires source_column"):
        report(
            plan_with((step("p1"),)),
            (
                evidence(
                    "p1",
                    source_column=None,
                ),
            ),
        )


def test_missing_output_column_cannot_be_proven() -> None:
    with pytest.raises(ValueError, match="requires output_column"):
        report(
            plan_with((step("p1"),)),
            (
                evidence(
                    "p1",
                    output_column=None,
                ),
            ),
        )


def test_stable_plan_order_output() -> None:
    result = report(
        plan_with((step("p2"), step("p1"))),
        (evidence("p1"), evidence("p2")),
    )
    assert [
        item.primitive_id for item in result.binding_evidence
    ] == ["p2", "p1"]


def test_stable_evidence_ordering() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence(
                "p1",
                status=PARTIAL,
                location="z.csv:1",
                kind="z",
            ),
            evidence(
                "p1",
                status=PARTIAL,
                location="a.csv:1",
                kind="a",
            ),
        ),
    )
    assert result.binding_evidence[0].evidence_location == (
        "a.csv:1 | z.csv:1"
    )


def test_report_conversion_succeeds_only_when_complete() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence(
                "p1",
                source_column="source",
                output_column="output",
            ),
        ),
    )
    contract = provenance_report_to_contract(result)
    assert contract.source_columns == ("source",)
    assert contract.output_columns == ("output",)


def test_incomplete_report_cannot_become_a_contract() -> None:
    result = report(plan_with((step("p1"),)), ())
    with pytest.raises(ValueError, match="incomplete"):
        provenance_report_to_contract(result)


def test_conflict_report_cannot_become_a_contract() -> None:
    result = report(
        plan_with((step("p1"),)),
        (
            evidence("p1", source_column="a", output_column="x"),
            evidence(
                "p1",
                source_column="b",
                output_column="x",
                location="manifest.csv:2",
            ),
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        provenance_report_to_contract(result)


def test_evidence_records_are_deterministic() -> None:
    record = evidence("p1")
    first = report(plan_with((step("p1"),)), (record,))
    second = report(plan_with((step("p1"),)), (record,))
    assert first == second


def test_plan_is_not_mutated() -> None:
    plan = plan_with((step("p1"),))
    before = repr(plan)
    report(plan, (evidence("p1"),))
    assert repr(plan) == before


def test_evidence_input_is_not_mutated() -> None:
    records = [evidence("p1")]
    before = deepcopy(records)
    report(plan_with((step("p1"),)), records)
    assert records == before


def ratebeer_plan():
    spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=Path(
            "configs/reproduction/tasks.yaml"
        ),
        semantics_config=Path(
            "configs/reproduction/task_semantics.yaml"
        ),
    )
    compiled = build_candidate_program(spec)
    program = next(
        item
        for item in build_default_candidates(compiled)
        if item.program_id
        == "baseline_plus_pairwise_temporal"
    )
    return plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={
            "beer_ratings",
            "place_ratings",
        },
    )


def test_activity_product_ratio_remain_outside_the_contract() -> None:
    plan = ratebeer_plan()
    records = tuple(
        PassthroughBindingEvidence(
            program_id=plan.program_id,
            primitive_id=step.primitive_id,
            source_column=f"{step.primitive_id}::source",
            output_column=f"{step.primitive_id}::output",
            evidence_kind="test",
            evidence_location="test",
            status=PROVEN,
            notes=(),
        )
        for step in plan.steps
        if step.lowering_mode == LoweringMode.PASSTHROUGH
    )
    contract = provenance_report_to_contract(
        build_passthrough_provenance_report(
            plan,
            evidence_records=records,
        )
    )
    assert ACTIVITY_PRODUCT not in contract.output_columns
    assert ACTIVITY_RATIO not in contract.output_columns


def test_no_filesystem_writes() -> None:
    original_open = builtins.open

    def checked_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError("filesystem write attempted")
        return original_open(file, mode, *args, **kwargs)

    def fail(*args, **kwargs):
        raise AssertionError("filesystem write attempted")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        builtins,
        "open",
        checked_open,
    )
    monkeypatch.setattr(
        pathlib.Path,
        "write_text",
        fail,
    )
    monkeypatch.setattr(
        pathlib.Path,
        "write_bytes",
        fail,
    )
    monkeypatch.setattr(
        pathlib.Path,
        "mkdir",
        fail,
    )
    try:
        result = report(plan_with((step("p1"),)), (evidence("p1"),))
    finally:
        monkeypatch.undo()

    assert result.complete is True


def test_no_model_gpu_network_or_experiment_dependency() -> None:
    import fdhg.compiler.passthrough_provenance as module

    names = set(module.__dict__)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "torch" not in names
    assert "subprocess" not in names
    assert "requests" not in names


def inspector_module():
    path = (
        Path.cwd()
        / "scripts/compiler/"
        "inspect_ratebeer_passthrough_provenance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "inspect_ratebeer_passthrough_provenance",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_user_count_ratebeer_mappings_rejected_for_pairwise() -> None:
    inspector = inspector_module()
    records = inspector.explicit_records_from_scoped_mapping({
        "dataset": "rel-ratebeer",
        "task": "user-count",
        "primitive_column_bindings": {
            "baseline::count": ("f_beer_ratings_count",),
        },
    })
    assert records == ()


def test_arxiv_author_mappings_rejected_for_pairwise() -> None:
    inspector = inspector_module()
    records = inspector.explicit_records_from_scoped_mapping({
        "dataset": "rel-arxiv",
        "task": "author-category",
        "primitive_column_bindings": {
            "baseline::history::window_count_short": (
                "dfs::author::past_paper_count_30d",
            ),
        },
    })
    assert records == ()


def test_unrelated_pairwise_string_does_not_scope_mapping() -> None:
    inspector = inspector_module()
    records = inspector.explicit_records_from_scoped_mapping({
        "dataset": "rel-ratebeer",
        "task": "user-count",
        "notes": "mentions user-place-liked_pairwise elsewhere",
        "primitive_column_bindings": {
            "baseline::count": ("f_beer_ratings_count",),
        },
    })
    assert records == ()


def test_explicitly_pairwise_scoped_mapping_is_accepted() -> None:
    inspector = inspector_module()
    records = inspector.explicit_records_from_scoped_mapping({
        "dataset": "rel-ratebeer",
        "task": "user-place-liked_pairwise",
        "primitive_column_bindings": {
            "baseline::count": ("f_pairwise_explicit_count",),
        },
    })
    assert records == (
        {
            "primitive_id": "baseline::count",
            "source_column": "f_pairwise_explicit_count",
            "output_column": "f_pairwise_explicit_count",
        },
    )


def test_existing_backend_without_task_registration_yields_no_evidence() -> None:
    inspector = inspector_module()
    plan = ratebeer_plan()
    evidence_records = []
    inspector.inspect_historical_python(
        plan=plan,
        commit="local",
        path="src/fdhg/compiler/existing_backend.py",
        text=Path("src/fdhg/compiler/existing_backend.py").read_text(
            encoding="utf-8"
        ),
        passthrough_ids=tuple(
            step.primitive_id
            for step in plan.steps
            if step.lowering_mode == LoweringMode.PASSTHROUGH
        ),
        evidence=evidence_records,
    )
    assert evidence_records == []


def test_server_schema_presence_remains_partial_not_proven() -> None:
    inspector = inspector_module()
    plan = ratebeer_plan()
    evidence_records = []
    extra_columns = set()
    base_columns = (
        "user_id",
        "candidate_place_id",
        "timestamp",
        "label",
        "f_beer_ratings_count",
        "f_beer_ratings_aroma_mean",
        "f_beer_ratings_aroma_std",
        "f_beer_ratings_aroma_max",
        "f_beer_ratings_days_since_last",
    )
    for relative_path in inspector.DFS_SCHEMA_PATHS:
        inspector.inspect_schema_columns(
            plan=plan,
            relative_path=relative_path,
            columns=base_columns,
            evidence=evidence_records,
            extra_legacy_columns=extra_columns,
        )
    inspector.inspect_schema_columns(
        plan=plan,
        relative_path=inspector.TEMPORAL_SCHEMA_PATH,
        columns=base_columns
        + (
            ACTIVITY_PRODUCT,
            ACTIVITY_RATIO,
        ),
        evidence=evidence_records,
        extra_legacy_columns=extra_columns,
    )
    report = build_passthrough_provenance_report(
        plan,
        evidence_records=tuple(evidence_records),
    )
    statuses = [row.status for row in report.binding_evidence]
    assert statuses.count(PARTIAL) == 5
    assert statuses.count(MISSING) == 10
    assert report.proven_bindings == ()
    assert report.complete is False
    assert extra_columns == {ACTIVITY_PRODUCT, ACTIVITY_RATIO}
