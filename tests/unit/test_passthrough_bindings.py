from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import inspect

import pytest

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materializer import (
    CandidateMaterializationPlan,
    LoweringMode,
    MaterializationAuditRow,
    PrimitiveMaterializationStep,
    plan_candidate_materialization,
)
from fdhg.compiler.passthrough_bindings import (
    PassthroughBindingCode,
    PassthroughBindingError,
    PassthroughColumnBinding,
    passthrough_contract_from_declared_outputs,
    passthrough_contract_to_records,
    resolve_passthrough_bindings,
    validate_passthrough_rows,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates


def step(
    primitive_id: str,
    mode: LoweringMode,
    outputs: tuple[str, ...] = (),
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
        target_time_col=None,
        related_col=None,
        window_days=None,
        cutoff_operator=None,
        output_columns=outputs,
        materializable=(mode != LoweringMode.UNSUPPORTED),
        temporally_safe=(mode != LoweringMode.UNSUPPORTED),
        requires_external_provider=(mode == LoweringMode.EXTERNAL),
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
                configured_cutoff_operator=None,
                temporally_safe=item.temporally_safe,
                materializable=item.materializable,
                requires_external_provider=(
                    item.requires_external_provider
                ),
                errors=(),
                warnings=(),
            )
            for item in steps
        ),
        materializable=True,
        temporally_safe=True,
        requires_external_provider=any(
            item.requires_external_provider for item in steps
        ),
    )


def resolve(plan, explicit):
    return resolve_passthrough_bindings(
        plan,
        explicit_bindings=explicit,
    )


def assert_binding_error(code, plan, explicit) -> None:
    with pytest.raises(PassthroughBindingError) as exc:
        resolve(plan, explicit)
    assert exc.value.code == code
    assert "program_id=program" in str(exc.value)


def test_one_passthrough_primitive_one_binding() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("source", "out"),)},
    )
    assert contract.source_columns == ("source",)
    assert contract.output_columns == ("out",)


def test_one_passthrough_primitive_multiple_bindings() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("a", "a"), ("b", "b"))},
    )
    assert [binding.primitive_id for binding in contract.bindings] == [
        "p",
        "p",
    ]
    assert contract.output_columns == ("a", "b")


def test_multiple_primitives_preserve_plan_order() -> None:
    plan = plan_with((
        step("p2", LoweringMode.PASSTHROUGH),
        step("p1", LoweringMode.PASSTHROUGH),
    ))
    contract = resolve(
        plan,
        {"p1": (("a", "a"),), "p2": (("b", "b"),)},
    )
    assert contract.output_columns == ("b", "a")


def test_per_primitive_binding_order_preserved() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("b", "b"), ("a", "a"))},
    )
    assert contract.output_columns == ("b", "a")


def test_source_and_output_names_may_match() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("x", "x"),)},
    )
    assert contract.bindings[0].source_column == "x"
    assert contract.bindings[0].output_column == "x"


def test_explicit_rename_supported() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("legacy_x", "x"),)},
    )
    assert contract.source_columns == ("legacy_x",)
    assert contract.output_columns == ("x",)


def test_unknown_primitive_rejected() -> None:
    assert_binding_error(
        PassthroughBindingCode.UNKNOWN_PRIMITIVE_ID,
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("x", "x"),), "missing": (("y", "y"),)},
    )


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        (LoweringMode.GENERATE, PassthroughBindingCode.NON_PASSTHROUGH_PRIMITIVE),
        (LoweringMode.EXTERNAL, PassthroughBindingCode.NON_PASSTHROUGH_PRIMITIVE),
        (LoweringMode.UNSUPPORTED, PassthroughBindingCode.NON_PASSTHROUGH_PRIMITIVE),
    ],
)
def test_non_passthrough_primitive_binding_rejected(
    mode,
    code,
) -> None:
    assert_binding_error(
        code,
        plan_with((step("p", mode),)),
        {"p": (("x", "x"),)},
    )


def test_missing_required_binding_rejected() -> None:
    assert_binding_error(
        PassthroughBindingCode.MISSING_PASSTHROUGH_BINDING,
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {},
    )


def test_duplicate_source_binding_rejected() -> None:
    assert_binding_error(
        PassthroughBindingCode.DUPLICATE_SOURCE_BINDING,
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("x", "a"), ("x", "b"))},
    )


def test_duplicate_output_column_rejected() -> None:
    assert_binding_error(
        PassthroughBindingCode.DUPLICATE_OUTPUT_COLUMN,
        plan_with((
            step("p1", LoweringMode.PASSTHROUGH),
            step("p2", LoweringMode.PASSTHROUGH),
        )),
        {"p1": (("a", "x"),), "p2": (("b", "x"),)},
    )


@pytest.mark.parametrize(
    "explicit",
    [
        {"p": (("", "x"),)},
        {"p": (("x", ""),)},
    ],
)
def test_malformed_empty_columns_rejected(explicit) -> None:
    assert_binding_error(
        PassthroughBindingCode.INVALID_BINDING,
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        explicit,
    )


def test_deterministic_repeated_resolution() -> None:
    plan = plan_with((step("p", LoweringMode.PASSTHROUGH),))
    explicit = {"p": (("x", "x"),)}
    assert resolve(plan, explicit) == resolve(plan, explicit)


def test_plan_and_input_bindings_not_mutated() -> None:
    plan = plan_with((step("p", LoweringMode.PASSTHROUGH),))
    explicit = {"p": (("x", "x"),)}
    before_plan = repr(plan)
    before_explicit = deepcopy(explicit)
    resolve(plan, explicit)
    assert repr(plan) == before_plan
    assert explicit == before_explicit


def test_row_validation_success_and_none_preserved() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("x", "x"),)},
    )
    rows = [{"x": None}, {"x": 1}]
    before = deepcopy(rows)
    validate_passthrough_rows(contract, target_rows=rows)
    assert rows == before


def test_missing_source_column_reports_row_index() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("x", "x"),)},
    )
    with pytest.raises(PassthroughBindingError) as exc:
        validate_passthrough_rows(
            contract,
            target_rows=[{"x": 1}, {"y": 2}],
        )
    assert exc.value.code == PassthroughBindingCode.MISSING_SOURCE_COLUMN
    assert "row_index=1" in str(exc.value)
    assert "source_column=x" in str(exc.value)


def test_provenance_record_ordering_deterministic() -> None:
    contract = resolve(
        plan_with((step("p", LoweringMode.PASSTHROUGH),)),
        {"p": (("b", "out_b"), ("a", "out_a"))},
    )
    assert passthrough_contract_to_records(contract) == (
        {
            "program_id": "program",
            "primitive_id": "p",
            "source_column": "b",
            "output_column": "out_b",
        },
        {
            "program_id": "program",
            "primitive_id": "p",
            "source_column": "a",
            "output_column": "out_a",
        },
    )


def test_compatibility_helper_for_declared_outputs() -> None:
    contract = passthrough_contract_from_declared_outputs(
        plan_with((
            step(
                "p",
                LoweringMode.PASSTHROUGH,
                outputs=("x", "y"),
            ),
        ))
    )
    assert contract.bindings == (
        PassthroughColumnBinding("program", "p", "x", "x"),
        PassthroughColumnBinding("program", "p", "y", "y"),
    )


def ratebeer_plan():
    spec = load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=Path("configs/reproduction/tasks.yaml"),
        semantics_config=Path("configs/reproduction/task_semantics.yaml"),
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
        available_source_tables={"beer_ratings", "place_ratings"},
    )


RATEBEER_KNOWN_BASELINE_BINDINGS = {
    "baseline::count": (
        ("f_beer_ratings_count", "f_beer_ratings_count"),
    ),
    "baseline::numeric_mean": (
        (
            "f_beer_ratings_aroma_mean",
            "f_beer_ratings_aroma_mean",
        ),
        (
            "f_beer_ratings_aroma_mean__is_missing",
            "f_beer_ratings_aroma_mean__is_missing",
        ),
    ),
    "baseline::numeric_std": (
        (
            "f_beer_ratings_aroma_std",
            "f_beer_ratings_aroma_std",
        ),
        (
            "f_beer_ratings_aroma_std__is_missing",
            "f_beer_ratings_aroma_std__is_missing",
        ),
    ),
    "baseline::numeric_max": (
        (
            "f_beer_ratings_aroma_max",
            "f_beer_ratings_aroma_max",
        ),
        (
            "f_beer_ratings_aroma_max__is_missing",
            "f_beer_ratings_aroma_max__is_missing",
        ),
    ),
    "baseline::days_since_last": (
        (
            "f_beer_ratings_days_since_last",
            "f_beer_ratings_days_since_last",
        ),
        (
            "f_beer_ratings_days_since_last__is_missing",
            "f_beer_ratings_days_since_last__is_missing",
        ),
    ),
}


def test_ratebeer_real_plan_known_legacy_baseline_bindings_are_partial() -> None:
    plan = ratebeer_plan()
    passthrough_steps = [
        item for item in plan.steps
        if item.lowering_mode == LoweringMode.PASSTHROUGH
    ]
    assert len(passthrough_steps) == 15
    assert sum(
        1
        for item in passthrough_steps
        if item.primitive_id in RATEBEER_KNOWN_BASELINE_BINDINGS
    ) == 5
    assert sum(
        len(columns)
        for columns in RATEBEER_KNOWN_BASELINE_BINDINGS.values()
    ) == 9
    with pytest.raises(PassthroughBindingError) as exc:
        resolve_passthrough_bindings(
            plan,
            explicit_bindings=RATEBEER_KNOWN_BASELINE_BINDINGS,
        )
    assert exc.value.code == (
        PassthroughBindingCode.MISSING_PASSTHROUGH_BINDING
    )
    assert "baseline::history::window_count_short" in str(exc.value)


def test_activity_product_ratio_remain_excluded() -> None:
    columns = tuple(
        column
        for values in RATEBEER_KNOWN_BASELINE_BINDINGS.values()
        for _, column in values
    )
    assert "f_pairtmp__user_place_activity_product" not in columns
    assert "f_pairtmp__user_place_activity_ratio" not in columns


def test_no_heavy_or_filesystem_dependency() -> None:
    import fdhg.compiler.passthrough_bindings as module

    names = set(module.__dict__)
    source = inspect.getsource(module)
    assert "pandas" not in names
    assert "pyarrow" not in names
    assert "subprocess" not in names
    assert "tabpfn" not in names
    assert "Path" not in names
    assert "open(" not in source
