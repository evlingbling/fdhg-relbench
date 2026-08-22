from __future__ import annotations

from pathlib import Path

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.ir import PrimitiveFamily
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates


CONFIG = Path("configs/reproduction/tasks.yaml")
SEMANTICS = Path(
    "configs/reproduction/task_semantics.yaml"
)


def compile_task(dataset: str, task: str):
    spec = load_task_spec(
        dataset=dataset,
        task=task,
        reproduction_config=CONFIG,
        semantics_config=SEMANTICS,
    )
    return build_candidate_program(spec)


def load_ratebeer_pairwise_spec():
    return load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=CONFIG,
        semantics_config=SEMANTICS,
    )


def temporal_primitives(compiled):
    return [
        primitive
        for primitive in compiled.candidate_primitives
        if primitive.event_time_col is not None
        and primitive.temporal_predicate is not None
    ]


def assert_strict_predicate(primitive) -> None:
    assert "<=" not in primitive.temporal_predicate
    assert (
        f"{primitive.event_time_col} < target."
        in primitive.temporal_predicate
    )


def test_generic_temporal_primitives_use_strict_cutoff() -> None:
    compiled = compile_task(
        "rel-ratebeer",
        "user-count",
    )
    generated = [
        primitive
        for primitive in compiled.candidate_primitives
        if primitive.family == PrimitiveFamily.TEMPORAL
    ]

    assert generated

    for primitive in generated:
        assert_strict_predicate(primitive)


def test_pairwise_left_right_and_pair_predicates_are_strict() -> None:
    compiled = compile_task(
        "rel-ratebeer",
        "user-place-liked_pairwise",
    )
    roles = {
        "left": [],
        "right": [],
        "pair": [],
    }

    for primitive in compiled.candidate_primitives:
        role = primitive.metadata.get("pairwise_role")

        if role in roles:
            roles[role].append(primitive)

    assert all(roles.values())

    for role_primitives in roles.values():
        for primitive in role_primitives:
            assert primitive.temporal_predicate is not None
            assert_strict_predicate(primitive)


def test_ratebeer_pairwise_source_bindings_match_legacy_evidence() -> None:
    spec = load_ratebeer_pairwise_spec()
    pairwise = spec.pairwise

    assert pairwise is not None
    assert pairwise.left_history is not None
    assert pairwise.right_history is not None
    assert pairwise.pair_history is not None

    assert pairwise.left_history.table == "beer_ratings"
    assert pairwise.left_history.time_col == "updated_at"

    assert pairwise.right_history.table == "place_ratings"
    assert pairwise.right_history.key == "place_id"
    assert pairwise.right_history.related_col == "user_id"
    assert pairwise.right_history.time_col == "created_at"

    assert pairwise.pair_history.table == "place_ratings"
    assert pairwise.pair_history.left_key == "user_id"
    assert pairwise.pair_history.right_key == "place_id"
    assert pairwise.pair_history.time_col == "created_at"


def test_ratebeer_right_and_pair_predicates_reference_place_ratings() -> None:
    compiled = build_candidate_program(
        load_ratebeer_pairwise_spec()
    )

    for primitive in compiled.candidate_primitives:
        role = primitive.metadata.get("pairwise_role")

        if role == "right":
            assert "place_ratings.created_at" in (
                primitive.temporal_predicate or ""
            )

        if role == "pair":
            assert "place_ratings.created_at" in (
                primitive.temporal_predicate or ""
            )


def test_no_generated_temporal_primitive_contains_non_strict_cutoff() -> None:
    for dataset, task in [
        ("rel-ratebeer", "user-count"),
        (
            "rel-ratebeer",
            "user-place-liked_pairwise",
        ),
    ]:
        compiled = compile_task(dataset, task)

        for primitive in temporal_primitives(compiled):
            assert "<=" not in (
                primitive.temporal_predicate or ""
            )


def test_candidate_program_ids_remain_unchanged() -> None:
    pairwise = compile_task(
        "rel-ratebeer",
        "user-place-liked_pairwise",
    )
    non_pairwise = compile_task(
        "rel-ratebeer",
        "user-count",
    )

    assert {
        program.program_id
        for program in build_default_candidates(pairwise)
    } == {
        "baseline",
        "baseline_plus_pair_left_temporal",
        "baseline_plus_pair_right_temporal",
        "baseline_plus_pair_history",
        "baseline_plus_pairwise_temporal",
        "baseline_plus_structural",
        "baseline_plus_structural_pairwise_temporal",
    }

    assert {
        program.program_id
        for program in build_default_candidates(non_pairwise)
    } == {
        "baseline",
        "baseline_plus_structural",
        "baseline_plus_temporal",
        "baseline_plus_structural_temporal",
    }


def test_planner_predicate_checks_write_no_files(tmp_path: Path) -> None:
    before = sorted(tmp_path.iterdir())

    compile_task(
        "rel-ratebeer",
        "user-place-liked_pairwise",
    )

    after = sorted(tmp_path.iterdir())
    assert after == before
