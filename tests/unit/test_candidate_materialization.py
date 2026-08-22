from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from fdhg.compiler.candidate_safety import ExplicitLoweringEvidence
from fdhg.compiler.config import load_task_spec
from fdhg.compiler.ir import (
    CompiledTask,
    PairwiseHistorySpec,
    PairwiseSpec,
    Primitive,
    PrimitiveFamily,
    TaskSpec,
)
from fdhg.compiler.materializer import (
    CandidateMaterializationRequest,
    materialize_candidate_program,
)
from fdhg.compiler.passthrough_provenance import (
    PassthroughBindingEvidence,
    PassthroughProvenanceReport,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import CandidateProgram, build_default_candidates


T0 = datetime(2026, 6, 1, 12, 0, 0)


def task_spec() -> TaskSpec:
    return TaskSpec(
        dataset="rel-example",
        task="pairwise",
        problem_type="binary",
        label_col="label",
        entity_key="user_id",
        target_time_col="timestamp",
        horizon_days=30,
        pairwise=PairwiseSpec(
            left_key="user_id",
            right_key="item_id",
            target_right_key="candidate_item_id",
            left_history=PairwiseHistorySpec(
                table="events",
                key="user_id",
                related_col="item_id",
                time_col="event_time",
            ),
        ),
    )


def temporal_primitive(
    primitive_id: str = "temporal::pairwise::left::count::30d",
    *,
    temporally_safe: bool = True,
) -> Primitive:
    return Primitive(
        primitive_id=primitive_id,
        family=PrimitiveFamily.TEMPORAL,
        operation="window_count",
        source_table="events",
        group_key="user_id",
        event_time_col="event_time",
        window_days=30,
        temporal_predicate="events.event_time < target.timestamp",
        temporally_safe=temporally_safe,
        metadata={"pairwise_role": "left"},
    )


def baseline_primitive() -> Primitive:
    return Primitive(
        primitive_id="baseline::count",
        family=PrimitiveFamily.BASELINE,
        operation="count",
        source_table="events",
        group_key="user_id",
        event_time_col="event_time",
    )


def structural_primitive() -> Primitive:
    return Primitive(
        primitive_id="structural::afd::majority_confidence",
        family=PrimitiveFamily.STRUCTURAL,
        operation="majority_confidence",
    )


def compiled_with(*primitives: Primitive) -> CompiledTask:
    return CompiledTask(
        task_spec=task_spec(),
        candidate_primitives=list(primitives),
    )


def program_for(
    *primitive_ids: str,
    program_id: str = "candidate",
) -> CandidateProgram:
    return CandidateProgram(
        program_id=program_id,
        primitive_ids=list(primitive_ids),
        families=["temporal"],
        description="Synthetic candidate.",
    )


def source_rows():
    return {
        "events": [
            {
                "user_id": "u1",
                "item_id": "i1",
                "event_time": T0 - timedelta(days=4),
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "event_time": T0 - timedelta(days=1),
            },
            {
                "user_id": "u1",
                "item_id": "same-time",
                "event_time": T0,
            },
        ]
    }


def train_rows():
    return [
        {
            "user_id": "u1",
            "candidate_item_id": "i9",
            "timestamp": T0,
            "label": 1,
        }
    ]


def validation_rows():
    return [
        {
            "user_id": "u1",
            "candidate_item_id": "i8",
            "timestamp": T0 + timedelta(days=1),
            "label": 0,
        }
    ]


def output_dir(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "outputs"
        / "e2e"
        / "rel-example_pairwise"
        / "candidates"
        / "candidate"
    )


def request(
    tmp_path: Path,
    compiled: CompiledTask,
    program: CandidateProgram,
    *,
    write: bool = True,
    **kwargs,
) -> CandidateMaterializationRequest:
    return CandidateMaterializationRequest(
        compiled=compiled,
        program=program,
        output_dir=output_dir(tmp_path),
        source_rows_by_table=source_rows(),
        train_target_rows=train_rows(),
        validation_target_rows=validation_rows(),
        write=write,
        **kwargs,
    )


def test_dry_run_performs_no_writes(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    result = materialize_candidate_program(
        request(tmp_path, compiled, program, write=False)
    )

    assert result.dry_run
    assert not output_dir(tmp_path).exists()


def test_successful_train_validation_materialization(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    result = materialize_candidate_program(request(tmp_path, compiled, program))

    assert result.selector_ready
    assert result.train_artifact and result.train_artifact.exists()
    assert result.validation_artifact and result.validation_artifact.exists()
    frame = pd.read_parquet(result.train_artifact)
    assert frame["f_pairwise__left__count_30d"].tolist() == [2]


def test_train_only_fit_state_and_validation_isolation(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    result = materialize_candidate_program(request(tmp_path, compiled, program))

    val_frame = pd.read_parquet(result.validation_artifact)
    assert val_frame["f_pairwise__left__count_30d"].tolist() == [3]
    assert "candidate_item_id" not in result.feature_columns


def test_temporal_cutoff_enforced(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    result = materialize_candidate_program(request(tmp_path, compiled, program))

    frame = pd.read_parquet(result.train_artifact)
    assert frame["f_pairwise__left__count_30d"].iloc[0] == 2


def test_unknown_primitive_failure_leaves_no_candidate(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for("temporal::unknown")

    with pytest.raises(ValueError, match="not materializable"):
        materialize_candidate_program(request(tmp_path, compiled, program))

    assert not output_dir(tmp_path).exists()


def test_partial_passthrough_binding_failure(tmp_path: Path) -> None:
    compiled = compiled_with(baseline_primitive())
    program = program_for("baseline::count")
    report = PassthroughProvenanceReport(
        program_id="candidate",
        passthrough_step_count=1,
        binding_evidence=(
            PassthroughBindingEvidence(
                program_id="candidate",
                primitive_id="baseline::count",
                source_column="baseline_count",
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

    with pytest.raises(ValueError, match="provenance"):
        materialize_candidate_program(
            request(
                tmp_path,
                compiled,
                program,
                passthrough_provenance_report=report,
            )
        )

    assert not output_dir(tmp_path).exists()


def test_complete_passthrough_binding_materializes(
    tmp_path: Path,
) -> None:
    compiled = compiled_with(baseline_primitive())
    program = program_for("baseline::count")
    proven = PassthroughBindingEvidence(
        program_id="candidate",
        primitive_id="baseline::count",
        source_column="baseline_count_src",
        output_column="baseline_count",
        evidence_kind="manifest",
        evidence_location="manifest",
        status="proven",
    )
    report = PassthroughProvenanceReport(
        program_id="candidate",
        passthrough_step_count=1,
        binding_evidence=(proven,),
        proven_bindings=(proven,),
        unresolved_primitive_ids=(),
        conflicting_primitive_ids=(),
        complete=True,
    )

    result = materialize_candidate_program(
        CandidateMaterializationRequest(
            compiled=compiled,
            program=program,
            output_dir=output_dir(tmp_path),
            source_rows_by_table=source_rows(),
            train_target_rows=[
                {
                    **train_rows()[0],
                    "baseline_count_src": 4,
                }
            ],
            validation_target_rows=[
                {
                    **validation_rows()[0],
                    "baseline_count_src": 5,
                }
            ],
            write=True,
            passthrough_provenance_report=report,
        )
    )

    assert result.selector_ready
    assert pd.read_parquet(result.train_artifact)["baseline_count"].tolist() == [4]


def test_task_mismatched_binding_failure(tmp_path: Path) -> None:
    compiled = compiled_with(structural_primitive())
    program = program_for("structural::afd::majority_confidence")
    evidence = ExplicitLoweringEvidence(
        dataset="rel-example",
        task="other-task",
        program_id="candidate",
        primitive_id="structural::afd::majority_confidence",
        source_table="events",
        source_column="src",
        output_column="out",
        status="proven",
        evidence_location="manifest",
    )

    with pytest.raises(ValueError, match="provenance"):
        materialize_candidate_program(
            request(
                tmp_path,
                compiled,
                program,
                explicit_lowering_evidence=(evidence,),
            )
        )

    assert not output_dir(tmp_path).exists()


def test_complete_external_binding_materializes(tmp_path: Path) -> None:
    compiled = compiled_with(structural_primitive())
    program = program_for("structural::afd::majority_confidence")
    evidence = ExplicitLoweringEvidence(
        dataset="rel-example",
        task="pairwise",
        program_id="candidate",
        primitive_id="structural::afd::majority_confidence",
        source_table="target",
        source_column="external_src",
        output_column="external_out",
        status="proven",
        evidence_location="provider-manifest",
    )

    result = materialize_candidate_program(
        CandidateMaterializationRequest(
            compiled=compiled,
            program=program,
            output_dir=output_dir(tmp_path),
            source_rows_by_table=source_rows(),
            train_target_rows=[
                {
                    **train_rows()[0],
                    "external_src": 0.75,
                }
            ],
            validation_target_rows=[
                {
                    **validation_rows()[0],
                    "external_src": 0.25,
                }
            ],
            write=True,
            explicit_lowering_evidence=(evidence,),
        )
    )

    assert result.selector_ready
    assert pd.read_parquet(result.validation_artifact)["external_out"].tolist() == [0.25]


def test_deterministic_bindings_and_manifest(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    result = materialize_candidate_program(request(tmp_path, compiled, program))

    bindings = json.loads(result.bindings_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert bindings["records"][0]["primitive_id"] == (
        "temporal::pairwise::left::count::30d"
    )
    assert manifest["feature_columns"] == ["f_pairwise__left__count_30d"]


def test_audit_files_emitted_automatically(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    result = materialize_candidate_program(request(tmp_path, compiled, program))

    assert {path.name for path in result.audit_paths} == {
        "temporal_safety_audit.csv",
        "leakage_safety_audit.csv",
        "lowering_provenance_audit.csv",
    }
    assert all(path.exists() for path in result.audit_paths)


def test_leakage_audit_failure_blocks_publication(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    with pytest.raises(ValueError, match="target_aggregate"):
        materialize_candidate_program(
            request(
                tmp_path,
                compiled,
                program,
                target_aggregate_columns=(
                    "f_pairwise__left__count_30d",
                ),
            )
        )

    assert not output_dir(tmp_path).exists()


def test_temporal_audit_failure_blocks_publication(tmp_path: Path) -> None:
    primitive = Primitive(
        primitive_id="temporal::pairwise::left::count::30d",
        family=PrimitiveFamily.TEMPORAL,
        operation="window_count",
        source_table="events",
        group_key="user_id",
        event_time_col="event_time",
        window_days=30,
        temporal_predicate="events.event_time <= target.timestamp",
        metadata={"pairwise_role": "left"},
    )
    compiled = compiled_with(primitive)
    program = program_for(primitive.primitive_id)

    with pytest.raises(ValueError, match="not materializable"):
        materialize_candidate_program(request(tmp_path, compiled, program))

    assert not output_dir(tmp_path).exists()


def test_provenance_failure_blocks_publication(tmp_path: Path) -> None:
    compiled = compiled_with(structural_primitive())
    program = program_for("structural::afd::majority_confidence")

    with pytest.raises(ValueError, match="provenance"):
        materialize_candidate_program(request(tmp_path, compiled, program))

    assert not output_dir(tmp_path).exists()


def test_atomic_write_behavior(tmp_path: Path, monkeypatch) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "fdhg.compiler.materializer._write_candidate_program_json",
        fail,
    )
    with pytest.raises(RuntimeError, match="boom"):
        materialize_candidate_program(request(tmp_path, compiled, program))

    assert not output_dir(tmp_path).exists()
    assert not list(output_dir(tmp_path).parent.glob("_candidate.*.tmp"))


def test_overwrite_refusal(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)
    output_dir(tmp_path).mkdir(parents=True)

    with pytest.raises(FileExistsError):
        materialize_candidate_program(request(tmp_path, compiled, program))


def test_valid_resume_reuse(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)
    materialize_candidate_program(request(tmp_path, compiled, program))

    result = materialize_candidate_program(request(tmp_path, compiled, program))

    assert result.reused


def test_invalid_stale_directory_rejected(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)
    output_dir(tmp_path).mkdir(parents=True)
    (output_dir(tmp_path) / "target_with_dfs_agg_train.parquet").touch()

    with pytest.raises(FileExistsError):
        materialize_candidate_program(request(tmp_path, compiled, program))


def test_discovery_sees_only_completed_candidates(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)
    result = materialize_candidate_program(request(tmp_path, compiled, program))
    private = result.output_dir.parent / "_private"
    private.mkdir()
    (private / "target_with_dfs_agg_train.parquet").touch()
    (private / "target_with_dfs_agg_val.parquet").touch()
    sweep = _load_sweep_module()

    discovered = sweep.discover_materialized_candidates(
        task_output_root=result.output_dir.parents[1],
        configured_candidates=[],
    )

    assert discovered == ["candidate"]


def test_no_test_split() -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)

    with pytest.raises(ValueError, match="train/validation"):
        materialize_candidate_program(
            CandidateMaterializationRequest(
                compiled=compiled,
                program=program,
                output_dir=Path("/tmp/candidate"),
                source_rows_by_table=source_rows(),
                train_target_rows=train_rows(),
                validation_target_rows=validation_rows(),
                write=False,
                validation_split="test",
            )
        )


def test_no_mutation_of_inputs(tmp_path: Path) -> None:
    compiled = compiled_with(temporal_primitive())
    program = program_for(temporal_primitive().primitive_id)
    rows = source_rows()
    targets = train_rows()
    before_rows = deepcopy(rows)
    before_targets = deepcopy(targets)

    materialize_candidate_program(
        CandidateMaterializationRequest(
            compiled=compiled,
            program=program,
            output_dir=output_dir(tmp_path),
            source_rows_by_table=rows,
            train_target_rows=targets,
            validation_target_rows=validation_rows(),
            write=True,
        )
    )

    assert rows == before_rows
    assert targets == before_targets


def test_ratebeer_unresolved_provenance_remains_blocked(tmp_path: Path) -> None:
    task = load_task_spec(
        dataset="rel-ratebeer",
        task="user-place-liked_pairwise",
        reproduction_config=Path("configs/reproduction/tasks.yaml"),
    )
    compiled = build_candidate_program(task)
    program = next(
        item
        for item in build_default_candidates(compiled)
        if item.program_id == "baseline"
    )
    source_rows_by_table = {
        task.child_table or "beer_ratings": (),
    }

    with pytest.raises(ValueError, match="provenance"):
        materialize_candidate_program(
            CandidateMaterializationRequest(
                compiled=compiled,
                program=program,
                output_dir=tmp_path / "ratebeer",
                source_rows_by_table=source_rows_by_table,
                train_target_rows=(),
                validation_target_rows=(),
                write=True,
            )
        )

    assert not (tmp_path / "ratebeer").exists()


def _load_sweep_module():
    path = Path("scripts/experiments/run_candidate_program_sweep.py")
    spec = importlib.util.spec_from_file_location(
        "run_candidate_program_sweep",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
