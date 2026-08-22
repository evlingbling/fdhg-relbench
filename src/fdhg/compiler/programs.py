from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import load_yaml
from .ir import CompiledTask, PrimitiveFamily


@dataclass(frozen=True)
class CandidateProgram:
    program_id: str
    primitive_ids: list[str]
    families: list[str]
    description: str
    estimated_feature_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_families(
    primitive_ids: list[str],
) -> list[str]:
    families = []

    for family in [
        PrimitiveFamily.BASELINE.value,
        PrimitiveFamily.STRUCTURAL.value,
        PrimitiveFamily.TEMPORAL.value,
        PrimitiveFamily.COVERAGE.value,
    ]:
        prefix = f"{family}::"

        if any(
            primitive_id.startswith(prefix)
            for primitive_id in primitive_ids
        ):
            families.append(family)

    return families


def build_pairwise_candidates(
    compiled: CompiledTask,
    *,
    baseline: list[str],
    structural: list[str],
    temporal: list[str],
) -> list[CandidateProgram]:
    by_role = {
        "left": [],
        "right": [],
        "pair": [],
    }

    primitive_by_id = {
        primitive.primitive_id: primitive
        for primitive in compiled.candidate_primitives
    }

    for primitive_id in temporal:
        primitive = primitive_by_id[primitive_id]

        role = primitive.metadata.get(
            "pairwise_role"
        )

        if role in by_role:
            by_role[role].append(
                primitive_id
            )

    programs = [
        CandidateProgram(
            program_id="baseline",
            primitive_ids=list(baseline),
            families=["baseline"],
            description="Relational baseline program.",
        )
    ]

    role_programs = [
        (
            "left",
            "baseline_plus_pair_left_temporal",
            "Baseline plus left-entity temporal history.",
        ),
        (
            "right",
            "baseline_plus_pair_right_temporal",
            "Baseline plus right-entity temporal history.",
        ),
        (
            "pair",
            "baseline_plus_pair_history",
            "Baseline plus prior pair-interaction history.",
        ),
    ]

    for role, program_id, description in role_programs:
        role_ids = by_role[role]

        if not role_ids:
            continue

        programs.append(
            CandidateProgram(
                program_id=program_id,
                primitive_ids=(
                    list(baseline)
                    + list(role_ids)
                ),
                families=[
                    "baseline",
                    "temporal",
                ],
                description=description,
                metadata={
                    "pairwise_role": role,
                },
            )
        )

    all_pairwise_temporal = (
        list(by_role["left"])
        + list(by_role["right"])
        + list(by_role["pair"])
    )

    if all_pairwise_temporal:
        programs.append(
            CandidateProgram(
                program_id=(
                    "baseline_plus_pairwise_temporal"
                ),
                primitive_ids=(
                    list(baseline)
                    + all_pairwise_temporal
                ),
                families=[
                    "baseline",
                    "temporal",
                ],
                description=(
                    "Baseline plus left, right, and "
                    "pair-history temporal programs."
                ),
                metadata={
                    "pairwise_specialization": True,
                },
            )
        )

    if structural:
        programs.append(
            CandidateProgram(
                program_id="baseline_plus_structural",
                primitive_ids=(
                    list(baseline)
                    + list(structural)
                ),
                families=[
                    "baseline",
                    "structural",
                ],
                description=(
                    "Baseline plus all generated "
                    "structural residuals."
                ),
            )
        )

    if structural and all_pairwise_temporal:
        programs.append(
            CandidateProgram(
                program_id=(
                    "baseline_plus_structural_"
                    "pairwise_temporal"
                ),
                primitive_ids=(
                    list(baseline)
                    + list(structural)
                    + all_pairwise_temporal
                ),
                families=[
                    "baseline",
                    "structural",
                    "temporal",
                ],
                description=(
                    "Baseline plus structural residuals "
                    "and complete pairwise temporal history."
                ),
                metadata={
                    "pairwise_specialization": True,
                },
            )
        )

    return programs


def build_default_candidates(
    compiled: CompiledTask,
) -> list[CandidateProgram]:
    by_family: dict[str, list[str]] = {}

    for primitive in compiled.candidate_primitives:
        family = primitive.family.value
        by_family.setdefault(family, []).append(
            primitive.primitive_id
        )

    baseline = by_family.get(
        PrimitiveFamily.BASELINE.value,
        [],
    )
    structural = by_family.get(
        PrimitiveFamily.STRUCTURAL.value,
        [],
    )
    temporal = by_family.get(
        PrimitiveFamily.TEMPORAL.value,
        [],
    )

    if compiled.task_spec.pairwise is not None:
        return build_pairwise_candidates(
            compiled,
            baseline=list(baseline),
            structural=list(structural),
            temporal=list(temporal),
        )

    programs = [
        CandidateProgram(
            program_id="baseline",
            primitive_ids=list(baseline),
            families=["baseline"],
            description=(
                "Relational baseline program."
            ),
        )
    ]

    if structural:
        programs.append(
            CandidateProgram(
                program_id="baseline_plus_structural",
                primitive_ids=(
                    list(baseline)
                    + list(structural)
                ),
                families=[
                    "baseline",
                    "structural",
                ],
                description=(
                    "Baseline plus all generated "
                    "structural residuals."
                ),
            )
        )

    if temporal:
        programs.append(
            CandidateProgram(
                program_id="baseline_plus_temporal",
                primitive_ids=(
                    list(baseline)
                    + list(temporal)
                ),
                families=[
                    "baseline",
                    "temporal",
                ],
                description=(
                    "Baseline plus generated temporal "
                    "residuals."
                ),
            )
        )

    if structural and temporal:
        programs.append(
            CandidateProgram(
                program_id=(
                    "baseline_plus_structural_temporal"
                ),
                primitive_ids=(
                    list(baseline)
                    + list(structural)
                    + list(temporal)
                ),
                families=[
                    "baseline",
                    "structural",
                    "temporal",
                ],
                description=(
                    "Baseline plus generated structural "
                    "and temporal residuals."
                ),
            )
        )

    return programs


def build_configured_candidates(
    compiled: CompiledTask,
    *,
    reproduction_config: Path,
    semantics_config: Path | None = None,
) -> list[CandidateProgram]:
    """Build default candidates plus explicit task-scoped declarations."""

    declarations = _candidate_declarations(
        compiled=compiled,
        reproduction_config=reproduction_config,
        semantics_config=semantics_config,
    )
    if declarations is None:
        return build_default_candidates(compiled)
    programs = build_default_candidates(compiled)
    declared = build_declared_candidates(
        compiled,
        declarations=declarations,
    )
    existing_ids = {program.program_id for program in programs}
    duplicate_programs = sorted(
        program.program_id
        for program in declared
        if program.program_id in existing_ids
    )
    if duplicate_programs:
        raise ValueError(
            "declared candidate program duplicates default program: "
            + ", ".join(duplicate_programs)
        )
    return programs + declared


def build_declared_candidates(
    compiled: CompiledTask,
    *,
    declarations,
) -> list[CandidateProgram]:
    if not isinstance(declarations, list):
        raise ValueError("candidate_programs must be a list")
    primitive_by_id = {
        primitive.primitive_id: primitive
        for primitive in compiled.candidate_primitives
    }
    programs: list[CandidateProgram] = []
    seen_programs: set[str] = set()
    for index, raw in enumerate(declarations):
        if not isinstance(raw, dict):
            raise ValueError("candidate_programs rows must be mappings")
        program_id = str(raw.get("program_id", ""))
        if not program_id:
            raise ValueError("candidate_programs rows require program_id")
        if program_id in seen_programs:
            raise ValueError(f"duplicate candidate program_id: {program_id}")
        seen_programs.add(program_id)
        primitive_ids = raw.get("primitive_ids")
        if (
            not isinstance(primitive_ids, list)
            or any(not isinstance(item, str) for item in primitive_ids)
        ):
            raise ValueError(
                f"candidate_programs[{index}].primitive_ids must be a string list"
            )
        duplicates = sorted({
            primitive_id
            for primitive_id in primitive_ids
            if primitive_ids.count(primitive_id) > 1
        })
        if duplicates:
            raise ValueError(
                f"duplicate primitive IDs in {program_id}: "
                + ", ".join(duplicates)
            )
        unknown = sorted(
            primitive_id
            for primitive_id in primitive_ids
            if primitive_id not in primitive_by_id
        )
        if unknown:
            raise ValueError(
                f"unknown primitive IDs in {program_id}: "
                + ", ".join(unknown)
            )
        programs.append(CandidateProgram(
            program_id=program_id,
            primitive_ids=list(primitive_ids),
            families=infer_families(list(primitive_ids)),
            description=str(
                raw.get(
                    "description",
                    "Task-scoped configured candidate program.",
                )
            ),
            estimated_feature_count=(
                None
                if raw.get("estimated_feature_count") is None
                else int(raw["estimated_feature_count"])
            ),
            metadata={
                "configured_candidate": True,
                **(
                    raw.get("metadata")
                    if isinstance(raw.get("metadata"), dict)
                    else {}
                ),
            },
        ))
    return programs


def _candidate_declarations(
    *,
    compiled: CompiledTask,
    reproduction_config: Path,
    semantics_config: Path | None,
):
    key = f"{compiled.task_spec.dataset}/{compiled.task_spec.task}"
    config = load_yaml(reproduction_config)
    tasks = config.get("tasks", config)
    raw_task = tasks.get(key, {}) if isinstance(tasks, dict) else {}
    if isinstance(raw_task, dict) and "candidate_programs" in raw_task:
        return raw_task["candidate_programs"]
    if isinstance(raw_task, dict):
        prepared = raw_task.get("prepared_artifacts", {})
        if isinstance(prepared, dict) and prepared.get("provider") == "onboarding":
            raw_manifest = prepared.get("onboarding_manifest", {})
            if isinstance(raw_manifest, dict) and raw_manifest.get("path"):
                path = Path(str(raw_manifest["path"]))
                if not path.is_absolute():
                    path = (reproduction_config.parent / path).resolve()
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if "candidate_programs" in manifest:
                    return manifest["candidate_programs"]
    if semantics_config is not None and semantics_config.exists():
        semantics = load_yaml(semantics_config).get(key, {})
        if isinstance(semantics, dict) and "candidate_programs" in semantics:
            return semantics["candidate_programs"]
    return None


def build_block_candidates(
    compiled: CompiledTask,
    realized_primitive_ids_by_program: (
        dict[str, list[str]] | None
    ) = None,
) -> list[CandidateProgram]:
    """
    Build logical candidate programs.

    Without an existing backend, construct default family-level
    candidates from the compiler IR.

    With an existing backend, preserve every backend program ID
    exactly. This allows distinct specializations such as compact
    and expanded structural programs to compete independently.
    """
    if realized_primitive_ids_by_program is None:
        return build_default_candidates(compiled)

    programs = []

    for program_id, primitive_ids in (
        realized_primitive_ids_by_program.items()
    ):
        ids = list(primitive_ids)

        programs.append(
            CandidateProgram(
                program_id=program_id,
                primitive_ids=ids,
                families=infer_families(ids),
                description=(
                    "Existing canonical candidate program "
                    "bound to the shared compiler IR."
                ),
                metadata={
                    "primitive_binding": (
                        "realized_existing_artifact"
                    ),
                },
            )
        )

    return programs
