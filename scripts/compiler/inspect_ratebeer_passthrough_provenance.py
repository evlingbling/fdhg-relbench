from __future__ import annotations

import ast
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fdhg.compiler.config import load_task_spec
from fdhg.compiler.materializer import (
    LoweringMode,
    plan_candidate_materialization,
)
from fdhg.compiler.passthrough_provenance import (
    PassthroughBindingEvidence,
    build_passthrough_provenance_report,
)
from fdhg.compiler.planner import build_candidate_program
from fdhg.compiler.programs import build_default_candidates


PROGRAM_ID = "baseline_plus_pairwise_temporal"
DATASET = "rel-ratebeer"
TASK = "user-place-liked_pairwise"
ACTIVITY_COLUMNS = (
    "f_pairtmp__user_place_activity_product",
    "f_pairtmp__user_place_activity_ratio",
)
DFS_SCHEMA_PATHS = (
    "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/dfs/"
    "target_with_dfs_agg_train.parquet",
    "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/dfs/"
    "target_with_dfs_agg_val.parquet",
)
TEMPORAL_SCHEMA_PATH = (
    "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
    "candidates/temporal_only/target_with_dfs_agg_train.parquet"
)


def main() -> None:
    args = parse_args()
    plan = build_ratebeer_plan()
    passthrough_ids = tuple(
        step.primitive_id
        for step in plan.steps
        if step.lowering_mode == LoweringMode.PASSTHROUGH
    )

    inspected: list[str] = []
    extra_legacy_columns: set[str] = set()
    evidence: list[PassthroughBindingEvidence] = []

    inspect_existing_manifests(
        plan=plan,
        passthrough_ids=passthrough_ids,
        inspected=inspected,
        evidence=evidence,
    )
    inspect_local_parquet_schemas(
        plan=plan,
        supplied_schemas=read_supplied_schemas(
            args.schema_columns_json
        ),
        inspected=inspected,
        evidence=evidence,
        extra_legacy_columns=extra_legacy_columns,
    )
    inspect_prepared_artifacts(
        plan=plan,
        supplied_schemas=read_supplied_schemas(
            args.schema_columns_json
        ),
        inspected=inspected,
        evidence=evidence,
        extra_legacy_columns=extra_legacy_columns,
    )
    inspect_legacy_code(inspected=inspected)
    inspect_existing_backend(inspected=inspected)
    inspect_discovery_code(inspected=inspected)
    inspect_archived_docs(inspected=inspected)
    inspect_git_history(
        plan=plan,
        passthrough_ids=passthrough_ids,
        inspected=inspected,
        evidence=evidence,
    )

    report = build_passthrough_provenance_report(
        plan,
        evidence_records=tuple(evidence),
    )

    rows = report.binding_evidence
    proven = tuple(row for row in rows if row.status == "proven")
    partial = tuple(row for row in rows if row.status == "partial")
    missing = tuple(row for row in rows if row.status == "missing")
    conflicting = tuple(
        row for row in rows if row.status == "conflicting"
    )

    print("PROGRAM_ID", report.program_id)
    print("PASSTHROUGH_STEP_COUNT", report.passthrough_step_count)
    print("PROVEN_PRIMITIVE_COUNT", len(proven))
    print("PARTIAL_PRIMITIVE_COUNT", len(partial))
    print("MISSING_PRIMITIVE_COUNT", len(missing))
    print("CONFLICTING_PRIMITIVE_COUNT", len(conflicting))
    print(
        "PROVEN_SOURCE_COLUMN_COUNT",
        len({row.source_column for row in proven}),
    )
    print(
        "PROVEN_OUTPUT_COLUMN_COUNT",
        len({row.output_column for row in proven}),
    )
    print("EXTRA_LEGACY_COLUMN_COUNT", len(extra_legacy_columns))
    print(
        "ACTIVITY_PRODUCT_RATIO_FOUND",
        bool(set(ACTIVITY_COLUMNS) & extra_legacy_columns),
    )
    print("PROVENANCE_REPORT_COMPLETE", report.complete)
    print()

    print("RATEBEER_LOGICAL_PRIMITIVES")
    for primitive_id in passthrough_ids:
        print(primitive_id)
    print()

    print("EVIDENCE_SOURCES_INSPECTED")
    for item in inspected:
        print(item)
    print()

    print(
        "PRIMITIVE_ID\tSTATUS\tSOURCE_COLUMN\tOUTPUT_COLUMN\t"
        "EVIDENCE_KIND\tEVIDENCE_LOCATION\tNOTES"
    )
    for row in rows:
        print(
            "\t".join((
                row.primitive_id,
                row.status,
                row.source_column or "",
                row.output_column or "",
                row.evidence_kind,
                row.evidence_location,
                " | ".join(row.notes),
            ))
        )
    print()

    print("EXTRA_LEGACY_COLUMNS")
    for column in sorted(extra_legacy_columns):
        print(column)
    print()

    print("REMOTE_READ_ONLY_COMMAND")
    print(remote_read_only_command())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RateBeer pairwise passthrough provenance "
            "inspection."
        )
    )
    parser.add_argument(
        "--schema-columns-json",
        type=Path,
        default=None,
        help=(
            "Optional read-only JSON mapping of relative parquet "
            "paths to ordered schema column names."
        ),
    )
    return parser.parse_args()


def build_ratebeer_plan():
    spec = load_task_spec(
        dataset=DATASET,
        task=TASK,
        reproduction_config=ROOT
        / "configs/reproduction/tasks.yaml",
        semantics_config=ROOT
        / "configs/reproduction/task_semantics.yaml",
    )
    compiled = build_candidate_program(spec)
    program = next(
        item
        for item in build_default_candidates(compiled)
        if item.program_id == PROGRAM_ID
    )
    return plan_candidate_materialization(
        compiled,
        program,
        available_source_tables={
            "beer_ratings",
            "place_ratings",
        },
    )


def inspect_existing_manifests(
    *,
    plan,
    passthrough_ids: tuple[str, ...],
    inspected: list[str],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    paths = [
        ROOT
        / "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
        "candidate_manifest.csv",
        ROOT
        / "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
        "dfs_feature_manifest.csv",
        ROOT
        / "configs/reproduction/legacy_task_specs.json",
        ROOT
        / "configs/reproduction/task_inventory_with_artifacts.csv",
    ]

    for path in paths:
        inspected.append(describe_path(path))
        if not path.exists():
            continue
        if path.suffix == ".json":
            inspect_json_manifest(
                plan=plan,
                path=path,
                passthrough_ids=passthrough_ids,
                evidence=evidence,
            )
        elif path.suffix == ".csv":
            inspect_csv_manifest(
                plan=plan,
                path=path,
                passthrough_ids=passthrough_ids,
                evidence=evidence,
            )


def inspect_json_manifest(
    *,
    plan,
    path: Path,
    passthrough_ids: tuple[str, ...],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    task_key = f"{DATASET}/{TASK}"
    section = data.get(task_key) if isinstance(data, dict) else None

    if not isinstance(section, dict):
        return

    append_explicit_records(
        plan=plan,
        records=explicit_records_from_scoped_mapping(section),
        passthrough_ids=passthrough_ids,
        evidence=evidence,
        kind="legacy-task-spec",
        location_prefix=str(path.relative_to(ROOT)),
    )


def inspect_csv_manifest(
    *,
    plan,
    path: Path,
    passthrough_ids: tuple[str, ...],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        fields = set(reader.fieldnames)
        if "primitive_id" not in fields:
            return
        output_field = first_present(
            fields,
            ("output_column", "feature_name", "column"),
        )
        source_field = first_present(
            fields,
            ("source_column", "input_column", "feature_name"),
        )
        if output_field is None and source_field is None:
            return
        for row_index, row in enumerate(reader, start=2):
            primitive_id = row.get("primitive_id")
            if primitive_id not in passthrough_ids:
                continue
            source_column = row.get(source_field) if source_field else None
            output_column = row.get(output_field) if output_field else None
            status = (
                "proven"
                if source_column and output_column
                else "partial"
            )
            evidence.append(
                PassthroughBindingEvidence(
                    program_id=plan.program_id,
                    primitive_id=primitive_id,
                    source_column=source_column or None,
                    output_column=output_column or None,
                    evidence_kind="manifest",
                    evidence_location=(
                        f"{path.relative_to(ROOT)}:{row_index}"
                    ),
                    status=status,
                    notes=("manifest row keyed by primitive_id",),
                )
            )


def inspect_local_parquet_schemas(
    *,
    plan,
    supplied_schemas: dict[str, tuple[str, ...]],
    inspected: list[str],
    evidence: list[PassthroughBindingEvidence],
    extra_legacy_columns: set[str],
) -> None:
    paths = [
        ROOT
        / "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
        "dfs/target_with_dfs_agg_train.parquet",
        ROOT
        / "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
        "candidates/temporal_only/target_with_dfs_agg_train.parquet",
        ROOT
        / "outputs/pairwise/rel-ratebeer_user-place-liked/"
        "dfs_train_pairwise.parquet",
        ROOT
        / "outputs/pairwise/rel-ratebeer_user-place-liked/"
        "fdhg_train_pairwise.parquet",
    ]
    inspect_parquet_paths(
        paths,
        plan=plan,
        supplied_schemas=supplied_schemas,
        inspected=inspected,
        evidence=evidence,
        extra_legacy_columns=extra_legacy_columns,
    )


def inspect_prepared_artifacts(
    *,
    plan,
    supplied_schemas: dict[str, tuple[str, ...]],
    inspected: list[str],
    evidence: list[PassthroughBindingEvidence],
    extra_legacy_columns: set[str],
) -> None:
    paths = [
        ROOT
        / "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
        "dfs/target_with_dfs_agg_val.parquet",
        ROOT
        / "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
        "inspect/rel-ratebeer_user-place-liked/"
        "table_beer_ratings.parquet",
        ROOT
        / "outputs/e2e/rel-ratebeer_user-place-liked_pairwise/"
        "inspect/rel-ratebeer_user-place-liked/"
        "table_place_ratings.parquet",
    ]
    inspect_parquet_paths(
        paths,
        plan=plan,
        supplied_schemas=supplied_schemas,
        inspected=inspected,
        evidence=evidence,
        extra_legacy_columns=extra_legacy_columns,
    )


def inspect_parquet_paths(
    paths: Iterable[Path],
    *,
    plan,
    supplied_schemas: dict[str, tuple[str, ...]],
    inspected: list[str],
    evidence: list[PassthroughBindingEvidence],
    extra_legacy_columns: set[str],
) -> None:
    for path in paths:
        inspected.append(describe_path(path))
        relative = str(path.relative_to(ROOT))
        if relative in supplied_schemas:
            columns = supplied_schemas[relative]
        elif path.exists():
            columns = read_parquet_columns(path)
        else:
            continue
        inspect_schema_columns(
            plan=plan,
            relative_path=relative,
            columns=columns,
            evidence=evidence,
            extra_legacy_columns=extra_legacy_columns,
        )


def inspect_schema_columns(
    *,
    plan,
    relative_path: str,
    columns: tuple[str, ...],
    evidence: list[PassthroughBindingEvidence],
    extra_legacy_columns: set[str],
) -> None:
    for column in columns:
        print(f"PARQUET_COLUMN {relative_path} {column}")
        if column in ACTIVITY_COLUMNS:
            extra_legacy_columns.add(column)

    if relative_path not in DFS_SCHEMA_PATHS:
        return

    schema_columns = frozenset(columns)
    for primitive_id, column in expected_dfs_baseline_columns(plan).items():
        if column not in schema_columns:
            continue
        evidence.append(
            PassthroughBindingEvidence(
                program_id=plan.program_id,
                primitive_id=primitive_id,
                source_column=None,
                output_column=column,
                evidence_kind="parquet-schema",
                evidence_location=f"parquet-schema:{relative_path}",
                status="partial",
                notes=(
                    "physical DFS column present in pairwise schema",
                    "schema does not explicitly map primitive_id to column",
                ),
            )
        )


def expected_dfs_baseline_columns(plan) -> dict[str, str]:
    by_id = {step.primitive_id: step for step in plan.steps}
    count_step = by_id.get("baseline::count")
    numeric_step = by_id.get("baseline::numeric_mean")
    days_step = by_id.get("baseline::days_since_last")

    child_table = (
        count_step.source_table
        if count_step is not None
        else None
    )
    numeric_col = (
        numeric_step.related_col
        if numeric_step is not None
        else None
    )

    columns: dict[str, str] = {}
    if child_table is not None:
        columns["baseline::count"] = f"f_{child_table}_count"
        if days_step is not None:
            columns["baseline::days_since_last"] = (
                f"f_{child_table}_days_since_last"
            )
    if child_table is not None and numeric_col is not None:
        columns["baseline::numeric_mean"] = (
            f"f_{child_table}_{numeric_col}_mean"
        )
        columns["baseline::numeric_std"] = (
            f"f_{child_table}_{numeric_col}_std"
        )
        columns["baseline::numeric_max"] = (
            f"f_{child_table}_{numeric_col}_max"
        )
    return columns


def read_supplied_schemas(
    path: Path | None,
) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            "--schema-columns-json must contain a JSON object"
        )
    schemas = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, list):
            raise ValueError(
                "--schema-columns-json maps paths to string lists"
            )
        schemas[key] = tuple(
            item for item in value if isinstance(item, str)
        )
    return schemas


def read_parquet_columns(path: Path) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "PARQUET_SCHEMA_UNAVAILABLE",
            path.relative_to(ROOT),
            "pyarrow.parquet is not importable",
        )
        return ()

    schema = pq.read_schema(path)
    return tuple(schema.names)


def inspect_legacy_code(*, inspected: list[str]) -> None:
    inspected.append(
        describe_path(
            ROOT
            / "scripts/experiments/"
            "generate_pairwise_temporal_candidates.py"
        )
    )


def inspect_existing_backend(*, inspected: list[str]) -> None:
    inspected.append(
        describe_path(ROOT / "src/fdhg/compiler/existing_backend.py")
    )


def inspect_discovery_code(*, inspected: list[str]) -> None:
    for relative in (
        "src/fdhg/compiler/discovery.py",
        "src/fdhg/compiler/manifest.py",
        "src/fdhg/compiler/provenance.py",
    ):
        inspected.append(describe_path(ROOT / relative))


def inspect_archived_docs(*, inspected: list[str]) -> None:
    for relative in (
        "docs/reproducibility_gap_report.md",
        "docs/paper_to_code_map.md",
        "configs/reproduction/task_inventory_with_artifacts.csv",
    ):
        inspected.append(describe_path(ROOT / relative))


def inspect_git_history(
    *,
    plan,
    passthrough_ids: tuple[str, ...],
    inspected: list[str],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    command = [
        "git",
        "log",
        "--all",
        "--format=%H",
        "--",
        "outputs/e2e/rel-ratebeer_user-place-liked_pairwise",
        "outputs/pairwise/rel-ratebeer_user-place-liked",
        "configs/reproduction/legacy_task_specs.json",
        "src/fdhg/compiler/existing_backend.py",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    commits = tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    )
    inspected.append(
        "git-log:"
        + (
            ",".join(commits[:8])
            if commits
            else "no matching commits"
        )
    )
    for commit in commits[:8]:
        inspect_git_commit(
            plan=plan,
            commit=commit,
            passthrough_ids=passthrough_ids,
            inspected=inspected,
            evidence=evidence,
        )


def inspect_git_commit(
    *,
    plan,
    commit: str,
    passthrough_ids: tuple[str, ...],
    inspected: list[str],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    tree = run_git((
        "ls-tree",
        "-r",
        "--name-only",
        commit,
    ))
    paths = tuple(
        line
        for line in tree.splitlines()
        if is_relevant_historical_path(line)
    )
    inspected.append(
        f"git-ls-tree:{commit}:{len(paths)} relevant files"
    )

    for path in paths:
        text = run_git(("show", f"{commit}:{path}"))
        if path.endswith(".json"):
            inspect_historical_json(
                plan=plan,
                commit=commit,
                path=path,
                text=text,
                passthrough_ids=passthrough_ids,
                evidence=evidence,
            )
        elif path.endswith(".csv"):
            inspect_historical_csv(
                plan=plan,
                commit=commit,
                path=path,
                text=text,
                passthrough_ids=passthrough_ids,
                evidence=evidence,
            )
        elif path.endswith(".py"):
            inspect_historical_python(
                plan=plan,
                commit=commit,
                path=path,
                text=text,
                passthrough_ids=passthrough_ids,
                evidence=evidence,
            )

    grep = run_git((
        "grep",
        "-n",
        "-E",
        "primitive_id|primitive_column_bindings|baseline::history::",
        commit,
        "--",
        "src",
        "scripts",
        "configs",
        "docs",
        "results",
    ))
    matched_lines = sum(1 for line in grep.splitlines() if line)
    inspected.append(
        f"git-grep:{commit}:explicit-mapping-candidates={matched_lines}"
    )


def is_relevant_historical_path(path: str) -> bool:
    if not path.endswith((".csv", ".json", ".py")):
        return False
    markers = (
        "rel-ratebeer_user-place-liked_pairwise",
        "rel-ratebeer_user-place-liked",
        "legacy_task_specs",
        "task_inventory_with_artifacts",
        "existing_backend.py",
        "manifest",
        "provenance",
    )
    return any(marker in path for marker in markers)


def inspect_historical_json(
    *,
    plan,
    commit: str,
    path: str,
    text: str,
    passthrough_ids: tuple[str, ...],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    append_explicit_records(
        plan=plan,
        records=explicit_records_from_json(data),
        passthrough_ids=passthrough_ids,
        evidence=evidence,
        kind="git-json",
        location_prefix=f"git:{commit}:{path}",
    )


def inspect_historical_csv(
    *,
    plan,
    commit: str,
    path: str,
    text: str,
    passthrough_ids: tuple[str, ...],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        return
    fields = set(reader.fieldnames)
    if "primitive_id" not in fields:
        return
    source_field = first_present(
        fields,
        ("source_column", "input_column"),
    )
    output_field = first_present(
        fields,
        ("output_column", "feature_name", "column"),
    )
    if source_field is None and output_field is None:
        return
    for row_number, row in enumerate(reader, start=2):
        primitive_id = row.get("primitive_id")
        if primitive_id not in passthrough_ids:
            continue
        source_column = row.get(source_field) if source_field else None
        output_column = row.get(output_field) if output_field else None
        append_explicit_record(
            plan=plan,
            primitive_id=primitive_id,
            source_column=source_column or None,
            output_column=output_column or None,
            evidence=evidence,
            kind="git-csv",
            location=f"git:{commit}:{path}:{row_number}",
        )


def inspect_historical_python(
    *,
    plan,
    commit: str,
    path: str,
    text: str,
    passthrough_ids: tuple[str, ...],
    evidence: list[PassthroughBindingEvidence],
) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        append_explicit_records(
            plan=plan,
            records=explicit_records_from_scoped_mapping(value),
            passthrough_ids=passthrough_ids,
            evidence=evidence,
            kind="git-python-literal",
            location_prefix=f"git:{commit}:{path}",
        )


def explicit_records_from_json(data) -> tuple[dict[str, str | None], ...]:
    return explicit_records_from_scoped_mapping(data)


def explicit_records_from_scoped_mapping(
    data,
    *,
    in_matching_scope: bool = False,
) -> tuple[dict[str, str | None], ...]:
    if isinstance(data, dict):
        return explicit_records_from_mapping(
            data,
            in_matching_scope=in_matching_scope,
        )
    if isinstance(data, list):
        return tuple(
            record
            for item in data
            for record in explicit_records_from_scoped_mapping(
                item,
                in_matching_scope=in_matching_scope,
            )
        )
    return ()


def explicit_records_from_mapping(
    data,
    *,
    in_matching_scope: bool,
) -> tuple[dict[str, str | None], ...]:
    records: list[dict[str, str | None]] = []
    if not isinstance(data, dict):
        return ()

    current_scope = (
        in_matching_scope
        or mapping_declares_matching_task(data)
    )

    if current_scope:
        primitive_id = data.get("primitive_id")
        if isinstance(primitive_id, str):
            source_column = data.get("source_column")
            output_column = data.get("output_column")
            if output_column is None:
                output_column = data.get("feature_name")
            if (
                isinstance(source_column, str)
                or isinstance(output_column, str)
            ):
                records.append({
                    "primitive_id": primitive_id,
                    "source_column": (
                        source_column
                        if isinstance(source_column, str)
                        else None
                    ),
                    "output_column": (
                        output_column
                        if isinstance(output_column, str)
                        else None
                    ),
                })

        bindings = data.get("primitive_column_bindings")
        if isinstance(bindings, dict):
            records.extend(
                records_from_primitive_column_bindings(bindings)
            )

    for key, value in data.items():
        nested_scope = (
            current_scope
            or key == f"{DATASET}/{TASK}"
            or key == f"{DATASET}_{TASK}"
        )
        if isinstance(value, (dict, list)):
            records.extend(
                explicit_records_from_scoped_mapping(
                    value,
                    in_matching_scope=nested_scope,
                )
            )

    return tuple(records)


def mapping_declares_matching_task(data: dict) -> bool:
    dataset = data.get("dataset")
    task = data.get("task")
    if dataset == DATASET and task == TASK:
        return True

    task_key = data.get("task_key") or data.get("relbench_task")
    return task_key == f"{DATASET}/{TASK}"


def records_from_primitive_column_bindings(
    bindings: dict,
) -> tuple[dict[str, str | None], ...]:
    records: list[dict[str, str | None]] = []
    for primitive_id, value in bindings.items():
        if not (
            isinstance(primitive_id, str)
            and primitive_id.startswith("baseline::")
        ):
            continue
        for column in normalize_column_tuple(value):
            records.append({
                "primitive_id": primitive_id,
                "source_column": column,
                "output_column": column,
            })
    return tuple(records)


def normalize_column_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def append_explicit_records(
    *,
    plan,
    records: tuple[dict[str, str | None], ...],
    passthrough_ids: tuple[str, ...],
    evidence: list[PassthroughBindingEvidence],
    kind: str,
    location_prefix: str,
) -> None:
    for index, record in enumerate(records, start=1):
        primitive_id = record["primitive_id"]
        if primitive_id not in passthrough_ids:
            continue
        append_explicit_record(
            plan=plan,
            primitive_id=primitive_id,
            source_column=record["source_column"],
            output_column=record["output_column"],
            evidence=evidence,
            kind=kind,
            location=f"{location_prefix}:{index}",
        )


def append_explicit_record(
    *,
    plan,
    primitive_id: str,
    source_column: str | None,
    output_column: str | None,
    evidence: list[PassthroughBindingEvidence],
    kind: str,
    location: str,
) -> None:
    status = "proven" if source_column and output_column else "partial"
    evidence.append(
        PassthroughBindingEvidence(
            program_id=plan.program_id,
            primitive_id=primitive_id,
            source_column=source_column,
            output_column=output_column,
            evidence_kind=kind,
            evidence_location=location,
            status=status,
            notes=("explicit primitive-to-column mapping",),
        )
    )


def run_git(args: tuple[str, ...]) -> str:
    result = subprocess.run(
        ("git",) + args,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout


def first_present(
    fields: set[str],
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def describe_path(path: Path) -> str:
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    return label if path.exists() else f"{label} ABSENT"


def remote_read_only_command() -> str:
    return (
        "cd /home/evelyn/fdhg-icl-paper && "
        "micromamba run -n fdhg310 python - <<'PY'\n"
        "from pathlib import Path\n"
        "import hashlib\n"
        "paths = [\n"
        "  'outputs/e2e/rel-ratebeer_user-place-liked_pairwise',\n"
        "  'outputs/pairwise/rel-ratebeer_user-place-liked',\n"
        "  'outputs/dfs_agg/rel-ratebeer_user-place-liked_pairwise_sample',\n"
        "  'outputs/fdhg_heuristic/rel-ratebeer_user-place-liked_pairwise_sample',\n"
        "]\n"
        "for root in paths:\n"
        "    p = Path(root)\n"
        "    print('PATH', root, 'EXISTS', p.exists())\n"
        "    if p.exists():\n"
        "        for child in sorted(p.rglob('*')):\n"
        "            if child.is_file() and child.suffix in {'.parquet','.csv','.json'}:\n"
        "                digest = hashlib.sha256()\n"
        "                with child.open('rb') as fh:\n"
        "                    for chunk in iter(lambda: fh.read(1024 * 1024), b''):\n"
        "                        digest.update(chunk)\n"
        "                h = digest.hexdigest()\n"
        "                print('FILE', child, 'SHA256', h)\n"
        "                if child.suffix == '.parquet':\n"
        "                    import pyarrow.parquet as pq\n"
        "                    print('SCHEMA', child, pq.read_schema(child).names)\n"
        "                elif child.suffix in {'.csv','.json'}:\n"
        "                    print('HEAD', child, child.read_text(errors='replace')[:2000])\n"
        "PY"
    )


if __name__ == "__main__":
    main()
