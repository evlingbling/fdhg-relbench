from __future__ import annotations

from .ir import (
    CompiledTask,
    Primitive,
    PrimitiveFamily,
    TaskSpec,
)


BASELINE_OPERATIONS = [
    "count",
    "numeric_mean",
    "numeric_std",
    "numeric_max",
    "days_since_last",
]

HISTORY_OPERATIONS = [
    "window_count_short",
    "window_count_aligned",
    "window_count_long",
    "past_unique_values",
    "past_unique_neighbors",
    "mean_group_size",
    "max_group_size",
    "incoming_event_count",
    "past_unique_sources",
    "incoming_event_count_long",
]

STRUCTURAL_OPERATIONS = [
    "majority_confidence",
    "entropy",
    "conflict_count",
    "support_count",
    "top1_margin",
    "unique_count",
    "last_observed_value",
]


def derive_temporal_windows(
    horizon_days: int | None,
) -> list[int]:
    if horizon_days is None or horizon_days <= 0:
        return []

    short = max(1, round(horizon_days / 3))
    aligned = int(horizon_days)
    long_window = int(horizon_days * 4)

    if 350 <= long_window <= 380:
        long_window = 365

    return sorted({
        short,
        aligned,
        long_window,
    })


def append_pairwise_primitives(
    *,
    primitives: list[Primitive],
    task_spec: TaskSpec,
) -> None:
    pairwise = task_spec.pairwise

    if pairwise is None:
        return

    windows = derive_temporal_windows(
        task_spec.horizon_days
    )

    role_specs = [
        (
            "left",
            pairwise.left_key,
            pairwise.left_history,
        ),
        (
            "right",
            pairwise.target_right_key,
            pairwise.right_history,
        ),
    ]

    for role, target_key, history in role_specs:
        if history is None:
            continue

        if history.key is None:
            raise ValueError(
                f"Pairwise {role} history requires 'key'"
            )

        for window_days in windows:
            predicate = None

            if history.time_col is not None:
                predicate = (
                    f"target.{task_spec.target_time_col}"
                    f" - {window_days}d"
                    f" < {history.table}.{history.time_col}"
                    f" < target.{task_spec.target_time_col}"
                )

            primitives.append(
                Primitive(
                    primitive_id=(
                        "temporal::pairwise::"
                        f"{role}::count::{window_days}d"
                    ),
                    family=PrimitiveFamily.TEMPORAL,
                    operation="window_count",
                    source_table=history.table,
                    group_key=history.key,
                    event_time_col=history.time_col,
                    window_days=window_days,
                    temporal_predicate=predicate,
                    temporally_safe=(
                        history.time_col is not None
                    ),
                    metadata={
                        "binding": "pairwise_history",
                        "pairwise_role": role,
                        "target_key": target_key,
                    },
                )
            )

            if (
                history.related_col is not None
                and window_days in windows[:2]
            ):
                primitives.append(
                    Primitive(
                        primitive_id=(
                            "temporal::pairwise::"
                            f"{role}::unique_neighbors::"
                            f"{window_days}d"
                        ),
                        family=PrimitiveFamily.TEMPORAL,
                        operation="past_unique_neighbors",
                        source_table=history.table,
                        group_key=history.key,
                        event_time_col=history.time_col,
                        numeric_col=history.related_col,
                        window_days=window_days,
                        temporal_predicate=predicate,
                        temporally_safe=(
                            history.time_col is not None
                        ),
                        metadata={
                            "binding": "pairwise_history",
                            "pairwise_role": role,
                            "target_key": target_key,
                            "related_col": (
                                history.related_col
                            ),
                        },
                    )
                )

        primitives.append(
            Primitive(
                primitive_id=(
                    "temporal::pairwise::"
                    f"{role}::days_since_last"
                ),
                family=PrimitiveFamily.TEMPORAL,
                operation="days_since_last",
                source_table=history.table,
                group_key=history.key,
                event_time_col=history.time_col,
                temporal_predicate=(
                    (
                        f"{history.table}.{history.time_col}"
                        f" < target."
                        f"{task_spec.target_time_col}"
                    )
                    if history.time_col is not None
                    else None
                ),
                temporally_safe=(
                    history.time_col is not None
                ),
                metadata={
                    "binding": "pairwise_history",
                    "pairwise_role": role,
                    "target_key": target_key,
                },
            )
        )

    pair_history = pairwise.pair_history

    if pair_history is None:
        return

    if (
        pair_history.left_key is None
        or pair_history.right_key is None
    ):
        raise ValueError(
            "Pair history requires left_key and right_key"
        )

    pair_metadata = {
        "binding": "pairwise_history",
        "pairwise_role": "pair",
        "target_left_key": pairwise.left_key,
        "target_right_key": (
            pairwise.target_right_key
        ),
        "source_left_key": (
            pair_history.left_key
        ),
        "source_right_key": (
            pair_history.right_key
        ),
    }

    pair_temporal_predicate = (
        (
            f"{pair_history.table}.{pair_history.time_col}"
            f" < target.{task_spec.target_time_col}"
        )
        if pair_history.time_col is not None
        else None
    )

    primitives.append(
        Primitive(
            primitive_id=(
                "temporal::pairwise::pair::prior_count"
            ),
            family=PrimitiveFamily.TEMPORAL,
            operation="prior_pair_count",
            source_table=pair_history.table,
            event_time_col=pair_history.time_col,
            temporal_predicate=pair_temporal_predicate,
            temporally_safe=(
                pair_history.time_col is not None
            ),
            metadata=pair_metadata,
        )
    )

    primitives.append(
        Primitive(
            primitive_id=(
                "temporal::pairwise::pair::"
                "days_since_last"
            ),
            family=PrimitiveFamily.TEMPORAL,
            operation="pair_days_since_last",
            source_table=pair_history.table,
            event_time_col=pair_history.time_col,
            temporal_predicate=pair_temporal_predicate,
            temporally_safe=(
                pair_history.time_col is not None
            ),
            metadata=pair_metadata,
        )
    )


def build_candidate_program(
    task_spec: TaskSpec,
) -> CompiledTask:
    primitives: list[Primitive] = []

    # Generic relational baseline primitives.
    if task_spec.child_table is not None:
        operations = (
            task_spec.baseline_operations
            if task_spec.baseline_operations is not None
            else tuple(BASELINE_OPERATIONS)
        )
        for operation in operations:
            temporal_predicate = None

            if task_spec.child_time_col is not None:
                temporal_predicate = (
                    f"{task_spec.child_table}."
                    f"{task_spec.child_time_col}"
                    f" < target.{task_spec.target_time_col}"
                )

            primitives.append(
                Primitive(
                    primitive_id=f"baseline::{operation}",
                    family=PrimitiveFamily.BASELINE,
                    operation=operation,
                    source_table=task_spec.child_table,
                    group_key=task_spec.entity_key,
                    event_time_col=task_spec.child_time_col,
                    numeric_col=task_spec.numeric_col,
                    temporal_predicate=temporal_predicate,
                    temporally_safe=(
                        task_spec.child_time_col is not None
                    ),
                )
            )

    # Generic relational-history primitives.
    #
    # These are logical candidates. A backend/lowerer decides which
    # physical columns realize them for a particular task.
    for operation in HISTORY_OPERATIONS:
        primitives.append(
            Primitive(
                primitive_id=(
                    f"baseline::history::{operation}"
                ),
                family=PrimitiveFamily.BASELINE,
                operation=operation,
                source_table=task_spec.child_table,
                group_key=task_spec.entity_key,
                event_time_col=task_spec.child_time_col,
                temporal_predicate=(
                    (
                        f"{task_spec.child_table}."
                        f"{task_spec.child_time_col}"
                        f" < target."
                        f"{task_spec.target_time_col}"
                    )
                    if (
                        task_spec.child_table is not None
                        and task_spec.child_time_col is not None
                    )
                    else None
                ),
                temporally_safe=(
                    task_spec.child_time_col is not None
                ),
                metadata={
                    "binding": "task_relational_history",
                    "optional": True,
                },
            )
        )

    # Structural residual candidates.
    #
    # The actual AFD edge or task-history relation is bound during
    # lowering.
    for operation in STRUCTURAL_OPERATIONS:
        primitives.append(
            Primitive(
                primitive_id=(
                    f"structural::afd::{operation}"
                ),
                family=PrimitiveFamily.STRUCTURAL,
                operation=operation,
                group_key=task_spec.entity_key,
                metadata={
                    "binding": "selected_afd_edge",
                    "requires_structural_binding": True,
                },
            )
        )

    # Pairwise role-conditioned temporal candidates.
    append_pairwise_primitives(
        primitives=primitives,
        task_spec=task_spec,
    )

    # Standard single-entity temporal candidates.
    if task_spec.pairwise is None:
        temporal_windows = derive_temporal_windows(
            task_spec.horizon_days
        )
    else:
        temporal_windows = []

    for window_days in temporal_windows:
        temporal_predicate = None

        if (
            task_spec.child_table is not None
            and task_spec.child_time_col is not None
        ):
            temporal_predicate = (
                f"target.{task_spec.target_time_col}"
                f" - {window_days}d"
                f" < {task_spec.child_table}."
                f"{task_spec.child_time_col}"
                f" < target.{task_spec.target_time_col}"
            )

        primitives.append(
            Primitive(
                primitive_id=(
                    f"temporal::count::{window_days}d"
                ),
                family=PrimitiveFamily.TEMPORAL,
                operation="window_count",
                source_table=task_spec.child_table,
                group_key=task_spec.entity_key,
                event_time_col=task_spec.child_time_col,
                window_days=window_days,
                temporal_predicate=temporal_predicate,
                temporally_safe=(
                    task_spec.child_time_col is not None
                ),
            )
        )

    return CompiledTask(
        task_spec=task_spec,
        candidate_primitives=primitives,
    )
