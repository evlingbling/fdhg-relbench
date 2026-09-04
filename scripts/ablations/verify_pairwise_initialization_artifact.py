#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DATASET = "rel-f1"
TASK = "driver-position"
SLUG = f"{DATASET}_{TASK}"

EXPECTED_AUTO = 5.08829369814525
EXPECTED_GREEDY = 5.081050623938082
EXPECTED_EDGES = [
    "standings:wins->points",
    "results:positionOrder->points",
    "results:points->number",
]
EXPECTED_INITIAL_PAIR = (
    "standings:wins->points||"
    "results:positionOrder->points"
)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root containing outputs/final-gate-51task-v2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    final_root = (
        args.artifact_root.resolve()
        / "outputs"
        / "final-gate-51task-v2"
    )

    task_root = final_root / SLUG

    greedy_file = (
        task_root
        / "strategies"
        / "greedy"
        / SLUG
        / "selected_variant.json"
    )

    discovery_file = (
        task_root
        / "discovery"
        / SLUG
        / "selected_variant.json"
    )

    greedy = load_json(greedy_file)
    discovery = load_json(discovery_file)

    scores = greedy["mean_scores"]

    auto_score = scores["auto_only"]
    greedy_score = (
        scores.get("auto_plus_fdhg")
        or scores.get("auto_plus_fdhg_greedy")
    )

    selected_edges = [
        str(x)
        for x in greedy.get(
            "strategy_selected_edge_ids", []
        )
    ]

    # Exact manuscript artifact checks.
    assert close(auto_score, EXPECTED_AUTO), (
        auto_score,
        EXPECTED_AUTO,
    )
    assert close(greedy_score, EXPECTED_GREEDY), (
        greedy_score,
        EXPECTED_GREEDY,
    )

    assert selected_edges == EXPECTED_EDGES, selected_edges

    assert (
        greedy.get("strategy_selected_edge_count")
        == len(EXPECTED_EDGES)
    )

    assert greedy.get("pairwise_rescue_used") is True
    assert (
        greedy.get("pairwise_rescue_reason")
        == "selected_pair_passed_gate"
    )
    assert (
        greedy.get("selected_initial_pair")
        == EXPECTED_INITIAL_PAIR
    )

    # No singleton passed before pairwise initialization.
    assert (
        greedy.get(
            "independent_screened_in_edge_ids", []
        )
        == []
    )

    # Train-only / leakage audit.
    assert greedy.get("test_split_accessed") is False
    assert (
        greedy.get(
            "official_validation_was_used_for_selection"
        )
        is False
    )

    assert discovery.get("test_split_accessed") is False
    assert (
        discovery.get(
            "official_validation_was_used_for_selection"
        )
        is False
    )

    summary = {
        "dataset": DATASET,
        "task": TASK,
        "metric": greedy.get("primary_metric"),
        "metric_direction": greedy.get(
            "metric_direction"
        ),
        "pairwise_rescue_off_auto_score": auto_score,
        "pairwise_rescue_on_greedy_score": greedy_score,
        "gain_rmse": auto_score - greedy_score,
        "singleton_edges_passing_before_rescue": 0,
        "pairwise_rescue_used": True,
        "pairwise_rescue_reason": greedy.get(
            "pairwise_rescue_reason"
        ),
        "selected_initial_pair": greedy.get(
            "selected_initial_pair"
        ),
        "selected_edge_count": len(selected_edges),
        "selected_edge_ids": selected_edges,
        "candidate_file_sha256": greedy.get(
            "candidate_file_sha256"
        ),
        "test_split_accessed": False,
        "official_validation_was_used_for_selection": False,
        "artifact_verified": True,
    }

    print(
        "\n===== TABLE 7 ARTIFACT VERIFICATION ====="
    )
    print(
        f"Auto / rescue-off fallback RMSE : "
        f"{auto_score:.10f}"
    )
    print(
        f"Greedy + pairwise rescue RMSE   : "
        f"{greedy_score:.10f}"
    )
    print(
        f"Gain                          : "
        f"{auto_score - greedy_score:.10f}"
    )
    print(
        f"Selected dependencies         : "
        f"{len(selected_edges)}"
    )
    print(
        "Initial pair                  : "
        f"{summary['selected_initial_pair']}"
    )
    print(
        "Test split accessed           : False"
    )
    print("STATUS                        : VERIFIED")

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print("WROTE:", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
