from fdhg.gate import classification_gate, regression_gate


def test_classification_gate_selects_consistent_gain() -> None:
    assert classification_gate(
        [0.70, 0.71, 0.69, 0.72],
        [0.72, 0.73, 0.70, 0.74],
    )


def test_classification_gate_rejects_one_seed_failure() -> None:
    assert not classification_gate(
        [0.70, 0.71],
        [0.72, 0.70],
    )


def test_regression_gate_selects_consistent_reduction() -> None:
    assert regression_gate(
        [1.0, 1.1, 0.9, 1.2],
        [0.9, 1.0, 0.8, 1.1],
    )


def test_regression_gate_rejects_tie() -> None:
    assert not regression_gate(
        [1.0, 1.1],
        [1.0, 1.0],
    )


def test_consistent_gate_supports_minimized_metric() -> None:
    from fdhg.gate import consistent_improvement_gate

    assert consistent_improvement_gate(
        [0.60, 0.59, 0.61],
        [0.58, 0.57, 0.60],
        direction="minimize",
    )


def test_consistent_gate_rejects_minimized_metric_tie() -> None:
    from fdhg.gate import consistent_improvement_gate

    assert not consistent_improvement_gate(
        [0.60, 0.59],
        [0.58, 0.59],
        direction="minimize",
    )


def test_consistent_gate_honors_minimum_improvement() -> None:
    from fdhg.gate import consistent_improvement_gate

    assert not consistent_improvement_gate(
        [0.70, 0.70],
        [0.71, 0.72],
        direction="maximize",
        min_improvement=0.01,
    )


def test_consistent_gate_rejects_nan() -> None:
    import pytest

    from fdhg.gate import consistent_improvement_gate

    with pytest.raises(ValueError, match="finite"):
        consistent_improvement_gate(
            [0.70, float("nan")],
            [0.72, 0.73],
            direction="maximize",
        )


def test_lexicographic_gate_accepts_primary_improvement() -> None:
    from fdhg.gate import lexicographic_improvement_gate

    assert lexicographic_improvement_gate(
        [0.70, 0.71],
        [0.71, 0.72],
        [0.80, 0.80],
        [0.79, 0.79],
        primary_direction="maximize",
        secondary_direction="maximize",
    )


def test_lexicographic_gate_accepts_tied_primary_with_mrr_gain() -> None:
    from fdhg.gate import lexicographic_improvement_gate

    assert lexicographic_improvement_gate(
        [0.70, 0.71],
        [0.70, 0.71],
        [0.80, 0.81],
        [0.81, 0.82],
        primary_direction="maximize",
        secondary_direction="maximize",
    )


def test_lexicographic_gate_rejects_tied_primary_without_mrr_gain() -> None:
    from fdhg.gate import lexicographic_improvement_gate

    assert not lexicographic_improvement_gate(
        [0.70, 0.71],
        [0.70, 0.71],
        [0.80, 0.81],
        [0.80, 0.82],
        primary_direction="maximize",
        secondary_direction="maximize",
    )


def test_lexicographic_gate_rejects_primary_regression() -> None:
    from fdhg.gate import lexicographic_improvement_gate

    assert not lexicographic_improvement_gate(
        [0.70, 0.71],
        [0.69, 0.72],
        [0.80, 0.81],
        [0.90, 0.90],
        primary_direction="maximize",
        secondary_direction="maximize",
    )


def test_lexicographic_gate_supports_minimize_secondary() -> None:
    from fdhg.gate import lexicographic_improvement_gate

    assert lexicographic_improvement_gate(
        [0.80, 0.81],
        [0.80, 0.81],
        [0.40, 0.39],
        [0.39, 0.38],
        primary_direction="maximize",
        secondary_direction="minimize",
    )
