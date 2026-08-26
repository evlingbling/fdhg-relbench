# FDHG RelBench

Reproducible research code for automatic relational feature engineering with functional dependencies on RelBench.

The pipeline compares Canonical DFS, Auto, Auto + Selected FDHG (Independent), and Auto + FDHG (Greedy). All feature and strategy selection uses train-only evidence. Official validation is reserved for final evaluation, and the test split is not accessed during selection.

## Installation

```bash
micromamba create -f environment.yml
micromamba activate fdhg310
pip install -e .
```

## Reproduction

Main 51-task sweep:

```bash
scripts/reproduce_main.sh
```

Ablations (cross-fold consistency, Auto-budget sensitivity, Random-K):

```bash
ARTIFACT_ROOT=/path/to/preserved/paper/artifacts scripts/reproduce_ablations.sh
```

Efficiency:

```bash
scripts/reproduce_efficiency.sh
```

Predictor generalization:

```bash
scripts/reproduce_generalization.sh
```

The generalization wrapper performs pre-flight validation, exports the frozen selected representation for each task, evaluates XGBoost and CatBoost over seeds 41--44, and verifies run completeness. Deterministic predictor limitations, such as CatBoost receiving an all-constant frozen design matrix, are recorded explicitly as structural skips rather than hidden as failures.

## Ablations

### Cross-fold consistency
Evaluates positive-fold thresholds 1/3, 2/3, and 3/3. We use 2/3 as a conservative majority criterion: 1/3 can admit unstable edges, while 3/3 can eliminate useful augmentations.

### Auto-budget sensitivity
Evaluates Auto feature budgets 4, 8, 12, and 16. The reported Auto score is the paired downstream evaluation score from the final FDHG manifest, not the Auto feature-selection objective.

### Random-K
Uses 4 representative tasks × 20 seeds = 80 trials. Random subsets are sampled from the same frozen candidate pool, with task-specific K fixed to the preserved manuscript experiment protocol and no post-sampling screening.

## Efficiency

The efficiency benchmark compares DFS, Auto, Auto + All FDHG, Auto + Selected FDHG (Independent), and Auto + FDHG (Greedy). Memory measurements use `/usr/bin/time -v`, so Linux is recommended.

## Selection and leakage protocol

Train-only inner folds are used for Auto feature selection, FDHG screening, Greedy edge selection, and final feature-construction strategy selection. Official validation is not used for selection. The RelBench test split is not accessed during feature or strategy selection.

Relevant manifest fields include:
- `test_split_accessed`
- `official_validation_was_used_for_selection`

## Reproducibility note

Preserved paper artifacts are the source of the exact numerical values reported in the manuscript. Current-code reruns reconstruct the same protocol, frozen candidate pools, Auto representations, folds, selection rules, and random seeds, but some numerical scores may differ because implementation bugs and reproducibility issues were fixed after some original experiments. Current code should not be modified merely to recover an older numerical value when the protocol is otherwise equivalent.

## Testing

```bash
pytest -q
python -m py_compile scripts/ablations/*.py
bash -n scripts/reproduce_main.sh
bash -n scripts/reproduce_ablations.sh
bash -n scripts/reproduce_efficiency.sh
bash -n scripts/reproduce_generalization.sh
git diff --check
```

Generated experiment outputs are written under `outputs/` and are gitignored unless explicitly promoted to paper-facing artifacts.
