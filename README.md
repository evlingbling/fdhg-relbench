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

## Paper-to-code map

The table below maps manuscript experiments to their reproduction entry points. Exact manuscript numbers are retained in the preserved paper artifacts; reruns reconstruct the corresponding protocol using the current implementation.

| Manuscript result | Reproduction entry point | Notes |
|---|---|---|
| Table 5 — Candidate-pool sensitivity | `scripts/ablations/run_candidate_pool_sensitivity.sh` | Evaluates Bcand in {16,32,64} from a common ordered candidate pool and verifies prefix nesting. Computationally expensive. |
| Table 6 — Cross-fold consistency | `scripts/reproduce_ablations.sh` | Evaluates positive-fold requirements 1/3, 2/3, and 3/3. |
| Table 7 — Pairwise initialization | `scripts/ablations/verify_pairwise_initialization_artifact.py` | Verifies the exact preserved manuscript artifact. `run_pairwise_initialization.py` reruns enabled/disabled behavior with current code and is computationally expensive. |
| Table 8 — Auto-budget sensitivity | `scripts/reproduce_ablations.sh` | Evaluates Auto budgets 4, 8, 12, and 16. |
| Table 9 — Independent vs Greedy | `scripts/ablations/collect_independent_vs_greedy.py` | Aggregates the 15-task common support where both strategies produce non-empty FDHG augmentations. |
| Table 10 — Random-K | `scripts/reproduce_ablations.sh` | Four representative tasks and 20 random subsets per task. |
| Table 11 — Efficiency | `scripts/reproduce_efficiency.sh` | Reports runtime and peak-memory measurements. |
| Tables 14--15 — Predictor generalization | `scripts/reproduce_generalization.sh` | Reuses the frozen selected representation across downstream predictors. |

The standard ablation wrapper excludes the two most expensive selector-search ablations. To include them, run:

```bash
RUN_EXPENSIVE_ABLATIONS=1 \\
ARTIFACT_ROOT=/path/to/preserved/paper/artifacts \\
scripts/reproduce_ablations.sh
```

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
