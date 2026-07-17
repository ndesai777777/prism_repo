# PRISM Engineering Audit & Refactoring Plan – Uplift Modeling Workflow

## Executive Summary

This document is a complete engineering review of `Uplift Model Code_rh06032026.ipynb`. No code has been modified, no files renamed, no refactoring performed. The purpose is to serve as the engineering specification for a future refactoring effort that will make this notebook clean, reproducible, and ready for plug-and-play dataset replacement.

---

## 1. Overall Notebook Architecture

### Current Structure

The notebook currently uses a numbered-section layout (Sections 1–20+) with additional unnumbered "WRITE-UP" and "DASHBOARD" sections inserted between and after the numbered flow:

| Section | Purpose |
|---------|---------|
| 1 | Install/check packages |
| 2 | Load packages |
| 3 | File paths |
| 4 | Helper functions |
| 5 | Read Excel file |
| 6 | Check required columns |
| 7 | Basic cleanup |
| 8 | Derive date features |
| 9 | Select predictors |
| 10 | Data type handling |
| 11 | Train/test split |
| 12 | Separate treated/control |
| 13 | Build model matrices |
| (unnumbered) | Data review summary |
| (unnumbered) | Predictor data dictionary & summary |
| (unnumbered) | Predictor distribution visuals |
| 14 | Train XGBoost models (baseline) |
| (unnumbered) | XGBoost CV grid search |
| (unnumbered) | Train treated model with CV grid |
| (unnumbered) | Train control model with CV grid |
| (unnumbered) | Test AUC for CV-tuned models |
| (unnumbered) | CPU-only GLMNet comparison |
| (unnumbered) | GLMNet test AUC |
| 15 | Score test sets |
| 16 | Decile summaries |
| (unnumbered) | X-Learner treatment effect comparison |
| (unnumbered) | Probability calibration & Brier scores |
| (unnumbered) | Factual outcome model diagnostics |
| (unnumbered) | Observed treated-control gap by decile |
| 17 | Variable importance |
| 18 | Score full files |
| 19 | Write outputs |
| 20 | Interpretation |
| (unnumbered) | Dashboard views |
| (unnumbered) | ROI per decile |
| (unnumbered) | Top benefit decile summary |
| (unnumbered) | Risk model driver outputs |
| (unnumbered) | Benefit score driver importance |
| (unnumbered) | X-Learner benefit score driver importance |
| (unnumbered) | GLMNet benefit score SHAP contributions |
| (unnumbered) | Model evaluation summary |
| (unnumbered) | Model recommendation summary |
| (unnumbered) | Risk tier vs benefit group outputs |
| (unnumbered) | Synthetic true-benefit validation |

### Issues Identified

1. **Inconsistent numbering**: The first 20 sections are numbered, but approximately 15 additional major analytical sections are inserted without numbers, breaking the logical flow.
2. **Mixed concerns in single cells**: Section 4 (Helper Functions) contains ~200 lines mixing GPU configuration flags, utility functions, model-fitting functions, and a custom pipeline class—all in one cell.
3. **Duplicated logic**: `safe_as_date`, `present_columns`, and `is_binary_indicator_column` / `is_binary_predictor` are defined in the notebook despite equivalent functions existing in `_prism_model_utils.py`.
4. **Hidden assumptions**: Section 8 derives date features and then immediately reverts the dataframe with `df = df_raw.copy()`. This is dead code that appears intentional ("reverse for now") but wastes reader attention and could silently break.
5. **Non-linear execution**: The notebook trains baseline XGBoost models in Section 14, then immediately retrains them with CV grid search in the next cells, overwriting the earlier `model_treated`/`model_control` variables. The baseline training is effectively dead code.
6. **Large monolithic cells**: The X-Learner section is a single massive cell (~200 lines) mixing model training, scoring, output-saving, and charting.

### Recommendations

| Recommendation | Why | Benefit | Risk |
|---|---|---|---|
| Adopt Task 1–8 structure from causal forest notebook | Consistency across PRISM workflows | Immediately navigable by any analyst | Medium – requires careful reordering |
| Extract model-fitting and SHAP logic into helper functions in `_prism_model_utils.py` | Reduces notebook length by ~40% | Single source of truth, easier testing | Low – pure extraction |
| Remove dead code in Section 8 and baseline Section 14 training | Reduces confusion | Cleaner execution path | Low – no behavior change |
| Break large cells into single-responsibility cells | Readability | Easier debugging, clearer markdown narrative | Low |

---

## 2. Notebook Organization – Proposed Task Structure

Based on the reference notebooks (`PRISM_Causal_Forest_Modeling_Workflow.ipynb` and `PRISM_Doubly_Robust_Modeling_Workflow.ipynb`), the uplift notebook should be reorganized into:

| Proposed Task | Contents | Maps From Current Sections |
|---|---|---|
| **Task 1: Understanding And Explaining The T-Learner Framework** | Sign convention, methodology description, Mermaid flowchart | Sections 1–3 (preamble) + Section 20 (interpretation) |
| **Task 2: Data Review** | Load data, check columns, cleanup, select predictors, data type handling, predictor data dictionary, distribution visuals, data review summary CSV | Sections 5–10, "WRITE-UP DATA REVIEW SUMMARY", "PREDICTOR DATA DICTIONARY", "PREDICTOR DISTRIBUTION VISUALS" |
| **Task 3: Factual Outcome Model Performance** | Train/test split, model training (XGBoost CV + GLMNet), test AUC, Brier scores, calibration, factual diagnostics, event counts | Sections 11–14, CV grid search, GLMNet comparison, "PROBABILITY CALIBRATION", "FACTUAL DIAGNOSTICS" |
| **Task 4: Treatment Effect Analysis** | Score test sets, build uplift results, X-Learner training and scoring, T-vs-X consistency summary | Section 15, "X-LEARNER TREATMENT EFFECT COMPARISON" |
| **Task 5: HTE Decile And High-Value Subgroup Analysis** | Decile summaries, observed gap analysis, uplift curve, top benefit decile summary | Section 16, "OBSERVED TREATED-CONTROL GAP", "TOP BENEFIT DECILE SUMMARY" |
| **Task 6: Variable Importance And Explainability** | SHAP risk drivers, benefit score drivers, X-Learner benefit drivers, GLMNet SHAP contributions | Section 17, "RISK MODEL DRIVER OUTPUTS", "BENEFIT SCORE DRIVER IMPORTANCE", "X-LEARNER BENEFIT DRIVER", "GLMNET BENEFIT SCORE SHAP" |
| **Task 7: Business Value Assessment** | ROI per decile, risk tier vs benefit group, targeting summaries, model evaluation summary, model recommendation summary | "ROI PER DECILE", "RISK TIER VS BENEFIT GROUP", "MODEL EVALUATION SUMMARY", "MODEL RECOMMENDATION SUMMARY" |
| **Task 8: Synthetic True-Benefit Validation** | True-benefit formula comparison (synthetic data only) | "SYNTHETIC TRUE-BENEFIT VALIDATION" |

### Why

- The causal forest and doubly robust notebooks already follow this pattern. Adopting it for uplift ensures any analyst familiar with one workflow can navigate all three.
- Markdown cells before each task provide context for non-technical reviewers.

### Expected Benefits

- Self-documenting notebook structure.
- Clear separation of concerns (data prep vs modeling vs analysis vs business value).
- Easier to skip or rerun individual analytical tasks.

### Possible Risks

- Reordering cells may temporarily break variable dependencies. Mitigation: ensure variables flow top-to-bottom with a single clean execution pass.
- The numbering will diverge from the R script sections. Mitigation: document the R-Python mapping in a comment at the top.

---

## 3. Inputs

### Current Input Configuration

Inputs are scattered across multiple cells:

| Input | Location | Current Value / Source |
|---|---|---|
| Dataset URL | Section 3 via `GITHUB_XLSX_URL` from utils | Raw GitHub URL to Excel file |
| Output root | Section 3 | `PROJECT_ROOT / 'Outputs' / 'Uplift' / 'Python'` |
| Random seed | Section 4 (helper functions cell) and Section 11 | `123` (hardcoded in multiple places) |
| Train fraction | Section 11 | `0.70` |
| GPU flag | Section 4 | `REQUIRE_GPU_FOR_XGBOOST = True` |
| CPU comparison flag | Section 4 | `RUN_CPU_ONLY_COMPARISON_MODELS = True` |
| CUDA device | Section 4 | `XGBOOST_CUDA_DEVICE = 0` |
| CV folds | Embedded in `fit_xgb_cv_grid`, `fit_elastic_net` | `5` (hardcoded in function defaults) |
| Max boosting rounds | Embedded in grid search call | `500` |
| XGBoost grid | Unnumbered cell after Section 14 | `product([3,4,5], [0.03,0.05,0.10], [1,5])` |
| Cost per ED visit | "ROI PER DECILE" section | `1200` |
| Cost per intervention | "ROI PER DECILE" section | `250` |
| Propensity clip bounds | `clipped_propensity` function | `lower=0.05, upper=0.95` |
| Risk tier thresholds | "RISK TIER" section | Hardcoded DataFrame |
| Outcome column | Section 6 | `'outcome_ed_90d'` |
| Treatment column | Section 6 | `'intervention_flag'` |

### Recommendations

All configurable values should be consolidated into a single **Configuration** cell at the top of the notebook (immediately after the package import cell). This mirrors the pattern in the causal forest notebook.

Proposed configuration block:

```python
# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION – Edit only this section when changing inputs
# ═══════════════════════════════════════════════════════════════════════
SEED = 123
TRAIN_FRACTION = 0.70
DATASET_PATH = GITHUB_XLSX_URL  # or local path
OUTPUT_DIR = PROJECT_ROOT / 'Outputs' / 'Uplift' / 'Python'
OUTCOME_COLUMN = 'outcome_ed_90d'
TREATMENT_COLUMN = 'intervention_flag'
REQUIRE_GPU_FOR_XGBOOST = True
XGBOOST_CUDA_DEVICE = 0
RUN_CPU_ONLY_COMPARISON_MODELS = True
CV_FOLDS = 5
MAX_BOOST_ROUNDS = 500
COST_PER_ED_VISIT = 1200
COST_PER_INTERVENTION = 250
PROPENSITY_CLIP_LOWER = 0.05
PROPENSITY_CLIP_UPPER = 0.95
```

---

## 4. Outputs

### CSV Outputs – Audit Table

| Output | Purpose | Keep | Remove | Optional | Comments |
|--------|---------|:----:|:------:|:--------:|---------|
| `data_review_summary.csv` | Dataset-level summary stats | ✓ | | | Core write-up artifact |
| `factual_event_count_summary.csv` | Train/test event counts by group | ✓ | | | Needed for calibration context |
| `factual_prediction_separation.csv` | AUC, mean predictions by actual class | ✓ | | | Model diagnostic |
| `factual_prediction_ranges.csv` | Min/max/quantiles of predictions | | | ✓ | Useful but secondary; could merge into separation CSV |
| `model_evaluation_summary.csv` | Cross-model comparison of all metrics | ✓ | | | Key decision artifact |
| `model_recommendation_summary.csv` | Human-readable model assessment | ✓ | | | Reporting artifact |
| `risk_tier_thresholds.csv` | Static lookup table | | | ✓ | Only relevant if risk tiers are used downstream |
| `risk_tier_population_summary.csv` | Members per risk tier | ✓ | | | Business context |
| `glmnet_true_benefit_validation_summary.csv` | Synthetic-only validation | | | ✓ | Remove once GenRocket data arrives |
| `glmnet_true_benefit_scored_test_comparison.csv` | Member-level true vs predicted benefit | | | ✓ | Debugging/synthetic only |
| `glmnet_true_benefit_decile_overlap_summary.csv` | Decile overlap with true benefit | | | ✓ | Synthetic only |
| `glmnet_shap_true_driver_alignment.csv` | SHAP vs true driver alignment | | | ✓ | Synthetic only |
| `glmnet_true_driver_recovery_summary.csv` | Driver recovery metrics | | | ✓ | Synthetic only |

#### Per-Model Outputs (T-Learner/XGBoost and T-Learner/GLMNet folders)

| Output | Purpose | Keep | Remove | Optional | Comments |
|--------|---------|:----:|:------:|:--------:|---------|
| `uplift_scored_output.csv` | Full-population scored file | ✓ | | | Primary deliverable |
| `uplift_decile_summary.csv` | Decile-level aggregation | ✓ | | | Primary deliverable |
| `uplift_roi_by_decile.csv` | ROI calculation per decile | ✓ | | | Business value artifact |
| `top_benefit_decile_summary.csv` | Top-decile deep dive | ✓ | | | Reporting |
| `uplift_observed_gap_by_decile.csv` | Observed gap with CIs | ✓ | | | Key validation |
| `uplift_curve_by_decile.csv` | Cumulative uplift curve data | ✓ | | | Validation |
| `uplift_curve_summary.csv` | AUC of uplift curve | ✓ | | | Summary metric |
| `model_brier_scores.csv` | Brier score per group | ✓ | | | Calibration |
| `calibration_by_decile.csv` | Pred vs obs by risk decile | ✓ | | | Calibration |
| `calibration_summary.csv` | Mean calibration error | ✓ | | | Calibration |
| `shap_importance_treated_control_models.csv` | SHAP for risk models | ✓ | | | Explainability |
| `shap_importance_benefit_score.csv` | SHAP for benefit score | ✓ | | | Explainability |

#### Per-Model Outputs (X-Learner/XGBoost and X-Learner/GLMNet folders)

| Output | Purpose | Keep | Remove | Optional | Comments |
|--------|---------|:----:|:------:|:--------:|---------|
| `xlearner_scored_test_output.csv` | X-Learner test scores | ✓ | | | Consistency check |
| `xlearner_decile_summary.csv` | X-Learner deciles | ✓ | | | Consistency check |
| `shared_propensity_scores.csv` | Propensity scores for all members | ✓ | | | Shared across frameworks |
| `xlearner_vs_tlearner_consistency_summary.csv` | Correlation/overlap metrics | ✓ | | | Key validation |
| `xlearner_benefit_driver_importance.csv` | GLMNet X-Learner drivers | ✓ | | | Explainability |

#### Predictor Distribution Outputs

| Output | Purpose | Keep | Remove | Optional | Comments |
|--------|---------|:----:|:------:|:--------:|---------|
| `predictor_data_dictionary.csv` | Variable metadata | ✓ | | | Reference artifact |
| `numeric_predictor_summary.csv` | Descriptive stats for numeric vars | ✓ | | | Data review |
| `categorical_predictor_summary.csv` | Descriptive stats for categorical vars | ✓ | | | Data review |
| `predictor_distribution_visual_index.csv` | Index of generated plots | | | ✓ | Convenience only |
| `numeric_predictor_distributions.pdf` | All numeric histograms | ✓ | | | Data review |
| `categorical_predictor_distributions.pdf` | All categorical bar charts | ✓ | | | Data review |
| Individual `.png` per predictor | Same content as PDFs | | | ✓ | Redundant if PDFs are kept; adds ~50 files |

### Chart/Graph Outputs – Audit Table

| Chart | Purpose | Keep | Remove | Optional | Comments |
|-------|---------|:----:|:------:|:--------:|---------|
| `dashboard_avg_benefit_by_decile.png` | Primary visual | ✓ | | | Core |
| `dashboard_observed_ed_rate_by_decile.png` | Context visual | ✓ | | | Core |
| `dashboard_treated_pct_by_decile.png` | Treatment penetration | ✓ | | | Core |
| `dashboard_predicted_treated_vs_control.png` | Treated vs control predictions | ✓ | | | Core |
| `dashboard_roi_net_savings_by_decile.png` | Business value | ✓ | | | Core |
| `dashboard_shap_treated_model.png` | Risk drivers – treated | ✓ | | | Core |
| `dashboard_shap_control_model.png` | Risk drivers – control | ✓ | | | Core |
| `dashboard_shap_benefit_score.png` | Benefit drivers | ✓ | | | Core |
| `dashboard_calibration_plot.png` | Calibration visual | ✓ | | | Core |
| `dashboard_observed_gap_by_decile.png` | Observed gap with CIs | ✓ | | | Validation |
| `dashboard_uplift_curve_by_decile.png` | Cumulative uplift curve | ✓ | | | Validation |
| `dashboard_xlearner_benefit_drivers.png` | X-Learner drivers | | | ✓ | Secondary; only if X-Learner is kept |
| `dashboard_t_vs_x_benefit_driver_comparison.png` | Side-by-side driver comparison | | | ✓ | Secondary |
| `dashboard_tlearner_risk_tier_by_benefit_group.png` | Risk tier segmentation | ✓ | | | Business reporting |
| `dashboard_glmnet_tlearner_global_benefit_shap.png` | SHAP permutation importance | | | ✓ | Adds ~15min runtime |
| Individual predictor histograms/bar charts | Per-variable distributions | | | ✓ | Excessive if PDFs exist |

### Outputs Identified as Unnecessary or Excessive

1. **Individual predictor PNG files** (~50 files): The same content is already in the consolidated PDFs. Recommendation: generate PDFs only, or gate individual PNGs behind a `SAVE_INDIVIDUAL_PREDICTOR_PLOTS = False` flag.

2. **Synthetic true-benefit CSVs** (5 files): These are only meaningful with the current synthetic dataset and will be invalid once GenRocket data arrives. Recommendation: gate behind a `SYNTHETIC_VALIDATION_ENABLED = True` flag; default to `False` once real data is loaded.

3. **GLMNet global benefit SHAP via permutation** (`glmnet_tlearner_global_benefit_shap_importance.csv`, `glmnet_xlearner_global_benefit_shap.csv`, member-level SHAP values): This is extremely slow (permutation SHAP on a wrapped predict function). The coefficient-contribution approach already provides an exact decomposition for linear models. Recommendation: mark as optional or remove.

### Outputs Recommended for Addition

| Missing Output | Purpose | Priority |
|---|---|---|
| `hyperparameter_tuning_summary.csv` | XGBoost CV grid search results for both treated/control models | High – aids reproducibility |
| `predictor_inventory.csv` | List of features used in the final model matrix (matching causal forest output) | High – aids dataset replacement |
| `configuration_snapshot.csv` | Dump of all config constants at runtime | Medium – aids reproducibility |

---

## 5. Output Folder Organization

### Current Structure

```
Outputs/Uplift/Python/
├── data_review_summary.csv
├── factual_event_count_summary.csv
├── factual_prediction_ranges.csv
├── factual_prediction_separation.csv
├── model_evaluation_summary.csv
├── model_recommendation_summary.csv
├── risk_tier_*.csv
├── glmnet_true_benefit_*.csv
├── Predictor_Distributions/
│   ├── predictor_data_dictionary.csv
│   ├── numeric_predictor_summary.csv
│   ├── categorical_predictor_summary.csv
│   ├── *.pdf
│   ├── Numeric/ (individual PNGs)
│   └── Categorical/ (individual PNGs)
├── T-Learner/
│   ├── XGBoost/ (CSVs + PNGs)
│   └── GLMNet/ (CSVs + PNGs)
└── X-Learner/
    ├── shared_propensity_scores.csv
    ├── xlearner_vs_tlearner_consistency_summary.csv
    ├── XGBoost/ (CSVs + PNGs)
    └── GLMNet/ (CSVs + PNGs)
```

### Issues

1. **Loose files at the root level**: `data_review_summary.csv`, `factual_*`, `model_*`, `risk_tier_*`, and `glmnet_true_benefit_*` CSVs sit at the Python root with no subfolder grouping.
2. **Inconsistent naming across models**: XGBoost and GLMNet folders share identically-named files (`uplift_scored_output.csv`, `uplift_decile_summary.csv`) which is good, but the root-level files don't follow a prefix convention.
3. **Predictor Distributions as a subfolder** is good, but individual PNGs in `Numeric/` and `Categorical/` subfolders are excessive.
4. **Legacy T-Learner folder at `Outputs/T-Learner/`** still exists alongside the new `Outputs/Uplift/Python/T-Learner/`. This creates confusion about which is canonical.

### Recommended Organization

```
Outputs/Uplift/Python/
├── 01_Data_Review/
│   ├── data_review_summary.csv
│   ├── predictor_data_dictionary.csv
│   ├── predictor_inventory.csv
│   ├── numeric_predictor_summary.csv
│   ├── categorical_predictor_summary.csv
│   ├── numeric_predictor_distributions.pdf
│   └── categorical_predictor_distributions.pdf
├── 02_Model_Performance/
│   ├── factual_event_count_summary.csv
│   ├── factual_prediction_separation.csv
│   ├── hyperparameter_tuning_summary.csv
│   └── configuration_snapshot.csv
├── 03_T-Learner/
│   ├── XGBoost/
│   │   ├── uplift_scored_output.csv
│   │   ├── uplift_decile_summary.csv
│   │   ├── (all current XGBoost CSVs and PNGs)
│   └── GLMNet/
│       ├── (all current GLMNet CSVs and PNGs)
├── 04_X-Learner/
│   ├── shared_propensity_scores.csv
│   ├── xlearner_vs_tlearner_consistency_summary.csv
│   ├── XGBoost/
│   └── GLMNet/
├── 05_Business_Value/
│   ├── model_evaluation_summary.csv
│   ├── model_recommendation_summary.csv
│   ├── risk_tier_thresholds.csv
│   ├── risk_tier_population_summary.csv
│   └── risk_tier_benefit_group_summary.csv
└── 06_Synthetic_Validation/ (removable)
    ├── glmnet_true_benefit_validation_summary.csv
    └── (other synthetic-only CSVs)
```

This structure lets an analyst immediately navigate by analytical concern.

---

## 6. Helper Functions

### Functions Defined in Notebook That Duplicate `_prism_model_utils.py`

| Function in Notebook | Equivalent in Utils | Action |
|---|---|---|
| `safe_as_date(values)` | `safe_as_date(values)` (Line 82) | Remove from notebook |
| `present_columns(columns, df)` | `present_columns(columns, df)` (Line 137) | Remove from notebook |
| `is_binary_indicator_column(series)` | No exact match, but `to_binary` handles conversion | Keep in notebook or extract |

### Functions That Should Move to `_prism_model_utils.py`

| Function | Reason |
|---|---|
| `fit_xgb_cv_grid(...)` | Reused across T-learner and potentially other workflows; general-purpose XGBoost hyperparameter tuning |
| `build_uplift_results(...)` | Core T-learner result construction; should be shared |
| `summarize_uplift_deciles(...)` | Standard decile summary; reusable |
| `build_roi_summary(...)` | ROI calculation used by multiple models |
| `wilson_score_interval(...)` | General statistical utility |
| `difference_in_proportions_ci(...)` | General statistical utility |
| `observed_gap_by_decile(...)` | Used by both T-learner and X-learner outputs |
| `uplift_curve_by_decile(...)` | Used by both T-learner and X-learner |
| `brier_scores_for_results(...)` | General model evaluation |
| `calibration_tables_for_results(...)` | General model evaluation |

### Functions That Are Notebook-Specific (Keep in Notebook)

| Function | Reason |
|---|---|
| `fit_elastic_net(...)` | GLMNet comparison is specific to this workflow's structure |
| `PrefitScaledLogisticPipeline` | Custom wrapper for the GLMNet prefit pattern |
| `PrefitScaledRegressionPipeline` | Custom wrapper for X-Learner GLMNet regression |
| `glmnet_contribution_importance_frame(...)` | GLMNet-specific attribution |
| `compute_synthetic_true_benefit(...)` | Synthetic data only; temporary |

---

## 7. Constants

### Constants Currently Scattered Throughout the Notebook

| Constant | Current Location | Value |
|---|---|---|
| `SEED` | Multiple cells (11, 14, grid search, elastic-net, propensity) | `123` |
| `TRAIN_FRACTION` | Section 11 | `0.70` |
| `REQUIRE_GPU_FOR_XGBOOST` | Section 4 | `True` |
| `RUN_CPU_ONLY_COMPARISON_MODELS` | Section 4 | `True` |
| `XGBOOST_CUDA_DEVICE` | Section 4 | `0` |
| `cost_per_ed_visit` | "ROI PER DECILE" cell | `1200` |
| `cost_per_intervention` | "ROI PER DECILE" cell | `250` |
| `nrounds_max` | Grid search call | `500` |
| `nfold` | Multiple defaults | `5` |
| `lower`/`upper` (propensity clip) | `clipped_propensity` function | `0.05` / `0.95` |
| `n=20` (top benefit display count) | `print_highest_benefit` | `20` |
| Risk tier boundaries | "RISK TIER" section | `[35, 55, 75]` |
| `max_items=5` (predictor examples) | `compact_examples` function | `5` |
| `top_n=20` (categorical bar chart) | `plot_categorical_distribution` | `20` |

### Recommendation

Create a single `# === CONFIGURATION ===` cell containing ALL of these as uppercase named constants. Every downstream reference should use the constant name rather than a magic number. This makes the notebook auditable at a glance and trivially adjustable.

### Additional Constants to Centralize

- `CANDIDATE_PREDICTORS_ALL` – the master predictor list from Section 9
- `REQUIRED_FIELDS` – from Section 6
- `RISK_TIER_THRESHOLDS` – the boundary DataFrame
- `BENEFIT_GROUP_DECILE_MAPPING` – deciles 1-2 = High, 3-7 = Medium, 8-10 = Low

---

## 8. Dead Code

| Item | Location | Type | Recommendation |
|---|---|---|---|
| Section 8 date feature derivation followed by `df = df_raw.copy()` | Section 8 | Dead code block | Remove entire section or integrate properly |
| Baseline XGBoost training in Section 14 (overwritten by CV grid) | Section 14 | Obsolete code | Remove; only the CV-tuned models are used |
| `# Backward-compatible aliases` (5+ occurrences) | Sections 15, 16, 17, 18, 19 | Legacy variables | Remove after confirming nothing downstream uses the aliases |
| `output_path = xgboost_output_path` alias | Section 3 | Legacy alias | Remove |
| `summary_path = xgboost_summary_path` alias | Section 3 | Legacy alias | Remove |
| `results_test = results_test_xgboost` alias | Section 15 | Legacy alias | Remove |
| `decile_summary = decile_summary_xgboost` alias | Section 16 | Legacy alias | Remove |
| `importance_treated = importance_treated_xgboost` alias | Section 17 | Legacy alias | Remove |
| `scored_full = scored_full_xgboost` alias | Section 18 | Legacy alias | Remove |
| `roi_summary = roi_summary_xgboost` alias | "ROI" section | Legacy alias | Remove |
| `from itertools import product` | Section 2 | Used only for grid construction; not dead but could be inlined | Keep |
| `model_df.columns` standalone cell (Section 10) | Section 10 | Debugging output | Remove |

### Unused Imports

None found. All imports in Section 2 are used downstream.

---

## 9. Dataset Replacement Readiness

### Potential Breakpoints When Replacing with GenRocket Dataset

| Issue | Location | Severity | Details |
|---|---|---|---|
| **Hardcoded predictor list** | Section 9 (`candidate_predictors_all`) | High | 56 column names are hardcoded. If the GenRocket dataset uses different names or omits some, the notebook will silently drop them without warning beyond a print statement. |
| **Hardcoded outcome/treatment names** | Section 6 | Medium | `'outcome_ed_90d'` and `'intervention_flag'` are assumed. If GenRocket uses different column names, it will immediately fail. |
| **Date column assumptions** | Section 7 | Medium | Assumes `'index_date'`, `'intervention_start_date'`, `'intervention_end_date'` exist with parseable date formats. |
| **Hardcoded `possible_numeric_cols`** | Section 10 | Medium | 24 column names are explicitly listed as expected numeric. New dataset may have different names. |
| **Synthetic true-benefit formula** | "SYNTHETIC TRUE-BENEFIT VALIDATION" | High | Uses `0.02 + 0.018 * ed_visits_last_6m + ...` which is meaningless for GenRocket data. Will produce invalid validation results. |
| **Risk tier boundaries** | "RISK TIER" section | Medium | Assumes `current_risk_score` exists and that thresholds at 35/55/75 are meaningful for the new data distribution. |
| **Predictor category/description lookups** | "PREDICTOR DATA DICTIONARY" section | Low | Hardcoded dictionaries; new columns won't have descriptions. |
| **Flag column detection by suffix** | Section 10 (`column.endswith('_flag')`) | Low | Assumes GenRocket uses `_flag` suffix convention. |
| **Assumption of balanced classes** | `fit_xgb_cv_grid` and `fit_elastic_net` | Medium | Both check `class_counts.min() >= 2` for CV folds but don't handle extreme imbalance gracefully. New data may have very different outcome prevalence. |
| **GitHub URL as data source** | Section 3 | Low | Will need to change to local path or new URL. Already uses a constant, so this is a one-line change. |

### Recommendations for Plug-and-Play Compatibility

1. **Dynamic predictor detection**: Instead of hardcoding `candidate_predictors_all`, derive candidates programmatically:
   - Exclude the outcome column, treatment column, ID columns, and date columns.
   - Use all remaining columns as candidates.
   - Optionally keep a `PREDICTOR_OVERRIDE` list for explicit inclusion/exclusion.

2. **Column validation with clear error messages**: Replace the silent `[col for col in candidates if col in df.columns]` pattern with an explicit report showing which expected columns are missing and which unexpected columns are present.

3. **Gate synthetic validation**: Wrap the true-benefit validation in `if SYNTHETIC_VALIDATION_ENABLED:` so it's automatically skipped for real data.

4. **Parameterize risk tier boundaries**: Move thresholds to the configuration section so they can be adjusted per dataset.

5. **Add a schema validation cell**: After loading the dataset, validate that required columns exist, outcome is binary, treatment is binary, and numeric columns are actually numeric. Fail fast with actionable error messages.

---

## 10. Reproducibility

### Can Another Developer Clone → Replace Dataset → Run → Get Outputs?

**Current answer: Partially, with caveats.**

### Obstacles

| Obstacle | Severity | Details |
|---|---|---|
| **GPU requirement** | High | `REQUIRE_GPU_FOR_XGBOOST = True` will fail on machines without NVIDIA CUDA. The notebook does not gracefully fall back to CPU-only XGBoost. |
| **No `requirements.txt` or `pyproject.toml`** | Medium | Package versions are not pinned. `_prism_model_utils.py` must be discoverable. |
| **Working directory assumption** | Medium | `project_root()` uses `Path.cwd()` heuristics. Running from a different directory may fail. |
| **Section 8 dead code** | Low | The `df = df_raw.copy()` at the end of Section 8 silently discards all date feature work. A new developer would be confused. |
| **No environment specification** | Medium | Python version, CUDA version, XGBoost version with GPU support – none documented. |
| **SHAP optional dependency** | Low | SHAP is imported with `try/except` and gracefully skipped, which is good. |
| **Large runtime** | Medium | Permutation SHAP for GLMNet benefit scores adds significant runtime (~15+ min). No progress indicator. |

### Recommendations

1. **Add a `requirements.txt`** at the repo root listing all dependencies with pinned versions.
2. **Add GPU fallback**: If `REQUIRE_GPU_FOR_XGBOOST = True` but no GPU is available, print a clear warning and fall back to CPU rather than raising. Or make the flag `False` by default.
3. **Add a "Quick Start" section** at the top of the notebook explaining: Python version, how to install deps, where to put the dataset file.
4. **Pin random seeds consistently**: Ensure every function using randomness accepts `seed` from the configuration section rather than defaulting to local `123`.
5. **Add runtime estimates**: In markdown cells, note expected runtime for expensive sections (CV grid search, SHAP).

---

## 11. Refactoring Risk Assessment

| Recommendation | Risk Level | What Could Break | Why | How to Avoid |
|---|---|---|---|---|
| Remove dead Section 8 code | Low | Nothing; code is already discarded by `df = df_raw.copy()` | No downstream dependency | Verify no variable leaks from Section 8 |
| Remove backward-compatible aliases | Low | External scripts that `import` from this notebook (unlikely for .ipynb) | Variable name changes | grep for alias usage in other files first |
| Remove baseline XGBoost training (Section 14) | Low | Nothing; CV models immediately overwrite | Models are overwritten two cells later | Confirm no test-set AUC from baseline is reported |
| Consolidate constants into config cell | Low | Typos when moving values | Copy-paste errors | Automated test: run notebook end-to-end after change |
| Extract helper functions to `_prism_model_utils.py` | Medium | Import order, circular dependencies | Functions reference notebook-scoped variables (e.g., `XGB_GPU_PARAMS`) | Pass configuration as arguments rather than using globals |
| Reorganize into Task 1–8 structure | Medium | Cell execution order dependencies | Variables defined in one task used in another | Map all variable dependencies before reordering; test with `Run All` |
| Change output folder structure | Medium | Downstream consumers expecting old paths (README generator, other notebooks) | File paths change | Update `generate_readme_tables.py` and any references simultaneously |
| Remove individual predictor PNGs | Low | Anyone bookmarking specific PNG paths | Paths disappear | Keep PDFs as replacement; announce change |
| Remove synthetic validation section | Low | Lose ability to validate against known truth | Only meaningful for synthetic data | Gate behind flag rather than hard-removing |
| Dynamic predictor detection | Medium | May include/exclude unexpected columns | New data schema differs | Add explicit exclusion list as safety net |
| GPU fallback logic | Medium | Could silently produce CPU-trained models when GPU was intended | Performance difference, potential reproducibility shift | Log clearly which device was used |

---

## 12. Recommended Refactoring Roadmap

### Phase 1: Safe Cleanup (No Behavior Changes)

**Estimated effort**: 1–2 hours  
**Risk**: Low

- [ ] Remove dead Section 8 code (date features followed by `df = df_raw.copy()`)
- [ ] Remove baseline XGBoost training in Section 14 (pre-CV models)
- [ ] Remove all backward-compatible alias variables
- [ ] Remove standalone `model_df.columns` debugging cell
- [ ] Remove duplicate function definitions (`safe_as_date`, `present_columns`)
- [ ] Consolidate all constants into a single Configuration cell at the top
- [ ] Add `requirements.txt` to the repo
- [ ] Verify notebook runs end-to-end after cleanup

### Phase 2: Notebook Restructuring into Tasks 1–8

**Estimated effort**: 3–4 hours  
**Risk**: Medium

- [ ] Create markdown header cells for each Analytical Task
- [ ] Move code cells into their appropriate tasks
- [ ] Ensure variable flow is strictly top-to-bottom
- [ ] Add markdown explanations before each major code section
- [ ] Add Mermaid flowchart in Task 1 (mirroring causal forest)
- [ ] Add runtime estimates in markdown for expensive sections
- [ ] Test with full `Run All Cells`

### Phase 3: Output Cleanup

**Estimated effort**: 2–3 hours  
**Risk**: Medium

- [ ] Implement new folder structure (01_Data_Review through 06_Synthetic_Validation)
- [ ] Gate individual predictor PNGs behind a config flag (default off)
- [ ] Gate synthetic validation behind `SYNTHETIC_VALIDATION_ENABLED` flag
- [ ] Gate GLMNet permutation SHAP behind `RUN_GLMNET_PERMUTATION_SHAP` flag
- [ ] Add `hyperparameter_tuning_summary.csv` output
- [ ] Add `predictor_inventory.csv` output
- [ ] Add `configuration_snapshot.csv` output
- [ ] Update `generate_readme_tables.py` for new paths
- [ ] Remove or archive legacy `Outputs/T-Learner/` folder
- [ ] Test with full `Run All Cells`

### Phase 4: Dataset Replacement Improvements

**Estimated effort**: 2–3 hours  
**Risk**: Medium

- [ ] Replace hardcoded `candidate_predictors_all` with dynamic detection + override list
- [ ] Add schema validation cell after data load
- [ ] Parameterize outcome/treatment column names via config
- [ ] Parameterize risk tier boundaries via config
- [ ] Add clear error messages for missing/unexpected columns
- [ ] Test with current synthetic data
- [ ] Document expected GenRocket schema requirements

### Phase 5: Final Engineering Polish

**Estimated effort**: 2–3 hours  
**Risk**: Low

- [ ] Extract reusable functions to `_prism_model_utils.py`
- [ ] Add GPU fallback logic (warn + continue on CPU if GPU unavailable)
- [ ] Pin random seeds through single `SEED` constant everywhere
- [ ] Add "Quick Start" instructions in notebook header
- [ ] Add output manifest (generated list of all files produced)
- [ ] Final end-to-end run and output verification
- [ ] Compare outputs against pre-refactoring baseline to confirm numerical equivalence

---

## Summary

The uplift notebook is functionally complete and produces extensive analytical outputs. However, it has grown organically and diverged from the cleaner architecture established in the causal forest and doubly robust workflows. The primary issues are:

1. Inconsistent structure (numbered + unnumbered sections)
2. Scattered configuration values
3. Dead code that confuses readers
4. Hardcoded assumptions that will break on dataset replacement
5. Output folder organization that mixes concerns

None of these issues affect current correctness—they affect maintainability, reproducibility, and readiness for the GenRocket dataset swap. The phased roadmap ensures each change can be validated independently before proceeding to the next.
