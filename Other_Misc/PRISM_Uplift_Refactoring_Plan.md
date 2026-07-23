# PRISM Uplift Modeling – Engineering Review & Refactoring Plan

## Objective

Make the uplift workflow plug-and-play for dataset replacement. When a new GenRocket dataset arrives, the analyst should only need to update the dataset path (and optionally a column mapping), then run the notebook top-to-bottom to regenerate all outputs.

---

## 1. Dataset Replacement Readiness (Highest Priority)

### 1.1 Hardcoded Predictor List

**Location**: Section 9 – `candidate_predictors_all` (56 column names)

**Problem**: The notebook hardcodes a list of 56 expected predictor names. If the GenRocket dataset uses different column names, renames features, or adds/removes fields, the notebook will silently drop missing columns without error (it uses a list comprehension filter) but will never pick up new relevant columns.

**Impact**: New predictors in the GenRocket data won't be used; renamed predictors will be silently excluded.

**Recommendation**: Replace with dynamic detection:
```python
EXCLUDE_COLUMNS = {OUTCOME_COLUMN, TREATMENT_COLUMN, 'index_date', 'intervention_start_date', 'intervention_end_date', 'member_id'}
candidate_predictors = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
```
Optionally keep a `PREDICTOR_OVERRIDE` list in configuration for explicit inclusion/exclusion.

---

### 1.2 Hardcoded Numeric Column List

**Location**: Section 10 – `possible_numeric_cols` (24 column names)

**Problem**: These 24 column names are listed to force numeric coercion. If GenRocket uses different names for the same concepts, they won't be coerced and will be treated as categorical, breaking the model matrix.

**Impact**: Incorrectly typed features → wrong model matrix → wrong model results.

**Recommendation**: Detect numeric columns programmatically. If a column has >90% parseable numeric values (after `pd.to_numeric(errors='coerce')`), treat it as numeric. Keep the explicit list only as an override.

---

### 1.3 Hardcoded Outcome and Treatment Column Names

**Location**: Section 6 – `required_fields = ['outcome_ed_90d', 'intervention_flag']`; referenced throughout.

**Problem**: If GenRocket names the outcome `ed_visit_90d` or the treatment `treatment_flag`, the notebook immediately fails.

**Impact**: Hard failure at the `require_columns` check.

**Recommendation**: Move to configuration:
```python
OUTCOME_COLUMN = 'outcome_ed_90d'
TREATMENT_COLUMN = 'intervention_flag'
```
Replace every downstream reference with these constants.

---

### 1.4 Risk Tier Thresholds

**Location**: Risk Tier section – hardcoded DataFrame with boundaries `[35, 55, 75]` on `current_risk_score`.

**Problem**: These thresholds are specific to the synthetic data's risk-score distribution. GenRocket may have a different scale, different range, or the column may not exist.

**Impact**: Risk tier assignments will be meaningless or crash.

**Recommendation**: Gate behind a config flag. If `current_risk_score` exists, derive thresholds from quantiles by default (e.g., 25th/50th/75th percentile) or allow explicit override:
```python
RISK_TIER_THRESHOLDS = [35, 55, 75]  # or 'auto' for quantile-based
RISK_TIER_LABELS = ['Low', 'Medium', 'High', 'Very High']
```

---

### 1.5 Minimum Sample Size Assumptions

**Location**: Section 12 – `if len(train_treated) < 50: raise ValueError(...)`

**Problem**: With a GenRocket dataset of different size or treatment prevalence, 50 may be too high or too low. The CV fold logic also assumes `class_counts.min() >= 2`.

**Impact**: Hard failure on small datasets; unstable models on extremely imbalanced datasets.

**Recommendation**: Make the threshold configurable: `MIN_GROUP_SIZE = 50`. Add a warning rather than an error when the threshold is close.

---

### 1.6 Synthetic True-Benefit Validation Formula

**Location**: "SYNTHETIC TRUE-BENEFIT VALIDATION" section – hardcoded formula:
```python
0.02 + 0.018 * ed_visits_last_6m + 0.015 * admits_last_6m + 0.018 * food_insecurity_flag + ...
```

**Problem**: This formula is the generating mechanism for the current synthetic dataset. It is meaningless for any other dataset.

**Impact**: Will produce invalid "validation" results; could be misleading.

**Recommendation**: Gate behind `SYNTHETIC_VALIDATION_ENABLED = False` (default off). Only set to `True` when running the synthetic dataset.

---

### 1.7 `current_risk_score` Dependency in Targeting Analysis

**Location**: ROI targeting comparison ranks by `current_risk_score` vs `benefit_score`.

**Problem**: If GenRocket doesn't include a column called `current_risk_score`, the targeting comparison will crash or need modification.

**Impact**: Targeting savings analysis fails.

**Recommendation**: Make the comparison column configurable:
```python
RISK_RANKING_COLUMN = 'current_risk_score'  # or None to skip comparison
```
If None, only produce uplift-based targeting analysis.

---

### 1.8 Date Column Assumptions

**Location**: Section 8 – derives features from `index_date`, `intervention_start_date`, `intervention_end_date`. Currently reverted with `df = df_raw.copy()` (dead code).

**Problem**: If GenRocket includes dates in different formats or names, the date logic will fail. Currently dead code anyway.

**Impact**: Low immediate risk since the section is bypassed. But if restored, format assumptions could break.

**Recommendation**: Remove the dead Section 8 code entirely. If date features are needed, use the `add_date_features()` function from `_prism_model_utils.py` which already handles missing columns gracefully.

---

### 1.9 Predictor Category/Description Dictionaries

**Location**: "WRITE-UP PREDICTOR DATA DICTIONARY" section – two large dictionaries mapping column names to categories and descriptions (>100 entries).

**Problem**: GenRocket columns not in these dictionaries will get generic labels ("Other / derived"). Not a crash risk but reduces documentation quality.

**Impact**: Low – cosmetic only. Data dictionary will be incomplete.

**Recommendation**: Auto-detect categories using naming heuristics (e.g., `*_flag` → "Binary indicator", `*_last_6m` → "Utilization") and allow override via a small config dict.

---

### 1.10 Flag Detection by Suffix

**Location**: Section 10 – `flag_like_cols = [col for col in model_df.columns if col.endswith('_flag')]`

**Problem**: Relies on `_flag` naming convention. If GenRocket uses different suffix (e.g., `_indicator`, `_yn`), binary columns won't be properly coerced.

**Impact**: Medium – could treat binary columns as multi-level categorical.

**Recommendation**: Expand detection: check columns with exactly 2 unique values in {0, 1, 'yes', 'no', 'Y', 'N', True, False} regardless of name suffix.

---

## 2. Hardcoded Results and Dataset-Specific Logic

### 2.1 ROI Cost Constants

**Location**: "ROI PER DECILE" section:
```python
cost_per_ed_visit = 1200
cost_per_intervention = 250
```

**Assessment**: These are business assumptions, not dataset-specific. However, they should be in configuration so they're visible and adjustable.

**Recommendation**: Move to config section at top of notebook.

---

### 2.2 Backward-Compatible Alias Variables

**Location**: Sections 3, 15, 16, 17, 18, ROI:
```python
output_path = xgboost_output_path
results_test = results_test_xgboost
decile_summary = decile_summary_xgboost
importance_treated = importance_treated_xgboost
scored_full = scored_full_xgboost
roi_summary = roi_summary_xgboost
```

**Assessment**: Legacy aliases from before the dual-model (XGBoost+GLMNet) structure was introduced. No external consumer uses these.

**Recommendation**: Remove in Phase 1. They add confusion without value.

---

### 2.3 AUC Strength Classification

**Location**: "MODEL RECOMMENDATION SUMMARY" section:
```python
def auc_strength(value):
    if value >= 0.90: return 'Excellent'
    if value >= 0.80: return 'Strong'
    ...
```

**Assessment**: This is a general-purpose classification. Not dataset-specific, but the thresholds are hardcoded.

**Recommendation**: Keep as-is. These are industry-standard interpretive ranges.

---

### 2.4 Baseline XGBoost Training (Overwritten)

**Location**: Section 14 – trains models with fixed params `max_depth=4, eta=0.05`, then immediately overwrites them with CV-tuned models.

**Assessment**: Dead code. The baseline models are never used.

**Recommendation**: Remove in Phase 1.

---

## 3. Outputs Folder Review

### Root-Level CSVs (`Outputs/Uplift/Python/`)

| Output | Purpose | Used in Report | Used by Other | Keep | Optional | Remove | Comments |
|--------|---------|:-:|:-:|:-:|:-:|:-:|---------|
| `data_review_summary.csv` | Dataset-level statistics | ✓ | — | ✓ | | | Core write-up |
| `factual_event_count_summary.csv` | Train/test event counts | ✓ | — | ✓ | | | |
| `factual_prediction_separation.csv` | AUC/separation diagnostics | ✓ | — | ✓ | | | |
| `factual_prediction_ranges.csv` | Prediction quantiles | ✓ | — | | ✓ | | Could merge into separation |
| `model_evaluation_summary.csv` | Cross-model comparison | ✓ | — | ✓ | | | Key decision artifact |
| `model_recommendation_summary.csv` | Human-readable assessment | ✓ | — | ✓ | | | |
| `risk_tier_thresholds.csv` | Static lookup | — | — | | ✓ | | Only if risk tiers used |
| `risk_tier_population_summary.csv` | Members per tier | ✓ | — | ✓ | | | |
| `glmnet_true_benefit_validation_summary.csv` | Synthetic-only validation | — | — | | | ✓ | Remove for real data |
| `glmnet_true_benefit_scored_test_comparison.csv` | Synthetic-only | — | — | | | ✓ | Remove for real data |
| `glmnet_true_benefit_decile_overlap_summary.csv` | Synthetic-only | — | — | | | ✓ | Remove for real data |
| `glmnet_shap_true_driver_alignment.csv` | Synthetic-only | — | — | | | ✓ | Remove for real data |
| `glmnet_true_driver_recovery_summary.csv` | Synthetic-only | — | — | | | ✓ | Remove for real data |

### Predictor Distributions (`Predictor_Distributions/`)

| Output | Purpose | Keep | Optional | Remove | Comments |
|--------|---------|:-:|:-:|:-:|---------|
| `predictor_data_dictionary.csv` | Variable metadata | ✓ | | | Reference |
| `numeric_predictor_summary.csv` | Descriptive stats | ✓ | | | |
| `categorical_predictor_summary.csv` | Descriptive stats | ✓ | | | |
| `numeric_predictor_distributions.pdf` | All numeric histograms | ✓ | | | |
| `categorical_predictor_distributions.pdf` | All categorical bars | ✓ | | | |
| `predictor_distribution_visual_index.csv` | Index of PNGs | | ✓ | | Convenience only |
| Individual PNGs (61 files) | Same content as PDFs | | | ✓ | Excessive; PDFs suffice |

### T-Learner Outputs (per model: XGBoost, GLMNet)

| Output | Purpose | Used in Report | Keep | Comments |
|--------|---------|:-:|:-:|---------|
| `uplift_scored_output.csv` | Full scored population | ✓ | ✓ | Primary deliverable |
| `uplift_decile_summary.csv` | Decile aggregation | ✓ | ✓ | Primary deliverable |
| `uplift_roi_by_decile.csv` | ROI per decile | ✓ | ✓ | |
| `top_benefit_decile_summary.csv` | Top decile profile | ✓ | ✓ | |
| `uplift_observed_gap_by_decile.csv` | Observed gap + CIs | ✓ | ✓ | Key validation |
| `uplift_curve_by_decile.csv` | Uplift curve data | ✓ | ✓ | |
| `uplift_curve_summary.csv` | AUC metric | ✓ | ✓ | |
| `model_brier_scores.csv` | Calibration | ✓ | ✓ | |
| `calibration_by_decile.csv` | Detailed calibration | ✓ | ✓ | |
| `calibration_summary.csv` | Calibration summary | ✓ | ✓ | |
| `shap_importance_treated_control_models.csv` | Risk drivers | ✓ | ✓ | |
| `shap_importance_benefit_score.csv` | Benefit drivers | ✓ | ✓ | |
| `cumulative_gross_savings_by_targeting.csv` | Targeting analysis | ✓ | ✓ | GLMNet only currently |
| All dashboard PNGs | Charts | ✓ | ✓ | |
| `glmnet_tlearner_global_benefit_shap_importance.csv` | Permutation SHAP | — | Optional | Slow; adds ~15min |
| `glmnet_tlearner_member_benefit_shap_values.csv` | Per-member SHAP | — | Optional | Large file; slow |
| `glmnet_tlearner_global_benefit_shap_reconciliation.csv` | SHAP reconciliation | — | Optional | |

### X-Learner Outputs

| Output | Purpose | Keep | Comments |
|--------|---------|:-:|---------|
| `shared_propensity_scores.csv` | Propensity for all members | ✓ | Shared across frameworks |
| `xlearner_vs_tlearner_consistency_summary.csv` | Correlation/overlap | ✓ | Key validation |
| `xlearner_scored_test_output.csv` | Test set scores | ✓ | Per model |
| `xlearner_decile_summary.csv` | Decile aggregation | ✓ | Per model |
| `xlearner_benefit_driver_importance.csv` | Benefit drivers | ✓ | |
| `shap_importance_benefit_score.csv` | Same as above (alias) | ✓ | |
| `xlearner_risk_tier_benefit_group_summary.csv` | Risk tier segmentation | ✓ | |
| `cumulative_gross_savings_by_targeting.csv` | Targeting analysis | ✓ | |
| All dashboard PNGs | Charts | ✓ | |

### Recommended Additions

| Missing Output | Purpose | Priority |
|---|---|---|
| `configuration_snapshot.csv` | Runtime config dump | High |
| `predictor_inventory.csv` | Final feature list used | High |
| `hyperparameter_tuning_summary.csv` | CV grid results | Medium |

### Recommended Removals

1. **61 individual predictor PNGs** – PDFs already consolidate these
2. **5 synthetic-only validation CSVs** – meaningless for real data
3. **GLMNet permutation SHAP files** (3 files) – extremely slow, coefficient contribution already provides exact decomposition

---

## 4. Code Maintainability

### 4.1 Duplicated Functions

| Function | Notebook | `_prism_model_utils.py` | Action |
|----------|:--------:|:----------------------:|--------|
| `safe_as_date()` | ✓ | ✓ | Remove from notebook |
| `present_columns()` | ✓ | ✓ | Remove from notebook |

### 4.2 Dead Code

| Item | Location | Action |
|------|----------|--------|
| Section 8 date features + `df = df_raw.copy()` revert | Section 8 | Remove entire section |
| Baseline XGBoost training (pre-CV) | Section 14 | Remove |
| `model_df.columns` standalone cell | Section 10 | Remove |
| Backward-compatible alias variables (6 instances) | Multiple | Remove |

### 4.3 Monolithic Cells

The X-Learner section is a single cell of ~200 lines mixing:
- Model training
- Scoring
- Output saving
- Charting
- Consistency computation

**Recommendation**: Break into single-responsibility cells (one per concern).

### 4.4 Repeated Reporting Pattern

The same save-CSV + make-chart + print pattern is repeated 15+ times. Functions like `save_probability_evaluation_outputs()` and `save_observed_gap_outputs()` already consolidate some of this, but many remain inline.

**Recommendation**: Extract remaining inline reporting blocks into functions.

---

## 5. Configuration

### Proposed Centralized Configuration Block

```python
# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit only this section when changing inputs
# ═══════════════════════════════════════════════════════════════════════

# --- Data Source ---
DATASET_PATH = GITHUB_XLSX_URL  # or local path to .xlsx
OUTCOME_COLUMN = 'outcome_ed_90d'
TREATMENT_COLUMN = 'intervention_flag'

# --- Modeling ---
SEED = 123
TRAIN_FRACTION = 0.70
CV_FOLDS = 5
MAX_BOOST_ROUNDS = 500
MIN_GROUP_SIZE = 50

# --- Runtime ---
REQUIRE_GPU_FOR_XGBOOST = True
XGBOOST_CUDA_DEVICE = 0
RUN_CPU_ONLY_COMPARISON_MODELS = True
SAVE_INDIVIDUAL_PREDICTOR_PLOTS = False
SYNTHETIC_VALIDATION_ENABLED = False
RUN_GLMNET_PERMUTATION_SHAP = False

# --- Business Assumptions ---
COST_PER_ED_VISIT = 1200
COST_PER_INTERVENTION = 250

# --- Risk Tiers ---
RISK_RANKING_COLUMN = 'current_risk_score'  # or None to skip
RISK_TIER_THRESHOLDS = [35, 55, 75]
RISK_TIER_LABELS = ['Low', 'Medium', 'High', 'Very High']

# --- Benefit Groups ---
HIGH_BENEFIT_DECILES = [1, 2]
MEDIUM_BENEFIT_DECILES = [3, 4, 5, 6, 7]
LOW_BENEFIT_DECILES = [8, 9, 10]

# --- Propensity ---
PROPENSITY_CLIP_LOWER = 0.05
PROPENSITY_CLIP_UPPER = 0.95

# --- Output ---
OUTPUT_DIR = PROJECT_ROOT / 'Outputs' / 'Uplift' / 'Python'
```

### Values Currently Scattered (must be centralized)

| Value | Current Location | Count of References |
|-------|-----------------|:--:|
| Seed `123` | Sections 4, 11, 14, grid search, elastic-net, propensity, X-learner | 12+ |
| Train fraction `0.70` | Section 11 | 1 |
| `cost_per_ed_visit = 1200` | ROI section | 2 |
| `cost_per_intervention = 250` | ROI section | 2 |
| Propensity clip `0.05/0.95` | `clipped_propensity` function | 1 |
| Nfold `5` | Multiple function defaults | 5+ |
| `nrounds_max=500` | Grid search call | 1 |
| Risk tier boundaries `[35,55,75]` | Risk tier section | 1 |
| Display top-N `20` | `print_highest_benefit`, SHAP charts | 5+ |

---

## 6. Reproducibility

### Can Another Analyst: Clone → Replace Data → Run → Get Outputs?

**Current Answer: No.**

### Obstacles

| Obstacle | Severity | Details |
|----------|----------|---------|
| **GPU mandatory** | High | `REQUIRE_GPU_FOR_XGBOOST = True` crashes without CUDA. No graceful fallback. |
| **No requirements.txt** | Medium | Package versions not pinned; Python version not documented. |
| **Working directory dependency** | Medium | `project_root()` relies on `Path(__file__).parents[1]` which works from `Code/` but may fail in other contexts. |
| **No environment spec** | Medium | XGBoost GPU build requires specific CUDA version. |
| **Dead code confusion** | Low | Section 8 derives features then discards them – confusing. |
| **SHAP optional but silent** | Low | If `shap` not installed, some outputs are simply missing without clear messaging. |
| **Large runtime** | Medium | Full run takes 20-45 min (CV grid search + permutation SHAP). No progress indicators. |

### Recommendations

1. Add `requirements.txt` with pinned versions
2. Add GPU fallback: `REQUIRE_GPU_FOR_XGBOOST = True` should warn-and-continue on CPU rather than crash
3. Add "Quick Start" instructions at notebook top
4. Gate expensive operations (permutation SHAP) behind flags that default to off
5. Add runtime estimates in markdown cells

---

## 7. Notebook Organization

The notebook uses numbered sections (1-20) plus ~15 unnumbered "WRITE-UP" sections. This works functionally but is harder to navigate than the Task 1-8 structure in the Causal Forest notebook.

**Recommendation** (lower priority): If reorganizing, adopt:

| Task | Contents |
|------|----------|
| Task 1 | Framework explanation, configuration |
| Task 2 | Data load, review, predictor exploration |
| Task 3 | Model training, CV, test AUC, calibration |
| Task 4 | Treatment effect estimation (T-learner + X-learner) |
| Task 5 | Decile analysis, observed gap, uplift curve |
| Task 6 | Variable importance, SHAP, benefit drivers |
| Task 7 | Business value (ROI, targeting, risk tiers) |

However, this is cosmetic and should only be done after Phases 1-4 are complete.

---

## 8. Refactoring Risk Assessment

| Recommendation | Risk | What Could Break | Mitigation |
|---|---|---|---|
| Remove dead Section 8 | Low | Nothing (already discarded) | Verify no variable leaks |
| Remove baseline XGBoost Section 14 | Low | Nothing (overwritten) | Confirm no test AUC from baseline |
| Remove backward-compatible aliases | Low | Nothing (no external consumers) | Grep for usage |
| Consolidate config into one cell | Low | Typos when moving values | Run end-to-end after |
| Remove individual predictor PNGs | Low | Broken bookmarks | PDFs as replacement |
| Remove synthetic validation section | Low | Lose synthetic truth check | Gate behind flag |
| Gate permutation SHAP behind flag | Low | Missing outputs if off | Document in config |
| Dynamic predictor detection | Medium | May include/exclude unexpected columns | Add exclusion list as safety |
| Parameterize outcome/treatment names | Medium | All downstream code must use constants | Search-and-replace carefully |
| Risk tier auto-thresholds | Medium | Different segmentation than before | Validate against existing |
| Extract functions to utils | Medium | Import order, global variable deps | Pass config as arguments |
| Reorganize into Task structure | Medium | Cell execution order breaks | Map dependencies first |
| Change output folder structure | Medium | README generator, other scripts break | Update all consumers |
| GPU fallback logic | Medium | Could silently produce CPU models | Log device used clearly |

---

## 9. Implementation Roadmap

### Phase 1: Safe Cleanup (No Behavioral Changes)

**Effort**: 1-2 hours | **Risk**: Low

- [ ] Remove dead Section 8 code
- [ ] Remove baseline XGBoost training in Section 14
- [ ] Remove backward-compatible alias variables (6 instances)
- [ ] Remove standalone `model_df.columns` debugging cell
- [ ] Remove duplicate `safe_as_date` and `present_columns` definitions
- [ ] Remove 61 individual predictor PNGs (keep PDFs)
- [ ] Verify notebook still runs end-to-end with identical outputs

### Phase 2: Configuration Centralization

**Effort**: 2-3 hours | **Risk**: Low

- [ ] Create single configuration cell at notebook top
- [ ] Replace all scattered `123` seeds with `SEED`
- [ ] Replace all `0.70` with `TRAIN_FRACTION`
- [ ] Replace ROI cost constants with `COST_PER_ED_VISIT` / `COST_PER_INTERVENTION`
- [ ] Replace risk tier hardcoded values with config constants
- [ ] Replace benefit group decile mappings with config
- [ ] Replace propensity clip bounds with config
- [ ] Parameterize outcome/treatment column names as constants
- [ ] Verify numerical equivalence after centralization

### Phase 3: Dataset Replacement Improvements

**Effort**: 3-4 hours | **Risk**: Medium

- [ ] Replace hardcoded `candidate_predictors_all` with dynamic detection + exclusion list
- [ ] Replace hardcoded `possible_numeric_cols` with heuristic detection + override
- [ ] Add schema validation cell (required columns, types, value checks)
- [ ] Gate synthetic validation behind `SYNTHETIC_VALIDATION_ENABLED` flag
- [ ] Make `RISK_RANKING_COLUMN` configurable (or None to skip)
- [ ] Expand flag detection beyond `_flag` suffix
- [ ] Add graceful GPU fallback (warn + run on CPU)
- [ ] Add `configuration_snapshot.csv` and `predictor_inventory.csv` outputs

### Phase 4: Outputs Cleanup

**Effort**: 2-3 hours | **Risk**: Medium

- [ ] Gate individual predictor PNGs behind `SAVE_INDIVIDUAL_PREDICTOR_PLOTS = False`
- [ ] Gate GLMNet permutation SHAP behind `RUN_GLMNET_PERMUTATION_SHAP = False`
- [ ] Move synthetic validation CSVs behind the config gate
- [ ] Add `hyperparameter_tuning_summary.csv` output
- [ ] Update `generate_readme_tables.py` for any path changes
- [ ] Verify all README auto-generation still works

### Phase 5: Code Maintainability

**Effort**: 3-4 hours | **Risk**: Medium

- [ ] Break X-Learner monolithic cell into single-responsibility cells
- [ ] Extract remaining inline reporting patterns into functions
- [ ] Move reusable functions (ROI, decile summary, targeting) to `_prism_model_utils.py`
- [ ] Add markdown explanations before each major code section
- [ ] Add runtime estimates for expensive sections
- [ ] Add `requirements.txt` to repo
- [ ] Final end-to-end run and output comparison against baseline

---

## Summary

The uplift notebook is functionally complete and produces comprehensive analytical outputs. The primary risks for dataset replacement are:

1. **56 hardcoded predictor names** that will silently exclude new columns
2. **24 hardcoded numeric column names** that control type coercion
3. **Risk tier thresholds** tied to the synthetic data distribution
4. **Synthetic validation code** that will produce meaningless results on real data
5. **GPU enforcement** that prevents running on non-CUDA machines

The phased roadmap addresses these in priority order, ensuring each change is independently verifiable. Phase 1-2 can be completed without any risk to current results. Phase 3 is the critical "plug-and-play enablement" phase. Phases 4-5 are quality-of-life improvements.

After completing Phases 1-3, replacing the dataset should require only:
1. Update `DATASET_PATH` in configuration
2. Optionally update `OUTCOME_COLUMN` / `TREATMENT_COLUMN` if renamed
3. Run notebook top-to-bottom
