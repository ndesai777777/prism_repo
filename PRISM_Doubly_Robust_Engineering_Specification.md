# PRISM Doubly Robust Learner — Engineering Specification

**Document Version:** 1.0  
**Status:** Draft  
**Author:** PRISM Analytics Engineering  
**Target Notebook:** `Code/PRISM_Doubly_Robust_Modeling_Workflow.ipynb`  
**Reference Implementation:** `Code/PRISM_Causal_Forest_Modeling_Workflow.ipynb`  
**Legacy Notebook (superseded):** `Code/Doubly Robust Code.ipynb`

---

## 1. Executive Summary

This specification defines the complete engineering requirements for refactoring the existing
`Doubly Robust Code.ipynb` into a production-quality PRISM workflow notebook. The target
notebook will use `econml.dr.ForestDRLearner` to estimate heterogeneous treatment effects
via doubly-robust pseudo-outcomes, following the same evaluation framework, output standards,
and code patterns established by the Causal Forest workflow.

The refactored notebook will produce:

- Member-level treatment-effect estimates (benefit scores)
- Diagnostic outputs validating estimation credibility (Level 1)
- Explainability and business-value outputs (Level 2)
- Cross-method consistency analysis against GLMNet T-learner, GLMNet X-learner, and Causal Forest
- Dashboard-quality charts and structured CSV tables

The legacy notebook produces 2 CSV files with no charts, no SHAP, and no evaluation structure.
The target notebook will produce approximately 20 CSV files, 8–10 PNG charts, and structured
markdown-ready tables across a two-level evaluation framework.

---

## 2. Scope

### 2.1 Included

| Area | Description |
|------|-------------|
| Estimator | `econml.dr.ForestDRLearner` with `RandomForestRegressor` nuisance models |
| Propensity | Shared propensity scores from X-learner workflow (with fallback) |
| Split | 70/30 stratified on `intervention_flag` + `outcome_ed_90d`, seed=123 |
| Predictors | Same 41-variable inventory (77 columns after one-hot encoding) |
| Evaluation | Two-level framework (Treatment-Effect Credibility + Explainability/Business Value) |
| Outputs | CSV tables, PNG charts, inline markdown summaries |
| Validation | True-benefit formula correlation, cross-method agreement |
| Cost model | $1,200/ED visit, $250/intervention |
| SHAP | Applied to final RF treatment-effect model |

### 2.2 Excluded

| Area | Rationale |
|------|-----------|
| Hyperparameter tuning grid | DR Learner pipeline is simpler; no grid search required |
| Confidence intervals / `tau_se` | `ForestDRLearner` does not provide inference objects like `CausalForestDML` |
| GPU acceleration | Not applicable (scikit-learn RandomForest backend) |
| Live deployment scoring | Out of scope; notebook is a modeling demonstration |
| Modification of legacy notebook | Legacy notebook preserved as-is for audit trail |

---

## 3. Design Principles

1. **Reproducibility** — Deterministic seed (123), stratified split, fixed predictor order.
2. **Consistency** — Same helper functions, naming conventions, and evaluation structure as Causal Forest.
3. **Transparency** — All intermediate outputs saved; pseudo-outcome diagnostics expose DR internals.
4. **Modularity** — Shared utilities imported from `_prism_model_utils.py`; no inline reimplementation.
5. **Idempotency** — Re-running the notebook overwrites outputs deterministically.
6. **Readability** — Markdown cells explain every analytical step; notebook is self-documenting.

---

## 4. Notebook Architecture

### 4.1 High-Level Cell Structure

| Cell # | Type | Content |
|--------|------|---------|
| 1 | Markdown | Title, sign convention, seed declaration |
| 2 | Markdown | Optional package install instructions |
| 3 | Code | Conditional install check (econml, shap, matplotlib) |
| 4 | Markdown | Background |
| 5 | Markdown | Business question |
| 6 | Markdown | Project objectives |
| 7 | Markdown | Task 1 header: Framework + Imports |
| 8 | Code | Imports, constants, PROJECT_ROOT, OUTPUT_DIR |
| 9 | Code | PATHS dict (all output paths) |
| 10 | Code | Helper functions (save_csv, save_current_figure, summarize_distribution, safe_corr, top_overlap) |
| 11 | Markdown | Task 2 header: Data Review |
| 12 | Code | Data load, clean, prepare, predictor inventory, data review summary |
| 13 | Markdown | Evaluation Roadmap |
| 14 | Markdown | Level 1 header: Treatment-Effect Credibility |
| 15 | Markdown | Task 3 header: Diagnostics |
| 16 | Code | Train/test split, propensity, event counts, propensity chart |
| 17 | Code | DR Learner fit, pseudo-outcome extraction, pseudo-outcome diagnostics |
| 18 | Markdown | Task 4 header: Treatment Effects + True-Benefit Validation |
| 19 | Code | Effect distribution, ATE summary, true-benefit correlation, effect histogram |
| 20 | Markdown | Task 5 header: Decile Analysis + Framework Consistency |
| 21 | Code | Decile summary, risk-tier × benefit-group, cross-method agreement matrix |
| 22 | Code | Framework consistency Spearman heatmap chart |
| 23 | Markdown | Level 1 Summary |
| 24 | Markdown | Level 2 header: Explainability And Business Value |
| 25 | Markdown | Task 6 header: Variable Importance + SHAP |
| 26 | Code | RF variable importance, SHAP values, SHAP bar chart |
| 27 | Markdown | Task 7 header: Business Value + Targeting |
| 28 | Code | Targeting summary, ROI calculations, policy comparison chart |
| 29 | Markdown | Level 2 Summary |
| 30 | Code | Final file inventory print |

### 4.2 Execution Order

Strictly sequential, top-to-bottom. No cell dependencies on external state beyond
the shared propensity CSV and the source Excel file.

---

## 5. Global Coding Standards

### 5.1 Constants

```python
SEED = 123
TRAIN_FRACTION = 0.70
OUTCOME_COL = 'outcome_ed_90d'
TREATMENT_COL = 'intervention_flag'
OUTPUT_DIR = ensure_output_folder(PROJECT_ROOT / 'Outputs' / 'Doubly-Robust' / 'Python')
COST_PER_ED_VISIT = 1200
COST_PER_INTERVENTION = 250
```

### 5.2 Imports (Required)

```python
from pathlib import Path
import importlib.util, sys, warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from econml.dr import ForestDRLearner
```

### 5.3 Helper Functions (Defined In-Notebook)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `save_csv` | `(df, path)` | Write CSV + print confirmation |
| `save_current_figure` | `(path)` | `tight_layout()` → `savefig(dpi=160)` → `show()` → print |
| `summarize_distribution` | `(values, label) → DataFrame` | 9-row percentile summary |
| `safe_corr` | `(a, b, method) → float` | NaN-safe correlation |
| `top_overlap` | `(a_scores, b_scores, share=0.10) → float` | Top-k Jaccard overlap |

### 5.4 Chart Standards

| Parameter | Value |
|-----------|-------|
| DPI | 160 |
| Default figsize | (8, 4.5) |
| Horizontal bar figsize | (9, 5.5) |
| Color palette | matplotlib defaults (tab10) |
| Font size | matplotlib defaults |
| Layout | `tight_layout()` before save |

### 5.5 CSV Naming Convention

```
doubly_robust_{descriptor}.csv
```

### 5.6 Chart Naming Convention

```
dashboard_doubly_robust_{descriptor}.png
```

---

## 6. Data Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│  PRP_1000_full_pretreatment.xlsx                                     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Clean → Binary encoding → Date features → prepare_model_frame()     │
│  → model_df (N rows × 41 raw predictors + outcome + treatment)       │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  make_design_matrix() → x_all (N × 77 one-hot columns)              │
└───────────┬───────────────────────────────┬─────────────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐     ┌───────────────────────────┐
│  x_train (70%)        │     │  x_test (30%)             │
│  y_train, w_train     │     │  y_test, w_test           │
└───────────┬───────────┘     └───────────┬───────────────┘
            │                               │
            ▼                               │
┌─────────────────────────────────────┐     │
│  ForestDRLearner.fit(Y, T, X, W)   │     │
│  • outcome model: RF → Ŷ residuals │     │
│  • propensity model: shared/LR     │     │
│  • final model: RF on pseudo-Y     │     │
└───────────┬─────────────────────────┘     │
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  dr_model.effect(x_test) → tau_hat                                   │
│  benefit_score = -tau_hat                                            │
│  Decile assignment, true-benefit correlation, SHAP, ROI              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Analytical Tasks

### 7.1 Task 1 — Framework, Imports, Helpers, Constants

**Objective:** Establish the computational environment, define all output paths, and declare
helper functions before any data processing.

**Methodology:**
1. Check for `econml` and `shap` availability; raise `ImportError` with install instructions if missing.
2. Resolve `CODE_DIR`, add to `sys.path`, import `_prism_model_utils` functions.
3. Define `PROJECT_ROOT`, `OUTPUT_DIR`, `SEED`, `TRAIN_FRACTION`, cost constants.
4. Define `PATHS` dict with all output file paths.
5. Define helper functions: `save_csv`, `save_current_figure`, `summarize_distribution`, `safe_corr`, `top_overlap`.

**Expected Analyses:** None (setup only).

**Expected Figures:** None.

**Expected Tables:** None.

**Expected CSV Outputs:** None.

**Acceptance Criteria:**
- [ ] All imports resolve without error in Python 3.10–3.12 with econml installed.
- [ ] `PATHS` dict contains entries for every CSV and PNG produced by the notebook.
- [ ] Helper functions match signatures and behavior from Causal Forest notebook.
- [ ] `OUTPUT_DIR` resolves to `Outputs/Doubly-Robust/Python`.

**Implementation Notes:**
- The `ForestDRLearner` does not require a separate `CausalForestDML` import.
- SHAP is applied to the underlying `RandomForestRegressor`, not to the econml wrapper.
- Propensity loading logic (`load_shared_propensity_scores`, `merge_shared_propensity`) reuses the Causal Forest pattern verbatim.

---

### 7.2 Task 2 — Data Review

**Objective:** Load the PRISM dataset, apply standard cleaning, produce a predictor inventory
and data review summary matching the Causal Forest format.

**Methodology:**
1. `read_prism_excel()` → `clean_names_simple()` → `require_columns()`.
2. `to_binary()` on outcome and treatment columns.
3. `add_date_features(df, include_duration=False)`.
4. Define `PREDICTOR_CATEGORIES` dict (6 categories, 41 features).
5. `prepare_model_frame()` → `model_df`.
6. Insert `member_id` column (sequential integer).
7. Build `predictor_inventory` DataFrame.
8. Build `data_review_summary` DataFrame.

**Expected Analyses:**
- Total members, treated/control counts, treatment rate.
- ED outcome events, outcome prevalence, observed ED rate by group.
- Predictor counts (continuous, binary, categorical).
- Model matrix dimensions after one-hot encoding.

**Expected Figures:** None.

**Expected Tables:**

| Table | Key Columns |
|-------|-------------|
| `predictor_inventory` | feature, category, included_in_model, source_dtype, unique_values |
| `data_review_summary` | metric, current_value |

**Expected CSV Outputs:**
- `doubly_robust_predictor_inventory.csv`
- `doubly_robust_data_review_summary.csv`

**Acceptance Criteria:**
- [ ] Predictor inventory lists all 41 features with correct category assignments.
- [ ] Data review summary includes train/test row counts after split.
- [ ] `model_df` has `member_id` as first column.
- [ ] One-hot encoded matrix has approximately 77 columns.

**Implementation Notes:**
- `PREDICTOR_CATEGORIES` must be identical to Causal Forest (same keys, same order).
- `NUMERIC_VARS` and `BINARY_EXTRA` lists must match Causal Forest exactly.

---

### 7.3 Task 3 — Diagnostics (Level 1)

**Objective:** Validate estimation credibility through event counts, propensity overlap,
and pseudo-outcome diagnostics unique to the DR approach.

**Methodology:**

1. **Train/Test Split:**
   ```python
   train_df, test_df = split_train_test(
       model_df, train_fraction=0.70, seed=123,
       stratify_columns=['intervention_flag', 'outcome_ed_90d']
   )
   ```
2. **Design matrices:** `make_design_matrix()` on feature columns only.
3. **Shared propensity scores:**
   - Attempt to load `Outputs/Uplift/Python/X-Learner/shared_propensity_scores.csv`.
   - Merge by `member_id` with split-label validation.
   - Fallback: fit `LogisticRegressionCV` (elastic-net, l1_ratio=0.5, AUC CV, StandardScaler).
   - Clip to [0.05, 0.95].
4. **Event count summary:** Treated/control × train/test with positive/negative counts.
5. **Propensity summary:** Source, AUC, mean/min/max/percentiles, clipping counts.
6. **Propensity overlap chart:** Histogram of propensity by treatment group (test set).
7. **Fit ForestDRLearner:**
   ```python
   dr_model = ForestDRLearner(
       model_regression=RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=123, n_jobs=-1),
       model_propensity=LogisticRegressionCV(...),  # or use shared propensity
       cv=5,
       min_samples_leaf=10,
       n_estimators=500,
       random_state=123,
   )
   dr_model.fit(Y=y_train, T=w_train, X=x_train, W=None)
   ```
8. **Pseudo-outcome diagnostics (DR-specific):**
   - Extract or reconstruct DR pseudo-outcomes from the fitted model.
   - Compute summary statistics (mean, std, percentiles, fraction negative).
   - Histogram of raw pseudo-outcomes.

**Expected Figures:**

| Figure | Filename | Figsize |
|--------|----------|---------|
| Propensity overlap histogram | `dashboard_doubly_robust_propensity_overlap.png` | (8, 4.5) |
| Pseudo-outcome distribution | `dashboard_doubly_robust_pseudo_outcome_distribution.png` | (8, 4.5) |

**Expected Tables:**

| Table | Key Columns |
|-------|-------------|
| `event_count_summary` | split, group, n, positive_ed_events, event_rate |
| `propensity_summary` | metric, value |
| `pseudo_outcome_summary` | metric, value |

**Expected CSV Outputs:**
- `doubly_robust_event_count_summary.csv`
- `doubly_robust_propensity_summary.csv`
- `doubly_robust_pseudo_outcome_summary.csv`

**Acceptance Criteria:**
- [ ] Stratified split preserves treatment × outcome proportions in train and test.
- [ ] Propensity AUC is computed on both train and test sets.
- [ ] Shared propensity file is used when available; fallback is documented in propensity_summary.
- [ ] Pseudo-outcome summary includes at minimum: mean, std, min, p10, p25, median, p75, p90, max, fraction_negative.
- [ ] Pseudo-outcome histogram clearly shows distribution shape (15–20 bins).
- [ ] ForestDRLearner is fit on training data only.

**Implementation Notes:**
- Pseudo-outcomes are the core DR innovation. The raw pseudo-outcome Γᵢ is:
  ```
  Γᵢ = μ₁(Xᵢ) - μ₀(Xᵢ) + T·(Yᵢ - μ₁(Xᵢ))/e(Xᵢ) - (1-T)·(Yᵢ - μ₀(Xᵢ))/(1-e(Xᵢ))
  ```
- If `ForestDRLearner` does not expose pseudo-outcomes directly, reconstruct them from nuisance predictions.
- The pseudo-outcome diagnostic is the DR-specific analog to the Causal Forest's uncertainty/SE diagnostic.

---

### 7.4 Task 4 — Treatment Effects + True-Benefit Validation (Level 1)

**Objective:** Compute member-level treatment effects, validate against the known
true-benefit formula, and produce scored output files.

**Methodology:**

1. **Predict treatment effects on test set:**
   ```python
   tau_hat = dr_model.effect(x_test).flatten()
   benefit_score = -tau_hat
   ```
2. **Effect distribution summary:** `summarize_distribution(benefit_score, 'benefit_score')`.
3. **ATE summary:**
   - Mean tau, mean benefit_score, fraction with positive benefit, fraction with negative tau.
4. **Scored test output:**
   - Columns: `member_id`, all features, `outcome_ed_90d`, `intervention_flag`, `propensity_score`, `tau_hat`, `benefit_score`, `benefit_rank`, `benefit_percentile`, `hte_decile`.
5. **True-benefit formula computation:**
   ```python
   true_benefit = (
       0.020
       + 0.018 * ed_visits_last_6m
       + 0.015 * admits_last_6m
       + 0.018 * food_insecurity_flag
       + 0.014 * transportation_barrier_flag
       + 0.012 * behavioral_health_risk_flag
       + 0.0006 * np.maximum(current_risk_score - 50, 0)
   )
   ```
6. **Validation metrics:**
   - Spearman correlation: `benefit_score` vs `true_benefit`.
   - Pearson correlation: `benefit_score` vs `true_benefit`.
   - Top-10% overlap: fraction of true top-10% members identified by model.
   - Top-20% overlap.
7. **Effect distribution histogram.**

**Expected Figures:**

| Figure | Filename | Figsize |
|--------|----------|---------|
| Benefit score distribution | `dashboard_doubly_robust_effect_distribution.png` | (8, 4.5) |

**Expected Tables:**

| Table | Key Columns |
|-------|-------------|
| `effect_distribution_summary` | metric, benefit_score |
| `ate_summary` | metric, value |
| `true_benefit_validation_summary` | metric, value |

**Expected CSV Outputs:**
- `doubly_robust_scored_test_output.csv`
- `doubly_robust_scored_output.csv` (full dataset, all members)
- `doubly_robust_effect_distribution_summary.csv`
- `doubly_robust_ate_summary.csv`
- `doubly_robust_true_benefit_validation_summary.csv`

**Acceptance Criteria:**
- [ ] `tau_hat` and `benefit_score` are computed for every test-set member (no NaN).
- [ ] True-benefit formula matches exactly across all PRISM workflows (same coefficients).
- [ ] Spearman correlation between model benefit and true benefit is reported.
- [ ] Top-10% and top-20% overlap metrics are computed and saved.
- [ ] Scored output includes `hte_decile` assigned via `ntile_desc()`.
- [ ] Full-dataset scored output (`scored_output`) includes train + test members.

**Implementation Notes:**
- For the full-dataset scored output, refit on all data or predict on all data using the training model. The Causal Forest notebook uses the training model to predict on all data.
- Sign convention: `benefit_score = -tau_hat` (negative tau means intervention reduces ED risk, which is beneficial).

---

### 7.5 Task 5 — Decile Analysis + Framework Consistency + Cross-Method Agreement (Level 1)

**Objective:** Stratify members into benefit deciles, analyze risk-tier alignment, compare
rankings against all three other PRISM methods, and produce a cross-method agreement matrix.

**Methodology:**

1. **Decile summary:**
   - Group test set by `hte_decile`.
   - Compute per decile: n, mean benefit_score, mean tau_hat, mean true_benefit, observed ED rate (treated), observed ED rate (control).
2. **Benefit decile bar chart:** Average benefit by decile.
3. **Risk-tier × benefit-group cross-tabulation:**
   - Assign `benefit_group` ∈ {High, Medium, Low} based on terciles.
   - Cross-tab with `risk_tier` (from raw data).
   - Stacked bar chart: risk tier distribution by benefit group.
4. **Top-decile profile:**
   - Members in decile 1: summary statistics of key features.
5. **Framework consistency (4-way comparison):**
   - Load scored outputs from:
     - `Outputs/Uplift/Python/T-Learner/` → T-learner benefit scores
     - `Outputs/Uplift/Python/X-Learner/` → X-learner benefit scores
     - `Outputs/Causal-Forests/Python/` → Causal Forest benefit scores
   - Merge by `member_id` on test-set members.
   - Compute pairwise Spearman correlations (4×4 matrix).
   - Compute pairwise top-10% overlap (4×4 matrix).
6. **Cross-method agreement heatmap (NEW for DR notebook):**
   - Spearman correlation heatmap across all 4 methods.
   - Annotated cells with correlation values.
7. **Top-benefit examples:** Top 10 members by DR benefit score with key features.

**Expected Figures:**

| Figure | Filename | Figsize |
|--------|----------|---------|
| Average benefit by decile | `dashboard_doubly_robust_avg_benefit_by_decile.png` | (8, 4.5) |
| Risk tier by benefit group | `dashboard_doubly_robust_risk_tier_by_benefit_group.png` | (8, 4.5) |
| Cross-method agreement heatmap | `dashboard_doubly_robust_cross_method_agreement.png` | (8, 6) |

**Expected Tables:**

| Table | Key Columns |
|-------|-------------|
| `decile_summary` | hte_decile, n, mean_benefit_score, mean_true_benefit, observed_ed_treated, observed_ed_control |
| `risk_tier_benefit_group_summary` | risk_tier, benefit_group, n, pct |
| `top_decile_profile` | feature, mean_value, median_value |
| `consistency_summary` | method_a, method_b, spearman_corr, top_10_overlap, top_20_overlap |
| `top_benefit_examples` | member_id, benefit_score, true_benefit, key features |

**Expected CSV Outputs:**
- `doubly_robust_decile_summary.csv`
- `doubly_robust_risk_tier_benefit_group_summary.csv`
- `doubly_robust_top_decile_profile.csv`
- `doubly_robust_cross_method_consistency_summary.csv`
- `doubly_robust_top_benefit_examples.csv`

**Acceptance Criteria:**
- [ ] Decile 1 has the highest mean benefit score; decile 10 has the lowest.
- [ ] Risk-tier × benefit-group table sums to total test-set size.
- [ ] Cross-method comparison includes all 4 methods (T-learner, X-learner, Causal Forest, DR).
- [ ] Heatmap is symmetric with 1.0 on the diagonal.
- [ ] Graceful handling if upstream scored outputs are missing (skip comparison, log warning).
- [ ] Top-benefit examples table includes both model benefit and true benefit columns.

**Implementation Notes:**
- Use `safe_corr()` for all pairwise correlations to handle NaN/missing gracefully.
- Use `top_overlap()` for Jaccard-style overlap at 10% and 20% thresholds.
- The cross-method heatmap is unique to the DR notebook (Causal Forest only does 2-way comparison).
- If any upstream output is missing, the consistency summary should still report available pairs.

---

### 7.6 Task 6 — Variable Importance + SHAP (Level 2)

**Objective:** Explain which features drive the DR treatment-effect estimates using both
native RF importance and SHAP values applied to the final treatment-effect model.

**Methodology:**

1. **Extract the final RF treatment-effect model:**
   - The `ForestDRLearner` internally trains a `RandomForestRegressor` on pseudo-outcomes.
   - Access via `dr_model.model_final_` or equivalent attribute.
   - If direct access is unavailable, train a surrogate RF on `(x_test, benefit_score)`.
2. **Native RF variable importance:**
   - `model.feature_importances_` (Gini/MDI importance).
   - Top 20 features sorted by importance.
3. **SHAP analysis:**
   ```python
   explainer = shap.TreeExplainer(final_rf_model)
   shap_values = explainer.shap_values(x_test)
   ```
   - Global mean |SHAP| importance (top 20 features).
   - Member-level SHAP values saved for downstream use.
4. **SHAP bar chart:** Horizontal bar of top 15 features by mean |SHAP|.

**Expected Figures:**

| Figure | Filename | Figsize |
|--------|----------|---------|
| Variable importance (RF native) | `dashboard_doubly_robust_variable_importance.png` | (9, 5.5) |
| SHAP global importance | `dashboard_doubly_robust_global_benefit_shap.png` | (9, 5.5) |

**Expected Tables:**

| Table | Key Columns |
|-------|-------------|
| `variable_importance` | feature, importance, rank |
| `global_benefit_shap_importance` | feature, mean_abs_shap, rank |

**Expected CSV Outputs:**
- `doubly_robust_variable_importance.csv`
- `doubly_robust_global_benefit_shap_importance.csv`
- `doubly_robust_member_benefit_shap_values.csv`

**Acceptance Criteria:**
- [ ] Variable importance sums to approximately 1.0 (normalized MDI).
- [ ] SHAP values are computed on test-set observations.
- [ ] SHAP importance table has at minimum top 20 features.
- [ ] Member-level SHAP CSV has one row per test-set member and one column per feature.
- [ ] Both charts use horizontal bar layout with (9, 5.5) figsize.

**Implementation Notes:**
- The `ForestDRLearner` in econml stores the final CATE model. Inspect `dr_model.model_final_` or `dr_model.model_cate`.
- If the internal model is a `SubsampledHonestForest`, use `shap.TreeExplainer` with `feature_perturbation='tree_path_dependent'`.
- Fallback approach: Train a standalone `RandomForestRegressor(n_estimators=500, min_samples_leaf=10)` on `(x_train, pseudo_outcomes_train)` and apply SHAP to that model.

---

### 7.7 Task 7 — Business Value + Targeting (Level 2)

**Objective:** Translate treatment-effect estimates into actionable targeting policies
with ROI projections and cost-benefit analysis.

**Methodology:**

1. **Targeting summary:**
   - For each targeting threshold (top 10%, 20%, 30%, 40%, all, none):
     - Number of members targeted.
     - Mean benefit score of targeted group.
     - Mean true benefit of targeted group.
     - Estimated ED rate reduction vs. no intervention.
     - Expected ED visits avoided per 1,000 members.
2. **Cost-benefit analysis:**
   ```python
   ed_savings = ed_visits_avoided * COST_PER_ED_VISIT  # $1,200
   intervention_cost = n_targeted * COST_PER_INTERVENTION  # $250
   net_savings = ed_savings - intervention_cost
   roi = net_savings / intervention_cost
   ```
3. **Policy comparison with historical:**
   - Compare DR-based targeting against historical treatment assignment.
   - Compute DR policy value using the doubly-robust estimator formula.
4. **Targeting chart:** Net savings by targeting threshold (bar chart).
5. **Risk vs. benefit by decile chart:** Dual-axis or grouped bar showing risk score vs. benefit score by decile.

**Expected Figures:**

| Figure | Filename | Figsize |
|--------|----------|---------|
| Net savings by targeting policy | `dashboard_doubly_robust_targeting_roi.png` | (8, 4.5) |
| Risk vs. benefit by decile | `dashboard_doubly_robust_risk_vs_benefit_by_decile.png` | (8, 4.5) |

**Expected Tables:**

| Table | Key Columns |
|-------|-------------|
| `targeting_summary` | policy, n_targeted, treatment_rate, mean_benefit, ed_visits_avoided_per_1000, ed_savings_per_1000, intervention_cost_per_1000, net_savings_per_1000, roi |

**Expected CSV Outputs:**
- `doubly_robust_targeting_summary.csv`

**Acceptance Criteria:**
- [ ] Targeting summary includes at minimum 6 policies (none, top 10%, 20%, 30%, 40%, all).
- [ ] Cost assumptions are stated as constants, not hardcoded inline.
- [ ] ROI is computed as `net_savings / intervention_cost` (undefined for "treat nobody" → NaN).
- [ ] Net savings chart clearly labels the breakeven line.
- [ ] Risk vs. benefit chart demonstrates that high risk ≠ high benefit.

**Implementation Notes:**
- The DR policy evaluation formula from the legacy notebook can be retained as a validation check:
  ```python
  def dr_policy_value(y, w, e, m1, m0, policy):
      e = np.clip(e, 0.05, 0.95)
      value_i = policy * (m1 + (w / e) * (y - m1)) + (1 - policy) * (m0 + ((1 - w) / (1 - e)) * (y - m0))
      return float(np.nanmean(value_i))
  ```
- However, the primary effect estimates come from `ForestDRLearner.effect()`, not from the legacy formula.

---

## 8. Output Standards

### 8.1 Folder Structure

```
Outputs/
└── Doubly-Robust/
    └── Python/
        ├── doubly_robust_predictor_inventory.csv
        ├── doubly_robust_data_review_summary.csv
        ├── doubly_robust_event_count_summary.csv
        ├── doubly_robust_propensity_summary.csv
        ├── doubly_robust_pseudo_outcome_summary.csv
        ├── doubly_robust_scored_output.csv
        ├── doubly_robust_scored_test_output.csv
        ├── doubly_robust_effect_distribution_summary.csv
        ├── doubly_robust_ate_summary.csv
        ├── doubly_robust_true_benefit_validation_summary.csv
        ├── doubly_robust_decile_summary.csv
        ├── doubly_robust_risk_tier_benefit_group_summary.csv
        ├── doubly_robust_top_decile_profile.csv
        ├── doubly_robust_cross_method_consistency_summary.csv
        ├── doubly_robust_top_benefit_examples.csv
        ├── doubly_robust_variable_importance.csv
        ├── doubly_robust_global_benefit_shap_importance.csv
        ├── doubly_robust_member_benefit_shap_values.csv
        ├── doubly_robust_targeting_summary.csv
        ├── dashboard_doubly_robust_propensity_overlap.png
        ├── dashboard_doubly_robust_pseudo_outcome_distribution.png
        ├── dashboard_doubly_robust_effect_distribution.png
        ├── dashboard_doubly_robust_avg_benefit_by_decile.png
        ├── dashboard_doubly_robust_risk_tier_by_benefit_group.png
        ├── dashboard_doubly_robust_cross_method_agreement.png
        ├── dashboard_doubly_robust_variable_importance.png
        ├── dashboard_doubly_robust_global_benefit_shap.png
        ├── dashboard_doubly_robust_targeting_roi.png
        └── dashboard_doubly_robust_risk_vs_benefit_by_decile.png
```

### 8.2 Complete Output Inventory

| # | Filename | Type | Produced By |
|---|----------|------|-------------|
| 1 | `doubly_robust_predictor_inventory.csv` | CSV | Task 2 |
| 2 | `doubly_robust_data_review_summary.csv` | CSV | Task 2 |
| 3 | `doubly_robust_event_count_summary.csv` | CSV | Task 3 |
| 4 | `doubly_robust_propensity_summary.csv` | CSV | Task 3 |
| 5 | `doubly_robust_pseudo_outcome_summary.csv` | CSV | Task 3 |
| 6 | `doubly_robust_scored_output.csv` | CSV | Task 4 |
| 7 | `doubly_robust_scored_test_output.csv` | CSV | Task 4 |
| 8 | `doubly_robust_effect_distribution_summary.csv` | CSV | Task 4 |
| 9 | `doubly_robust_ate_summary.csv` | CSV | Task 4 |
| 10 | `doubly_robust_true_benefit_validation_summary.csv` | CSV | Task 4 |
| 11 | `doubly_robust_decile_summary.csv` | CSV | Task 5 |
| 12 | `doubly_robust_risk_tier_benefit_group_summary.csv` | CSV | Task 5 |
| 13 | `doubly_robust_top_decile_profile.csv` | CSV | Task 5 |
| 14 | `doubly_robust_cross_method_consistency_summary.csv` | CSV | Task 5 |
| 15 | `doubly_robust_top_benefit_examples.csv` | CSV | Task 5 |
| 16 | `doubly_robust_variable_importance.csv` | CSV | Task 6 |
| 17 | `doubly_robust_global_benefit_shap_importance.csv` | CSV | Task 6 |
| 18 | `doubly_robust_member_benefit_shap_values.csv` | CSV | Task 6 |
| 19 | `doubly_robust_targeting_summary.csv` | CSV | Task 7 |
| 20 | `dashboard_doubly_robust_propensity_overlap.png` | PNG | Task 3 |
| 21 | `dashboard_doubly_robust_pseudo_outcome_distribution.png` | PNG | Task 3 |
| 22 | `dashboard_doubly_robust_effect_distribution.png` | PNG | Task 4 |
| 23 | `dashboard_doubly_robust_avg_benefit_by_decile.png` | PNG | Task 5 |
| 24 | `dashboard_doubly_robust_risk_tier_by_benefit_group.png` | PNG | Task 5 |
| 25 | `dashboard_doubly_robust_cross_method_agreement.png` | PNG | Task 5 |
| 26 | `dashboard_doubly_robust_variable_importance.png` | PNG | Task 6 |
| 27 | `dashboard_doubly_robust_global_benefit_shap.png` | PNG | Task 6 |
| 28 | `dashboard_doubly_robust_targeting_roi.png` | PNG | Task 7 |
| 29 | `dashboard_doubly_robust_risk_vs_benefit_by_decile.png` | PNG | Task 7 |

**Total:** 19 CSV files + 10 PNG charts = 29 output artifacts.

### 8.3 CSV Format Standards

- UTF-8 encoding, no BOM.
- No row index (`index=False`).
- Float precision: default pandas (6 significant digits).
- Column names: snake_case.
- Missing values: empty string (pandas default CSV behavior).

### 8.4 PATHS Dict Template

```python
PATHS = {
    # Task 2
    'predictor_inventory': OUTPUT_DIR / 'doubly_robust_predictor_inventory.csv',
    'data_review_summary': OUTPUT_DIR / 'doubly_robust_data_review_summary.csv',
    # Task 3
    'event_count_summary': OUTPUT_DIR / 'doubly_robust_event_count_summary.csv',
    'propensity_summary': OUTPUT_DIR / 'doubly_robust_propensity_summary.csv',
    'pseudo_outcome_summary': OUTPUT_DIR / 'doubly_robust_pseudo_outcome_summary.csv',
    'propensity_chart': OUTPUT_DIR / 'dashboard_doubly_robust_propensity_overlap.png',
    'pseudo_outcome_chart': OUTPUT_DIR / 'dashboard_doubly_robust_pseudo_outcome_distribution.png',
    # Task 4
    'scored_output': OUTPUT_DIR / 'doubly_robust_scored_output.csv',
    'test_scored_output': OUTPUT_DIR / 'doubly_robust_scored_test_output.csv',
    'effect_distribution_summary': OUTPUT_DIR / 'doubly_robust_effect_distribution_summary.csv',
    'ate_summary': OUTPUT_DIR / 'doubly_robust_ate_summary.csv',
    'true_benefit_validation_summary': OUTPUT_DIR / 'doubly_robust_true_benefit_validation_summary.csv',
    'effect_distribution_chart': OUTPUT_DIR / 'dashboard_doubly_robust_effect_distribution.png',
    # Task 5
    'decile_summary': OUTPUT_DIR / 'doubly_robust_decile_summary.csv',
    'risk_tier_benefit_group_summary': OUTPUT_DIR / 'doubly_robust_risk_tier_benefit_group_summary.csv',
    'top_decile_profile': OUTPUT_DIR / 'doubly_robust_top_decile_profile.csv',
    'consistency_summary': OUTPUT_DIR / 'doubly_robust_cross_method_consistency_summary.csv',
    'top_benefit_examples': OUTPUT_DIR / 'doubly_robust_top_benefit_examples.csv',
    'benefit_decile_chart': OUTPUT_DIR / 'dashboard_doubly_robust_avg_benefit_by_decile.png',
    'risk_tier_benefit_group_chart': OUTPUT_DIR / 'dashboard_doubly_robust_risk_tier_by_benefit_group.png',
    'cross_method_chart': OUTPUT_DIR / 'dashboard_doubly_robust_cross_method_agreement.png',
    # Task 6
    'variable_importance': OUTPUT_DIR / 'doubly_robust_variable_importance.csv',
    'shap_importance': OUTPUT_DIR / 'doubly_robust_global_benefit_shap_importance.csv',
    'shap_values': OUTPUT_DIR / 'doubly_robust_member_benefit_shap_values.csv',
    'variable_importance_chart': OUTPUT_DIR / 'dashboard_doubly_robust_variable_importance.png',
    'shap_chart': OUTPUT_DIR / 'dashboard_doubly_robust_global_benefit_shap.png',
    # Task 7
    'targeting_summary': OUTPUT_DIR / 'doubly_robust_targeting_summary.csv',
    'targeting_chart': OUTPUT_DIR / 'dashboard_doubly_robust_targeting_roi.png',
    'risk_benefit_decile_chart': OUTPUT_DIR / 'dashboard_doubly_robust_risk_vs_benefit_by_decile.png',
}
```

---

## 9. Evaluation Philosophy

### 9.1 Two-Level Framework

The PRISM evaluation framework separates analytical credibility from business utility.
Both levels must pass for a model to be considered stakeholder-ready.

| Level | Name | Purpose | Tasks |
|-------|------|---------|-------|
| 1 | Treatment-Effect Credibility | Validate that the model's treatment-effect estimates are statistically meaningful and internally consistent | Tasks 3, 4, 5 |
| 2 | Explainability And Business Value | Demonstrate that the model produces interpretable, actionable recommendations with quantifiable ROI | Tasks 6, 7 |

### 9.2 Level 1 — Treatment-Effect Credibility

Level 1 asks: "Are the estimated treatment effects trustworthy?"

**Key questions addressed:**
- Is there sufficient overlap between treated and control groups (propensity)?
- Are pseudo-outcomes well-behaved (not dominated by extreme values)?
- Does the effect distribution show meaningful heterogeneity (not constant)?
- Does the model's ranking correlate with the known true-benefit formula?
- Do top-benefit members overlap across multiple estimation methods?

**Level 1 pass criteria:**
- Propensity AUC > 0.55 (treatment is not random but is predictable).
- Pseudo-outcome standard deviation is finite and not dominated by outliers.
- Spearman correlation with true benefit > 0.30.
- Top-10% overlap with at least one other method > 0.20.
- Decile 1 mean benefit > Decile 10 mean benefit (monotonic separation).

### 9.3 Level 2 — Explainability And Business Value

Level 2 asks: "Can we explain and act on the results?"

**Key questions addressed:**
- Which features drive the treatment-effect heterogeneity?
- Are the key drivers clinically and operationally sensible?
- Does benefit-based targeting produce positive net savings under realistic cost assumptions?
- At what targeting threshold does ROI turn positive?

**Level 2 pass criteria:**
- SHAP importance identifies at least 3 clinically meaningful features in the top 10.
- Net savings is positive for at least one targeting threshold (top 10%, 20%, or 30%).
- Variable importance ranking is interpretable to a non-technical stakeholder.

### 9.4 Level Summaries

Each level concludes with a markdown cell summarizing:
1. Key findings (2–3 bullet points).
2. Pass/fail assessment with supporting evidence.
3. Limitations or caveats.

---

## 10. Deliverables

### 10.1 Implementation Checklist

- [ ] **Notebook file:** `Code/PRISM_Doubly_Robust_Modeling_Workflow.ipynb`
- [ ] **Output folder:** `Outputs/Doubly-Robust/Python/` with all 29 artifacts
- [ ] **README update:** `PRISM_Doubly_Robust_Modeling_README.md` summarizing results
- [ ] **Legacy notebook preserved:** `Code/Doubly Robust Code.ipynb` unchanged
- [ ] **Shared utilities:** No modifications to `_prism_model_utils.py` required

### 10.2 Notebook Quality Checklist

- [ ] All cells execute top-to-bottom without error (Restart & Run All).
- [ ] No hardcoded absolute paths.
- [ ] All outputs saved via `save_csv()` and `save_current_figure()`.
- [ ] Every code cell preceded by a markdown explanation cell.
- [ ] No commented-out debug code in final version.
- [ ] All charts render at 160 DPI.
- [ ] `warnings.filterwarnings('ignore', category=UserWarning)` set globally.
- [ ] Print statements confirm each save operation.

### 10.3 Analytical Quality Checklist

- [ ] Stratified split matches Causal Forest (same seed, same stratification columns).
- [ ] Propensity source is documented (shared file vs. fallback model).
- [ ] True-benefit formula coefficients match specification exactly.
- [ ] Cross-method consistency includes all 4 methods when available.
- [ ] Sign convention is consistent: `benefit_score = -tau_hat` throughout.
- [ ] ATE reported with appropriate context (not interpreted as individual-level).
- [ ] Pseudo-outcome diagnostics clearly explain what the values represent.

---

## 11. Success Criteria

### 11.1 Functional Success

| Criterion | Measurement |
|-----------|-------------|
| Notebook executes end-to-end | Restart & Run All completes without error |
| All 29 output files created | File count in output folder = 29 |
| Scored output has correct schema | Columns match specification; no NaN in benefit_score |
| Charts render correctly | All PNGs are non-empty, ≥10 KB |
| Cross-method comparison works | At minimum DR vs. Causal Forest pair is reported |

### 11.2 Analytical Success

| Criterion | Threshold |
|-----------|-----------|
| Propensity overlap | AUC ∈ [0.55, 0.90] |
| Effect heterogeneity | benefit_score std > 0.001 |
| True-benefit correlation | Spearman ρ > 0.20 |
| Decile separation | Decile 1 mean > Decile 10 mean |
| Positive ROI | Net savings > 0 for at least one targeting policy |

### 11.3 Code Quality

| Criterion | Standard |
|-----------|----------|
| PEP 8 compliance | No errors from `flake8 --max-line-length=120` |
| Type consistency | All numeric arrays are `float64` or `int64` |
| Reproducibility | Two consecutive runs produce identical CSV content |
| Memory efficiency | Peak memory < 2 GB for 1,000-row dataset |

---

## 12. Appendix

### A. Deferred Items

| Item | Reason | Target Phase |
|------|--------|--------------|
| `DRLearner` with custom neural-net nuisance models | Complexity; requires PyTorch | Phase 2 |
| Confidence intervals via bootstrap | Computationally expensive for demonstration | Phase 2 |
| Interactive dashboard (Plotly/Streamlit) | Out of scope for notebook deliverable | Phase 3 |
| Causal sensitivity analysis (Rosenbaum bounds) | Advanced methodology not required for demonstration | Phase 2 |
| Live-data validation | Requires production data pipeline | Phase 3 |

### B. Key Differences From Causal Forest Notebook

| Aspect | Causal Forest | Doubly Robust |
|--------|--------------|---------------|
| Estimator | `CausalForestDML` | `ForestDRLearner` |
| Estimation mechanism | Honest splitting + local CATE | Pseudo-outcome regression |
| Hyperparameter tuning | Grid search (3 candidates) | Not required |
| Confidence intervals | Available via `effect_inference()` | Not available |
| Unique diagnostic | Uncertainty summary (tau_se) | Pseudo-outcome distribution |
| Cross-method comparison | 2-way (vs. uplift) | 4-way (vs. all methods) |
| SHAP target | Causal forest model effect | Final RF on pseudo-outcomes |

### C. Shared Propensity Score Contract

The shared propensity file (`Outputs/Uplift/Python/X-Learner/shared_propensity_scores.csv`) must have:

| Column | Type | Description |
|--------|------|-------------|
| `member_id` | int | Sequential integer matching `model_df.member_id` |
| `split` | str | Either `'train'` or `'test'` |
| `propensity_score` | float | Clipped to [0.05, 0.95] |

**Contract rules:**
- No duplicate `member_id` values.
- Every member in `model_df` must have a corresponding row.
- If file is missing or malformed, notebook falls back to internal propensity model.

### D. True-Benefit Formula Reference

```python
true_benefit = (
    0.020
    + 0.018 * ed_visits_last_6m
    + 0.015 * admits_last_6m
    + 0.018 * food_insecurity_flag
    + 0.014 * transportation_barrier_flag
    + 0.012 * behavioral_health_risk_flag
    + 0.0006 * np.maximum(current_risk_score - 50, 0)
)
```

This formula is identical across all four PRISM workflows. It serves as a validation benchmark,
not as a model input. The model should recover this signal without explicit access to the formula.

### E. ForestDRLearner API Reference

```python
from econml.dr import ForestDRLearner

dr_model = ForestDRLearner(
    model_regression=RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=123,
        n_jobs=-1,
    ),
    model_propensity=LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 30),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=123),
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.5],
        scoring='roc_auc',
        max_iter=10000,
        random_state=123,
    ),
    cv=5,
    min_samples_leaf=10,
    n_estimators=500,
    random_state=123,
)

# Fit
dr_model.fit(Y=y_train, T=w_train, X=x_train, W=None)

# Predict treatment effects
tau_hat = dr_model.effect(x_test).flatten()

# Access final model for SHAP
final_rf = dr_model.model_final_  # or dr_model.model_cate
```

### F. Migration From Legacy Notebook

| Legacy Feature | Disposition |
|---------------|-------------|
| `fit_xgb_binary()` nuisance models | Replaced by RF inside ForestDRLearner |
| `dr_policy_value()` function | Retained as validation utility only |
| `predict_xgb()` / `clip_probs()` | Not needed (handled internally by econml) |
| 2-file CSV output | Expanded to 19 CSV outputs |
| No charts | Expanded to 10 PNG charts |
| No SHAP | Full SHAP analysis added |
| No stratified split | Stratified on treatment + outcome |
| No shared propensity | Shared propensity from X-learner |
| No evaluation framework | Two-level evaluation framework |

---

*End of specification.*
