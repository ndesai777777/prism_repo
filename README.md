# PRISM — Predictive Risk Intervention Strategy for Members

PRISM applies causal inference and uplift modeling to identify care management members who would benefit most from intervention, measured by reduction in emergency department (ED) visits within 90 days. The project implements four complementary modeling approaches — T-Learner, X-Learner, Causal Forest, and Doubly Robust Learner — plus a Clinical Confidence Layer that clusters high-benefit members into actionable archetypes. All workflows run on a 1,000-member synthetic dataset to demonstrate methodology before production deployment.

---

## Business Context

Traditional care management targeting relies on **risk scores** — members most likely to use the ED are prioritized for outreach. However, high-risk members are not necessarily the same members who would *change behavior* in response to intervention.

PRISM reframes the targeting question using **heterogeneous treatment effect (HTE) estimation**: for each member, estimate the *causal difference* in ED utilization between receiving intervention and not receiving it. Members with the largest negative treatment effects (greatest predicted benefit) are prioritized.

Key business value:
- Directs limited care management resources toward members most likely to respond
- Quantifies individual-level intervention benefit rather than population-average effects
- Provides clinical confidence scoring to support care manager decision-making
- Demonstrates that benefit-based targeting outperforms risk-based targeting at every threshold tested

---

## Repository Structure

```
prism_repo/
├── README.md                                          ← This file
├── PRISM_Intervention_Benefit_Modeling_README.md       ← Uplift modeling detailed README
├── PRISM_Causal_Forest_Modeling_README.md              ← Causal Forest detailed README
├── PRISM_Doubly_Robust_Modeling_README.md              ← Doubly Robust detailed README
├── PRISM_BEST_Confidence_Readme.md                    ← Clinical Confidence Layer detailed README
│
├── Code/
│   ├── Uplift Model Code_rh06032026.ipynb             ← T-Learner & X-Learner (Python)
│   ├── PRISM_Causal_Forest_Modeling_Workflow.ipynb     ← Causal Forest (EconML GRF)
│   ├── PRISM_Doubly_Robust_Modeling_Workflow.ipynb     ← Doubly Robust Learner (EconML)
│   ├── PRISM_Clinical_Confidence_Layer_v2.ipynb       ← GMM clustering + confidence scoring
│   ├── _prism_model_utils.py                          ← Shared utilities (split, design matrix, GPU)
│   ├── generate_all_readmes.py                        ← Unified README generator (91 generators)
│   ├── generate_readme_tables.py                      ← Uplift-specific generators (delegates)
│   ├── Causal Forests Model Code.R                    ← R reference implementation
│   ├── Doubly Robust Code.R                           ← R reference implementation
│   ├── T-Learner Model Code.R                         ← R reference implementation
│   └── T-Learner Model Code.ipynb                     ← Legacy notebook
│
├── DataSets/
│   └── PRP_1000_full_pretreatment.xlsx                ← 1,000-member synthetic dataset
│
├── Outputs/
│   ├── Uplift/Python/                                 ← T-Learner & X-Learner outputs
│   │   ├── T-Learner/GLMNet/                          ← Scored members, decile summaries, charts
│   │   ├── X-Learner/GLMNet/                          ← Scored members, decile summaries, charts
│   │   └── X-Learner/XGBoost/                         ← Scored members, decile summaries, charts
│   ├── Causal-Forests/Python/                         ← Scored outputs, SHAP, targeting summaries
│   ├── Doubly-Robust/Python/                          ← Scored outputs, cross-method agreement
│   └── Clinical-Confidence-Layer/Python/              ← Clustering outputs, confidence tiers
│
└── Other_Misc/                                        ← Supporting documents and exploratory data
```

---

## Modeling Approaches

### 1. Uplift Modeling (T-Learner & X-Learner)

| Aspect | Detail |
|--------|--------|
| **Notebook** | `Code/Uplift Model Code_rh06032026.ipynb` |
| **Methods** | T-Learner (GLMNet, XGBoost) · X-Learner (GLMNet, XGBoost) |
| **Predictors** | 41 features across demographics, clinical, SDoH, utilization, pharmacy, risk scores |
| **Key design** | Shared propensity model across X-Learner variants; elastic-net regularization for GLMNet |
| **Output** | Individual CATE estimates, decile summaries, benefit-vs-risk targeting comparison |

The T-Learner fits separate outcome models for treated and control groups, then differences them. The X-Learner refines this by using the opposite group's model to impute counterfactual outcomes, then blends via propensity weighting — particularly effective under treatment imbalance.

### 2. Causal Forest

| Aspect | Detail |
|--------|--------|
| **Notebook** | `Code/PRISM_Causal_Forest_Modeling_Workflow.ipynb` |
| **Method** | Generalized Random Forest (EconML `CausalForestDML`) |
| **Predictors** | 56 candidate features (filtered to available columns) |
| **Key design** | Honest splitting, hyperparameter tuning, SHAP-based variable importance |
| **Output** | Member-level τ(x) estimates, SHAP values, uncertainty quantification |

Causal Forests use honest sample splitting (separate samples for partition learning vs. estimation) to produce valid confidence intervals on individual treatment effects, providing interpretable importance rankings via SHAP.

### 3. Doubly Robust Learner

| Aspect | Detail |
|--------|--------|
| **Notebook** | `Code/PRISM_Doubly_Robust_Modeling_Workflow.ipynb` |
| **Method** | DR-Learner (`LinearDRLearner` from EconML) |
| **Predictors** | Same candidate set as Causal Forest |
| **Key design** | Doubly robust estimation (consistent if either outcome or propensity model is correct) |
| **Cross-method agreement** | Spearman ρ ≈ 0.9 with Causal Forest rankings |
| **Output** | Member-level CATE, SHAP importance, cross-method consistency metrics |

The DR-Learner combines inverse propensity weighting with outcome regression, providing robustness against model misspecification in either component.

### 4. Clinical Confidence Layer

| Aspect | Detail |
|--------|--------|
| **Notebook** | `Code/PRISM_Clinical_Confidence_Layer_v2.ipynb` |
| **Method** | GMM clustering on top-20% high-benefit members |
| **Features** | 10 continuous clinical features → PCA (7 components) |
| **Confidence score** | Two-signal: posterior probability × Mahalanobis typicality |
| **Validation** | HDBSCAN cross-verification, bootstrap stability analysis |
| **Current status** | Machinery works as designed; archetype split has WEAK bootstrap stability |

This layer sits downstream of the CATE models. It clusters high-benefit members into clinical archetypes (e.g., "high utilizers with chronic conditions" vs. "behavioral health–driven ED use") and assigns confidence scores indicating how prototypical each member is of their archetype.

---

## Data Description

**Source:** `DataSets/PRP_1000_full_pretreatment.xlsx`

| Property | Value |
|----------|-------|
| Sample size | 1,000 members (synthetic) |
| Treatment variable | `intervention_flag` (binary: 1 = received care management) |
| Outcome variable | `outcome_ed_90d` (binary: 1 = ED visit within 90 days) |
| Candidate predictors | 56 features across 8 categories |

**Predictor categories:**

| Category | Examples |
|----------|----------|
| Demographics | `age`, `gender`, `race_ethnicity` |
| Clinical | `percolator_clinical_score`, `chronic_condition_count`, `comorbidity_index` |
| Social Determinants (SDoH) | `sdoh_composite_score`, `housing_instability_flag` |
| Utilization History | `ed_visits_last_6m`, `inpatient_admits_last_12m`, `total_claims_last_12m` |
| Pharmacy | `polypharmacy_flag`, `medication_adherence_score` |
| Risk Scores | `current_risk_score`, `prior_risk_score` |
| Behavioral Health | `bh_diagnosis_flag`, `substance_use_flag` |
| Program Engagement | `prior_outreach_attempts`, `engagement_score` |

---

## Environment Setup

### Primary Execution Environment

All notebooks are designed to run on **AWS SageMaker** with GPU support (XGBoost CUDA acceleration).

### Python Dependencies

```
python >= 3.9
pandas
numpy
scikit-learn
xgboost              # GPU-enabled (tree_method='gpu_hist')
econml               # CausalForestDML, LinearDRLearner
shap
matplotlib
seaborn
plotly
openpyxl             # Excel I/O
scipy
hdbscan              # Clinical Confidence Layer cross-verification
```

### Reproducibility

All notebooks use **seed = 123** for deterministic train/test splits, model fitting, and clustering. Results are fully reproducible given the same environment and data.

---

## How to Run

### Execution Order

The notebooks are independent of each other (no cross-notebook dependencies) but share the same input dataset and utility module. Recommended order:

```
1. Uplift Model Code_rh06032026.ipynb        → Produces Outputs/Uplift/Python/
2. PRISM_Causal_Forest_Modeling_Workflow.ipynb → Produces Outputs/Causal-Forests/Python/
3. PRISM_Doubly_Robust_Modeling_Workflow.ipynb → Produces Outputs/Doubly-Robust/Python/
4. PRISM_Clinical_Confidence_Layer_v2.ipynb   → Produces Outputs/Clinical-Confidence-Layer/Python/
```

### What Each Notebook Produces

| Notebook | Key Outputs |
|----------|-------------|
| Uplift Model | Scored member files (CATE per member), decile summaries, benefit-vs-risk targeting charts, model diagnostics |
| Causal Forest | Scored output with τ(x) + confidence intervals, SHAP importance CSVs, variable importance plots, targeting summaries |
| Doubly Robust | Scored output with CATE, cross-method agreement metrics (vs. Causal Forest), SHAP values, decile summaries |
| Clinical Confidence | Cluster assignments, confidence tier CSVs, archetype profiles, stability diagnostics, PCA visualizations |

### Shared Utilities

`Code/_prism_model_utils.py` provides:
- Consistent train/test splitting (70/30, stratified by treatment)
- Design matrix construction (one-hot encoding, imputation)
- GPU availability detection and XGBoost device configuration
- Common output formatting and export helpers

---

## README Auto-Generation

Each sub-project has a detailed README that auto-regenerates from model outputs. The reporting pipeline:

```
Notebook execution → CSV/PNG outputs → generate_all_readmes.py → Markdown READMEs
```

### Running the Generator

```bash
python Code/generate_all_readmes.py
```

This script contains **91 generator functions** that read output CSVs and produce formatted Markdown tables, inline metrics, and chart references for all four sub-project READMEs. Each notebook's final cell calls this script automatically after execution.

### Sub-Project READMEs

| README File | Content |
|-------------|---------|
| `PRISM_Intervention_Benefit_Modeling_README.md` | Uplift model results, decile tables, targeting comparisons |
| `PRISM_Causal_Forest_Modeling_README.md` | Causal Forest results, SHAP importance, uncertainty analysis |
| `PRISM_Doubly_Robust_Modeling_README.md` | DR-Learner results, cross-method agreement |
| `PRISM_BEST_Confidence_Readme.md` | Clinical Confidence Layer results, cluster profiles, stability |

---

## Key Results Summary

### Cross-Method Comparison

| Metric | T-Learner (GLMNet) | X-Learner (GLMNet) | X-Learner (XGBoost) | Causal Forest | Doubly Robust |
|--------|--------------------|--------------------|---------------------|---------------|---------------|
| Estimation approach | Two separate models | Counterfactual imputation | Counterfactual imputation | Honest forest splitting | Doubly robust AIPW |
| Regularization | Elastic net | Elastic net | Tree-based | Honest splitting | Linear final stage |
| SHAP importance | ✗ | ✗ | ✗ | ✓ | ✓ |
| Confidence intervals | ✗ | ✗ | ✗ | ✓ | ✓ |
| Cross-method agreement | — | — | — | Spearman ρ ≈ 0.9 (vs DR) | Spearman ρ ≈ 0.9 (vs CF) |

### Consistent Findings Across Methods

1. **Benefit-based targeting outperforms risk-based targeting** at every intervention threshold tested (top 10%, 20%, 30%, etc.)
2. **Top drivers of intervention benefit** (consistent across Causal Forest and DR-Learner SHAP):
   - `percolator_clinical_score`
   - `age`
   - `current_risk_score`
   - `ed_visits_last_6m`
3. **High cross-method agreement** (Spearman ρ ≈ 0.9) — member rankings are stable across estimation strategies
4. **Identifiable high-benefit subpopulation** — top decile shows meaningfully larger treatment effects than population average

---

## Current Status & Recommendations

### Production-Ready Components

| Component | Status | Recommendation |
|-----------|--------|----------------|
| Causal Forest CATE estimation | ✅ Validated | Ready for pilot scoring |
| Doubly Robust CATE estimation | ✅ Validated | Ready for pilot scoring (cross-validates CF) |
| Uplift Models (T/X-Learner) | ✅ Validated | Suitable for ensemble or comparison |
| Benefit-based targeting logic | ✅ Validated | Replace or augment risk-based prioritization |
| SHAP-based explainability | ✅ Validated | Ready for clinical review dashboards |

### Components Requiring Further Work

| Component | Status | Issue | Next Step |
|-----------|--------|-------|-----------|
| Clinical Confidence Layer — archetype clustering | ⚠️ WEAK stability | Bootstrap stability below threshold | Shadow-mode only; increase sample size or simplify cluster count |
| Clinical Confidence Layer — confidence scoring | ✅ Machinery works | Dependent on stable clusters | Deploy scoring once clusters validate |

### Recommended Next Steps

1. **Pilot deployment** — Score production members with Causal Forest or DR-Learner; rank by predicted benefit
2. **A/B validation** — Randomize outreach among top-benefit members to measure real-world lift
3. **Scale dataset** — Re-run Clinical Confidence Layer on full population (>1,000 members) to test archetype stability
4. **Dashboard integration** — Surface SHAP explanations and confidence scores in care manager tooling
5. **Model refresh cadence** — Retrain quarterly as new outcome data accumulates

---

## Contributing

This repository is maintained by the **Acentra Health Data Science & Analytics team**.

### Workflow

1. Create a feature branch from `main`
2. Run the relevant notebook(s) end-to-end to regenerate outputs
3. Run `python Code/generate_all_readmes.py` to update sub-project READMEs
4. Commit outputs and READMEs together with code changes
5. Open a pull request for review

### Contact

For questions about methodology, data, or access, reach out to the PRISM project team via internal channels.

---

*Last updated: July 2026*
