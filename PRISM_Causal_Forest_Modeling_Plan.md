# PRISM Causal Forest Modeling README And Notebook Plan

This planning document describes what should go into the separate causal forest README and how the existing causal forest notebook should be improved before final results are written up. It is intended as a review document before editing `Code/Causal Forests Model Code.ipynb` or generating the final causal forest modeling README.

The causal forest analysis should be treated as a companion to `PRISM_Intervention_Benefit_Modeling_README.md`. The project background, business question, outcome variable, treatment variable, and broad predictor set remain the same. The new README should focus specifically on the causal forest workflow, how it estimates heterogeneous treatment effects, and how its results compare with the existing T-learner and X-learner uplift modeling work.

## Recommended File Strategy

Create a separate README for the causal forest analysis:

```text
PRISM_Causal_Forest_Modeling_README.md
```

Keep the existing README focused on the current uplift modeling workflow:

```text
PRISM_Intervention_Benefit_Modeling_README.md
```

The causal forest README should include a short opening note explaining that it is a companion analysis. The existing uplift README can later receive a short cross-reference near the end that points readers to the causal forest README.

Suggested companion note for the causal forest README:

```markdown
This README is a companion to `PRISM_Intervention_Benefit_Modeling_README.md`.
The project background, business question, outcome variable, treatment variable,
and predictor set remain the same. This document focuses specifically on the
causal forest modeling workflow and its estimates of heterogeneous treatment
effects.
```

## Reproducibility Rule

Use the same random seed as the existing T-learner and X-learner workflow:

```text
seed = 123
```

The current uplift modeling notebook, T-learner notebook, doubly robust notebook, and existing causal forest notebook all use `123` for train/test splitting and model reproducibility. The causal forest notebook should keep this same seed wherever a random process is used:

- Train/test split
- Causal forest model fitting
- Cross-fitting or nuisance model fitting, if applicable
- Random forest, gradient boosting, or other nuisance models
- Any bootstrap, subsampling, or sensitivity checks

## Final Causal Forest README Outline

The causal forest README should follow the same report-style structure as the existing intervention benefit modeling README, but the analytical tasks should be adapted to causal forest modeling.

Recommended final section order:

```text
# PRISM Causal Forest Modeling Report Draft
## Background
## Business Question
## Project Objectives
## Analytical Task 1: Understanding And Explaining The Causal Forest Framework
## Analytical Task 2: Data Review
## Analytical Task 3: Causal Forest Diagnostics And Estimation Credibility
## Analytical Task 4: Treatment Effect Analysis
## Analytical Task 5: HTE Decile And High-Value Subgroup Analysis
## Analytical Task 6: Variable Importance And Explainability
## Analytical Task 7: Business Value Assessment
## Analytical Task 8: Client Perspective
## Recommendation
## Presentation Summary
## Reproducibility
```

This keeps the same spine as `PRISM_Intervention_Benefit_Modeling_README.md`. The main intentional difference is Analytical Task 3. In the uplift README, Task 3 evaluates factual outcome-model performance because the T-learner and X-learner depend on outcome models. In the causal forest README, Task 3 should instead evaluate whether the causal forest treatment-effect estimates are credible enough to interpret.

## Title And Opening Summary

Suggested title:

```markdown
# PRISM Causal Forest Modeling Report Draft
```

The opening paragraph should explain:

- This report summarizes the causal forest modeling workflow.
- The analysis estimates heterogeneous treatment effects for 90-day ED utilization.
- The causal forest model is a third treatment-effect framework, alongside the existing T-learner and X-learner work.
- Outputs are saved separately from the uplift modeling outputs.

Suggested output folder:

```text
Outputs/Causal-Forests/Python
```

If the R script remains part of the workflow, also mention:

```text
Outputs/Causal-Forests/R
```

## Background

Reuse most of the existing background from `PRISM_Intervention_Benefit_Modeling_README.md`.

The key message should remain:

- Care management programs need to prioritize members when intervention resources are limited.
- High baseline ED risk does not always mean high expected intervention benefit.
- The goal is to identify members who are most likely to benefit from intervention, not simply members who are most likely to have an ED visit.
- The dataset is synthetic and should be interpreted as a reproducible modeling demonstration.

The causal forest-specific addition should be:

- Causal forest is used to estimate whether treatment benefit varies across members.
- This member-level variation is called heterogeneous treatment effect, or HTE.

## Business Question

Use the same core business question, adapted slightly for causal forest:

```markdown
Which members are most likely to benefit from intervention in terms of reducing
90-day emergency department utilization, based on causal forest estimates of
heterogeneous treatment effects?
```

Supporting questions:

- Which member characteristics are associated with larger estimated treatment benefit?
- Do causal forest high-benefit groups look similar to the T-learner and X-learner high-benefit groups?
- Can causal forest help identify high-value subgroups for care management review?
- What limitations should be considered before operational use?

## Project Objectives

This section should explain that the causal forest analysis is meant to:

- Estimate member-level heterogeneous treatment effects.
- Rank members by estimated intervention benefit.
- Identify high-benefit deciles or subgroups.
- Compare causal forest rankings with T-learner and X-learner rankings.
- Provide a partial explainability layer through variable importance and subgroup patterns.
- Document limitations around synthetic data, small sample size, rare outcomes, treatment assignment, and overlap.

## Analytical Task 1: Understanding And Explaining The Causal Forest Framework

This should be the most important setup section. It should explain causal forest in plain language before showing results.

### Outcome Variable

Use:

```text
outcome_ed_90d
```

Explain that `outcome_ed_90d` is a binary indicator for whether the member had emergency department utilization within 90 days.

### Treatment Variable

Use:

```text
intervention_flag
```

Explain that `1` means the member received the care management intervention and `0` means the member did not.

### Predictor Variables

Reuse the existing predictor categories:

- Demographics
- Clinical conditions
- Social determinants of health
- Utilization
- Pharmacy
- Risk scores
- Outreach or engagement fields, if retained in the causal forest notebook

The README should state whether the causal forest model uses the same predictor inventory as the T-learner and X-learner workflow. If the current causal forest notebook uses a simplified predictor set, that simplification should be documented clearly.

### Train/Test Methodology

Use the same train/test logic as the existing uplift workflow unless there is a strong reason to change it:

```text
70% training
30% held-out test
seed = 123
```

The split should preserve treatment/outcome structure when possible. The existing uplift README explains that stratification by treatment status and ED outcome is important because the outcome is rare. The causal forest README should include the same rationale.

### What A Causal Forest Estimates

Explain that a causal forest estimates a treatment effect for each member:

```text
tau_hat = estimated effect of treatment on outcome_ed_90d
```

Because `outcome_ed_90d` is a bad outcome, the sign needs to be handled carefully:

```text
tau_hat < 0 means intervention is estimated to reduce ED risk
tau_hat > 0 means intervention is estimated to increase ED risk
```

For business interpretation, define:

```text
benefit_score = -tau_hat
```

Then:

```text
higher benefit_score = larger estimated ED risk reduction from intervention
```

This sign convention should be used consistently throughout the notebook and README.

### How Causal Forest Differs From T-Learner And X-Learner

This subsection should compare the frameworks:

| Framework | Main idea | Main output |
|---|---|---|
| T-learner | Train separate treated and control outcome models, then subtract predictions | `pred_ed_if_control - pred_ed_if_treated` |
| X-learner | Impute treatment effects, model them, then combine using propensity weights | Weighted treatment-effect estimate |
| Causal forest | Use trees to directly learn where treatment effects vary across members | `tau_hat` and `benefit_score` |

The important message:

- T-learner and X-learner are counterfactual outcome modeling approaches.
- Causal forest is more directly focused on treatment-effect heterogeneity and subgroup discovery.
- Causal forest is useful when the main question is: "For whom does the intervention appear most beneficial?"

### Modeling Technique Used

Explain the actual implementation:

- Python notebook uses `econml.dml.CausalForestDML`.
- R script uses `grf::causal_forest`.

The final README should focus on whichever notebook is treated as the primary workflow. Since the existing uplift README is Python-oriented, the Python causal forest notebook should likely become the main report source.

If using `CausalForestDML`, explain:

- The model estimates treatment effects while adjusting for covariates.
- It uses nuisance models for outcome and treatment assignment.
- It can estimate heterogeneous treatment effects at the member level.
- It can provide treatment-effect uncertainty estimates or intervals depending on implementation.

Avoid overclaiming. The README should say the model estimates treatment effects under causal assumptions; it does not prove causality from synthetic observational data.

## Analytical Task 2: Data Review

This section should mirror the existing data review section.

Include a table with:

| Metric | Current value |
|---|---:|
| Total members | TBD |
| Treated members | TBD |
| Untreated/control members | TBD |
| Treatment rate | TBD |
| ED outcome events | TBD |
| Outcome prevalence | TBD |
| Treated observed ED rate | TBD |
| Control observed ED rate | TBD |
| Final predictors before one-hot encoding | TBD |
| Model matrix columns after encoding | TBD |

Add causal forest-specific interpretation:

- Rare ED outcomes can make HTE estimates unstable.
- Non-random treatment assignment can create confounding.
- Causal forest depends on overlap: members with similar features should exist in both treated and control groups.
- If certain high-risk subgroups are almost always treated or almost never treated, treatment-effect estimates for those subgroups will be less reliable.

Expected notebook output:

```text
causal_forest_data_review_summary.csv
```

## Analytical Task 3: Causal Forest Diagnostics And Estimation Credibility

This section should sit in the same location as "Analytical Task 3: Model Performance" in the uplift README, but it should not be written as an outcome-prediction performance section. Causal forest does not primarily produce two factual ED risk models for report interpretation the way the T-learner does. Instead, this section should answer:

- Is there enough treated/control and outcome information to support HTE estimation?
- Is there adequate treatment overlap/common support?
- What is the distribution of estimated treatment effects?
- How uncertain are the treatment-effect estimates?
- Are the estimates stable enough to support decile and subgroup interpretation?

Use this section as the causal forest equivalent of "can we trust the modeling foundation enough to continue?"

Recommended diagnostics:

### Event Counts And Modeling Constraints

Include train/test counts by treatment and outcome:

| Split | Group | N | Positive ED events | Negative ED events | Event rate |
|---|---|---:|---:|---:|---:|
| Train | Treated | TBD | TBD | TBD | TBD |
| Train | Control | TBD | TBD | TBD | TBD |
| Test | Treated | TBD | TBD | TBD | TBD |
| Test | Control | TBD | TBD | TBD | TBD |

Expected notebook output:

```text
causal_forest_event_count_summary.csv
```

### Propensity And Overlap Checks

Add a treatment assignment model or use the causal forest nuisance treatment model to summarize propensity overlap.

Useful outputs:

- Minimum, maximum, mean, and percentile distribution of propensity scores
- Treated/control propensity histograms
- Share of members with very low or very high estimated propensity
- Optional trimmed sample sensitivity check

Suggested table:

| Metric | Value |
|---|---:|
| Mean propensity | TBD |
| Min propensity | TBD |
| 5th percentile | TBD |
| Median propensity | TBD |
| 95th percentile | TBD |
| Max propensity | TBD |
| Members below 0.05 | TBD |
| Members above 0.95 | TBD |

Expected notebook outputs:

```text
causal_forest_propensity_summary.csv
dashboard_propensity_overlap.png
```

### Treatment Effect Distribution

Summarize the distribution of `tau_hat` and `benefit_score`.

Suggested table:

| Metric | tau_hat | benefit_score |
|---|---:|---:|
| Mean | TBD | TBD |
| Std dev | TBD | TBD |
| Min | TBD | TBD |
| 10th percentile | TBD | TBD |
| 25th percentile | TBD | TBD |
| Median | TBD | TBD |
| 75th percentile | TBD | TBD |
| 90th percentile | TBD | TBD |
| Max | TBD | TBD |

Expected notebook outputs:

```text
causal_forest_effect_distribution_summary.csv
dashboard_causal_forest_effect_distribution.png
```

### Uncertainty Checks

If standard errors are available, summarize `tau_se`.

Recommended fields:

- Average standard error by decile
- Percent of members with confidence intervals entirely below zero for `tau_hat`
- Percent of members with confidence intervals crossing zero
- Top decile average uncertainty

Expected notebook output:

```text
causal_forest_uncertainty_summary.csv
```

## Analytical Task 4: Treatment Effect Analysis

This should be the core causal forest results section.

Explain:

- Each member receives `tau_hat`.
- Since ED utilization is undesirable, a negative `tau_hat` indicates estimated benefit.
- The business-facing `benefit_score` is calculated as `-tau_hat`.
- Members can be ranked by `benefit_score`.

Suggested member example table:

| Member profile | Actual outcome | Treatment flag | tau_hat | tau_se | Benefit score | HTE decile | Outreach interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Highest benefit | TBD | TBD | TBD | TBD | TBD | 1 | Strong outreach candidate based on estimated ED reduction. |
| High risk, low benefit | TBD | TBD | TBD | TBD | TBD | TBD | High baseline risk but limited estimated impactability. |
| Low risk, high benefit | TBD | TBD | TBD | TBD | TBD | TBD | May be missed by risk-only targeting. |
| Lowest benefit | TBD | TBD | TBD | TBD | TBD | 10 | Not prioritized by causal forest benefit score. |

Expected notebook output:

```text
causal_forest_top_benefit_examples.csv
```

Optional chart:

```text
dashboard_causal_forest_member_effect_examples.png
```

## Analytical Task 5: HTE Decile And High-Value Subgroup Analysis

This section maps directly to the methodology coverage plan: causal forest covers high-value subgroup identification through deciles and HTE.

Rank members by:

```text
benefit_score = -tau_hat
```

Assign:

```text
hte_decile = 1 for highest estimated benefit
hte_decile = 10 for lowest estimated benefit
```

Suggested decile table:

| HTE decile | N | Avg tau_hat | Avg benefit score | Avg tau_se | Observed ED rate | Treatment pct | Avg current risk score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 9 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Recommended charts:

```text
dashboard_causal_forest_avg_benefit_by_decile.png
dashboard_causal_forest_tau_by_decile.png
dashboard_causal_forest_risk_vs_benefit_by_decile.png
```

The written interpretation should answer:

- Does the causal forest create a clear gradient in estimated benefit?
- Are the highest-benefit deciles also the highest-risk deciles?
- Does causal forest identify impactable members who would not be prioritized by current risk alone?
- Are decile-level standard errors large enough to weaken confidence?

Expected notebook output:

```text
causal_forest_decile_summary.csv
```

### Framework Consistency Check

This should be included inside Analytical Task 5 rather than becoming its own analytical task. That keeps the causal forest README aligned with the current uplift README, where Task 5 is also the decile/uplift-ranking section.

The causal forest README should not exist in isolation from the prior uplift modeling work. After the causal forest decile results are shown, compare the causal forest ranking with the GLMNet T-learner and GLMNet X-learner rankings.

Compare causal forest benefit scores with:

- GLMNet T-learner benefit scores
- GLMNet X-learner benefit scores
- Optional XGBoost T-learner benefit scores if still useful
- Current risk score

Recommended comparison metrics:

- Pearson correlation
- Spearman rank correlation
- Top decile overlap
- Top 20% overlap
- Average causal forest benefit among T-learner top decile
- Average T-learner benefit among causal forest top decile

Suggested table:

| Comparison | Pearson corr | Spearman corr | Top decile overlap | Top 20% overlap |
|---|---:|---:|---:|---:|
| Causal forest vs GLMNet T-learner | TBD | TBD | TBD | TBD |
| Causal forest vs GLMNet X-learner | TBD | TBD | TBD | TBD |
| Causal forest vs current risk score | TBD | TBD | TBD | TBD |

Recommended chart:

```text
dashboard_causal_forest_vs_uplift_comparison.png
```

Expected notebook output:

```text
causal_forest_vs_uplift_consistency_summary.csv
```

Interpretation guidance:

- Strong positive rank correlation would suggest that the frameworks identify similar high-benefit members.
- Weak correlation does not automatically mean causal forest is wrong; it may mean the frameworks are learning different kinds of heterogeneity.
- Low top-decile overlap should be discussed carefully because operational targeting would differ by model.

## Analytical Task 6: Variable Importance And Explainability

The methodology coverage sheet marks causal forest explainability as partial. The README should be careful and not overstate what variable importance means.

Include:

- Causal forest variable importance
- Top features associated with treatment-effect heterogeneity
- Optional subgroup profiles for top HTE decile
- Optional comparison between causal forest variable importance and T/X learner benefit drivers

Suggested table:

| Rank | Feature | Causal forest importance |
|---:|---|---:|
| 1 | TBD | TBD |
| 2 | TBD | TBD |
| 3 | TBD | TBD |
| 4 | TBD | TBD |
| 5 | TBD | TBD |

Recommended chart:

```text
dashboard_causal_forest_variable_importance.png
```

Expected notebook output:

```text
causal_forest_variable_importance.csv
```

Interpretation language:

```markdown
Causal forest variable importance identifies variables that help the model split
members into groups with different estimated treatment effects. It should not be
read as a definitive causal explanation of why intervention works.
```

Optional subgroup profile table:

| Feature | Top HTE decile mean/rate | Other deciles mean/rate | Difference |
|---|---:|---:|---:|
| TBD | TBD | TBD | TBD |

Expected notebook output:

```text
causal_forest_top_decile_profile.csv
```

## Analytical Task 7: Business Value Assessment

This section should occupy the same position as "Analytical Task 7: Business Value Assessment" in the uplift README, but it should be narrower unless ROI simulation is added to the causal forest workflow.

Because the methodology coverage sheet marks causal forest as not currently covering alternative strategy simulation, the causal forest README should not make ROI the main story unless that logic is added later. Instead, this section should assess business value in terms of targeting usefulness and subgroup discovery.

Focus instead on:

- How causal forest could help identify high-benefit subgroups.
- How the top HTE decile could be reviewed by care management leaders.
- How causal forest can serve as a challenger model to T-learner and X-learner prioritization.
- Why the results should be treated as exploratory.

Recommended business-value outputs:

```text
causal_forest_targeting_summary.csv
causal_forest_top_decile_profile.csv
```

If ROI is added later, this section can be expanded to mirror the uplift README more closely by comparing causal forest targeting against current-risk targeting. Until then, keep the language as "potential operational value" rather than estimated savings.

## Analytical Task 8: Client Perspective

This section should mirror the role of the client perspective section in the uplift README. It should translate the causal forest findings into stakeholder language and clearly separate promising insight from operational proof.

Core limitations:

- Synthetic dataset
- Small sample size
- Rare ED outcome
- Observational treatment assignment
- Potential confounding
- Overlap/common support concerns
- Treatment-effect uncertainty
- Sensitivity to modeling assumptions
- Need for live-data validation

Suggested operational workflow:

1. Score members using the causal forest model.
2. Convert `tau_hat` into `benefit_score = -tau_hat`.
3. Rank members by benefit score.
4. Assign HTE deciles.
5. Review top HTE decile member profiles and variable importance.
6. Compare with T-learner and X-learner high-benefit members.
7. Validate prospectively or on live historical data before operational deployment.

## Recommendation

The recommendation should say whether causal forest should be used as:

- A primary prioritization model,
- A challenger model to the T-learner and X-learner,
- A subgroup-discovery tool,
- Or an exploratory method requiring more validation.

Suggested initial recommendation before final results:

```markdown
The causal forest model should be presented as a challenger and subgroup-discovery
framework rather than a replacement for the T-learner and X-learner workflow.
Its main value is estimating heterogeneous treatment effects directly and helping
identify high-benefit member subgroups. Final operational use would require
stronger validation on live data, overlap checks, treatment-effect uncertainty
review, and comparison against existing uplift rankings.
```

## Presentation Summary

Suggested slide flow:

1. Business problem: high risk is not always high benefit.
2. Why causal forest: direct HTE estimation.
3. Data review and modeling constraints.
4. Causal forest diagnostics and estimation credibility.
5. Treatment-effect analysis.
6. HTE decile and high-value subgroup findings.
7. Comparison with T-learner and X-learner rankings.
8. Variable importance and explainability.
9. Business value and operational use.
10. Client perspective, limitations, and recommendation.

## Reproducibility

The final README should list:

- Primary notebook:

```text
Code/Causal Forests Model Code.ipynb
```

- Optional R comparison script:

```text
Code/Causal Forests Model Code.R
```

- Output folder:

```text
Outputs/Causal-Forests/Python
```

- Seed:

```text
123
```

- Key output files:

```text
causal_forest_scored_output.csv
causal_forest_decile_summary.csv
causal_forest_variable_importance.csv
causal_forest_data_review_summary.csv
causal_forest_event_count_summary.csv
causal_forest_propensity_summary.csv
causal_forest_effect_distribution_summary.csv
causal_forest_uncertainty_summary.csv
causal_forest_top_benefit_examples.csv
causal_forest_vs_uplift_consistency_summary.csv
causal_forest_top_decile_profile.csv
```

The final README should also explain how generated tables and charts are refreshed. If a causal forest README table generator is added, use a command like:

```bash
python Code/generate_causal_forest_readme_tables.py
```

If causal forest output generation is folded into the existing table generator, document:

```bash
python Code/generate_readme_tables.py
```

## Existing Notebook Improvement Plan

The current `Code/Causal Forests Model Code.ipynb` already has a useful starting structure:

- Package loading
- File paths
- Helper functions
- Data reading
- Required fields
- Date features
- Simplified predictors
- Data type handling
- Model matrix creation
- Train/test split with `seed=123`
- `CausalForestDML` fitting
- Individual treatment-effect prediction
- Decile summary
- Variable importance
- Full-file scoring
- CSV output
- Basic interpretation

The notebook should be improved into a report-ready workflow with stronger diagnostics, cleaner outputs, and README-ready tables/charts.

Implementation note added during build-out:

- The report-ready causal forest workflow has been implemented as `Code/PRISM_Causal_Forest_Modeling_Workflow.ipynb`.
- This notebook follows the section order in this plan and is intended to be the primary causal forest modeling artifact.
- The workflow keeps `SEED = 123`, uses `econml.dml.CausalForestDML`, writes the planned CSV/chart outputs, and preserves the causal forest sign convention `benefit_score = -tau_hat`.
- Local execution requires a Python environment compatible with `econml`. The previously available Python 3.14 environment could not install `econml` because its scikit-learn dependency requires an older compatible wheel or a local C/C++ build toolchain.

## Notebook Section Plan

### 1. Notebook Header

Add a clear markdown header:

```markdown
# PRISM Causal Forest Modeling

This notebook estimates heterogeneous treatment effects for 90-day ED utilization
using a causal forest model. It is a companion to the T-learner and X-learner
uplift modeling workflow.
```

Include:

- Outcome
- Treatment
- Seed
- Input data
- Output folder
- Primary interpretation of `tau_hat` and `benefit_score`

### 2. Configuration Cell

Create a single configuration cell near the top:

```python
SEED = 123
TRAIN_FRACTION = 0.70
OUTCOME_COL = "outcome_ed_90d"
TREATMENT_COL = "intervention_flag"
OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Causal-Forests" / "Python"
```

All later code should refer to these constants.

### 3. Package And Environment Setup

Keep package loading, but make it cleaner:

- Avoid installing packages inside the notebook unless absolutely necessary.
- If package installation cells remain, mark them as optional.
- Print package versions for important libraries.

Recommended packages:

- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`
- `econml`
- project utilities from `_prism_model_utils.py`

### 4. File Paths

Confirm:

```text
Outputs/Causal-Forests/Python
```

Add paths for all expected outputs.

Recommended output path variables:

```python
scored_output_path
decile_summary_path
variable_importance_path
data_review_summary_path
event_count_summary_path
propensity_summary_path
effect_distribution_summary_path
uncertainty_summary_path
top_benefit_examples_path
consistency_summary_path
top_decile_profile_path
```

### 5. Data Load And Required Field Checks

Keep existing checks, but make them stricter:

- Confirm outcome column exists.
- Confirm treatment column exists.
- Confirm member identifier column exists, if available.
- Confirm no invalid treatment values outside `0` and `1`.
- Confirm no invalid outcome values outside `0` and `1`.

Output a short data audit table.

### 6. Predictor Definition

This needs a deliberate decision.

Option A: Use the same predictor set as the T/X learner workflow.

Option B: Use a simplified causal forest predictor set.

Recommendation:

- Use the same predictor categories and as close to the same predictor set as possible.
- If causal forest requires simplification, document exactly which variables were removed and why.

The notebook should output:

```text
causal_forest_predictor_inventory.csv
```

with columns like:

```text
feature
category
data_type
included_in_model
reason_if_excluded
```

### 7. Train/Test Split

Use:

```python
train_df, test_df = split_train_test(model_df, train_fraction=0.70, seed=123)
```

or equivalent.

Confirm that the split is stratified by treatment and outcome if the helper supports it.

Output:

```text
causal_forest_event_count_summary.csv
```

### 8. Model Matrix Creation

Use the same preprocessing approach as the T/X learner workflow where practical:

- Numeric variables handled consistently.
- Binary variables kept as 0/1.
- Categorical variables one-hot encoded.
- Missing values handled consistently.
- Train/test columns aligned.

The notebook should print:

- Number of raw predictors
- Number of model matrix columns after encoding
- Number of dropped columns, if any

### 9. Propensity And Overlap Diagnostics

Add this section before fitting or immediately after nuisance model fitting.

Recommended approach:

- Fit a simple propensity model on training data using `SEED = 123`.
- Predict propensity on train and test.
- Summarize distribution overall and by actual treatment group.
- Plot treated and control propensity distributions.

Output:

```text
causal_forest_propensity_summary.csv
dashboard_propensity_overlap.png
```

This is important because causal forest estimates are less reliable when overlap is weak.

### 10. Fit Causal Forest Model

Keep `CausalForestDML`, but move all important parameters into a readable cell.

Use `random_state=SEED`.

Recommended parameters to document:

- Number of trees
- Minimum leaf size
- Subforest size, if set
- Nuisance outcome model
- Nuisance treatment model
- Whether treatment is discrete
- Cross-fitting folds, if used

The notebook should print the final model configuration.

### 11. Estimate Treatment Effects On Test Set

Generate:

```python
tau_hat
tau_se
benefit_score = -tau_hat
hte_decile
```

Use `hte_decile` as the preferred causal forest naming, while optionally also keeping `uplift_decile` for consistency with prior outputs.

Recommended:

```python
results_test["tau_hat"] = tau_test
results_test["tau_se"] = tau_se_test
results_test["benefit_score"] = -results_test["tau_hat"]
results_test["hte_decile"] = ntile_desc(results_test["benefit_score"], 10)
results_test["uplift_decile"] = results_test["hte_decile"]
```

### 12. Average Treatment Effect Summary

Add a section that reports:

- Mean `tau_hat`
- Mean `benefit_score`
- Confidence interval if available
- Interpretation in plain English

Output:

```text
causal_forest_ate_summary.csv
```

### 13. Treatment Effect Distribution

Add a table and chart summarizing `tau_hat` and `benefit_score`.

Output:

```text
causal_forest_effect_distribution_summary.csv
dashboard_causal_forest_effect_distribution.png
```

### 14. HTE Decile Summary

Improve the current decile summary to include:

- `hte_decile`
- `n`
- `avg_tau_hat`
- `avg_benefit_score`
- `avg_tau_se`
- `observed_ed_rate`
- `treatment_pct`
- `avg_current_risk_score`, if available
- Optional confidence interval fields

Output:

```text
causal_forest_decile_summary.csv
dashboard_causal_forest_avg_benefit_by_decile.png
dashboard_causal_forest_tau_by_decile.png
```

### 15. Member Example Selection

Add a reproducible way to select:

- Highest benefit member
- High risk, low benefit member
- Low risk, high benefit member
- Lowest benefit member

Output:

```text
causal_forest_top_benefit_examples.csv
```

### 16. Variable Importance

Keep variable importance, but make the output README-ready.

Output:

```text
causal_forest_variable_importance.csv
dashboard_causal_forest_variable_importance.png
```

Include a note:

```text
Variable importance reflects features used to split members into groups with
different estimated treatment effects. It is not a definitive causal explanation.
```

### 17. High-Benefit Subgroup Profile

Compare top HTE decile versus all other deciles.

Recommended fields:

- Mean/rate in top HTE decile
- Mean/rate outside top HTE decile
- Difference

Output:

```text
causal_forest_top_decile_profile.csv
```

### 18. Comparison With T-Learner And X-Learner Outputs

If existing T/X learner outputs are available, load:

```text
Outputs/Uplift/Python/T-Learner/GLMNet/uplift_scored_output.csv
Outputs/Uplift/Python/X-Learner/GLMNet/...
```

The exact X-learner scored output path should be confirmed during implementation.

Compare on a stable member key if available. If no stable member ID exists, compare only if row alignment is defensible and documented.

Output:

```text
causal_forest_vs_uplift_consistency_summary.csv
dashboard_causal_forest_vs_uplift_comparison.png
```

### 19. Save Full Scored Output

The full scored file should include:

- Member identifier, if available
- Outcome
- Treatment
- Key risk score fields
- `tau_hat`
- `tau_se`
- `benefit_score`
- `hte_decile`
- `uplift_decile`

Output:

```text
causal_forest_scored_output.csv
```

### 20. README Snippet Or Table Generator

After the notebook outputs are stable, create one of two options:

Option A:

```text
Code/generate_causal_forest_readme_tables.py
```

Option B:

Extend:

```text
Code/generate_readme_tables.py
```

Recommendation:

Use a separate causal forest table generator first. It will be easier to review and less likely to disrupt the existing uplift README generation.

## Minimum Output Checklist Before Writing Final README

The final causal forest README should not be written until these files exist:

- `causal_forest_scored_output.csv`
- `causal_forest_decile_summary.csv`
- `causal_forest_variable_importance.csv`
- `causal_forest_data_review_summary.csv`
- `causal_forest_event_count_summary.csv`
- `causal_forest_effect_distribution_summary.csv`
- `causal_forest_top_benefit_examples.csv`

Strongly recommended additional outputs:

- `causal_forest_propensity_summary.csv`
- `causal_forest_uncertainty_summary.csv`
- `causal_forest_top_decile_profile.csv`
- `causal_forest_vs_uplift_consistency_summary.csv`
- `dashboard_propensity_overlap.png`
- `dashboard_causal_forest_effect_distribution.png`
- `dashboard_causal_forest_avg_benefit_by_decile.png`
- `dashboard_causal_forest_variable_importance.png`
- `dashboard_causal_forest_vs_uplift_comparison.png`

## Main Review Questions Before Implementation

Before editing the causal forest notebook, decide:

1. Should the Python notebook be the primary causal forest workflow, with the R script kept as a reference?
2. Should the causal forest use the exact same predictor set as the T/X learner workflow, or keep the simplified predictor set currently in the notebook?
3. Is there a stable member identifier that can be used to compare causal forest, T-learner, and X-learner scores?
4. Should the final causal forest README include T/X learner comparison as a core section or an appendix-style section?
5. Should ROI remain out of scope for causal forest until the model diagnostics and decile outputs are stronger?

## Recommended Direction

Use a separate causal forest README and treat causal forest as a companion model. The final story should be:

- The T-learner and X-learner README explains counterfactual uplift modeling.
- The causal forest README explains direct heterogeneous treatment-effect modeling.
- Both analyses answer the same business question, but from different modeling frameworks.
- The causal forest model should initially be positioned as a challenger and subgroup-discovery tool, not as a replacement for the existing uplift modeling workflow.
