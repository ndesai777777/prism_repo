# PRISM Intervention Benefit Modeling Report Guide

This README-style guide is ordered to match the project requirements document:
`Internship Project_PRISM Intervention Benefit Modeling (2).pdf`.

Use this file as the structure for the written report and presentation. Replace bracketed placeholders with final values from the notebook outputs after rerunning:

- Notebook: `Code/Uplift Model Code_rh06032026.ipynb`
- Main output root: `Outputs/Uplift/Python`
- XGBoost outputs: `Outputs/Uplift/Python/XGBoost`
- GLMNet outputs: `Outputs/Uplift/Python/GLMNet`

## Background

Care management programs must decide which members should receive intervention when resources are limited. The key project idea is that the highest-risk members are not always the members who benefit most from intervention.

This project evaluates an intervention benefit, or uplift, modeling framework to estimate the impact of intervention on future 90-day emergency department utilization. The goal is to identify members who are most likely to benefit from care management activities.

The analysis uses the PRP synthetic dataset. Although synthetic, the dataset is intended to resemble the structure, variables, and business use cases of a real Medicaid population. Results should therefore be communicated as if they were being presented to a Medicaid health plan leadership team, while clearly stating that validation on future live data would still be required.

The project should also document a reproducible modeling workflow that can later be applied to a live production dataset.

## Business Question

Primary question:

> Which members are most likely to benefit from intervention in terms of reducing 90-day ED utilization?

Additional questions to address:

- What factors drive future ED utilization?
- What factors appear to influence intervention effectiveness?
- How can intervention resources be prioritized using model results?
- What business value could be generated through targeted intervention?

## Project Objectives

The report should evaluate, explain, and communicate results from the intervention benefit modeling framework.

Include the following objectives:

- Explain the modeling framework clearly for both technical and non-technical audiences.
- Evaluate technical model performance.
- Evaluate practical business usefulness for care management prioritization.
- Interpret model outputs for business stakeholders.
- Identify data, modeling, and operational considerations before live deployment.
- Demonstrate a reproducible workflow that can be reused with a future client dataset.

## Analytical Task 1: Understanding And Explaining The Modeling Framework

### Outcome Variable

Include:

- Outcome: `outcome_ed_90d`
- Interpretation: binary indicator for whether a member had Emergency Department utilization within 90 days.
- Modeling goal: estimate the probability of this outcome under treated and untreated scenarios.

### Treatment Variable

Include:

- Treatment: `intervention_flag`
- Interpretation: whether the member received the care management intervention, 0 indicating no intervention and 1 indicating intervention.

### Predictor Variables

Summarize predictor categories:

- Demographics, such as age, gender, language, county, plan type.
- Clinical conditions, such as diabetes, CHF, COPD, asthma, depression, anxiety, CKD.
- Social needs, such as food insecurity, housing instability, transportation barriers, utilities insecurity.
- Prior utilization and cost, such as ED visits, admits, PCP visits, specialist visits, total cost.
- Medication and risk fields, such as medication adherence, high-cost drug flag, opioid flag, polypharmacy flag, percolator scores, current risk score, risk tier.

Source:

- Feature setup in `Code/Uplift Model Code_rh06032026.ipynb`
- Output: `Outputs/Uplift/Python/data_review_summary.csv` after rerunning the notebook.

### Train/Test Methodology

Include:

- The dataset is split into training and test data.
- Treated and control members are separated within the training set.
- Models are trained on training data and evaluated on held-out test data.
- Cross-validation is used for tuning/comparison.

### T-Learner Framework

Explain:

The notebook uses a T-learner approach, meaning it trains two separate outcome models:

1. Treated model: estimates `P(ED | Treated, member features)`
2. Control model: estimates `P(ED | Control, member features)`

The predicted benefit score is:

```text
benefit_score = pred_ed_if_control - pred_ed_if_treated
```

Higher benefit score means the model predicts a larger reduction in ED risk if the member receives intervention.

### Why Separate Treated And Untreated Models Are Built

Include:

- The relationship between predictors and ED risk may differ depending on whether a member receives intervention.
- A single risk model predicts future ED utilization, but does not directly estimate intervention benefit.
- Separate treated and control models allow each member to receive two counterfactual risk estimates:
  - predicted ED risk if treated
  - predicted ED risk if untreated

### How Uplift Scores Are Generated

Include:

```text
pred_ed_if_treated = predicted probability of ED within 90 days if treated
pred_ed_if_control = predicted probability of ED within 90 days if untreated
benefit_score = pred_ed_if_control - pred_ed_if_treated
uplift_decile = decile ranking of benefit_score, where 1 = highest predicted benefit
```

## Analytical Task 2: Data Review

Summarize:

- Number of members: `[fill from data_review_summary.csv]`
- Number treated: `[fill from data_review_summary.csv]`
- Number untreated/control: `[fill from data_review_summary.csv]`
- Treatment rate: `[fill from data_review_summary.csv]`
- Outcome prevalence: `[fill from data_review_summary.csv]`
- Treated outcome rate: `[fill from data_review_summary.csv]`
- Control outcome rate: `[fill from data_review_summary.csv]`
- Number of predictors: `[fill from data_review_summary.csv]`

Use:

- `Outputs/Uplift/Python/data_review_summary.csv`

Discuss data quality and modeling assumptions:

- The dataset is synthetic and may not fully represent live client data.
- Treatment assignment may not be randomized.
- Observed treated-control differences may reflect confounding.
- Rare outcome prevalence can cause predicted probabilities to cluster near zero.
- Future deployment would require validation on live data and monitoring over time.

## Analytical Task 3: Model Performance

The project guidelines ask for model performance for both treated and untreated models.

### Required Metrics

For both XGBoost and GLMNet, report:

- Treated test AUC
- Control test AUC
- Treated cross-validation AUC
- Control cross-validation AUC
- Treated Brier score
- Control Brier score
- Treated calibration error
- Control calibration error

Use:

- `Outputs/Uplift/Python/model_evaluation_summary.csv`
- `Outputs/Uplift/Python/XGBoost/model_brier_scores.csv`
- `Outputs/Uplift/Python/GLMNet/model_brier_scores.csv`
- `Outputs/Uplift/Python/XGBoost/calibration_summary.csv`
- `Outputs/Uplift/Python/GLMNet/calibration_summary.csv`
- `Outputs/Uplift/Python/XGBoost/calibration_by_decile.csv`
- `Outputs/Uplift/Python/GLMNet/calibration_by_decile.csv`

### AUC Interpretation Scale

Use the project guideline scale:

| AUC | Interpretation |
|---:|---|
| 0.50 | No discrimination, similar to coin flip |
| 0.60 | Weak |
| 0.70 | Acceptable |
| 0.80 | Strong |
| 0.90+ | Excellent |

### How To Discuss AUC

Include:

- AUC evaluates whether each risk model ranks ED risk well.
- AUC is useful, but it is not the final objective of the uplift model.
- The business objective is ranking members by expected intervention benefit, not simply classifying ED utilization.

### How To Discuss Brier Score

Include:

- Brier score measures probability accuracy at the individual level.
- Lower Brier score is better.
- Brier score is important because ROI depends on the magnitude of predicted probabilities, not just ranking.

### How To Discuss Calibration Error

Include:

- Calibration compares average predicted ED rate to observed ED rate within predicted-risk deciles.
- Treated calibration is evaluated only among actually treated members using `pred_ed_if_treated`.
- Control calibration is evaluated only among actually untreated/control members using `pred_ed_if_control`.
- Lower absolute calibration error means the predicted probabilities are more believable.

### Sensitivity And Specificity

The guidelines mention sensitivity and specificity as optional.

Recommended wording:

Sensitivity and specificity are less central for this project because the goal is not to classify ED visits directly. The uplift framework estimates:

```text
P(ED | Control) - P(ED | Treated)
```

The more relevant question is whether members can be ranked by expected intervention benefit.

## Analytical Task 4: Treatment Effect Analysis

Explain:

- `pred_ed_if_treated`: estimated ED probability if a member receives intervention.
- `pred_ed_if_control`: estimated ED probability if a member does not receive intervention.
- `benefit_score`: estimated ED risk reduction from intervention.
- `uplift_decile`: member ranking based on predicted benefit.

Use examples from:

- `Outputs/Uplift/Python/XGBoost/uplift_scored_output.csv`
- `Outputs/Uplift/Python/GLMNet/uplift_scored_output.csv`

Suggested example table columns:

| Member | Actual outcome | Treatment flag | Predicted ED if treated | Predicted ED if control | Benefit score | Uplift decile |
|---|---:|---:|---:|---:|---:|---:|
| Example 1 | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Example 2 | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

Key interpretation:

- A member with high control risk and meaningfully lower treated risk has high predicted benefit.
- A member with high ED risk in both scenarios may be high risk but not necessarily high benefit.

## Analytical Task 5: Uplift Decile Analysis

Review and summarize uplift decile results.

For each model, include:

- Average benefit score by decile
- Observed ED rate by decile
- Predicted ED risk if treated by decile
- Predicted ED risk if control by decile
- Treatment penetration by decile
- Observed treated-control ED gap by decile

Use:

- `Outputs/Uplift/Python/XGBoost/uplift_decile_summary.csv`
- `Outputs/Uplift/Python/GLMNet/uplift_decile_summary.csv`
- `Outputs/Uplift/Python/XGBoost/uplift_observed_gap_by_decile.csv`
- `Outputs/Uplift/Python/GLMNet/uplift_observed_gap_by_decile.csv`
- `Outputs/Uplift/Python/XGBoost/top_benefit_decile_summary.csv`
- `Outputs/Uplift/Python/GLMNet/top_benefit_decile_summary.csv`

### Questions To Answer

Which deciles appear to receive the greatest predicted benefit?

- Usually decile 1 should have the highest average benefit score.
- Confirm using `avg_benefit_score` or `avg_predicted_benefit`.

Are there deciles that appear unlikely to benefit?

- Look at lower deciles.
- If benefit is near zero or negative, the model suggests limited benefit.

How could intervention resources be prioritized?

- Prioritize decile 1 first, then additional deciles depending on intervention capacity, observed validation, and ROI assumptions.

### Important Caution

High observed ED rate in the top benefit decile does not by itself prove uplift quality. It may indicate that the model is finding high-risk members.

The more relevant validation is:

```text
observed_control_minus_treated_gap = control_observed_ed_rate - treated_observed_ed_rate
```

If the top benefit decile has a positive observed gap, it directionally supports the model's benefit ranking. If the gap is negative or unstable, discuss sample size and confounding.

## Analytical Task 6: Variable Importance And Explainability

The guidelines ask for both traditional importance and SHAP analysis.

### Risk Drivers

Risk drivers explain what predicts ED risk within treated and control models.

Use:

- `Outputs/Uplift/Python/XGBoost/shap_importance_treated_control_models.csv`
- `Outputs/Uplift/Python/GLMNet/shap_importance_treated_control_models.csv`
- `Outputs/Uplift/Python/XGBoost/dashboard_shap_treated_model.png`
- `Outputs/Uplift/Python/XGBoost/dashboard_shap_control_model.png`
- `Outputs/Uplift/Python/GLMNet/dashboard_shap_treated_model.png`
- `Outputs/Uplift/Python/GLMNet/dashboard_shap_control_model.png`

Discuss:

- Top predictors in treated model.
- Top predictors in untreated/control model.
- Whether the same variables are important in both models.
- Whether findings are clinically and operationally reasonable.

### Benefit Drivers

Benefit drivers explain what influences predicted treatment benefit:

```text
pred_ed_if_control - pred_ed_if_treated
```

Use:

- `Outputs/Uplift/Python/XGBoost/shap_importance_benefit_score.csv`
- `Outputs/Uplift/Python/GLMNet/shap_importance_benefit_score.csv`
- `Outputs/Uplift/Python/XGBoost/dashboard_shap_benefit_score.png`
- `Outputs/Uplift/Python/GLMNet/dashboard_shap_benefit_score.png`

For XGBoost:

```text
benefit_shap = control_shap - treated_shap
```

For GLMNet:

```text
benefit_contribution = control_contribution - treated_contribution
```

Explain:

- `mean_abs_benefit_shap` or `mean_abs_benefit_contribution` shows magnitude of influence on benefit, regardless of direction.
- `mean_signed_benefit_shap` or `mean_signed_benefit_contribution` shows whether the feature tends to increase or decrease predicted benefit.
- `pct_positive_benefit_shap` or `pct_positive_benefit_contribution` shows the share of members where the feature increased predicted benefit.

### P-Values And Significance Tests

The guidelines note that XGBoost does not naturally produce p-values or confidence intervals like traditional regression.

Recommended wording:

For machine learning models such as XGBoost, variable importance is typically evaluated using out-of-sample performance, feature importance, SHAP values, and stability across folds or samples rather than classical p-values.

### Collinearity

Include if available:

- Correlation matrix for numeric predictors.
- VIF table from a logistic regression benchmark, if calculated.

Recommended wording:

Collinearity is generally a larger interpretability concern for regression-style models than for tree-based models such as XGBoost. It can still affect coefficient interpretation in GLMNet, so correlated predictors should be reviewed before live deployment.

## Analytical Task 7: Business Value Assessment

Review the ROI analysis generated by the model.

Report:

- Estimated ED visits avoided
- Estimated gross savings
- Estimated intervention costs
- Estimated net savings
- ROI by uplift decile

Use:

- `Outputs/Uplift/Python/XGBoost/uplift_roi_by_decile.csv`
- `Outputs/Uplift/Python/GLMNet/uplift_roi_by_decile.csv`
- `Outputs/Uplift/Python/XGBoost/dashboard_roi_net_savings_by_decile.png`
- `Outputs/Uplift/Python/GLMNet/dashboard_roi_net_savings_by_decile.png`

### ROI Formula

Include:

```text
expected_ed_rate_reduction = avg_benefit_score
expected_ed_visits_avoided = n * expected_ed_rate_reduction
gross_savings = expected_ed_visits_avoided * cost_per_ed_visit
intervention_cost = n * cost_per_intervention
net_savings = gross_savings - intervention_cost
roi = net_savings / intervention_cost
```

### ROI Assumptions

Document:

- Cost per ED visit: `[fill from notebook, currently 1200 if unchanged]`
- Cost per intervention: `[fill from notebook, currently 250 if unchanged]`
- The ROI values are estimates and depend heavily on these assumptions.
- ROI should be treated as directional until validated on live data.

### Business Interpretation

Discuss:

- Whether the top uplift deciles show enough predicted benefit to justify intervention costs.
- Whether targeting only high-benefit deciles improves ROI compared with intervening broadly.
- Whether estimated savings are sensitive to ED cost, intervention cost, and model calibration.

## Analytical Task 8: Client Perspective

Assume the audience is a Medicaid health plan executive team.

Prepare concise responses to these questions.

### How Accurate Is The Model?

Discuss:

- Treated and control AUC.
- Brier scores.
- Calibration error.
- Whether performance is acceptable, weak, or strong according to the AUC scale.

Use:

- `Outputs/Uplift/Python/model_evaluation_summary.csv`

### Why Should We Trust The Results?

Discuss:

- Held-out test evaluation.
- Cross-validation.
- Calibration and Brier checks.
- Explainability outputs.
- Consistency or differences between XGBoost and GLMNet.
- Need for future validation on live data.

### Which Members Should Be Targeted?

Discuss:

- Members in uplift decile 1 first.
- Expand to additional deciles depending on operational capacity and ROI.
- Do not target purely by risk score; target by predicted benefit.

Use:

- `Outputs/Uplift/Python/XGBoost/top_benefit_decile_summary.csv`
- `Outputs/Uplift/Python/GLMNet/top_benefit_decile_summary.csv`

### What Variables Drive Predictions?

Discuss:

- Risk drivers from treated/control model-driver outputs.
- Benefit drivers from benefit-driver outputs.
- Separate "risk drivers" from "benefit drivers" in the write-up.

### How Explainable Are The Results?

Discuss:

- XGBoost has SHAP explanations.
- GLMNet has standardized coefficient/logit contribution explanations.
- Benefit-driver analysis helps explain why members are ranked as high benefit.

### What Limitations Should We Be Aware Of?

Include:

- Synthetic data may not reflect live population behavior.
- Treatment assignment may be confounded.
- Observed treated-control gaps are not randomized treatment effects.
- Small sample sizes within deciles can create unstable observed rates.
- Rare outcome prevalence can compress predicted probabilities toward zero.
- ROI depends on cost assumptions and model calibration.
- Future live-data validation is required before operational deployment.

### What Assumptions Were Made In ROI Calculations?

Include:

- Average predicted benefit is treated as expected ED rate reduction.
- Cost per ED visit and intervention cost are fixed assumptions.
- Every targeted member receives the intervention.
- Model-estimated benefit translates into actual avoided ED visits.

### How Would These Results Be Used Operationally?

Describe:

1. Score members using the trained uplift workflow.
2. Rank members by benefit score.
3. Assign uplift deciles.
4. Prioritize outreach beginning with decile 1.
5. Review benefit drivers for operational context.
6. Track outcomes after intervention.
7. Recalibrate and retrain on live data over time.

## Written Report Requirements

The written report should include these sections:

1. Business problem
2. Methodology
3. Data review
4. Model performance
5. Treatment effect analysis
6. Uplift decile findings
7. Explainability results
8. Business value and ROI
9. Limitations
10. Recommendations

### Suggested Recommendation Language

Use language like:

The uplift modeling framework provides a structured way to prioritize members by predicted intervention benefit rather than baseline ED risk alone. AUC should be interpreted as a diagnostic for the treated and control risk models, while uplift decile performance, calibration, observed treated-control gaps, and ROI estimates should guide business interpretation.

If the top deciles show positive predicted benefit, acceptable calibration, and directionally favorable observed treated-control gaps, the model can support targeted intervention prioritization. However, because the current data are synthetic and treatment assignment may be confounded, the workflow should be validated on live data before production deployment.

## Presentation Requirements

The presentation should summarize:

- Business objective
- Modeling approach
- Data overview
- Key model performance findings
- Uplift decile results
- Explainability results
- Business value and ROI
- Limitations
- Potential next steps

### Suggested Slide Order

1. Project objective and business question
2. Why high risk is not always high benefit
3. Data overview
4. T-learner methodology
5. Model performance summary
6. Uplift decile results
7. Top benefit decile interpretation
8. Risk drivers vs benefit drivers
9. ROI and business value
10. Limitations and deployment considerations
11. Recommendations and next steps

## Reproducibility Guide

To reproduce:

1. Open `Code/Uplift Model Code_rh06032026.ipynb`.
2. Restart the kernel.
3. Run all cells in order.
4. Confirm outputs are written to:
   - `Outputs/Uplift/Python/XGBoost`
   - `Outputs/Uplift/Python/GLMNet`
5. Use the generated CSVs and PNGs to fill the report and presentation.

Key output files:

- `Outputs/Uplift/Python/data_review_summary.csv`
- `Outputs/Uplift/Python/model_evaluation_summary.csv`
- `Outputs/Uplift/Python/model_recommendation_summary.csv`
- `Outputs/Uplift/Python/XGBoost/uplift_decile_summary.csv`
- `Outputs/Uplift/Python/GLMNet/uplift_decile_summary.csv`
- `Outputs/Uplift/Python/XGBoost/uplift_roi_by_decile.csv`
- `Outputs/Uplift/Python/GLMNet/uplift_roi_by_decile.csv`
- `Outputs/Uplift/Python/XGBoost/calibration_summary.csv`
- `Outputs/Uplift/Python/GLMNet/calibration_summary.csv`
- `Outputs/Uplift/Python/XGBoost/uplift_observed_gap_by_decile.csv`
- `Outputs/Uplift/Python/GLMNet/uplift_observed_gap_by_decile.csv`
- `Outputs/Uplift/Python/XGBoost/top_benefit_decile_summary.csv`
- `Outputs/Uplift/Python/GLMNet/top_benefit_decile_summary.csv`
- `Outputs/Uplift/Python/XGBoost/shap_importance_treated_control_models.csv`
- `Outputs/Uplift/Python/GLMNet/shap_importance_treated_control_models.csv`
- `Outputs/Uplift/Python/XGBoost/shap_importance_benefit_score.csv`
- `Outputs/Uplift/Python/GLMNet/shap_importance_benefit_score.csv`

## Final Checklist Before Submitting

- [ ] Business question is clearly stated.
- [ ] Data review includes member counts, treatment split, and outcome prevalence.
- [ ] T-learner method is explained in plain language.
- [ ] XGBoost and GLMNet are both evaluated.
- [ ] Treated and control AUC values are reported.
- [ ] Brier and calibration results are interpreted.
- [ ] Uplift decile results are summarized.
- [ ] Top benefit decile is discussed.
- [ ] Observed treated-control gaps are interpreted cautiously.
- [ ] Risk drivers and benefit drivers are separated.
- [ ] ROI assumptions are documented.
- [ ] Limitations are explicit.
- [ ] Recommendations are tied back to operational prioritization.
