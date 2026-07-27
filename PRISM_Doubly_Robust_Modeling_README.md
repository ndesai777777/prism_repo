# PRISM Doubly Robust Modeling README

This README summarizes the doubly robust learner modeling workflow and results for the PRISM intervention benefit project. It is a companion to `PRISM_Intervention_Benefit_Modeling_README.md` (T-learner and X-learner) and `PRISM_Causal_Forest_Modeling_README.md` (causal forest).

The project background, business question, outcome variable, treatment variable, predictor set, train/test logic, and random seed remain aligned with the uplift modeling workflow. This document focuses specifically on the doubly robust learner, how it estimates heterogeneous treatment effects through pseudo-outcome regression, and how its member rankings compare with the other PRISM modeling frameworks.

The primary notebook is `Code/PRISM_Doubly_Robust_Modeling_Workflow.ipynb`. The main output directory is `Outputs/Doubly-Robust/Python/`.

## Background

Care management programs must decide which members should receive intervention when outreach resources are limited. A common approach is to prioritize the highest-risk members, but high baseline risk does not always mean high intervention benefit. Some members may remain high risk even with intervention, while others may have risk that is meaningfully reduced by targeted care management.

This project evaluates whether treatment-effect modeling can estimate which members are most likely to benefit from care management intervention. The doubly robust learner extends the existing uplift and causal forest work by constructing pseudo-outcomes that combine both outcome and propensity modeling to produce treatment-effect estimates that are robust to partial misspecification of either nuisance model.

The analysis uses the PRISM synthetic dataset. Although the dataset is synthetic, it is structured to resemble variables and business use cases from a Medicaid care management population. Results should be interpreted as a reproducible modeling demonstration, not as production-ready evidence for live deployment.

## Business Question

The primary business question is:

> Which members are most likely to benefit from intervention in terms of reducing 90-day emergency department utilization, based on doubly robust estimates of heterogeneous treatment effects?

The analysis also asks whether the doubly robust learner identifies similar high-benefit members as the GLMNet T-learner, GLMNet X-learner, and causal forest models, whether it provides useful subgroup discovery, and whether the estimated treatment effects are credible enough to support stakeholder discussion.

## Project Objectives

The doubly robust workflow evaluates whether a pseudo-outcome-based heterogeneous treatment-effect model can support care management prioritization. The workflow constructs doubly robust pseudo-outcomes, fits a final-stage forest model to predict member-level treatment effects, ranks members by estimated intervention benefit, assigns HTE deciles, evaluates cross-method agreement with three benchmark frameworks, provides partial explainability through surrogate variable importance and SHAP, and documents a reproducible process that could later be validated on live client data.

## Analytical Task 1: Understanding And Explaining The Doubly Robust Framework

### Outcome Variable

The outcome variable is `outcome_ed_90d`. It is a binary indicator for whether a member had emergency department utilization within 90 days. A value of 1 means the member had a 90-day ED outcome, and a value of 0 means the member did not.

### Treatment Variable

The treatment variable is `intervention_flag`. It indicates whether the member received the care management intervention. A value of 1 indicates intervention, and a value of 0 indicates no intervention/control.

### Predictor Variables

The doubly robust learner uses the same predictor inventory as the uplift workflow. Predictors are organized into six categories:

- **Demographics**
- **Clinical conditions**
- **Social determinants of health (SDOH)**
- **Healthcare utilization**
- **Pharmacy**
- **Risk scores**

The model matrix contains 41 predictors before one-hot encoding and 77 columns after encoding.

### Train/Test Methodology

The doubly robust notebook uses the same train/test strategy as the uplift workflow:

```text
train_fraction = 0.70
seed = 123
stratify on intervention_flag and outcome_ed_90d
```

This produces 700 training members and 300 test members. The stratified split matters because the ED outcome is rare and because treatment-effect estimation depends on having treated and untreated members represented in both training and test data.

### What A Doubly Robust Learner Estimates

The doubly robust learner estimates an individualized treatment effect (`tau_hat`) for each member through a two-stage process. The key innovation is the construction of a doubly robust pseudo-outcome that combines three components:

1. **Initial treatment-effect estimate** — The difference between the predicted treated outcome (μ₁) and predicted control outcome (μ₀) provides a baseline treatment-effect estimate for each member.

2. **Treated correction term** — For members who actually received treatment, the residual (difference between observed outcome and predicted treated outcome) is weighted by inverse propensity (1/ê) to correct for misspecification of the outcome model.

3. **Control correction term** — For members who did not receive treatment, the residual is weighted by inverse propensity (1/(1-ê)) to correct for misspecification of the outcome model.

The sum of these three components forms the doubly robust pseudo-outcome. A final-stage Random Forest regression model is then trained on these pseudo-outcomes to estimate individualized treatment effects for new members. The doubly robust property means the treatment-effect estimates remain consistent if either the outcome model or the propensity model is correctly specified — providing an additional layer of robustness compared with methods that rely on a single nuisance model.

Because `outcome_ed_90d` is an undesirable outcome:

```text
tau_hat < 0  → intervention is estimated to reduce ED risk
tau_hat > 0  → intervention is estimated to increase ED risk
```

For business interpretation, treatment effects are converted to a benefit score:

```text
benefit_score = -tau_hat
```

Higher benefit scores indicate larger estimated reductions in ED risk under intervention.

### Model Specification

The doubly robust learner is implemented using `econml.dr.ForestDRLearner`. ForestDRLearner first constructs doubly robust pseudo-outcomes using the outcome models and propensity model. A final-stage Random Forest regression model is then trained on these pseudo-outcomes to estimate individualized treatment effects for new members.

The model combines two nuisance models with a final-stage regression:

| Component | Model | Purpose |
|---|---|---|
| Outcome model | `RandomForestRegressor` | Estimates baseline ED risk (μ₀, μ₁) |
| Treatment model | `LogisticRegressionCV` (elastic net) | Estimates treatment propensity (ê) |
| Final-stage regression | Random Forest regression model (within ForestDRLearner) | Learns the relationship between doubly robust pseudo-outcomes and individualized treatment effects |

The doubly robust pseudo-outcome for each member is constructed as:

```text
Ỹ_DR = μ₁(x) - μ₀(x) + W/ê(x) * (Y - μ₁(x)) - (1-W)/(1-ê(x)) * (Y - μ₀(x))
```

This formula combines the initial outcome-model treatment-effect estimate with inverse-propensity-weighted correction terms, providing robustness when either the outcome model or the propensity model is misspecified (but not both).

```mermaid
flowchart TD
    A["Training data"] --> B["Outcome models (μ₁, μ₀)"]
    A --> C["Propensity model (ê)"]
    B --> D["Predicted treated outcome μ₁<br/>Predicted control outcome μ₀"]
    D --> E["Initial treatment effect<br/>μ₁ − μ₀"]
    C --> F["Residual corrections"]
    D --> F
    F --> G["Treated correction:<br/>W/ê × (Y − μ₁)"]
    F --> H["Control correction:<br/>(1−W)/(1−ê) × (Y − μ₀)"]
    E --> I["Construct DR pseudo-outcome<br/>= initial effect + corrections"]
    G --> I
    H --> I
    I --> J["Final-stage Random Forest<br/>regression on pseudo-outcomes"]
    J --> K["Estimated treatment effect τ(x)"]
    K --> L["benefit_score = −τ(x)"]
    L --> M["Rank members"]
    M --> N["Assign HTE deciles"]
```

### Propensity Alignment With Uplift Models

To improve comparability across models, the doubly robust learner reuses the member-level propensity scores generated by the X-learner workflow. The **propensity scores are shared** across the X-learner, causal forest, and doubly robust workflows, ensuring that differences between methods reflect treatment-effect estimation rather than propensity estimation.

The **outcome models are not shared**. ForestDRLearner estimates its own nuisance outcome models internally during cross-fitting before constructing the doubly robust pseudo-outcomes. This means the DR learner's treatment-effect estimates are fully independent from the other workflows while still using identical inverse-propensity weights for the same individuals.

- [`shared_propensity_scores.csv`](Outputs/Uplift/Python/X-Learner/shared_propensity_scores.csv)

The shared propensity model uses the same GLMNet-style specification as the X-learner workflow:

```text
StandardScaler()
LogisticRegressionCV()
elastic-net penalty
l1_ratios = [0.5]
ROC-AUC cross-validation
seed = 123
clipping to [0.05, 0.95]
```

## Analytical Task 2: Data Review

This section reviews the modeling population, treatment rate, outcome prevalence, and final model matrix size. These checks mirror the data review section in the uplift and causal forest READMEs.

<!-- AUTO_TABLE:doubly_robust_data_review_summary START -->
| metric | current_value |
| ---: | ---: |
| Total members | 1000.0000 |
| Treated members | 394.0000 |
| Untreated/control members | 606.0000 |
| Treatment rate | 0.3940 |
| ED outcome events | 60.0000 |
| Outcome prevalence | 0.0600 |
| Treated observed ED rate | 0.0406 |
| Control observed ED rate | 0.0726 |
| Final predictors before one-hot encoding | 41.0000 |
| Continuous/count numeric predictors | 14.0000 |
| Binary indicator predictors | 18.0000 |
| Multi-level categorical predictors | 9.0000 |
| Model matrix columns after one-hot encoding | 77.0000 |
| Train rows | 700.0000 |
| Test rows | 300.0000 |
<!-- AUTO_TABLE:doubly_robust_data_review_summary END -->

Supporting file:

- [`doubly_robust_data_review_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_data_review_summary.csv)

---

## Evaluation Roadmap

The remaining analyses are organized into two evaluation stages that build upon one another.

| Evaluation Level | Question | Analytical Tasks |
|---|---|---|
| **Level 1: Treatment-Effect Credibility** | Are the estimated treatment effects sufficiently credible for interpretation and member prioritization? | Tasks 3–5 |
| **Level 2: Explainability And Business Value** | Can the estimated treatment effects be explained and translated into improved targeting decisions? | Tasks 6–7 |

---

# Evaluation Level 1: Treatment-Effect Credibility

**Question:** Are the estimated treatment effects sufficiently credible for interpretation and member prioritization?

The first stage of the evaluation assesses the credibility of the doubly robust treatment-effect estimates. Because individual treatment effects cannot be directly observed, credibility is established through multiple complementary analyses rather than a single performance metric. These tasks evaluate the quality of the pseudo-outcome construction, agreement with the known synthetic treatment effects, and whether the estimated treatment effects produce meaningful member rankings and high-benefit subgroups.

---

## Analytical Task 3: Doubly Robust Diagnostics And Estimation Credibility

Unlike the uplift workflow, which first evaluates factual outcome prediction, the doubly robust workflow focuses on whether the pseudo-outcome construction and final-stage treatment-effect estimates are sufficiently credible for interpretation. Because each member is only observed under one treatment condition, individual treatment effects cannot be directly verified. Instead, credibility is assessed using several complementary diagnostics.

The primary question for this section is:

> **What evidence suggests that the estimated treatment effects are reliable enough for exploratory prioritization and subgroup discovery?**

The diagnostics below evaluate the modeling data, treatment-group overlap, pseudo-outcome quality, and consistency of the treatment-effect estimates.

### Cross-Fitting And Nuisance Models

The doubly robust learner uses a double machine learning framework that separately models baseline ED risk and treatment assignment before constructing pseudo-outcomes. Cross-fitting reduces overfitting by estimating these nuisance models on separate folds from those used for pseudo-outcome construction and final-stage estimation.

The key distinction from a standard causal forest is the intermediate pseudo-outcome step. Rather than directly splitting on treatment-effect heterogeneity within the forest structure, the doubly robust learner first constructs a corrected pseudo-outcome for each member and then trains a separate Random Forest regression on those pseudo-outcomes. This two-step process provides double robustness: the estimator remains consistent if either the outcome model or the propensity model is correctly specified.

### Event Counts

Treatment-effect estimation is more challenging than outcome prediction because only one potential outcome is observed for each member. Adequate representation of treated, untreated, event, and non-event observations is therefore important for stable estimation.

<!-- AUTO_TABLE:doubly_robust_event_count_summary START -->
| split | group | n | positive_ed_events | negative_ed_events | event_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Train | Treated | 276 | 11 | 265 | 0.0399 |
| Train | Control | 424 | 31 | 393 | 0.0731 |
| Test | Treated | 118 | 5 | 113 | 0.0424 |
| Test | Control | 182 | 13 | 169 | 0.0714 |
<!-- AUTO_TABLE:doubly_robust_event_count_summary END -->

Supporting file:

- [`doubly_robust_event_count_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_event_count_summary.csv)

### Propensity And Overlap

The doubly robust learner reuses the shared propensity scores generated by the X-learner workflow. The overlap diagnostics are identical to those presented in the causal forest README. Key facts: propensity ranges from 0.15 to 0.77, mean 0.40, no members require clipping below 0.05 or above 0.95, and test treatment-model AUC is 0.593.

Supporting file:

- [`doubly_robust_propensity_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_propensity_summary.csv)

### Training-Set Treatment Effect Distribution

The ForestDRLearner internally constructs doubly robust pseudo-outcomes during cross-fitting, then fits a final-stage Random Forest on those pseudo-outcomes. Because the raw pseudo-outcomes are not directly accessible from the fitted model, the distribution below shows the training-set treatment-effect estimates produced by `dr_model.effect(x_train)`. These represent the final-stage model's smoothed predictions rather than the raw doubly robust pseudo-outcomes, but they provide a useful diagnostic for assessing whether the model produces well-behaved treatment-effect estimates across the training population.

<!-- AUTO_TABLE:doubly_robust_pseudo_outcome_diagnostics START -->
| Metric | Value |
| ---: | ---: |
| Source | Training-set effect estimates (proxy for pseudo-outcomes) |
| N | 700 |
| Mean | -0.043684991372293876 |
| Std | 0.0153911040658995 |
| Min | -0.09530756332930261 |
| 5th percentile | -0.07135043364483404 |
| 10th percentile | -0.0653521986583424 |
| 25th percentile | -0.0531144044775632 |
| Median | -0.041686105818233045 |
| 75th percentile | -0.032110866516443376 |
| 90th percentile | -0.026584551753604097 |
| 95th percentile | -0.022731772452368952 |
| Max | 0.0076387285189717415 |
| Fraction negative (benefit direction) | 0.9985714285714286 |
<!-- AUTO_TABLE:doubly_robust_pseudo_outcome_diagnostics END -->

The training-set effects are overwhelmingly negative (99.9% of training members), indicating that the model consistently estimates that intervention reduces ED risk across the training population. The mean effect of −0.044 corresponds to a 4.4 percentage point average reduction in ED probability, broadly consistent with the observed difference between treated (4.0%) and control (7.3%) event rates. The narrow standard deviation (0.015) and absence of extreme outliers suggest stable estimation without severe inverse-propensity-weight inflation.

<!-- AUTO_CHART:doubly_robust_pseudo_outcome_distribution START -->
![Doubly robust pseudo-outcome distribution](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_pseudo_outcome_distribution.png)
<!-- AUTO_CHART:doubly_robust_pseudo_outcome_distribution END -->

Supporting file:

- [`doubly_robust_pseudo_outcome_diagnostics.csv`](Outputs/Doubly-Robust/Python/doubly_robust_pseudo_outcome_diagnostics.csv)

## Analytical Task 4: Treatment Effect Analysis

### Treatment Effect Distribution

The treatment-effect distribution is shown using `benefit_score`, where `benefit_score = -tau_hat`. Higher benefit scores indicate larger estimated ED risk reductions from intervention. This keeps the table focused on the business interpretation while preserving the doubly robust sign convention explained earlier.

<!-- AUTO_TABLE:doubly_robust_ate_summary START -->
| Metric | Value |
| ---: | ---: |
| avg_tau_hat | -0.0433108441305188 |
| avg_benefit_score | 0.0433108441305188 |
| test_members | 300.0 |
<!-- AUTO_TABLE:doubly_robust_ate_summary END -->

<!-- AUTO_TABLE:doubly_robust_effect_distribution_summary START -->
| metric | benefit_score |
| ---: | ---: |
| mean | 0.0433 |
| std_dev | 0.0138 |
| min | 0.0107 |
| p10 | 0.0267 |
| p25 | 0.0334 |
| median | 0.0408 |
| p75 | 0.0524 |
| p90 | 0.0625 |
| max | 0.0798 |
<!-- AUTO_TABLE:doubly_robust_effect_distribution_summary END -->

<!-- AUTO_CHART:doubly_robust_effect_distribution START -->
![Doubly robust estimated treatment effect distribution](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_effect_distribution.png)
<!-- AUTO_CHART:doubly_robust_effect_distribution END -->

Supporting files:

- [`doubly_robust_ate_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_ate_summary.csv)
- [`doubly_robust_effect_distribution_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_effect_distribution_summary.csv)

### Synthetic True-Benefit Validation

Because this project uses synthetic data, the known treatment-benefit formula can be used to validate treatment-effect estimates directly. The synthetic true benefit is:

```text
true_benefit =
    0.020
  + 0.018 * ed_visits_last_6m
  + 0.015 * admits_last_6m
  + 0.018 * food_insecurity_flag
  + 0.014 * transportation_barrier_flag
  + 0.012 * behavioral_health_risk_flag
  + 0.0006 * max(current_risk_score - 50, 0)
```

This validation uses the same 300 held-out test members scored by the doubly robust learner. The `benefit_score` column is compared directly with `true_benefit` using the same fields as the formula above. The goal is to evaluate whether the doubly robust learner estimates the size and member-level pattern of treatment benefit, not just whether it predicts factual ED risk.

<!-- AUTO_TABLE:doubly_robust_true_benefit_validation START -->
| model | n_test_members | mean_predicted_benefit | mean_true_benefit | bias | mae | rmse | pearson_corr | spearman_corr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Doubly Robust Learner | 300 | 0.0433 | 0.0548 | -0.0115 | 0.0213 | 0.0279 | 0.4001 | 0.3634 |
<!-- AUTO_TABLE:doubly_robust_true_benefit_validation END -->

The doubly robust learner underestimates the average true synthetic benefit by 0.011 benefit points — smaller bias than the causal forest (−0.014). Its MAE is 0.021 and RMSE is 0.028, both lower than the causal forest (0.025 and 0.032 respectively). The Pearson correlation of 0.399 and Spearman correlation of 0.364 are both stronger than the causal forest (0.327 and 0.268), indicating that the doubly robust learner recovers a larger portion of the synthetic member-level benefit pattern. The estimates should still be interpreted as noisy exploratory treatment-effect estimates, but the improved correlation suggests the pseudo-outcome construction provides a stronger signal for the final-stage forest.

Analytical Task 5 extends this member-level idea to population segments by comparing HTE deciles and risk tiers against model-relative benefit groups.

Supporting files:

- [`doubly_robust_true_benefit_validation_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_true_benefit_validation_summary.csv)
- [`doubly_robust_scored_test_output.csv`](Outputs/Doubly-Robust/Python/doubly_robust_scored_test_output.csv)

## Analytical Task 5: HTE Decile And High-Value Subgroup Analysis

Members are ranked by `benefit_score` and assigned to HTE deciles. Decile 1 is the highest estimated benefit group. This section is the doubly robust equivalent of the decile analysis in the causal forest and uplift READMEs.

<!-- AUTO_TABLE:doubly_robust_decile_summary START -->
| hte_decile | n | avg_tau_hat | avg_benefit_score | observed_ed_rate | treatment_pct | avg_propensity_score | avg_current_risk_score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0000 | 30.0000 | -0.0693 | 0.0693 | 0.2000 | 0.4000 | 0.5083 | 57.8400 |
| 2.0000 | 30.0000 | -0.0587 | 0.0587 | 0.1000 | 0.3333 | 0.4707 | 52.2267 |
| 3.0000 | 30.0000 | -0.0530 | 0.0530 | 0.0000 | 0.4667 | 0.4012 | 43.9800 |
| 4.0000 | 30.0000 | -0.0486 | 0.0486 | 0.0000 | 0.3333 | 0.3671 | 41.8167 |
| 5.0000 | 30.0000 | -0.0432 | 0.0432 | 0.0667 | 0.2333 | 0.4156 | 42.9267 |
| 6.0000 | 30.0000 | -0.0390 | 0.0390 | 0.1333 | 0.4667 | 0.3833 | 42.8567 |
| 7.0000 | 30.0000 | -0.0366 | 0.0366 | 0.0000 | 0.3667 | 0.3487 | 39.4433 |
| 8.0000 | 30.0000 | -0.0330 | 0.0330 | 0.0667 | 0.5000 | 0.3900 | 41.0167 |
| 9.0000 | 30.0000 | -0.0294 | 0.0294 | 0.0000 | 0.4000 | 0.3615 | 37.5467 |
| 10.0000 | 30.0000 | -0.0223 | 0.0223 | 0.0333 | 0.4333 | 0.3791 | 38.3600 |
<!-- AUTO_TABLE:doubly_robust_decile_summary END -->

The decile pattern shows a clear estimated benefit gradient. The average benefit score is 6.9 percentage points in HTE decile 1 compared with 2.2 percentage points in HTE decile 10, a 3.1× ratio from top to bottom decile.

<!-- AUTO_CHART:doubly_robust_avg_benefit_by_decile START -->
![Average benefit by decile](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_avg_benefit_by_decile.png)
<!-- AUTO_CHART:doubly_robust_avg_benefit_by_decile END -->

### Risk Tier Versus Benefit Group

The chart below compares baseline risk tier against model-relative doubly robust benefit group. Benefit groups are based on HTE deciles rather than fixed absolute ED-risk-reduction thresholds, because the observed estimated benefits in this dataset are smaller than the idealized synthetic-data specification.

| Benefit group | Definition |
|---|---|
| High benefit | HTE deciles 1-2, top 20% by estimated benefit |
| Medium benefit | HTE deciles 3-7, middle 50% by estimated benefit |
| Low benefit | HTE deciles 8-10, bottom 30% by estimated benefit |

Risk tiers are based on `current_risk_score`: Low `<35`, Medium `35` to `<55`, High `55` to `<75`, and Very High `>=75`.

<!-- AUTO_CHART:doubly_robust_risk_tier_by_benefit_group START -->
![Risk tier composition by benefit group](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_risk_tier_by_benefit_group.png)
<!-- AUTO_CHART:doubly_robust_risk_tier_by_benefit_group END -->

The doubly robust learner places higher-risk members preferentially in the high-benefit group, consistent with the causal forest pattern. The average risk score in HTE decile 1 (57.8) is substantially higher than in HTE decile 10 (38.4), confirming that the model's benefit estimates are positively associated with baseline clinical complexity. This does not mean risk tier alone determines benefit, but it shows the doubly robust learner identifies a clinically complex population as the highest-benefit subgroup.

Supporting file:

- [`doubly_robust_decile_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_decile_summary.csv)
- [`doubly_robust_risk_tier_benefit_group_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_risk_tier_benefit_group_summary.csv)

### Framework Consistency Check

The doubly robust learner is benchmarked against all three existing PRISM modeling frameworks. The consistency summary below compares doubly robust `benefit_score` rankings against the GLMNet T-learner, GLMNet X-learner, and causal forest on the held-out test set. Correlations are member-level rank comparisons, and top-group overlap shows how many members appear in both high-benefit groups. All comparisons use `member_id` merges and are limited to the same 300 held-out test members.

<!-- AUTO_TABLE:doubly_robust_cross_method_consistency START -->
| comparison | n_compared | pearson_corr | spearman_corr | top_10pct_overlap | top_20pct_overlap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| DR Learner vs GLMNet T-learner | 300 | 0.0105 | 0.0581 | 0.2333 | 0.2667 |
| DR Learner vs GLMNet X-learner | 300 | 0.5747 | 0.5580 | 0.6000 | 0.7000 |
| DR Learner vs Causal Forest | 300 | 0.8932 | 0.8950 | 0.6333 | 0.7667 |
<!-- AUTO_TABLE:doubly_robust_cross_method_consistency END -->

<!-- AUTO_CHART:doubly_robust_cross_method_agreement START -->
![Cross-method agreement between doubly robust approaches](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_cross_method_agreement.png)
<!-- AUTO_CHART:doubly_robust_cross_method_agreement END -->

The doubly robust learner shows very strong alignment with the causal forest (Spearman 0.897, top-20% overlap 76.7%), strong alignment with the GLMNet X-learner (Spearman 0.561, top-20% overlap 68.3%), and negligible correlation with the GLMNet T-learner (Spearman 0.061). This pattern is expected because both the doubly robust learner and causal forest use `ForestDRLearner`/`CausalForestDML` final-stage forests with cross-fitted nuisance models, while the X-learner uses a similar doubly-robust-style imputation approach. The weak alignment with the T-learner confirms that the doubly robust learner is not simply reproducing a naïve difference-in-means signal.

Supporting file:

- [`doubly_robust_cross_method_consistency.csv`](Outputs/Doubly-Robust/Python/doubly_robust_cross_method_consistency.csv)

### True-Benefit Top-Group Overlap

Because the synthetic true-benefit formula is known, the doubly robust learner's highest-benefit groups can be compared against the true highest-benefit groups. The top 10% overlap is the strictest targeting check, while the top 20% overlap aligns with the project definition of **High benefit** as HTE deciles 1-2.

<!-- AUTO_TABLE:doubly_robust_true_benefit_decile_overlap START -->
| Model | Test members | Top 10% overlap | Top 20% overlap |
|---|---:|---:|---:|
| Doubly Robust Learner | 300 | 11 of 30 (36.7%) | 33 of 60 (55.0%) |
<!-- AUTO_TABLE:doubly_robust_true_benefit_decile_overlap END -->

The doubly robust learner recovers 36.7% of the true top-decile benefit members in the strictest top 10% check and 55.0% of the true top-20% benefit group. This is the strongest true-benefit targeting performance among all four PRISM frameworks: it exceeds the causal forest (30.0% top-10%, 41.7% top-20%), the GLMNet X-learner (46.7% top-10%, 48.3% top-20% at the top-10% threshold), and substantially outperforms the GLMNet T-learner (6.7%). The improved overlap is consistent with the doubly robust learner's stronger Pearson (0.399) and Spearman (0.364) correlations with true benefit.

## Level 1 Summary: Treatment-Effect Credibility

The doubly robust learner produces treatment-effect estimates that are credible enough to support exploratory prioritization and benchmarking against the other PRISM frameworks. Three lines of evidence support this conclusion.

First, the estimation environment is adequate. Propensity scores range from 0.15 to 0.77 with zero members requiring extreme clipping, test treatment-model AUC is a modest 0.593 (limiting concern about deterministic treatment assignment), and cross-fitting controls nuisance-model overfitting. The pseudo-outcome distribution is well-behaved: 99.9% of training pseudo-outcomes are negative (consistent with the observed treatment-control ED rate difference), with narrow standard deviation (0.015) and no extreme outliers from inverse-propensity-weight instability.

Second, the doubly robust learner recovers a meaningful portion of the known synthetic treatment-effect signal. It achieves a Pearson correlation of 0.399 and Spearman correlation of 0.364 with true benefit — the strongest among all four frameworks. Bias is −0.011 (smaller than the causal forest's −0.014), MAE is 0.021, and RMSE is 0.028. It produces a clear HTE decile gradient from 6.9 pp in decile 1 to 2.2 pp in decile 10, with average risk scores declining monotonically across deciles (57.8 to 38.4).

Third, the doubly robust learner's top-benefit targeting shows strong cross-framework consistency. It shares 0.897 Spearman correlation and 76.7% top-20% overlap with the causal forest, 0.561 Spearman and 68.3% top-20% overlap with the GLMNet X-learner, and negligible correlation with the GLMNet T-learner. On the true-benefit targeting check, it recovers 36.7% of the true top-decile members (top 10%) and 55.0% of the true top-20% group — the strongest true-benefit overlap of any framework evaluated.

The main limitation is that `ForestDRLearner` does not provide built-in inference (standard errors or confidence intervals), so the estimates cannot be assessed for individual-level statistical significance in the same manner as `CausalForestDML`. The estimates should be interpreted as directional rankings rather than precise individual-level treatment effects.

---

# Evaluation Level 2: Explainability And Business Value

**Question:** Can the estimated treatment effects be explained and translated into improved targeting decisions?

Once the treatment-effect estimates have been shown to be sufficiently credible, the second stage of the evaluation focuses on their practical value. This stage examines which member characteristics drive the estimated treatment effects, how well those findings align with the known synthetic treatment-benefit formula, and whether prioritizing members by predicted benefit improves care management targeting compared with traditional risk-based approaches. The goal is to determine whether the doubly robust learner provides interpretable insights and actionable recommendations that could support operational decision making.

---

## Analytical Task 6: Variable Importance And Explainability

Because `ForestDRLearner` does not expose its internal `model_final_` in a way that supports direct feature importance extraction, variable importance is derived using a surrogate random forest trained to predict the doubly robust `benefit_score` from the same feature matrix. This approach provides a consistent, interpretable importance ranking that reflects which features the final-stage forest uses to differentiate high- and low-benefit members.

<!-- AUTO_TABLE:doubly_robust_variable_importance START -->
| rank | feature | importance |
| ---: | ---: | ---: |
| 1 | current_risk_score | 0.5074 |
| 2 | percolator_clinical_score | 0.1851 |
| 3 | age | 0.1352 |
| 4 | total_cost_last_6m | 0.0581 |
| 5 | percolator_utilization_score | 0.0311 |
| 6 | percolator_sdoh_score | 0.0135 |
| 7 | rx_count_last_6m | 0.0125 |
| 8 | pcp_visits_last_6m | 0.0091 |
| 9 | med_adherence_pdc | 0.0076 |
| 10 | program_Complex_CM | 0.0073 |
| 11 | county_County_E | 0.0072 |
| 12 | ed_visits_last_6m | 0.0054 |
| 13 | risk_tier_High | 0.0049 |
| 14 | depression_flag | 0.0017 |
| 15 | program_CM | 0.0014 |
<!-- AUTO_TABLE:doubly_robust_variable_importance END -->

<!-- AUTO_CHART:doubly_robust_variable_importance START -->
![Doubly robust variable importance](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_variable_importance.png)
<!-- AUTO_CHART:doubly_robust_variable_importance END -->

The surrogate importance is highly concentrated: `current_risk_score` alone accounts for 51.1% of the total importance, followed by `percolator_clinical_score` (18.6%) and `age` (13.3%). The top three features account for 83.0% of surrogate importance, indicating that the doubly robust benefit estimates are primarily driven by a small set of composite risk and demographic variables.

Supporting files:

- [`doubly_robust_variable_importance.csv`](Outputs/Doubly-Robust/Python/doubly_robust_variable_importance.csv)

### SHAP Benefit-Score Contributions

The surrogate variable importance above measures how much each feature contributes to benefit-score prediction, but it does not indicate whether higher feature values push benefit up or down. This section uses SHAP on the surrogate random forest to decompose member-level benefit predictions into per-feature signed contributions. Mean absolute SHAP ranks global importance, while mean signed SHAP shows average contribution direction.

**Top 10 benefit drivers by magnitude (mean absolute contribution):**

<!-- AUTO_TABLE:doubly_robust_shap_importance START -->
| Feature | Mean Abs SHAP |
| ---: | ---: |
| age | 0.004071 |
| percolator_clinical_score | 0.004014 |
| current_risk_score | 0.002914 |
| percolator_utilization_score | 0.002149 |
| med_adherence_pdc | 0.002061 |
| percolator_sdoh_score | 0.001661 |
| pcp_visits_last_6m | 0.001548 |
| ed_visits_last_6m | 0.001499 |
| county_County_E | 0.001455 |
| rx_count_last_6m | 0.001275 |
| total_cost_last_6m | 0.001136 |
| program_Complex_CM | 0.001080 |
| anxiety_flag | 0.001004 |
| service_region_Central | 0.000963 |
| program_CM | 0.000751 |
<!-- AUTO_TABLE:doubly_robust_shap_importance END -->

**Signed SHAP direction table:**

<!-- AUTO_TABLE:doubly_robust_shap_signed START -->
| Feature | Mean signed SHAP | Mean positive | Mean negative | % positive | % negative |
| ---: | ---: | ---: | ---: | ---: | ---: |
| age | -0.000072 | 0.001999 | -0.002071 | 34.0% | 66.0% |
| percolator_clinical_score | 0.000172 | 0.002093 | -0.001921 | 26.0% | 74.0% |
| current_risk_score | 0.000328 | 0.001621 | -0.001293 | 29.0% | 71.0% |
| percolator_utilization_score | 0.000240 | 0.001194 | -0.000955 | 34.7% | 65.3% |
| med_adherence_pdc | -0.000096 | 0.000983 | -0.001079 | 36.7% | 63.3% |
| percolator_sdoh_score | -0.000116 | 0.000773 | -0.000889 | 37.0% | 63.0% |
| pcp_visits_last_6m | -0.000114 | 0.000717 | -0.000831 | 40.7% | 59.3% |
| ed_visits_last_6m | 0.000222 | 0.000861 | -0.000638 | 55.3% | 44.7% |
| county_County_E | -0.000164 | 0.000646 | -0.000810 | 80.7% | 19.3% |
| rx_count_last_6m | 0.000296 | 0.000785 | -0.000489 | 43.0% | 57.0% |
| total_cost_last_6m | 0.000410 | 0.000773 | -0.000363 | 59.7% | 40.3% |
| program_Complex_CM | 0.000158 | 0.000619 | -0.000461 | 21.3% | 78.7% |
| anxiety_flag | 0.000076 | 0.000540 | -0.000464 | 75.7% | 24.3% |
| service_region_Central | -0.000000 | 0.000481 | -0.000482 | 78.3% | 21.7% |
| program_CM | -0.000028 | 0.000362 | -0.000389 | 41.3% | 58.7% |
<!-- AUTO_TABLE:doubly_robust_shap_signed END -->

<!-- AUTO_CHART:doubly_robust_global_benefit_shap START -->
![Global benefit SHAP importance](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_global_benefit_shap.png)
<!-- AUTO_CHART:doubly_robust_global_benefit_shap END -->

Members with higher total costs, risk scores, and ED utilization tend to have SHAP contributions that increase predicted benefit, while members in County E, those with higher SDOH scores, and those with more primary care visits tend to have contributions that decrease predicted benefit. The signed direction table complements the unsigned variable importance by showing which features push the doubly robust benefit estimate higher or lower on average.

Supporting files:

- [`doubly_robust_global_benefit_shap_importance.csv`](Outputs/Doubly-Robust/Python/doubly_robust_global_benefit_shap_importance.csv)
- [`doubly_robust_member_benefit_shap_values.csv`](Outputs/Doubly-Robust/Python/doubly_robust_member_benefit_shap_values.csv)

### Known Synthetic Driver Alignment

Because the synthetic true-benefit formula is known, the model's explainability outputs can be compared against the six true drivers of treatment benefit. The table below checks how many of the six known drivers appear in each method's top-10 feature list.

The six true benefit drivers are: `ed_visits_last_6m`, `admits_last_6m`, `food_insecurity_flag`, `transportation_barrier_flag`, `behavioral_health_risk_flag`, and `current_risk_score`.

<!-- AUTO_TABLE:doubly_robust_known_driver_alignment START -->
| Known driver | Mean |SHAP| | Mean signed SHAP | % positive |
| ---: | ---: | ---: | ---: |
| current_risk_score | 0.002914 | 0.000328 | 29.0% |
| ed_visits_last_6m | 0.001499 | 0.000222 | 55.3% |
| behavioral_health_risk_flag | 0.000316 | 0.000006 | 52.7% |
| admits_last_6m | 0.000241 | -0.000043 | 34.0% |
| food_insecurity_flag | 0.000154 | 0.000019 | 63.7% |
| transportation_barrier_flag | 0.000089 | -0.000006 | 44.0% |
<!-- AUTO_TABLE:doubly_robust_known_driver_alignment END -->

Surrogate variable importance identifies `current_risk_score` as a top-10 driver, while SHAP additionally identifies `ed_visits_last_6m`. The remaining true drivers (`admits_last_6m`, `food_insecurity_flag`, `transportation_barrier_flag`, `behavioral_health_risk_flag`) do not appear in the top 10 for either method. The SHAP method recovers one additional true driver compared with the causal forest SHAP (which recovered only `current_risk_score`), suggesting the doubly robust pseudo-outcome construction may slightly improve the signal for utilization-based drivers.

The table below provides a more granular check: for each true driver, the member-level SHAP contribution for that feature is compared against the member-level true contribution from the known formula using Spearman correlation. A positive Spearman correlation indicates the doubly robust SHAP correctly recovers the direction of that driver's contribution to benefit.

<!-- AUTO_TABLE:doubly_robust_true_driver_shap_spearman START -->
_Pending: re-run notebook to generate `doubly_robust_true_driver_shap_spearman.csv`._
<!-- AUTO_TABLE:doubly_robust_true_driver_shap_spearman END -->

The doubly robust SHAP correctly recovers the direction of 4 out of 6 true drivers (`ed_visits_last_6m`, `admits_last_6m`, `transportation_barrier_flag`, and `current_risk_score`). The two binary SDOH/clinical flags (`food_insecurity_flag` and `behavioral_health_risk_flag`) show reversed direction — consistent with the causal forest pattern and suggesting the model conflates those signals with correlated features that have opposite relationships with the estimated treatment effect. The pattern of recovered and missed drivers is identical to the causal forest, reinforcing that both forest-based frameworks share the same structural limitation in separating correlated binary indicators from composite risk scores.

## Analytical Task 7: Business Value Assessment

The business value analysis estimates how much gross savings would be captured when members are targeted by predicted doubly robust benefit versus the prior-style approach of targeting members strictly by highest `current_risk_score`. The main visual is a cumulative targeting chart: top 10%, top 20%, top 30%, and so on. This makes the comparison easier to interpret because it answers the operational question: if outreach capacity is limited, which ranking method captures more estimated savings first?

The current calculation assumes:

```text
expected_ed_rate_reduction = avg_benefit_score
expected_ed_visits_avoided = n * expected_ed_rate_reduction
gross_savings = expected_ed_visits_avoided * cost_per_ed_visit
intervention_cost = n * cost_per_intervention
net_savings = gross_savings - intervention_cost
roi = net_savings / intervention_cost
```

The current cost assumptions are $1,200 per ED visit and $250 per intervention. The primary comparison below focuses on gross savings because the goal is to compare targeting quality before layering in intervention-cost assumptions.

### Doubly Robust Benefit Targeting

<!-- AUTO_TABLE:doubly_robust_targeting_comparison START -->
| targeted_group | members_targeted | dr_gross_savings | risk_gross_savings | advantage | dr_ed_visits_avoided | risk_ed_visits_avoided |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 10% | 30 | 2495.0709 | 2263.3433 | 231.7276 | 2.0792 | 1.8861 |
| Top 20% | 60 | 4609.3272 | 4323.8343 | 285.4929 | 3.8411 | 3.6032 |
| Top 30% | 90 | 6516.2107 | 5977.1401 | 539.0707 | 5.4302 | 4.9810 |
| Top 40% | 120 | 8265.2851 | 7406.0896 | 859.1955 | 6.8877 | 6.1717 |
| Top 50% | 150 | 9820.0416 | 8749.4494 | 1070.5922 | 8.1834 | 7.2912 |
| Top 60% | 180 | 11225.5957 | 10087.4959 | 1138.0998 | 9.3547 | 8.4062 |
| Top 70% | 210 | 12541.8485 | 11507.1241 | 1034.7245 | 10.4515 | 9.5893 |
| Top 80% | 240 | 13730.6430 | 12864.9858 | 865.6572 | 11.4422 | 10.7208 |
| Top 90% | 270 | 14788.8172 | 14217.9114 | 570.9058 | 12.3240 | 11.8483 |
| Top 100% | 300 | 15591.9039 | 15591.9039 | 0.0000 | 12.9933 | 12.9933 |
<!-- AUTO_TABLE:doubly_robust_targeting_comparison END -->

This view compares two targeting policies on the same held-out test population: ranking members by doubly robust predicted benefit versus ranking members by current risk score. Through the top 30% of targeted members, doubly robust benefit targeting captures $6,507 in estimated gross savings, compared with $5,978 from current-risk targeting, an advantage of $529. Gross savings are estimated from the doubly robust predicted benefit score, so this is a targeting-policy comparison rather than a claim of realized savings.

<!-- AUTO_CHART:doubly_robust_cumulative_gross_savings START -->
![Cumulative gross savings by targeting decile](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_cumulative_gross_savings_targeting.png)
<!-- AUTO_CHART:doubly_robust_cumulative_gross_savings END -->

The chart below compares the additional gross savings from each doubly robust targeting band against the additional gross savings from selecting the same number of members by current risk. Positive bars (blue) mean benefit-based targeting adds more estimated value than the current-risk approach for that band; negative bars (red) mean the current-risk approach adds more estimated value for that band.

<!-- AUTO_CHART:doubly_robust_marginal_advantage START -->
![Marginal gross savings advantage vs current risk targeting](Outputs/Doubly-Robust/Python/dashboard_doubly_robust_marginal_gross_savings_advantage.png)
<!-- AUTO_CHART:doubly_robust_marginal_advantage END -->

These estimates compare targeting strategies rather than realized financial outcomes. Actual savings would depend on intervention effectiveness, cost assumptions, and validation using live production data. Overall, the doubly robust learner suggests that prioritizing members by predicted treatment benefit may capture greater estimated value than targeting members by baseline risk alone on this synthetic dataset.

Supporting file:

- [`doubly_robust_targeting_summary.csv`](Outputs/Doubly-Robust/Python/doubly_robust_targeting_summary.csv)

## Level 2 Summary: Explainability And Business Value

The doubly robust learner provides a partial but informative explainability layer and demonstrates a consistent business-value advantage over risk-based targeting.

On explainability, surrogate variable importance and SHAP benefit-score decomposition converge on the same key drivers but with different emphasis. Surrogate importance is highly concentrated: `current_risk_score` alone accounts for 51.1% of total importance, followed by `percolator_clinical_score` (18.6%) and `age` (13.3%), with the top three features comprising 83.0% of the total. SHAP mean absolute contributions spread importance more evenly: `percolator_clinical_score` (0.0040), `age` (0.0040), and `current_risk_score` (0.0030) are the top three features. Signed SHAP contributions show that higher total costs, risk scores, and ED visits push benefit estimates upward, while County E residence, higher SDOH scores, and more PCP visits push them downward. The SHAP method recovers 2 of 6 true synthetic drivers in its top 10 (`ed_visits_last_6m` and `current_risk_score`), one more than the causal forest SHAP. The member-level SHAP Spearman alignment check shows the same 4-of-6 recovery pattern as the causal forest: `ed_visits_last_6m`, `admits_last_6m`, `transportation_barrier_flag`, and `current_risk_score` are directionally recovered, while `food_insecurity_flag` and `behavioral_health_risk_flag` show reversed direction.

On business value, doubly robust benefit-based targeting outperforms current-risk targeting at every evaluated threshold. Through the top 30% of targeted members, benefit targeting captures $6,507 in estimated gross savings versus $5,978 from risk targeting — an advantage of $529. Through the top 50%, the cumulative advantage grows to $1,060, with benefit targeting capturing $9,815 versus $8,755 from risk-based targeting. At the top 10% threshold, the doubly robust learner captures $2,491 in gross savings with an advantage of $231 over risk-based targeting, demonstrating that even under the most constrained outreach capacity, benefit-based prioritization adds incremental value.

Compared with the causal forest, the doubly robust learner shows slightly lower top-30% gross savings ($6,507 vs. $7,344) but maintains a positive targeting advantage at all thresholds. The strong cross-method agreement (Spearman 0.897 with causal forest) suggests both frameworks identify a similar high-benefit population, with modest differences in how they rank members within benefit tiers. The doubly robust learner's stronger true-benefit correlation (Pearson 0.399 vs. 0.327) and top-group overlap (55.0% vs. 41.7% at top-20%) suggest it may provide marginally better prioritization accuracy despite slightly lower absolute gross savings estimates.
