"""Generate the Funds Combined causal-forest report from executed outputs."""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Outputs" / "Causal-Forests_Funds" / "Python"
TARGET = ROOT / "PRISM_Causal_Forest_Modeling_Funds_README.md"
REL = "Outputs/Causal-Forests_Funds/Python"


def read(name: str) -> pd.DataFrame:
    path = OUTPUT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def scalar(frame: pd.DataFrame, key: str, key_col: str = "metric", value_col: str = "value"):
    match = frame.loc[frame[key_col].astype(str).eq(key), value_col]
    if match.empty:
        raise KeyError(f"{key!r} not found in {key_col}")
    return match.iloc[0]


def number(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.{digits}f}"


def pct(value, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.{digits}f}%"


def money(value, digits: int = 0) -> str:
    if pd.isna(value):
        return "NA"
    sign = "-" if float(value) < 0 else ""
    return f"{sign}${abs(float(value)):,.{digits}f}"


def md_table(headers: list[str], rows: list[list[object]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["left"] * len(headers)
    markers = {"left": ":---", "right": "---:", "center": ":---:"}
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(markers[a] for a in aligns) + " |",
    ]
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


data_review = read("causal_forest_data_review_summary.csv")
audit = read("causal_forest_preprocessing_audit_summary.csv").iloc[0]
events = read("causal_forest_event_count_summary.csv")
propensity = read("causal_forest_propensity_summary.csv")
tuning = read("causal_forest_hyperparameter_tuning_summary.csv")
ate = read("causal_forest_ate_summary.csv")
effects = read("causal_forest_effect_distribution_summary.csv")
uncertainty = read("causal_forest_uncertainty_summary.csv")
deciles = read("causal_forest_decile_summary.csv")
gaps = read("causal_forest_observed_gap_by_decile.csv")
consistency = read("causal_forest_vs_uplift_consistency_summary.csv")
risk = read("causal_forest_risk_tier_benefit_group_summary.csv")
importance = read("causal_forest_variable_importance.csv")
shap = read("causal_forest_global_benefit_shap_importance.csv")
profile = read("causal_forest_top_decile_profile.csv")
targeting = read("causal_forest_targeting_summary.csv")
cumulative = read("causal_forest_cumulative_gross_savings_by_targeting.csv")
marginal = read("causal_forest_marginal_gross_savings_advantage_vs_current_risk.csv")
output_index = read("causal_forest_output_index.csv")

review_lookup = dict(zip(data_review["metric"], data_review["current_value"]))
selected = tuning.loc[tuning["selected"].astype(str).str.lower().isin(["true", "1"])].iloc[0]
top = deciles.loc[deciles["hte_decile"].eq(1)].iloc[0]
bottom = deciles.loc[deciles["hte_decile"].eq(10)].iloc[0]
top_gap = gaps.loc[gaps["hte_decile"].eq(1)].iloc[0]

event_table = md_table(
    ["Split", "Group", "N", "ED events", "No ED event", "Event rate"],
    [[r["split"], r["group"], number(r["n"]), number(r["positive_ed_events"]),
      number(r["negative_ed_events"]), pct(r["event_rate"])] for _, r in events.iterrows()],
    ["left", "left", "right", "right", "right", "right"],
)

tuning_table = md_table(
    ["Candidate", "Trees", "Min leaf", "Max depth", "Top-bottom gap", "Avg Spearman", "Top-decile Jaccard", "Selection score", "Selected"],
    [[number(r["candidate_id"]), number(r["n_estimators"]), number(r["min_samples_leaf"]),
      "None" if pd.isna(r["max_depth"]) else number(r["max_depth"]),
      number(r["validation_top_bottom_benefit_gap"], 4), number(r["avg_spearman_vs_other_candidates"], 3),
      number(r["avg_top_decile_jaccard_vs_other_candidates"], 3), number(r["selection_score"], 1),
      "Yes" if str(r["selected"]).lower() in ["true", "1"] else "No"] for _, r in tuning.iterrows()],
    ["right", "right", "right", "right", "right", "right", "right", "right", "center"],
)

decile_table = md_table(
    ["HTE decile", "N", "Avg benefit", "Avg tau", "Avg SE", "Observed ED rate", "Treated", "Avg propensity", "Avg current risk"],
    [[number(r["hte_decile"]), number(r["n"]), pct(r["avg_benefit_score"], 2), pct(r["avg_tau_hat"], 2),
      pct(r["avg_tau_se"], 2), pct(r["observed_ed_rate"]), pct(r["treatment_pct"]),
      number(r["avg_propensity_score"], 3), number(r["avg_current_risk_score"], 2)] for _, r in deciles.iterrows()],
    ["right"] * 9,
)

gap_table = md_table(
    ["HTE decile", "Treated N", "Control N", "Predicted benefit", "Treated ED rate", "Control ED rate", "Observed gap"],
    [[number(r["hte_decile"]), number(r["treated_n"]), number(r["control_n"]), pct(r["avg_predicted_benefit"], 2),
      pct(r["treated_observed_ed_rate"]), pct(r["control_observed_ed_rate"]),
      pct(r["observed_control_minus_treated_gap"], 2)] for _, r in gaps.iterrows()],
    ["right"] * 7,
)

consistency_table = md_table(
    ["Benchmark", "Role", "N", "Pearson", "Spearman", "Top-decile overlap", "Top-20% overlap", "Alignment"],
    [[r["comparison"].replace("Causal forest vs ", ""), r["benchmark_role"], number(r["n_compared"]),
      number(r["pearson_corr"], 3), number(r["spearman_corr"], 3), pct(r["top_decile_overlap"]),
      pct(r["top_20pct_overlap"]), r["comparison_basis"].replace("_", " ")] for _, r in consistency.iterrows()],
    ["left", "left", "right", "right", "right", "right", "right", "left"],
)

risk_pivot = risk.pivot(index="risk_tier", columns="benefit_group", values="pct_within_risk_tier").fillna(0)
risk_counts = risk.groupby("risk_tier", as_index=True)["risk_tier_members"].max()
risk_table = md_table(
    ["Original risk tier", "Members", "High benefit", "Medium benefit", "Low benefit"],
    [[number(tier), number(risk_counts.loc[tier]), pct(risk_pivot.loc[tier, "High benefit"]),
      pct(risk_pivot.loc[tier, "Medium benefit"]), pct(risk_pivot.loc[tier, "Low benefit"])]
     for tier in risk_pivot.index],
    ["right"] * 5,
)

importance_table = md_table(
    ["Rank", "Encoded feature", "Forest importance"],
    [[number(r["rank"]), r["feature"], number(r["importance"], 4)] for _, r in importance.head(15).iterrows()],
    ["right", "left", "right"],
)

shap_table = md_table(
    ["Rank", "Encoded feature", "Mean |SHAP|", "Mean signed SHAP", "% positive", "Explained rows"],
    [[i + 1, r["feature"], number(r["mean_abs_benefit_shap"], 5), number(r["mean_signed_benefit_shap"], 5),
      pct(r["pct_positive_benefit_shap"]), number(r["explained_test_rows"])] for i, (_, r) in enumerate(shap.head(15).iterrows())],
    ["right", "left", "right", "right", "right", "right"],
)

profile_table = md_table(
    ["Feature", "Top-decile mean/rate", "Other-decile mean/rate", "Difference"],
    [[r["feature"], number(r["top_hte_decile_mean_or_rate"], 3),
      number(r["other_deciles_mean_or_rate"], 3), number(r["difference"], 3)] for _, r in profile.iterrows()],
    ["left", "right", "right", "right"],
)

targeting_table = md_table(
    ["HTE decile", "N", "Avg benefit", "Expected ED visits avoided", "Gross savings", "Intervention cost", "Net savings", "ROI"],
    [[number(r["hte_decile"]), number(r["n"]), pct(r["avg_benefit_score"], 2),
      number(r["expected_ed_visits_avoided"], 2), money(r["gross_savings"]), money(r["intervention_cost"]),
      money(r["net_savings"]), pct(r["roi"])] for _, r in targeting.iterrows()],
    ["right"] * 8,
)

top50 = cumulative.loc[cumulative["through_decile"].eq(5)].copy()
top50_table = md_table(
    ["Targeting approach", "Members", "Population", "Expected ED visits avoided", "Gross savings"],
    [[r["targeting_approach"], number(r["n"]), pct(r["population_fraction_targeted"]),
      number(r["cumulative_estimated_ed_visits_avoided"], 2), money(r["cumulative_gross_savings"])]
     for _, r in top50.iterrows()],
    ["left", "right", "right", "right", "right"],
)

prop_train_auc = float(scalar(propensity, "Train treatment model AUC"))
prop_test_auc = float(scalar(propensity, "Test treatment model AUC"))
prop_lower = int(float(scalar(propensity, "Members at lower clip (0.05)")))
prop_upper = int(float(scalar(propensity, "Members at upper clip (0.95)")))
mean_benefit = float(scalar(ate, "avg_benefit_score"))
mean_tau = float(scalar(ate, "avg_tau_hat"))
ci_below = int(float(scalar(uncertainty, "Members with tau CI entirely below zero")))
ci_cross = int(float(scalar(uncertainty, "Members with tau CI crossing zero")))
ci_above = int(float(scalar(uncertainty, "Members with tau CI entirely above zero")))

xgb_t = consistency[(consistency["benchmark_model"] == "XGBoost") & (consistency["benchmark_framework"] == "T-Learner")].iloc[0]
xgb_x = consistency[(consistency["benchmark_model"] == "XGBoost") & (consistency["benchmark_framework"] == "X-Learner")].iloc[0]

cf_top50 = top50[top50["targeting_approach"] == "Causal forest benefit score"].iloc[0]
risk_top50 = top50[top50["targeting_approach"] == "Current risk score"].iloc[0]
top50_advantage = cf_top50["cumulative_gross_savings"] - risk_top50["cumulative_gross_savings"]

report = f"""# PRISM Causal Forest Modeling — Funds Combined Report

This report summarizes the completed causal-forest workflow for the Funds Combined population. It follows the analytical order and evaluation structure of `PRISM_Causal_Forest_Modeling_README.md`, while applying the Funds-specific preprocessing established in `Code/Uplift Model Code_Funds_Combined.ipynb`.

The primary notebook is `Code/PRISM_Causal_Forest_Modeling_Funds_Workflow.ipynb`. The output directory is `{REL}`.

## Background

Care-management targeting should distinguish baseline risk from impactability. A high-risk member may remain high risk even after intervention, while a lower-risk member may be more responsive. The causal forest estimates heterogeneous treatment effects (HTE): how the modeled effect of intervention on 90-day ED utilization varies across baseline member characteristics.

Funds Combined is observational and does not contain known individual counterfactual outcomes. The workflow therefore evaluates treatment overlap, uncertainty, stability across model specifications, held-out HTE separation, consistency with existing Funds uplift models, and operational targeting implications. None of these diagnostics alone proves member-level causality.

## Business Question

Which Funds Combined members are most likely to benefit from intervention through a reduction in 90-day emergency-department utilization, and does causal-forest ranking add useful information beyond baseline risk and the existing Funds uplift models?

## Project Objectives

- Reproduce the Funds treatment and outcome derivations exactly.
- Restrict modeling to the same leakage-safe baseline predictor allowlist.
- Estimate member-level heterogeneous treatment effects.
- Quantify overlap, uncertainty, ranking stability, and held-out separation.
- Compare causal-forest rankings with Funds XGBoost and fixed GLMNet T- and X-learners.
- Produce decision-ready decile, risk-tier, explainability, and savings outputs.

## Analytical Task 1: Understanding And Explaining The Causal Forest Framework

### Outcome Variable

`outcome_ed_90d` is converted to a binary ED-event indicator. Missing values are filled with zero before binarization, zero means no ED visit, and any positive count becomes one. This retained {number(audit['missing_outcome_filled_with_zero'])} rows whose outcome was originally missing.

### Treatment Variable

Treatment is derived jointly from normalized source `intervention_flag` and `OptOut_flag`:

- Treated: source intervention equals 1 and opt-out does not equal 1.
- Control: opt-out equals 1 and source intervention does not equal 1, including missing source intervention values.
- Contradictory: both signals equal 1.
- Unresolved: neither signal establishes treatment.

The current data contained {number(audit['contradictory_treatment_rows'])} contradictory and {number(audit['unresolved_treatment_rows'])} unresolved rows. Both raw source flags remain in audits and scored outputs but are excluded from all predictors.

### Predictor Variables

The final model uses {number(audit['retained_predictor_count'])} baseline predictors and {number(audit['encoded_feature_count'])} encoded columns. It retains demographics, coverage, clinical and behavioral-health flags, social-need indicators, baseline utilization, cost, medication measures, risk scores, original Funds risk tier, and the derived death indicator.

Post-treatment program and engagement fields, alternate outcomes, raw treatment sources, `member_id`, raw `date_of_death`, and `total_cost_6MO` are excluded. `total_cost_last_6m` remains because it is a distinct baseline measure. The zero-variance fields removed were `{audit['zero_variance_predictors_removed']}`.

Detailed inventory: [`causal_forest_predictor_inventory.csv`]({REL}/causal_forest_predictor_inventory.csv)

### Train/Test Methodology

The workflow uses a 70/30 split with seed 123, stratified jointly by derived treatment and binary outcome. It produces {number(audit['train_rows'])} training rows and {number(audit['test_rows'])} test rows. Model selection occurs within an additional 80/20 split of the training data; the final test set is not used to choose hyperparameters.

Source `member_id` is not unique ({number(audit['duplicate_member_id_observations_retained'])} duplicate observations). It is preserved as requested, while a unique `analysis_row_id` provides unambiguous row-level scoring and framework alignment.

### What A Causal Forest Estimates

For member covariates `X`, treatment `T`, and binary outcome `Y`, the model estimates:

```text
tau_hat(X) = E[Y | do(T=1), X] - E[Y | do(T=0), X]
benefit_score(X) = -tau_hat(X)
```

Negative `tau_hat` implies lower modeled ED risk under intervention. Positive `benefit_score` therefore represents estimated ED-risk reduction.

### Model Specification

- Estimator: `econml.dml.CausalForestDML`
- Outcome nuisance model: random-forest regression, 300 trees, minimum leaf 10
- Treatment nuisance model: fixed Funds elastic-net logistic specification (`l1_ratio=0.5`, `C=1.0`)
- Cross-fitting: five stratified folds
- Candidate causal forests: 400–800 trees, minimum leaf 5–20, unlimited or depth-8 trees
- Selection: 50% validation separation rank, 30% average rank-stability rank, 20% top-decile stability rank
- Final selected forest: {number(selected['n_estimators'])} trees, minimum leaf {number(selected['min_samples_leaf'])}, max depth {('None' if pd.isna(selected['max_depth']) else number(selected['max_depth']))}

Thread count was set to one for execution compatibility in the Windows sandbox; estimator definitions, hyperparameters, folds, and seed were unchanged.

### Propensity Alignment With Funds Uplift Models

The propensity model matches the fixed Funds uplift specification rather than the cross-validated PRP GLMNet propensity. The existing shared propensity file cannot safely be joined by `member_id` because the Funds source intentionally contains duplicates; the causal workflow therefore refits the identical fixed specification on the identical training split.

## Analytical Task 2: Data Review

{md_table(
    ['Metric', 'Value'],
    [[metric, number(value, 4)] for metric, value in review_lookup.items()],
    ['left', 'right'],
)}

Key review findings:

- All {number(audit['modeling_rows'])} resolved rows were retained.
- {number(audit['outcome_positive_rows'])} members/rows had at least one ED event ({pct(audit['outcome_positive_rows'] / audit['modeling_rows'])}).
- Treatment prevalence was {pct(audit['treated_rows'] / audit['modeling_rows'])}.
- {number(audit['death_flag_rows_retained'])} death-flag rows remained in the analysis; raw death dates were not modeled.
- Original Funds risk tiers 1–5 were preserved; missing or invalid values were left unassigned rather than reconstructed from current risk score.
- Executable leakage checks passed against both raw predictor names and encoded matrix columns.

Primary files:

- [`causal_forest_data_review_summary.csv`]({REL}/causal_forest_data_review_summary.csv)
- [`causal_forest_preprocessing_audit_summary.csv`]({REL}/causal_forest_preprocessing_audit_summary.csv)
- [`causal_forest_preprocessing_column_audit.csv`]({REL}/causal_forest_preprocessing_column_audit.csv)

## Evaluation Roadmap

The report separates two questions:

1. **Treatment-effect credibility:** data support, overlap, uncertainty, model stability, held-out separation, and cross-framework consistency.
2. **Explainability and business value:** drivers, subgroup profiles, risk-tier segmentation, and the economics of benefit-based targeting.

# Evaluation Level 1: Treatment-Effect Credibility

## Analytical Task 3: Causal Forest Diagnostics And Estimation Credibility

### Cross-Fitting And Nuisance Models

Five-fold cross-fitting estimates outcome and treatment nuisance functions out of fold before final HTE fitting. This reduces reuse bias relative to fitting nuisance components and treatment effects on the same predictions, but it does not remove all risk of residual confounding in observational data.

### Event Counts

{event_table}

Both outcome classes appear within treatment group in both train and test partitions, supporting the requested cross-fitting and held-out diagnostics.

### Propensity And Overlap Checks

The treatment model AUC was {prop_train_auc:.3f} in training and {prop_test_auc:.3f} on test data. On the held-out set, {number(prop_lower)} members were clipped to 0.05 and {number(prop_upper)} were clipped to 0.95. These tails indicate some limited-overlap observations, but most scores remain inside the clipping boundaries.

![Funds propensity score overlap]({REL}/dashboard_propensity_overlap.png)

Detailed file: [`causal_forest_propensity_summary.csv`]({REL}/causal_forest_propensity_summary.csv)

### Uncertainty Assessment

The held-out mean tau standard error was {number(scalar(uncertainty, 'Mean tau standard error'), 4)}. Of {number(len(read('causal_forest_scored_test_output.csv')))} test rows:

- {number(ci_below)} had a 95% tau interval entirely below zero.
- {number(ci_cross)} had an interval crossing zero.
- {number(ci_above)} had an interval entirely above zero.

The large crossing-zero count ({pct(ci_cross / len(read('causal_forest_scored_test_output.csv')))} of test rows) is the main caution against treating individual scores as certain causal facts. Ranking and subgroup summaries are more defensible uses than deterministic member-level claims.

Detailed file: [`causal_forest_uncertainty_summary.csv`]({REL}/causal_forest_uncertainty_summary.csv)

### Model Selection

{tuning_table}

Candidate 2 was selected because it provided the best combined stability/separation score. Candidate 1 produced the largest validation top-bottom gap, but its lower cross-candidate rank and top-decile stability made it less attractive for operational targeting.

## Analytical Task 4: Treatment Effect Analysis

### Treatment Effect Distribution

The mean held-out `tau_hat` was {pct(mean_tau, 2)}, corresponding to mean predicted benefit of {pct(mean_benefit, 2)}. The median benefit was {pct(float(effects.loc[effects['metric'].eq('median'), 'benefit_score'].iloc[0]), 2)}; the 10th–90th percentile range was {pct(float(effects.loc[effects['metric'].eq('p10'), 'benefit_score'].iloc[0]), 2)} to {pct(float(effects.loc[effects['metric'].eq('p90'), 'benefit_score'].iloc[0]), 2)}.

![Funds causal forest effect distribution]({REL}/dashboard_causal_forest_effect_distribution.png)

Primary files:

- [`causal_forest_ate_summary.csv`]({REL}/causal_forest_ate_summary.csv)
- [`causal_forest_effect_distribution_summary.csv`]({REL}/causal_forest_effect_distribution_summary.csv)

### Synthetic True-Benefit Validation

**Not applicable.** Funds Combined does not include known counterfactual treatment effects. No synthetic formula was copied from the PRP demonstration data and no ground truth was fabricated. The explicit status is recorded in [`causal_forest_true_benefit_validation_summary.csv`]({REL}/causal_forest_true_benefit_validation_summary.csv).

## Analytical Task 5: HTE Decile And High-Value Subgroup Analysis

{decile_table}

The average predicted benefit declines monotonically from {pct(top['avg_benefit_score'], 2)} in decile 1 to {pct(bottom['avg_benefit_score'], 2)} in decile 10. Deciles 8–10 have negative average predicted benefit, meaning the model does not support broad intervention of those groups under this specification.

![Average Funds causal-forest benefit by decile]({REL}/dashboard_causal_forest_avg_benefit_by_decile.png)

### Observed Treatment/Outcome Gap By HTE Decile

{gap_table}

The top decile had an observed control-minus-treated ED-rate gap of {pct(top_gap['observed_control_minus_treated_gap'], 2)} versus average predicted benefit of {pct(top_gap['avg_predicted_benefit'], 2)}. This strong directional pattern is useful corroboration, but it is an unadjusted factual contrast and must not be interpreted as counterfactual validation.

![Predicted benefit and observed gap by decile]({REL}/dashboard_causal_forest_observed_gap_by_decile.png)

### Risk Tier Versus Benefit Group

{risk_table}

Benefit is not a relabeling of original risk tier. For example, {pct(risk_pivot.loc[1, 'High benefit'])} of tier-1 observations and {pct(risk_pivot.loc[5, 'High benefit'])} of tier-5 observations fall in the causal-forest high-benefit tercile. The relationship is non-monotonic and should be interpreted alongside overlap and uncertainty.

![Risk tier composition by causal-forest benefit group]({REL}/dashboard_causal_forest_risk_tier_by_benefit_group.png)

### Framework Consistency Check

{consistency_table}

The causal forest shows moderate agreement with the stronger XGBoost uplift models: Spearman correlations are {number(xgb_t['spearman_corr'], 3)} against the T-learner and {number(xgb_x['spearman_corr'], 3)} against the X-learner, with top-decile overlaps of {pct(xgb_t['top_decile_overlap'])} and {pct(xgb_x['top_decile_overlap'])}. Agreement with the fixed GLMNet variants is weak, matching the Funds uplift report's conclusion that those models have weaker factual outcome performance.

Because `member_id` is duplicated, comparisons were aligned by validated source-row order or reproduced test-row order—not by a false one-to-one member-ID merge.

### True-Benefit Top-Group Overlap

**Not applicable.** No known true-benefit ranking exists for Funds Combined.

## Level 1 Summary: Treatment-Effect Credibility

The causal forest produces clear held-out rank separation and moderate agreement with both XGBoost uplift frameworks. Treatment groups have adequate event counts, and most propensity scores are not at the clipping limits. However, {pct(ci_cross / len(read('causal_forest_scored_test_output.csv')))} of individual treatment-effect intervals cross zero. The model is therefore best positioned as a challenger, ranking, and subgroup-discovery tool—not as proof that a specific member will benefit.

# Evaluation Level 2: Explainability And Business Value

## Analytical Task 6: Variable Importance And Explainability

### Forest Split Importance

{importance_table}

![Causal forest variable importance]({REL}/dashboard_causal_forest_variable_importance.png)

Split importance identifies features the forest uses to partition treatment-effect heterogeneity; it does not indicate direction and is not a causal attribution.

### SHAP Benefit-Score Contributions

Permutation SHAP was computed on a deterministic 100-row held-out sample with a 50-row training background. This bounds computation for the 371-column encoded design while leaving all fitting and scoring on the full prescribed samples.

{shap_table}

![Global benefit SHAP importance]({REL}/dashboard_causal_forest_global_benefit_shap.png)

SHAP describes how features move this model's benefit prediction relative to its background expectation. It explains model behavior, not intervention mechanisms or clinical causation.

### Top-Decile Profile

{profile_table}

The top decile has materially higher average baseline cost, percolator score, current risk score, and housing-instability prevalence than other deciles, but lower average prior six-month ED visits and admissions. That mixed pattern illustrates why impactability is not reducible to simple high-risk targeting.

### Known Synthetic Driver Alignment

**Not applicable.** Funds Combined has no synthetic data-generating mechanism or known true treatment-effect drivers.

## Analytical Task 7: Business Value Assessment

The business-value scenario uses the same assumptions as the Funds uplift work: $1,200 per avoided ED visit and $250 intervention cost per targeted member. Values are model-based scenario estimates, not booked savings.

### Causal Forest Benefit Targeting

{targeting_table}

At current assumptions, every HTE decile has negative net savings because even the top-decile gross savings of {money(top['avg_benefit_score'] * top['n'] * 1200)} are below its {money(top['n'] * 250)} intervention cost. The top-decile ROI is {pct(targeting.loc[targeting['hte_decile'].eq(1), 'roi'].iloc[0])}. This does not mean ranking is useless; it means the current cost/benefit assumptions do not support a positive financial return at full per-member intervention cost.

### Comparison With Current-Risk Targeting

{top50_table}

At approximately 50% of the test population, causal-forest targeting yields {money(cf_top50['cumulative_gross_savings'])} in modeled gross savings versus {money(risk_top50['cumulative_gross_savings'])} for current-risk targeting, a modeled advantage of {money(top50_advantage)} before intervention costs.

![Cumulative gross savings by targeting approach]({REL}/dashboard_cumulative_gross_savings_targeting.png)

The marginal advantage is positive through the fifth targeting decile and turns negative thereafter, consistent with limiting outreach to higher-benefit groups rather than expanding indiscriminately.

![Marginal gross savings advantage versus current risk]({REL}/dashboard_marginal_gross_savings_advantage_vs_current_risk.png)

## Level 2 Summary: Explainability And Business Value

The model's main benefit drivers are baseline severity, cost, utilization, medication, and selected social/clinical factors. Causal-forest targeting ranks modeled benefit more efficiently than current-risk targeting in the upper half of the population, but the assumed $250 intervention cost exceeds modeled gross savings even in the top decile. Operational use should therefore focus on lower-cost interventions, narrower targeting, or prospective validation of larger realized effects.

## Client Perspective And Recommendation

Use the Funds causal forest as a **challenger and subgroup-discovery model**. It provides useful rank separation and moderate consistency with the stronger XGBoost uplift workflows, but individual uncertainty remains high and observational data cannot establish unmeasured-confounding robustness.

Recommended next steps:

1. Validate top-decile and top-tercile performance prospectively or in a randomized rollout.
2. Investigate limited-overlap cases and consider overlap trimming or weighting as a sensitivity analysis.
3. Calibrate intervention-cost scenarios and identify lower-cost outreach modalities.
4. Monitor subgroup stability, fairness, and drift before using scores in production.
5. Keep original risk tier and predicted benefit as separate operational dimensions.

## Reproducibility And Output Index

- Notebook: [`Code/PRISM_Causal_Forest_Modeling_Funds_Workflow.ipynb`](Code/PRISM_Causal_Forest_Modeling_Funds_Workflow.ipynb)
- Input: `DataSets/Funds_combined.csv`
- Output root: [`{REL}`]({REL})
- Seed: `123`
- Executed environment: Python 3.13, econml 0.16.0
- Expected artifacts: {number(len(output_index))}
- Successfully generated artifacts: {number(output_index['exists'].sum())}
- Missing artifacts: {number((~output_index['exists'].astype(bool)).sum())}
- Leakage assertions: passed
- Synthetic true-benefit artifacts: explicit `not_applicable` status only; no fabricated ground truth

Complete machine-readable index: [`causal_forest_output_index.csv`]({REL}/causal_forest_output_index.csv)

## Interpretation Guardrails

- `benefit_score` is an estimate, not an observed individual counterfactual.
- Unadjusted treated/control gaps are descriptive corroboration, not causal validation.
- SHAP and split importance explain the fitted model, not biological or program mechanisms.
- Financial values are assumption-based scenarios and exclude broader care costs and benefits.
- Duplicate `member_id` observations are retained by design; use `analysis_row_id` for row-level reconciliation.
"""

TARGET.write_text(report, encoding="utf-8")
print(f"Wrote {TARGET} ({len(report):,} characters).")
