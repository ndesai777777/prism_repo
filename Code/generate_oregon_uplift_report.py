"""Generate the Oregon uplift-model Markdown report from executed notebook outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "Outputs" / "Uplift_Oregon" / "Python"
REPORT_PATH = ROOT / "PRISM_Intervention_Benefit_Modeling_Oregon_README.md"


def load(relative_path: str) -> pd.DataFrame:
    path = OUTPUT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    return pd.read_csv(path)


def number(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def percent(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{100 * value:.{digits}f}%"


def dollars(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "—"
    return f"${value:,.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered.append("| " + " | ".join(str(value).replace("\n", " ").replace("|", "\\|") for value in row) + " |")
    return "\n".join(rendered)


def image(relative_path: str, alt_text: str) -> str:
    path = OUTPUT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Report image is missing: {path}")
    report_relative = path.relative_to(ROOT).as_posix()
    return f"![{alt_text}]({report_relative})"


def main() -> None:
    review = load("data_review_summary.csv").iloc[0]
    audit = load("preprocessing_audit_summary.csv").iloc[0]
    event_counts = load("factual_event_count_summary.csv")
    evaluation = load("model_evaluation_summary.csv")
    recommendation = load("model_recommendation_summary.csv")
    prediction_ranges = load("factual_prediction_ranges.csv")
    separation = load("factual_prediction_separation.csv")
    risk_tiers = load("risk_tier_population_summary.csv")
    column_audit = load("preprocessing_column_audit.csv")
    parity = load("output_parity_manifest.csv")

    xgb_t_deciles = load("T-Learner/XGBoost/uplift_observed_gap_by_decile.csv")
    glm_t_deciles = load("T-Learner/GLMNet/uplift_observed_gap_by_decile.csv")
    xgb_t_top = load("T-Learner/XGBoost/top_benefit_decile_summary.csv").iloc[0]
    glm_t_top = load("T-Learner/GLMNet/top_benefit_decile_summary.csv").iloc[0]
    xgb_t_roi = load("T-Learner/XGBoost/uplift_roi_by_decile.csv")
    xgb_t_shap = load("T-Learner/XGBoost/shap_importance_benefit_score.csv")
    glm_t_shap = load("T-Learner/GLMNet/shap_importance_benefit_score.csv")

    consistency = load("X-Learner/xlearner_vs_tlearner_consistency_summary.csv")
    xgb_x_deciles = load("X-Learner/XGBoost/uplift_observed_gap_by_decile.csv")
    xgb_x_roi = load("X-Learner/XGBoost/xlearner_roi_by_decile.csv")
    xgb_x_drivers = load("X-Learner/XGBoost/xlearner_benefit_driver_importance.csv")
    xgb_x_savings = load("X-Learner/XGBoost/cumulative_gross_savings_summary_top50.csv")

    post_treatment = column_audit[column_audit["role"].eq("post-treatment field")][
        ["original_oregon_column", "status", "reason"]
    ]
    excluded_other = column_audit[
        column_audit["status"].eq("excluded")
        & ~column_audit["role"].isin(["post-treatment field", "identifier/grouping key", "treatment", "outcome"])
    ][["original_oregon_column", "role", "reason"]]

    evaluation_rows = []
    for _, row in evaluation.iterrows():
        evaluation_rows.append(
            [
                row["model"],
                number(row["treated_cv_auc"]),
                number(row["control_cv_auc"]),
                number(row["treated_test_auc"]),
                number(row["control_test_auc"]),
                number(row["treated_brier_score"]),
                number(row["control_brier_score"]),
                number(row["treated_calibration_error"]),
                number(row["control_calibration_error"]),
            ]
        )

    event_rows = [
        [
            row["split"],
            row["group"],
            int(row["n"]),
            int(row["positive_ed_events"]),
            int(row["negative_ed_events"]),
            percent(row["event_rate"]),
        ]
        for _, row in event_counts.iterrows()
    ]

    separation_rows = [
        [
            row["model"],
            row["group"],
            number(row["auc"]),
            number(row["avg_pred_actual_positive"]),
            number(row["avg_pred_actual_negative"]),
            number(row["avg_pred_positive_minus_negative"]),
        ]
        for _, row in separation.iterrows()
    ]

    range_rows = [
        [
            row["model"],
            row["group"],
            number(row["min_pred"]),
            number(row["p10_pred"]),
            number(row["median_pred"]),
            number(row["mean_pred"]),
            number(row["p90_pred"]),
            number(row["max_pred"]),
        ]
        for _, row in prediction_ranges.iterrows()
    ]

    risk_rows = [
        [
            int(row["risk_tier"]),
            row["risk_tier_label"],
            int(row["members"]),
            percent(row["pct_population"]),
            number(row["min_current_risk_score"]),
            number(row["max_current_risk_score"]),
            number(row["avg_current_risk_score"]),
        ]
        for _, row in risk_tiers.iterrows()
    ]

    def decile_rows(frame: pd.DataFrame) -> list[list[object]]:
        return [
            [
                int(row["uplift_decile"]),
                int(row["n"]),
                number(row["avg_predicted_benefit"]),
                number(row["observed_control_minus_treated_gap"]),
                f"{number(row['observed_gap_ci_lower_95'])} to {number(row['observed_gap_ci_upper_95'])}",
                int(row["treated_n"]),
                int(row["control_n"]),
                percent(row["treated_pct"]),
            ]
            for _, row in frame.iterrows()
        ]

    consistency_rows = [
        [
            row["model"],
            number(row["pearson_benefit_score_corr"]),
            number(row["spearman_benefit_score_corr"]),
            percent(row["top_decile_overlap_pct"]),
            number(row["t_learner_mean_benefit_score"]),
            number(row["x_learner_mean_benefit_score"]),
        ]
        for _, row in consistency.iterrows()
    ]

    def driver_rows(frame: pd.DataFrame, limit: int = 15) -> list[list[object]]:
        absolute_column = (
            "mean_abs_benefit_shap"
            if "mean_abs_benefit_shap" in frame.columns
            else "mean_abs_benefit_contribution"
        )
        signed_column = (
            "mean_signed_benefit_shap"
            if "mean_signed_benefit_shap" in frame.columns
            else "mean_signed_benefit_contribution"
        )
        positive_column = (
            "pct_positive_benefit_shap"
            if "pct_positive_benefit_shap" in frame.columns
            else "pct_positive_benefit_contribution"
        )
        return [
            [
                rank,
                row["feature"],
                number(row[absolute_column]),
                number(row[signed_column]),
                percent(row[positive_column]),
            ]
            for rank, (_, row) in enumerate(frame.head(limit).iterrows(), start=1)
        ]

    roi_rows = [
        [
            int(row["uplift_decile"]),
            int(row["n"]),
            number(row["expected_ed_rate_reduction"]),
            number(row["expected_ed_visits_avoided"], 2),
            dollars(row["gross_savings"]),
            dollars(row["intervention_cost"]),
            dollars(row["net_savings"]),
            percent(row["roi"]),
        ]
        for _, row in xgb_t_roi.iterrows()
    ]

    x_roi_rows = [
        [
            int(row["uplift_decile"]),
            int(row["n"]),
            number(row["expected_ed_rate_reduction"]),
            number(row["expected_ed_visits_avoided"], 2),
            dollars(row["gross_savings"]),
            dollars(row["intervention_cost"]),
            dollars(row["net_savings"]),
            percent(row["roi"]),
        ]
        for _, row in xgb_x_roi.iterrows()
    ]

    recommendation_rows = [
        [
            row["model"],
            row["risk_prediction_strength"],
            row["probability_calibration"],
            row["benefit_ranking_quality"],
        ]
        for _, row in recommendation.iterrows()
    ]

    parity_counts = parity.groupby(["applicability_status", "generated_status"], dropna=False).size().to_dict()
    missing_applicable = parity[
        parity["applicability_status"].eq("applicable") & ~parity["generated_status"].astype(bool)
    ]["corresponding_prp_relative_path"].tolist()

    content = f"""# PRISM Intervention Benefit Modeling — Oregon 2-Year Dataset

## Background

This report reproduces the analytical workflow, model families, validation sequence, charts, and output structure of the Funds Combined PRISM intervention-benefit notebook using `DataSets/Oregon_2YearDataset.csv`. The executed analysis is in `Code/Uplift Model Code_Oregon_2YearDataset.ipynb`; all generated artifacts are isolated under `Outputs/Uplift_Oregon/`.

The outcome is a binary indicator of any emergency-department use within 90 days. The treatment is PRISM engagement. A positive benefit score means the model predicts a lower 90-day ED-event probability under engagement than under no engagement.

> This is an observational, model-based prioritization analysis. The individual benefit scores are not directly observed causal effects. Small treatment/control counts within some test-set deciles produce wide confidence intervals, so operational use should follow prospective validation.

> The committed notebook is configured to require a SageMaker CUDA GPU on device 0. The checked-in numerical outputs in this report are the completed local CPU validation snapshot produced before enabling the GPU requirement. A full SageMaker rerun will replace those outputs; regenerate this report afterward if fitted results change.

## Business Question

Which Oregon members are most likely to benefit from PRISM engagement, measured as an estimated reduction in the probability of any ED utilization within 90 days?

## Project Objectives

1. Reproduce the Funds Combined methodology with Oregon-specific preprocessing.
2. Prevent leakage by excluding treatment indicators, outcomes, and all known post-intervention fields from the predictor matrix.
3. Compare XGBoost and regularized logistic-regression (GLMNet-equivalent) T-Learners.
4. Add X-Learners using the same propensity-score and evaluation framework as the source notebook.
5. Evaluate outcome-model discrimination and calibration, uplift ranking, uncertainty, explainability, and illustrative business value.

## Analytical Task 1: Modeling Framework

### Outcome

`outcome_ed_90d` was numerically coerced, missing values were assigned to zero as requested, and all values greater than zero were binarized to one. The source count distribution was preserved in the preprocessing audit. No source values were overwritten.

### Treatment

`Engaged_flag = 1` defines treated records and `Engaged_flag = 0` defines controls. `OptOut_flag` was used only to validate the treatment definition and was then excluded. The two fields were exact inverses for all {int(audit['modeling_rows']):,} modeling records; no contradictory or unresolved treatment rows were found.

### Predictors and leakage controls

Only baseline variables were eligible as predictors. The following post-treatment fields were detected and excluded before preprocessing:

{markdown_table(['Source field', 'Status', 'Reason'], [[r['original_oregon_column'], r['status'], r['reason']] for _, r in post_treatment.iterrows()])}

Additional exclusions were:

{markdown_table(['Source field', 'Role', 'Reason'], [[r['original_oregon_column'], r['role'], r['reason']] for _, r in excluded_other.iterrows()])}

The member identifier was retained only for grouped splitting and result traceability. The treatment and outcome were never predictors. No death-related variable was present in this Oregon file; no death flag was created, and no record was removed based on death.

### Preprocessing and split design

- All {int(audit['raw_rows']):,} source records were retained, including {int(audit['repeated_member_episodes_retained']):,} repeated member episodes. There were {int(audit['exact_duplicate_rows']):,} exact duplicate rows.
- The 70/30 split was grouped by `member_id`, producing zero member overlap between train and test.
- Numeric imputation medians, categorical levels, unseen-category handling, and zero-variance filtering were learned from training data only.
- Risk tiers 1–5 were retained. Missing and out-of-range source values were mapped to an explicit `Missing` level.
- `total_cost_last_6m` was converted from its source currency-like representation to numeric before train-only median imputation.
- Three training-set zero-variance fields—`client_contract`, `plan_type`, and `program`—were removed.
- The final model used {int(audit['retained_predictor_count'])} baseline predictors and {int(audit['encoded_feature_count'])} encoded features.
- Explicit pre-model and final encoded-feature leakage assertions passed.

### Workflow

The notebook keeps the Funds Combined sequence and choices: descriptive review, predictor distributions, group-aware holdout, treatment-specific outcome models, T-Learner benefit scores, GLMNet comparison, overlap-weighted sensitivity analysis, X-Learner construction, decile diagnostics, SHAP-style explanation, and ROI targeting summaries. Hyperparameter grids, cross-validation structure, random seeds, and evaluation methods were retained. The committed notebook now matches the Funds GPU requirement: XGBoost uses `tree_method='hist'` with `device='cuda:0'`, and explicit assertions verify that fitted boosters used CUDA.

The Funds source notebook does not define separate Qini or cumulative-gain calculations. Its cumulative uplift-by-targeted-fraction curve is reproduced as written; no additional metric was invented because doing so would change the authoritative source method.

## Analytical Task 2: Data Review

{markdown_table(
    ['Measure', 'Value'],
    [
        ['Analytical records', f"{int(review['total_analytical_records']):,}"],
        ['Unique members', f"{int(review['unique_members']):,}"],
        ['Treated records', f"{int(review['treated_records']):,} ({percent(review['treatment_rate'])})"],
        ['Control records', f"{int(review['control_records']):,} ({percent(1 - review['treatment_rate'])})"],
        ['90-day ED events', f"{int(review['outcome_events']):,} ({percent(review['outcome_prevalence'])})"],
        ['Treated outcome rate', percent(review['treated_outcome_rate'])],
        ['Control outcome rate', percent(review['control_outcome_rate'])],
        ['Training records / unique members', f"{int(audit['train_rows']):,} / {int(audit['train_unique_members']):,}"],
        ['Test records / unique members', f"{int(audit['test_rows']):,} / {int(audit['test_unique_members']):,}"],
        ['Member overlap across train/test', str(int(audit['train_test_member_overlap']))],
    ],
)}

The lower observed event rate among engaged records is descriptive and unadjusted. It should not be interpreted as the intervention effect because engagement was not randomized and baseline risk may differ between groups.

### Risk Tier Definition

The source risk tier was used as a baseline categorical predictor rather than recalculated from the current risk score. Among the {int(risk_tiers['members'].sum()):,} records with valid tiers, the distribution was:

{markdown_table(['Tier', 'Label', 'Records', 'Share of valid tiers', 'Min score', 'Max score', 'Average score'], risk_rows)}

There were {int(audit['missing_source_risk_tier_rows']):,} missing risk-tier values and {int(audit['out_of_range_source_risk_tier_rows']):,} out-of-range values; both were assigned to the explicit `Missing` category for modeling.

## Evaluation Roadmap

The model is evaluated at three levels:

1. **Outcome-model validity:** can each treatment-specific model distinguish records with and without a 90-day ED event, and are its probabilities calibrated?
2. **Uplift-model validity:** does the estimated benefit ranking show useful separation, stable results across learner frameworks, and reasonable agreement with observed within-decile gaps?
3. **Operational value:** which variables drive the scores, and what do illustrative targeting economics look like under fixed cost assumptions?

## Evaluation Level 1: Outcome Model Validation

### Analytical Task 3: Model Performance

#### Event counts

{markdown_table(['Split', 'Group', 'Records', 'ED events', 'No ED event', 'Event rate'], event_rows)}

The control test subgroup contains only {int(event_counts[(event_counts['split'] == 'Test') & (event_counts['group'] == 'Control')]['n'].iloc[0])} records and {int(event_counts[(event_counts['split'] == 'Test') & (event_counts['group'] == 'Control')]['positive_ed_events'].iloc[0])} events. This limits the precision of both outcome-model and uplift validation.

#### Discrimination, Brier score, and calibration

{markdown_table(
    ['Model', 'Treated CV AUC', 'Control CV AUC', 'Treated test AUC', 'Control test AUC', 'Treated Brier', 'Control Brier', 'Treated calibration error', 'Control calibration error'],
    evaluation_rows,
)}

The XGBoost models achieved treated/control test AUCs of {number(evaluation.loc[evaluation['model'].eq('XGBoost'), 'treated_test_auc'].iloc[0])} and {number(evaluation.loc[evaluation['model'].eq('XGBoost'), 'control_test_auc'].iloc[0])}. GLMNet was similar in the treated subgroup and stronger in the control subgroup, although the control estimate is based on the smaller test sample. Neither method had uniformly strong calibration, so raw probabilities and derived benefit magnitudes should be treated cautiously.

{image('T-Learner/XGBoost/dashboard_calibration_plot.png', 'XGBoost calibration by treatment group')}

#### Prediction separation

{markdown_table(['Model', 'Group', 'AUC', 'Mean prediction: event', 'Mean prediction: no event', 'Mean separation'], separation_rows)}

#### Prediction ranges

{markdown_table(['Model', 'Group', 'Minimum', '10th percentile', 'Median', 'Mean', '90th percentile', 'Maximum'], range_rows)}

### Level 1 Summary

{markdown_table(['Model', 'Risk prediction', 'Calibration', 'Benefit-ranking diagnostic'], recommendation_rows)}

Both model families have useful risk discrimination, but probability calibration and small subgroup sizes remain material limitations. Benefit ranking should therefore be validated with ranking-based and uncertainty-aware diagnostics rather than judged from AUC alone.

## Evaluation Level 2: Uplift Model Validation

### Analytical Task 4: Treatment-Effect Estimation

The T-Learner estimates two potential-outcome risks for each test record: predicted ED risk under engagement and predicted ED risk under no engagement. Benefit is `predicted control risk − predicted treated risk`; larger positive values represent greater predicted reduction in ED-event probability.

The source data do not contain observed individual counterfactuals, so true-benefit correlations, true top-decile overlap, and PEHE cannot be computed. Instead, the report uses observed treated-versus-control gaps within ranked deciles, Newcombe-Wilson confidence intervals, overlap weighting, and T-versus-X-Learner consistency as diagnostics.

### Analytical Task 5: Uplift Decile Analysis

#### XGBoost T-Learner

{markdown_table(['Decile', 'N', 'Average predicted benefit', 'Observed control − treated gap', '95% CI for observed gap', 'Treated N', 'Control N', 'Treated share'], decile_rows(xgb_t_deciles))}

The top XGBoost T-Learner decile contained {int(xgb_t_top['top_decile_n'])} records and had average predicted benefit {number(xgb_t_top['top_decile_avg_predicted_benefit'])}. Its observed control-minus-treated gap was {number(xgb_t_top['top_decile_observed_control_minus_treated_gap'])}; the 95% interval was {number(xgb_t_deciles.iloc[0]['observed_gap_ci_lower_95'])} to {number(xgb_t_deciles.iloc[0]['observed_gap_ci_upper_95'])}. The interval includes zero, reflecting the limited decile sample.

{image('T-Learner/XGBoost/dashboard_avg_benefit_by_decile.png', 'XGBoost T-Learner average predicted benefit by decile')}

{image('T-Learner/XGBoost/dashboard_observed_gap_by_decile.png', 'XGBoost T-Learner observed ED-rate gap by decile')}

{image('T-Learner/XGBoost/dashboard_uplift_curve_by_decile.png', 'XGBoost T-Learner uplift curve')}

Across deciles, the Spearman correlation between predicted benefit and observed gap was {number(evaluation.loc[evaluation['model'].eq('XGBoost'), 'benefit_gap_spearman_corr_by_decile'].iloc[0])}, with mean absolute predicted-versus-observed gap {number(evaluation.loc[evaluation['model'].eq('XGBoost'), 'mean_abs_predicted_minus_observed_gap_by_decile'].iloc[0])}. This does not establish reliable causal ranking; it supports treating the current result as a candidate model for further validation.

#### GLMNet T-Learner

The GLMNet top decile had average predicted benefit {number(glm_t_top['top_decile_avg_predicted_benefit'])} and observed gap {number(glm_t_top['top_decile_observed_control_minus_treated_gap'])}. The predicted magnitude was outside the observed-gap 95% interval, indicating probable overstatement of the absolute benefit scale even though its decile rank correlation was positive.

{markdown_table(['Decile', 'N', 'Average predicted benefit', 'Observed control − treated gap', '95% CI for observed gap', 'Treated N', 'Control N', 'Treated share'], decile_rows(glm_t_deciles))}

#### XGBoost X-Learner

{markdown_table(['Decile', 'N', 'Average predicted benefit', 'Observed control − treated gap', '95% CI for observed gap', 'Treated N', 'Control N', 'Treated share'], decile_rows(xgb_x_deciles))}

{image('X-Learner/XGBoost/dashboard_avg_benefit_by_decile.png', 'XGBoost X-Learner average predicted benefit by decile')}

{image('X-Learner/XGBoost/dashboard_observed_gap_by_decile.png', 'XGBoost X-Learner observed ED-rate gap by decile')}

#### T-Learner versus X-Learner consistency

{markdown_table(['Model family', 'Pearson correlation', 'Spearman correlation', 'Top-decile overlap', 'T-Learner mean benefit', 'X-Learner mean benefit'], consistency_rows)}

The XGBoost implementations had moderate score correlation and {percent(consistency.loc[consistency['model'].eq('XGBoost'), 'top_decile_overlap_pct'].iloc[0])} top-decile overlap. GLMNet showed little agreement across learner frameworks. Framework sensitivity is another reason to avoid using a single fitted benefit magnitude as a causal estimate.

### Level 2 Summary

The Oregon models produce differentiated benefit rankings, but the observed-gap pattern is noisy and sensitive to the learner framework. The XGBoost T-Learner is the closest direct reproduction of the original primary workflow; the X-Learner is best treated as a sensitivity analysis. A prospective or quasi-experimental validation design is needed before deployment.

## Evaluation Level 3: Operational Interpretation

### Analytical Task 6: Variable Importance and Explainability

#### XGBoost T-Learner benefit drivers

{markdown_table(['Rank', 'Feature', 'Mean absolute benefit SHAP', 'Mean signed benefit SHAP', 'Positive SHAP share'], driver_rows(xgb_t_shap))}

{image('T-Learner/XGBoost/dashboard_shap_benefit_score.png', 'XGBoost T-Learner SHAP benefit drivers')}

The leading benefit-score drivers were recent ED utilization, recent total cost, percolator score, recent 30-day ED utilization, and age. SHAP values explain model behavior; they do not establish that changing a feature would change treatment benefit.

#### XGBoost X-Learner benefit drivers

{markdown_table(['Rank', 'Feature', 'Mean absolute benefit SHAP', 'Mean signed benefit SHAP', 'Positive SHAP share'], driver_rows(xgb_x_drivers))}

{image('X-Learner/XGBoost/dashboard_xlearner_benefit_drivers.png', 'XGBoost X-Learner benefit drivers')}

#### GLMNet comparison

The GLMNet benefit explanation remains available as a transparent linear-model comparison. Its leading absolute contribution terms were:

{markdown_table(['Rank', 'Feature', 'Mean absolute benefit contribution', 'Mean signed contribution', 'Positive contribution share'], driver_rows(glm_t_shap, 10))}

### Analytical Task 7: Illustrative Business Value

The notebook retains the Funds Combined business assumptions: **$1,200 gross value per avoided ED event** and **$250 intervention cost per targeted record**. These are scenario inputs, not measured Oregon financial outcomes. ROI is `(gross savings − intervention cost) / intervention cost`.

#### XGBoost T-Learner targeting

{markdown_table(['Decile', 'N', 'Expected ED-rate reduction', 'Expected ED events avoided', 'Gross savings', 'Intervention cost', 'Net savings', 'ROI'], roi_rows)}

For the top T-Learner decile, estimated gross savings were {dollars(xgb_t_top['top_decile_gross_savings'])}, intervention cost was {dollars(xgb_t_top['top_decile_intervention_cost'])}, and net savings were {dollars(xgb_t_top['top_decile_net_savings'])}, corresponding to {percent(xgb_t_top['top_decile_roi'])} illustrative ROI. The negative value means the predicted benefit does not clear the specified cost threshold for that complete decile.

{image('T-Learner/XGBoost/dashboard_roi_net_savings_by_decile.png', 'XGBoost T-Learner net savings by decile')}

#### XGBoost X-Learner targeting

{markdown_table(['Decile', 'N', 'Expected ED-rate reduction', 'Expected ED events avoided', 'Gross savings', 'Intervention cost', 'Net savings', 'ROI'], x_roi_rows)}

The X-Learner produces a more favorable first-decile scenario but negative returns in later deciles. Because the T- and X-Learners disagree materially for some records, these values should be used for scenario planning rather than booked savings.

{image('X-Learner/XGBoost/dashboard_xlearner_roi_net_savings_by_decile.png', 'XGBoost X-Learner net savings by decile')}

At approximately the top 50% of the test population, cumulative modeled gross savings were:

{markdown_table(['Targeting approach', 'Population fraction', 'N', 'Cumulative gross savings'], [[r['targeting_approach'], percent(r['population_fraction_targeted']), int(r['n']), dollars(r['cumulative_gross_savings'])] for _, r in xgb_x_savings.iterrows()])}

{image('X-Learner/XGBoost/dashboard_cumulative_gross_savings_targeting.png', 'Cumulative gross savings by targeting approach')}

{image('X-Learner/XGBoost/dashboard_marginal_gross_savings_advantage_vs_current_risk.png', 'Marginal gross-savings advantage of uplift targeting')}

### Level 3 Summary

The models consistently emphasize recent utilization and cost-related baseline signals. Under the retained economic assumptions, the XGBoost T-Learner's full top decile is not cost-saving, while the X-Learner's first decile is positive. That contrast reinforces the need for prospective testing, explicit intervention-capacity constraints, and sensitivity analyses over unit cost and avoided-event value.

## Recommended Next Steps

1. Run a prospective pilot with randomized or defensible quasi-experimental assignment among high-ranked members.
2. Predefine ranking metrics, minimum detectable effect, and subgroup sample sizes before evaluation.
3. Recalibrate outcome probabilities and reassess learner-framework stability on a later Oregon cohort.
4. Replace illustrative cost assumptions with Oregon-specific allowed amounts and intervention delivery costs.
5. Monitor overlap, treatment propensity, and score drift; do not score members outside the baseline population without review.
6. Review fairness and operational feasibility across age, gender, geography, dual-eligibility status, and available demographic groups before deployment.

## Reproducibility and Output Audit

- Source data: `DataSets/Oregon_2YearDataset.csv` (read-only during analysis)
- Executed notebook: `Code/Uplift Model Code_Oregon_2YearDataset.ipynb`
- Notebook builder: `Code/build_oregon_uplift_notebook.py`
- Report generator: `Code/generate_oregon_uplift_report.py`
- Oregon outputs: `Outputs/Uplift_Oregon/`
- Random seed: 123
- Split: grouped by `member_id`; zero member overlap
- Preprocessing: fit on training data only
- Committed notebook backend: SageMaker CUDA GPU required on device 0
- Checked-in result snapshot backend: {audit['xgboost_execution_backend']}
- Leakage assertions: passed
- Death handling: no death field present; no death-derived feature and no death-based exclusion

The parity manifest records {parity_counts.get(('applicable', True), 0)} generated applicable artifacts, {parity_counts.get(('applicable', False), 0)} applicable artifacts not generated, {parity_counts.get(('oregon_specific', True), parity_counts.get(('funds_specific', True), 0))} Oregon-specific generated artifacts, and {parity_counts.get(('not_applicable', False), 0)} synthetic-ground-truth artifacts marked not applicable. The six applicable-but-absent files are GLMNet T-Learner targeting outputs that were not produced by the Funds source workflow:

{chr(10).join(f'- `{path}`' for path in missing_applicable)}

These absences are explicit in `Outputs/Uplift_Oregon/Python/output_parity_manifest.csv`; no missing artifact is silently treated as generated.
"""

    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
