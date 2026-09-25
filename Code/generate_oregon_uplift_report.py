"""Build the Funds-style Oregon uplift report from committed model artifacts.

The modeling notebook remains the source of truth for fitted models.  This
script performs no refitting; it formats the saved Oregon results and derives
the same cumulative targeting views used in the Funds report from the saved
held-out test scores.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "Outputs" / "Uplift_Oregon" / "Python"
REPORT_PATH = ROOT / "PRISM_Intervention_Benefit_Modeling_Oregon_README.md"
ED_EVENT_VALUE = 1_200.0


def load(relative_path: str) -> pd.DataFrame:
    path = OUTPUT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    return pd.read_csv(path)


def number(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def integer(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"


def percent(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def dollars(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "—"
    numeric = float(value)
    if numeric < 0:
        return f"-${abs(numeric):,.{digits}f}"
    return f"${numeric:,.{digits}f}"


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def image(relative_path: str, alt: str) -> str:
    path = OUTPUT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Report image is missing: {path}")
    return f"![{alt}]({rel(path)})"


def side_by_side(left_path: str, left_alt: str, right_path: str, right_alt: str) -> str:
    left = rel(OUTPUT_ROOT / left_path)
    right = rel(OUTPUT_ROOT / right_path)
    for path in (ROOT / left, ROOT / right):
        if not path.exists():
            raise FileNotFoundError(f"Report image is missing: {path}")
    return (
        '<table><tr><td width="50%" valign="top">'
        f'<img src="{left}" alt="{left_alt}"></td>'
        '<td width="50%" valign="top">'
        f'<img src="{right}" alt="{right_alt}"></td></tr></table>'
    )


def make_tlearner_targeting_artifacts(scored: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the Funds cumulative/marginal targeting calculation for T scores."""
    out_dir = OUTPUT_ROOT / "T-Learner" / "XGBoost"
    n_total = len(scored)
    decile_size = n_total // 10
    if decile_size == 0:
        raise ValueError("The held-out score file is too small for decile targeting.")

    rows: list[dict[str, float | int | str]] = []
    approaches = {
        "Uplift score": "t_learner_benefit_score",
        "Current risk score": "current_risk_score",
    }
    for label, ranking_column in approaches.items():
        ranked = scored.sort_values(ranking_column, ascending=False, na_position="last")
        for decile in range(1, 11):
            top_n = n_total if decile == 10 else decile * decile_size
            selected = ranked.head(top_n)
            avoided = float(selected["t_learner_benefit_score"].sum())
            rows.append(
                {
                    "targeting_approach": label,
                    "through_decile": decile,
                    "population_fraction_targeted": top_n / n_total,
                    "n": top_n,
                    "cumulative_estimated_ed_visits_avoided": avoided,
                    "cumulative_gross_savings": avoided * ED_EVENT_VALUE,
                }
            )

    cumulative = pd.DataFrame(rows)
    cumulative.to_csv(out_dir / "cumulative_gross_savings_by_targeting.csv", index=False)

    marginal_parts: list[pd.DataFrame] = []
    for approach, group in cumulative.groupby("targeting_approach", sort=False):
        part = group.sort_values("through_decile").copy()
        part["targeting_approach"] = approach
        part["decile"] = part["through_decile"]
        part["marginal_gross_savings"] = part["cumulative_gross_savings"].diff().fillna(
            part["cumulative_gross_savings"]
        )
        marginal_parts.append(
            part[["targeting_approach", "decile", "marginal_gross_savings", "cumulative_gross_savings"]]
        )
    marginal = pd.concat(marginal_parts, ignore_index=True)
    marginal.to_csv(out_dir / "marginal_gross_savings_by_targeting.csv", index=False)

    uplift = marginal[marginal["targeting_approach"].eq("Uplift score")].set_index("decile")
    risk = marginal[marginal["targeting_approach"].eq("Current risk score")].set_index("decile")
    advantage = pd.DataFrame(
        {
            "decile": uplift.index,
            "uplift_marginal_gross_savings": uplift["marginal_gross_savings"].values,
            "risk_marginal_gross_savings": risk["marginal_gross_savings"].values,
        }
    )
    advantage["decile"] = pd.to_numeric(advantage["decile"], errors="raise").astype(int)
    advantage["marginal_advantage"] = (
        advantage["uplift_marginal_gross_savings"] - advantage["risk_marginal_gross_savings"]
    )
    advantage.to_csv(out_dir / "marginal_gross_savings_advantage_vs_current_risk.csv", index=False)

    top50 = cumulative[cumulative["through_decile"].eq(5)].copy()
    top50.to_csv(out_dir / "cumulative_gross_savings_summary_top50.csv", index=False)

    colors = {"Uplift score": "#2F5597", "Current risk score": "#ED7D31"}
    fig, ax = plt.subplots(figsize=(8.5, 5.25))
    for approach, group in cumulative.groupby("targeting_approach", sort=False):
        ax.plot(
            group["population_fraction_targeted"] * 100,
            group["cumulative_gross_savings"],
            marker="o",
            linewidth=2.2,
            label=approach,
            color=colors[approach],
        )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title("XGBoost T-Learner: Cumulative Gross Savings by Targeting Method")
    ax.set_xlabel("Population targeted (%)")
    ax.set_ylabel("Cumulative gross savings ($)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "dashboard_cumulative_gross_savings_targeting.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.25))
    ax.bar(advantage["decile"], advantage["marginal_advantage"], color="#2F5597")
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_title("XGBoost T-Learner: Marginal Savings Advantage vs Current Risk")
    ax.set_xlabel("Targeting decile")
    ax.set_ylabel("Uplift minus risk marginal gross savings ($)")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "dashboard_marginal_gross_savings_advantage_vs_current_risk.png", dpi=200)
    plt.close(fig)

    return cumulative


def main() -> None:
    review = load("data_review_summary.csv").iloc[0]
    audit = load("preprocessing_audit_summary.csv").iloc[0]
    events = load("factual_event_count_summary.csv")
    evaluation = load("model_evaluation_summary.csv")
    prediction_ranges = load("factual_prediction_ranges.csv")
    separation = load("factual_prediction_separation.csv")
    risk_population = load("risk_tier_population_summary.csv")
    column_audit = load("preprocessing_column_audit.csv")

    t_gap = load("T-Learner/XGBoost/uplift_observed_gap_by_decile.csv")
    t_top = load("T-Learner/XGBoost/top_benefit_decile_summary.csv").iloc[0]
    t_roi = load("T-Learner/XGBoost/uplift_roi_by_decile.csv")
    t_risk = load("T-Learner/XGBoost/tlearner_risk_tier_benefit_group_summary.csv")
    t_shap_models = load("T-Learner/XGBoost/shap_importance_treated_control_models.csv")
    t_shap_benefit = load("T-Learner/XGBoost/shap_importance_benefit_score.csv")

    glm_top = load("T-Learner/GLMNet/top_benefit_decile_summary.csv").iloc[0]
    glm_benefit = load("T-Learner/GLMNet/shap_importance_benefit_score.csv")

    x_decile = load("X-Learner/XGBoost/xlearner_decile_summary.csv")
    x_gap = load("X-Learner/XGBoost/uplift_observed_gap_by_decile.csv")
    x_roi = load("X-Learner/XGBoost/xlearner_roi_by_decile.csv")
    x_risk = load("X-Learner/XGBoost/xlearner_risk_tier_benefit_group_summary.csv")
    x_benefit = load("X-Learner/XGBoost/xlearner_benefit_driver_importance.csv")
    consistency = load("X-Learner/xlearner_vs_tlearner_consistency_summary.csv")
    x_test = load("X-Learner/XGBoost/xlearner_scored_test_output.csv")
    x_cumulative = load("X-Learner/XGBoost/cumulative_gross_savings_by_targeting.csv")
    t_cumulative = make_tlearner_targeting_artifacts(x_test)

    xgb_eval = evaluation[evaluation["model"].eq("XGBoost")].iloc[0]
    glm_eval = evaluation[evaluation["model"].eq("GLMNET")].iloc[0]
    avg_xgb_auc = np.mean([xgb_eval["treated_test_auc"], xgb_eval["control_test_auc"]])
    avg_glm_auc = np.mean([glm_eval["treated_test_auc"], glm_eval["control_test_auc"]])
    avg_xgb_cal = np.mean([xgb_eval["treated_calibration_error"], xgb_eval["control_calibration_error"]])
    avg_glm_cal = np.mean([glm_eval["treated_calibration_error"], glm_eval["control_calibration_error"]])

    post_treatment = column_audit[column_audit["role"].eq("post-treatment variable")]
    event_rows = [
        [r["split"], r["group"], integer(r["n"]), integer(r["positive_ed_events"]), integer(r["negative_ed_events"]), percent(r["event_rate"])]
        for _, r in events.iterrows()
    ]
    performance_rows = [
        [
            r["model"],
            number(r["treated_cv_auc"]),
            number(r["control_cv_auc"]),
            number(r["treated_test_auc"]),
            number(r["control_test_auc"]),
            number(r["treated_brier_score"]),
            number(r["control_brier_score"]),
            number(r["treated_calibration_error"]),
            number(r["control_calibration_error"]),
        ]
        for _, r in evaluation.iterrows()
    ]
    separation_rows = [
        [r["model"], r["group"], number(r["auc"]), number(r["avg_pred_actual_positive"]), number(r["avg_pred_actual_negative"]), number(r["avg_pred_positive_minus_negative"])]
        for _, r in separation.iterrows()
    ]
    range_rows = [
        [r["model"], r["group"], number(r["min_pred"]), number(r["p10_pred"]), number(r["median_pred"]), number(r["mean_pred"]), number(r["p90_pred"]), number(r["max_pred"])]
        for _, r in prediction_ranges.iterrows()
    ]
    risk_rows = [
        [integer(r["risk_tier"]), r["risk_tier_label"], integer(r["members"]), percent(r["pct_population"]), number(r["min_current_risk_score"]), number(r["max_current_risk_score"]), number(r["avg_current_risk_score"])]
        for _, r in risk_population.iterrows()
    ]

    def gap_rows(frame: pd.DataFrame) -> list[list[object]]:
        return [
            [
                integer(r["uplift_decile"]),
                integer(r["n"]),
                number(r["avg_predicted_benefit"]),
                number(r["observed_control_minus_treated_gap"]),
                f"{number(r['observed_gap_ci_lower_95'])} to {number(r['observed_gap_ci_upper_95'])}",
                integer(r["treated_n"]),
                integer(r["control_n"]),
            ]
            for _, r in frame.iterrows()
        ]

    def shap_rows(frame: pd.DataFrame, limit: int = 10) -> list[list[object]]:
        absolute = "mean_abs_benefit_shap" if "mean_abs_benefit_shap" in frame else "mean_abs_benefit_contribution"
        signed = "mean_signed_benefit_shap" if "mean_signed_benefit_shap" in frame else "mean_signed_benefit_contribution"
        positive = "pct_positive_benefit_shap" if "pct_positive_benefit_shap" in frame else "pct_positive_benefit_contribution"
        return [
            [rank, r["feature"], number(r[absolute]), number(r[signed]), percent(r[positive])]
            for rank, (_, r) in enumerate(frame.head(limit).iterrows(), 1)
        ]

    def risk_mix_rows(frame: pd.DataFrame) -> list[list[object]]:
        pivot = frame.pivot(index="risk_tier", columns="benefit_group", values="pct_within_risk_tier").fillna(0)
        return [
            [integer(idx), percent(row.get("High benefit", 0)), percent(row.get("Medium benefit", 0)), percent(row.get("Low benefit", 0))]
            for idx, row in pivot.iterrows()
        ]

    def roi_rows(frame: pd.DataFrame) -> list[list[object]]:
        return [
            [integer(r["uplift_decile"]), integer(r["n"]), number(r["expected_ed_rate_reduction"]), number(r["expected_ed_visits_avoided"], 2), dollars(r["gross_savings"]), dollars(r["intervention_cost"]), dollars(r["net_savings"]), percent(r["roi"])]
            for _, r in frame.iterrows()
        ]

    def targeting_rows(frame: pd.DataFrame) -> list[list[object]]:
        return [
            [r["targeting_approach"], integer(r["through_decile"]), percent(r["population_fraction_targeted"]), integer(r["n"]), number(r["cumulative_estimated_ed_visits_avoided"], 2), dollars(r["cumulative_gross_savings"])]
            for _, r in frame[frame["through_decile"].isin([1, 3, 5, 10])].iterrows()
        ]

    t_model_shap = {
        name: group.sort_values("mean_abs_shap", ascending=False).head(10)
        for name, group in t_shap_models.groupby("model")
    }
    consistency_rows = [
        [r["model"], number(r["pearson_benefit_score_corr"]), number(r["spearman_benefit_score_corr"]), percent(r["top_decile_overlap_pct"]), number(r["t_learner_mean_benefit_score"]), number(r["x_learner_mean_benefit_score"])]
        for _, r in consistency.iterrows()
    ]

    report = f"""# PRISM Intervention Benefit Modeling Oregon Report Draft

## Background

PRISM is intended to focus intervention resources on members most likely to benefit, not simply those most likely to have an emergency department event. This Oregon report mirrors the organization, methods, exhibits, and decision sequence of the **PRISM Intervention Benefit Modeling Funds Report Draft**, while using the Oregon two-year cohort and Oregon-specific preprocessing.

The analysis includes {integer(review['total_analytical_records'])} episodes from {integer(review['unique_members'])} unique members. The outcome is a binary indicator of any emergency department use within 90 days. Treatment is engagement in PRISM. A positive benefit score means the model predicts lower 90-day ED-event risk under engagement than under no engagement.

> **Interpretation boundary.** This is an observational modeling analysis. Individual counterfactual outcomes are not observed, treatment was not randomized, and decile-level treatment/control counts are small. Results support model comparison and pilot design; they do not by themselves establish a causal intervention effect.

## Business Question

Which Oregon members are most likely to benefit from PRISM engagement, measured as a modeled reduction in the probability of an ED event within 90 days?

## Project Objectives

1. Apply the same T-Learner and X-Learner workflow used in the Funds Combined analysis.
2. Use only variables available before intervention and explicitly exclude treatment, outcome, and post-intervention information from predictors.
3. Compare XGBoost with regularized logistic regression using the same model-selection and evaluation logic as the Funds report.
4. Evaluate factual outcome prediction, benefit ranking, framework consistency, explainability, and illustrative targeting value.
5. Produce an Oregon report with the same reporting structure and visual treatment as the Funds report.

## Analytical Task 1: Understanding the Modeling Framework

### Outcome Variable

`outcome_ed_90d` is the outcome of interest. It was converted to binary form: any value greater than zero is 1 and zero is 0. Missing values were assigned to 0 as specified. The final cohort contains {integer(review['outcome_events'])} events, an overall prevalence of {percent(review['outcome_prevalence'])}.

### Treatment Variable

`Engaged_flag = 1` defines treatment and `Engaged_flag = 0` defines control. `OptOut_flag` contains the inverse treatment information and was used only as a validation check before being excluded. The fields agreed for all modeling records; there were {integer(audit['contradictory_treatment_rows'])} contradictions and {integer(audit['unresolved_treatment_rows'])} unresolved rows.

### Predictor Variables

Only baseline variables were eligible. The preprocessing audit retained {integer(audit['retained_predictor_count'])} source predictors and produced {integer(audit['encoded_feature_count'])} encoded model features. Numeric imputation, categorical levels, and zero-variance filtering were learned from training data only.

The following post-intervention variables were detected and excluded before any model matrix was built:

<!-- AUTO_TABLE: oregon_post_treatment_exclusions -->
{table(['Excluded post-intervention variable', 'Reason'], [[r['original_oregon_column'], r['reason']] for _, r in post_treatment.iterrows()])}

The member identifier was retained only for grouped splitting and traceability. Treatment and outcome fields were never predictors. No death-related field was available in the Oregon source, so no death flag could be derived; no member was excluded on the basis of death. The 192 missing and 93 out-of-range source risk-tier values were modeled as an explicit `Missing` category. Training-set zero-variance fields `client_contract`, `plan_type`, and `program` were removed.

### Overall Modeling Workflow

```mermaid
flowchart LR
    A[Oregon two-year data] --> B[Outcome and treatment validation]
    B --> C[Remove post-treatment and leakage fields]
    C --> D[Member-grouped 70/30 split]
    D --> E[Train-only preprocessing]
    E --> F[T-Learner: XGBoost and GLMNet]
    E --> G[X-Learner: XGBoost and GLMNet]
    F --> H[Outcome validation]
    G --> H
    H --> I[Benefit deciles and uncertainty]
    I --> J[Explainability and business value]
```

All {integer(audit['raw_rows'])} source episodes were retained, including {integer(audit['repeated_member_episodes_retained'])} repeated member episodes. Grouping the split by `member_id` produced {integer(audit['train_rows'])} training rows and {integer(audit['test_rows'])} test rows with zero member overlap. The SageMaker execution audit records: `{audit['xgboost_execution_backend']}`.

### Treatment-Effect Frameworks

#### T-Learner

The T-Learner fits one outcome model among engaged records and a second among controls. Both models score every held-out record. The estimated benefit is:

`predicted ED risk without engagement − predicted ED risk with engagement`

```mermaid
flowchart LR
    A[Training data] --> B[Treated subset]
    A --> C[Control subset]
    B --> D[Outcome model under treatment]
    C --> E[Outcome model under control]
    D --> F[Predicted treated risk]
    E --> G[Predicted control risk]
    F --> H[Benefit = control risk - treated risk]
    G --> H
```

#### X-Learner

The X-Learner first models factual outcomes, imputes treatment effects for each treatment group, fits treatment-effect models, and combines their predictions using the estimated propensity score. It is included as a sensitivity analysis because it handles treatment-group imbalance differently from the T-Learner.

```mermaid
flowchart LR
    A[Factual outcome models] --> B[Imputed treated effects]
    A --> C[Imputed control effects]
    B --> D[Treated effect model]
    C --> E[Control effect model]
    D --> F[Propensity-weighted combination]
    E --> F
    F --> G[X-Learner benefit score]
```

### Modeling Techniques

The report retains the Funds modeling choices: XGBoost and regularized logistic regression, the same hyperparameter-search logic, stratified cross-validation within treatment group, fixed random seeds, held-out member-grouped evaluation, T- and X-Learner constructions, benefit deciles, SHAP-style explanations, and the same illustrative economic assumptions. XGBoost used the SageMaker CUDA device recorded in the audit; no CPU refit was substituted for these results.

## Analytical Task 2: Data Review

<!-- AUTO_TABLE: oregon_data_review -->
{table(['Measure', 'Oregon result'], [
    ['Analytical records', integer(review['total_analytical_records'])],
    ['Unique members', integer(review['unique_members'])],
    ['Treated records', f"{integer(review['treated_records'])} ({percent(review['treatment_rate'])})"],
    ['Control records', f"{integer(review['control_records'])} ({percent(1-review['treatment_rate'])})"],
    ['90-day ED events', f"{integer(review['outcome_events'])} ({percent(review['outcome_prevalence'])})"],
    ['Treated outcome rate', percent(review['treated_outcome_rate'])],
    ['Control outcome rate', percent(review['control_outcome_rate'])],
    ['Training rows / unique members', f"{integer(audit['train_rows'])} / {integer(audit['train_unique_members'])}"],
    ['Test rows / unique members', f"{integer(audit['test_rows'])} / {integer(audit['test_unique_members'])}"],
    ['Member overlap across splits', integer(audit['train_test_member_overlap'])],
] )}

The raw treated outcome rate ({percent(review['treated_outcome_rate'])}) is lower than the raw control outcome rate ({percent(review['control_outcome_rate'])}). That difference is descriptive, not an adjusted treatment effect, because engagement was not randomized.

### Risk Tier Definition

Oregon source risk tiers were retained as baseline categories rather than reconstructed from the current risk score. Valid tiers covered {integer(risk_population['members'].sum())} records.

<!-- AUTO_TABLE: oregon_risk_tier_definition -->
{table(['Risk tier', 'Label', 'Records', 'Share of valid tiers', 'Minimum score', 'Maximum score', 'Average score'], risk_rows)}

## Evaluation Roadmap

The Funds report evaluates models in three linked levels. Oregon follows the same sequence:

1. **Outcome model validation:** discrimination, factual separation, Brier score, and calibration.
2. **Uplift model validation:** benefit-score behavior, decile ranking, uncertainty, risk-versus-benefit mix, and agreement between T- and X-Learners.
3. **Operational interpretation:** model drivers and illustrative business value under fixed economic assumptions.

# Evaluation Level 1: Outcome Model Validation

## Analytical Task 3: Model Performance

### Event Counts And Modeling Constraints

<!-- AUTO_TABLE: oregon_factual_event_counts -->
{table(['Split', 'Treatment group', 'Records', 'ED events', 'No ED event', 'Event rate'], event_rows)}

The held-out control group contains 116 records and 32 events. These counts are adequate for an initial comparison but still make subgroup AUC, calibration, and decile-level estimates imprecise. Wide confidence intervals later in the report are therefore expected rather than anomalous.

### Discrimination Performance

<!-- AUTO_TABLE: oregon_model_performance -->
{table(['Model', 'Treated CV AUC', 'Control CV AUC', 'Treated test AUC', 'Control test AUC', 'Treated Brier', 'Control Brier', 'Treated calibration error', 'Control calibration error'], performance_rows)}

GLMNet has the higher mean held-out factual AUC ({number(avg_glm_auc)} versus {number(avg_xgb_auc)} for XGBoost), driven by its control-group AUC. XGBoost is slightly stronger for treated records and has the better average calibration error ({number(avg_xgb_cal)} versus {number(avg_glm_cal)}). The Oregon result is therefore mixed rather than a simple winner on every metric.

### Factual Discrimination And Prediction Separation

<!-- AUTO_TABLE: oregon_prediction_separation -->
{table(['Model', 'Group', 'AUC', 'Mean prediction: event', 'Mean prediction: no event', 'Difference'], separation_rows)}

{side_by_side('T-Learner/XGBoost/dashboard_predicted_treated_vs_control.png', 'XGBoost factual prediction distributions', 'T-Learner/GLMNet/dashboard_predicted_treated_vs_control.png', 'GLMNet factual prediction distributions')}

### Brier Score And Calibration

Both model families have similar average Brier scores, while XGBoost has lower mean calibration error. Calibration matters directly here because a benefit score is the difference between two predicted probabilities; bias in either potential-outcome model can distort the estimated treatment-effect scale.

<!-- AUTO_CHART: oregon_calibration_comparison -->
{side_by_side('T-Learner/XGBoost/dashboard_calibration_plot.png', 'XGBoost calibration', 'T-Learner/GLMNet/dashboard_calibration_plot.png', 'GLMNet calibration')}

### Factual Prediction Range And Rare-Outcome Interpretation

<!-- AUTO_TABLE: oregon_prediction_ranges -->
{table(['Model', 'Group', 'Minimum', 'P10', 'Median', 'Mean', 'P90', 'Maximum'], range_rows)}

The event is not extremely rare overall, but subgroup sizes remain limited. Some probabilities approach the edges of the observed range, especially for GLMNet; this helps explain why GLMNet can rank records well while overstating the magnitude of its top predicted benefit.

### Model Performance Takeaway

For direct methodological parity with the Funds report, XGBoost remains the primary Oregon uplift model and GLMNet remains the transparent sensitivity model. That choice is supported by XGBoost's stronger calibration and plausible top-decile benefit scale, not by universal AUC dominance. GLMNet's higher average factual AUC is reported explicitly and should be revisited in future cohorts.

## Level 1 Summary: Outcome Model Validation

The factual outcome models contain useful signal: held-out AUCs range from {number(evaluation[['treated_test_auc','control_test_auc']].min().min())} to {number(evaluation[['treated_test_auc','control_test_auc']].max().max())}. However, discrimination alone does not validate uplift ranking. XGBoost offers the better probability calibration, while GLMNet has stronger average discrimination. Both therefore move forward to treatment-effect evaluation, with XGBoost as the report's primary model and GLMNet as a sensitivity check.

# Evaluation Level 2: Uplift Model Validation

## Analytical Task 4: Treatment Effect Analysis

For each held-out member episode, the T-Learner produces predicted ED risk under engagement and predicted ED risk under control. The difference is the modeled benefit. X-Learner scores provide a second estimate built from imputed treatment effects and propensity-weighted combination.

The top XGBoost T-Learner decile averages {number(t_top['top_decile_avg_predicted_benefit'])} predicted benefit—approximately {number(100*t_top['top_decile_avg_predicted_benefit'],1)} percentage points of modeled ED-risk reduction. Its observed control-minus-treated gap is {number(t_top['top_decile_observed_control_minus_treated_gap'])}, but the 95% interval ({number(xgb_eval['top_decile_observed_gap_ci_lower_95'])} to {number(xgb_eval['top_decile_observed_gap_ci_upper_95'])}) includes zero.

### Synthetic True-Benefit Validation

The Funds workflow includes synthetic-data checks because true individual treatment effects are known in simulation. The Oregon observational file has no observed counterfactual and therefore no true individual benefit label. PEHE, true-benefit correlation, and true top-group overlap are not identifiable here. Oregon uses the corresponding real-data diagnostics available without inventing ground truth: observed within-decile treated/control gaps with Wilson intervals, T-versus-X consistency, and sensitivity to model family.

## Analytical Task 5: Uplift Decile Analysis

The strongest predicted benefit should appear in decile 1 and decrease toward decile 10. Predicted ranking is clear for both XGBoost learners, but observed gaps are noisy.

<!-- AUTO_TABLE: oregon_tlearner_deciles -->
{table(['Decile', 'N', 'Average predicted benefit', 'Observed control - treated gap', '95% CI', 'Treated N', 'Control N'], gap_rows(t_gap))}

<!-- AUTO_CHART: oregon_tlearner_decile_pair -->
{side_by_side('T-Learner/XGBoost/dashboard_avg_benefit_by_decile.png', 'T-Learner predicted benefit by decile', 'T-Learner/XGBoost/dashboard_observed_gap_by_decile.png', 'T-Learner observed gap by decile')}

The XGBoost T-Learner's decile Spearman correlation between predicted benefit and observed gap is {number(xgb_eval['benefit_gap_spearman_corr_by_decile'])}. The negative, near-zero value means the observed gap does not decline monotonically with predicted benefit. This is the central validation weakness of the current T-Learner result.

The XGBoost X-Learner produces a wider score spread, from {number(x_decile.iloc[0]['avg_benefit_score'])} in decile 1 to {number(x_decile.iloc[-1]['avg_benefit_score'])} in decile 10. Its first-decile observed gap is {number(x_gap.iloc[0]['observed_control_minus_treated_gap'])}, with a 95% interval of {number(x_gap.iloc[0]['observed_gap_ci_lower_95'])} to {number(x_gap.iloc[0]['observed_gap_ci_upper_95'])}.

<!-- AUTO_TABLE: oregon_xlearner_deciles -->
{table(['Decile', 'N', 'Average predicted benefit', 'Observed control - treated gap', '95% CI', 'Treated N', 'Control N'], gap_rows(x_gap))}

<!-- AUTO_CHART: oregon_xlearner_decile_pair -->
{side_by_side('X-Learner/XGBoost/dashboard_avg_benefit_by_decile.png', 'X-Learner predicted benefit by decile', 'X-Learner/XGBoost/dashboard_observed_gap_by_decile.png', 'X-Learner observed gap by decile')}

GLMNet's T-Learner has a positive decile Spearman correlation ({number(glm_eval['benefit_gap_spearman_corr_by_decile'])}), but its top predicted benefit ({number(glm_top['top_decile_avg_predicted_benefit'])}) lies above the observed-gap 95% interval ({number(glm_eval['top_decile_observed_gap_ci_lower_95'])} to {number(glm_eval['top_decile_observed_gap_ci_upper_95'])}). It may rank more consistently while overstating absolute benefit.

### Risk Tier Versus Benefit Group

Current risk and modeled benefit answer different questions. Within each source risk tier, members were divided into high-, medium-, and low-benefit thirds based on the respective model score.

<!-- AUTO_TABLE: oregon_tlearner_risk_benefit_mix -->
{table(['Risk tier', 'High benefit', 'Medium benefit', 'Low benefit'], risk_mix_rows(t_risk))}

For comparison, the X-Learner risk-tier mix is:

<!-- AUTO_TABLE: oregon_xlearner_risk_benefit_mix -->
{table(['Risk tier', 'High benefit', 'Medium benefit', 'Low benefit'], risk_mix_rows(x_risk))}

<!-- AUTO_CHART: oregon_risk_benefit_pair -->
{side_by_side('T-Learner/XGBoost/dashboard_tlearner_risk_tier_by_benefit_group.png', 'T-Learner risk tier and benefit group', 'X-Learner/XGBoost/dashboard_xlearner_risk_tier_by_benefit_group.png', 'X-Learner risk tier and benefit group')}

High modeled benefit appears in multiple risk tiers rather than only in the highest-risk tier. For example, {percent(t_risk[(t_risk['risk_tier'].eq(1)) & (t_risk['benefit_group'].eq('High benefit'))]['pct_within_risk_tier'].iloc[0])} of valid tier-1 test records fall in the T-Learner high-benefit group. Risk-only targeting would therefore select a meaningfully different population.

### Framework Consistency

<!-- AUTO_TABLE: oregon_framework_consistency -->
{table(['Model family', 'Pearson correlation', 'Spearman correlation', 'Top-decile overlap', 'T-Learner mean benefit', 'X-Learner mean benefit'], consistency_rows)}

XGBoost has moderate T-versus-X score agreement (Spearman {number(consistency[consistency['model'].eq('XGBoost')]['spearman_benefit_score_corr'].iloc[0])}), while GLMNet agreement is weak. Moderate score correlation with limited top-decile overlap means framework choice materially changes which individual records are prioritized.

### True-Benefit Top-Group Overlap

True-benefit top-group overlap cannot be calculated for Oregon because individual counterfactual benefit is unobserved. The report instead shows T-versus-X top-decile overlap: {percent(consistency[consistency['model'].eq('XGBoost')]['top_decile_overlap_pct'].iloc[0])} for XGBoost. This is a stability diagnostic, not a truth benchmark.

## Level 2 Summary: Uplift Model Validation

The Oregon analysis produces clear predicted score gradients but mixed empirical validation. The XGBoost T-Learner's top predicted magnitude is compatible with its wide observed interval, yet its observed gaps are not monotonic. The X-Learner's first decile has a positive observed-gap interval, but the two frameworks overlap on only about one-third of top-decile records. These findings support a targeted prospective pilot and do not support interpreting the scores as proven individual causal effects.

# Evaluation Level 3: Operational Evaluation

## Analytical Task 6: Variable Importance and Explainability

### Risk Drivers

The treated and control factual models emphasize related but not identical risk drivers. These are predictors of 90-day ED outcome under each observed treatment condition; they are not automatically drivers of treatment benefit.

<!-- AUTO_TABLE: oregon_factual_shap_treated -->
{table(['Rank', 'Treated-model feature', 'Mean absolute SHAP'], [[i, r['feature'], number(r['mean_abs_shap'])] for i, (_, r) in enumerate(t_model_shap['Treated Model'].iterrows(), 1)])}

<!-- AUTO_TABLE: oregon_factual_shap_control -->
{table(['Rank', 'Control-model feature', 'Mean absolute SHAP'], [[i, r['feature'], number(r['mean_abs_shap'])] for i, (_, r) in enumerate(t_model_shap['Control Model'].iterrows(), 1)])}

{side_by_side('T-Learner/XGBoost/dashboard_shap_treated_model.png', 'Treated factual model SHAP', 'T-Learner/XGBoost/dashboard_shap_control_model.png', 'Control factual model SHAP')}

### Explainability Approaches

The report uses model-specific explanation methods. XGBoost uses TreeSHAP-derived contributions; GLMNet uses exact linear contributions on the transformed model matrix. Absolute values show importance, signed values show average direction in the modeled benefit score, and the positive share shows how often the feature contribution raises predicted benefit. None of these quantities implies that intervening on the feature would change benefit.

### Coefficient-Based Benefit Contributions

<!-- AUTO_TABLE: oregon_glmnet_benefit_contributions -->
{table(['Rank', 'Feature', 'Mean absolute contribution', 'Mean signed contribution', 'Positive contribution share'], shap_rows(glm_benefit, 10))}

GLMNet provides a transparent sensitivity view, but its top-decile benefit magnitude was outside the observed interval. Coefficients and contributions should therefore be read as model mechanics rather than causal effect modifiers.

### SHAP Benefit-Score Contributions

<!-- AUTO_TABLE: oregon_tlearner_benefit_shap -->
{table(['Rank', 'T-Learner feature', 'Mean absolute benefit SHAP', 'Mean signed SHAP', 'Positive SHAP share'], shap_rows(t_shap_benefit, 12))}

<!-- AUTO_TABLE: oregon_xlearner_benefit_shap -->
{table(['Rank', 'X-Learner feature', 'Mean absolute benefit SHAP', 'Mean signed SHAP', 'Positive SHAP share'], shap_rows(x_benefit, 12))}

<!-- AUTO_CHART: oregon_benefit_shap_pair -->
{side_by_side('T-Learner/XGBoost/dashboard_shap_benefit_score.png', 'T-Learner benefit-score SHAP', 'X-Learner/XGBoost/dashboard_xlearner_benefit_drivers.png', 'X-Learner benefit-score SHAP')}

Recent ED utilization, recent total cost, percolator score, age, and current risk score dominate the T-Learner benefit explanation. The X-Learner also emphasizes recent ED use, total cost, and age, though at a smaller contribution scale. The overlap is reassuring at the population level; the limited top-decile member overlap shows that similar global drivers do not imply identical individual rankings.

### Known Synthetic Driver Alignment

There are no known synthetic treatment-effect drivers in the Oregon observational data. Accordingly, the report does not claim alignment to a known causal data-generating process. The closest available check is cross-framework agreement on broad driver families, which is descriptive only.

## Analytical Task 7: Business Value Assessment

As in the Funds report, the scenario assigns **$1,200 gross value per modeled ED event avoided** and **$250 intervention cost per targeted record**. These are illustrative assumptions, not measured Oregon allowed amounts or program costs. Gross savings are shown separately from intervention cost, and results inherit all uncertainty in the benefit scores.

### XGBoost T-Learner Targeting

<!-- AUTO_TABLE: oregon_tlearner_roi -->
{table(['Decile', 'N', 'Expected ED-rate reduction', 'Expected ED events avoided', 'Gross savings', 'Intervention cost', 'Net savings', 'ROI'], roi_rows(t_roi))}

The complete top decile has {integer(t_top['top_decile_n'])} records, {number(t_top['top_decile_estimated_ed_visits_avoided'],2)} modeled ED events avoided, {dollars(t_top['top_decile_gross_savings'])} gross savings, and {dollars(t_top['top_decile_net_savings'])} net savings. Its illustrative ROI is {percent(t_top['top_decile_roi'])}, so the full decile does not clear the assumed $250 per-person intervention cost.

{image('T-Learner/XGBoost/dashboard_roi_net_savings_by_decile.png', 'T-Learner net savings by decile')}

The Funds-style cumulative comparison ranks the same held-out test population either by modeled T-Learner benefit or by current risk. Avoided events are always summed from the T-Learner benefit score so only the targeting order changes.

<!-- AUTO_TABLE: oregon_tlearner_cumulative_targeting -->
{table(['Targeting approach', 'Through decile', 'Population targeted', 'N', 'Modeled ED events avoided', 'Cumulative gross savings'], targeting_rows(t_cumulative))}

The ROI table follows the notebook's decile assignment, whose first bin contains 39 records. The cumulative Funds-style comparison uses fixed 38-record increments (`382 // 10`) and places all remaining records in the final step. This intentional convention explains the small difference between first-decile gross savings in the two displays.

<!-- AUTO_CHART: oregon_tlearner_targeting_pair -->
{side_by_side('T-Learner/XGBoost/dashboard_cumulative_gross_savings_targeting.png', 'T-Learner cumulative targeting value', 'T-Learner/XGBoost/dashboard_marginal_gross_savings_advantage_vs_current_risk.png', 'T-Learner marginal advantage versus risk')}

At approximately 50% of the test population, T-Learner uplift ranking produces {dollars(t_cumulative[(t_cumulative['targeting_approach'].eq('Uplift score')) & (t_cumulative['through_decile'].eq(5))]['cumulative_gross_savings'].iloc[0])} in modeled gross savings, compared with {dollars(t_cumulative[(t_cumulative['targeting_approach'].eq('Current risk score')) & (t_cumulative['through_decile'].eq(5))]['cumulative_gross_savings'].iloc[0])} under current-risk ranking.

### XGBoost X-Learner Targeting

<!-- AUTO_TABLE: oregon_xlearner_roi -->
{table(['Decile', 'N', 'Expected ED-rate reduction', 'Expected ED events avoided', 'Gross savings', 'Intervention cost', 'Net savings', 'ROI'], roi_rows(x_roi))}

{image('X-Learner/XGBoost/dashboard_xlearner_roi_net_savings_by_decile.png', 'X-Learner net savings by decile')}

<!-- AUTO_TABLE: oregon_xlearner_cumulative_targeting -->
{table(['Targeting approach', 'Through decile', 'Population targeted', 'N', 'Modeled ED events avoided', 'Cumulative gross savings'], targeting_rows(x_cumulative))}

<!-- AUTO_CHART: oregon_xlearner_targeting_pair -->
{side_by_side('X-Learner/XGBoost/dashboard_cumulative_gross_savings_targeting.png', 'X-Learner cumulative targeting value', 'X-Learner/XGBoost/dashboard_marginal_gross_savings_advantage_vs_current_risk.png', 'X-Learner marginal advantage versus risk')}

The X-Learner shows stronger modeled value in its first few deciles than the T-Learner, but later marginal value turns negative. Because the frameworks prioritize substantially different records, the apparent economic advantage is model-dependent and must be validated prospectively before it is treated as expected savings.

## Level 3 Summary: Operational Evaluation

Recent ED use, cost, percolator score, and age are prominent in the Oregon benefit models. Uplift-based ranking generates a different allocation from current-risk ranking and can concentrate modeled gross value earlier in the targeting curve. However, the full XGBoost T-Learner top decile has negative net value under the retained cost assumptions, while the X-Learner is more favorable. That disagreement, together with noisy observed decile gaps, makes a controlled pilot with Oregon-specific cost inputs the appropriate next decision point.

---

Supporting files: [executed Oregon notebook](Code/Uplift%20Model%20Code_Oregon_2YearDataset.ipynb), [preprocessing audit](Outputs/Uplift_Oregon/Python/preprocessing_audit_summary.csv), [column-level leakage audit](Outputs/Uplift_Oregon/Python/preprocessing_column_audit.csv), [model evaluation summary](Outputs/Uplift_Oregon/Python/model_evaluation_summary.csv), and [output parity manifest](Outputs/Uplift_Oregon/Python/output_parity_manifest.csv).
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
