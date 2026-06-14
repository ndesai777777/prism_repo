"""Expanded uplift model workflow with XGBoost, dashboards, ROI, and SHAP.

Python equivalent of ``uplift code.R``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _prism_model_utils import (
    add_date_features,
    clean_names_simple,
    ensure_output_folder,
    fit_xgb_binary,
    make_design_matrix,
    ntile_desc,
    predict_xgb,
    prepare_model_frame,
    print_distribution,
    read_prism_excel,
    require_columns,
    shap_importance_frame,
    split_train_test,
    to_binary,
    xgb_importance_frame,
)


PREDICTOR_VARS = [
    "client_contract",
    "service_region",
    "program",
    "case_manager_name",
    "age",
    "gender",
    "dual_eligible",
    "county",
    "plan_type",
    "language",
    "living_alone_flag",
    "diabetes_flag",
    "chf_flag",
    "copd_flag",
    "asthma_flag",
    "depression_flag",
    "anxiety_flag",
    "substance_use_flag",
    "ckd_flag",
    "pregnancy_flag",
    "behavioral_health_risk_flag",
    "food_insecurity_flag",
    "housing_instability_flag",
    "transportation_barrier_flag",
    "utilities_insecurity_flag",
    "pcp_visits_last_6m",
    "specialist_visits_last_6m",
    "ed_visits_last_30d",
    "ed_visits_last_6m",
    "admits_last_6m",
    "observation_stays_last_6m",
    "total_cost_last_6m",
    "rx_count_last_6m",
    "med_adherence_pdc",
    "high_cost_drug_flag",
    "opioid_flag",
    "polypharmacy_flag",
    "percolator_utilization_score",
    "percolator_clinical_score",
    "percolator_sdoh_score",
    "current_risk_score",
    "risk_tier",
    "intervention_type",
    "intervention_days_active",
    "touches_per_month",
    "outreach_attempts",
    "successful_contacts",
    "avg_call_duration_min",
    "max_call_duration_min",
    "notes_escalation_flag",
    "community_referral_flag",
    "pharmacy_review_flag",
    "engagement_level",
    "days_to_intervention_start",
    "intervention_start_month",
    "intervention_start_wday",
]


NUMERIC_VARS = [
    "age",
    "pcp_visits_last_6m",
    "specialist_visits_last_6m",
    "ed_visits_last_30d",
    "ed_visits_last_6m",
    "admits_last_6m",
    "observation_stays_last_6m",
    "total_cost_last_6m",
    "rx_count_last_6m",
    "med_adherence_pdc",
    "percolator_utilization_score",
    "percolator_clinical_score",
    "percolator_sdoh_score",
    "current_risk_score",
    "intervention_days_active",
    "touches_per_month",
    "outreach_attempts",
    "successful_contacts",
    "avg_call_duration_min",
    "max_call_duration_min",
    "days_to_intervention_start",
    "intervention_start_month",
    "intervention_start_wday",
]


def save_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    path: Path,
    width: float = 8,
    height: float = 5,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    output_folder = ensure_output_folder(Path("Outputs") / "Uplift")
    output_path = output_folder / "uplift_scored_output.csv"
    summary_path = output_folder / "uplift_decile_summary.csv"

    df = read_prism_excel()
    df.columns = clean_names_simple(df.columns)
    require_columns(df, ["outcome_ed_90d", "intervention_flag"])

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}\n")
    print("Column names after cleaning:")
    print(list(df.columns))
    print()

    df = df.copy()
    df["intervention_flag"] = to_binary(df["intervention_flag"])
    df["outcome_ed_90d"] = to_binary(df["outcome_ed_90d"])
    print_distribution("Outcome distribution", df["outcome_ed_90d"])
    print_distribution("Treatment distribution", df["intervention_flag"])

    df = add_date_features(df, include_duration=True)
    model_df = prepare_model_frame(df, PREDICTOR_VARS, NUMERIC_VARS)

    print(f"Modeling rows after dropping missing outcome/treatment: {len(model_df)}\n")
    print("Final modeling columns:")
    print(list(model_df.columns))
    print()

    train_df, test_df = split_train_test(model_df, train_fraction=0.70, seed=123)
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}\n")

    train_treated = train_df[train_df["intervention_flag"] == 1].copy()
    train_control = train_df[train_df["intervention_flag"] == 0].copy()
    print(f"Training treated rows: {len(train_treated)}")
    print(f"Training control rows: {len(train_control)}\n")

    if len(train_treated) < 50:
        raise ValueError("Too few treated rows to train a stable model.")
    if len(train_control) < 50:
        raise ValueError("Too few control rows to train a stable model.")

    feature_cols = [col for col in model_df.columns if col not in ["outcome_ed_90d", "intervention_flag"]]
    combined_matrix, matrices = make_design_matrix(
        [
            train_treated[feature_cols],
            train_control[feature_cols],
            test_df[feature_cols],
        ]
    )
    x_treated, x_control, x_test = matrices
    model_columns = list(combined_matrix.columns)

    y_treated = train_treated["outcome_ed_90d"].astype(float).to_numpy()
    y_control = train_control["outcome_ed_90d"].astype(float).to_numpy()

    print("Unique y_treated values:")
    print(sorted(pd.Series(y_treated).unique()))
    print("\nUnique y_control values:")
    print(sorted(pd.Series(y_control).unique()))
    print()

    model_treated = fit_xgb_binary(x_treated, y_treated, nrounds=150, seed=123)
    model_control = fit_xgb_binary(x_control, y_control, nrounds=150, seed=123)
    print("Models trained successfully.\n")

    p_treated = predict_xgb(model_treated, x_test)
    p_control = predict_xgb(model_control, x_test)

    results_test = test_df.copy()
    results_test["pred_ed_if_treated"] = p_treated
    results_test["pred_ed_if_control"] = p_control
    results_test["benefit_score"] = results_test["pred_ed_if_control"] - results_test["pred_ed_if_treated"]
    results_test["uplift_bad_outcome"] = results_test["pred_ed_if_treated"] - results_test["pred_ed_if_control"]
    results_test["uplift_decile"] = ntile_desc(results_test["benefit_score"], 10)

    print("Top 20 highest-benefit members:")
    print(
        results_test.sort_values("benefit_score", ascending=False)[
            [
                "outcome_ed_90d",
                "intervention_flag",
                "pred_ed_if_treated",
                "pred_ed_if_control",
                "benefit_score",
                "uplift_decile",
            ]
        ].head(20)
    )
    print()

    decile_summary = (
        results_test.groupby("uplift_decile", as_index=False)
        .agg(
            n=("uplift_decile", "size"),
            avg_benefit_score=("benefit_score", "mean"),
            observed_ed_rate=("outcome_ed_90d", "mean"),
            treated_pct=("intervention_flag", "mean"),
            avg_pred_ed_if_treated=("pred_ed_if_treated", "mean"),
            avg_pred_ed_if_control=("pred_ed_if_control", "mean"),
        )
        .sort_values("uplift_decile")
    )
    print("Decile summary:")
    print(decile_summary)
    print()

    importance_treated = xgb_importance_frame(model_treated)
    importance_control = xgb_importance_frame(model_control)
    print("Top variables in treated model:")
    print(importance_treated.head(20))
    print("\nTop variables in control model:")
    print(importance_control.head(20))
    print()

    full_x_df = model_df[feature_cols]
    _, [full_matrix] = make_design_matrix([full_x_df])
    full_matrix = full_matrix.reindex(columns=model_columns, fill_value=0.0)

    scored_full = model_df.copy()
    scored_full["pred_ed_if_treated"] = predict_xgb(model_treated, full_matrix)
    scored_full["pred_ed_if_control"] = predict_xgb(model_control, full_matrix)
    scored_full["benefit_score"] = scored_full["pred_ed_if_control"] - scored_full["pred_ed_if_treated"]
    scored_full["uplift_bad_outcome"] = scored_full["pred_ed_if_treated"] - scored_full["pred_ed_if_control"]
    scored_full["uplift_decile"] = ntile_desc(scored_full["benefit_score"], 10)

    scored_full.to_csv(output_path, index=False)
    decile_summary.to_csv(summary_path, index=False)
    print(f"Scored full file written to:\n{output_path}\n")
    print(f"Decile summary written to:\n{summary_path}\n")

    print("INTERPRETATION:")
    print("- pred_ed_if_treated = predicted probability of ED within 90d if treated")
    print("- pred_ed_if_control = predicted probability of ED within 90d if not treated")
    print("- benefit_score = pred_ed_if_control - pred_ed_if_treated")
    print("- Higher benefit_score means treatment is predicted to reduce ED risk more")
    print("- Uplift decile 1 = highest predicted treatment benefit")

    save_bar_chart(
        decile_summary,
        "uplift_decile",
        "avg_benefit_score",
        "Average Predicted Intervention Benefit by Uplift Decile",
        "Uplift Decile: 1 = Highest Predicted Benefit",
        "Average Benefit Score",
        output_folder / "dashboard_avg_benefit_by_decile.png",
    )
    save_bar_chart(
        decile_summary,
        "uplift_decile",
        "observed_ed_rate",
        "Observed 90-Day ED Rate by Uplift Decile",
        "Uplift Decile",
        "Observed ED Rate",
        output_folder / "dashboard_observed_ed_rate_by_decile.png",
    )
    save_bar_chart(
        decile_summary,
        "uplift_decile",
        "treated_pct",
        "Current Treatment Penetration by Uplift Decile",
        "Uplift Decile",
        "Percent Treated",
        output_folder / "dashboard_treated_pct_by_decile.png",
    )

    decile_long = decile_summary.melt(
        id_vars="uplift_decile",
        value_vars=["avg_pred_ed_if_treated", "avg_pred_ed_if_control"],
        var_name="scenario",
        value_name="predicted_ed_rate",
    )
    pivot = decile_long.pivot(index="uplift_decile", columns="scenario", values="predicted_ed_rate")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Predicted ED Risk: Treated vs Control by Decile")
    ax.set_xlabel("Uplift Decile")
    ax.set_ylabel("Predicted ED Rate")
    ax.figure.tight_layout()
    ax.figure.savefig(output_folder / "dashboard_predicted_treated_vs_control.png", dpi=150)
    plt.close(ax.figure)
    print(f"Dashboard charts saved to:\n{output_folder}\n")

    cost_per_ed_visit = 1200
    cost_per_intervention = 250
    roi_summary = decile_summary.copy()
    roi_summary["expected_ed_rate_reduction"] = roi_summary["avg_benefit_score"]
    roi_summary["expected_ed_visits_avoided"] = (
        roi_summary["n"] * roi_summary["expected_ed_rate_reduction"]
    )
    roi_summary["gross_savings"] = roi_summary["expected_ed_visits_avoided"] * cost_per_ed_visit
    roi_summary["intervention_cost"] = roi_summary["n"] * cost_per_intervention
    roi_summary["net_savings"] = roi_summary["gross_savings"] - roi_summary["intervention_cost"]
    roi_summary["roi"] = roi_summary["net_savings"] / roi_summary["intervention_cost"]
    print(roi_summary)
    roi_summary.to_csv(output_folder / "uplift_roi_by_decile.csv", index=False)
    save_bar_chart(
        roi_summary,
        "uplift_decile",
        "net_savings",
        "Estimated Net Savings by Uplift Decile",
        "Uplift Decile",
        "Estimated Net Savings",
        output_folder / "dashboard_roi_net_savings_by_decile.png",
    )
    print("ROI summary saved.")

    shap_treated_importance = shap_importance_frame(model_treated, x_test, "Treated Model")
    shap_control_importance = shap_importance_frame(model_control, x_test, "Control Model")
    shap_importance_combined = pd.concat(
        [shap_treated_importance, shap_control_importance], ignore_index=True
    )
    shap_importance_combined.to_csv(
        output_folder / "shap_importance_treated_control_models.csv", index=False
    )

    top_treated = shap_treated_importance.nlargest(20, "mean_abs_shap").sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_treated["feature"], top_treated["mean_abs_shap"])
    ax.set_title("Top SHAP Drivers: Treated Model")
    ax.set_xlabel("Mean Absolute SHAP Contribution")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_folder / "dashboard_shap_treated_model.png", dpi=150)
    plt.close(fig)

    top_control = shap_control_importance.nlargest(20, "mean_abs_shap").sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_control["feature"], top_control["mean_abs_shap"])
    ax.set_title("Top SHAP Drivers: Control Model")
    ax.set_xlabel("Mean Absolute SHAP Contribution")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_folder / "dashboard_shap_control_model.png", dpi=150)
    plt.close(fig)
    print("SHAP outputs saved.")


if __name__ == "__main__":
    main()
