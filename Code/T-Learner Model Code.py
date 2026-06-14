"""T-learner uplift model with XGBoost.

Python equivalent of ``T-Learner Model Code.R``.
"""

from __future__ import annotations

from pathlib import Path

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
    present_columns,
    print_distribution,
    read_prism_excel,
    require_columns,
    to_binary,
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


def main() -> None:
    output_folder = ensure_output_folder(Path("Outputs") / "T-Learner")
    output_path = output_folder / "t_learner_scored_output.csv"
    summary_path = output_folder / "t_learner_decile_summary.csv"

    df = read_prism_excel()
    df.columns = clean_names_simple(df.columns)
    require_columns(df, ["outcome_ed_90d", "intervention_flag"])

    df = df.copy()
    df["outcome_ed_90d"] = to_binary(df["outcome_ed_90d"])
    df["intervention_flag"] = to_binary(df["intervention_flag"])

    print_distribution("Outcome distribution", df["outcome_ed_90d"])
    print_distribution("Treatment distribution", df["intervention_flag"])

    df = add_date_features(df, include_duration=False)
    model_df = prepare_model_frame(df, PREDICTOR_VARS, NUMERIC_VARS)

    print(f"Modeling rows after dropping missing outcome/treatment: {len(model_df)}\n")
    print("Final modeling columns:")
    print(list(model_df.columns))
    print()

    from _prism_model_utils import split_train_test

    train_df, test_df = split_train_test(model_df, train_fraction=0.70, seed=123)
    train_treated = train_df[train_df["intervention_flag"] == 1].copy()
    train_control = train_df[train_df["intervention_flag"] == 0].copy()

    print(f"Training treated rows: {len(train_treated)}")
    print(f"Training control rows: {len(train_control)}")

    feature_cols = [col for col in model_df.columns if col not in ["outcome_ed_90d", "intervention_flag"]]
    _, matrices = make_design_matrix(
        [
            train_treated[feature_cols],
            train_control[feature_cols],
            test_df[feature_cols],
        ]
    )
    x_treated, x_control, x_test = matrices

    y_treated = train_treated["outcome_ed_90d"].astype(float).to_numpy()
    y_control = train_control["outcome_ed_90d"].astype(float).to_numpy()

    model_treated = fit_xgb_binary(x_treated, y_treated, nrounds=150, seed=123)
    model_control = fit_xgb_binary(x_control, y_control, nrounds=150, seed=123)
    print("T-Learner models trained successfully.")

    p_treated = predict_xgb(model_treated, x_test)
    p_control = predict_xgb(model_control, x_test)

    results_test = test_df.copy()
    results_test["pred_ed_if_treated"] = p_treated
    results_test["pred_ed_if_control"] = p_control
    results_test["benefit_score"] = results_test["pred_ed_if_control"] - results_test["pred_ed_if_treated"]
    results_test["treatment_effect_bad_outcome"] = (
        results_test["pred_ed_if_treated"] - results_test["pred_ed_if_control"]
    )
    results_test["uplift_decile"] = ntile_desc(results_test["benefit_score"], 10)

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

    print(decile_summary)

    full_x_df = model_df[feature_cols]
    _, [full_matrix] = make_design_matrix([full_x_df])
    full_matrix = full_matrix.reindex(columns=x_test.columns, fill_value=0.0)

    scored_full = model_df.copy()
    scored_full["pred_ed_if_treated"] = predict_xgb(model_treated, full_matrix)
    scored_full["pred_ed_if_control"] = predict_xgb(model_control, full_matrix)
    scored_full["benefit_score"] = scored_full["pred_ed_if_control"] - scored_full["pred_ed_if_treated"]
    scored_full["treatment_effect_bad_outcome"] = (
        scored_full["pred_ed_if_treated"] - scored_full["pred_ed_if_control"]
    )
    scored_full["uplift_decile"] = ntile_desc(scored_full["benefit_score"], 10)

    scored_full.to_csv(output_path, index=False)
    decile_summary.to_csv(summary_path, index=False)

    print(f"Scored output saved to:\n{output_path}")
    print(f"Decile summary saved to:\n{summary_path}")
    print("\nINTERPRETATION:")
    print("- pred_ed_if_treated = predicted ED probability if treated")
    print("- pred_ed_if_control = predicted ED probability if untreated")
    print("- benefit_score = pred_ed_if_control - pred_ed_if_treated")
    print("- Higher benefit_score = greater expected ED reduction from intervention")
    print("- uplift_decile 1 = highest expected treatment benefit")


if __name__ == "__main__":
    main()
