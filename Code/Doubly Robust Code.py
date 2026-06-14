"""Doubly robust off-policy evaluation.

Python equivalent of ``Doubly Robust Code.R``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _prism_model_utils import (
    add_date_features,
    clean_names_simple,
    clip_probs,
    ensure_output_folder,
    fit_xgb_binary,
    make_design_matrix,
    predict_xgb,
    prepare_model_frame,
    print_distribution,
    read_prism_excel,
    require_columns,
    split_train_test,
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
    "engaged",
    "opted_out",
    "engagement_length",
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
    "engagement_length",
    "days_to_intervention_start",
    "intervention_start_month",
    "intervention_start_wday",
]


def dr_policy_value(
    y: np.ndarray,
    w: np.ndarray,
    e: np.ndarray,
    m1: np.ndarray,
    m0: np.ndarray,
    policy: np.ndarray,
) -> float:
    e = clip_probs(e)
    value_i = policy * (m1 + (w / e) * (y - m1)) + (1 - policy) * (
        m0 + ((1 - w) / (1 - e)) * (y - m0)
    )
    return float(np.nanmean(value_i))


def main() -> None:
    output_folder = ensure_output_folder(Path("Outputs") / "Doubly-Robust")
    output_path = output_folder / "doubly_robust_policy_evaluation.csv"
    scored_output_path = output_folder / "doubly_robust_scored_members.csv"

    df = read_prism_excel()
    df.columns = clean_names_simple(df.columns)
    require_columns(df, ["outcome_ed_90d", "intervention_flag"])

    df = df.copy()
    df["outcome_ed_90d"] = to_binary(df["outcome_ed_90d"])
    df["intervention_flag"] = to_binary(df["intervention_flag"])

    print_distribution("Outcome distribution", df["outcome_ed_90d"])
    print_distribution("Treatment distribution", df["intervention_flag"])

    df = add_date_features(df, include_duration=False)
    model_df = prepare_model_frame(
        df,
        PREDICTOR_VARS,
        NUMERIC_VARS,
        binary_extra=["engaged", "opted_out", "dual_eligible"],
    )

    train_df, test_df = split_train_test(model_df, train_fraction=0.70, seed=123)
    feature_cols = [col for col in model_df.columns if col not in ["outcome_ed_90d", "intervention_flag"]]

    _, [x_train, x_test] = make_design_matrix([train_df[feature_cols], test_df[feature_cols]])

    y_train = train_df["outcome_ed_90d"].astype(float).to_numpy()
    w_train = train_df["intervention_flag"].astype(float).to_numpy()
    y_test = test_df["outcome_ed_90d"].astype(float).to_numpy()
    w_test = test_df["intervention_flag"].astype(float).to_numpy()

    propensity_model = fit_xgb_binary(x_train, w_train, nrounds=150, seed=123)

    treated_mask = (train_df["intervention_flag"] == 1).to_numpy()
    control_mask = (train_df["intervention_flag"] == 0).to_numpy()
    if treated_mask.sum() < 50:
        raise ValueError("Too few treated rows to train a stable outcome model.")
    if control_mask.sum() < 50:
        raise ValueError("Too few control rows to train a stable outcome model.")

    outcome_model_treated = fit_xgb_binary(x_train.iloc[treated_mask], y_train[treated_mask], seed=123)
    outcome_model_control = fit_xgb_binary(x_train.iloc[control_mask], y_train[control_mask], seed=123)

    e_hat = clip_probs(predict_xgb(propensity_model, x_test))
    m1_hat = predict_xgb(outcome_model_treated, x_test)
    m0_hat = predict_xgb(outcome_model_control, x_test)

    scored_test = test_df.copy()
    scored_test["propensity_score"] = e_hat
    scored_test["pred_ed_if_treated"] = m1_hat
    scored_test["pred_ed_if_control"] = m0_hat
    scored_test["benefit_score"] = scored_test["pred_ed_if_control"] - scored_test["pred_ed_if_treated"]
    scored_test["benefit_rank"] = scored_test["benefit_score"].rank(method="first", ascending=False)
    scored_test["benefit_percentile"] = scored_test["benefit_rank"] / len(scored_test)

    policy_historical = w_test
    policy_treat_none = np.zeros_like(w_test)
    policy_treat_all = np.ones_like(w_test)
    policy_top_10 = (scored_test["benefit_percentile"].to_numpy() <= 0.10).astype(float)
    policy_top_20 = (scored_test["benefit_percentile"].to_numpy() <= 0.20).astype(float)
    policy_top_30 = (scored_test["benefit_percentile"].to_numpy() <= 0.30).astype(float)
    policy_top_40 = (scored_test["benefit_percentile"].to_numpy() <= 0.40).astype(float)

    policy_names = [
        "Historical observed policy",
        "Treat nobody",
        "Treat everybody",
        "Treat top 10% by benefit score",
        "Treat top 20% by benefit score",
        "Treat top 30% by benefit score",
        "Treat top 40% by benefit score",
    ]
    policies = [
        policy_historical,
        policy_treat_none,
        policy_treat_all,
        policy_top_10,
        policy_top_20,
        policy_top_30,
        policy_top_40,
    ]

    policy_results = pd.DataFrame(
        {
            "policy": policy_names,
            "treatment_rate": [float(np.mean(policy)) for policy in policies],
            "estimated_ed_rate": [
                dr_policy_value(y_test, w_test, e_hat, m1_hat, m0_hat, policy)
                for policy in policies
            ],
        }
    )

    historical_ed_rate = float(
        policy_results.loc[
            policy_results["policy"] == "Historical observed policy", "estimated_ed_rate"
        ].iloc[0]
    )
    policy_results["estimated_ed_rate_reduction_vs_historical"] = (
        historical_ed_rate - policy_results["estimated_ed_rate"]
    )
    policy_results["expected_ed_visits_avoided_per_1000"] = (
        policy_results["estimated_ed_rate_reduction_vs_historical"] * 1000
    )
    policy_results = policy_results.sort_values("estimated_ed_rate").reset_index(drop=True)
    print(policy_results)

    cost_per_ed_visit = 1200
    cost_per_intervention = 250
    policy_results["expected_ed_savings_per_1000"] = (
        policy_results["expected_ed_visits_avoided_per_1000"] * cost_per_ed_visit
    )
    policy_results["intervention_cost_per_1000"] = (
        policy_results["treatment_rate"] * 1000 * cost_per_intervention
    )
    policy_results["net_savings_per_1000"] = (
        policy_results["expected_ed_savings_per_1000"]
        - policy_results["intervention_cost_per_1000"]
    )
    print(policy_results)

    policy_results.to_csv(output_path, index=False)
    scored_test.to_csv(scored_output_path, index=False)

    print(f"Policy evaluation output saved to:\n{output_path}\n")
    print(f"Scored member output saved to:\n{scored_output_path}\n")
    print("INTERPRETATION:")
    print("- estimated_ed_rate = doubly robust estimated ED rate under that policy")
    print("- Lower estimated_ed_rate is better")
    print("- expected_ed_visits_avoided_per_1000 compares each policy to historical targeting")
    print("- net_savings_per_1000 applies your cost assumptions")


if __name__ == "__main__":
    main()
