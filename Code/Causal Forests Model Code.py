"""Causal forest model.

Python equivalent of ``Causal Forests Model Code.R``.

This script uses ``econml``'s CausalForestDML, which is the closest common
Python equivalent to R's ``grf::causal_forest``. Install the causal-forest
dependencies before running this script:

    pip install econml scikit-learn
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _prism_model_utils import (
    add_date_features,
    clean_names_simple,
    ensure_output_folder,
    make_design_matrix,
    ntile_desc,
    prepare_model_frame,
    print_distribution,
    read_prism_excel,
    require_columns,
    split_train_test,
    to_binary,
)


PREDICTOR_VARS = [
    "age",
    "gender",
    "dual_eligible",
    "diabetes_flag",
    "chf_flag",
    "copd_flag",
    "asthma_flag",
    "depression_flag",
    "anxiety_flag",
    "substance_use_flag",
    "ckd_flag",
    "food_insecurity_flag",
    "housing_instability_flag",
    "transportation_barrier_flag",
    "ed_visits_last_30d",
    "ed_visits_last_6m",
    "admits_last_6m",
    "total_cost_last_6m",
    "percolator_utilization_score",
    "percolator_clinical_score",
    "current_risk_score",
    "touches_per_month",
    "outreach_attempts",
    "successful_contacts",
    "avg_call_duration_min",
    "community_referral_flag",
    "pharmacy_review_flag",
    "engagement_level",
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


def load_causal_forest_dependencies():
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing causal-forest Python dependencies. Install them with:\n"
            "  pip install econml scikit-learn\n"
            f"Original import error: {exc}"
        ) from exc
    return CausalForestDML, RandomForestClassifier, RandomForestRegressor


def effect_standard_errors(cf_model, x_matrix: pd.DataFrame) -> np.ndarray:
    try:
        inference = cf_model.effect_inference(x_matrix)
        return np.asarray(inference.stderr, dtype=float)
    except Exception:
        return np.full(len(x_matrix), np.nan)


def main() -> None:
    CausalForestDML, RandomForestClassifier, RandomForestRegressor = load_causal_forest_dependencies()

    output_folder = ensure_output_folder(Path("Outputs") / "Causal-Forests")
    output_path = output_folder / "causal_forest_scored_output.csv"
    summary_path = output_folder / "causal_forest_decile_summary.csv"
    importance_path = output_folder / "causal_forest_variable_importance.csv"

    df = read_prism_excel()
    df.columns = clean_names_simple(df.columns)
    require_columns(df, ["outcome_ed_90d", "intervention_flag"])

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}\n")
    print(list(df.columns))

    df = df.copy()
    df["outcome_ed_90d"] = to_binary(df["outcome_ed_90d"])
    df["intervention_flag"] = to_binary(df["intervention_flag"])
    print_distribution("Outcome distribution", df["outcome_ed_90d"])
    print_distribution("Treatment distribution", df["intervention_flag"])

    df = add_date_features(df, include_duration=True)
    model_df = prepare_model_frame(
        df,
        PREDICTOR_VARS,
        NUMERIC_VARS,
        binary_extra=["engaged", "opted_out", "dual_eligible"],
    )

    feature_cols = [col for col in model_df.columns if col not in ["outcome_ed_90d", "intervention_flag"]]
    design_matrix, [x_all] = make_design_matrix([model_df[feature_cols]])
    x_all.columns = design_matrix.columns

    y_all = model_df["outcome_ed_90d"].astype(float).to_numpy()
    w_all = model_df["intervention_flag"].astype(float).to_numpy()

    observed_y = set(pd.Series(y_all).dropna().unique())
    observed_w = set(pd.Series(w_all).dropna().unique())
    if not observed_y.issubset({0.0, 1.0}):
        raise ValueError("Outcome contains values other than 0 and 1.")
    if not observed_w.issubset({0.0, 1.0}):
        raise ValueError("Treatment contains values other than 0 and 1.")

    print("Final model matrix dimensions:")
    print(x_all.shape)

    train_df, test_df = split_train_test(model_df, train_fraction=0.70, seed=123)
    train_index = train_df.index
    test_index = test_df.index

    x_train = x_all.loc[train_index].copy()
    y_train = model_df.loc[train_index, "outcome_ed_90d"].astype(float).to_numpy()
    w_train = model_df.loc[train_index, "intervention_flag"].astype(float).to_numpy()

    x_test = x_all.loc[test_index].copy()

    cf_model = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=123,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=123,
            n_jobs=-1,
        ),
        discrete_treatment=True,
        n_estimators=2000,
        min_samples_leaf=5,
        random_state=123,
        inference=True,
    )
    cf_model.fit(y_train, w_train, X=x_train)
    print("Causal forest trained successfully.\n")

    try:
        ate = cf_model.ate(x_train)
        ate_interval = cf_model.ate_interval(x_train)
        print("Average Treatment Effect estimate:")
        print({"ate": float(ate), "lower": float(ate_interval[0]), "upper": float(ate_interval[1])})
        print()
    except Exception as exc:
        print(f"Average Treatment Effect estimate unavailable: {exc}\n")

    tau_test = np.asarray(cf_model.effect(x_test), dtype=float)
    tau_se_test = effect_standard_errors(cf_model, x_test)

    results_test = model_df.loc[test_index].copy()
    results_test["tau_hat"] = tau_test
    results_test["tau_se"] = tau_se_test
    results_test["benefit_score"] = -results_test["tau_hat"]
    results_test["uplift_decile"] = ntile_desc(results_test["benefit_score"], 10)

    decile_summary = (
        results_test.groupby("uplift_decile", as_index=False)
        .agg(
            n=("uplift_decile", "size"),
            avg_tau_hat=("tau_hat", "mean"),
            avg_benefit_score=("benefit_score", "mean"),
            observed_ed_rate=("outcome_ed_90d", "mean"),
            treated_pct=("intervention_flag", "mean"),
            avg_tau_se=("tau_se", "mean"),
        )
        .sort_values("uplift_decile")
    )
    print("Decile summary:")
    print(decile_summary)

    importances = getattr(cf_model, "feature_importances_", np.full(x_train.shape[1], np.nan))
    importance_df = (
        pd.DataFrame({"feature": x_train.columns, "importance": np.asarray(importances, dtype=float)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("Top variable importance:")
    print(importance_df.head(25))

    tau_full = np.asarray(cf_model.effect(x_all), dtype=float)
    tau_se_full = effect_standard_errors(cf_model, x_all)
    scored_full = model_df.copy()
    scored_full["tau_hat"] = tau_full
    scored_full["tau_se"] = tau_se_full
    scored_full["benefit_score"] = -scored_full["tau_hat"]
    scored_full["uplift_decile"] = ntile_desc(scored_full["benefit_score"], 10)

    scored_full.to_csv(output_path, index=False)
    decile_summary.to_csv(summary_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    print(f"Scored output saved to:\n{output_path}\n")
    print(f"Decile summary saved to:\n{summary_path}\n")
    print(f"Variable importance saved to:\n{importance_path}\n")

    print("INTERPRETATION:")
    print("- tau_hat = estimated treatment effect on outcome_ed_90d")
    print("- Because outcome_ed_90d is bad, negative tau_hat means intervention reduced ED risk")
    print("- benefit_score = -tau_hat")
    print("- Higher benefit_score means larger estimated ED reduction from intervention")
    print("- uplift_decile 1 = highest estimated intervention benefit")


if __name__ == "__main__":
    main()
