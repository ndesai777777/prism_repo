"""Repeated-seed sensitivity checks for the PRISM uplift workflow.

This script mirrors the main uplift notebook's T-learner setup, but writes all
diagnostics to a separate sensitivity folder so report-linked outputs are not
changed.
"""

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from _prism_model_utils import (
    clean_names_simple,
    ensure_output_folder,
    impute_categorical,
    impute_numeric,
    make_design_matrix,
    ntile_desc,
    present_columns,
    project_root,
    split_train_test,
    to_binary,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

PROJECT_ROOT = project_root()
DATA_PATH = PROJECT_ROOT / "DataSets" / "PRP_1000_full_pretreatment.xlsx"
OUTPUT_ROOT = ensure_output_folder(PROJECT_ROOT / "Outputs" / "Uplift" / "Python_Sensitivity")

SEEDS = [123, 456, 789, 101, 202]
TRAIN_FRACTION = 0.70
TOP_DECILE = 1

BASELINE_XGB_GRID = [
    {"max_depth": max_depth, "eta": eta, "min_child_weight": min_child_weight}
    for max_depth, eta, min_child_weight in product([3, 4, 5], [0.03, 0.05, 0.10], [1, 5])
]

REGULARIZED_XGB_GRID = [
    {
        "max_depth": max_depth,
        "eta": eta,
        "min_child_weight": min_child_weight,
        "lambda": reg_lambda,
        "alpha": reg_alpha,
        "subsample": 0.70,
        "colsample_bytree": 0.70,
    }
    for max_depth, eta, min_child_weight, reg_lambda, reg_alpha in product(
        [2, 3], [0.03, 0.05], [5, 10], [5.0], [1.0]
    )
]

CANDIDATE_PREDICTORS_ALL = [
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

POSSIBLE_NUMERIC_COLS = [
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


def safe_auc(y_true, y_pred):
    y_series = pd.Series(y_true).dropna()
    if y_series.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_pred))


def prepare_model_frame():
    df = pd.read_excel(DATA_PATH)
    df.columns = clean_names_simple(df.columns)
    df["intervention_flag"] = to_binary(df["intervention_flag"])
    df["outcome_ed_90d"] = to_binary(df["outcome_ed_90d"])

    candidate_predictors = [column for column in CANDIDATE_PREDICTORS_ALL if column in df.columns]
    model_df = df[["outcome_ed_90d", "intervention_flag", *candidate_predictors]].copy()
    model_df = model_df[
        model_df["outcome_ed_90d"].notna() & model_df["intervention_flag"].notna()
    ].copy()

    for column in [column for column in model_df.columns if column.endswith("_flag")]:
        model_df[column] = to_binary(model_df[column])

    for column in present_columns(POSSIBLE_NUMERIC_COLS, model_df):
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    for column in model_df.columns:
        if column in ["outcome_ed_90d", "intervention_flag"]:
            continue
        if pd.api.types.is_numeric_dtype(model_df[column]):
            model_df[column] = impute_numeric(model_df[column])
        else:
            model_df[column] = impute_categorical(model_df[column])

    unique_counts = model_df.apply(lambda column: column.dropna().nunique())
    keep_cols = list(unique_counts[unique_counts > 1].index)
    return model_df.loc[:, keep_cols].copy().reset_index(drop=True)


def make_dmatrix(x_matrix, y=None):
    if y is None:
        return xgb.DMatrix(x_matrix, feature_names=list(x_matrix.columns))
    return xgb.DMatrix(x_matrix, label=np.asarray(y, dtype=float), feature_names=list(x_matrix.columns))


def xgb_params(grid_params, seed):
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": grid_params["max_depth"],
        "eta": grid_params["eta"],
        "min_child_weight": grid_params["min_child_weight"],
        "subsample": grid_params.get("subsample", 0.8),
        "colsample_bytree": grid_params.get("colsample_bytree", 0.8),
        "lambda": grid_params.get("lambda", 1.0),
        "alpha": grid_params.get("alpha", 0.0),
        "seed": seed,
    }


def fit_xgb_cv_grid(x_matrix, y, grid, seed, nrounds_max=500, nfold=5):
    y_array = np.asarray(y, dtype=float)
    class_counts = pd.Series(y_array).value_counts()
    folds = int(min(nfold, class_counts.min())) if len(class_counts) > 1 else 0
    if folds < 2:
        raise ValueError("Need at least two outcome classes with at least two rows each for XGBoost CV.")

    dtrain = make_dmatrix(x_matrix, y_array)
    best = None
    rows = []
    for grid_params in grid:
        params = xgb_params(grid_params, seed)
        cv_result = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=nrounds_max,
            nfold=folds,
            stratified=True,
            early_stopping_rounds=20,
            seed=seed,
            verbose_eval=False,
        )
        best_iter = int(cv_result["test-auc-mean"].idxmax())
        row = {
            **grid_params,
            "best_nrounds": best_iter + 1,
            "cv_auc": float(cv_result.loc[best_iter, "test-auc-mean"]),
        }
        rows.append(row)
        if best is None or row["cv_auc"] > best["cv_auc"]:
            best = row

    model = xgb.train(
        params=xgb_params(best, seed),
        dtrain=dtrain,
        num_boost_round=int(best["best_nrounds"]),
        verbose_eval=False,
    )
    return {
        "model": model,
        "best_cv_auc": float(best["cv_auc"]),
        "best_nrounds": int(best["best_nrounds"]),
        "best_params": best,
        "search_results": pd.DataFrame(rows),
    }


def fit_glmnet(x_matrix, y, seed, prefit_scaler):
    y_array = np.asarray(y, dtype=float)
    class_counts = pd.Series(y_array).value_counts()
    folds = int(min(5, class_counts.min())) if len(class_counts) > 1 else 0
    if folds < 2:
        raise ValueError("Need at least two outcome classes with at least two rows each for GLMNet CV.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 30),
        cv=cv,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=np.round(np.arange(0, 1.01, 0.1), 1),
        scoring="roc_auc",
        max_iter=10000,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    x_scaled = prefit_scaler.transform(x_matrix)
    model.fit(x_scaled, y_array)
    scores = model.scores_[1.0]
    return {
        "model": model,
        "scaler": prefit_scaler,
        "best_cv_auc": float(scores.mean(axis=0).max()),
        "best_lambda": float(model.C_[0]),
        "best_alpha": float(model.l1_ratio_[0]),
    }


def glmnet_predict(fit, x_matrix):
    return fit["model"].predict_proba(fit["scaler"].transform(x_matrix))[:, 1]


def build_uplift_results(base_df, pred_treated, pred_control):
    results = base_df.copy()
    results["pred_ed_if_treated"] = pred_treated
    results["pred_ed_if_control"] = pred_control
    results["benefit_score"] = results["pred_ed_if_control"] - results["pred_ed_if_treated"]
    results["uplift_decile"] = ntile_desc(results["benefit_score"], 10)
    return results


def factual_metrics(test_df, pred_treated, pred_control):
    treated_mask = test_df["intervention_flag"].to_numpy() == 1
    factual_pred = np.where(treated_mask, pred_treated, pred_control)
    return {
        "test_factual_auc": safe_auc(test_df["outcome_ed_90d"], factual_pred),
        "test_factual_brier": float(brier_score_loss(test_df["outcome_ed_90d"], factual_pred)),
    }


def top_decile_summary(results, model_name, seed):
    top = results[results["uplift_decile"] == TOP_DECILE].copy()
    treated = top[top["intervention_flag"] == 1]
    control = top[top["intervention_flag"] == 0]
    return {
        "seed": seed,
        "model": model_name,
        "top_decile_n": int(len(top)),
        "top_decile_events": int(top["outcome_ed_90d"].sum()),
        "top_decile_observed_ed_rate": float(top["outcome_ed_90d"].mean()),
        "top_decile_treated_n": int(len(treated)),
        "top_decile_treated_events": int(treated["outcome_ed_90d"].sum()) if len(treated) else 0,
        "top_decile_treated_ed_rate": float(treated["outcome_ed_90d"].mean()) if len(treated) else np.nan,
        "top_decile_control_n": int(len(control)),
        "top_decile_control_events": int(control["outcome_ed_90d"].sum()) if len(control) else 0,
        "top_decile_control_ed_rate": float(control["outcome_ed_90d"].mean()) if len(control) else np.nan,
        "top_decile_avg_benefit_score": float(top["benefit_score"].mean()),
    }


def event_count_rows(seed, train_df, test_df):
    rows = []
    for split_name, frame in [("train", train_df), ("test", test_df)]:
        for treatment_value, label in [(1.0, "treated"), (0.0, "control")]:
            subgroup = frame[frame["intervention_flag"] == treatment_value]
            rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "group": label,
                    "n": int(len(subgroup)),
                    "events": int(subgroup["outcome_ed_90d"].sum()),
                    "event_rate": float(subgroup["outcome_ed_90d"].mean()) if len(subgroup) else np.nan,
                }
            )
    return rows


def model_metric_row(seed, model_name, group_name, fit, test_df, test_positions, predictions):
    y_test = test_df["outcome_ed_90d"].iloc[test_positions]
    test_auc = safe_auc(y_test, predictions)
    return {
        "seed": seed,
        "model": model_name,
        "outcome_model_group": group_name,
        "cv_auc": fit["best_cv_auc"],
        "test_auc": test_auc,
        "cv_minus_test_auc": fit["best_cv_auc"] - test_auc if pd.notna(test_auc) else np.nan,
        "best_nrounds": fit.get("best_nrounds", np.nan),
        "best_alpha": fit.get("best_alpha", np.nan),
        "best_lambda": fit.get("best_lambda", np.nan),
    }


def run_one_seed(model_df, seed):
    train_df, test_df = split_train_test(
        model_df,
        train_fraction=TRAIN_FRACTION,
        seed=seed,
        stratify_columns=["intervention_flag", "outcome_ed_90d"],
    )

    train_treated = train_df[train_df["intervention_flag"] == 1].copy()
    train_control = train_df[train_df["intervention_flag"] == 0].copy()
    feature_cols = [column for column in model_df.columns if column not in ["outcome_ed_90d", "intervention_flag"]]

    combined_matrix, split_matrices = make_design_matrix(
        [
            train_treated[feature_cols],
            train_control[feature_cols],
            test_df[feature_cols],
            model_df[feature_cols],
        ]
    )
    x_treated, x_control, x_test, x_full = split_matrices
    y_treated = train_treated["outcome_ed_90d"].astype(float).to_numpy()
    y_control = train_control["outcome_ed_90d"].astype(float).to_numpy()

    test_treated_pos = np.where(test_df["intervention_flag"].to_numpy() == 1)[0]
    test_control_pos = np.where(test_df["intervention_flag"].to_numpy() == 0)[0]

    rows = []
    top_rows = []
    top_member_rows = []
    event_rows = event_count_rows(seed, train_df, test_df)

    model_specs = {
        "XGBoost": BASELINE_XGB_GRID,
        "XGBoost_Regularized": REGULARIZED_XGB_GRID,
    }

    for model_name, grid in model_specs.items():
        treated_fit = fit_xgb_cv_grid(x_treated, y_treated, grid=grid, seed=seed)
        control_fit = fit_xgb_cv_grid(x_control, y_control, grid=grid, seed=seed)

        pred_treated_group = treated_fit["model"].predict(make_dmatrix(x_test.iloc[test_treated_pos]))
        pred_control_group = control_fit["model"].predict(make_dmatrix(x_test.iloc[test_control_pos]))
        rows.append(model_metric_row(seed, model_name, "treated", treated_fit, test_df, test_treated_pos, pred_treated_group))
        rows.append(model_metric_row(seed, model_name, "control", control_fit, test_df, test_control_pos, pred_control_group))

        pred_treated_all = treated_fit["model"].predict(make_dmatrix(x_test))
        pred_control_all = control_fit["model"].predict(make_dmatrix(x_test))
        rows[-2].update(factual_metrics(test_df, pred_treated_all, pred_control_all))
        rows[-1].update(factual_metrics(test_df, pred_treated_all, pred_control_all))

        full_results = build_uplift_results(
            model_df,
            treated_fit["model"].predict(make_dmatrix(x_full)),
            control_fit["model"].predict(make_dmatrix(x_full)),
        )
        top_rows.append(top_decile_summary(full_results, model_name, seed))
        for member_index in full_results.index[full_results["uplift_decile"] == TOP_DECILE]:
            top_member_rows.append({"seed": seed, "model": model_name, "member_index": int(member_index)})

    scaler = StandardScaler().fit(pd.concat([x_treated, x_control], axis=0))
    glm_treated = fit_glmnet(x_treated, y_treated, seed=seed, prefit_scaler=scaler)
    glm_control = fit_glmnet(x_control, y_control, seed=seed, prefit_scaler=scaler)

    pred_treated_group = glmnet_predict(glm_treated, x_test.iloc[test_treated_pos])
    pred_control_group = glmnet_predict(glm_control, x_test.iloc[test_control_pos])
    rows.append(model_metric_row(seed, "GLMNet", "treated", glm_treated, test_df, test_treated_pos, pred_treated_group))
    rows.append(model_metric_row(seed, "GLMNet", "control", glm_control, test_df, test_control_pos, pred_control_group))

    pred_treated_all = glmnet_predict(glm_treated, x_test)
    pred_control_all = glmnet_predict(glm_control, x_test)
    rows[-2].update(factual_metrics(test_df, pred_treated_all, pred_control_all))
    rows[-1].update(factual_metrics(test_df, pred_treated_all, pred_control_all))

    full_results = build_uplift_results(model_df, glmnet_predict(glm_treated, x_full), glmnet_predict(glm_control, x_full))
    top_rows.append(top_decile_summary(full_results, "GLMNet", seed))
    for member_index in full_results.index[full_results["uplift_decile"] == TOP_DECILE]:
        top_member_rows.append({"seed": seed, "model": "GLMNet", "member_index": int(member_index)})

    return rows, event_rows, top_rows, top_member_rows


def summarize_metrics(metrics_df):
    return (
        metrics_df.groupby(["model", "outcome_model_group"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_cv_auc=("cv_auc", "mean"),
            sd_cv_auc=("cv_auc", "std"),
            mean_test_auc=("test_auc", "mean"),
            sd_test_auc=("test_auc", "std"),
            min_test_auc=("test_auc", "min"),
            max_test_auc=("test_auc", "max"),
            mean_cv_minus_test_auc=("cv_minus_test_auc", "mean"),
            mean_factual_auc=("test_factual_auc", "mean"),
            mean_factual_brier=("test_factual_brier", "mean"),
        )
        .sort_values(["outcome_model_group", "mean_test_auc"], ascending=[True, False])
    )


def summarize_top_decile(top_df):
    return (
        top_df.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_top_decile_ed_rate=("top_decile_observed_ed_rate", "mean"),
            sd_top_decile_ed_rate=("top_decile_observed_ed_rate", "std"),
            mean_top_decile_treated_ed_rate=("top_decile_treated_ed_rate", "mean"),
            mean_top_decile_control_ed_rate=("top_decile_control_ed_rate", "mean"),
            mean_top_decile_avg_benefit_score=("top_decile_avg_benefit_score", "mean"),
        )
        .sort_values("mean_top_decile_avg_benefit_score", ascending=False)
    )


def top_decile_jaccard(top_member_df):
    rows = []
    for model_name, model_members in top_member_df.groupby("model"):
        seed_sets = {
            seed: set(seed_df["member_index"].astype(int))
            for seed, seed_df in model_members.groupby("seed")
        }
        for seed_a, seed_b in combinations(sorted(seed_sets), 2):
            union = seed_sets[seed_a] | seed_sets[seed_b]
            intersection = seed_sets[seed_a] & seed_sets[seed_b]
            rows.append(
                {
                    "model": model_name,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "top_decile_jaccard": len(intersection) / len(union) if union else np.nan,
                    "overlap_count": len(intersection),
                    "union_count": len(union),
                }
            )
    return pd.DataFrame(rows)


def write_charts(metrics_df, summary_df, top_jaccard_df):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = metrics_df.copy()
    plot_df["series"] = plot_df["model"] + " " + plot_df["outcome_model_group"]
    for series, frame in plot_df.groupby("series"):
        ax.plot(frame["seed"], frame["test_auc"], marker="o", label=series)
    ax.set_title("Held-out AUC by Seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test AUC")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "sensitivity_test_auc_by_seed.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_df = summary_df.copy()
    bar_df["series"] = bar_df["model"] + " " + bar_df["outcome_model_group"]
    ax.barh(bar_df["series"], bar_df["mean_cv_minus_test_auc"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Average CV Minus Test AUC")
    ax.set_xlabel("CV AUC - Test AUC")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "sensitivity_cv_test_gap.png", dpi=150)
    plt.close(fig)

    if not top_jaccard_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        jaccard_summary = top_jaccard_df.groupby("model")["top_decile_jaccard"].mean().sort_values()
        ax.barh(jaccard_summary.index, jaccard_summary.values)
        ax.set_title("Average Top-Decile Member Overlap Across Seed Pairs")
        ax.set_xlabel("Mean Jaccard overlap")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / "sensitivity_top_decile_overlap.png", dpi=150)
        plt.close(fig)


def markdown_table(df, columns, float_digits=3):
    if df.empty:
        return "_No rows._"
    display_df = df.loc[:, columns].copy()
    for column in display_df.select_dtypes(include=[float]).columns:
        display_df[column] = display_df[column].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
    return display_df.to_markdown(index=False)


def write_findings_readme(model_df, metrics_df, summary_df, event_df, top_summary_df, top_jaccard_df):
    event_pivot = event_df.pivot_table(index=["seed", "split"], columns="group", values=["n", "events"], aggfunc="first")
    event_pivot.columns = [f"{metric}_{group}" for metric, group in event_pivot.columns]
    event_pivot = event_pivot.reset_index()

    control_summary = summary_df[summary_df["outcome_model_group"] == "control"].sort_values("mean_test_auc", ascending=False)
    treated_summary = summary_df[summary_df["outcome_model_group"] == "treated"].sort_values("mean_test_auc", ascending=False)
    jaccard_summary = (
        top_jaccard_df.groupby("model", as_index=False)
        .agg(mean_top_decile_jaccard=("top_decile_jaccard", "mean"), min_top_decile_jaccard=("top_decile_jaccard", "min"))
        .sort_values("mean_top_decile_jaccard", ascending=False)
        if not top_jaccard_df.empty
        else pd.DataFrame(columns=["model", "mean_top_decile_jaccard", "min_top_decile_jaccard"])
    )

    lines = [
        "# PRISM Uplift Seed Sensitivity Findings",
        "",
        "This README is generated by `Code/uplift_seed_sensitivity_analysis.py`. It reports repeated random-seed checks without modifying the main uplift notebook outputs.",
        "",
        "## Scope",
        "",
        f"- Dataset rows analyzed: {len(model_df):,}",
        f"- Outcome events: {int(model_df['outcome_ed_90d'].sum()):,}",
        f"- Outcome prevalence: {model_df['outcome_ed_90d'].mean():.3f}",
        f"- Seeds: {', '.join(str(seed) for seed in sorted(metrics_df['seed'].unique()))}",
        "- Split: 70/30 stratified by treatment flag and ED outcome",
        "- Models: baseline XGBoost grid, regularized XGBoost grid, and GLMNet elastic-net logistic regression",
        "",
        "## Main Takeaways",
        "",
    ]

    best_control = control_summary.iloc[0]
    best_treated = treated_summary.iloc[0]
    most_stable = jaccard_summary.iloc[0] if not jaccard_summary.empty else None
    largest_gap = summary_df.sort_values("mean_cv_minus_test_auc", ascending=False).iloc[0]

    lines.extend(
        [
            f"- Best average control-model test AUC: {best_control['model']} ({best_control['mean_test_auc']:.3f}).",
            f"- Best average treated-model test AUC: {best_treated['model']} ({best_treated['mean_test_auc']:.3f}).",
            f"- Largest average CV-to-test AUC drop: {largest_gap['model']} {largest_gap['outcome_model_group']} ({largest_gap['mean_cv_minus_test_auc']:.3f}).",
        ]
    )
    if most_stable is not None:
        lines.append(
            f"- Most stable full-population top-decile membership: {most_stable['model']} "
            f"(mean Jaccard {most_stable['mean_top_decile_jaccard']:.3f})."
        )

    lines.extend(
        [
            "",
            "Interpretation note: low treated/control test-event counts can make subgroup AUC volatile. Large positive CV-minus-test gaps are a warning sign that the model may be fitting split-specific signal.",
            "",
            "## Average AUC And Calibration Metrics",
            "",
            markdown_table(
                summary_df,
                [
                    "model",
                    "outcome_model_group",
                    "seeds",
                    "mean_cv_auc",
                    "mean_test_auc",
                    "sd_test_auc",
                    "mean_cv_minus_test_auc",
                    "mean_factual_auc",
                    "mean_factual_brier",
                ],
            ),
            "",
            "## Test Event Counts By Seed",
            "",
            markdown_table(event_pivot, list(event_pivot.columns), float_digits=0),
            "",
            "## Top-Decile Uplift Stability",
            "",
            markdown_table(
                top_summary_df,
                [
                    "model",
                    "seeds",
                    "mean_top_decile_ed_rate",
                    "sd_top_decile_ed_rate",
                    "mean_top_decile_treated_ed_rate",
                    "mean_top_decile_control_ed_rate",
                    "mean_top_decile_avg_benefit_score",
                ],
            ),
            "",
            "## Top-Decile Member Overlap",
            "",
            markdown_table(jaccard_summary, ["model", "mean_top_decile_jaccard", "min_top_decile_jaccard"]),
            "",
            "## Output Files",
            "",
            "- `sensitivity_seed_metrics.csv`: per-seed CV/test AUC, CV-test gap, factual AUC, and Brier score.",
            "- `sensitivity_summary_by_model.csv`: cross-seed model summary.",
            "- `sensitivity_event_counts_by_seed.csv`: subgroup event counts behind each seed's test AUC.",
            "- `sensitivity_top_decile_by_seed.csv`: full-population top-decile outcome diagnostics by seed.",
            "- `sensitivity_top_decile_jaccard.csv`: top-decile member overlap across seed pairs.",
            "- `sensitivity_top_decile_membership.csv`: member indices selected into top decile by seed/model.",
            "- `sensitivity_test_auc_by_seed.png`, `sensitivity_cv_test_gap.png`, `sensitivity_top_decile_overlap.png`: diagnostic charts.",
            "",
        ]
    )
    path = OUTPUT_ROOT / "PRISM_Uplift_Seed_Sensitivity_README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    model_df = prepare_model_frame()
    all_metric_rows = []
    all_event_rows = []
    all_top_rows = []
    all_top_member_rows = []

    for seed in SEEDS:
        print(f"Running seed {seed}...")
        metric_rows, event_rows, top_rows, top_member_rows = run_one_seed(model_df, seed)
        all_metric_rows.extend(metric_rows)
        all_event_rows.extend(event_rows)
        all_top_rows.extend(top_rows)
        all_top_member_rows.extend(top_member_rows)

    metrics_df = pd.DataFrame(all_metric_rows)
    event_df = pd.DataFrame(all_event_rows)
    top_df = pd.DataFrame(all_top_rows)
    top_member_df = pd.DataFrame(all_top_member_rows)
    summary_df = summarize_metrics(metrics_df)
    top_summary_df = summarize_top_decile(top_df)
    top_jaccard_df = top_decile_jaccard(top_member_df)

    metrics_df.to_csv(OUTPUT_ROOT / "sensitivity_seed_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_ROOT / "sensitivity_summary_by_model.csv", index=False)
    event_df.to_csv(OUTPUT_ROOT / "sensitivity_event_counts_by_seed.csv", index=False)
    top_df.to_csv(OUTPUT_ROOT / "sensitivity_top_decile_by_seed.csv", index=False)
    top_summary_df.to_csv(OUTPUT_ROOT / "sensitivity_top_decile_summary_by_model.csv", index=False)
    top_member_df.to_csv(OUTPUT_ROOT / "sensitivity_top_decile_membership.csv", index=False)
    top_jaccard_df.to_csv(OUTPUT_ROOT / "sensitivity_top_decile_jaccard.csv", index=False)

    write_charts(metrics_df, summary_df, top_jaccard_df)
    readme_path = write_findings_readme(model_df, metrics_df, summary_df, event_df, top_summary_df, top_jaccard_df)

    print()
    print("Sensitivity analysis complete.")
    print("Output folder:", OUTPUT_ROOT)
    print("Findings README:", readme_path)
    print()
    print(summary_df)
    return {
        "model_df": model_df,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "event_df": event_df,
        "top_df": top_df,
        "top_summary_df": top_summary_df,
        "top_jaccard_df": top_jaccard_df,
        "readme_path": readme_path,
    }


if __name__ == "__main__":
    main()
