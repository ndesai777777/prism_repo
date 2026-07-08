"""Report-ready PRISM causal forest modeling workflow."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

if __package__ is None:
    code_dir = Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

from _prism_model_utils import (
    add_date_features,
    clean_names_simple,
    ensure_output_folder,
    make_design_matrix,
    ntile_desc,
    prepare_model_frame,
    read_prism_excel,
    require_columns,
    split_train_test,
    to_binary,
)


SEED = 123
TRAIN_FRACTION = 0.70
OUTCOME_COL = "outcome_ed_90d"
TREATMENT_COL = "intervention_flag"

PREDICTOR_CATEGORIES = {
    "demographics": [
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
    ],
    "clinical_conditions": [
        "diabetes_flag",
        "chf_flag",
        "copd_flag",
        "asthma_flag",
        "depression_flag",
        "anxiety_flag",
        "substance_use_flag",
        "ckd_flag",
        "behavioral_health_risk_flag",
    ],
    "sdoh": [
        "food_insecurity_flag",
        "housing_instability_flag",
        "transportation_barrier_flag",
        "utilities_insecurity_flag",
    ],
    "utilization": [
        "pcp_visits_last_6m",
        "specialist_visits_last_6m",
        "ed_visits_last_30d",
        "ed_visits_last_6m",
        "admits_last_6m",
        "observation_stays_last_6m",
    ],
    "pharmacy": [
        "total_cost_last_6m",
        "rx_count_last_6m",
        "med_adherence_pdc",
        "high_cost_drug_flag",
        "opioid_flag",
        "polypharmacy_flag",
    ],
    "risk_scores": [
        "percolator_utilization_score",
        "percolator_clinical_score",
        "percolator_sdoh_score",
        "current_risk_score",
        "risk_tier",
    ],
}

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
    "intervention_start_month",
    "intervention_start_wday",
    "days_to_intervention_start",
]
BINARY_EXTRA = ["dual_eligible", "living_alone_flag"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "scored_output": output_dir / "causal_forest_scored_output.csv",
        "test_scored_output": output_dir / "causal_forest_scored_test_output.csv",
        "data_review_summary": output_dir / "causal_forest_data_review_summary.csv",
        "predictor_inventory": output_dir / "causal_forest_predictor_inventory.csv",
        "event_count_summary": output_dir / "causal_forest_event_count_summary.csv",
        "propensity_summary": output_dir / "causal_forest_propensity_summary.csv",
        "effect_distribution_summary": output_dir / "causal_forest_effect_distribution_summary.csv",
        "uncertainty_summary": output_dir / "causal_forest_uncertainty_summary.csv",
        "ate_summary": output_dir / "causal_forest_ate_summary.csv",
        "decile_summary": output_dir / "causal_forest_decile_summary.csv",
        "top_benefit_examples": output_dir / "causal_forest_top_benefit_examples.csv",
        "variable_importance": output_dir / "causal_forest_variable_importance.csv",
        "top_decile_profile": output_dir / "causal_forest_top_decile_profile.csv",
        "targeting_summary": output_dir / "causal_forest_targeting_summary.csv",
        "consistency_summary": output_dir / "causal_forest_vs_uplift_consistency_summary.csv",
        "propensity_chart": output_dir / "dashboard_propensity_overlap.png",
        "effect_distribution_chart": output_dir / "dashboard_causal_forest_effect_distribution.png",
        "benefit_decile_chart": output_dir / "dashboard_causal_forest_avg_benefit_by_decile.png",
        "tau_decile_chart": output_dir / "dashboard_causal_forest_tau_by_decile.png",
        "risk_benefit_decile_chart": output_dir / "dashboard_causal_forest_risk_vs_benefit_by_decile.png",
        "variable_importance_chart": output_dir / "dashboard_causal_forest_variable_importance.png",
        "comparison_chart": output_dir / "dashboard_causal_forest_vs_uplift_comparison.png",
    }


def require_econml():
    if importlib.util.find_spec("econml") is None:
        raise ImportError(
            "Missing required package: econml. Install it with `pip install econml`, "
            "then rerun the causal forest workflow."
        )
    from econml.dml import CausalForestDML

    return CausalForestDML


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def effect_standard_errors(cf_model, x_matrix: pd.DataFrame) -> np.ndarray:
    try:
        inference = cf_model.effect_inference(x_matrix)
        return np.asarray(inference.stderr, dtype=float)
    except Exception as exc:
        print(f"Standard errors not available: {exc}")
        return np.full(len(x_matrix), np.nan)


def summarize_binary_by_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, frame in [("Train", train_df), ("Test", test_df)]:
        for group_value, group_label in [(1.0, "Treated"), (0.0, "Control")]:
            subset = frame[frame[TREATMENT_COL] == group_value]
            positive = int((subset[OUTCOME_COL] == 1).sum())
            n = int(len(subset))
            rows.append(
                {
                    "split": split_name,
                    "group": group_label,
                    "n": n,
                    "positive_ed_events": positive,
                    "negative_ed_events": n - positive,
                    "event_rate": positive / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_distribution(values: pd.Series, label: str) -> pd.DataFrame:
    series = pd.Series(values, dtype=float).dropna()
    return pd.DataFrame(
        {
            "metric": [
                "mean",
                "std_dev",
                "min",
                "p10",
                "p25",
                "median",
                "p75",
                "p90",
                "max",
            ],
            label: [
                series.mean(),
                series.std(),
                series.min(),
                series.quantile(0.10),
                series.quantile(0.25),
                series.median(),
                series.quantile(0.75),
                series.quantile(0.90),
                series.max(),
            ],
        }
    )


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    joined = pd.concat([pd.Series(a, dtype=float), pd.Series(b, dtype=float)], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method=method))


def top_overlap(a_scores: pd.Series, b_scores: pd.Series, share: float = 0.10) -> float:
    a = pd.Series(a_scores).reset_index(drop=True)
    b = pd.Series(b_scores).reset_index(drop=True)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    k = max(1, int(np.floor(n * share)))
    a_top = set(a.iloc[:n].nlargest(k).index)
    b_top = set(b.iloc[:n].nlargest(k).index)
    return len(a_top & b_top) / k


def propensity_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan


def member_examples(results: pd.DataFrame) -> pd.DataFrame:
    ranked = results.sort_values("benefit_score", ascending=False).copy()
    if ranked.empty:
        return pd.DataFrame()

    candidates = [("Highest benefit", ranked.iloc[0]), ("Lowest benefit", ranked.iloc[-1])]
    if "current_risk_score" in ranked.columns:
        high_risk = ranked["current_risk_score"].quantile(0.75)
        low_risk = ranked["current_risk_score"].quantile(0.50)
        low_benefit = ranked["benefit_score"].quantile(0.25)
        high_benefit = ranked["benefit_score"].quantile(0.75)

        subset = ranked[
            (ranked["current_risk_score"] >= high_risk)
            & (ranked["benefit_score"] <= low_benefit)
        ]
        if not subset.empty:
            candidates.append(("High risk, low benefit", subset.iloc[0]))

        subset = ranked[
            (ranked["current_risk_score"] <= low_risk)
            & (ranked["benefit_score"] >= high_benefit)
        ]
        if not subset.empty:
            candidates.append(("Low risk, high benefit", subset.iloc[0]))

    interpretation = {
        "Highest benefit": "Strong outreach candidate based on estimated ED risk reduction.",
        "Lowest benefit": "Lowest priority by causal forest benefit score.",
        "High risk, low benefit": "High baseline risk but limited estimated impactability.",
        "Low risk, high benefit": "May be missed by risk-only targeting but appears impactable.",
    }
    rows = []
    for label, row in candidates:
        rows.append(
            {
                "member_profile": label,
                "row_id": int(row["row_id"]),
                "actual_outcome": row[OUTCOME_COL],
                "treatment_flag": row[TREATMENT_COL],
                "current_risk_score": row.get("current_risk_score", np.nan),
                "tau_hat": row["tau_hat"],
                "tau_se": row.get("tau_se", np.nan),
                "benefit_score": row["benefit_score"],
                "hte_decile": row["hte_decile"],
                "outreach_interpretation": interpretation[label],
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["member_profile", "row_id"])


def plot_outputs(
    paths: dict[str, Path],
    results_test: pd.DataFrame,
    decile_summary: pd.DataFrame,
    importance_df: pd.DataFrame,
    consistency_summary: pd.DataFrame,
    test_propensity: np.ndarray,
    w_test: np.ndarray,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(test_propensity[w_test == 1], bins=15, alpha=0.65, label="Treated", color="#2F6B9A")
    plt.hist(test_propensity[w_test == 0], bins=15, alpha=0.65, label="Control", color="#D9822B")
    plt.xlabel("Estimated propensity for intervention")
    plt.ylabel("Members")
    plt.title("Causal Forest Propensity Overlap Check")
    plt.legend()
    save_current_figure(paths["propensity_chart"])

    plt.figure(figsize=(8, 4.5))
    plt.hist(results_test["benefit_score"], bins=20, color="#3B7A57", alpha=0.85)
    plt.axvline(results_test["benefit_score"].mean(), color="#1F2933", linestyle="--")
    plt.xlabel("Benefit score (-tau_hat)")
    plt.ylabel("Members")
    plt.title("Causal Forest Estimated Benefit Distribution")
    save_current_figure(paths["effect_distribution_chart"])

    plt.figure(figsize=(8, 4.5))
    plt.bar(decile_summary["hte_decile"].astype(str), decile_summary["avg_benefit_score"], color="#3B7A57")
    plt.xlabel("HTE decile (1 = highest estimated benefit)")
    plt.ylabel("Average benefit score")
    plt.title("Causal Forest Average Estimated Benefit By HTE Decile")
    save_current_figure(paths["benefit_decile_chart"])

    plt.figure(figsize=(8, 4.5))
    plt.bar(decile_summary["hte_decile"].astype(str), decile_summary["avg_tau_hat"], color="#6B4E71")
    plt.axhline(0, color="#1F2933", linewidth=1)
    plt.xlabel("HTE decile (1 = highest estimated benefit)")
    plt.ylabel("Average tau_hat")
    plt.title("Causal Forest Average Treatment Effect By HTE Decile")
    save_current_figure(paths["tau_decile_chart"])

    if "avg_current_risk_score" in decile_summary.columns:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.bar(decile_summary["hte_decile"].astype(str), decile_summary["avg_benefit_score"], color="#3B7A57", alpha=0.8)
        ax1.set_xlabel("HTE decile (1 = highest estimated benefit)")
        ax1.set_ylabel("Average benefit score")
        ax2 = ax1.twinx()
        ax2.plot(decile_summary["hte_decile"].astype(str), decile_summary["avg_current_risk_score"], color="#C2410C", marker="o")
        ax2.set_ylabel("Average current risk score")
        plt.title("Causal Forest Benefit Versus Current Risk By Decile")
        save_current_figure(paths["risk_benefit_decile_chart"])

    plot_df = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5.5))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#2F6B9A")
    plt.xlabel("Causal forest importance")
    plt.ylabel("Feature")
    plt.title("Top Causal Forest HTE Split Features")
    save_current_figure(paths["variable_importance_chart"])

    if not consistency_summary.empty:
        plot_df = consistency_summary.set_index("comparison")[["spearman_corr"]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(8, 4.5))
            plt.barh(plot_df.index, plot_df["spearman_corr"], color="#2F6B9A")
            plt.axvline(0, color="#1F2933", linewidth=1)
            plt.xlabel("Spearman correlation")
            plt.title("Causal Forest Ranking Consistency Checks")
            save_current_figure(paths["comparison_chart"])


def run_workflow(write_outputs: bool = True) -> dict[str, pd.DataFrame]:
    CausalForestDML = require_econml()
    warnings.filterwarnings("ignore", category=UserWarning)
    np.random.seed(SEED)

    root = project_root()
    output_dir = ensure_output_folder(root / "Outputs" / "Causal-Forests" / "Python")
    paths = output_paths(output_dir)
    print(f"Project root: {root}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {SEED}")

    df = read_prism_excel()
    df.columns = clean_names_simple(df.columns)
    require_columns(df, [OUTCOME_COL, TREATMENT_COL])
    df[OUTCOME_COL] = to_binary(df[OUTCOME_COL])
    df[TREATMENT_COL] = to_binary(df[TREATMENT_COL])
    df = add_date_features(df, include_duration=False)

    predictor_vars = [feature for features in PREDICTOR_CATEGORIES.values() for feature in features]
    present_predictors = [feature for feature in predictor_vars if feature in df.columns]

    inventory_rows = []
    for category, features in PREDICTOR_CATEGORIES.items():
        for feature in features:
            inventory_rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "included_in_model": feature in present_predictors,
                    "reason_if_excluded": "" if feature in present_predictors else "Column not present in source data",
                    "source_dtype": str(df[feature].dtype) if feature in df.columns else "missing",
                    "unique_values": df[feature].nunique(dropna=True) if feature in df.columns else 0,
                }
            )
    predictor_inventory = pd.DataFrame(inventory_rows)

    model_df = prepare_model_frame(df, present_predictors, NUMERIC_VARS, BINARY_EXTRA)
    model_df.insert(0, "row_id", np.arange(len(model_df)))
    feature_cols_raw = [
        col for col in model_df.columns if col not in ["row_id", OUTCOME_COL, TREATMENT_COL]
    ]
    continuous_count = len([col for col in feature_cols_raw if col in NUMERIC_VARS])
    binary_count = len([col for col in feature_cols_raw if col.endswith("_flag") or col in BINARY_EXTRA])
    categorical_count = len(feature_cols_raw) - continuous_count - binary_count

    feature_frame = model_df.drop(columns=["row_id", OUTCOME_COL, TREATMENT_COL])
    _, [x_all] = make_design_matrix([feature_frame])

    data_review_summary = pd.DataFrame(
        {
            "metric": [
                "Total members",
                "Treated members",
                "Untreated/control members",
                "Treatment rate",
                "ED outcome events",
                "Outcome prevalence",
                "Treated observed ED rate",
                "Control observed ED rate",
                "Final predictors before one-hot encoding",
                "Continuous/count numeric predictors",
                "Binary indicator predictors",
                "Multi-level categorical predictors",
                "Model matrix columns after one-hot encoding",
            ],
            "current_value": [
                len(model_df),
                int((model_df[TREATMENT_COL] == 1).sum()),
                int((model_df[TREATMENT_COL] == 0).sum()),
                model_df[TREATMENT_COL].mean(),
                int((model_df[OUTCOME_COL] == 1).sum()),
                model_df[OUTCOME_COL].mean(),
                model_df.loc[model_df[TREATMENT_COL] == 1, OUTCOME_COL].mean(),
                model_df.loc[model_df[TREATMENT_COL] == 0, OUTCOME_COL].mean(),
                len(feature_cols_raw),
                continuous_count,
                binary_count,
                categorical_count,
                x_all.shape[1],
            ],
        }
    )

    train_df, test_df = split_train_test(
        model_df,
        train_fraction=TRAIN_FRACTION,
        seed=SEED,
        stratify_columns=[TREATMENT_COL, OUTCOME_COL],
    )
    x_train = x_all.loc[train_df.index].reset_index(drop=True)
    x_test = x_all.loc[test_df.index].reset_index(drop=True)
    y_train = train_df[OUTCOME_COL].astype(float).to_numpy()
    w_train = train_df[TREATMENT_COL].astype(float).to_numpy()
    y_test = test_df[OUTCOME_COL].astype(float).to_numpy()
    w_test = test_df[TREATMENT_COL].astype(float).to_numpy()

    event_count_summary = summarize_binary_by_split(train_df, test_df)

    propensity_model = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED)
    propensity_model.fit(x_train, w_train)
    train_propensity = propensity_model.predict_proba(x_train)[:, 1]
    test_propensity = propensity_model.predict_proba(x_test)[:, 1]
    all_propensity = propensity_model.predict_proba(x_all)[:, 1]
    propensity_series = pd.Series(test_propensity, dtype=float)
    propensity_summary = pd.DataFrame(
        {
            "metric": [
                "Train treatment model AUC",
                "Test treatment model AUC",
                "Mean propensity",
                "Min propensity",
                "5th percentile",
                "Median propensity",
                "95th percentile",
                "Max propensity",
                "Members below 0.05",
                "Members above 0.95",
            ],
            "value": [
                propensity_auc(w_train, train_propensity),
                propensity_auc(w_test, test_propensity),
                propensity_series.mean(),
                propensity_series.min(),
                propensity_series.quantile(0.05),
                propensity_series.median(),
                propensity_series.quantile(0.95),
                propensity_series.max(),
                int((propensity_series < 0.05).sum()),
                int((propensity_series > 0.95).sum()),
            ],
        }
    )

    cf_model = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1
        ),
        model_t=RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1
        ),
        discrete_treatment=True,
        n_estimators=800,
        min_samples_leaf=10,
        max_depth=None,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED,
    )
    cf_model.fit(y_train, w_train, X=x_train)

    results_test = test_df.reset_index(drop=True).copy()
    results_test["tau_hat"] = np.asarray(cf_model.effect(x_test), dtype=float)
    results_test["tau_se"] = effect_standard_errors(cf_model, x_test)
    results_test["benefit_score"] = -results_test["tau_hat"]
    results_test["hte_decile"] = ntile_desc(results_test["benefit_score"], 10).to_numpy()
    results_test["uplift_decile"] = results_test["hte_decile"]
    results_test["propensity_score"] = test_propensity
    ci_multiplier = 1.96
    results_test["tau_ci_lower"] = results_test["tau_hat"] - ci_multiplier * results_test["tau_se"]
    results_test["tau_ci_upper"] = results_test["tau_hat"] + ci_multiplier * results_test["tau_se"]
    results_test["benefit_ci_lower"] = -results_test["tau_ci_upper"]
    results_test["benefit_ci_upper"] = -results_test["tau_ci_lower"]

    ate_summary = pd.DataFrame(
        {
            "metric": ["avg_tau_hat", "avg_benefit_score", "test_members"],
            "value": [results_test["tau_hat"].mean(), results_test["benefit_score"].mean(), len(results_test)],
        }
    )
    effect_distribution_summary = summarize_distribution(results_test["tau_hat"], "tau_hat").merge(
        summarize_distribution(results_test["benefit_score"], "benefit_score"),
        on="metric",
        how="outer",
    )
    uncertainty_summary = pd.DataFrame(
        {
            "metric": [
                "Mean tau standard error",
                "Median tau standard error",
                "Members with tau CI entirely below zero",
                "Members with tau CI crossing zero",
                "Members with tau CI entirely above zero",
                "Top HTE decile mean tau standard error",
            ],
            "value": [
                results_test["tau_se"].mean(),
                results_test["tau_se"].median(),
                int((results_test["tau_ci_upper"] < 0).sum()),
                int(((results_test["tau_ci_lower"] <= 0) & (results_test["tau_ci_upper"] >= 0)).sum()),
                int((results_test["tau_ci_lower"] > 0).sum()),
                results_test.loc[results_test["hte_decile"] == 1, "tau_se"].mean(),
            ],
        }
    )

    agg_map = {
        "n": ("hte_decile", "size"),
        "avg_tau_hat": ("tau_hat", "mean"),
        "avg_benefit_score": ("benefit_score", "mean"),
        "avg_tau_se": ("tau_se", "mean"),
        "observed_ed_rate": (OUTCOME_COL, "mean"),
        "treatment_pct": (TREATMENT_COL, "mean"),
        "avg_propensity_score": ("propensity_score", "mean"),
    }
    if "current_risk_score" in results_test.columns:
        agg_map["avg_current_risk_score"] = ("current_risk_score", "mean")
    decile_summary = results_test.groupby("hte_decile", as_index=False).agg(**agg_map)
    decile_summary["uplift_decile"] = decile_summary["hte_decile"]

    top_benefit_examples = member_examples(results_test)

    importances = getattr(cf_model, "feature_importances_", np.full(x_train.shape[1], np.nan))
    importance_df = (
        pd.DataFrame({"feature": x_train.columns, "importance": np.asarray(importances, dtype=float)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))

    top_decile_profile = make_top_decile_profile(results_test)

    scored_full = model_df.copy()
    scored_full["tau_hat"] = np.asarray(cf_model.effect(x_all), dtype=float)
    scored_full["tau_se"] = effect_standard_errors(cf_model, x_all)
    scored_full["benefit_score"] = -scored_full["tau_hat"]
    scored_full["hte_decile"] = ntile_desc(scored_full["benefit_score"], 10).to_numpy()
    scored_full["uplift_decile"] = scored_full["hte_decile"]
    scored_full["propensity_score"] = all_propensity
    scored_full["tau_ci_lower"] = scored_full["tau_hat"] - ci_multiplier * scored_full["tau_se"]
    scored_full["tau_ci_upper"] = scored_full["tau_hat"] + ci_multiplier * scored_full["tau_se"]
    scored_full["benefit_ci_lower"] = -scored_full["tau_ci_upper"]
    scored_full["benefit_ci_upper"] = -scored_full["tau_ci_lower"]

    consistency_summary = make_consistency_summary(root, scored_full, results_test)
    targeting_summary = decile_summary[
        ["hte_decile", "n", "avg_benefit_score", "observed_ed_rate", "treatment_pct"]
    ].copy()
    if "avg_current_risk_score" in decile_summary.columns:
        targeting_summary["avg_current_risk_score"] = decile_summary["avg_current_risk_score"]
    targeting_summary["cumulative_members"] = targeting_summary["n"].cumsum()
    targeting_summary["cumulative_expected_ed_reductions"] = (
        targeting_summary["avg_benefit_score"] * targeting_summary["n"]
    ).cumsum()

    outputs = {
        "data_review_summary": data_review_summary,
        "predictor_inventory": predictor_inventory,
        "event_count_summary": event_count_summary,
        "propensity_summary": propensity_summary,
        "ate_summary": ate_summary,
        "effect_distribution_summary": effect_distribution_summary,
        "uncertainty_summary": uncertainty_summary,
        "decile_summary": decile_summary,
        "top_benefit_examples": top_benefit_examples,
        "variable_importance": importance_df,
        "top_decile_profile": top_decile_profile,
        "targeting_summary": targeting_summary,
        "consistency_summary": consistency_summary,
        "test_scored_output": results_test,
        "scored_output": scored_full,
    }

    if write_outputs:
        for key, frame in outputs.items():
            if key in paths:
                save_csv(frame, paths[key])
        plot_outputs(paths, results_test, decile_summary, importance_df, consistency_summary, test_propensity, w_test)

    return outputs


def make_top_decile_profile(results_test: pd.DataFrame) -> pd.DataFrame:
    profile_features = [
        "current_risk_score",
        "percolator_utilization_score",
        "percolator_clinical_score",
        "percolator_sdoh_score",
        "ed_visits_last_6m",
        "admits_last_6m",
        "total_cost_last_6m",
        "behavioral_health_risk_flag",
        "food_insecurity_flag",
        "housing_instability_flag",
        "transportation_barrier_flag",
        "utilities_insecurity_flag",
        "dual_eligible",
    ]
    profile_features = [feature for feature in profile_features if feature in results_test.columns]
    top_decile_mask = results_test["hte_decile"] == 1
    rows = []
    for feature in profile_features:
        top_value = pd.to_numeric(results_test.loc[top_decile_mask, feature], errors="coerce").mean()
        other_value = pd.to_numeric(results_test.loc[~top_decile_mask, feature], errors="coerce").mean()
        rows.append(
            {
                "feature": feature,
                "top_hte_decile_mean_or_rate": top_value,
                "other_deciles_mean_or_rate": other_value,
                "difference": top_value - other_value,
            }
        )
    return pd.DataFrame(rows).sort_values("difference", key=lambda s: s.abs(), ascending=False)


def make_consistency_summary(root: Path, scored_full: pd.DataFrame, results_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t_path = root / "Outputs" / "Uplift" / "Python" / "T-Learner" / "GLMNet" / "uplift_scored_output.csv"
    if t_path.exists():
        t_df = pd.read_csv(t_path)
        n = min(len(scored_full), len(t_df))
        t_scores = t_df["benefit_score"].iloc[:n] if "benefit_score" in t_df.columns else pd.Series(np.nan, index=range(n))
        cf_scores = scored_full["benefit_score"].iloc[:n]
        rows.append(
            {
                "comparison": "Causal forest vs GLMNet T-learner full output",
                "comparison_basis": "row_order_full_file_no_stable_member_id",
                "n_compared": n,
                "pearson_corr": safe_corr(cf_scores, t_scores, "pearson"),
                "spearman_corr": safe_corr(cf_scores, t_scores, "spearman"),
                "top_decile_overlap": top_overlap(cf_scores, t_scores, 0.10),
                "top_20pct_overlap": top_overlap(cf_scores, t_scores, 0.20),
            }
        )

    x_path = root / "Outputs" / "Uplift" / "Python" / "X-Learner" / "GLMNet" / "xlearner_scored_test_output.csv"
    if x_path.exists():
        x_df = pd.read_csv(x_path)
        n = min(len(results_test), len(x_df))
        x_scores = x_df["benefit_score"].iloc[:n] if "benefit_score" in x_df.columns else pd.Series(np.nan, index=range(n))
        cf_scores = results_test["benefit_score"].iloc[:n]
        rows.append(
            {
                "comparison": "Causal forest vs GLMNet X-learner test output",
                "comparison_basis": "row_order_test_file_no_stable_member_id",
                "n_compared": n,
                "pearson_corr": safe_corr(cf_scores, x_scores, "pearson"),
                "spearman_corr": safe_corr(cf_scores, x_scores, "spearman"),
                "top_decile_overlap": top_overlap(cf_scores, x_scores, 0.10),
                "top_20pct_overlap": top_overlap(cf_scores, x_scores, 0.20),
            }
        )

    if "current_risk_score" in results_test.columns:
        rows.append(
            {
                "comparison": "Causal forest vs current risk score test output",
                "comparison_basis": "test_set_direct_columns",
                "n_compared": len(results_test),
                "pearson_corr": safe_corr(results_test["benefit_score"], results_test["current_risk_score"], "pearson"),
                "spearman_corr": safe_corr(results_test["benefit_score"], results_test["current_risk_score"], "spearman"),
                "top_decile_overlap": top_overlap(results_test["benefit_score"], results_test["current_risk_score"], 0.10),
                "top_20pct_overlap": top_overlap(results_test["benefit_score"], results_test["current_risk_score"], 0.20),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_workflow(write_outputs=True)
