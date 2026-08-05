"""Generate the preservation-first Phase 2 PRISM synthetic causal benchmark.

This script treats ``Genrocket_10k_seed1_updated.csv`` as an immutable baseline.
It repairs deterministic feature dependencies, estimates a transparent calibrated
propensity model while preserving the observed treatment realization, and creates
coherent potential-outcome probabilities.

The script is intentionally self-contained and deterministic.  All generated files
are written under new names; the baseline CSV is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


RNG_SEED = 20260805
OUTCOME_TARGET_COUNT = 800
PROPENSITY_LOWER_BOUND = 0.08
PROPENSITY_UPPER_BOUND = 0.92

FLAG_GROUPS = {
    "clinical": [
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
    "medication": ["high_cost_drug_flag", "polypharmacy_flag", "opioid_flag"],
}


def zscore(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr))
    if std == 0:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def binary(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.upper().eq("Y").astype(int).to_numpy()


def yn(values: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(values, dtype=int) == 1, "Y", "N")


def fixed_count_draw(score: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Randomized weighted selection that preserves a binary marginal exactly."""
    score = np.asarray(score, dtype=float)
    if not 0 <= count <= len(score):
        raise ValueError(f"Invalid fixed count {count} for {len(score)} rows")
    if count == 0:
        return np.zeros(len(score), dtype=int)
    if count == len(score):
        return np.ones(len(score), dtype=int)
    randomized_priority = score + rng.gumbel(loc=0.0, scale=1.0, size=len(score))
    selected = np.argpartition(randomized_priority, -count)[-count:]
    result = np.zeros(len(score), dtype=int)
    result[selected] = 1
    return result


def quantile_map_to_reference(raw_score: np.ndarray, reference: pd.Series) -> np.ndarray:
    """Assign the exact reference multiset according to a new continuous ranking."""
    raw_score = np.asarray(raw_score, dtype=float)
    order = np.argsort(raw_score, kind="mergesort")
    sorted_reference = np.sort(pd.to_numeric(reference).to_numpy())
    mapped = np.empty(len(raw_score), dtype=sorted_reference.dtype)
    mapped[order] = sorted_reference
    return mapped


def calibrate_intercept(
    linear_predictor: np.ndarray,
    target_mean: float,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, np.ndarray]:
    """Find an intercept yielding the requested mean probability."""

    def probabilities(intercept: float) -> np.ndarray:
        values = expit(intercept + linear_predictor)
        if lower is not None or upper is not None:
            values = np.clip(values, lower if lower is not None else 0.0, upper if upper is not None else 1.0)
        return values

    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if float(probabilities(mid).mean()) < target_mean:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2.0
    return intercept, probabilities(intercept)


def risk_tier_from_score(score: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    return np.select(
        [values <= 25, values <= 50, values <= 75],
        ["Low", "Medium", "High"],
        default="Very High",
    )


def make_latent_and_repaired_features(
    baseline: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    revised = baseline.copy(deep=True)
    n = len(revised)

    age_z = zscore(revised["age"])
    utilization_z = zscore(revised["utilization_score"])
    cost_z = zscore(np.log1p(revised["total_cost_last_6m"]))
    rx_z = zscore(np.log1p(revised["rx_count_last_6m"]))
    legacy_risk_z = zscore(revised["currentRiskScore"])
    legacy_sdoh = binary(revised["food_insecurity_flag"])
    legacy_behavioral = binary(revised["behavioral_health_risk_flag"])
    legacy_metabolic = binary(revised["diabetes_flag"])
    living_alone = binary(revised["living_alone_flag"])
    non_english = revised["language"].ne("English").astype(int).to_numpy()
    rural = revised["service_region"].isin(["Rural", "Frontier"]).astype(int).to_numpy()
    female = revised["gender"].eq("F").astype(int).to_numpy()
    opioid = binary(revised["opioid_flag"])

    latent_social = zscore(
        0.30 * zscore(legacy_sdoh)
        + 0.30 * living_alone
        + 0.20 * non_english
        + 0.18 * rural
        - 0.12 * age_z
        + rng.normal(0.0, 1.0, n)
    )
    latent_behavioral = zscore(
        0.52 * latent_social
        + 0.18 * zscore(legacy_behavioral)
        - 0.18 * age_z
        + 0.12 * living_alone
        + rng.normal(0.0, 0.95, n)
    )
    latent_metabolic = zscore(
        0.55 * age_z
        + 0.34 * utilization_z
        + 0.17 * cost_z
        + 0.12 * zscore(legacy_metabolic)
        + rng.normal(0.0, 0.90, n)
    )

    original_counts = {
        col: int(revised[col].astype(str).str.upper().eq("Y").sum())
        for col in [
            *FLAG_GROUPS["clinical"],
            *FLAG_GROUPS["sdoh"],
            *FLAG_GROUPS["medication"],
            "dual_eligible",
        ]
    }

    food = fixed_count_draw(
        1.75 * latent_social + 0.18 * (1.0 - revised["med_adherence_pdc"].to_numpy()) + rng.normal(0, 0.45, n),
        original_counts["food_insecurity_flag"],
        rng,
    )
    housing = fixed_count_draw(
        1.70 * latent_social + 0.25 * living_alone + rng.normal(0, 0.48, n),
        original_counts["housing_instability_flag"],
        rng,
    )
    transportation = fixed_count_draw(
        1.55 * latent_social + 0.24 * rural + rng.normal(0, 0.52, n),
        original_counts["transportation_barrier_flag"],
        rng,
    )
    utilities = fixed_count_draw(
        1.60 * latent_social - 0.12 * cost_z + rng.normal(0, 0.52, n),
        original_counts["utilities_insecurity_flag"],
        rng,
    )

    depression = fixed_count_draw(
        2.10 * latent_behavioral + 0.12 * living_alone + rng.normal(0, 0.45, n),
        original_counts["depression_flag"],
        rng,
    )
    anxiety = fixed_count_draw(
        2.00 * latent_behavioral + 0.12 * female + rng.normal(0, 0.48, n),
        original_counts["anxiety_flag"],
        rng,
    )
    substance = fixed_count_draw(
        1.75 * latent_behavioral - 0.16 * age_z + 0.20 * opioid + rng.normal(0, 0.55, n),
        original_counts["substance_use_flag"],
        rng,
    )
    behavioral_risk = fixed_count_draw(
        1.60 * latent_behavioral
        + 0.45 * depression
        + 0.35 * anxiety
        + 0.45 * substance
        + rng.normal(0, 0.55, n),
        original_counts["behavioral_health_risk_flag"],
        rng,
    )

    diabetes = fixed_count_draw(
        1.85 * latent_metabolic + 0.20 * age_z + rng.normal(0, 0.48, n),
        original_counts["diabetes_flag"],
        rng,
    )
    ckd = fixed_count_draw(
        1.50 * latent_metabolic
        + 0.26 * legacy_risk_z
        + 0.45 * diabetes
        + rng.normal(0, 0.60, n),
        original_counts["ckd_flag"],
        rng,
    )

    repaired_flags = {
        "food_insecurity_flag": food,
        "housing_instability_flag": housing,
        "transportation_barrier_flag": transportation,
        "utilities_insecurity_flag": utilities,
        "depression_flag": depression,
        "anxiety_flag": anxiety,
        "substance_use_flag": substance,
        "behavioral_health_risk_flag": behavioral_risk,
        "diabetes_flag": diabetes,
        "ckd_flag": ckd,
    }
    for column, values in repaired_flags.items():
        revised[column] = yn(values)

    clinical_raw = (
        1.8 * diabetes
        + 1.9 * ckd
        + 2.1 * binary(revised["chf_flag"])
        + 1.3 * binary(revised["copd_flag"])
        + 0.9 * binary(revised["asthma_flag"])
        + 1.2 * depression
        + 0.9 * anxiety
        + 1.4 * substance
        + 1.1 * behavioral_risk
        + 1.1 * diabetes * ckd
        + 0.8 * depression * substance
        + rng.normal(0.0, 1.15, n)
    )
    revised["Clinical Score"] = quantile_map_to_reference(clinical_raw, baseline["Clinical Score"]).astype(int)
    revised["SDOH Score"] = food + housing + transportation + utilities

    clinical_z = zscore(revised["Clinical Score"])
    sdoh_z = zscore(revised["SDOH Score"])
    current_risk_raw = (
        0.52 * utilization_z
        + 0.36 * clinical_z
        + 0.24 * sdoh_z
        + 0.12 * cost_z
        + rng.normal(0.0, 0.28, n)
    )
    revised["currentRiskScore"] = quantile_map_to_reference(
        current_risk_raw, baseline["currentRiskScore"]
    ).astype(int)
    revised["risk_tier"] = risk_tier_from_score(revised["currentRiskScore"])

    plan_score = (
        1.15 * revised["plan_type"].isin(["Medicare", "Medicare Advantage"]).astype(int).to_numpy()
        + 0.62 * revised["plan_type"].eq("Medicaid").astype(int).to_numpy()
        - 1.20 * revised["plan_type"].isin(["Commercial", "Marketplace"]).astype(int).to_numpy()
        + 0.42 * age_z
        + 0.18 * clinical_z
        + rng.normal(0.0, 0.82, n)
    )
    dual = fixed_count_draw(plan_score, original_counts["dual_eligible"], rng)
    revised["dual_eligible"] = yn(dual)

    high_cost_drug = fixed_count_draw(
        0.40 * cost_z
        + 0.30 * rx_z
        + 2.50 * binary(revised["chf_flag"])
        + 0.17 * diabetes
        + 0.17 * ckd
        + rng.normal(0.0, 0.95, n),
        original_counts["high_cost_drug_flag"],
        rng,
    )
    polypharmacy = fixed_count_draw(
        1.38 * rx_z
        + 0.22 * clinical_z
        + 0.16 * age_z
        + 0.14 * diabetes
        + 0.12 * ckd
        + rng.normal(0.0, 0.78, n),
        original_counts["polypharmacy_flag"],
        rng,
    )
    revised["high_cost_drug_flag"] = yn(high_cost_drug)
    revised["polypharmacy_flag"] = yn(polypharmacy)

    latent_audit = pd.DataFrame(
        {
            "id": revised["id"],
            "member_id": revised["member_id"],
            "latent_social_vulnerability": latent_social,
            "latent_behavioral_health": latent_behavioral,
            "latent_metabolic_renal": latent_metabolic,
            "clinical_score_noise_rank": zscore(clinical_raw),
            "current_risk_noise_rank": zscore(current_risk_raw),
        }
    )
    return revised, latent_audit


def propensity_design(df: pd.DataFrame) -> pd.DataFrame:
    risk_z = zscore(df["currentRiskScore"])
    clinical_z = zscore(df["Clinical Score"])
    sdoh_z = zscore(df["SDOH Score"])
    utilization_z = zscore(df["utilization_score"])
    ed6_z = zscore(np.log1p(df["ed_visits_last_6m"]))
    admits_z = zscore(np.log1p(df["admits_last_6m"]))
    pdc_z = zscore(df["med_adherence_pdc"])
    age_z = zscore(df["age"])
    ccm = df["program"].eq("Complex Care Management").astype(int).to_numpy()
    behavioral = binary(df["behavioral_health_risk_flag"])
    dual = binary(df["dual_eligible"])
    diabetes = binary(df["diabetes_flag"])
    living = binary(df["living_alone_flag"])
    return pd.DataFrame(
        {
            "age_z": age_z,
            "age_over_65": np.maximum(df["age"].to_numpy() - 65.0, 0.0) / 10.0,
            "current_risk_z": risk_z,
            "clinical_score_z": clinical_z,
            "sdoh_score_z": sdoh_z,
            "utilization_score_z": utilization_z,
            "log_ed6_z": ed6_z,
            "log_admits_z": admits_z,
            "med_adherence_z": pdc_z,
            "dual_eligible": dual,
            "diabetes": diabetes,
            "behavioral_health_risk": behavioral,
            "living_alone": living,
            "complex_care_program": ccm,
            "risk_x_complex_care": risk_z * ccm,
            "behavioral_x_sdoh": behavioral * sdoh_z,
            "ed6_x_complex_care": ed6_z * ccm,
        }
    )


def fit_repaired_propensity(
    df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    x = propensity_design(df)
    treatment = binary(df["intervention_flag"])
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=5000, random_state=RNG_SEED)
    model.fit(x, treatment)
    slopes = model.coef_.ravel()
    linear_no_intercept = x.to_numpy() @ slopes
    intercept, propensity = calibrate_intercept(
        linear_no_intercept,
        float(treatment.mean()),
        PROPENSITY_LOWER_BOUND,
        PROPENSITY_UPPER_BOUND,
    )
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", *x.columns],
            "coefficient": [intercept, *slopes],
            "transformation": [
                "constant",
                "z(age)",
                "max(age-65,0)/10",
                "z(currentRiskScore)",
                "z(Clinical Score)",
                "z(SDOH Score)",
                "z(utilization_score)",
                "z(log1p(ed_visits_last_6m))",
                "z(log1p(admits_last_6m))",
                "z(med_adherence_pdc)",
                "I(dual eligible)",
                "I(diabetes)",
                "I(behavioral health risk)",
                "I(living alone)",
                "I(Complex Care Management)",
                "z(current risk) * I(Complex Care Management)",
                "I(behavioral health risk) * z(SDOH score)",
                "z(log1p(ED6)) * I(Complex Care Management)",
            ],
        }
    )
    propensity_contribution = np.array(
        [abs(intercept), *np.mean(np.abs(x.to_numpy() * slopes), axis=0)]
    )
    coefficients["mean_absolute_logit_contribution"] = propensity_contribution
    coefficients["relative_contribution_pct"] = (
        100.0 * propensity_contribution / propensity_contribution.sum()
    )
    coefficients["relative_feature_contribution_pct"] = 0.0
    feature_total = propensity_contribution[1:].sum()
    coefficients.loc[
        coefficients["term"].ne("intercept"),
        "relative_feature_contribution_pct",
    ] = 100.0 * propensity_contribution[1:] / feature_total
    return propensity, coefficients, x


def make_potential_outcomes(
    df: pd.DataFrame,
    legacy_effect: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, float, np.ndarray]:
    treatment = binary(df["intervention_flag"])
    ed30 = df["ed_visits_last_30d"].to_numpy(dtype=float)
    ed6 = df["ed_visits_last_6m"].to_numpy(dtype=float)
    admits = df["admits_last_6m"].to_numpy(dtype=float)
    ed30_any = (ed30 > 0).astype(float)
    admits_any = (admits > 0).astype(float)
    prior_ed = np.maximum(ed6 - ed30, 0.0)
    risk_z = zscore(df["currentRiskScore"])
    clinical_z = zscore(df["Clinical Score"])
    sdoh_z = zscore(df["SDOH Score"])
    age_z = zscore(df["age"])
    low_adherence = zscore(np.maximum(0.80 - df["med_adherence_pdc"].to_numpy(), 0.0))
    living = binary(df["living_alone_flag"])
    ccm = df["program"].eq("Complex Care Management").astype(int).to_numpy()
    behavioral = binary(df["behavioral_health_risk_flag"])
    substance = binary(df["substance_use_flag"])
    chf = binary(df["chf_flag"])
    copd = binary(df["copd_flag"])
    utilization_z = zscore(df["utilization_score"])
    legacy_effect_z = zscore(legacy_effect)

    baseline_terms = pd.DataFrame(
        {
            "I_ed30_positive": ed30_any,
            "log1p_ed30": np.log1p(ed30),
            "log1p_prior_ed6": np.log1p(prior_ed),
            "log1p_admits": np.log1p(admits),
            "age_z": age_z,
            "age_over_65": np.maximum(df["age"].to_numpy() - 65.0, 0.0) / 10.0,
            "current_risk_z": risk_z,
            "clinical_score_z": clinical_z,
            "sdoh_score_z": sdoh_z,
            "living_alone": living,
            "low_adherence_z": low_adherence,
            "chf": chf,
            "copd": copd,
            "ed30_x_admission": ed30_any * admits_any,
            "legacy_benefit_risk_proxy_z": legacy_effect_z,
        }
    )
    baseline_coefficients = np.array(
        [2.35, 0.62, 0.40, 0.32, 0.08, 0.18, 0.30, 0.17, 0.13, 0.12, 0.11, 0.16, 0.12, 0.22, 0.45]
    )
    baseline_linear = baseline_terms.to_numpy() @ baseline_coefficients

    response_terms = pd.DataFrame(
        {
            "legacy_effect_z": legacy_effect_z,
            "complex_care_program": ccm,
            "behavioral_health_risk": behavioral,
            "substance_use": substance,
            "sdoh_score_z": sdoh_z,
            "clinical_score_z": clinical_z,
            "low_adherence_z": low_adherence,
            "utilization_score_z": utilization_z,
            "complex_care_x_utilization": ccm * utilization_z,
            "behavioral_x_sdoh": behavioral * sdoh_z,
            "clinical_x_behavioral": clinical_z * behavioral,
        }
    )
    response_coefficients = np.array([0.65, 0.24, 0.18, 0.17, 0.13, 0.12, 0.10, 0.08, 0.18, 0.15, 0.10])
    response_linear = -0.25 + response_terms.to_numpy() @ response_coefficients
    relative_risk_reduction = 0.04 + 0.31 * expit(response_linear)

    def factual_probabilities(intercept: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu0_values = expit(intercept + baseline_linear)
        mu1_values = mu0_values * (1.0 - relative_risk_reduction)
        factual_values = np.where(treatment == 1, mu1_values, mu0_values)
        return mu0_values, mu1_values, factual_values

    lo, hi = -20.0, 20.0
    target_rate = OUTCOME_TARGET_COUNT / len(df)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if float(factual_probabilities(mid)[2].mean()) < target_rate:
            lo = mid
        else:
            hi = mid
    outcome_intercept = (lo + hi) / 2.0
    mu0, mu1, factual = factual_probabilities(outcome_intercept)

    coefficient_rows = [
        {
            "function": "baseline_outcome_logit",
            "term": "intercept",
            "coefficient": outcome_intercept,
            "mean_absolute_linear_contribution": abs(outcome_intercept),
        },
        *[
            {
                "function": "baseline_outcome_logit",
                "term": term,
                "coefficient": coefficient,
                "mean_absolute_linear_contribution": float(
                    np.mean(np.abs(baseline_terms[term].to_numpy() * coefficient))
                ),
            }
            for term, coefficient in zip(
                baseline_terms.columns, baseline_coefficients
            )
        ],
        {
            "function": "relative_response_logit",
            "term": "intercept",
            "coefficient": -0.25,
            "mean_absolute_linear_contribution": 0.25,
        },
        *[
            {
                "function": "relative_response_logit",
                "term": term,
                "coefficient": coefficient,
                "mean_absolute_linear_contribution": float(
                    np.mean(np.abs(response_terms[term].to_numpy() * coefficient))
                ),
            }
            for term, coefficient in zip(
                response_terms.columns, response_coefficients
            )
        ],
        {
            "function": "relative_response",
            "term": "lower_bound",
            "coefficient": 0.04,
            "mean_absolute_linear_contribution": np.nan,
        },
        {
            "function": "relative_response",
            "term": "sigmoid_range",
            "coefficient": 0.31,
            "mean_absolute_linear_contribution": np.nan,
        },
    ]
    coefficient_frame = pd.DataFrame(coefficient_rows)
    coefficient_frame["relative_contribution_pct"] = np.nan
    coefficient_frame["relative_feature_contribution_pct"] = np.nan
    for function in ["baseline_outcome_logit", "relative_response_logit"]:
        mask = coefficient_frame["function"].eq(function)
        denominator = coefficient_frame.loc[
            mask, "mean_absolute_linear_contribution"
        ].sum()
        coefficient_frame.loc[mask, "relative_contribution_pct"] = (
            100.0
            * coefficient_frame.loc[mask, "mean_absolute_linear_contribution"]
            / denominator
        )
        feature_mask = mask & coefficient_frame["term"].ne("intercept")
        feature_denominator = coefficient_frame.loc[
            feature_mask, "mean_absolute_linear_contribution"
        ].sum()
        coefficient_frame.loc[
            feature_mask, "relative_feature_contribution_pct"
        ] = (
            100.0
            * coefficient_frame.loc[
                feature_mask, "mean_absolute_linear_contribution"
            ]
            / feature_denominator
        )
    return (
        mu0,
        mu1,
        relative_risk_reduction,
        coefficient_frame,
        outcome_intercept,
        factual,
    )


def draw_exact_outcomes(probability: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Find a documented Bernoulli seed producing exactly the preserved event count."""
    best: tuple[int, np.ndarray, int] | None = None
    for offset in range(100_000):
        seed = RNG_SEED + 100_000 + offset
        outcome = (np.random.default_rng(seed).random(len(probability)) < probability).astype(int)
        distance = abs(int(outcome.sum()) - OUTCOME_TARGET_COUNT)
        if best is None or distance < best[0]:
            best = (distance, outcome, seed)
        if distance == 0:
            zero_ed_events = int(outcome[df["ed_visits_last_30d"].eq(0).to_numpy()].sum())
            positive_ed_nonevents = int(
                (1 - outcome[df["ed_visits_last_30d"].gt(0).to_numpy()]).sum()
            )
            if zero_ed_events > 0 and positive_ed_nonevents > 0:
                return outcome, seed
    assert best is not None
    raise RuntimeError(
        f"No exact Bernoulli realization found; closest count was {int(best[1].sum())} at seed {best[2]}"
    )


def pairwise_phi(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame({column: binary(df[column]) for column in columns}).corr()
    rows: list[dict[str, float | str]] = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            rows.append({"feature_1": left, "feature_2": right, "correlation": matrix.loc[left, right]})
    return pd.DataFrame(rows).sort_values("correlation", key=lambda x: x.abs(), ascending=False)


def summarize_distribution(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric = [
        "age",
        "Clinical Score",
        "SDOH Score",
        "utilization_score",
        "currentRiskScore",
        "pcp_visits_last_6m",
        "specialist_visits_last_6m",
        "ed_visits_last_30d",
        "ed_visits_last_6m",
        "admits_last_6m",
        "observation_stays_last_6m",
        "total_cost_last_6m",
        "rx_count_last_6m",
        "med_adherence_pdc",
    ]
    for column in numeric:
        for metric, function in {
            "mean": np.mean,
            "std": lambda x: np.std(x, ddof=1),
            "median": np.median,
            "p05": lambda x: np.quantile(x, 0.05),
            "p95": lambda x: np.quantile(x, 0.95),
        }.items():
            old_value = float(function(pd.to_numeric(original[column])))
            new_value = float(function(pd.to_numeric(revised[column])))
            rows.append(
                {
                    "variable": column,
                    "metric": metric,
                    "original": old_value,
                    "revised": new_value,
                    "absolute_difference": abs(new_value - old_value),
                }
            )
    return pd.DataFrame(rows)


def prevalence_table(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "living_alone_flag",
        "dual_eligible",
        *FLAG_GROUPS["clinical"],
        *FLAG_GROUPS["sdoh"],
        "pregnancy_flag",
        *FLAG_GROUPS["medication"],
    ]
    rows = []
    for column in flags:
        old = float(binary(original[column]).mean())
        new = float(binary(revised[column]).mean())
        rows.append(
            {
                "variable": column,
                "original_prevalence": old,
                "revised_prevalence": new,
                "difference_percentage_points": 100.0 * (new - old),
            }
        )
    return pd.DataFrame(rows)


def ed_validation(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, data in [("original", original), ("revised", revised)]:
        y = data["outcome_ed_90d"].astype(int)
        ed = data["ed_visits_last_30d"].astype(int)
        rows.append(
            {
                "dataset": label,
                "overall_event_rate": float(y.mean()),
                "event_rate_ed30_zero": float(y[ed == 0].mean()),
                "event_rate_ed30_positive": float(y[ed > 0].mean()),
                "events_ed30_zero": int(y[ed == 0].sum()),
                "non_events_ed30_positive": int((1 - y[ed > 0]).sum()),
                "ed30_auc": float(roc_auc_score(y, ed)),
                "ed30_outcome_correlation": float(np.corrcoef(ed, y)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def treatment_validation(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    treatment = binary(revised["intervention_flag"])
    rows: list[dict[str, object]] = []
    for label, data, propensity_column in [
        ("legacy", original, "Propensity_Score"),
        ("repaired", revised, "propensity_score"),
    ]:
        p = data[propensity_column].astype(float).to_numpy()
        rows.extend(
            [
                {"dataset": label, "metric": "treatment_prevalence", "value": float(treatment.mean())},
                {"dataset": label, "metric": "mean_propensity", "value": float(p.mean())},
                {"dataset": label, "metric": "propensity_min", "value": float(p.min())},
                {"dataset": label, "metric": "propensity_p05", "value": float(np.quantile(p, 0.05))},
                {"dataset": label, "metric": "propensity_p50", "value": float(np.quantile(p, 0.50))},
                {"dataset": label, "metric": "propensity_p95", "value": float(np.quantile(p, 0.95))},
                {"dataset": label, "metric": "propensity_max", "value": float(p.max())},
                {"dataset": label, "metric": "propensity_brier", "value": float(brier_score_loss(treatment, p))},
                {"dataset": label, "metric": "propensity_auc", "value": float(roc_auc_score(treatment, p))},
            ]
        )
    return pd.DataFrame(rows)


def effect_validation(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    legacy = original["true_treatment_effect"].astype(float).to_numpy()
    effect = revised["true_treatment_effect"].astype(float).to_numpy()
    treatment = binary(revised["intervention_flag"])
    identity_error = np.abs(
        revised["mu0"].to_numpy() - revised["mu1"].to_numpy() - effect
    )
    return pd.DataFrame(
        [
            {"metric": "legacy_effect_mean", "value": float(legacy.mean())},
            {"metric": "revised_effect_mean", "value": float(effect.mean())},
            {"metric": "revised_effect_std", "value": float(effect.std(ddof=1))},
            {"metric": "revised_effect_min", "value": float(effect.min())},
            {"metric": "revised_effect_max", "value": float(effect.max())},
            {"metric": "legacy_revised_spearman", "value": float(spearmanr(legacy, effect).statistic)},
            {"metric": "revised_ATT", "value": float(effect[treatment == 1].mean())},
            {"metric": "revised_ATC", "value": float(effect[treatment == 0].mean())},
            {"metric": "max_mu_identity_error", "value": float(identity_error.max())},
            {"metric": "mu0_mean", "value": float(revised["mu0"].mean())},
            {"metric": "mu1_mean", "value": float(revised["mu1"].mean())},
        ]
    )


def preservation_audit(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in original.columns:
        left = original[column].fillna("__ACTUAL_NULL__").astype(str)
        right = revised[column].fillna("__ACTUAL_NULL__").astype(str)
        rows.append(
            {
                "column": column,
                "changed_rows": int((left != right).sum()),
                "row_values_identical": bool(left.equals(right)),
                "marginal_distribution_identical": bool(
                    left.value_counts(dropna=False).sort_index().equals(
                        right.value_counts(dropna=False).sort_index()
                    )
                ),
                "status": "modified" if not left.equals(right) else "preserved",
            }
        )
    for column in revised.columns:
        if column not in original.columns:
            rows.append(
                {
                    "column": column,
                    "changed_rows": len(revised),
                    "row_values_identical": False,
                    "marginal_distribution_identical": False,
                    "status": "added",
                }
            )
    return pd.DataFrame(rows)


def propensity_calibration_by_decile(revised: pd.DataFrame) -> pd.DataFrame:
    result = revised[["propensity_score", "intervention_flag"]].copy()
    result["treated"] = binary(result["intervention_flag"])
    result["propensity_decile"] = pd.qcut(
        result["propensity_score"], 10, labels=False, duplicates="drop"
    ) + 1
    return (
        result.groupby("propensity_decile", as_index=False)
        .agg(
            members=("treated", "size"),
            mean_propensity=("propensity_score", "mean"),
            observed_treatment_rate=("treated", "mean"),
        )
        .assign(
            calibration_difference=lambda x: x["observed_treatment_rate"]
            - x["mean_propensity"]
        )
    )


def effect_rank_validation(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "legacy_effect": original["true_treatment_effect"].astype(float),
            "revised_effect": revised["true_treatment_effect"].astype(float),
        }
    )
    frame["legacy_decile"] = pd.qcut(
        frame["legacy_effect"].rank(method="first"), 10, labels=False
    ) + 1
    return (
        frame.groupby("legacy_decile", as_index=False)
        .agg(
            members=("revised_effect", "size"),
            mean_legacy_effect=("legacy_effect", "mean"),
            mean_revised_effect=("revised_effect", "mean"),
            median_revised_effect=("revised_effect", "median"),
        )
    )


def selected_correlation_comparison(original: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("food_insecurity_flag", "housing_instability_flag"),
        ("food_insecurity_flag", "transportation_barrier_flag"),
        ("food_insecurity_flag", "utilities_insecurity_flag"),
        ("depression_flag", "anxiety_flag"),
        ("depression_flag", "behavioral_health_risk_flag"),
        ("depression_flag", "substance_use_flag"),
        ("diabetes_flag", "ckd_flag"),
        ("chf_flag", "high_cost_drug_flag"),
        ("polypharmacy_flag", "rx_count_last_6m"),
        ("program", "polypharmacy_flag"),
    ]

    def corr(data: pd.DataFrame, left: str, right: str) -> float:
        left_values = binary(data[left]) if data[left].dtype == object or str(data[left].dtype).startswith("str") else data[left].to_numpy(dtype=float)
        if right == "program":
            right_values = data[right].eq("Complex Care Management").astype(int).to_numpy()
        else:
            right_values = binary(data[right]) if data[right].dtype == object or str(data[right].dtype).startswith("str") else data[right].to_numpy(dtype=float)
        if left == "program":
            left_values = data[left].eq("Complex Care Management").astype(int).to_numpy()
        return float(np.corrcoef(left_values, right_values)[0, 1])

    return pd.DataFrame(
        [
            {
                "feature_1": left,
                "feature_2": right,
                "original_correlation": corr(original, left, right),
                "revised_correlation": corr(revised, left, right),
            }
            for left, right in pairs
        ]
    )


def save_plots(original: pd.DataFrame, revised: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(original["Propensity_Score"], bins=np.linspace(0, 1, 31), alpha=0.55, label="Legacy", color="#6B7280")
    ax.hist(revised["propensity_score"], bins=np.linspace(0, 1, 31), alpha=0.65, label="Repaired", color="#2563EB")
    ax.set(xlabel="Propensity score", ylabel="Members", title="Legacy and repaired propensity distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "propensity_distribution_comparison.png", dpi=180)
    plt.close(fig)

    ed_rates = []
    for label, data in [("Original", original), ("Revised", revised)]:
        for group, mask in [("ED30 = 0", data["ed_visits_last_30d"].eq(0)), ("ED30 > 0", data["ed_visits_last_30d"].gt(0))]:
            ed_rates.append({"dataset": label, "group": group, "rate": data.loc[mask, "outcome_ed_90d"].mean()})
    rate_df = pd.DataFrame(ed_rates)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.36
    for index, label in enumerate(["Original", "Revised"]):
        values = rate_df.loc[rate_df["dataset"].eq(label), "rate"].to_numpy()
        ax.bar(x + (index - 0.5) * width, values, width, label=label)
    ax.set_xticks(x, ["ED30 = 0", "ED30 > 0"])
    ax.set(ylabel="90-day ED outcome rate", title="ED history remains predictive without determinism", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ed30_outcome_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(original["true_treatment_effect"], bins=24, alpha=0.50, label="Legacy score", color="#6B7280")
    ax.hist(revised["true_treatment_effect"], bins=30, alpha=0.70, label="Revised probability effect", color="#059669")
    ax.set(xlabel="Individual absolute risk reduction", ylabel="Members", title="Legacy and probability-coherent treatment effect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "treatment_effect_distribution_comparison.png", dpi=180)
    plt.close(fig)

    heat_columns = [
        "diabetes_flag",
        "ckd_flag",
        "depression_flag",
        "anxiety_flag",
        "substance_use_flag",
        "behavioral_health_risk_flag",
        *FLAG_GROUPS["sdoh"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), sharex=True, sharey=True)
    for ax, label, data in zip(axes, ["Original", "Revised"], [original, revised]):
        matrix = pd.DataFrame({column: binary(data[column]) for column in heat_columns}).corr()
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(label)
        ax.set_xticks(range(len(heat_columns)), heat_columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(heat_columns)), heat_columns, fontsize=7)
    colorbar_axis = fig.add_axes([0.91, 0.22, 0.016, 0.58])
    fig.colorbar(image, cax=colorbar_axis, label="Pearson/phi correlation")
    fig.suptitle("Clinical and SDOH dependency repair")
    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.29, top=0.86, wspace=0.16)
    fig.savefig(output_dir / "dependency_correlation_heatmaps.png", dpi=180)
    plt.close(fig)


def assert_invariants(original: pd.DataFrame, revised: pd.DataFrame) -> None:
    assert len(original) == len(revised) == 10_000
    assert original["id"].equals(revised["id"])
    assert original["member_id"].equals(revised["member_id"])
    assert original["intervention_flag"].equals(revised["intervention_flag"])
    assert original["treatmentAssignment"].equals(revised["treatmentAssignment"])
    assert int(revised["outcome_ed_90d"].sum()) == OUTCOME_TARGET_COUNT
    assert revised.loc[revised["ed_visits_last_30d"].eq(0), "outcome_ed_90d"].sum() > 0
    assert (1 - revised.loc[revised["ed_visits_last_30d"].gt(0), "outcome_ed_90d"]).sum() > 0
    assert np.all((revised["mu0"] >= 0) & (revised["mu0"] <= 1))
    assert np.all((revised["mu1"] >= 0) & (revised["mu1"] <= 1))
    assert np.all(revised["mu1"] <= revised["mu0"])
    assert np.max(np.abs(revised["mu0"] - revised["mu1"] - revised["true_treatment_effect"])) < 1e-12
    assert np.all((revised["propensity_score"] >= PROPENSITY_LOWER_BOUND) & (revised["propensity_score"] <= PROPENSITY_UPPER_BOUND))
    assert math.isclose(binary(revised["intervention_flag"]).mean(), revised["propensity_score"].mean(), abs_tol=1e-12)

    exact_prevalence_flags = [
        "dual_eligible",
        "diabetes_flag",
        "depression_flag",
        "anxiety_flag",
        "substance_use_flag",
        "ckd_flag",
        "behavioral_health_risk_flag",
        *FLAG_GROUPS["sdoh"],
        "high_cost_drug_flag",
        "polypharmacy_flag",
    ]
    for column in exact_prevalence_flags:
        assert int(binary(original[column]).sum()) == int(binary(revised[column]).sum()), column

    intentionally_changed = {
        "dual_eligible",
        "diabetes_flag",
        "depression_flag",
        "anxiety_flag",
        "substance_use_flag",
        "ckd_flag",
        "behavioral_health_risk_flag",
        *FLAG_GROUPS["sdoh"],
        "Clinical Score",
        "SDOH Score",
        "currentRiskScore",
        "risk_tier",
        "high_cost_drug_flag",
        "polypharmacy_flag",
        "outcome_ed_90d",
        "Propensity_Score",
        "true_treatment_effect",
    }
    for column in original.columns:
        if column not in intentionally_changed:
            assert original[column].equals(revised[column]), f"Unexpected change in {column}"

    corr = selected_correlation_comparison(original, revised).set_index(
        ["feature_1", "feature_2"]
    )["revised_correlation"]
    for pair in [
        ("food_insecurity_flag", "housing_instability_flag"),
        ("food_insecurity_flag", "transportation_barrier_flag"),
        ("food_insecurity_flag", "utilities_insecurity_flag"),
    ]:
        assert 0.25 <= corr.loc[pair] <= 0.60
    assert 0.35 <= corr.loc[("depression_flag", "anxiety_flag")] <= 0.70
    assert 0.35 <= corr.loc[("diabetes_flag", "ckd_flag")] <= 0.60
    assert corr.loc[("polypharmacy_flag", "rx_count_last_6m")] > 0.30


def generate(project_root: Path) -> dict[str, object]:
    baseline_path = project_root / "DataSets" / "Genrocket_10k_seed1_updated.csv"
    revised_path = project_root / "DataSets" / "Genrocket_10k_seed1_causal_revised.csv"
    output_dir = project_root / "Outputs" / "Synthetic-Data-Validation" / "Phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(baseline_path, low_memory=False)
    rng = np.random.default_rng(RNG_SEED)
    revised, latent_audit = make_latent_and_repaired_features(baseline, rng)

    legacy_ground_truth_audit = pd.DataFrame(
        {
            "id": baseline["id"],
            "member_id": baseline["member_id"],
            "legacy_propensity_score": baseline["Propensity_Score"].astype(float),
            "legacy_true_treatment_effect": baseline["true_treatment_effect"].astype(float),
            "legacy_outcome_ed_90d": baseline["outcome_ed_90d"].astype(int),
        }
    )

    propensity, propensity_coefficients, propensity_x = fit_repaired_propensity(revised)
    revised["propensity_score"] = propensity
    revised["Propensity_Score"] = propensity

    mu0, mu1, relative_response, outcome_coefficients, outcome_intercept, factual_probability = make_potential_outcomes(
        revised, baseline["true_treatment_effect"].astype(float).to_numpy()
    )
    revised["mu0"] = mu0
    revised["mu1"] = mu1
    revised["true_treatment_effect"] = mu0 - mu1
    revised["relative_risk_reduction"] = relative_response
    revised["factual_outcome_probability"] = factual_probability
    outcome, outcome_seed = draw_exact_outcomes(factual_probability, revised)
    revised["outcome_ed_90d"] = outcome
    revised["dgp_version"] = "PRISM-preservation-first-v1"
    revised["dgp_seed"] = RNG_SEED
    revised["outcome_draw_seed"] = outcome_seed

    latent_audit["treatment"] = binary(revised["intervention_flag"])
    latent_audit["propensity_score"] = propensity
    latent_audit["relative_risk_reduction"] = relative_response
    latent_audit["factual_outcome_probability"] = factual_probability

    assert_invariants(baseline, revised)

    revised.to_csv(revised_path, index=False)
    latent_audit.to_csv(output_dir / "latent_variable_audit.csv", index=False)
    legacy_ground_truth_audit.to_csv(
        output_dir / "legacy_ground_truth_audit.csv", index=False
    )
    propensity_x.assign(id=revised["id"], member_id=revised["member_id"]).to_csv(
        output_dir / "propensity_design_matrix.csv", index=False
    )
    propensity_coefficients.to_csv(output_dir / "propensity_function_coefficients.csv", index=False)
    outcome_coefficients.to_csv(output_dir / "outcome_and_response_function_coefficients.csv", index=False)

    distribution = summarize_distribution(baseline, revised)
    prevalence = prevalence_table(baseline, revised)
    correlation = selected_correlation_comparison(baseline, revised)
    ed = ed_validation(baseline, revised)
    treatment = treatment_validation(baseline, revised)
    effect = effect_validation(baseline, revised)
    preservation = preservation_audit(baseline, revised)
    propensity_calibration = propensity_calibration_by_decile(revised)
    effect_rank = effect_rank_validation(baseline, revised)
    all_binary_columns = [
        "dual_eligible",
        *FLAG_GROUPS["clinical"],
        *FLAG_GROUPS["sdoh"],
        *FLAG_GROUPS["medication"],
    ]
    strongest_original = pairwise_phi(baseline, all_binary_columns).head(40)
    strongest_revised = pairwise_phi(revised, all_binary_columns).head(40)

    distribution.to_csv(output_dir / "distribution_validation.csv", index=False)
    prevalence.to_csv(output_dir / "prevalence_validation.csv", index=False)
    correlation.to_csv(output_dir / "correlation_validation.csv", index=False)
    ed.to_csv(output_dir / "ed_visit_validation.csv", index=False)
    treatment.to_csv(output_dir / "treatment_validation.csv", index=False)
    effect.to_csv(output_dir / "treatment_effect_validation.csv", index=False)
    preservation.to_csv(output_dir / "column_preservation_audit.csv", index=False)
    propensity_calibration.to_csv(output_dir / "propensity_calibration_by_decile.csv", index=False)
    effect_rank.to_csv(output_dir / "legacy_effect_rank_validation.csv", index=False)
    strongest_original.to_csv(output_dir / "strongest_binary_correlations_original.csv", index=False)
    strongest_revised.to_csv(output_dir / "strongest_binary_correlations_revised.csv", index=False)
    save_plots(baseline, revised, output_dir)

    metrics = {
        "baseline_path": str(baseline_path),
        "revised_path": str(revised_path),
        "validation_output_dir": str(output_dir),
        "rows": len(revised),
        "columns": len(revised.columns),
        "rng_seed": RNG_SEED,
        "outcome_draw_seed": outcome_seed,
        "treatment_prevalence": float(binary(revised["intervention_flag"]).mean()),
        "outcome_prevalence": float(revised["outcome_ed_90d"].mean()),
        "mean_propensity": float(revised["propensity_score"].mean()),
        "propensity_min": float(revised["propensity_score"].min()),
        "propensity_max": float(revised["propensity_score"].max()),
        "ed30_zero_event_rate": float(revised.loc[revised["ed_visits_last_30d"].eq(0), "outcome_ed_90d"].mean()),
        "ed30_positive_event_rate": float(revised.loc[revised["ed_visits_last_30d"].gt(0), "outcome_ed_90d"].mean()),
        "ed30_auc": float(roc_auc_score(revised["outcome_ed_90d"], revised["ed_visits_last_30d"])),
        "ate": float(revised["true_treatment_effect"].mean()),
        "att": float(revised.loc[revised["intervention_flag"].eq("Y"), "true_treatment_effect"].mean()),
        "atc": float(revised.loc[revised["intervention_flag"].eq("N"), "true_treatment_effect"].mean()),
        "legacy_effect_spearman": float(
            spearmanr(
                baseline["true_treatment_effect"],
                revised["true_treatment_effect"],
            ).statistic
        ),
        "max_mu_identity_error": float(np.abs(revised["mu0"] - revised["mu1"] - revised["true_treatment_effect"]).max()),
        "outcome_intercept": outcome_intercept,
    }
    with (output_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Inner PRISM project root containing DataSets, Code, and Outputs.",
    )
    args = parser.parse_args()
    metrics = generate(args.project_root.resolve())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
