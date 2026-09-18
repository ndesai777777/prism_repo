"""Generate Funds Task 7 targeting-policy tables and charts.

This applies the same business-value methodology used in the PRP report to the
held-out Funds test population for the XGBoost T-learner and X-learner.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from _prism_model_utils import split_train_test  # noqa: E402


OUTPUT_ROOT = ROOT / "Outputs" / "Uplift_Funds" / "Python"
ED_VISIT_COST = 1_200.0


def held_out_tlearner_scores() -> pd.DataFrame:
    """Recreate the notebook's stratified held-out test set."""
    scored = pd.read_csv(
        OUTPUT_ROOT / "T-Learner" / "XGBoost" / "uplift_scored_output.csv"
    )
    _, test_df = split_train_test(
        scored,
        train_fraction=0.70,
        seed=123,
        stratify_columns=["intervention_flag", "outcome_ed_90d"],
    )
    return test_df.reset_index(drop=True)


def cumulative_targeting(scored: pd.DataFrame) -> pd.DataFrame:
    """Compare uplift and current-risk rankings on identical population sizes."""
    n_total = len(scored)
    decile_size = max(1, n_total // 10)
    rows: list[dict[str, float | int | str]] = []

    for approach, sort_column in [
        ("Uplift score", "benefit_score"),
        ("Current risk score", "current_risk_score"),
    ]:
        ranked = scored.sort_values(sort_column, ascending=False).reset_index(drop=True)
        for decile in range(1, 11):
            top_n = n_total if decile == 10 else decile * decile_size
            selected = ranked.head(top_n)
            avoided = float(selected["benefit_score"].sum())
            rows.append(
                {
                    "targeting_approach": approach,
                    "through_decile": decile,
                    "population_fraction_targeted": top_n / n_total,
                    "n": top_n,
                    "cumulative_estimated_ed_visits_avoided": avoided,
                    "cumulative_gross_savings": avoided * ED_VISIT_COST,
                }
            )

    return pd.DataFrame(rows)


def marginal_targeting(cumulative: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate each expansion band's value and uplift-over-risk advantage."""
    marginal_parts: list[pd.DataFrame] = []
    for approach, group in cumulative.groupby("targeting_approach", sort=False):
        group = group.sort_values("through_decile").copy()
        group["targeting_band"] = group["through_decile"].map(
            lambda d: f"Top {(int(d) - 1) * 10}-{int(d) * 10}%"
        )
        group["additional_members"] = group["n"].diff().fillna(group["n"])
        group["additional_ed_visits_avoided"] = (
            group["cumulative_estimated_ed_visits_avoided"]
            .diff()
            .fillna(group["cumulative_estimated_ed_visits_avoided"])
        )
        group["additional_gross_savings"] = (
            group["cumulative_gross_savings"]
            .diff()
            .fillna(group["cumulative_gross_savings"])
        )
        marginal_parts.append(
            group[
                [
                    "targeting_approach",
                    "through_decile",
                    "targeting_band",
                    "additional_members",
                    "additional_ed_visits_avoided",
                    "additional_gross_savings",
                ]
            ]
        )

    marginal = pd.concat(marginal_parts, ignore_index=True)
    uplift = marginal[marginal["targeting_approach"] == "Uplift score"].reset_index(drop=True)
    risk = marginal[marginal["targeting_approach"] == "Current risk score"].reset_index(drop=True)
    advantage = pd.DataFrame(
        {
            "targeting_approach_model": "XGBoost benefit score",
            "through_decile": uplift["through_decile"],
            "targeting_band": uplift["targeting_band"],
            "additional_members": uplift["additional_members"].astype(int),
            "additional_ed_visits_avoided_model": uplift["additional_ed_visits_avoided"],
            "additional_gross_savings_model": uplift["additional_gross_savings"],
            "targeting_approach_current_risk": "Current risk score",
            "additional_ed_visits_avoided_current_risk": risk["additional_ed_visits_avoided"],
            "additional_gross_savings_current_risk": risk["additional_gross_savings"],
            "additional_ed_visits_avoided_advantage": (
                uplift["additional_ed_visits_avoided"]
                - risk["additional_ed_visits_avoided"]
            ),
            "additional_gross_savings_advantage": (
                uplift["additional_gross_savings"]
                - risk["additional_gross_savings"]
            ),
        }
    )
    return marginal, advantage


def save_cumulative_chart(cumulative: pd.DataFrame, model_name: str, path: Path) -> None:
    chart_df = cumulative[cumulative["population_fraction_targeted"] <= 0.50]
    fig, ax = plt.subplots(figsize=(8.5, 5.25))
    for approach, group in chart_df.groupby("targeting_approach"):
        ax.plot(
            group["population_fraction_targeted"] * 100,
            group["cumulative_gross_savings"],
            marker="o",
            linewidth=2,
            label=approach,
        )
    ax.set_title(f"{model_name} Cumulative Gross Savings Through Top Targeted Deciles")
    ax.set_xlabel("Population targeted (%)")
    ax.set_ylabel("Cumulative gross savings")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_marginal_chart(advantage: pd.DataFrame, model_name: str, path: Path) -> None:
    values = advantage["additional_gross_savings_advantage"]
    colors = ["#377fd0" if value >= 0 else "#ba4d4a" for value in values]
    fig, ax = plt.subplots(figsize=(12, 6.67))
    ax.bar(advantage["targeting_band"], values, color=colors)
    ax.axhline(0, color="#333333", linewidth=1.2)
    ax.set_title(
        f"{model_name}: Additional Gross Savings Advantage vs Current Risk",
        fontsize=17,
    )
    ax.set_xlabel("Targeting expansion band", fontsize=13)
    ax.set_ylabel("Additional gross savings advantage", fontsize=13)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_top50_summary(cumulative: pd.DataFrame, model_key: str, path: Path) -> None:
    rows = []
    for decile in range(1, 6):
        uplift = cumulative[
            (cumulative["targeting_approach"] == "Uplift score")
            & (cumulative["through_decile"] == decile)
        ].iloc[0]
        risk = cumulative[
            (cumulative["targeting_approach"] == "Current risk score")
            & (cumulative["through_decile"] == decile)
        ].iloc[0]
        rows.append(
            {
                "targeted_group": f"Top {decile * 10}%",
                "members_targeted": int(uplift["n"]),
                f"{model_key}_gross_savings": uplift["cumulative_gross_savings"],
                "current_risk_gross_savings": risk["cumulative_gross_savings"],
                f"{model_key}_advantage": (
                    uplift["cumulative_gross_savings"]
                    - risk["cumulative_gross_savings"]
                ),
                f"{model_key}_ed_visits_avoided": uplift[
                    "cumulative_estimated_ed_visits_avoided"
                ],
                "current_risk_ed_visits_avoided": risk[
                    "cumulative_estimated_ed_visits_avoided"
                ],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def generate_for_model(
    scored: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    model_key: str,
) -> None:
    cumulative = cumulative_targeting(scored)
    marginal, advantage = marginal_targeting(cumulative)

    cumulative.to_csv(output_dir / "cumulative_gross_savings_by_targeting.csv", index=False)
    marginal.to_csv(output_dir / "marginal_gross_savings_by_targeting.csv", index=False)
    advantage.to_csv(
        output_dir / "marginal_gross_savings_advantage_vs_current_risk.csv",
        index=False,
    )
    save_top50_summary(
        cumulative,
        model_key,
        output_dir / "cumulative_gross_savings_summary_top50.csv",
    )
    save_cumulative_chart(
        cumulative,
        model_name,
        output_dir / "dashboard_cumulative_gross_savings_targeting.png",
    )
    save_marginal_chart(
        advantage,
        model_name,
        output_dir / "dashboard_marginal_gross_savings_advantage_vs_current_risk.png",
    )


def main() -> None:
    t_dir = OUTPUT_ROOT / "T-Learner" / "XGBoost"
    x_dir = OUTPUT_ROOT / "X-Learner" / "XGBoost"
    generate_for_model(
        held_out_tlearner_scores(),
        t_dir,
        "XGBoost T-Learner",
        "t_learner",
    )
    generate_for_model(
        pd.read_csv(x_dir / "xlearner_scored_test_output.csv"),
        x_dir,
        "XGBoost X-Learner",
        "x_learner",
    )


if __name__ == "__main__":
    main()
