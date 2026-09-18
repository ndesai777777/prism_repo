"""Regenerate Funds risk-tier charts using equal predicted-benefit terciles."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

from _prism_model_utils import split_train_test  # noqa: E402


OUTPUT_ROOT = ROOT / "Outputs" / "Uplift_Funds" / "Python"
RISK_TIER_ORDER = ["1", "2", "3", "4", "5"]
RISK_TIER_DISPLAY = {
    "1": "Tier 1 (lowest risk)",
    "2": "Tier 2",
    "3": "Tier 3",
    "4": "Tier 4",
    "5": "Tier 5 (highest risk)",
}
BENEFIT_GROUP_ORDER = ["High benefit", "Medium benefit", "Low benefit"]
BENEFIT_GROUP_COLORS = {
    "High benefit": "#1f77b4",
    "Medium benefit": "#70ad47",
    "Low benefit": "#a6a6a6",
}
BENEFIT_GROUP_DEFINITIONS = {
    "High benefit": "predicted-benefit tercile (top third)",
    "Medium benefit": "predicted-benefit tercile (middle third)",
    "Low benefit": "predicted-benefit tercile (bottom third)",
}


def held_out_tlearner_scores(path: Path) -> pd.DataFrame:
    """Recreate the notebook's stratified held-out test population."""
    scored = pd.read_csv(path)
    _, test_df = split_train_test(
        scored,
        train_fraction=0.70,
        seed=123,
        stratify_columns=["intervention_flag", "outcome_ed_90d"],
    )
    return test_df.reset_index(drop=True)


def assign_benefit_terciles(benefit_score: pd.Series) -> pd.Categorical:
    """Assign equal-frequency benefit groups, breaking ties by input order."""
    benefit_rank = benefit_score.rank(method="first", ascending=False)
    return pd.qcut(benefit_rank, q=3, labels=BENEFIT_GROUP_ORDER)


def build_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize tercile composition within the original Funds risk tiers."""
    df = scored.copy()
    df["benefit_group"] = assign_benefit_terciles(df["benefit_score"])
    df["risk_tier"] = pd.to_numeric(df["risk_tier"], errors="coerce").astype("Int64").astype(str)
    df = df.loc[df["risk_tier"].isin(RISK_TIER_ORDER)].copy()
    df["risk_tier"] = pd.Categorical(
        df["risk_tier"], categories=RISK_TIER_ORDER, ordered=True
    )
    df["benefit_group"] = pd.Categorical(
        df["benefit_group"], categories=BENEFIT_GROUP_ORDER, ordered=True
    )

    summary = (
        df.groupby(["risk_tier", "benefit_group"], observed=False)
        .size()
        .reset_index(name="members")
    )
    tier_totals = (
        df.groupby("risk_tier", observed=False)
        .size()
        .rename("risk_tier_members")
        .reset_index()
    )
    summary = summary.merge(tier_totals, on="risk_tier", how="left")
    summary["pct_within_risk_tier"] = np.where(
        summary["risk_tier_members"] > 0,
        summary["members"] / summary["risk_tier_members"],
        np.nan,
    )
    summary["benefit_group_definition"] = summary["benefit_group"].map(
        BENEFIT_GROUP_DEFINITIONS
    )
    return summary


def save_chart(summary: pd.DataFrame, title: str, path: Path) -> None:
    """Save the 100% stacked risk-tier composition chart."""
    pct = (
        summary.pivot(
            index="risk_tier",
            columns="benefit_group",
            values="pct_within_risk_tier",
        )
        .reindex(RISK_TIER_ORDER)
        .fillna(0)
    )
    counts = (
        summary.drop_duplicates("risk_tier")
        .set_index("risk_tier")
        .reindex(RISK_TIER_ORDER)["risk_tier_members"]
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    bottom = np.zeros(len(pct))
    x = np.arange(len(pct.index))
    for group in BENEFIT_GROUP_ORDER:
        values = pct[group].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=group,
            color=BENEFIT_GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.8,
        )
        for idx, value in enumerate(values):
            if value >= 0.07:
                ax.text(
                    idx,
                    bottom[idx] + value / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if group == "High benefit" else "#222222",
                )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{RISK_TIER_DISPLAY[tier]}\n(n={counts.loc[tier]})"
            for tier in RISK_TIER_ORDER
        ]
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Original Funds risk tier (1 = lowest, 5 = highest)", labelpad=14)
    ax.set_ylabel("Percent of members within risk tier")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate(scored: pd.DataFrame, output_dir: Path, prefix: str, title: str) -> None:
    summary = build_summary(scored)
    summary.to_csv(output_dir / f"{prefix}_risk_tier_benefit_group_summary.csv", index=False)
    save_chart(
        summary,
        title,
        output_dir / f"dashboard_{prefix}_risk_tier_by_benefit_group.png",
    )

    overall = assign_benefit_terciles(scored["benefit_score"])
    print(title)
    print(pd.Series(overall).value_counts(sort=False).to_dict())


def main() -> None:
    configurations = [
        (
            held_out_tlearner_scores(
                OUTPUT_ROOT / "T-Learner" / "XGBoost" / "uplift_scored_output.csv"
            ),
            OUTPUT_ROOT / "T-Learner" / "XGBoost",
            "tlearner",
            "XGBoost T-learner benefit tercile distribution by risk tier (test set)",
        ),
        (
            held_out_tlearner_scores(
                OUTPUT_ROOT / "T-Learner" / "GLMNet" / "uplift_scored_output.csv"
            ),
            OUTPUT_ROOT / "T-Learner" / "GLMNet",
            "tlearner",
            "GLMNet T-learner benefit tercile distribution by risk tier (test set)",
        ),
        (
            pd.read_csv(
                OUTPUT_ROOT
                / "X-Learner"
                / "XGBoost"
                / "xlearner_scored_test_output.csv"
            ),
            OUTPUT_ROOT / "X-Learner" / "XGBoost",
            "xlearner",
            "XGBoost X-learner benefit tercile distribution by risk tier (test set)",
        ),
        (
            pd.read_csv(
                OUTPUT_ROOT
                / "X-Learner"
                / "GLMNet"
                / "xlearner_scored_test_output.csv"
            ),
            OUTPUT_ROOT / "X-Learner" / "GLMNet",
            "xlearner",
            "GLMNet X-learner benefit tercile distribution by risk tier (test set)",
        ),
    ]

    for scored, output_dir, prefix, title in configurations:
        generate(scored, output_dir, prefix, title)


if __name__ == "__main__":
    main()
