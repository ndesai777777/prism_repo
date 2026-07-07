"""Refresh generated result tables in the PRISM report README.

Run from the project root:

    python Code/generate_readme_tables.py

The script reads the current CSV outputs under Outputs/Uplift/Python and replaces
Markdown blocks marked with:

    <!-- AUTO_TABLE:table_name START -->
    ...
    <!-- AUTO_TABLE:table_name END -->
"""

from __future__ import annotations

import csv
import html
import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from _prism_model_utils import split_train_test


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "PRISM_Intervention_Benefit_Modeling_README.md"
OUTPUT_ROOT = ROOT / "Outputs" / "Uplift" / "Python"
TLEARNER_ROOT = OUTPUT_ROOT / "T-Learner"
GLMNET_ROOT = TLEARNER_ROOT / "GLMNet"
XLEARNER_GLMNET_ROOT = OUTPUT_ROOT / "X-Learner" / "GLMNet"
XLEARNER_ROOT = OUTPUT_ROOT / "X-Learner"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(value: object, digits: int = 4) -> str:
    if value is None or str(value).strip() == "":
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "N/A"
    return f"{number:.{digits}f}"


def pct(value: object, digits: int = 1) -> str:
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}%"


def money(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.2f}"


def markdown_image(alt: str, path: str) -> str:
    return f"![{alt}]({path})"


def repo_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def missing_output_note(path: Path) -> str:
    rel = repo_rel(path)
    return f"_Pending: run the notebook to generate `{rel}`._"


def yes_no(value: object) -> str:
    return "Yes" if str(value).strip().lower() in {"true", "1", "yes"} else "No"


def first_present(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"None of these columns were found: {keys}")


def markdown_table(headers: list[str], rows: list[list[object]], align_right: bool = True) -> str:
    align = "---:" if align_right else "---"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([align] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def html_table(headers: list[str], rows: list[list[object]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(item))}</td>" for item in row)
            + "</tr>"
        )
    return "<table><thead><tr>" + header_html + "</tr></thead><tbody>" + "".join(row_html) + "</tbody></table>"


def data_review_table() -> str:
    row = read_csv(OUTPUT_ROOT / "data_review_summary.csv")[0]
    rows = [
        ["Total members", f"{int(float(row['total_members'])):,}"],
        ["Treated members", f"{int(float(row['treated_members'])):,}"],
        ["Untreated/control members", f"{int(float(row['control_members'])):,}"],
        ["Treatment rate", pct(row["treatment_rate"])],
        ["ED outcome events", f"{int(float(row['outcome_events'])):,}"],
        ["Outcome prevalence", pct(row["outcome_prevalence"])],
        ["Treated observed ED rate", pct(row["treated_outcome_rate"])],
        ["Control observed ED rate", pct(row["control_outcome_rate"])],
        ["Final predictors before one-hot encoding", int(float(row["number_of_predictors"]))],
    ]
    if "number_of_continuous_count_predictors" in row:
        rows.extend(
            [
                ["Continuous/count numeric predictors", int(float(row["number_of_continuous_count_predictors"]))],
                ["Binary indicator predictors", int(float(row["number_of_binary_indicator_predictors"]))],
                ["Multi-level categorical predictors", int(float(row["number_of_multilevel_categorical_predictors"]))],
            ]
        )
    rows.append(["Model matrix columns after one-hot encoding", int(float(row["number_of_model_matrix_columns"]))])
    return markdown_table(["Metric", "Current value"], rows)


def model_performance_table() -> str:
    rows = []
    for row in read_csv(OUTPUT_ROOT / "model_evaluation_summary.csv"):
        rows.append(
            [
                row["model"].replace("GLMNET", "GLMNet"),
                fnum(row["treated_cv_auc"]),
                fnum(row["control_cv_auc"]),
                fnum(row["treated_test_auc"]),
                fnum(row["control_test_auc"]),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Treated CV AUC",
            "Control CV AUC",
            "Treated test AUC",
            "Control test AUC",
        ],
        rows,
    )


def brier_calibration_table() -> str:
    rows = []
    for row in read_csv(OUTPUT_ROOT / "model_evaluation_summary.csv"):
        rows.append(
            [
                row["model"].replace("GLMNET", "GLMNet"),
                fnum(row["treated_brier_score"]),
                fnum(row["control_brier_score"]),
                fnum(row["treated_calibration_error"]),
                fnum(row["control_calibration_error"]),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Treated Brier",
            "Control Brier",
            "Treated calibration error",
            "Control calibration error",
        ],
        rows,
    )


def factual_event_counts_table() -> str:
    rows = []
    for row in read_csv(OUTPUT_ROOT / "factual_event_count_summary.csv"):
        rows.append(
            [
                row["split"],
                row["group"],
                int(float(row["n"])),
                int(float(row["positive_ed_events"])),
                int(float(row["negative_ed_events"])),
                pct(row["event_rate"]),
            ]
        )
    return markdown_table(
        ["Split", "Group", "N", "Positive ED events", "Negative ED events", "Event rate"],
        rows,
    )


def factual_prediction_separation_table() -> str:
    rows = []
    for row in read_csv(OUTPUT_ROOT / "factual_prediction_separation.csv"):
        rows.append(
            [
                row["model"].replace("GLMNET", "GLMNet"),
                row["group"],
                int(float(row["positive_ed_events"])),
                int(float(row["negative_ed_events"])),
                fnum(row["auc"]),
                fnum(row["avg_pred_actual_positive"]),
                fnum(row["avg_pred_actual_negative"]),
                fnum(row["avg_pred_positive_minus_negative"]),
                fnum(row["median_pred_actual_positive"]),
                fnum(row["median_pred_actual_negative"]),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Group",
            "Positive events",
            "Negative events",
            "AUC",
            "Avg pred for ED=1",
            "Avg pred for ED=0",
            "Avg difference",
            "Median pred for ED=1",
            "Median pred for ED=0",
        ],
        rows,
    )


def factual_prediction_ranges_table() -> str:
    rows = []
    for row in read_csv(OUTPUT_ROOT / "factual_prediction_ranges.csv"):
        rows.append(
            [
                row["model"].replace("GLMNET", "GLMNet"),
                row["group"],
                fnum(row["min_pred"]),
                fnum(row["p10_pred"]),
                fnum(row["median_pred"]),
                fnum(row["mean_pred"]),
                fnum(row["p90_pred"]),
                fnum(row["max_pred"]),
            ]
        )
    return markdown_table(
        ["Model", "Group", "Min", "P10", "Median", "Mean", "P90", "Max"],
        rows,
    )


def observed_gap_table() -> str:
    rows = []
    for model_folder, model_label in [("XGBoost", "XGBoost"), ("GLMNet", "GLMNet")]:
        path = TLEARNER_ROOT / model_folder / "uplift_observed_gap_by_decile.csv"
        for row in read_csv(path):
            rows.append(
                [
                    model_label,
                    int(float(row["uplift_decile"])),
                    int(float(row["n"])),
                    int(float(row["treated_n"])),
                    int(float(row["control_n"])),
                    fnum(row["avg_predicted_benefit"]),
                    fnum(row["observed_control_minus_treated_gap"]),
                    fnum(first_present(row, ["gap_ci_lower_95", "observed_gap_ci_lower_95"])),
                    fnum(first_present(row, ["gap_ci_upper_95", "observed_gap_ci_upper_95"])),
                    yes_no(
                        first_present(
                            row,
                            [
                                "predicted_benefit_within_gap_ci_95",
                                "predicted_benefit_within_observed_gap_ci_95",
                            ],
                        )
                    ),
                ]
            )
    return markdown_table(
        [
            "Model",
            "Uplift decile",
            "N",
            "Treated N",
            "Control N",
            "Avg predicted benefit",
            "Observed control-treated gap",
            "Observed gap 95% CI lower",
            "Observed gap 95% CI upper",
            "Predicted benefit within observed gap 95% CI",
        ],
        rows,
    )


def top_benefit_examples_table() -> str:
    scored = pd.read_csv(GLMNET_ROOT / "uplift_scored_output.csv").copy()
    scored["benefit_score"] = pd.to_numeric(scored["benefit_score"])
    scored["pred_ed_if_treated"] = pd.to_numeric(scored["pred_ed_if_treated"])
    scored["pred_ed_if_control"] = pd.to_numeric(scored["pred_ed_if_control"])

    high_benefit = scored.sort_values("benefit_score", ascending=False).iloc[0]
    low_positive_benefit = scored[scored["benefit_score"] >= 0].copy()
    if low_positive_benefit.empty:
        low_positive_benefit = scored.copy()
    high_risk_cutoff = scored["pred_ed_if_control"].quantile(0.75)
    low_risk_cutoff = scored["pred_ed_if_control"].quantile(0.25)
    high_risk_low_benefit_pool = low_positive_benefit[
        low_positive_benefit["pred_ed_if_control"] >= high_risk_cutoff
    ].copy()
    low_risk_low_benefit_pool = low_positive_benefit[
        low_positive_benefit["pred_ed_if_control"] <= low_risk_cutoff
    ].copy()
    if high_risk_low_benefit_pool.empty:
        high_risk_low_benefit_pool = low_positive_benefit.copy()
    if low_risk_low_benefit_pool.empty:
        low_risk_low_benefit_pool = low_positive_benefit.copy()

    high_risk_low_benefit = high_risk_low_benefit_pool.sort_values(
        "benefit_score",
        ascending=True,
    ).iloc[0]
    low_risk_low_benefit = low_risk_low_benefit_pool.sort_values(
        "benefit_score",
        ascending=True,
    ).iloc[0]
    sleeping_dog = scored.sort_values("benefit_score", ascending=True).iloc[0]

    example_specs = [
        (
            "Highest benefit",
            high_benefit,
            "Strong outreach candidate because predicted ED risk is much lower under treatment.",
        ),
        (
            "High risk, low benefit",
            high_risk_low_benefit,
            "Clinically higher risk, but the predicted intervention benefit is small.",
        ),
        (
            "Low risk, low benefit",
            low_risk_low_benefit,
            "Lower outreach priority because baseline ED risk and predicted benefit are both low.",
        ),
        (
            "Sleeping dog / lowest benefit",
            sleeping_dog,
            "Not prioritized by uplift score because predicted benefit is lowest in the scored population.",
        ),
    ]

    rows = []
    for label, row, interpretation in example_specs:
        rows.append(
            [
                label,
                int(float(row["outcome_ed_90d"])),
                int(float(row["intervention_flag"])),
                fnum(row["pred_ed_if_treated"]),
                fnum(row["pred_ed_if_control"]),
                fnum(row["benefit_score"]),
                int(float(row["uplift_decile"])),
                interpretation,
            ]
        )
    return markdown_table(
        [
            "Member profile",
            "Actual outcome",
            "Treatment flag",
            "Predicted ED if treated",
            "Predicted ED if control",
            "Benefit score",
            "Uplift decile",
            "Outreach interpretation",
        ],
        rows,
        align_right=False,
    )


def learner_decile_rows(rows_in: list[dict[str, str]]) -> list[list[object]]:
    rows = []
    for row in rows_in:
        rows.append(
            [
                int(float(row["uplift_decile"])),
                int(float(row["n"])),
                fnum(row["avg_benefit_score"]),
                fnum(row["observed_ed_rate"]),
                fnum(row["avg_pred_ed_if_treated"]),
                fnum(row["avg_pred_ed_if_control"]),
                fnum(row["treated_pct"]),
            ]
        )
    return rows


def learner_decile_table(rows_in: list[dict[str, str]]) -> str:
    rows = learner_decile_rows(rows_in)
    return markdown_table(
        [
            "Uplift decile",
            "N",
            "Avg benefit score",
            "Observed ED rate",
            "Avg predicted ED if treated",
            "Avg predicted ED if control",
            "Treatment pct",
        ],
        rows,
    )


def glmnet_decile_table() -> str:
    return learner_decile_table(read_csv(GLMNET_ROOT / "uplift_decile_summary.csv"))


def glmnet_xlearner_decile_path() -> Path:
    return XLEARNER_GLMNET_ROOT / "xlearner_decile_summary.csv"


def glmnet_t_vs_x_decile_table() -> str:
    x_path = glmnet_xlearner_decile_path()
    if not x_path.exists():
        return missing_output_note(x_path)

    t_rows = read_csv(GLMNET_ROOT / "uplift_decile_summary.csv")
    x_rows = read_csv(x_path)
    t_headers = [
        "Uplift decile",
        "N",
        "Avg T-learner benefit score",
        "Observed ED rate",
        "Avg predicted ED if treated",
        "Avg predicted ED if control",
        "Treatment pct",
    ]
    x_headers = [
        "Uplift decile",
        "N",
        "Avg X-learner benefit score",
        "Observed ED rate",
        "Avg outcome-model ED if treated",
        "Avg outcome-model ED if control",
        "Treatment pct",
    ]
    t_table = html_table(t_headers, learner_decile_rows(t_rows))
    x_table = html_table(x_headers, learner_decile_rows(x_rows))
    return "\n".join(
        [
            '<table><tr>',
            '<td valign="top" width="50%"><strong>GLMNet T-learner deciles</strong>',
            t_table,
            '</td>',
            '<td valign="top" width="50%"><strong>GLMNet X-learner deciles</strong>',
            x_table,
            '</td>',
            '</tr></table>',
            "",
            "_Note: In the T-learner table, benefit is the direct contrast between the "
            "control and treated outcome-model predictions. In the X-learner table, "
            "benefit is the final weighted treatment-effect-model estimate; the treated "
            "and control outcome-model columns are included only as risk context._",
        ]
    )


def xlearner_glmnet_avg_benefit_chart_path() -> Path:
    return XLEARNER_GLMNET_ROOT / "dashboard_avg_benefit_by_decile.png"


def ensure_xlearner_glmnet_avg_benefit_chart() -> Path | None:
    x_path = glmnet_xlearner_decile_path()
    if not x_path.exists():
        return None

    chart_path = xlearner_glmnet_avg_benefit_chart_path()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(x_path).sort_values("uplift_decile")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["uplift_decile"].astype(str), df["avg_benefit_score"])
    ax.set_title("GLMNet X-Learner: Average Predicted Benefit by Decile")
    ax.set_xlabel("Uplift Decile")
    ax.set_ylabel("Average Predicted Benefit")
    ax.axhline(0, color="#333333", linewidth=1)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def glmnet_t_vs_x_chart_block() -> str:
    x_chart = ensure_xlearner_glmnet_avg_benefit_chart()
    if x_chart is None:
        return missing_output_note(glmnet_xlearner_decile_path())

    return "\n".join(
        [
            "| GLMNet T-learner | GLMNet X-learner |",
            "|---|---|",
            "| "
            + markdown_image(
                "GLMNet T-learner average predicted benefit by uplift decile",
                "Outputs/Uplift/Python/T-Learner/GLMNet/dashboard_avg_benefit_by_decile.png",
            )
            + " | "
            + markdown_image(
                "GLMNet X-learner average predicted benefit by uplift decile",
                "Outputs/Uplift/Python/X-Learner/GLMNet/dashboard_avg_benefit_by_decile.png",
            )
            + " |",
        ]
    )


def glmnet_xlearner_consistency_table() -> str:
    path = XLEARNER_GLMNET_ROOT / "xlearner_scored_test_output.csv"
    if not path.exists():
        return missing_output_note(path)

    scored = pd.read_csv(path)
    required = {"t_learner_benefit_score", "benefit_score", "uplift_decile"}
    missing = sorted(required - set(scored.columns))
    if missing:
        return (
            "_Pending: GLMNet X-learner scored output is missing required columns: "
            + ", ".join(f"`{column}`" for column in missing)
            + "._"
        )

    t_decile = pd.qcut(
        scored["t_learner_benefit_score"].rank(method="first", ascending=False),
        q=10,
        labels=False,
    ) + 1
    t_top = set(scored.index[t_decile == 1])
    x_top = set(scored.index[scored["uplift_decile"] == 1])
    top_overlap = len(t_top & x_top) / len(t_top) if t_top else float("nan")

    rows = [
        [
            "GLMNet",
            fnum(scored["t_learner_benefit_score"].corr(scored["benefit_score"], method="pearson")),
            fnum(scored["t_learner_benefit_score"].corr(scored["benefit_score"], method="spearman")),
            pct(top_overlap),
            fnum(scored["t_learner_benefit_score"].mean()),
            fnum(scored["benefit_score"].mean()),
        ]
    ]

    return markdown_table(
        [
            "Model",
            "Pearson benefit score corr",
            "Spearman benefit score corr",
            "Top decile overlap",
            "T-learner mean benefit score",
            "X-learner mean benefit score",
        ],
        rows,
    )


def glmnet_benefit_magnitude_table() -> str:
    t_path = GLMNET_ROOT / "shap_importance_benefit_score.csv"
    x_path = XLEARNER_GLMNET_ROOT / "xlearner_benefit_driver_importance.csv"
    if not x_path.exists():
        return missing_output_note(x_path)

    t_rows = [
        [f"`{row['feature']}`", fnum(row["mean_abs_benefit_contribution"])]
        for row in sorted(
            read_csv(t_path),
            key=lambda item: float(item["mean_abs_benefit_contribution"]),
            reverse=True,
        )[:5]
    ]
    x_rows = [
        [f"`{row['feature']}`", fnum(row["mean_abs_xlearner_benefit_contribution"])]
        for row in sorted(
            read_csv(x_path),
            key=lambda item: float(item["mean_abs_xlearner_benefit_contribution"]),
            reverse=True,
        )[:5]
    ]
    headers = ["Feature", "Mean absolute contribution"]
    return "\n".join(
        [
            '<table><tr>',
            '<td valign="top" width="50%"><strong>GLMNet T-learner</strong>',
            html_table(headers, t_rows),
            '</td>',
            '<td valign="top" width="50%"><strong>GLMNet X-learner</strong>',
            html_table(headers, x_rows),
            '</td>',
            '</tr></table>',
        ]
    )


def glmnet_benefit_signed_table() -> str:
    t_path = GLMNET_ROOT / "shap_importance_benefit_score.csv"
    x_path = XLEARNER_GLMNET_ROOT / "xlearner_benefit_driver_importance.csv"
    if not x_path.exists():
        return missing_output_note(x_path)

    def signed_rows(path: Path, signed_col: str) -> list[list[str]]:
        data = read_csv(path)
        positive_rows = sorted(
            [row for row in data if float(row[signed_col]) > 0],
            key=lambda row: float(row[signed_col]),
            reverse=True,
        )[:5]
        negative_rows = sorted(
            [row for row in data if float(row[signed_col]) < 0],
            key=lambda row: float(row[signed_col]),
        )[:5]

        rows = []
        for row in positive_rows:
            rows.append(
                [
                    "Increase predicted benefit",
                    f"`{row['feature']}`",
                    fnum(row[signed_col]),
                ]
            )
        for row in negative_rows:
            rows.append(
                [
                    "Decrease predicted benefit",
                    f"`{row['feature']}`",
                    fnum(row[signed_col]),
                ]
            )
        return rows

    headers = ["Direction", "Feature", "Mean signed contribution"]
    return "\n".join(
        [
            '<table><tr>',
            '<td valign="top" width="50%"><strong>GLMNet T-learner</strong>',
            html_table(headers, signed_rows(t_path, "mean_signed_benefit_contribution")),
            '</td>',
            '<td valign="top" width="50%"><strong>GLMNet X-learner</strong>',
            html_table(headers, signed_rows(x_path, "mean_signed_xlearner_benefit_contribution")),
            '</td>',
            '</tr></table>',
        ]
    )


def glmnet_benefit_driver_chart_block() -> str:
    comparison_path = XLEARNER_GLMNET_ROOT / "dashboard_t_vs_x_benefit_driver_comparison.png"
    if comparison_path.exists():
        return markdown_image(
            "GLMNet T-learner versus X-learner top drivers of predicted treatment benefit",
            "Outputs/Uplift/Python/X-Learner/GLMNet/dashboard_t_vs_x_benefit_driver_comparison.png",
        )

    x_chart_path = XLEARNER_GLMNET_ROOT / "dashboard_xlearner_benefit_drivers.png"
    if not x_chart_path.exists():
        return missing_output_note(x_chart_path)

    return "\n".join(
        [
            "| GLMNet T-learner | GLMNet X-learner |",
            "|---|---|",
            "| "
            + markdown_image(
                "GLMNet T-learner benefit-driver importance",
                "Outputs/Uplift/Python/T-Learner/GLMNet/dashboard_shap_benefit_score.png",
            )
            + " | "
            + markdown_image(
                "GLMNet X-learner benefit-driver importance",
                "Outputs/Uplift/Python/X-Learner/GLMNet/dashboard_xlearner_benefit_drivers.png",
            )
            + " |",
        ]
    )


def ensure_glmnet_calibration_chart() -> Path:
    path = GLMNET_ROOT / "calibration_by_decile.csv"
    chart_path = GLMNET_ROOT / "dashboard_calibration_plot.png"
    df = pd.read_csv(path)
    groups = [group for group in ["Control", "Treated"] if group in set(df["group"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    max_rate = max(df["avg_predicted_ed_rate"].max(), df["observed_ed_rate"].max(), 0.01)
    for ax, group in zip(axes, groups):
        group_df = df[df["group"] == group].sort_values("pred_risk_decile")
        x_values = range(len(group_df))
        bar_width = 0.38
        ax.bar(
            [x - bar_width / 2 for x in x_values],
            group_df["avg_predicted_ed_rate"],
            width=bar_width,
            label="Avg predicted ED rate",
        )
        ax.bar(
            [x + bar_width / 2 for x in x_values],
            group_df["observed_ed_rate"],
            width=bar_width,
            label="Observed ED rate",
        )
        ax.set_title(f"GLMNet: {group} Calibration")
        ax.set_xlabel("Predicted risk decile: 1 = highest risk")
        ax.set_xticks(list(x_values))
        ax.set_xticklabels(group_df["pred_risk_decile"].astype(str))
        ax.set_ylim(0, max_rate * 1.2)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("ED rate")
    axes[-1].legend(loc="upper right")
    fig.suptitle("GLMNet: Predicted vs Observed ED Rate By Factual Group")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def ensure_glmnet_predicted_treated_vs_control_chart() -> Path:
    path = GLMNET_ROOT / "uplift_decile_summary.csv"
    chart_path = GLMNET_ROOT / "dashboard_predicted_treated_vs_control.png"
    df = pd.read_csv(path).sort_values("uplift_decile")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = range(len(df))
    bar_width = 0.38
    ax.bar(
        [x - bar_width / 2 for x in x_values],
        df["avg_pred_ed_if_treated"],
        width=bar_width,
        label="Predicted ED if treated",
    )
    ax.bar(
        [x + bar_width / 2 for x in x_values],
        df["avg_pred_ed_if_control"],
        width=bar_width,
        label="Predicted ED if control",
    )
    ax.set_title("GLMNet Predicted ED Risk By Uplift Decile")
    ax.set_xlabel("Uplift decile")
    ax.set_ylabel("Average predicted ED risk")
    ax.set_xticks(list(x_values))
    ax.set_xticklabels(df["uplift_decile"].astype(str))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def ensure_glmnet_avg_benefit_chart() -> Path:
    path = GLMNET_ROOT / "uplift_decile_summary.csv"
    chart_path = GLMNET_ROOT / "dashboard_avg_benefit_by_decile.png"
    df = pd.read_csv(path).sort_values("uplift_decile")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["uplift_decile"].astype(str), df["avg_benefit_score"])
    ax.set_title("GLMNet Average Predicted Benefit By Uplift Decile")
    ax.set_xlabel("Uplift decile")
    ax.set_ylabel("Average predicted benefit")
    ax.axhline(0, color="#333333", linewidth=1)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def ensure_glmnet_benefit_driver_chart() -> Path:
    path = GLMNET_ROOT / "shap_importance_benefit_score.csv"
    chart_path = GLMNET_ROOT / "dashboard_shap_benefit_score.png"
    df = pd.read_csv(path).head(20).sort_values("mean_abs_benefit_contribution")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"], df["mean_abs_benefit_contribution"])
    ax.set_title("GLMNet T-Learner: Top Drivers of Predicted Treatment Benefit")
    ax.set_xlabel("Mean absolute shared-standardized logit contribution")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def ensure_glmnet_roi_chart() -> Path:
    chart_path = GLMNET_ROOT / "dashboard_cumulative_gross_savings_targeting.png"
    df = cumulative_gross_savings_by_targeting()
    chart_df = df[df["population_fraction_targeted"] <= 0.50]
    fig, ax = plt.subplots(figsize=(8.5, 5.25))
    for approach, group_df in chart_df.groupby("targeting_approach"):
        ax.plot(
            group_df["population_fraction_targeted"] * 100,
            group_df["cumulative_gross_savings"],
            marker="o",
            linewidth=2,
            label=approach,
        )
    ax.set_title("GLMNet Cumulative Gross Savings Through Top Targeted Deciles")
    ax.set_xlabel("Population targeted (%)")
    ax.set_ylabel("Cumulative gross savings")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def chart_image(alt: str, chart_path: Path) -> str:
    return markdown_image(alt, repo_rel(chart_path))


def glmnet_benefit_driver_interpretation() -> str:
    data = read_csv(GLMNET_ROOT / "shap_importance_benefit_score.csv")
    magnitude = sorted(
        data,
        key=lambda row: float(row["mean_abs_benefit_contribution"]),
        reverse=True,
    )[:5]
    positive = sorted(
        [row for row in data if float(row["mean_signed_benefit_contribution"]) > 0],
        key=lambda row: float(row["mean_signed_benefit_contribution"]),
        reverse=True,
    )[:3]
    negative = sorted(
        [row for row in data if float(row["mean_signed_benefit_contribution"]) < 0],
        key=lambda row: float(row["mean_signed_benefit_contribution"]),
    )[:1]

    mag_text = ", ".join(f"`{row['feature']}`" for row in magnitude)
    pos_text = ", ".join(f"`{row['feature']}`" for row in positive)
    neg_text = f"`{negative[0]['feature']}`" if negative else "no negative signed contributors"
    return (
        f"For GLMNet, {mag_text} are the largest benefit-driver features by "
        "absolute contribution-difference magnitude. By signed value, "
        f"{pos_text} have the strongest positive average contribution to predicted "
        f"benefit. The strongest negative signed contributor is {neg_text}."
    )


def top_decile_comparison_table() -> str:
    rows = []
    for model_folder, model_label in [("XGBoost", "XGBoost"), ("GLMNet", "GLMNet")]:
        row = read_csv(TLEARNER_ROOT / model_folder / "top_benefit_decile_summary.csv")[0]
        rows.append(
            [
                model_label,
                int(float(row["top_decile_n"])),
                int(float(row["top_decile_treated_n"])),
                int(float(row["top_decile_control_n"])),
                fnum(row["top_decile_avg_predicted_benefit"]),
                fnum(row["top_decile_observed_ed_rate"]),
                fnum(row["top_decile_treated_observed_ed_rate"]),
                fnum(row["top_decile_control_observed_ed_rate"]),
                fnum(row["top_decile_observed_control_minus_treated_gap"]),
                fnum(row["top_decile_treated_pct"]),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Top decile n",
            "Treated n",
            "Control n",
            "Avg predicted benefit",
            "Observed ED rate",
            "Treated observed ED rate",
            "Control observed ED rate",
            "Observed control-treated gap",
            "Treatment pct",
        ],
        rows,
    )


def roi_table() -> str:
    df = cumulative_gross_savings_by_targeting()
    cutoff_rows = []
    for cutoff in [0.10, 0.20, 0.30, 0.40, 0.50]:
        uplift = df[
            (df["targeting_approach"] == "Uplift score")
            & (df["population_fraction_targeted"].round(2) == cutoff)
        ].iloc[0]
        risk = df[
            (df["targeting_approach"] == "Current risk score")
            & (df["population_fraction_targeted"].round(2) == cutoff)
        ].iloc[0]
        cutoff_rows.append(
            [
                f"Top {int(cutoff * 100)}%",
                int(uplift["n"]),
                money(uplift["cumulative_gross_savings"]),
                money(risk["cumulative_gross_savings"]),
                money(uplift["cumulative_gross_savings"] - risk["cumulative_gross_savings"]),
                fnum(uplift["cumulative_estimated_ed_visits_avoided"]),
                fnum(risk["cumulative_estimated_ed_visits_avoided"]),
            ]
        )
    return markdown_table(
        [
            "Targeted group",
            "Members targeted",
            "Uplift gross savings",
            "Current-risk gross savings",
            "Uplift advantage",
            "Uplift ED visits avoided",
            "Current-risk ED visits avoided",
        ],
        cutoff_rows,
        align_right=False,
    )


def cumulative_gross_savings_by_targeting() -> pd.DataFrame:
    scored = pd.read_csv(GLMNET_ROOT / "uplift_scored_output.csv")
    _, test_df = split_train_test(
        scored,
        train_fraction=0.70,
        seed=123,
        stratify_columns=["intervention_flag", "outcome_ed_90d"],
    )

    n_total = len(test_df)
    decile_size = max(1, n_total // 10)

    ranking_specs = [
        ("Uplift score", "benefit_score", False),
        ("Current risk score", "current_risk_score", False),
    ]
    rows = []
    for approach, sort_col, ascending in ranking_specs:
        ranked = test_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
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
                    "cumulative_gross_savings": avoided * 1200,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(GLMNET_ROOT / "cumulative_gross_savings_by_targeting.csv", index=False)
    return out


def roi_interpretation() -> str:
    df = cumulative_gross_savings_by_targeting()
    top_30_uplift = df[
        (df["targeting_approach"] == "Uplift score")
        & (df["population_fraction_targeted"].round(2) == 0.30)
    ].iloc[0]
    top_30_risk = df[
        (df["targeting_approach"] == "Current risk score")
        & (df["population_fraction_targeted"].round(2) == 0.30)
    ].iloc[0]
    advantage = top_30_uplift["cumulative_gross_savings"] - top_30_risk["cumulative_gross_savings"]
    return (
        "This view compares two targeting policies on the same held-out test population: "
        "ranking members by GLMNet predicted uplift versus ranking members by current risk score. "
        f"Through the top 30% of targeted members, uplift targeting captures "
        f"{money(top_30_uplift['cumulative_gross_savings'])} in estimated gross savings, compared with "
        f"{money(top_30_risk['cumulative_gross_savings'])} from current-risk targeting, an uplift advantage "
        f"of {money(advantage)}. Gross savings are estimated from the GLMNet predicted benefit score, so "
        "this is a targeting-policy comparison rather than a claim of realized savings."
    )


def predicted_risk_by_decile_interpretation() -> str:
    deciles = read_csv(GLMNET_ROOT / "uplift_decile_summary.csv")
    rows = [
        {
            "decile": int(float(row["uplift_decile"])),
            "treated": float(row["avg_pred_ed_if_treated"]),
            "control": float(row["avg_pred_ed_if_control"]),
            "benefit": float(row["avg_benefit_score"]),
        }
        for row in deciles
    ]
    top_uplift = next(row for row in rows if row["decile"] == 1)
    highest_control_risk = max(rows, key=lambda row: row["control"])

    return (
        "This chart shows why uplift targeting is different from risk-based targeting. "
        "A historical risk-ranking approach would mostly look at the orange bars, which "
        "represent predicted ED risk without treatment. In the current GLMNet output, "
        f"decile {highest_control_risk['decile']} has the highest average predicted ED risk "
        f"without treatment ({fnum(highest_control_risk['control'])}), but its predicted "
        f"treatment benefit is {fnum(highest_control_risk['benefit'])}. Decile 1 is prioritized "
        f"because its treatment-versus-control gap is larger: predicted ED risk falls from "
        f"{fnum(top_uplift['control'])} without treatment to {fnum(top_uplift['treated'])} "
        f"with treatment, for an average predicted benefit of {fnum(top_uplift['benefit'])}. "
        "In other words, the orange bar reflects baseline risk, while the gap between the "
        "orange and blue bars reflects expected impactability."
    )


TABLE_GENERATORS: dict[str, Callable[[], str]] = {
    "data_review_summary": data_review_table,
    "model_performance_summary": model_performance_table,
    "brier_calibration_summary": brier_calibration_table,
    "factual_event_counts": factual_event_counts_table,
    "factual_prediction_separation": factual_prediction_separation_table,
    "factual_prediction_ranges": factual_prediction_ranges_table,
    "observed_gap_by_decile": observed_gap_table,
    "top_benefit_examples": top_benefit_examples_table,
    "glmnet_uplift_decile_summary": glmnet_decile_table,
    "glmnet_t_vs_x_decile_summary": glmnet_t_vs_x_decile_table,
    "glmnet_xlearner_consistency_summary": glmnet_xlearner_consistency_table,
    "glmnet_benefit_magnitude": glmnet_benefit_magnitude_table,
    "glmnet_benefit_signed": glmnet_benefit_signed_table,
    "top_decile_comparison": top_decile_comparison_table,
    "roi_summary": roi_table,
}


TEXT_GENERATORS: dict[str, Callable[[], str]] = {
    "glmnet_benefit_driver_interpretation": glmnet_benefit_driver_interpretation,
    "predicted_risk_by_decile_interpretation": predicted_risk_by_decile_interpretation,
    "roi_interpretation": roi_interpretation,
}


CHART_GENERATORS: dict[str, Callable[[], str]] = {
    "glmnet_calibration_plot": lambda: chart_image(
        "GLMNet calibration plot",
        ensure_glmnet_calibration_chart(),
    ),
    "glmnet_predicted_treated_vs_control": lambda: chart_image(
        "GLMNet predicted ED risk if treated versus control",
        ensure_glmnet_predicted_treated_vs_control_chart(),
    ),
    "glmnet_avg_benefit_by_decile": lambda: chart_image(
        "GLMNet average predicted benefit by uplift decile",
        ensure_glmnet_avg_benefit_chart(),
    ),
    "glmnet_t_vs_x_avg_benefit_charts": glmnet_t_vs_x_chart_block,
    "glmnet_benefit_driver_chart": glmnet_benefit_driver_chart_block,
    "glmnet_roi_by_decile": lambda: chart_image(
        "GLMNet cumulative gross savings by targeting approach",
        ensure_glmnet_roi_chart(),
    ),
}


def replace_block(text: str, kind: str, name: str, content: str) -> str:
    start = f"<!-- AUTO_{kind}:{name} START -->"
    end = f"<!-- AUTO_{kind}:{name} END -->"
    pattern = re.compile(
        rf"{re.escape(start)}\n.*?\n{re.escape(end)}",
        flags=re.DOTALL,
    )
    replacement = f"{start}\n{content}\n{end}"
    updated, count = pattern.subn(replacement, text)
    if count > 1:
        raise ValueError(f"Expected at most one generated block for {name}, found {count}.")
    return updated


def main() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    for name, generator in TABLE_GENERATORS.items():
        text = replace_block(text, "TABLE", name, generator())
    for name, generator in TEXT_GENERATORS.items():
        text = replace_block(text, "TEXT", name, generator())
    for name, generator in CHART_GENERATORS.items():
        text = replace_block(text, "CHART", name, generator())
    README_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"Updated generated README tables and charts in {README_PATH}")


if __name__ == "__main__":
    main()
