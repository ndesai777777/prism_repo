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


def missing_output_note(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
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
        ["Model matrix columns after one-hot encoding", int(float(row["number_of_model_matrix_columns"]))],
    ]
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


def factual_event_rate_threshold_classification_table() -> str:
    rows = []
    for row in read_csv(OUTPUT_ROOT / "factual_event_rate_threshold_classification.csv"):
        rows.append(
            [
                row["model"].replace("GLMNET", "GLMNet"),
                row["group"],
                pct(row["threshold"]),
                int(float(row["predicted_positive_n"])),
                int(float(row["true_positive"])),
                int(float(row["false_positive"])),
                int(float(row["true_negative"])),
                int(float(row["false_negative"])),
                fnum(row["precision"]),
                fnum(row["recall"]),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Group",
            "Threshold",
            "Predicted positive N",
            "True positives",
            "False positives",
            "True negatives",
            "False negatives",
            "Precision",
            "Recall",
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
    rows = []
    for model_folder, model_label in [("XGBoost", "XGBoost"), ("GLMNet", "GLMNet")]:
        scored = sorted(
            read_csv(TLEARNER_ROOT / model_folder / "uplift_scored_output.csv"),
            key=lambda row: float(row["benefit_score"]),
            reverse=True,
        )[:2]
        for index, row in enumerate(scored):
            rows.append(
                [
                    model_label,
                    "Highest benefit" if index == 0 else "Second highest benefit",
                    int(float(row["outcome_ed_90d"])),
                    int(float(row["intervention_flag"])),
                    fnum(row["pred_ed_if_treated"]),
                    fnum(row["pred_ed_if_control"]),
                    fnum(row["benefit_score"]),
                    int(float(row["uplift_decile"])),
                ]
            )
    return markdown_table(
        [
            "Model",
            "Example",
            "Actual outcome",
            "Treatment flag",
            "Predicted ED if treated",
            "Predicted ED if control",
            "Benefit score",
            "Uplift decile",
        ],
        rows,
    )


def glmnet_decile_table() -> str:
    rows = []
    for row in read_csv(GLMNET_ROOT / "uplift_decile_summary.csv"):
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


def glmnet_xlearner_decile_path() -> Path:
    return XLEARNER_GLMNET_ROOT / "xlearner_decile_summary.csv"


def glmnet_t_vs_x_decile_table() -> str:
    x_path = glmnet_xlearner_decile_path()
    if not x_path.exists():
        return missing_output_note(x_path)

    t_rows = read_csv(GLMNET_ROOT / "uplift_decile_summary.csv")
    x_rows = read_csv(x_path)
    x_by_decile = {int(float(row["uplift_decile"])): row for row in x_rows}
    rows = []
    for t_row in t_rows:
        decile = int(float(t_row["uplift_decile"]))
        x_row = x_by_decile[decile]
        rows.append(
            [
                decile,
                int(float(t_row["n"])),
                fnum(t_row["avg_benefit_score"]),
                fnum(x_row["avg_benefit_score"]),
                fnum(t_row["observed_ed_rate"]),
                fnum(x_row["observed_ed_rate"]),
                fnum(t_row["treated_pct"]),
                fnum(x_row["treated_pct"]),
            ]
        )
    return markdown_table(
        [
            "Uplift decile",
            "N",
            "T-learner avg benefit",
            "X-learner avg benefit",
            "T-learner observed ED rate",
            "X-learner observed ED rate",
            "T-learner treatment pct",
            "X-learner treatment pct",
        ],
        rows,
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
    path = XLEARNER_ROOT / "xlearner_vs_tlearner_consistency_summary.csv"
    if not path.exists():
        return missing_output_note(path)

    rows = []
    for row in read_csv(path):
        if row["model"].strip().upper() != "GLMNET":
            continue
        rows.append(
            [
                row["model"].replace("GLMNET", "GLMNet"),
                fnum(row["pearson_benefit_score_corr"]),
                fnum(row["spearman_benefit_score_corr"]),
                pct(row["top_decile_overlap_pct"]),
                fnum(row["t_learner_mean_benefit_score"]),
                fnum(row["x_learner_mean_benefit_score"]),
            ]
        )

    if not rows:
        return "_Pending: GLMNet consistency row was not found in the X-learner consistency output._"

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
    rows = sorted(
        read_csv(GLMNET_ROOT / "shap_importance_benefit_score.csv"),
        key=lambda row: float(row["mean_abs_benefit_contribution"]),
        reverse=True,
    )[:5]
    features = ", ".join(f"`{row['feature']}`" for row in rows)
    return markdown_table(["Top current benefit drivers by magnitude"], [[features]], align_right=False)


def glmnet_benefit_signed_table() -> str:
    data = read_csv(GLMNET_ROOT / "shap_importance_benefit_score.csv")
    positive_rows = sorted(
        [row for row in data if float(row["mean_signed_benefit_contribution"]) > 0],
        key=lambda row: float(row["mean_signed_benefit_contribution"]),
        reverse=True,
    )[:5]
    negative_rows = sorted(
        [row for row in data if float(row["mean_signed_benefit_contribution"]) < 0],
        key=lambda row: float(row["mean_signed_benefit_contribution"]),
    )[:5]

    rows = []
    for row in positive_rows:
        rows.append(
            [
                "Increase predicted benefit",
                f"`{row['feature']}`",
                fnum(row["mean_signed_benefit_contribution"]),
            ]
        )
    for row in negative_rows:
        rows.append(
            [
                "Decrease predicted benefit",
                f"`{row['feature']}`",
                fnum(row["mean_signed_benefit_contribution"]),
            ]
        )

    return markdown_table(
        [
            "Direction",
            "Top current benefit drivers by signed value",
            "Mean signed benefit contribution",
        ],
        rows,
        align_right=False,
    )


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
    glmnet_row = read_csv(GLMNET_ROOT / "top_benefit_decile_summary.csv")[0]
    scored = pd.read_csv(GLMNET_ROOT / "uplift_scored_output.csv")
    _, test_df = split_train_test(
        scored,
        train_fraction=0.70,
        seed=123,
        stratify_columns=["intervention_flag", "outcome_ed_90d"],
    )
    risk_top = test_df.sort_values("current_risk_score", ascending=False).head(
        int(float(glmnet_row["top_decile_n"]))
    )

    def roi_values(frame: pd.DataFrame) -> dict[str, float]:
        n = len(frame)
        avg_benefit = float(frame["benefit_score"].mean())
        avoided = n * avg_benefit
        gross = avoided * 1200
        intervention_cost = n * 250
        net = gross - intervention_cost
        roi = net / intervention_cost
        return {
            "n": n,
            "avg_benefit": avg_benefit,
            "avoided": avoided,
            "gross": gross,
            "intervention_cost": intervention_cost,
            "net": net,
            "roi": roi,
        }

    risk_roi = roi_values(risk_top)
    rows = [
        [
            "GLMNet uplift score",
            int(float(glmnet_row["top_decile_n"])),
            fnum(glmnet_row["top_decile_avg_predicted_benefit"]),
            fnum(glmnet_row["top_decile_estimated_ed_visits_avoided"]),
            money(glmnet_row["top_decile_gross_savings"]),
            money(glmnet_row["top_decile_intervention_cost"]),
            money(glmnet_row["top_decile_net_savings"]),
            fnum(glmnet_row["top_decile_roi"]),
        ],
        [
            "Current risk score",
            risk_roi["n"],
            fnum(risk_roi["avg_benefit"]),
            fnum(risk_roi["avoided"]),
            money(risk_roi["gross"]),
            money(risk_roi["intervention_cost"]),
            money(risk_roi["net"]),
            fnum(risk_roi["roi"]),
        ],
    ]
    return markdown_table(
        [
            "Targeting approach",
            "Top decile n",
            "Avg predicted benefit",
            "Estimated ED visits avoided",
            "Gross savings",
            "Intervention cost",
            "Net savings",
            "ROI",
        ],
        rows,
        align_right=False,
    )


def roi_interpretation() -> str:
    glmnet_row = read_csv(GLMNET_ROOT / "top_benefit_decile_summary.csv")[0]
    scored = pd.read_csv(GLMNET_ROOT / "uplift_scored_output.csv")
    _, test_df = split_train_test(
        scored,
        train_fraction=0.70,
        seed=123,
        stratify_columns=["intervention_flag", "outcome_ed_90d"],
    )
    n = int(float(glmnet_row["top_decile_n"]))
    risk_top = test_df.sort_values("current_risk_score", ascending=False).head(n)
    risk_avg_benefit = float(risk_top["benefit_score"].mean())
    risk_avoided = n * risk_avg_benefit
    risk_gross = risk_avoided * 1200
    risk_cost = n * 250
    risk_net = risk_gross - risk_cost
    risk_roi = risk_net / risk_cost

    return (
        "GLMNet uplift targeting estimates "
        f"{fnum(glmnet_row['top_decile_estimated_ed_visits_avoided'])} avoided ED visits "
        "in the top benefit decile. Targeting the top decile by current risk score instead "
        f"estimates {fnum(risk_avoided)} avoided ED visits. Under the current assumptions, "
        "the estimated gross savings do not exceed intervention costs in either approach, "
        "so ROI remains negative. However, uplift-based targeting produces a less negative "
        "ROI than current-risk targeting because it selects members with higher average "
        f"predicted intervention benefit ({fnum(glmnet_row['top_decile_roi'])} versus {fnum(risk_roi)})."
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
    "roi_interpretation": roi_interpretation,
}


CHART_GENERATORS: dict[str, Callable[[], str]] = {
    "glmnet_calibration_plot": lambda: markdown_image(
        "GLMNet calibration plot",
        "Outputs/Uplift/Python/T-Learner/GLMNet/dashboard_calibration_plot.png",
    ),
    "glmnet_predicted_treated_vs_control": lambda: markdown_image(
        "GLMNet predicted ED risk if treated versus control",
        "Outputs/Uplift/Python/T-Learner/GLMNet/dashboard_predicted_treated_vs_control.png",
    ),
    "glmnet_avg_benefit_by_decile": lambda: markdown_image(
        "GLMNet average predicted benefit by uplift decile",
        "Outputs/Uplift/Python/T-Learner/GLMNet/dashboard_avg_benefit_by_decile.png",
    ),
    "glmnet_t_vs_x_avg_benefit_charts": glmnet_t_vs_x_chart_block,
    "glmnet_benefit_driver_chart": lambda: markdown_image(
        "GLMNet benefit-driver importance",
        "Outputs/Uplift/Python/T-Learner/GLMNet/dashboard_shap_benefit_score.png",
    ),
    "glmnet_roi_by_decile": lambda: markdown_image(
        "GLMNet ROI net savings by uplift decile",
        "Outputs/Uplift/Python/T-Learner/GLMNet/dashboard_roi_net_savings_by_decile.png",
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
    print(f"Updated generated tables in {README_PATH}")


if __name__ == "__main__":
    main()
