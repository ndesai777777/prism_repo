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


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "PRISM_Intervention_Benefit_Modeling_README.md"
OUTPUT_ROOT = ROOT / "Outputs" / "Uplift" / "Python"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fnum(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
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


def yes_no(value: object) -> str:
    return "Yes" if str(value).strip().lower() in {"true", "1", "yes"} else "No"


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
                fnum(row["treated_brier_score"]),
                fnum(row["control_brier_score"]),
                fnum(row["treated_calibration_error"]),
                fnum(row["control_calibration_error"]),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Treated CV AUC",
            "Control CV AUC",
            "Treated test AUC",
            "Control test AUC",
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


def factual_top_risk_capture_table() -> str:
    rows = []
    for row in read_csv(OUTPUT_ROOT / "factual_top_risk_capture.csv"):
        rows.append(
            [
                row["model"].replace("GLMNET", "GLMNet"),
                row["group"],
                int(float(row["top_risk_n"])),
                int(float(row["top_risk_events"])),
                int(float(row["total_events"])),
                pct(row["top_risk_event_rate"]),
                pct(row["event_capture_rate"]),
                fnum(row["top_risk_lift_vs_overall"], 2),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Group",
            "Top-risk N",
            "Top-risk events",
            "Total events",
            "Top-risk event rate",
            "Event capture rate",
            "Lift vs overall",
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
        path = OUTPUT_ROOT / model_folder / "uplift_observed_gap_by_decile.csv"
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
                    fnum(row["gap_ci_lower_95"]),
                    fnum(row["gap_ci_upper_95"]),
                    yes_no(row["predicted_benefit_within_gap_ci_95"]),
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
            read_csv(OUTPUT_ROOT / model_folder / "uplift_scored_output.csv"),
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
    for row in read_csv(OUTPUT_ROOT / "GLMNet" / "uplift_decile_summary.csv"):
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


def top_decile_comparison_table() -> str:
    rows = []
    for model_folder, model_label in [("XGBoost", "XGBoost"), ("GLMNet", "GLMNet")]:
        row = read_csv(OUTPUT_ROOT / model_folder / "top_benefit_decile_summary.csv")[0]
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
    rows = []
    for model_folder, model_label in [("XGBoost", "XGBoost"), ("GLMNet", "GLMNet")]:
        row = read_csv(OUTPUT_ROOT / model_folder / "top_benefit_decile_summary.csv")[0]
        rows.append(
            [
                model_label,
                int(float(row["top_decile_n"])),
                fnum(row["top_decile_estimated_ed_visits_avoided"]),
                money(row["top_decile_gross_savings"]),
                money(row["top_decile_intervention_cost"]),
                money(row["top_decile_net_savings"]),
                fnum(row["top_decile_roi"]),
            ]
        )
    return markdown_table(
        [
            "Model",
            "Top decile n",
            "Estimated ED visits avoided",
            "Gross savings",
            "Intervention cost",
            "Net savings",
            "ROI",
        ],
        rows,
    )


GENERATORS: dict[str, Callable[[], str]] = {
    "data_review_summary": data_review_table,
    "model_performance_summary": model_performance_table,
    "factual_event_counts": factual_event_counts_table,
    "factual_prediction_separation": factual_prediction_separation_table,
    "factual_top_risk_capture": factual_top_risk_capture_table,
    "factual_prediction_ranges": factual_prediction_ranges_table,
    "observed_gap_by_decile": observed_gap_table,
    "top_benefit_examples": top_benefit_examples_table,
    "glmnet_uplift_decile_summary": glmnet_decile_table,
    "top_decile_comparison": top_decile_comparison_table,
    "roi_summary": roi_table,
}


def replace_block(text: str, name: str, content: str) -> str:
    start = f"<!-- AUTO_TABLE:{name} START -->"
    end = f"<!-- AUTO_TABLE:{name} END -->"
    pattern = re.compile(
        rf"{re.escape(start)}\n.*?\n{re.escape(end)}",
        flags=re.DOTALL,
    )
    replacement = f"{start}\n{content}\n{end}"
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(f"Expected exactly one generated block for {name}, found {count}.")
    return updated


def main() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    for name, generator in GENERATORS.items():
        text = replace_block(text, name, generator())
    README_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"Updated generated tables in {README_PATH}")


if __name__ == "__main__":
    main()
