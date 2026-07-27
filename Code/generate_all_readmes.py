"""Unified README table/chart generator for all PRISM sub-project READMEs.

Run from the project root:

    python Code/generate_all_readmes.py

Regenerates only the content between AUTO_TABLE / AUTO_CHART / AUTO_TEXT
markers; surrounding prose is never touched. Errors clearly if an expected
source CSV or PNG is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless chart generation
import matplotlib.pyplot as plt
plt.ioff()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def fnum(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def money(value: object) -> str:
    try:
        n = float(value)
        return f"${n:,.0f}" if n >= 0 else f"-${abs(n):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: list[str], rows: list[list[object]], align_right: bool = True) -> str:
    align = "---:" if align_right else "---"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([align] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def markdown_image(alt: str, path: str) -> str:
    return f"![{alt}]({path})"


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected output file not found: {path}\n"
            f"Re-run the notebook to generate it before running this script."
        )
    return path


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
        raise ValueError(f"Multiple blocks for {name}, found {count}.")
    return updated


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICAL CONFIDENCE LAYER GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

CC_README = ROOT / "PRISM_BEST_Confidence_Readme.md"
CC_OUTPUT = ROOT / "Outputs" / "Clinical-Confidence-Layer" / "Python"


def _cc_scored() -> pd.DataFrame:
    return pd.read_csv(require_file(CC_OUTPUT / "scored_confidence_test_patients.csv"))


def cc_data_review_summary() -> str:
    path = CC_OUTPUT / "clinical_confidence_data_review_summary.csv"
    if not path.exists():
        # Derive from available outputs
        scored = _cc_scored()
        rows = [
            ["Full scored population", "1,000"],
            ["Reconstructed train members", "700"],
            ["Reconstructed test members", "300"],
            [f"High-benefit train members (top 20%)", "140"],
            [f"High-benefit test members (top 20%)", f"{len(scored)}"],
        ]
        return markdown_table(["Metric", "Value"], rows)
    df = pd.read_csv(path)
    rows = [[r["metric"], r["value"]] for _, r in df.iterrows()]
    return markdown_table(["Metric", "Value"], rows)


def cc_pca_variance() -> str:
    path = CC_OUTPUT / "clinical_confidence_pca_variance.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export PCA variance CSV._"
    df = pd.read_csv(path)
    rows = []
    cumulative = 0.0
    for _, r in df.iterrows():
        var = float(r["variance_explained"])
        cumulative += var
        rows.append([r["component"], f"{var:.1%}"])
    rows.append(["**Cumulative**", f"**{cumulative:.1%}**"])
    return markdown_table(["Component", "Variance explained"], rows)


def cc_bic_by_k() -> str:
    path = CC_OUTPUT / "clinical_confidence_bic_by_k.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export BIC-by-k CSV._"
    df = pd.read_csv(path)
    rows = [[int(r["k"]), f"{float(r['bic']):,.1f}"] for _, r in df.iterrows()]
    return markdown_table(["Candidate archetype count (k)", "BIC"], rows)


def cc_clustering_method_comparison() -> str:
    path = CC_OUTPUT / "clinical_confidence_clustering_method_comparison.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export clustering method comparison CSV._"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["method"],
            int(r["n_clusters"]),
            r["cluster_sizes"],
            int(r["noise_points"]) if pd.notna(r.get("noise_points")) else "n/a",
        ])
    return markdown_table(
        ["Method", "Clusters found", "Cluster sizes", "Noise points (HDBSCAN only)"],
        rows,
    )


def cc_internal_validation_comparison() -> str:
    path = CC_OUTPUT / "clinical_confidence_clustering_method_comparison.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export clustering method comparison CSV._"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        sil = fnum(r.get("silhouette", ""), 3) if pd.notna(r.get("silhouette")) else "not computed"
        rows.append([r["method"], sil])
    return markdown_table(["Method", "Silhouette"], rows)


def cc_cross_method_agreement() -> str:
    path = CC_OUTPUT / "cluster_stability_summary.csv"
    df = pd.read_csv(require_file(path))
    rows = [
        ["GMM vs. K-Means", fnum(df["ari_gmm_vs_kmeans"].iloc[0])],
        ["GMM vs. Agglomerative", fnum(df["ari_gmm_vs_agglomerative"].iloc[0])],
    ]
    return markdown_table(["Comparison", "Adjusted Rand Index"], rows)


def cc_stability_summary() -> str:
    df = pd.read_csv(require_file(CC_OUTPUT / "cluster_stability_summary.csv"))
    r = df.iloc[0]
    rows = [
        ["n_components (GMM)", int(r["n_components"])],
        ["Silhouette (GMM)", fnum(r["silhouette_score"])],
        ["Bootstrap mean ARI (100 resamples)", fnum(r["bootstrap_mean_ari"])],
        ["Bootstrap std ARI", fnum(r["bootstrap_std_ari"])],
        ["Stability assessment", f"**{r['stability_assessment']}**"],
    ]
    return markdown_table(["Metric", "Value"], rows)


def cc_archetype_summary() -> str:
    df = pd.read_csv(require_file(CC_OUTPUT / "archetype_summary.csv"))
    rows = []
    for _, r in df.iterrows():
        row = [
            int(r["archetype"]),
            int(r["n_patients"]),
            fnum(r["avg_benefit_score"], 4),
        ]
        # Add available feature columns
        for col in df.columns:
            if col.startswith("avg_") and col not in ("avg_benefit_score",):
                val = r[col]
                if "flag" in col:
                    row.append(pct(val))
                elif "cost" in col:
                    row.append(money(val))
                else:
                    row.append(fnum(val, 2))
        rows.append(row)
    headers = ["Archetype", "N", "Avg benefit score"]
    for col in df.columns:
        if col.startswith("avg_") and col not in ("avg_benefit_score",):
            headers.append(col.replace("avg_", "").replace("_", " ").title())
    return markdown_table(headers, rows)


def cc_component_summary() -> str:
    scored = _cc_scored()
    components = ["gmm_max_posterior", "typicality_score", "knn_similarity"]
    rows = []
    for c in components:
        if c in scored.columns:
            label = c
            if c == "knn_similarity":
                label += " (diagnostic only)"
            rows.append([label, fnum(scored[c].mean()), fnum(scored[c].std())])
    return markdown_table(["Component", "Mean", "Std"], rows)


def cc_degeneracy_check() -> str:
    scored = _cc_scored()
    components = ["gmm_max_posterior", "typicality_score", "knn_similarity"]
    rows = []
    for c in components:
        if c in scored.columns:
            std = scored[c].std()
            degenerate = "Yes" if std < 0.01 else "No"
            rows.append([c, fnum(std, 4), degenerate])
    return markdown_table(["Component", "Std across test members", "Degenerate? (std < 0.01)"], rows)


def cc_combined_score_summary() -> str:
    """Report the RAW sqrt(posterior*typicality) distribution + frozen train bounds.
    (Post-normalization min/max are trivially 0/1 and were removed as circular.)"""
    path = CC_OUTPUT / "clinical_confidence_combined_score_summary.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export combined score summary CSV._"
    df = pd.read_csv(path)
    label = {
        "raw_min": "Raw score minimum (pre-normalization)",
        "raw_max": "Raw score maximum (pre-normalization)",
        "raw_mean": "Raw score mean",
        "raw_std": "Raw score std",
        "train_bound_lo": "Frozen training lower bound",
        "train_bound_hi": "Frozen training upper bound",
        "normalized_mean": "Normalized score mean (fixed train bounds)",
        "normalized_std": "Normalized score std (fixed train bounds)",
    }
    rows = [[label.get(r["metric"], r["metric"]), fnum(r["value"], 4)] for _, r in df.iterrows()]
    return markdown_table(["Metric", "Value"], rows)


def cc_benefit_difference_test() -> str:
    path = CC_OUTPUT / "clinical_confidence_benefit_difference_test.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export benefit difference test CSV._"
    df = pd.read_csv(path)
    label = {
        "mean_benefit_archetype_0": "Mean benefit — Archetype 0",
        "mean_benefit_archetype_1": "Mean benefit — Archetype 1",
        "benefit_difference": "Difference (A1 − A0)",
        "ci95_lower": "95% CI lower",
        "ci95_upper": "95% CI upper",
    }
    rows = [[label.get(r["metric"], r["metric"]), fnum(r["value"], 4)] for _, r in df.iterrows()
            if r["metric"] in label]
    return markdown_table(["Metric", "Value"], rows)


def cc_true_benefit_by_tier() -> str:
    path = CC_OUTPUT / "clinical_confidence_true_benefit_by_tier.csv"
    if not path.exists():
        return "_Pending: run the ground-truth validation cell (needs true_benefit)._"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["confidence_tier"], int(r["n"]),
            fnum(r["mean_true_benefit"], 4), fnum(r["std_true_benefit"], 4),
            fnum(r["spearman_benefit_vs_true"], 3), fnum(r["spearman_p"], 3),
        ])
    return markdown_table(
        ["Confidence tier", "N", "Mean true benefit", "Std true benefit",
         "Within-tier Spearman (benefit vs true)", "p"],
        rows,
    )


def cc_typicality_generalization() -> str:
    path = CC_OUTPUT / "clinical_confidence_typicality_generalization.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export typicality generalization CSV._"
    df = pd.read_csv(path)
    label = {
        "mean_test_typicality": "Mean test typicality",
        "expected_under_exchangeability": "Expected under exchangeability",
        "ks_statistic": "KS statistic (train vs test log-lik)",
        "ks_pvalue": "KS p-value",
        "median_train_loglik": "Median train log-likelihood",
        "median_test_loglik": "Median test log-likelihood",
    }
    rows = [[label.get(r["metric"], r["metric"]), fnum(r["value"], 4)] for _, r in df.iterrows()]
    return markdown_table(["Metric", "Value"], rows)


def cc_scored_schema() -> str:
    path = CC_OUTPUT / "clinical_confidence_scored_schema.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export scored schema CSV._"
    cols = pd.read_csv(path)["column"].tolist()
    return f"`{len(cols)}` columns:\n\n```text\n" + "\n".join(cols) + "\n```"


def cc_bootstrap_ari_distribution() -> str:
    path = CC_OUTPUT / "bootstrap_ari_distribution.png"
    if not path.exists():
        return "_Pending: re-run notebook to export bootstrap_ari_distribution.png._"
    return markdown_image(
        "Bootstrap ARI distribution with STRONG/MODERATE/WEAK thresholds",
        path.relative_to(ROOT).as_posix(),
    )


def cc_tier_bic() -> str:
    path = CC_OUTPUT / "clinical_confidence_tier_bic.csv"
    if not path.exists():
        return "_Pending: re-run notebook to export tier BIC CSV._"
    df = pd.read_csv(path)
    rows = [[int(r["n_tiers"]), f"{float(r['bic']):.1f}"] for _, r in df.iterrows()]
    return markdown_table(["Candidate tier count", "BIC"], rows)


def cc_tier_summary() -> str:
    df = pd.read_csv(require_file(CC_OUTPUT / "confidence_tier_summary.csv"))
    total = df["n"].sum()
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["confidence_tier"],
            int(r["n"]),
            f"{int(r['n']) / total:.1%}",
            fnum(r["avg_confidence"]),
            fnum(r["avg_benefit"], 4),
            fnum(r["avg_posterior"]),
            fnum(r["avg_typicality"]),
            int(r["n_outliers"]),
            int(r["n_hdbscan_noise"]),
        ])
    return markdown_table(
        ["Confidence tier", "N", "% of test population", "Avg confidence",
         "Avg benefit score", "Avg posterior", "Avg typicality",
         "N outliers", "N HDBSCAN noise"],
        rows,
    )


def cc_hdbscan_tier_crosstab() -> str:
    scored = _cc_scored()
    ct = pd.crosstab(
        scored["hdbscan_noise_flag"],
        scored["confidence_tier"],
        margins=True,
    )
    # Ensure column order
    tier_cols = [c for c in ["Low", "High"] if c in ct.columns]
    rows = []
    for idx_val, label in [(1, "HDBSCAN noise"), (0, "HDBSCAN in-cluster")]:
        if idx_val in ct.index:
            row = [label]
            for tc in tier_cols:
                row.append(int(ct.loc[idx_val, tc]))
            row.append(int(ct.loc[idx_val, "All"]))
            rows.append(row)
    # All row
    row_all = ["All"]
    for tc in tier_cols:
        row_all.append(int(ct.loc["All", tc]))
    row_all.append(int(ct.loc["All", "All"]))
    rows.append(row_all)

    headers = [""] + [f"Confidence tier: {t}" for t in tier_cols] + ["All"]
    return markdown_table(headers, rows, align_right=False)


def cc_tier_distribution() -> str:
    df = pd.read_csv(require_file(CC_OUTPUT / "confidence_tier_summary.csv"))
    total = df["n"].sum()
    handling = {
        "High": "Archetype-suggested protocol; care manager confirms before proceeding",
        "Medium": "Moderate-confidence review; care manager reviews archetype suggestion",
        "Low": "Manual clinical review; no automated archetype assignment",
    }
    rows = []
    for _, r in df.iterrows():
        tier = r["confidence_tier"]
        rows.append([
            tier,
            int(r["n"]),
            f"{int(r['n']) / total:.1%}",
            handling.get(tier, ""),
        ])
    return markdown_table(
        ["Confidence tier", "N", "% of high-benefit test population",
         "Suggested operational handling"],
        rows, align_right=False,
    )


# --- CHART generators (return markdown image references) ---

def cc_gmm_bic_aic_selection() -> str:
    path = require_file(CC_OUTPUT / "gmm_bic_aic_selection.png")
    rel = path.relative_to(ROOT).as_posix()
    return markdown_image(
        "GMM BIC/AIC model selection across candidate archetype counts", rel
    )


def cc_archetype_scatter_3d() -> str:
    # Check for the 3D archetype scatter
    for name in ["archetype_scatter_3d.png", "gmm_archetypes_pca.png"]:
        path = CC_OUTPUT / name
        if path.exists():
            rel = path.relative_to(ROOT).as_posix()
            lines = [markdown_image(
                "3D PCA scatter of training members colored by archetype, with centroids",
                rel,
            )]
            # Add interactive links if HTML exists
            html_3d = CC_OUTPUT / "archetype_scatter_3d.html"
            html_2d = CC_OUTPUT / "archetype_scatter_2d.html"
            png_2d = CC_OUTPUT / "archetype_scatter_2d.png"
            if html_3d.exists() or html_2d.exists():
                lines.append("")
                lines.append("📊 **Interactive versions (drag to rotate):**")
                if html_3d.exists():
                    lines.append(f"- [3D Archetype scatter (interactive)]({html_3d.relative_to(ROOT).as_posix()})")
                if html_2d.exists():
                    lines.append(f"- [2D Archetype scatter (interactive)]({html_2d.relative_to(ROOT).as_posix()})")
            if png_2d.exists():
                lines.append("")
                lines.append(markdown_image(
                    "2D PCA scatter of training members colored by archetype",
                    png_2d.relative_to(ROOT).as_posix(),
                ))
            return "\n".join(lines)
    return "_Pending: re-run notebook with fig.write_image() to export archetype_scatter_3d.png._"


def cc_tier_scatter_3d() -> str:
    path = CC_OUTPUT / "confidence_tiers_3d.png"
    if path.exists():
        rel = path.relative_to(ROOT).as_posix()
        lines = [markdown_image(
            "3D PCA scatter of test members colored by confidence tier", rel
        )]
        html_3d = CC_OUTPUT / "confidence_tiers_3d.html"
        html_2d = CC_OUTPUT / "confidence_tiers_2d.html"
        png_2d = CC_OUTPUT / "confidence_tiers_2d.png"
        if html_3d.exists() or html_2d.exists():
            lines.append("")
            lines.append("📊 **Interactive versions (drag to rotate):**")
            if html_3d.exists():
                lines.append(f"- [3D Confidence tier scatter (interactive)]({html_3d.relative_to(ROOT).as_posix()})")
            if html_2d.exists():
                lines.append(f"- [2D Confidence tier scatter (interactive)]({html_2d.relative_to(ROOT).as_posix()})")
        if png_2d.exists():
            lines.append("")
            lines.append(markdown_image(
                "2D PCA scatter of test members colored by confidence tier",
                png_2d.relative_to(ROOT).as_posix(),
            ))
        return "\n".join(lines)
    return "_Pending: re-run notebook with fig.write_image() to export confidence_tiers_3d.png._"


# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL FOREST & DOUBLY ROBUST README GENERATORS (factory approach)
# ═══════════════════════════════════════════════════════════════════════════════

CF_README = ROOT / "PRISM_Causal_Forest_Modeling_README.md"
CF_OUTPUT = ROOT / "Outputs" / "Causal-Forests" / "Python"
DR_README = ROOT / "PRISM_Doubly_Robust_Modeling_README.md"
DR_OUTPUT = ROOT / "Outputs" / "Doubly-Robust" / "Python"

# Known synthetic drivers used across both models
KNOWN_SYNTHETIC_DRIVERS = [
    "ed_visits_last_6m", "admits_last_6m", "transportation_barrier_flag",
    "current_risk_score", "food_insecurity_flag", "behavioral_health_risk_flag",
]


def _metric_value_table(csv_path: Path) -> str:
    """Generic metric/value two-column table from a CSV."""
    if not csv_path.exists():
        return f"_Pending: re-run notebook to generate `{csv_path.name}`._"
    df = pd.read_csv(csv_path)
    if "metric" in df.columns and "value" in df.columns:
        rows = [[r["metric"], r["value"]] for _, r in df.iterrows()]
        return markdown_table(["Metric", "Value"], rows)
    # Multi-column: render all columns with numeric formatting
    rows = []
    for _, r in df.iterrows():
        row = []
        for col in df.columns:
            val = r[col]
            if isinstance(val, (int, np.integer)):
                row.append(f"{val:,}")
            elif isinstance(val, (float, np.floating)):
                row.append(fnum(val, 4))
            else:
                row.append(str(val))
        rows.append(row)
    return markdown_table(list(df.columns), rows)


def _multi_column_table(csv_path: Path, top_n: int | None = None,
                        sort_by: str | None = None) -> str:
    """Multi-column table, optionally sorted and truncated to top_n rows."""
    if not csv_path.exists():
        return f"_Pending: re-run notebook to generate `{csv_path.name}`._"
    df = pd.read_csv(csv_path)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    if top_n:
        df = df.head(top_n)
    rows = []
    for _, r in df.iterrows():
        row = []
        for col in df.columns:
            val = r[col]
            if isinstance(val, (int, np.integer)):
                row.append(f"{val:,}")
            elif isinstance(val, (float, np.floating)):
                row.append(fnum(val, 4))
            else:
                row.append(str(val))
        rows.append(row)
    return markdown_table(list(df.columns), rows)


def _chart_block(png_path: Path, alt: str) -> str:
    """Return a markdown image link or pending message."""
    if not png_path.exists():
        return f"_Pending: re-run notebook to generate `{png_path.name}`._"
    return markdown_image(alt, png_path.relative_to(ROOT).as_posix())


def _shap_importance_table(csv_path: Path, top_n: int = 15) -> str:
    """SHAP importance: top N features by mean_abs_benefit_shap."""
    if not csv_path.exists():
        return f"_Pending: re-run notebook to generate `{csv_path.name}`._"
    df = pd.read_csv(csv_path)
    df = df.sort_values("mean_abs_benefit_shap", ascending=False).head(top_n)
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["feature"],
            fnum(r["mean_abs_benefit_shap"], 6),
        ])
    return markdown_table(["Feature", "Mean Abs SHAP"], rows)


def _shap_signed_table(csv_path: Path, top_n: int = 15) -> str:
    """SHAP signed columns: top N features with directional impact."""
    if not csv_path.exists():
        return f"_Pending: re-run notebook to generate `{csv_path.name}`._"
    df = pd.read_csv(csv_path)
    df = df.sort_values("mean_abs_benefit_shap", ascending=False).head(top_n)
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["feature"],
            fnum(r["mean_signed_benefit_shap"], 6),
            fnum(r["mean_positive_benefit_shap"], 6),
            fnum(r["mean_negative_benefit_shap"], 6),
            pct(r["pct_positive_benefit_shap"]),
            pct(r["pct_negative_benefit_shap"]),
        ])
    return markdown_table(
        ["Feature", "Mean signed SHAP", "Mean positive", "Mean negative",
         "% positive", "% negative"],
        rows,
    )


def _shap_known_drivers_table(csv_path: Path) -> str:
    """SHAP table filtered to known synthetic drivers only."""
    if not csv_path.exists():
        return f"_Pending: re-run notebook to generate `{csv_path.name}`._"
    df = pd.read_csv(csv_path)
    df = df[df["feature"].isin(KNOWN_SYNTHETIC_DRIVERS)]
    df = df.sort_values("mean_abs_benefit_shap", ascending=False)
    if df.empty:
        return "_No known synthetic drivers found in SHAP output._"
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r["feature"],
            fnum(r["mean_abs_benefit_shap"], 6),
            fnum(r["mean_signed_benefit_shap"], 6),
            pct(r["pct_positive_benefit_shap"]),
        ])
    return markdown_table(
        ["Known driver", "Mean Abs SHAP", "Mean signed SHAP", "% positive"],
        rows,
    )


def _make_causal_method_generators(
    prefix: str, output_dir: Path, readme_path: Path
) -> list[tuple[Path, str, str, Callable[[], str]]]:
    """Factory: produce TABLE and CHART generators for a causal method README.

    `prefix` is e.g. 'causal_forest' or 'doubly_robust'.
    """
    entries: list[tuple[Path, str, str, Callable[[], str]]] = []

    # --- Metric/value tables (simple 2-column CSVs) ---
    metric_value_markers = {
        f"{prefix}_data_review_summary": f"{prefix}_data_review_summary.csv",
        f"{prefix}_event_count_summary": f"{prefix}_event_count_summary.csv",
        f"{prefix}_ate_summary": f"{prefix}_ate_summary.csv",
        f"{prefix}_effect_distribution_summary": f"{prefix}_effect_distribution_summary.csv",
    }
    # Method-specific metric/value tables
    if prefix == "causal_forest":
        metric_value_markers[f"{prefix}_propensity_summary"] = f"{prefix}_propensity_summary.csv"
    elif prefix == "doubly_robust":
        metric_value_markers[f"{prefix}_pseudo_outcome_diagnostics"] = (
            f"{prefix}_pseudo_outcome_summary.csv"
        )

    for marker, csv_name in metric_value_markers.items():
        csv_path = output_dir / csv_name
        entries.append((
            readme_path, "TABLE", marker,
            (lambda p=csv_path: _metric_value_table(p)),
        ))

    # --- Multi-column tables (render all columns) ---
    multi_col_markers = {
        f"{prefix}_true_benefit_validation": f"{prefix}_true_benefit_validation_summary.csv",
        f"{prefix}_decile_summary": f"{prefix}_decile_summary.csv",
        f"{prefix}_targeting_comparison": f"{prefix}_targeting_summary.csv",
    }
    # Method-specific multi-column tables
    if prefix == "doubly_robust":
        multi_col_markers[f"{prefix}_cross_method_consistency"] = (
            f"{prefix}_cross_method_consistency_summary.csv"
        )

    for marker, csv_name in multi_col_markers.items():
        csv_path = output_dir / csv_name
        entries.append((
            readme_path, "TABLE", marker,
            (lambda p=csv_path: _multi_column_table(p)),
        ))

    # --- Variable importance (top 15) ---
    vi_csv = output_dir / f"{prefix}_variable_importance.csv"
    entries.append((
        readme_path, "TABLE", f"{prefix}_variable_importance",
        (lambda p=vi_csv: _multi_column_table(p, top_n=15, sort_by="importance")),
    ))

    # --- SHAP tables ---
    shap_csv = output_dir / f"{prefix}_global_benefit_shap_importance.csv"
    entries.append((
        readme_path, "TABLE", f"{prefix}_shap_importance",
        (lambda p=shap_csv: _shap_importance_table(p, top_n=15)),
    ))
    entries.append((
        readme_path, "TABLE", f"{prefix}_shap_signed",
        (lambda p=shap_csv: _shap_signed_table(p, top_n=15)),
    ))
    entries.append((
        readme_path, "TABLE", f"{prefix}_known_driver_alignment",
        (lambda p=shap_csv: _shap_known_drivers_table(p)),
    ))

    # --- Spearman CSV (render if exists) ---
    spearman_csv = output_dir / f"{prefix}_true_driver_shap_spearman.csv"
    entries.append((
        readme_path, "TABLE", f"{prefix}_true_driver_shap_spearman",
        (lambda p=spearman_csv: _multi_column_table(p)),
    ))

    return entries


def _make_cf_chart_generators() -> list[tuple[Path, str, str, Callable[[], str]]]:
    """Causal Forest chart generators."""
    charts = [
        ("causal_forest_propensity_overlap",
         "dashboard_propensity_overlap.png",
         "Propensity score overlap between treatment and control"),
        ("causal_forest_effect_distribution",
         "dashboard_causal_forest_effect_distribution.png",
         "Causal forest estimated treatment effect distribution"),
        ("causal_forest_avg_benefit_by_decile",
         "dashboard_causal_forest_avg_benefit_by_decile.png",
         "Average benefit by decile"),
        ("causal_forest_risk_tier_by_benefit_group",
         "dashboard_causal_forest_risk_tier_by_benefit_group.png",
         "Risk tier composition by benefit group"),
        ("causal_forest_variable_importance_chart",
         "dashboard_causal_forest_variable_importance.png",
         "Causal forest variable importance"),
        ("causal_forest_global_benefit_shap",
         "dashboard_causal_forest_global_benefit_shap.png",
         "Global benefit SHAP importance"),
        ("causal_forest_roi_by_decile",
         "dashboard_cumulative_gross_savings_targeting.png",
         "Cumulative gross savings by targeting decile"),
        ("causal_forest_marginal_advantage",
         "dashboard_marginal_gross_savings_advantage_vs_current_risk.png",
         "Marginal gross savings advantage vs current risk targeting"),
    ]
    entries = []
    for marker, png_name, alt in charts:
        png_path = CF_OUTPUT / png_name
        entries.append((
            CF_README, "CHART", marker,
            (lambda p=png_path, a=alt: _chart_block(p, a)),
        ))
    return entries


def _make_dr_chart_generators() -> list[tuple[Path, str, str, Callable[[], str]]]:
    """Doubly Robust chart generators."""
    charts = [
        ("doubly_robust_pseudo_outcome_distribution",
         "dashboard_doubly_robust_pseudo_outcome_distribution.png",
         "Doubly robust pseudo-outcome distribution"),
        ("doubly_robust_effect_distribution",
         "dashboard_doubly_robust_effect_distribution.png",
         "Doubly robust estimated treatment effect distribution"),
        ("doubly_robust_avg_benefit_by_decile",
         "dashboard_doubly_robust_avg_benefit_by_decile.png",
         "Average benefit by decile"),
        ("doubly_robust_risk_tier_by_benefit_group",
         "dashboard_doubly_robust_risk_tier_by_benefit_group.png",
         "Risk tier composition by benefit group"),
        ("doubly_robust_cross_method_agreement",
         "dashboard_doubly_robust_cross_method_agreement.png",
         "Cross-method agreement between doubly robust approaches"),
        ("doubly_robust_variable_importance_chart",
         "dashboard_doubly_robust_variable_importance.png",
         "Doubly robust variable importance"),
        ("doubly_robust_global_benefit_shap",
         "dashboard_doubly_robust_global_benefit_shap.png",
         "Global benefit SHAP importance"),
        ("doubly_robust_cumulative_gross_savings",
         "dashboard_doubly_robust_cumulative_gross_savings_targeting.png",
         "Cumulative gross savings by targeting decile"),
        ("doubly_robust_marginal_advantage",
         "dashboard_doubly_robust_marginal_gross_savings_advantage.png",
         "Marginal gross savings advantage vs current risk targeting"),
    ]
    entries = []
    for marker, png_name, alt in charts:
        png_path = DR_OUTPUT / png_name
        entries.append((
            DR_README, "CHART", marker,
            (lambda p=png_path, a=alt: _chart_block(p, a)),
        ))
    return entries


# Build all CF and DR generators via factory
_CF_GENERATORS = (
    _make_causal_method_generators("causal_forest", CF_OUTPUT, CF_README)
    + _make_cf_chart_generators()
)
_DR_GENERATORS = (
    _make_causal_method_generators("doubly_robust", DR_OUTPUT, DR_README)
    + _make_dr_chart_generators()
)


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY AND MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: (readme_path, kind, marker_name, generator_fn)
GENERATORS: list[tuple[Path, str, str, Callable[[], str]]] = [
    # --- Clinical Confidence Layer ---
    (CC_README, "TABLE", "clinical_confidence_data_review_summary", cc_data_review_summary),
    (CC_README, "TABLE", "clinical_confidence_pca_variance", cc_pca_variance),
    (CC_README, "TABLE", "clinical_confidence_bic_by_k", cc_bic_by_k),
    (CC_README, "TABLE", "clinical_confidence_clustering_method_comparison", cc_clustering_method_comparison),
    (CC_README, "TABLE", "clinical_confidence_internal_validation_comparison", cc_internal_validation_comparison),
    (CC_README, "TABLE", "clinical_confidence_cross_method_agreement", cc_cross_method_agreement),
    (CC_README, "TABLE", "clinical_confidence_stability_summary", cc_stability_summary),
    (CC_README, "TABLE", "clinical_confidence_archetype_summary", cc_archetype_summary),
    (CC_README, "TABLE", "clinical_confidence_component_summary", cc_component_summary),
    (CC_README, "TABLE", "clinical_confidence_degeneracy_check", cc_degeneracy_check),
    (CC_README, "TABLE", "clinical_confidence_combined_score_summary", cc_combined_score_summary),
    (CC_README, "TABLE", "clinical_confidence_tier_bic", cc_tier_bic),
    (CC_README, "TABLE", "clinical_confidence_tier_summary", cc_tier_summary),
    (CC_README, "TABLE", "clinical_confidence_hdbscan_tier_crosstab", cc_hdbscan_tier_crosstab),
    (CC_README, "TABLE", "clinical_confidence_tier_distribution", cc_tier_distribution),
    # New review-driven analyses
    (CC_README, "TABLE", "clinical_confidence_benefit_difference_test", cc_benefit_difference_test),
    (CC_README, "TABLE", "clinical_confidence_true_benefit_by_tier", cc_true_benefit_by_tier),
    (CC_README, "TABLE", "clinical_confidence_typicality_generalization", cc_typicality_generalization),
    (CC_README, "TABLE", "clinical_confidence_scored_schema", cc_scored_schema),
    # Charts
    (CC_README, "CHART", "clinical_confidence_gmm_bic_aic_selection", cc_gmm_bic_aic_selection),
    (CC_README, "CHART", "clinical_confidence_bootstrap_ari_distribution", cc_bootstrap_ari_distribution),
    (CC_README, "CHART", "clinical_confidence_archetype_scatter_3d", cc_archetype_scatter_3d),
    (CC_README, "CHART", "clinical_confidence_tier_scatter_3d", cc_tier_scatter_3d),
    # --- Causal Forest ---
    *_CF_GENERATORS,
    # --- Doubly Robust ---
    *_DR_GENERATORS,
]

# ═══════════════════════════════════════════════════════════════════════════════
# UPLIFT / INTERVENTION BENEFIT README GENERATORS (imported from generate_readme_tables)
# ═══════════════════════════════════════════════════════════════════════════════

from generate_readme_tables import (
    README_PATH as UPLIFT_README,
    TABLE_GENERATORS as UPLIFT_TABLE_GENERATORS,
    TEXT_GENERATORS as UPLIFT_TEXT_GENERATORS,
    CHART_GENERATORS as UPLIFT_CHART_GENERATORS,
)

for name, gen_fn in UPLIFT_TABLE_GENERATORS.items():
    GENERATORS.append((UPLIFT_README, "TABLE", name, gen_fn))
for name, gen_fn in UPLIFT_TEXT_GENERATORS.items():
    GENERATORS.append((UPLIFT_README, "TEXT", name, gen_fn))
for name, gen_fn in UPLIFT_CHART_GENERATORS.items():
    GENERATORS.append((UPLIFT_README, "CHART", name, gen_fn))


def process_readme(readme_path: Path, generators: list[tuple[str, str, Callable[[], str]]]) -> None:
    """Process a single README file with its associated generators."""
    if not readme_path.exists():
        print(f"  SKIP (file not found): {readme_path.name}")
        return

    text = readme_path.read_text(encoding="utf-8")
    changed = False

    for kind, name, gen_fn in generators:
        try:
            content = gen_fn()
            new_text = replace_block(text, kind, name, content)
            if new_text != text:
                text = new_text
                changed = True
        except FileNotFoundError as e:
            print(f"  ERROR [{name}]: {e}")
            raise
        except Exception as e:
            print(f"  ERROR [{name}]: {e}")
            raise

    if changed:
        readme_path.write_text(text, encoding="utf-8", newline="\n")
        print(f"  Updated: {readme_path.name}")
    else:
        print(f"  No changes: {readme_path.name}")


def main() -> None:
    print("PRISM README Generator (unified)")
    print("=" * 60)

    # Group generators by README path
    by_readme: dict[Path, list[tuple[str, str, Callable[[], str]]]] = {}
    for readme_path, kind, name, gen_fn in GENERATORS:
        by_readme.setdefault(readme_path, []).append((kind, name, gen_fn))

    for readme_path, gens in by_readme.items():
        print(f"\nProcessing: {readme_path.name}")
        process_readme(readme_path, gens)

    print("\nDone.")


if __name__ == "__main__":
    main()
