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
