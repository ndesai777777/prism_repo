# PRISM Clinical Confidence Layer Using High-Benefit Patient Clustering (Revised Plan)

## Objective

The purpose of this analysis is **not** to improve the treatment-effect estimates produced by the primary model (T-learner / X-learner / DR-learner / causal forest — whichever is selected as best). Instead, the objective is to build an **unsupervised Clinical Confidence Layer** that answers a narrower, purely descriptive question:

> **Does this patient's clinical profile resemble previously observed high-benefit patients, or does it fall in a region of the feature space we have little historical precedent for?**

This is a **resemblance / novelty-detection** statement, not an accuracy statement. It does not claim that the model's benefit prediction is more or less likely to be *correct* — only that it is more or less *precedented*. That distinction matters for how the score should be communicated to clinicians and care managers: "well-supported by historical patterns" vs. "novel profile, interpret with more clinical judgment," not "will work" vs. "won't work."

A true-benefit ground-truth check is available (see Appendix, Step 10) because this project uses synthetic data with a known benefit formula — but it is treated as an optional sanity check on the framework, not a required component of the core deliverable.

**Status note:** Model training (Step 1) and test-set scoring (Step 6) are already complete, implemented in the DR-learner notebook. This plan picks up from those existing outputs — the predicted individual treatment effects for both the 700-member training set and the 300-member test set are treated as given inputs below, not steps to (re)build. Everything from Step 2 onward (constructing the high-benefit reference population, feature selection, clustering, and the confidence layer) is new work.

**Execution note:** When this plan is implemented, the training and test predicted treatment effects should be pulled directly from the DR-learner notebook's existing outputs (e.g., loading the saved predictions/dataframe it produces) rather than re-running or reimplementing any part of model training or test scoring. Step 2 and Step 6 below start from those loaded values.

---

## Overall Workflow

```text
Training Data (700 Patients)
            │
            ▼
   Primary Treatment-Effect Model  [DONE — DR-learner notebook]
            │
            ▼
Predicted Individual Treatment Effects  [DONE — existing output]
            │
            ▼
 Select Top 20% Predicted High Benefit  (~140 patients)
            │
            ▼
 Select Feature Subset for Clustering
 (SHAP-selected, ~10-15 features)
            │
            ▼
 Discover High-Benefit Patient Archetypes
 (GMM primary; K-means + hierarchical stability check;
  HDBSCAN diagnostic layer)
            │
            ▼
 Validate Cluster Stability (bootstrap resampling)
            │
            ▼
 Build Historical Reference Library
──────────────────────────────────────────
Held-Out Test Data (300 Patients)
            │
            ▼
   Primary Treatment-Effect Model  [DONE — DR-learner notebook]
            │
            ▼
 Select Top 20% Predicted High Benefit (~60 patients)
            │
            ▼
 Compare Each Patient to Historical Clusters
 (same reduced feature space)
            │
            ▼
 Similarity Metrics (GMM posterior, k-NN, density)
            │
            ▼
 Clinical Confidence Score
            │
            ▼
 [Optional Appendix] Ground-Truth Sanity Check
```

---

## Step 1 – Train the Primary Treatment-Effect Model *(Already Complete)*

Done in the DR-learner notebook. The trained model has already produced, for every training member:
- Individual treatment effect
- Predicted benefit score

These existing predictions form the basis for constructing the historical reference population below. No clustering is performed on the full population — only on patients predicted to benefit most. This plan consumes those outputs rather than regenerating them — implementation should load the predictions directly from the DR-learner notebook (e.g., its saved output dataframe), not reimplement or rerun the model.

---

## Step 2 – Construct the Historical High-Benefit Reference Population

Rank all training patients by predicted treatment benefit:
- **High Benefit:** Top 20% (~140 patients)
- **Medium Benefit:** Middle 50%
- **Low Benefit:** Bottom 30%

Only the High Benefit group feeds the Clinical Confidence Layer.

**Known constraint to carry forward:** ~140 patients against the full 41-feature set is a thin observations-to-features ratio for stable clustering. This motivates Step 3 below — it is not an optional refinement, it is a prerequisite for the rest of the pipeline to produce trustworthy archetypes.

---

## Step 3 – Select a Reduced Feature Set for Clustering

Before clustering, narrow the 41 raw features to a smaller set that is both clinically nameable and relevant to benefit:

- **Primary approach:** SHAP feature importance from the trained treatment-effect model, selecting the top ~10-15 features. This ties archetype definitions to *why the model believes a patient benefits*, keeps every retained dimension directly interpretable (no PCA back-mapping needed), and is the approach usable on real (non-synthetic) data going forward.
- **Why not PCA:** PCA reduces dimensionality but optimizes for variance explained, not benefit relevance — a component can be dominated by a feature that varies a lot but has nothing to do with treatment benefit. It also requires back-mapping components to original features to keep cluster summaries clinically readable. SHAP-based selection avoids both problems.
- **Note for the appendix check:** because this project has a known synthetic benefit formula, the *true* benefit-driving features are also available. This isn't used to build the primary pipeline (it wouldn't be available on real data), but is used in Step 10 as a one-time comparison against the SHAP-selected set.

This reduced feature set is used consistently in both Step 4 (clustering) and Step 7 (fitting test patients to archetypes) — the two steps must operate in the same feature space.

---

## Step 4 – Discover High-Benefit Patient Archetypes

Cluster the ~140 historical high-benefit patients, using the reduced feature set from Step 3, on the question:

> **Are there distinct clinical phenotypes among patients predicted to benefit substantially from intervention?**

**Method selection, given n≈140 and the need for a graded (not just hard) cluster-fit signal:**

| Method | Role | Rationale |
|---|---|---|
| **Gaussian Mixture Model (GMM)** | Primary | Produces posterior cluster-membership probabilities natively — exactly the graded "how well does this patient fit" signal the confidence score needs, rather than a hard label requiring separate distance calculations. |
| **K-means** | Stability cross-check | Simple, fast, useful as an independent method to confirm GMM-discovered structure isn't an artifact of distributional assumptions. |
| **Hierarchical (agglomerative)** | Stability cross-check | More sample-efficient at small n; dendrogram gives a visual, defensible way to choose k rather than guessing upfront; handles Gower distance well if any retained features are binary/categorical. |
| **HDBSCAN** | Diagnostic layer | Only method with a native "doesn't belong to any cluster" concept — conceptually closest to novelty detection. Treated cautiously as primary output given thin per-cluster sample sizes (~25-30 patients per archetype), but useful for flagging borderline vs. clean cluster members even within the GMM structure. |

For each discovered archetype (from the primary GMM solution), compute:
- Number of patients
- Cluster centroid (mapped back to original clinical feature values, not PC/latent space)
- Average predicted benefit
- Average risk score
- Demographic summary
- Clinical summary
- Utilization summary
- SDOH summary

Cluster descriptions should emerge from the data; example categories (frequent ED utilizers, complex chronic disease, behavioral health, SDOH-driven, medication adherence) are illustrative only.

---

## Step 5 – Validate Cluster Stability

Given the small reference population, stability validation is a **gating step**, not a downstream nicety — if archetypes aren't reproducible at n≈140, they shouldn't be presented as clinical phenotypes.

- Compute Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index for the primary GMM solution.
- **Bootstrap resampling:** repeatedly resample the 140 historical high-benefit patients, re-cluster each time, and measure consensus / Jaccard similarity of cluster assignments across resamples.
- **Cross-method agreement:** compare GMM archetypes against K-means and hierarchical solutions on the same reduced feature set — consistent structure across methods increases confidence the phenotypes are real, not an artifact of one algorithm's assumptions.
- If stability is weak: reduce k, revisit the feature subset from Step 3, or reconsider whether hard archetypes are the right framing (see note in Step 8 on a continuous alternative).

---

## Step 6 – Score the Held-Out Test Population *(Already Complete)*

Done in the DR-learner notebook. The trained model has already scored the independent 300-member test population. This step now consists of loading those existing test-set predictions directly from the DR-learner notebook's output and selecting the top 20% (~60 predicted high-benefit patients) from them, exactly as in the training population — no re-scoring needed.

---

## Step 7 – Compare Test Patients to Historical High-Benefit Clusters

For each of the ~60 predicted high-benefit test patients, using the **same reduced feature set from Step 3**, measure fit against the historical archetypes:

1. **GMM posterior probability** — the model's native measure of how strongly a patient belongs to each historical cluster (e.g., Cluster 1: 92%, Cluster 2: 6%, Cluster 3: 2%). This is now the primary similarity signal, not an optional add-on.
2. **Nearest-neighbor distance** — mean/median distance to the *k* nearest historical high-benefit patients; patients surrounded by many similar historical members are better supported.
3. **Local density** — whether the patient sits in a dense or sparse region of the historical reference population; sparse regions indicate extrapolation beyond well-supported territory.
4. **HDBSCAN cross-check (diagnostic)** — does HDBSCAN treat this patient as a clean member of an existing dense region, or as noise/borderline? Used as a secondary flag, not a primary score input, given the sample-size caveat from Step 4.

---

## Step 8 – Construct the Clinical Confidence Score

Combine the similarity measures — GMM posterior probability, nearest-neighbor similarity, and local density — into a single Clinical Confidence Score. Exact weighting explored empirically.

```text
Clinical Confidence
=
GMM Posterior Membership
+
Nearest-Neighbor Similarity
+
Local Density
```

This score answers: **how strongly is this predicted high-benefit patient supported by previously observed high-benefit patients?**

It should **never** be interpreted as the probability that treatment will succeed, and — per the scope decision in this revision — it should also not be presented as a statement about the *accuracy* of the benefit prediction unless the Step 10 appendix check is run and supports that stronger claim. Absent that check, the honest framing is resemblance to precedent, not reliability of prediction.

---

## Step 9 – Assign Confidence Categories

### High Confidence
- High GMM posterior membership in a historical archetype
- Many nearby historical neighbors
- Dense local region

> This patient's clinical profile closely resembles previously observed high-benefit patients. The treatment recommendation has strong historical precedent.

### Medium Confidence
- Moderate posterior membership, split across archetypes
- Moderate neighbor agreement

> This patient shares characteristics with historical high-benefit patients but also exhibits some unique features. Clinical judgment should complement the model recommendation.

### Low Confidence
- Low posterior membership across all archetypes
- Few nearby neighbors, sparse local region
- Frequently flagged as noise/borderline by HDBSCAN

> Although the model predicts substantial treatment benefit, this patient's clinical profile has little historical precedent among previously observed high-benefit patients. The prediction is being made in a relatively unexplored region of the feature space and should be interpreted with additional clinical judgment.

---

## Step 9a – Visualize the Archetypes and Confidence Tiers

This is a plotting layer only — it does not feed back into clustering or the confidence score. It exists to make the archetypes and confidence tiers interpretable to a non-technical audience.

- **Project the same reduced (SHAP-selected) feature set used for clustering** — not the original 41 features — down to 2D via PCA, purely for visualization. This is a distinct, uncontroversial use of PCA from the one ruled out earlier for clustering itself: here it's just a coordinate system for plotting clusters GMM already found, not an input to forming them.
- Report variance explained by PC1+PC2 on the plot itself, so viewers don't over-read apparent separation or overlap as ground truth — at ~10-15 features, two components may only capture 40-60% of variance.
- Color training points by their GMM archetype assignment (not by anything PCA produces).
- Fit PCA once on the training archetype population; **transform** (don't refit) the ~60 test high-benefit patients into the same coordinate space, and overlay them with a distinct marker, colored by confidence tier. A low-confidence patient landing visibly outside all dense archetype clouds communicates "sparse/novel region" more intuitively than a table of posterior probabilities.
- **UMAP** is worth trying alongside PCA — it often separates clusters more cleanly at this scale by preserving local neighborhood structure rather than global variance. Same caveat in reverse: don't use UMAP output for anything beyond visualization, since inter-cluster distances in a UMAP plot aren't meaningful, only local grouping is.
- **Pairwise clinical-feature plots** (e.g., ED utilization vs. comorbidity burden, colored by cluster) are worth including alongside PCA/UMAP — for a non-technical audience, named clinical axes read more clearly than "PC1 vs PC2."

Minimal implementation:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords = pca.fit_transform(X_reduced)  # X_reduced = SHAP-selected training feature matrix
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

plt.scatter(coords[:, 0], coords[:, 1], c=gmm_labels, cmap="tab10", alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

# Test patients: transform, don't refit, so they land in the same space
test_coords = pca.transform(X_test_reduced)
```

---

## Interpretation

This framework does not validate the treatment-effect model. It answers a separate, complementary question:

- **Treatment-effect model:** *Which patients are predicted to benefit from intervention?*
- **Clinical Confidence Layer:** *How well does this patient's clinical profile resemble previously observed high-benefit patients?*

Together they give care managers and clinicians both a benefit prediction and a sense of how much historical precedent stands behind it — without overstating what the confidence score can claim on its own.

---

## Appendix (Optional) — Step 10: Ground-Truth Sanity Check

Because this project uses synthetic data with a known true-benefit formula, an optional, one-time check is available — **not** as a required part of the deliverable, and **not** framed as validating the model, but as a sanity read on whether the confidence framework is picking up a real signal.

**What it would do differently from the core pipeline:** re-run Step 3's feature selection using the true benefit-driving features (instead of SHAP), rebuild the confidence pipeline (Steps 4-9) on that feature set, and compare against the SHAP-based version using:

- **True-benefit trend by confidence tier** — does true benefit decline monotonically High → Medium → Low, and how much cleaner is that separation under true-formula features vs. SHAP-selected features?
- **Precision of high-benefit identification** — among predicted high-benefit patients, what fraction are genuinely high true-benefit, broken out by confidence tier and by feature-selection method?

If run, this comparison can inform which feature-selection approach to carry forward — but since the true formula won't be available on real claims data, this stays an appendix-level check on the *method*, not a component that real-world deployments of this framework would include.
