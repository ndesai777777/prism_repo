# PRISM Clinical Confidence Layer Using High-Benefit Patient Clustering

## Objective

The primary purpose of this analysis is **not** to improve the treatment-effect estimates produced by the ForestDRLearner. Instead, the objective is to develop a **Clinical Confidence Layer** that quantifies how much historical evidence supports an individual treatment recommendation.

The underlying intuition is that a patient predicted to benefit from intervention should inspire greater confidence if they closely resemble previously observed high-benefit patients. Conversely, if a patient is predicted to benefit but possesses a clinical profile unlike any historically observed high-benefit individual, then the recommendation should be interpreted more cautiously.

This framework therefore estimates **confidence in the prediction**, **not** confidence that treatment will succeed.

---

# Overall Workflow

```text
Training Data (700 Patients)
            │
            ▼
     ForestDRLearner
            │
            ▼
Predicted Individual Treatment Effects
            │
            ▼
 Select Top 20% Predicted High Benefit
            │
            ▼
 Discover High-Benefit Patient Clusters
            │
            ▼
 Build Historical Reference Library
──────────────────────────────────────────
Held-Out Test Data (300 Patients)
            │
            ▼
     ForestDRLearner
            │
            ▼
 Select Top 20% Predicted High Benefit
            │
            ▼
 Compare Each Patient to Historical Clusters
            │
            ▼
 Similarity Metrics
            │
            ▼
 Clinical Confidence Score
```

---

# Step 1 – Train the ForestDRLearner

Train the ForestDRLearner exactly as currently implemented using the 700-member training dataset.

For every training member estimate:

- Individual treatment effect
- Predicted benefit score

These predictions become the basis for constructing the historical reference population.

**Importantly, no clustering is performed on the entire population.**

The objective is specifically to characterize patients that the model predicts will benefit the most from intervention.

---

# Step 2 – Construct the Historical High-Benefit Reference Population

Rank every training patient according to predicted treatment benefit.

For consistency with the remainder of the PRISM project:

- **High Benefit:** Top 20%
- **Medium Benefit:** Middle 50%
- **Low Benefit:** Bottom 30%

Only the **High Benefit** group is used to build the Clinical Confidence Layer.

For the training population:

```text
700 Patients

↓

Top 20%

↓

Approximately 140 Historical High-Benefit Patients
```

These members become the historical reference population against which future predictions will be compared.

---

# Step 3 – Discover High-Benefit Patient Archetypes

Perform unsupervised clustering **only** on the historical high-benefit patients.

The purpose of clustering is **not** to discover treatment effects.

Treatment effects have already been estimated by the ForestDRLearner.

Instead, clustering answers the following question:

> **Are there distinct clinical phenotypes among patients predicted to benefit substantially from intervention?**

Potential clusters may represent groups such as:

- Frequent ED utilizers
- Complex chronic disease patients
- Behavioral health patients
- Social determinant–driven patients
- Medication adherence patients

These labels are only examples.

The actual cluster descriptions should emerge naturally from the data.

For every discovered cluster calculate:

- Number of patients
- Cluster centroid
- Average predicted benefit
- Average risk score
- Demographic summary
- Clinical summary
- Utilization summary
- Social Determinants of Health (SDOH) summary

These summaries create clinically interpretable "patient archetypes" that represent different types of members predicted to benefit from intervention.

---

# Step 4 – Validate Cluster Stability

Because clustering is exploratory, it is important to verify that the discovered patient groups are stable and reproducible.

Evaluate clustering quality using:

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index

Optionally compare multiple clustering algorithms:

- K-Means
- Gaussian Mixture Models
- HDBSCAN

The objective is to determine whether similar high-benefit patient groupings emerge consistently across different clustering techniques.

This directly addresses concerns regarding reproducibility and robustness of the discovered clinical phenotypes.

---

# Step 5 – Score the Held-Out Test Population

Using the trained ForestDRLearner,

predict treatment benefit for the independent 300-member test population.

Again,

identify only the predicted high-benefit patients.

Example:

```text
300 Patients

↓

Top 20%

↓

Approximately 60 Predicted High-Benefit Patients
```

These patients are now evaluated against the historical reference clusters.

---

# Step 6 – Compare Test Patients to Historical High-Benefit Clusters

For every predicted high-benefit patient in the test set,

measure how well they fit the historical high-benefit clusters.

The objective is **not** to determine whether treatment will work.

The objective is to determine whether this patient resembles previously observed high-benefit patients.

Possible similarity metrics include:

## 1. Distance to Nearest Cluster Centroid

Measure how close each patient is to the center of the most similar historical cluster.

Smaller distance indicates stronger similarity.

---

## 2. Nearest Neighbor Distance

Locate the patient's nearest historical high-benefit patients.

Compute:

- Mean distance
- Median distance
- Distance to the *k* nearest neighbors

Patients surrounded by many similar historical members receive stronger support.

---

## 3. Local Density

Determine whether the patient lies within:

- a dense historical region

or

- a sparse historical region

Patients located in sparse regions represent extrapolation beyond well-supported portions of the feature space.

---

## 4. Cluster Membership Confidence (Optional)

If Gaussian Mixture Models are used,

obtain posterior cluster probabilities.

Example:

```text
Cluster 1   92%

Cluster 2    6%

Cluster 3    2%
```

Higher posterior probability indicates stronger cluster membership.

---

# Step 7 – Construct the Clinical Confidence Score

Rather than relying on a single metric,

combine several complementary similarity measures.

For example:

```text
Clinical Confidence

=

Cluster Similarity

+

Nearest-Neighbor Similarity

+

Local Density
```

The exact weighting can be explored empirically.

The resulting score answers:

> **How strongly is this predicted high-benefit patient supported by previously observed high-benefit patients?**

Importantly,

this score should **never** be interpreted as the probability that treatment will work.

Instead,

it reflects the amount of historical evidence supporting the model's prediction.

---

# Step 8 – Assign Confidence Categories

Patients can then be grouped into clinically interpretable confidence levels.

---

## High Confidence

Characteristics:

- Close to historical cluster
- Many nearby neighbors
- Dense local region

Interpretation:

> This patient's clinical profile closely resembles previously observed high-benefit patients. The treatment recommendation is strongly supported by historical evidence.

---

## Medium Confidence

Characteristics:

- Moderate similarity
- Moderate neighbor agreement

Interpretation:

> This patient shares characteristics with historical high-benefit patients but also exhibits some unique features. Clinical judgment should complement the model recommendation.

---

## Low Confidence

Characteristics:

- Large cluster distance
- Few nearby neighbors
- Sparse local region

Interpretation:

> Although the ForestDRLearner predicts substantial treatment benefit, this patient's clinical profile is not well represented among historical high-benefit patients. The prediction should therefore be interpreted cautiously because it is being made in a relatively unexplored region of the feature space.

---

# Step 9 – Evaluate the Clinical Confidence Framework

Because this project uses synthetic data,

true treatment benefit is available.

This provides a unique opportunity to evaluate whether the confidence framework behaves as intended.

---

## Analysis 1 – Treatment Effect Accuracy by Confidence Group

Compare:

- High Confidence
- Medium Confidence
- Low Confidence

using:

- Mean Absolute Error (MAE)
- RMSE
- Pearson Correlation
- Spearman Correlation

**Hypothesis**

Higher-confidence patients should demonstrate more accurate treatment-effect estimates.

---

## Analysis 2 – Precision of High-Benefit Identification

Among patients predicted to be high benefit,

compare:

- High Confidence
- Medium Confidence
- Low Confidence

using:

- Percentage truly high benefit
- Average true benefit
- Average prediction error

**Hypothesis**

High-confidence predictions should contain a greater proportion of genuinely high-benefit patients.

---

## Analysis 3 – Distribution of True Benefit

Compare the distribution of known treatment benefit across confidence groups.

Example:

| Confidence | Average True Benefit |
|------------|---------------------:|
| High | 8.9% |
| Medium | 7.4% |
| Low | 4.8% |

A clear monotonic trend would support the usefulness of the Clinical Confidence Layer.

---

# Interpretation

This framework should **not** be presented as validating the ForestDRLearner.

Instead,

the clustering layer provides additional context for interpreting treatment recommendations.

The ForestDRLearner answers:

> **Which patients are predicted to benefit from intervention?**

The Clinical Confidence Layer answers:

> **How well does this patient's clinical profile resemble previously observed high-benefit patients?**

These questions are complementary.

The first estimates individualized treatment benefit.

The second quantifies how much historical evidence supports that estimate.

Together, they produce a more transparent and clinically interpretable decision-support framework that augments the causal model with an estimate of prediction confidence, helping care managers and clinicians identify recommendations that are both high-impact and well-supported by historical patient patterns.