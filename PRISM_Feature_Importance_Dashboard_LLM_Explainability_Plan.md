# PRISM Feature Importance Dashboard — LLM Explainability Plan

## Three-Layer Explainability Architecture

### Overview

To improve explainability, reproducibility, and LLM reliability, the explainability pipeline is
explicitly divided into three independent layers:

1. **Global Model Knowledge Layer**
2. **Member Attribution Layer**
3. **Explanation Priority Layer**

The LLM should not perform feature ranking or statistical calculations. Instead, all mathematical
operations should be completed before the LLM is called. The LLM's responsibility is to convert
structured evidence into a concise clinical narrative.

This architecture makes explanations **deterministic**, **auditable**, and **easier to validate**.

---

## Layer 1: Global Model Knowledge

The first layer captures information that is true for the model as a whole and changes only when
the model is retrained.

This dataset should contain **one record per feature**.

### Recommended Fields

| Field | Description |
|-------|-------------|
| Feature name | Internal column name |
| Preferred display name | Human-readable label for dashboards |
| Clinical category | Grouping (Risk, Utilization, Demographics, etc.) |
| Global feature importance | Raw mean absolute SHAP contribution |
| Normalized global feature importance | Scaled 0–1 relative to the top feature |
| Global feature rank | Ordinal rank by importance |
| Average signed contribution | Mean signed SHAP across the population |
| Mean absolute contribution | Mean |SHAP| across the population |
| Whether the feature is allowed in LLM explanations | Boolean governance flag |

### Example

```json
{
  "current_risk_score": {
    "normalized_importance": 0.90,
    "global_rank": 2,
    "avg_signed_contribution": 0.0048,
    "clinical_category": "Risk"
  }
}
```

This layer represents permanent knowledge about the trained model rather than any individual
member.

---

## Layer 2: Member Attribution

The second layer stores member-specific model outputs.

This layer contains **one JSON object for each scored member**.

### Recommended Information

| Field | Description |
|-------|-------------|
| Member identifier | Unique ID |
| Benefit score | Predicted CATE (treatment effect) |
| Benefit group | High / Medium / Low benefit classification |
| Uplift decile | 1–10 ranking by benefit score |
| Predicted ED risk with intervention | μ₁(x) |
| Predicted ED risk without intervention | μ₀(x) |
| Clinical confidence score | From the Clinical Confidence Layer |
| Risk tier | Current risk stratification |
| Local feature contributions | Per-feature SHAP values for this member |
| Feature values | Actual observed feature values |
| Population comparison statistics | Percentile, z-score, difference from mean |

### Example

```json
{
  "member_id": 12345,
  "benefit_score": 0.054,
  "confidence_score": 0.93,
  "risk_tier": "High",
  "features": [
    {
      "feature": "current_risk_score",
      "feature_value": 72.1,
      "population_mean": 43.7,
      "population_percentile": 96,
      "local_contribution": 0.0120
    }
  ]
}
```

This layer contains only facts about the selected member. **No feature ranking should occur here.**

---

## Layer 3: Explanation Priority

The third layer determines which features deserve emphasis in the explanation.

This layer combines global model knowledge with member-specific attribution.

### Recommended Calculations

**Weighted priority:**

```
weighted_priority = local_signed_contribution × normalized_global_feature_importance
```

Two values should be computed:

| Metric | Formula | Purpose |
|--------|---------|---------|
| `absolute_driver_strength` | `abs(local_contribution) × normalized_global_importance` | Determines which features deserve attention |
| `signed_driver_priority` | `local_signed_contribution × normalized_global_importance` | Preserves whether a feature increases or decreases estimated benefit |

### Explanation Roles

Each feature should be assigned an explanation role based on its priority scores:

| Role | Criteria |
|------|----------|
| **Primary positive driver** | High absolute strength, positive signed priority |
| **Supporting positive driver** | Moderate absolute strength, positive signed priority |
| **Offsetting factor** | Moderate-to-high absolute strength, negative signed priority |
| **Context only** | Low absolute strength (included for completeness) |
| **Suppressed** | Governance-excluded or negligible contribution |

This layer exists solely to determine explanation priority. It is not intended to replace SHAP
values or become a new feature importance metric.

---

## LLM Design Philosophy

The LLM should **never** calculate rankings or statistical metrics.

Instead, it should receive three structured inputs:

1. Global feature knowledge
2. Member attribution information
3. Explanation priority information

The explanation priority layer determines **which** features should be emphasized.
The member attribution layer provides **supporting statistics**.
The global feature layer provides **population-level context**.

The LLM simply combines these three sources into a readable explanation.

---

## Statistical Context

The LLM should have access to both the weighted priority score and the raw local contribution.
These values answer different questions:

| Value | Question it answers |
|-------|---------------------|
| Weighted priority score | "Which features should receive the most attention?" |
| Raw local contribution | "How much did this feature change this member's estimated benefit?" |

Because both values are available, the LLM can generate explanations such as:

> Current risk score is one of the model's strongest predictors of intervention benefit overall.
> For this member, it contributes substantially more than the average contribution observed across
> the population and is therefore one of the primary reasons this member is recommended for
> outreach.

The LLM should not infer statistical relationships itself. Instead, all comparison metrics should
be computed before the prompt is generated.

### Pre-Computed Comparison Metrics

| Metric | Description |
|--------|-------------|
| Difference from average signed contribution | `local_contribution − avg_signed_contribution` |
| Contribution ratio relative to population average | `local_contribution / avg_signed_contribution` |
| Population percentile | Member's feature value rank within the population |
| Difference from population mean | `feature_value − population_mean` |

These derived metrics should be included as structured fields whenever possible.

---

## Clinical Confidence Integration

The clinical confidence score should become a **first-class input** to the explainability pipeline
rather than simply another dashboard metric.

Every member explanation should include:

| Field | Example |
|-------|---------|
| Confidence score | 0.93 |
| Confidence level | High / Moderate / Low |
| Confidence interpretation | Narrative sentence |

The LLM should explain not only **why** a member is predicted to benefit from intervention, but
also **how confident** the model is in that recommendation.

### High Confidence Example

> This member is classified as a high-benefit outreach candidate with high prediction confidence.
> The estimated benefit is supported by stable treatment-effect estimates and strong agreement
> among the primary explanatory features.

### Lower Confidence Example

> Although this member is predicted to have a high intervention benefit, the confidence associated
> with this estimate is moderate. The recommendation should therefore be interpreted alongside
> clinical judgment.

This distinction is important because predicted benefit and confidence represent **different
concepts**.

---

## Separation of Responsibilities

The explainability system should clearly separate responsibilities:

### Analytics Layer

Responsible for:

- Computing benefit scores
- Computing confidence scores
- Computing local feature contributions (SHAP)
- Computing global feature importance
- Computing weighted explanation priorities
- Computing comparison statistics
- Assigning explanation roles

### LLM Layer

Responsible only for:

- Producing fluent natural-language explanations
- Summarizing structured evidence
- Preserving contribution direction
- Explaining confidence
- Avoiding causal overstatements

The LLM should **never** perform ranking, weighting, thresholding, or feature selection. Those
operations belong entirely within the analytics pipeline.

---

## Benefits of This Architecture

| Benefit | Description |
|---------|-------------|
| Clear separation | Modeling, analytics, and narrative generation are independent |
| Deterministic logic | Explanation priorities can be audited and reproduced |
| Minimal hallucination risk | LLM receives only structured evidence, not raw data |
| Easier governance | All ranking decisions occur before prompt generation |
| Simpler maintenance | Prompt wording can change without modifying analytics |
| Reusable framework | Supports future model families beyond the initial DR Learner |

---

## Implementation Roadmap

| Phase | Deliverable |
|-------|-------------|
| 1 | Export global model knowledge JSON from existing SHAP outputs |
| 2 | Export member attribution JSON from scored output + member SHAP values |
| 3 | Build explanation priority calculator (Layer 3 logic) |
| 4 | Design LLM prompt template consuming all three layers |
| 5 | Integrate clinical confidence score as first-class input |
| 6 | Dashboard integration and user testing |

---

*Last updated: July 2026*
