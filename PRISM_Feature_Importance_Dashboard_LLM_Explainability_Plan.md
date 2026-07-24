# PRISM Feature Importance Dashboard — LLM Explainability Plan

---

## Part 1: Overall Dashboard System Architecture

### Purpose

The explainability dashboard should be designed as a complete decision-support application rather
than simply a visualization tool.

Its purpose is to combine:

- Uplift modeling
- Clinical confidence estimation
- Explainability analytics
- LLM-generated narratives

into a single workflow that supports outreach prioritization for care management teams.

The dashboard should **not** perform model training. Instead, it should consume outputs from the
trained DR Learner pipeline and transform those outputs into actionable clinical explanations.

---

### End-to-End Application Workflow

```text
Historical Data
        │
        ▼
Train DR Learner
        │
        ▼
Persist Trained Model
        │
──────────────────────────────────────────────────────────────
        Production Scoring Begins
──────────────────────────────────────────────────────────────
        │
Receive New Member Population
        │
        ▼
Run DR Learner Predictions
        │
        ▼
Generate Benefit Scores
        │
        ▼
Generate Clinical Confidence Scores
        │
        ▼
Generate Local Feature Attribution
        │
        ▼
Load Global Model Statistics
        │
        ▼
Calculate Explanation Priority Scores
        │
        ▼
Store Structured Dashboard Data
        │
        ▼
Dashboard Application
        │
        ▼
LLM Narrative Generation
        │
        ▼
Care Manager Reviews Recommendation
```

---

### Step 1: Model Training

The DR Learner should be trained offline using historical intervention data. Once model development
is complete, the selected production model should be frozen and versioned.

The dashboard should never retrain the model. Instead, it should simply load the approved
production model.

---

### Step 2: Population Scoring

Whenever a new population of members becomes available, the production DR Learner should score
every member.

For each member the model should estimate:

| Output | Description |
|--------|-------------|
| Predicted ED risk without intervention | μ₀(x) |
| Predicted ED risk with intervention | μ₁(x) |
| Estimated intervention benefit | μ₀(x) − μ₁(x) |

Where:

```
benefit_score = predicted ED risk without intervention − predicted ED risk with intervention
```

Every member therefore receives an estimated treatment benefit.

---

### Step 3: Outreach Budget Selection

The dashboard should allow a program manager to determine how many members can realistically
receive intervention.

Rather than using a fixed treatment threshold, the dashboard should support operational constraints
such as:

- Maximum number of outreach slots
- Outreach budget
- Percentage of the population to target

**Example:**

```
Available outreach capacity: 25,000 members
```

The application should then automatically select the highest-ranked members according to predicted
intervention benefit. This allows the targeting strategy to adapt to changing operational resources
without modifying the underlying model.

---

### Step 4: Clinical Confidence Assessment

For every selected outreach candidate, the dashboard should calculate a clinical confidence score
using the confidence estimation methodology developed separately.

Benefit and confidence represent different concepts:

| Concept | What it estimates |
|---------|-------------------|
| Benefit | Expected intervention impact |
| Confidence | Reliability of that prediction |

Both values should always be displayed together.

| Benefit | Confidence | Interpretation |
|---------|------------|----------------|
| High | High | Strong recommendation |
| High | Low | Potential outreach candidate requiring greater clinical review |

---

### Step 5: Local Attribution

For every scored member, the attribution pipeline should estimate feature-level contributions
explaining why the model predicted the observed intervention benefit.

Depending on the selected implementation, this may be based on:

- GLMNet contribution differences
- SHAP-style probability contributions
- Another approved additive attribution method

The output should be **one signed contribution per feature**:

- Positive contributions increase estimated intervention benefit
- Negative contributions decrease estimated intervention benefit

---

### Step 6: Global Model Statistics

Separately from member scoring, the system should maintain global model statistics describing
feature behavior across the full scored population.

| Statistic | Description |
|-----------|-------------|
| Global feature importance | Raw mean absolute contribution |
| Normalized feature importance | Scaled 0–1 relative to top feature |
| Average signed contribution | Population mean signed SHAP |
| Mean absolute contribution | Population mean |SHAP| |
| Clinical category | Feature grouping |
| Preferred display name | Human-readable label |

These statistics should only change when the production model is retrained.

---

### Step 7: Explanation Priority Calculation

The dashboard should not ask the LLM to determine which features are important. Instead, the
analytics pipeline should calculate explanation priorities before any LLM interaction.

**Recommended calculation:**

```
signed_driver_priority = local_signed_contribution × normalized_global_importance
```

Additional metrics computed at this stage:

| Metric | Purpose |
|--------|---------|
| Absolute driver strength | Determines which features deserve attention |
| Difference from average contribution | Contextualizes the member vs. population |
| Contribution ratio | Relative magnitude vs. average |
| Population percentile | Member's feature value rank |

These values become structured evidence for the LLM.

---

### Step 8: Dashboard Data Layer

Before the dashboard is displayed, all analytics should already be complete. The dashboard should
load structured data rather than recompute model outputs.

**Recommended data assets:**

| Asset | Contents |
|-------|----------|
| Global Model Knowledge | One record per feature |
| Member Attribution | One record per member |
| Explanation Priority | Ranked explanation drivers for each member |
| LLM Explanation Input | Curated structured information sent to the language model |

This separation allows analytics and visualization to evolve independently.

---

### Step 9: Dashboard User Experience

The dashboard should present information in the following order.

#### Population View

Displays:

- Distribution of benefit scores
- Distribution of confidence scores
- Benefit deciles
- Risk tiers
- Number of members selected for outreach

Program managers first determine outreach capacity. The application then identifies the
highest-ranked members.

#### Member Selection

After selecting a member, the dashboard retrieves:

- Benefit prediction
- Confidence score
- Attribution data
- Ranked explanation drivers

No model computation occurs at this stage. All values have already been generated by the analytics
pipeline.

#### Explainability View

The dashboard displays:

| Section | Content |
|---------|---------|
| Member summary | Benefit score, confidence score, risk tier, predicted ED risk with/without intervention |
| Waterfall plot | Positive drivers, negative drivers |
| Feature comparisons | Member values versus population distribution |
| Global importance | Population-level feature ranking |
| Explanation priority | Derived ranking for this member |

The dashboard should expose both raw attribution values and the derived explanation ranking.

---

### Step 10: LLM Narrative Generation

The LLM represents the **final stage** of the application. It is not part of the predictive model.

Its responsibility is to translate structured analytics into clinician-friendly language.

The LLM should receive four structured inputs:

1. Global Model Knowledge
2. Member Attribution
3. Explanation Priority
4. Member Summary (benefit, confidence, risk, predictions)

**The LLM should never:**

- Compute feature rankings
- Perform statistical calculations
- Select important features
- Infer causal relationships

Instead, it should describe the evidence produced by the analytics pipeline. This design makes
every statement in the narrative traceable to structured model outputs.

---

### Recommended Technology Stack

**Prototype implementation:**

| Component | Technology |
|-----------|------------|
| Backend analytics | Python (Pandas, NumPy, SHAP or GLMNet attribution) |
| Development environment | VS Code |
| Predictive model | DR Learner (persisted) |
| LLM integration | OpenAI API |
| Dashboard frontend | Streamlit |

The dashboard should initially operate as a local application:

- The Python backend performs all analytics
- The Streamlit frontend displays dashboard components
- The OpenAI API is called only when a member explanation is requested

This architecture minimizes LLM costs while ensuring that explanations always reflect the latest
structured analytics.

---
---

## Part 2: Three-Layer Explainability Architecture

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

### Layer 1: Global Model Knowledge

The first layer captures information that is true for the model as a whole and changes only when
the model is retrained.

This dataset should contain **one record per feature**.

#### Recommended Fields

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

#### Example

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

### Layer 2: Member Attribution

The second layer stores member-specific model outputs.

This layer contains **one JSON object for each scored member**.

#### Recommended Information

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

#### Example

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

### Layer 3: Explanation Priority

The third layer determines which features deserve emphasis in the explanation.

This layer combines global model knowledge with member-specific attribution.

#### Recommended Calculations

**Weighted priority:**

```
weighted_priority = local_signed_contribution × normalized_global_feature_importance
```

Two values should be computed:

| Metric | Formula | Purpose |
|--------|---------|---------|
| `absolute_driver_strength` | `abs(local_contribution) × normalized_global_importance` | Determines which features deserve attention |
| `signed_driver_priority` | `local_signed_contribution × normalized_global_importance` | Preserves whether a feature increases or decreases estimated benefit |

#### Explanation Roles

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

### LLM Design Philosophy

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

### Statistical Context

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

#### Pre-Computed Comparison Metrics

| Metric | Description |
|--------|-------------|
| Difference from average signed contribution | `local_contribution − avg_signed_contribution` |
| Contribution ratio relative to population average | `local_contribution / avg_signed_contribution` |
| Population percentile | Member's feature value rank within the population |
| Difference from population mean | `feature_value − population_mean` |

These derived metrics should be included as structured fields whenever possible.

---

### Clinical Confidence Integration

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

#### High Confidence Example

> This member is classified as a high-benefit outreach candidate with high prediction confidence.
> The estimated benefit is supported by stable treatment-effect estimates and strong agreement
> among the primary explanatory features.

#### Lower Confidence Example

> Although this member is predicted to have a high intervention benefit, the confidence associated
> with this estimate is moderate. The recommendation should therefore be interpreted alongside
> clinical judgment.

This distinction is important because predicted benefit and confidence represent **different
concepts**.

---

### Separation of Responsibilities

The explainability system should clearly separate responsibilities:

#### Analytics Layer

Responsible for:

- Computing benefit scores
- Computing confidence scores
- Computing local feature contributions (SHAP)
- Computing global feature importance
- Computing weighted explanation priorities
- Computing comparison statistics
- Assigning explanation roles

#### LLM Layer

Responsible only for:

- Producing fluent natural-language explanations
- Summarizing structured evidence
- Preserving contribution direction
- Explaining confidence
- Avoiding causal overstatements

The LLM should **never** perform ranking, weighting, thresholding, or feature selection. Those
operations belong entirely within the analytics pipeline.

---

### Benefits of This Architecture

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
| 6 | Build Streamlit dashboard prototype (population view → member view → explanation view) |
| 7 | Connect OpenAI API for on-demand narrative generation |
| 8 | User testing with care management analysts |

---

*Last updated: July 2026*
