# PRISM Feature Importance Dashboard And LLM Explainability Plan

## Purpose

This planning document defines the desired explainability layer for the PRISM intervention benefit modeling workflow. The goal is to support a future dashboard where a care management user can select a member and understand why the model estimates that member as high, medium, or low intervention benefit.

The dashboard should not only show that a member has a high predicted benefit score. It should explain the main feature-level reasons behind that benefit score, distinguish positive and negative benefit drivers, compare the member's drivers against global model behavior, and provide an LLM-generated narrative that is grounded in structured model outputs.

The intended end state is:

```text
This member is a strong outreach candidate because they are in the top uplift decile,
their predicted ED risk is meaningfully lower under intervention than under no intervention,
and their strongest benefit-increasing drivers are recent ED utilization, elevated risk score,
and clinical complexity.
```

The explanation should be useful to business, clinical, and analytics audiences while remaining faithful to the model.

## Core Principle

The dashboard should separate three related but different ideas:

| Concept | Question answered | Example |
|---|---|---|
| Predicted benefit | How much does the model estimate this member benefits from intervention? | Benefit score = 0.0538 |
| Local feature contribution | Which features pushed this member's benefit score up or down? | `current_risk_score` contributed +0.012 |
| Global feature importance | Which features matter most across the population? | `ed_visits_last_6m` is a top global benefit driver |

The strongest explanation comes from the intersection of local and global evidence:

```text
globally important feature
+
large signed contribution for this member
=
strong member-specific explanation driver
```

This avoids over-explaining features that happen to have a local contribution but are weak, noisy, administrative, or not clinically meaningful at the population level.

## Modeling Context

The current uplift modeling workflow estimates treatment benefit as:

```text
benefit_score = pred_ed_if_control - pred_ed_if_treated
```

Where:

| Field | Meaning |
|---|---|
| `pred_ed_if_control` | Predicted 90-day ED probability if the member does not receive intervention |
| `pred_ed_if_treated` | Predicted 90-day ED probability if the member receives intervention |
| `benefit_score` | Estimated ED risk reduction from intervention |
| `uplift_decile` | Member's ranking by predicted benefit, where decile 1 is highest benefit |

A higher benefit score means the model estimates a larger reduction in ED risk under intervention.

For member-level explainability, feature contributions should explain why the model estimated a larger or smaller treatment benefit for that member.

## Desired Dashboard User Story

A user opens the dashboard and selects a member.

The dashboard should answer:

1. Is this member recommended for outreach?
2. How high is this member's predicted intervention benefit?
3. Is the member high risk, high benefit, both, or neither?
4. Which features are driving the benefit score upward?
5. Which features are reducing or offsetting the benefit score?
6. Are those features important globally, or are they only local/contextual signals?
7. How does this member compare with the population on those features?
8. What plain-English explanation should a care management stakeholder read?

## Dashboard Layout

### Section 1: Member Summary Header

The first section should provide the member-level targeting decision at a glance.

Recommended fields:

| Field | Description |
|---|---|
| `member_id` | Stable member identifier |
| `benefit_score` | Predicted intervention benefit |
| `uplift_decile` | Benefit ranking group |
| `benefit_group` | High, medium, or low benefit group |
| `pred_ed_if_control` | Predicted ED risk without intervention |
| `pred_ed_if_treated` | Predicted ED risk with intervention |
| `risk_tier` | Low, Medium, High, or Very High baseline risk tier |
| `current_risk_score` | Current risk score |
| `actual_outcome` | Observed 90-day ED outcome, if available |
| `treatment_flag` | Whether member historically received intervention |

Recommended visual:

```text
Member 12345
Benefit group: High benefit
Uplift decile: 1
Benefit score: 0.0538
Predicted ED without intervention: 0.0839
Predicted ED with intervention: 0.0301
Risk tier: Medium
```

Interpretation:

```text
This member is ranked in the top benefit decile. The model estimates that intervention
reduces 90-day ED risk by 5.4 percentage points.
```

### Section 2: Benefit Score Waterfall

This section should explain how feature contributions combine to explain the member's benefit estimate.

Recommended visual:

- Waterfall chart
- Starts from baseline/average benefit
- Adds positive feature contributions
- Subtracts negative feature contributions
- Ends at the member's predicted benefit score

Conceptual structure:

```text
Average model benefit
    + current_risk_score contribution
    + ed_visits_last_6m contribution
    + total_cost_last_6m contribution
    - pcp_visits_last_6m contribution
    - opioid_flag contribution
= member benefit score
```

Important note:

For SHAP-style explanations, feature contributions usually sum to:

```text
model output = baseline + sum(feature contributions)
```

Therefore, the dashboard should include the baseline term. It should not imply that feature contributions alone sum to the benefit score unless the baseline is included.

Recommended dashboard language:

```text
Feature contributions explain how this member's predicted benefit differs from
the model's average predicted benefit.
```

### Section 3: Top Positive And Negative Benefit Drivers

This section should separate benefit-increasing and benefit-decreasing features.

The sign must be preserved.

| Sign | Meaning |
|---|---|
| Positive contribution | Feature increases estimated benefit |
| Negative contribution | Feature decreases estimated benefit |
| Near zero contribution | Feature has little member-specific effect on benefit |

Recommended visual:

- Diverging horizontal bar chart
- Positive drivers to the right
- Negative drivers to the left
- Color coding:
  - Positive benefit drivers: green or blue
  - Negative benefit drivers: red or muted orange
  - Contextual/low-importance drivers: gray

Example:

| Feature | Contribution | Direction |
|---|---:|---|
| `current_risk_score` | +0.0120 | Increases benefit |
| `ed_visits_last_6m` | +0.0090 | Increases benefit |
| `total_cost_last_6m` | +0.0065 | Increases benefit |
| `pcp_visits_last_6m` | -0.0040 | Decreases benefit |
| `opioid_flag` | -0.0025 | Decreases benefit |

LLM instruction:

```text
Do not describe negative contributors as reasons to target the member.
Negative contributors are offsetting factors.
```

### Section 4: Global Importance Weighted Local Drivers

This is the central concept from the planning discussion.

Not all local feature contributions should receive equal attention. A feature should be highlighted when it is both:

1. Important globally across the model population
2. Important locally for the selected member

Recommended priority score:

```text
absolute_driver_strength =
    abs(member_feature_contribution)
    * normalized_global_feature_importance
```

Recommended signed priority:

```text
signed_driver_priority =
    member_feature_contribution
    * normalized_global_feature_importance
```

Use:

| Metric | Use |
|---|---|
| `absolute_driver_strength` | Rank the most important explanation drivers |
| `signed_driver_priority` | Preserve benefit-increasing vs benefit-decreasing direction |

Example:

| Feature | Global importance | Member contribution | Signed priority | Explanation role |
|---|---:|---:|---:|---|
| `current_risk_score` | 0.90 | +0.0120 | +0.0108 | Primary positive driver |
| `ed_visits_last_6m` | 1.00 | +0.0090 | +0.0090 | Primary positive driver |
| `total_cost_last_6m` | 0.65 | +0.0065 | +0.0042 | Supporting positive driver |
| `county_County_B` | 0.08 | +0.0040 | +0.0003 | Context only |
| `pcp_visits_last_6m` | 0.50 | -0.0040 | -0.0020 | Offsetting factor |

This prevents weak global features from dominating the LLM explanation.

Example bad explanation to avoid:

```text
This member is high benefit because they live in County B.
```

Better explanation:

```text
County is present as a minor contextual signal, but the primary benefit drivers are
current risk score, recent ED use, and clinical/utilization history.
```

### Section 5: Feature Value Compared With Population

Feature contribution and feature value are not the same thing.

The dashboard should show both:

| Field | Meaning |
|---|---|
| Feature value | The member's actual value |
| Population average | Average value across scored population |
| Percentile | Where the member falls relative to others |
| Local contribution | How much the feature changed this member's benefit estimate |

Recommended visual:

- Feature comparison strip
- Population range as a line
- Population average marker
- Member value marker
- Contribution direction and magnitude shown beside it

Example:

```text
current_risk_score
Population average: 43.7
Member value: 72.1
Percentile: 96th
Contribution to benefit: +0.012
```

The LLM can then say:

```text
This member's current risk score is far above average and contributes positively
to the estimated benefit score.
```

But the LLM should not infer causality from feature value alone. It should rely on the local contribution field for model explanation.

### Section 6: Risk Tier Versus Benefit Context

The dashboard should show whether the member is:

| Group | Meaning |
|---|---|
| High risk, high benefit | Strong outreach candidate |
| High risk, low benefit | Clinically risky but not estimated as highly impactable |
| Low/moderate risk, high benefit | May be missed by risk-only targeting |
| Low risk, low benefit | Lower outreach priority |

Risk tier thresholds:

| Risk tier | Current risk score rule |
|---|---|
| Low | `< 35` |
| Medium | `35` to `< 55` |
| High | `55` to `< 75` |
| Very High | `>= 75` |

Benefit group definitions:

| Benefit group | Definition |
|---|---|
| High benefit | Uplift deciles 1-2, top 20% |
| Medium benefit | Uplift deciles 3-7, middle 50% |
| Low benefit | Uplift deciles 8-10, bottom 30% |

Recommended visual:

- Risk tier by benefit group stacked bar chart
- Highlight selected member's risk tier and benefit group

Purpose:

```text
Risk tier captures baseline risk.
Benefit group captures estimated impactability.
They are related but not interchangeable.
```

### Section 7: LLM Explanation Panel

The LLM should generate a concise explanation grounded entirely in structured model outputs.

Recommended narrative structure:

1. Targeting summary
2. Predicted risk contrast
3. Top positive benefit drivers
4. Offsetting negative drivers
5. Risk-versus-benefit interpretation
6. Caveat

Example output:

```text
This member is a strong outreach candidate because they are in uplift decile 1,
the highest predicted benefit group. The model estimates that intervention lowers
their 90-day ED risk from 8.4% without intervention to 3.0% with intervention,
for an estimated benefit of 5.4 percentage points.

The strongest positive benefit drivers are current risk score, recent ED utilization,
and total cost in the last six months. These features are both important global
benefit drivers and unusually influential for this member.

Some features slightly offset the benefit estimate, including PCP visit history,
but the positive drivers are stronger overall. This explanation should be interpreted
as model-based prioritization support, not as proof that intervention will prevent
an ED visit for this individual.
```

## Required Data Tables

### Member-Level Scored Output

One row per member.

Required fields:

| Field | Description |
|---|---|
| `member_id` | Stable member identifier |
| `benefit_score` | Predicted intervention benefit |
| `uplift_decile` | Benefit ranking |
| `benefit_group` | High, medium, low |
| `pred_ed_if_treated` | Predicted ED risk with intervention |
| `pred_ed_if_control` | Predicted ED risk without intervention |
| `risk_tier` | Risk tier |
| `current_risk_score` | Current risk score |
| `outcome_ed_90d` | Observed outcome |
| `intervention_flag` | Historical treatment flag |

### Member-Level Benefit Attribution Table

One row per member-feature pair.

Recommended fields:

| Field | Description |
|---|---|
| `member_id` | Stable member identifier |
| `feature` | Feature name |
| `feature_value` | Member's value for the feature |
| `population_mean` | Population average for the feature |
| `population_percentile` | Member percentile for numeric features |
| `global_feature_importance` | Overall feature importance for benefit |
| `normalized_global_feature_importance` | Global importance scaled 0 to 1 |
| `member_feature_contribution` | Local signed contribution to benefit |
| `absolute_member_contribution` | Absolute local contribution |
| `signed_driver_priority` | Local signed contribution weighted by global importance |
| `absolute_driver_strength` | Absolute local contribution weighted by global importance |
| `direction` | Increases benefit, decreases benefit, or neutral |
| `explanation_role` | Primary driver, supporting driver, offsetting factor, context only |
| `rank_for_member` | Rank within member explanation |

### Global Benefit Feature Importance Table

One row per feature.

Recommended fields:

| Field | Description |
|---|---|
| `feature` | Feature name |
| `avg_signed_contribution` | Average signed contribution to benefit |
| `mean_abs_contribution` | Average absolute contribution |
| `global_rank` | Rank by global benefit importance |
| `clinical_category` | Risk, utilization, clinical, SDOH, pharmacy, demographic, administrative |
| `llm_explanation_allowed` | Whether the feature should be described in LLM narrative |
| `preferred_display_name` | Human-readable feature name |

### LLM Explanation Input Table

One row per member, already curated for LLM use.

Recommended fields:

| Field | Description |
|---|---|
| `member_id` | Stable member identifier |
| `benefit_score` | Predicted benefit |
| `uplift_decile` | Benefit decile |
| `benefit_group` | High, medium, low |
| `pred_ed_if_control` | Predicted ED risk without intervention |
| `pred_ed_if_treated` | Predicted ED risk with intervention |
| `risk_tier` | Risk tier |
| `top_positive_drivers_json` | Top benefit-increasing drivers |
| `top_negative_drivers_json` | Top benefit-decreasing drivers |
| `contextual_features_json` | Low-priority context features |
| `dashboard_caveats` | Required caveats |

## Attribution Method Options

### Option 1: GLMNet Contribution Difference

For GLMNet logistic regression, one approach is:

```text
feature contribution to benefit =
    feature contribution in control model
    - feature contribution in treated model
```

This aligns with:

```text
benefit_score = pred_ed_if_control - pred_ed_if_treated
```

However, the cleanest additive decomposition is usually on the log-odds scale, not the raw probability scale.

Pros:

- Easy to compute from GLMNet coefficients
- Transparent
- Fast
- Good for dashboard prototypes

Cons:

- Log-odds contributions may not sum directly to probability benefit
- Needs careful explanation for business users

### Option 2: SHAP-Style Probability Contribution

Use SHAP values or an equivalent local attribution method on predicted probabilities.

Possible approach:

```text
control_probability_shap_contribution
- treated_probability_shap_contribution
= local contribution to benefit
```

Pros:

- More aligned with probability-scale business interpretation
- Can produce local contribution tables
- Better for LLM explanation

Cons:

- More computationally complex
- Need to verify additivity assumptions
- More sensitive to implementation details

### Option 3: Dashboard Hybrid

Use GLMNet signed contribution difference for the first dashboard version and clearly label it as a model contribution estimate. Later, refine to probability-scale SHAP if needed.

Recommended first implementation:

```text
Start with GLMNet contribution difference.
Add global weighting.
Preserve sign.
Build LLM explanations from top weighted signed drivers.
Validate explanations manually on selected members.
```

## Driver Classification Rules

Each member-feature contribution should be classified.

Suggested logic:

```text
if signed_driver_priority > positive_threshold:
    explanation_role = "Primary positive driver" or "Supporting positive driver"
elif signed_driver_priority < negative_threshold:
    explanation_role = "Offsetting factor"
elif abs(member_feature_contribution) is high but global importance is low:
    explanation_role = "Context only"
else:
    explanation_role = "Not emphasized"
```

Potential thresholds:

| Role | Suggested rule |
|---|---|
| Primary positive driver | Top 3 positive `signed_driver_priority` values |
| Supporting positive driver | Positive driver ranks 4-8 |
| Offsetting factor | Top 3 negative `signed_driver_priority` values |
| Context only | High local contribution but low global importance |
| Suppressed | Administrative or sensitive feature not suitable for narrative |

## LLM Guardrails

The LLM should be constrained by structured evidence.

### Required LLM Rules

1. Mention the member's benefit decile and benefit score.
2. Mention predicted ED risk with and without intervention.
3. Only describe features present in the attribution table.
4. Preserve contribution sign.
5. Do not describe negative contributors as reasons to target.
6. Prioritize features with high global importance and high local contribution.
7. Treat low-global-importance features as context only.
8. Do not claim causal certainty.
9. Do not claim the intervention will definitely prevent an ED visit.
10. Avoid overemphasizing demographic, geographic, or administrative features unless approved.

### Suggested LLM Prompt Template

```text
You are generating a care-management model explanation.

Use only the structured data provided.
The model estimates intervention benefit as:
benefit_score = predicted ED risk without intervention - predicted ED risk with intervention.

Positive feature contributions increase predicted benefit.
Negative feature contributions decrease predicted benefit.
Prioritize features that have both high local contribution and high global feature importance.
Do not overemphasize context-only features.
Do not imply causal certainty.

Member summary:
{member_summary}

Top positive drivers:
{top_positive_drivers}

Top negative drivers:
{top_negative_drivers}

Contextual features:
{contextual_features}

Write a concise explanation with:
1. Outreach recommendation summary
2. Main benefit drivers
3. Offsetting factors
4. Caveat
```

## Recommended Dashboard Tabs

### Tab 1: Member Explanation

Primary user-facing view.

Contains:

- Member summary
- Benefit score and decile
- Predicted ED risk with/without intervention
- LLM narrative
- Top positive and negative drivers
- Feature contribution waterfall

### Tab 2: Feature Driver Details

Analyst-facing view.

Contains:

- Full member-feature attribution table
- Local contribution
- Global importance
- Weighted driver priority
- Feature percentile
- Feature value versus population

### Tab 3: Population Context

Population-level view.

Contains:

- Risk tier by benefit group stacked bar chart
- Uplift decile summaries
- Global benefit driver rankings
- Distribution of benefit scores
- Examples of high risk/low benefit and moderate risk/high benefit members

### Tab 4: LLM Audit

Governance view.

Contains:

- LLM input JSON
- LLM generated explanation
- Features used in narrative
- Features suppressed from narrative
- Sign-check validation
- Human reviewer notes

## Example Dashboard Explanation Data

Example table for one member:

| Feature | Member value | Population avg | Percentile | Local contribution | Global rank | Signed priority | Role |
|---|---:|---:|---:|---:|---:|---:|---|
| `current_risk_score` | 72.1 | 43.7 | 96 | +0.0120 | 2 | +0.0108 | Primary positive driver |
| `ed_visits_last_6m` | 3 | 0.8 | 91 | +0.0090 | 1 | +0.0090 | Primary positive driver |
| `total_cost_last_6m` | 8200 | 4200 | 88 | +0.0065 | 5 | +0.0042 | Supporting positive driver |
| `pcp_visits_last_6m` | 0 | 2.1 | 12 | -0.0040 | 7 | -0.0020 | Offsetting factor |
| `county_County_B` | 1 | NA | NA | +0.0040 | 43 | +0.0003 | Context only |

Example LLM narrative:

```text
This member is a strong outreach candidate because they are in the top uplift decile
and have an estimated benefit score of 0.054. The model estimates ED risk of 8.4%
without intervention compared with 3.0% with intervention.

The strongest positive benefit drivers are current risk score, recent ED utilization,
and total cost in the last six months. These features are important in the model overall
and contribute more than average for this member.

PCP visit history slightly offsets the benefit estimate, but the positive drivers are
stronger overall. This explanation reflects model-based prioritization and should not be
interpreted as proof that intervention will prevent an ED visit.
```

## Implementation Phases

### Phase 1: Define Attribution Outputs

Tasks:

1. Confirm the selected model family for local attribution.
2. Decide whether first implementation uses GLMNet contribution difference or SHAP-style probability contributions.
3. Create member-level attribution table.
4. Create global benefit feature importance table.
5. Add feature display names and clinical categories.

Deliverables:

- `member_benefit_attribution.csv`
- `global_benefit_feature_importance.csv`
- `feature_display_dictionary.csv`

### Phase 2: Build Dashboard Data Layer

Tasks:

1. Join member scores to attribution data.
2. Calculate global-normalized importance.
3. Calculate signed and absolute driver priority.
4. Classify driver roles.
5. Create LLM-ready member explanation input.

Deliverables:

- `member_explanation_dashboard_data.csv`
- `member_llm_explanation_inputs.jsonl`

### Phase 3: Build Visuals

Tasks:

1. Member summary card
2. Benefit score waterfall
3. Diverging positive/negative driver chart
4. Feature value versus population markers
5. Risk tier versus benefit group chart
6. Global feature importance chart

Deliverables:

- Dashboard prototype
- Static PNG examples for README
- Example member explanation pages

### Phase 4: Add LLM Narrative

Tasks:

1. Build prompt template.
2. Generate explanations for top benefit members.
3. Add sign and feature-use validation.
4. Review explanations manually.
5. Add caveats and governance notes.

Deliverables:

- `member_llm_explanations.csv`
- LLM prompt template
- LLM validation summary

### Phase 5: Governance And Review

Tasks:

1. Confirm no sensitive/protected features are overemphasized.
2. Confirm no causal overclaims.
3. Confirm negative drivers are not described as targeting reasons.
4. Confirm explanations are stable across reruns.
5. Confirm clinical/business reviewers understand the narrative.

Deliverables:

- Explainability QA checklist
- Reviewer sign-off notes
- Final dashboard documentation

## Key Design Decisions

### Use Model-Relative Benefit Buckets

Do not use fixed idealized treatment-effect thresholds such as 40%, 20%, 5%, and 0% if the observed model benefits are smaller. Use model-relative buckets:

| Benefit group | Rule |
|---|---|
| High benefit | Uplift deciles 1-2 |
| Medium benefit | Uplift deciles 3-7 |
| Low benefit | Uplift deciles 8-10 |

### Preserve Sign Everywhere

Every local feature contribution should keep direction.

```text
positive = increases benefit
negative = decreases benefit
```

### Weight Local Explanations By Global Importance

The LLM should not treat all local contributions equally. The main explanation should prioritize features that are both globally meaningful and locally influential.

### Keep Context Separate From Drivers

Features can be useful context without being headline reasons for targeting.

Example:

```text
County or case manager may appear in the model, but unless globally important and approved
for narrative use, these should be context only.
```

### Explain Risk Versus Benefit

The dashboard should repeatedly reinforce:

```text
High risk does not always mean high benefit.
High benefit means the model estimates a larger reduction in ED risk under intervention.
```

## Open Questions

1. Should member-level attribution use GLMNet coefficient contribution or probability-scale SHAP?
2. Should X-learner explanations be separate from T-learner explanations?
3. Should demographic/geographic features be suppressed from LLM narrative by default?
4. Should explanations be generated only for top-decile members or all scored members?
5. Should the LLM narrative be stored as an output artifact or generated on demand?
6. Should the dashboard compare T-learner and X-learner explanations for the same member?
7. How should explanations be validated with clinical stakeholders?

## Recommended Next Step

The next best technical step is to create a prototype member-level attribution table for the selected GLMNet uplift model.

Minimum viable output:

```text
member_id
feature
feature_value
member_feature_contribution
global_feature_importance
signed_driver_priority
direction
explanation_role
rank_for_member
```

Once this table exists, the dashboard and LLM layer become much easier to build because every explanation can be grounded in structured data rather than free-form interpretation.
