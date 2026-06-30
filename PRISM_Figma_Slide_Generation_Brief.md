# PRISM Data Team Presentation: Figma Slide Generation Brief

## Presentation Context

Create a professional 5-slide presentation for a 15-minute talk to the Data team about the PRISM project.

The presentation should tell a clear data-science story:

**Problem -> measurable outcome -> dataset -> modeling method -> decision support**

The tone should be polished, clear, and analytical. The audience is a Data team, so the slides can include modeling concepts, but they should remain intuitive and presentation-friendly. Avoid overcrowding slides with text. Use clean diagrams, comparison visuals, and simple flow charts.

## Visual Style Direction

- Professional, modern analytics presentation.
- Use a clean white or very light background.
- Use restrained accent colors such as navy, teal, muted green, and gold.
- Avoid heavy gradients or decorative visuals.
- Favor diagrams, structured tables, and clear visual hierarchy.
- Use concise slide text with speaker notes carrying more explanation.
- The project name PRISM should feel central and credible.

## Slide 1: Targeting Who We Can Help Most

### Main Message

PRISM is shifting from identifying who is most at risk to identifying who is most likely to benefit from a program.

### Slide Title

**Targeting Who We Can Help Most**

### On-Slide Content

- Traditional targeting focuses on members with the highest predicted risk.
- PRISM asks a different question: who is most likely to improve if they receive support?
- High risk does not always mean high program benefit.
- The goal is to prioritize members where intervention can make the biggest difference.

### Visual Direction

Create a side-by-side comparison:

```text
Old approach:
High predicted risk
-> prioritize for program

New PRISM approach:
High expected benefit
-> prioritize for program
```

Make the old approach visually neutral and the new PRISM approach visually emphasized.

### Speaker Notes

Previously, the focus was mainly on finding the people at greatest risk. That is useful, but it does not necessarily tell us who the program can actually help. PRISM reframes the targeting question around expected benefit. We want to know who is likely to have a better outcome because they received the program.

## Slide 2: What We Are Measuring

### Main Message

The project estimates benefit using a measurable 90-day ED outcome and program participation.

### Slide Title

**What We Are Measuring**

### On-Slide Content

- Dependent variable: `ed_outcome_90_days`
- This indicates whether a member had an ED outcome within 90 days.
- Treatment variable: whether the member received the program.
- The modeling goal is to estimate how program participation changes the likelihood of the ED outcome.

### Why This Outcome Matters

- ED utilization is tied to cost and resource planning.
- Reducing ED outcomes can indicate better member health.
- A 90-day window gives programs a concrete evaluation period.
- It creates a practical way to measure potential program impact.

### Visual Direction

Create a simple causal-style flow:

```text
Program received?
Yes / No
      ↓
ED outcome within 90 days
0 = no ED outcome
1 = ED outcome occurred
```

Include three small callouts around the outcome:

- Cost signal
- Clinical signal
- Planning signal

### Speaker Notes

Our dependent variable is `ed_outcome_90_days`, which tells us whether the member had an ED outcome within 90 days. This is useful because it connects to both financial and clinical impact. If we can identify members whose ED risk may be reduced by the program, we can support better health outcomes and help programs plan resources more effectively.

Important clarification: if `ed_outcome_90_days = 1` means an ED outcome occurred, then program benefit means reducing the probability of that outcome.

## Slide 3: How the Data Is Structured

### Main Message

Each row represents one member, with features, program participation, and outcome.

### Slide Title

**How the Data Is Structured**

### On-Slide Content

- Each row represents a person/member.
- Each member has feature information available before or around program targeting.
- Each member either received the program or did not.
- Each member has an observed 90-day ED outcome: `0` or `1`.

### Feature Categories

Show five feature categories:

- Demographics
- Clinical conditions
- Prior utilization
- Program/member history
- Social, behavioral, or risk-related factors

These category names can be adjusted if the final data dictionary uses different labels.

### Visual Direction

Create a clean table or row-level schema:

```text
Member features        Program received?        ED outcome 90 days
------------------------------------------------------------------
Demographics           Yes / No                 0 / 1
Clinical history
Prior utilization
Program history
Social/risk factors
```

The visual should communicate that one dataset row equals one member.

### Speaker Notes

The dataset is organized at the member level. Each row contains information about a person, whether they received the program, and whether they had the ED outcome within 90 days. The features give the model context about who the person is, their health history, prior utilization, and other risk-related factors.

## Slide 4: Estimating Program Benefit

### Main Message

A T-learner estimates two possible outcomes for each member: what may happen with the program and what may happen without it.

### Slide Title

**Estimating Program Benefit**

### On-Slide Content

- The T-learner uses two separate models.
- One model learns from members who received the program.
- One model learns from members who did not receive the program.
- For each member, the models estimate:
  - predicted ED risk if treated
  - predicted ED risk if untreated
- Estimated benefit is the difference between those two predictions.

### Formula

```text
Estimated benefit =
Predicted ED risk without program
-
Predicted ED risk with program
```

If the outcome is negative, like ED utilization:

```text
Higher estimated benefit = larger expected reduction in ED risk
```

### Visual Direction

Create a clear T-learner diagram:

```text
Member features
      ↓
 -------------------------
| Model A: Treated        | -> predicted ED risk with program
| Model B: Untreated      | -> predicted ED risk without program
 -------------------------
      ↓
Difference = estimated benefit
```

The model boxes should be visually parallel. The final difference calculation should be highlighted.

### Speaker Notes

The T-learner helps us estimate individualized program benefit. It does this by training one model on people who received the program and another model on people who did not. Then, for each member, we estimate their expected ED risk under both scenarios. The difference gives us an estimated benefit from receiving the program.

Keep the explanation intuitive. The goal is to explain the counterfactual comparison, not every causal assumption.

## Slide 5: From Model Output to Program Decisions

### Main Message

The final goal is not just a prediction score, but a decision-support tool that helps programs prioritize members and explain why.

### Slide Title

**From Model Output to Program Decisions**

### On-Slide Content

- Rank members by estimated benefit.
- Split members into benefit deciles.
- Focus on the highest-benefit groups.
- Use SHAP to explain what features are driving predicted benefit.
- Analyze which feature patterns are associated with the largest estimated benefit.
- Long-term goal: dashboard showing who should be prioritized and why.

### Brief SHAP Explanation

- SHAP explains how each feature contributes to an individual prediction.
- It helps identify which member characteristics push predicted benefit higher or lower.
- This makes the model more interpretable and useful for program teams.

### Visual Direction

Create a left-to-right or top-to-bottom workflow:

```text
Estimated benefit score
      ↓
Rank members
      ↓
Benefit deciles
      ↓
Top-benefit groups
      ↓
SHAP explanations
      ↓
Dashboard recommendation
```

Include a small dashboard mockup or simplified decision card showing:

- Member priority
- Estimated benefit
- Top drivers of benefit
- Recommendation rationale

### Speaker Notes

Once we estimate benefit, we can rank members and group them into deciles. The highest-benefit deciles become the groups we would most want to understand and potentially prioritize. SHAP gives us a way to explain what is driving the predicted benefit, so the final output is not just a black-box score. The eventual goal is a dashboard that identifies strong candidates for treatment and explains which features make them good candidates.

## Suggested 15-Minute Timing

- Slide 1: 2 minutes
- Slide 2: 3 minutes
- Slide 3: 2.5 minutes
- Slide 4: 4 minutes
- Slide 5: 3.5 minutes

## Overall Storyline

PRISM moves beyond predicting risk. It estimates who is most likely to benefit from intervention, using 90-day ED outcomes, member-level features, and a T-learner approach. The final goal is a transparent dashboard that helps programs prioritize members and understand why they are good candidates.

## Important Language Guidance

Use the phrase **estimated benefit** or **predicted benefit**, not guaranteed benefit.

Use **program participation** or **treatment indicator** instead of only saying independent variable.

Be clear that if `ed_outcome_90_days = 1` represents an ED outcome, then benefit means reducing the probability of that outcome.

