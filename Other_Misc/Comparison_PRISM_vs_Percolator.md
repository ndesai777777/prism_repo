# PRISM Retrospective Validation Plan

## Comparing PRISM and Percolator Using Historical Treatment Outcomes

---

# Objective

The objective of this analysis is to evaluate whether PRISM retrospectively identifies patients who historically demonstrated greater improvement following care management intervention than patients prioritized by the existing Percolator risk-based approach.

This is intended to be a **retrospective validation study** using observed historical outcomes. It is **not** intended to demonstrate what would have happened had PRISM been operationally deployed.

---

# Primary Research Question

> **Among patients who actually received care management, does PRISM identify patients who historically demonstrated greater improvement in emergency department utilization than those prioritized by Percolator?**

---

# Proposed Methodology

## Step 1 — Apply PRISM Retrospectively

Run the PRISM model on the complete historical patient population.

Rank every patient according to predicted treatment benefit.

---

## Step 2 — Identify the PRISM High-Benefit Cohort

Select the top 100 patients according to the PRISM benefit score.

---

## Step 3 — Restrict to Historically Treated Patients

From the PRISM top 100 patients, identify only those who actually received care management historically.

For example,

- Top 100 PRISM patients
- 40 actually received intervention

These 40 patients become the **PRISM evaluation cohort**.

---

## Step 4 — Construct the Percolator Comparison Cohort

Identify the top 40 patients who were historically prioritized by Percolator and received care management.

These patients become the **Percolator comparison cohort**.

Both groups therefore consist entirely of patients who actually received intervention, allowing all outcome measurements to be based on observed historical data rather than model predictions.

---

# Outcome Definition

The primary outcome is:

- **90-day ED outcome following care management intervention**

Baseline information will include:

- 30-day utilization before intervention
- additional pre-treatment utilization measures
- demographic characteristics
- clinical characteristics
- risk scores
- other relevant baseline variables

Importantly, this analysis does **not** compare equal time windows before and after treatment.

Instead, baseline utilization variables are used to adjust for pre-existing differences between the two groups while the primary endpoint remains the observed 90-day ED outcome.

---

# Statistical Analysis

A simple comparison of the average 90-day ED outcome between groups would likely be biased because the PRISM-selected and Percolator-selected patients may differ in important baseline characteristics.

Therefore, the primary analysis will adjust for these baseline differences using multivariable regression.

A conceptual model is

\[
ED_{90d,i}
=
\beta_0
+
\beta_1 PRISM_i
+
\beta_2 Util30_i
+
\beta_3 OtherUtil_i
+
\beta_4 RiskScore_i
+
\beta_5 Age_i
+
\cdots
+
\varepsilon_i
\]

where

- \(ED_{90d,i}\) = observed 90-day ED outcome
- \(PRISM_i\) = indicator that the patient belongs to the PRISM-selected cohort
- \(Util30_i\) = baseline 30-day utilization
- \(OtherUtil_i\) = additional baseline utilization variables
- \(RiskScore_i\) = baseline risk score
- \(Age_i\) = patient age
- \(\varepsilon_i\) = random error

The primary parameter of interest is

\[
\beta_1
\]

which estimates the difference in post-treatment ED outcome between the PRISM and Percolator cohorts after accounting for baseline differences.

---

# Planned Outputs

The analysis will generate:

## Cohort Summary

- Number of patients in each cohort
- Baseline utilization characteristics
- Baseline clinical characteristics
- Baseline demographic characteristics

---

## Outcome Summary

For each cohort:

- Mean 90-day ED outcome
- Standard deviation
- Confidence interval

---

## Adjusted Analysis

Regression output including:

- Estimated PRISM effect
- Standard error
- 95% confidence interval
- p-value

---

## Visualizations

Potential figures include:

- Baseline utilization comparison
- Distribution of 90-day ED outcomes
- Adjusted comparison of PRISM versus Percolator
- Forest plot of adjusted estimates

---

# Interpretation

If the PRISM cohort demonstrates significantly better adjusted 90-day ED outcomes than the Percolator cohort, the results would support the following conclusion:

> **Among patients who historically received care management, PRISM retrospectively identified patients who demonstrated greater observed improvement than those prioritized by the existing Percolator approach.**

This would provide retrospective evidence that PRISM is more effective at identifying treatment-responsive patients.

---

# Important Limitation

This analysis **does not estimate the causal effect of deploying PRISM**.

Historically, treatment decisions were made using Percolator, not PRISM.

Therefore, this study cannot claim:

> "PRISM would have reduced ED utilization by X."

Instead, it evaluates whether PRISM preferentially identifies patients who historically experienced better outcomes following care management.

---

# Proposed Conclusion

This retrospective validation uses observed historical outcomes rather than model-predicted treatment effects.

If successful, the analysis would demonstrate that PRISM consistently identifies patients who historically benefited more from care management than those prioritized by the existing Percolator risk-based strategy, while appropriately adjusting for differences in baseline utilization and clinical characteristics.
