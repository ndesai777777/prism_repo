# PRISM Retrospective Validation Plan

## Comparing PRISM and Percolator Using Historical Treatment Outcomes

## Objective

The objective of this analysis is to evaluate whether PRISM retrospectively identifies patients who historically demonstrated better outcomes following care management intervention than patients prioritized by the existing Percolator risk-based approach.

This is intended to be a **retrospective validation study** using observed historical outcomes. It is **not** intended to estimate what would have happened had PRISM been operationally deployed.

---

# Primary Research Question

> **Among patients who actually received care management, does PRISM identify patients who historically demonstrated better post-treatment ED outcomes than those prioritized by Percolator?**

---

# Proposed Methodology

## Step 1 – Apply PRISM Retrospectively

Run the PRISM model on the complete historical patient population.

Rank every patient according to predicted treatment benefit.

---

## Step 2 – Identify the PRISM High-Benefit Cohort

Select the top 100 patients according to the PRISM benefit score.

---

## Step 3 – Restrict to Historically Treated Patients

From the PRISM top 100 patients, identify only those who actually received care management historically.

For example:

- Top 100 PRISM patients
- 40 actually received intervention

These 40 patients become the **PRISM evaluation cohort**.

---

## Step 4 – Construct the Percolator Comparison Cohort

Identify the top 40 patients who were historically prioritized by Percolator and received care management.

These patients become the **Percolator comparison cohort**.

Both groups therefore consist entirely of patients who actually received intervention, allowing all outcome measurements to be based on observed historical data rather than model predictions.

---

# Outcome Definition

The primary endpoint is:

- **90-day ED outcome following care management intervention**

Baseline information used for adjustment will include:

- 30-day utilization before intervention
- Additional pre-treatment utilization measures
- Demographic characteristics
- Clinical characteristics
- Risk scores
- Other relevant baseline variables

> **Note:** This analysis does **not** compare identical time windows before and after treatment. The baseline utilization variables are used to adjust for differences in patient severity prior to intervention, while the primary outcome remains the observed 90-day ED outcome after intervention.

---
## Conceptual Model

The analysis will estimate the 90-day ED outcome as a function of:

```text
90-Day ED Outcome =
    Intercept
  + PRISM Cohort Indicator
  + Baseline 30-Day Utilization
  + Other Baseline Utilization Measures
  + Baseline Risk Score
  + Age
  + Other Clinical Characteristics
  + Random Error
```

The primary quantity of interest is the coefficient associated with the **PRISM Cohort Indicator**.

If this coefficient is statistically significant and indicates a lower adjusted 90-day ED outcome, it would suggest that PRISM identifies patients who historically experienced better outcomes following care management than those prioritized by Percolator.
# Planned Outputs

## 1. Cohort Summary

For each cohort:

- Number of patients
- Baseline utilization characteristics
- Baseline clinical characteristics
- Baseline demographic characteristics

---

## 2. Outcome Summary

For each cohort:

- Mean 90-day ED outcome
- Standard deviation
- 95% confidence interval

---

## 3. Adjusted Regression Results

Report:

- Estimated PRISM effect ($\beta_1$)
- Standard error
- 95% confidence interval
- p-value

---

## 4. Visualizations

Potential figures include:

- Baseline utilization comparison
- Distribution of 90-day ED outcomes
- Adjusted comparison of PRISM versus Percolator
- Forest plot of adjusted estimates

---

# Interpretation

If the PRISM cohort demonstrates significantly better adjusted 90-day ED outcomes than the Percolator cohort, the results would support the following conclusion:

> **Among patients who historically received care management, PRISM retrospectively identified patients who demonstrated better observed post-treatment outcomes than those prioritized by the historical Percolator approach.**

This would provide retrospective evidence that PRISM is more effective at identifying treatment-responsive patients.

---

# Important Limitation

This analysis **does not estimate the causal effect of deploying PRISM**.

Historically, treatment decisions were made using Percolator rather than PRISM.

Therefore, this study **cannot** claim:

> "PRISM would have reduced ED utilization by X."

Instead, it evaluates whether PRISM preferentially identifies patients who historically experienced better outcomes following care management.

---

# Proposed Conclusion

This retrospective validation uses **observed historical outcomes** rather than model-predicted treatment effects.

If successful, the analysis would demonstrate that PRISM consistently identifies patients who historically experienced better outcomes following care management than those prioritized by the existing Percolator risk-based strategy, while appropriately adjusting for differences in baseline utilization and clinical characteristics.

This would provide an independent historical validation of PRISM's ability to identify treatment-responsive patients without relying on PRISM's own predicted treatment effects as the evaluation metric.
