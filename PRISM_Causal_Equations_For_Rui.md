# PRISM Synthetic Medicaid Dataset: Causal Equations and True Feature Importance

This note documents the causal data-generating process used in the Phase 2 revised synthetic Medicaid dataset. The purpose is to make clear how the dataset defines:

- true untreated risk,
- true treated risk,
- true treatment benefit,
- and true feature importance for validating uplift and causal models.

The key point is that the dataset contains known potential outcomes. That means we are not estimating the truth from observed outcomes; the synthetic generator explicitly defines the truth.

## 1. Notation

For member `i`:

```text
T_i = 1 if the member receives the intervention, 0 otherwise
Y_i = observed 90-day ED outcome
X_i = member covariates
mu0_i = P(Y_i = 1 | do(T_i = 0), X_i)
mu1_i = P(Y_i = 1 | do(T_i = 1), X_i)
```

The individual treatment effect, also called the true benefit or ITE, is:

```text
ITE_i = true_treatment_effect_i = mu0_i - mu1_i
```

Positive values mean the intervention reduces ED risk.

## 2. True Untreated Risk

The untreated risk is the probability that member `i` would have a 90-day ED event if they did not receive the intervention:

```text
mu0_i = sigmoid(alpha + f0_i)
```

where:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

The calibrated intercept is:

```text
alpha = -3.847888502054383
```

The baseline-risk score `f0_i` is:

```text
f0_i =
  2.35 * I(ed_visits_last_30d_i > 0)
  + 0.62 * log(1 + ed_visits_last_30d_i)
  + 0.40 * log(1 + prior_ed_visits_6m_i)
  + 0.32 * log(1 + admits_last_6m_i)
  + 0.08 * age_z_i
  + 0.18 * age_over_65_scaled_i
  + 0.30 * currentRiskScore_z_i
  + 0.17 * clinical_score_z_i
  + 0.13 * sdoh_score_z_i
  + 0.12 * living_alone_i
  + 0.11 * low_adherence_z_i
  + 0.16 * chf_i
  + 0.12 * copd_i
  + 0.22 * I(ed_visits_last_30d_i > 0) * I(admits_last_6m_i > 0)
  + 0.45 * legacy_benefit_risk_proxy_z_i
```

Derived terms:

```text
prior_ed_visits_6m_i = max(ed_visits_last_6m_i - ed_visits_last_30d_i, 0)
age_over_65_scaled_i = max(age_i - 65, 0) / 10
low_adherence_z_i = zscore(max(0.80 - med_adherence_pdc_i, 0))
legacy_benefit_risk_proxy_z_i = zscore(original baseline true_treatment_effect_i)
```

Interpretation: `mu0_i` is the member's true baseline ED risk without treatment. It is driven mostly by recent ED use, prior utilization, admissions, risk score, clinical burden, SDOH burden, adherence problems, age, CHF/COPD, and a retained legacy risk/benefit proxy from the original synthetic dataset.

The intercept `alpha` is not a clinical risk factor. It is a calibration constant chosen so that the final generated dataset preserves the target overall 90-day ED outcome prevalence of 8%.

## 3. True Treatment Response

Treatment response is modeled as a relative risk reduction:

```text
response_i = 0.04 + 0.31 * sigmoid(h_i)
```

This means the relative risk reduction ranges approximately from 4% to 35%.

In the implemented Phase 2 generator, the treatment-response score is:

```text
h_i =
  -0.25
  + 0.65 * legacy_effect_z_i
  + 0.24 * complex_care_program_i
  + 0.18 * behavioral_health_risk_i
  + 0.17 * substance_use_i
  + 0.13 * sdoh_score_z_i
  + 0.12 * clinical_score_z_i
  + 0.10 * low_adherence_z_i
  + 0.08 * utilization_score_z_i
  + 0.18 * complex_care_program_i * utilization_score_z_i
  + 0.15 * behavioral_health_risk_i * sdoh_score_z_i
  + 0.10 * clinical_score_z_i * behavioral_health_risk_i
```

Interpretation: members are modeled as benefiting more from the intervention when they have higher legacy response signal, complex-care program fit, behavioral health complexity, substance-use risk, SDOH burden, clinical burden, adherence gaps, and utilization intensity.

## 4. True Treated Risk

The treated risk is:

```text
mu1_i = mu0_i * (1 - response_i)
```

Because `response_i` is constrained to be positive, treatment only reduces risk in this synthetic DGP:

```text
0 <= mu1_i <= mu0_i <= 1
```

## 5. True Treatment Benefit

The individual treatment effect is:

```text
ITE_i = mu0_i - mu1_i
```

Substituting the definition of `mu1_i`:

```text
ITE_i = mu0_i - mu0_i * (1 - response_i)
```

which simplifies to:

```text
ITE_i = mu0_i * response_i
```

So the absolute benefit depends on two things:

```text
true benefit = baseline risk * relative response
```

Example:

```text
mu0_i = 0.20
response_i = 0.25

mu1_i = 0.20 * (1 - 0.25) = 0.15
ITE_i = 0.20 - 0.15 = 0.05
```

This member has a 5 percentage point absolute reduction in 90-day ED risk.

## 6. Why Not Every Feature Appears in Every Equation

The generator is a causal structural model, not a kitchen-sink predictive model. Different variables have different roles:

```text
Some variables affect baseline ED risk:       mu0_i
Some variables affect treatment response:     response_i and h_i
Some variables affect treatment assignment:   propensity_score_i
Some variables are aliases, audit fields, or preserved descriptive fields
```

If a variable is absent from `h_i`, it has no direct effect on relative treatment response. However, it may still affect absolute treatment benefit through `mu0_i`.

For example, ED utilization variables are major drivers of `mu0_i`. Even if they do not directly modify `h_i`, they can still increase absolute benefit because:

```text
ITE_i = mu0_i * response_i
```

Higher untreated risk creates more room for risk reduction.

## 7. True Feature Importance for Treatment Benefit

Because the synthetic generator defines the true treatment effect, we can compute true feature importance directly from the known equations. This gives us a principled benchmark for checking whether uplift models recover the real drivers of benefit.

There are several useful definitions.

## 7.1 Coefficient Importance

The simplest measure is the coefficient in the response equation:

```text
h_i = beta_0 + beta_1 x_1i + ... + beta_p x_pi
```

Large positive `beta_j` means the feature directly increases relative treatment response.

This is easy to explain, but incomplete because it ignores:

- how common the feature is,
- the nonlinear sigmoid transformation,
- baseline risk `mu0_i`,
- and whether the feature also affects baseline risk.

## 7.2 Average Marginal Effect on Relative Response

For a feature `x_j` that appears in `h_i` with coefficient `beta_j`, the marginal effect on relative response is:

```text
d(response_i) / d(x_j)
  = 0.31 * sigmoid(h_i) * (1 - sigmoid(h_i)) * beta_j
```

A global importance score can be calculated by averaging this across all members:

```text
AME_response_j =
  average_i [0.31 * sigmoid(h_i) * (1 - sigmoid(h_i)) * beta_j]
```

This tells us how strongly the feature changes relative treatment response.

## 7.3 Average Marginal Effect on True Treatment Benefit

The more important target for uplift modeling is the effect on absolute treatment benefit:

```text
ITE_i = mu0_i * response_i
```

If `x_j` affects response through `h_i` but does not affect `mu0_i`, then:

```text
d(ITE_i) / d(x_j)
  = mu0_i * 0.31 * sigmoid(h_i) * (1 - sigmoid(h_i)) * beta_j
```

The global importance score is:

```text
AME_ITE_j =
  average_i [mu0_i * 0.31 * sigmoid(h_i) * (1 - sigmoid(h_i)) * beta_j]
```

This is often the most principled definition of true global feature importance for benefit because it captures:

- coefficient size,
- sigmoid nonlinearity,
- baseline risk,
- and the feature distribution in the population.

## 7.4 Total Marginal Effect When a Feature Affects Both Risk and Response

Some features may affect both `mu0_i` and `response_i`. In that case, the total derivative has two parts:

```text
ITE_i = mu0_i * response_i
```

Therefore:

```text
d(ITE_i) / d(x_j)
  = response_i * d(mu0_i) / d(x_j)
    + mu0_i * d(response_i) / d(x_j)
```

For a feature with coefficient `gamma_j` in the baseline-risk equation `f0_i`:

```text
d(mu0_i) / d(x_j)
  = mu0_i * (1 - mu0_i) * gamma_j
```

For a feature with coefficient `beta_j` in the response equation `h_i`:

```text
d(response_i) / d(x_j)
  = 0.31 * sigmoid(h_i) * (1 - sigmoid(h_i)) * beta_j
```

So the total marginal effect is:

```text
d(ITE_i) / d(x_j)
  = response_i * mu0_i * (1 - mu0_i) * gamma_j
    + mu0_i * 0.31 * sigmoid(h_i) * (1 - sigmoid(h_i)) * beta_j
```

If a feature appears only in `f0_i`, then `beta_j = 0`.

If a feature appears only in `h_i`, then `gamma_j = 0`.

If a feature appears in both, both terms contribute.

## 8. Binary Features and Practical Importance

For binary features, such as `complex_care_program_i` or `behavioral_health_risk_i`, an even more intuitive importance measure is the average counterfactual contrast:

```text
Importance_j =
  average_i [ITE_i(x_j = 1) - ITE_i(x_j = 0)]
```

This answers:

```text
How much would true treatment benefit change if this feature were switched on versus off,
holding the rest of the member profile fixed?
```

This is often easier to explain to nontechnical audiences than a derivative.

## 9. How to Validate Model-Recovered Feature Importance

A model can be evaluated by comparing its learned importance ranking to the true importance ranking from the generator.

Recommended validation workflow:

```text
1. Compute true feature importance from the known DGP.
2. Fit the uplift or causal model using the approved modeling feature set.
3. Compute model-estimated feature importance.
4. Compare model ranking vs. true ranking.
```

Useful comparison metrics:

```text
Spearman rank correlation:
  Does the model rank the true drivers correctly?

Top-k overlap:
  How many of the true top 5 or top 10 drivers are recovered?

Signed agreement:
  Does the model learn the correct direction of benefit?

Calibration by predicted benefit:
  Do members predicted to benefit more actually have higher true ITE?
```

The cleanest validation target is not observed `Y_i`, because observed outcomes are noisy Bernoulli draws. The cleanest target is the known synthetic truth:

```text
true_treatment_effect_i = mu0_i - mu1_i
```

## 10. Summary for Managers

The revised synthetic dataset gives every member a known true untreated risk, known true treated risk, and known true treatment benefit.

```text
Untreated risk:
mu0_i = sigmoid(alpha + f0_i)

Relative treatment response:
response_i = 0.04 + 0.31 * sigmoid(h_i)

Treated risk:
mu1_i = mu0_i * (1 - response_i)

True benefit:
true_treatment_effect_i = mu0_i - mu1_i = mu0_i * response_i
```

This makes the dataset useful for validating causal and uplift models because we can check whether a model recovers the actual synthetic drivers of benefit, rather than merely predicting noisy observed outcomes.

