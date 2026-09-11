# PRISM Synthetic Medicaid Dataset: Causal Equations and True Feature Importance

This note documents the causal data-generating process used in the Phase 2 revised synthetic Medicaid dataset. Its purpose is to make clear how the dataset defines true untreated risk, true treated risk, true treatment benefit, and true feature importance for validating uplift and causal models.

The key point is that the dataset contains known potential outcomes. We are not estimating the truth from observed outcomes; the synthetic generator explicitly defines it.

## 1. Notation

For member $i$:

$$
\begin{aligned}
T_i &= 1 && \text{if the member receives the intervention, and } 0 \text{ otherwise}, \\
Y_i &= \text{observed 90-day ED outcome}, \\
X_i &= \text{member covariates}, \\
\mu_{0i} &= P(Y_i = 1 \mid \operatorname{do}(T_i = 0), X_i), \\
\mu_{1i} &= P(Y_i = 1 \mid \operatorname{do}(T_i = 1), X_i).
\end{aligned}
$$

The individual treatment effect (ITE), also called the true benefit, is:

$$
\operatorname{ITE}_i = \text{true\_treatment\_effect}_i = \mu_{0i} - \mu_{1i}.
$$

Positive values mean the intervention reduces ED risk.

## 2. True Untreated Risk

The untreated risk is the probability that member $i$ would have a 90-day ED event if they did not receive the intervention:

$$
\mu_{0i} = \operatorname{sigmoid}(\alpha + f_{0i}),
\qquad
\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.
$$

The calibrated intercept is $\alpha = -3.847888502054383$.

The baseline-risk score is:

$$
\begin{aligned}
f_{0i} ={}& 2.35\,\mathbb{I}(\mathrm{ed\_visits\_last\_30d}_i > 0)
 + 0.62\log(1 + \mathrm{ed\_visits\_last\_30d}_i) \\
&+ 0.40\log(1 + \mathrm{prior\_ed\_visits\_6m}_i)
 + 0.32\log(1 + \mathrm{admits\_last\_6m}_i) \\
&+ 0.08\,\mathrm{age\_z}_i
 + 0.18\,\mathrm{age\_over\_65\_scaled}_i
 + 0.30\,\mathrm{currentRiskScore\_z}_i \\
&+ 0.17\,\mathrm{clinical\_score\_z}_i
 + 0.13\,\mathrm{sdoh\_score\_z}_i
 + 0.12\,\mathrm{living\_alone}_i \\
&+ 0.11\,\mathrm{low\_adherence\_z}_i
 + 0.16\,\mathrm{chf}_i
 + 0.12\,\mathrm{copd}_i \\
&+ 0.22\,\mathbb{I}(\mathrm{ed\_visits\_last\_30d}_i > 0)\,\mathbb{I}(\mathrm{admits\_last\_6m}_i > 0) \\
&+ 0.45\,\mathrm{legacy\_benefit\_risk\_proxy\_z}_i.
\end{aligned}
$$

The derived terms are:

$$
\begin{aligned}
\mathrm{prior\_ed\_visits\_6m}_i &= \max(\mathrm{ed\_visits\_last\_6m}_i - \mathrm{ed\_visits\_last\_30d}_i, 0), \\
\mathrm{age\_over\_65\_scaled}_i &= \frac{\max(\mathrm{age}_i - 65, 0)}{10}, \\
\mathrm{low\_adherence\_z}_i &= \operatorname{zscore}\!\left(\max(0.80 - \mathrm{med\_adherence\_pdc}_i, 0)\right), \\
\mathrm{legacy\_benefit\_risk\_proxy\_z}_i &= \operatorname{zscore}(\text{original baseline true treatment effect}_i).
\end{aligned}
$$

Interpretation: $\mu_{0i}$ is the member's true baseline ED risk without treatment. It is driven mostly by recent ED use, prior utilization, admissions, risk score, clinical burden, SDOH burden, adherence problems, age, CHF/COPD, and a retained legacy risk/benefit proxy.

The intercept $\alpha$ is not a clinical risk factor. It is a calibration constant chosen so the final generated dataset preserves the target overall 90-day ED outcome prevalence of 8%.

## 3. True Treatment Response

Treatment response is modeled as a relative risk reduction:

$$
\mathrm{response}_i = 0.04 + 0.31\,\operatorname{sigmoid}(h_i).
$$

Since $0 < \operatorname{sigmoid}(h_i) < 1$, the response lies approximately between 4% and 35%:

$$
0.04 < \mathrm{response}_i < 0.35.
$$

In the implemented Phase 2 generator, the treatment-response score is:

$$
\begin{aligned}
h_i ={}& -0.25
 + 0.65\,\mathrm{legacy\_effect\_z}_i
 + 0.24\,\mathrm{complex\_care\_program}_i \\
&+ 0.18\,\mathrm{behavioral\_health\_risk}_i
 + 0.17\,\mathrm{substance\_use}_i
 + 0.13\,\mathrm{sdoh\_score\_z}_i \\
&+ 0.12\,\mathrm{clinical\_score\_z}_i
 + 0.10\,\mathrm{low\_adherence\_z}_i
 + 0.08\,\mathrm{utilization\_score\_z}_i \\
&+ 0.18\,\mathrm{complex\_care\_program}_i\,\mathrm{utilization\_score\_z}_i \\
&+ 0.15\,\mathrm{behavioral\_health\_risk}_i\,\mathrm{sdoh\_score\_z}_i \\
&+ 0.10\,\mathrm{clinical\_score\_z}_i\,\mathrm{behavioral\_health\_risk}_i.
\end{aligned}
$$

Interpretation: members are modeled as benefiting more from the intervention when they have higher legacy response signal, complex-care program fit, behavioral-health complexity, substance-use risk, SDOH burden, clinical burden, adherence gaps, and utilization intensity.

## 4. True Treated Risk

The treated risk is:

$$
\mu_{1i} = \mu_{0i}\left(1 - \mathrm{response}_i\right).
$$

Because $\mathrm{response}_i$ is positive, treatment only reduces risk in this synthetic DGP:

$$
0 \leq \mu_{1i} \leq \mu_{0i} \leq 1.
$$

## 5. True Treatment Benefit

The individual treatment effect is:

$$
\begin{aligned}
\operatorname{ITE}_i
&= \mu_{0i} - \mu_{1i} \\
&= \mu_{0i} - \mu_{0i}(1 - \mathrm{response}_i) \\
&= \mu_{0i}\,\mathrm{response}_i.
\end{aligned}
$$

Therefore, absolute benefit depends on baseline risk and relative response:

$$
\text{true benefit} = \text{baseline risk} \times \text{relative response}.
$$

For example, if $\mu_{0i}=0.20$ and $\mathrm{response}_i=0.25$:

$$
\mu_{1i} = 0.20(1-0.25)=0.15,
\qquad
\operatorname{ITE}_i = 0.20-0.15=0.05.
$$

This member has a 5-percentage-point absolute reduction in 90-day ED risk.

## 6. Why Not Every Feature Appears in Every Equation

The generator is a causal structural model, not a kitchen-sink predictive model. Variables have different roles:

| Role | Example target |
| --- | --- |
| Baseline ED-risk driver | $\mu_{0i}$ |
| Treatment-response driver | $\mathrm{response}_i$ and $h_i$ |
| Treatment-assignment driver | $\mathrm{propensity\_score}_i$ |
| Preserved descriptive, alias, or audit field | No direct causal role |

If a variable is absent from $h_i$, it has no direct effect on relative treatment response. It may still affect absolute treatment benefit through $\mu_{0i}$:

$$
\operatorname{ITE}_i = \mu_{0i}\,\mathrm{response}_i.
$$

For example, ED utilization variables are major drivers of $\mu_{0i}$. Even when they do not directly modify $h_i$, they can increase absolute benefit because higher untreated risk creates more room for risk reduction.

## 7. True Feature Importance for Treatment Benefit

Because the synthetic generator defines the true treatment effect, true feature importance can be computed directly from the known equations. This provides a benchmark for checking whether uplift models recover the real drivers of benefit.

### 7.1 Coefficient Importance

The simplest measure is the coefficient in the response equation:

$$
h_i = \beta_0 + \beta_1x_{1i} + \cdots + \beta_px_{pi}.
$$

A large positive $\beta_j$ means the feature directly increases relative treatment response. This is easy to explain, but incomplete: it ignores prevalence, sigmoid nonlinearity, baseline risk, and any effect of the feature on baseline risk.

### 7.2 Average Marginal Effect on Relative Response

For a feature $x_j$ appearing in $h_i$ with coefficient $\beta_j$:

$$
\frac{\partial\,\mathrm{response}_i}{\partial x_j}
= 0.31\,\operatorname{sigmoid}(h_i)\left[1-\operatorname{sigmoid}(h_i)\right]\beta_j.
$$

Its global importance score is the average across members:

$$
\operatorname{AME}^{\mathrm{response}}_j
= \frac{1}{n}\sum_{i=1}^{n}
0.31\,\operatorname{sigmoid}(h_i)\left[1-\operatorname{sigmoid}(h_i)\right]\beta_j.
$$

### 7.3 Average Marginal Effect on True Treatment Benefit

The more important target for uplift modeling is absolute treatment benefit:

$$
\operatorname{ITE}_i = \mu_{0i}\,\mathrm{response}_i.
$$

If $x_j$ affects response but does not affect $\mu_{0i}$:

$$
\frac{\partial\,\operatorname{ITE}_i}{\partial x_j}
= \mu_{0i}\,0.31\,\operatorname{sigmoid}(h_i)
\left[1-\operatorname{sigmoid}(h_i)\right]\beta_j.
$$

The global importance score is:

$$
\operatorname{AME}^{\mathrm{ITE}}_j
= \frac{1}{n}\sum_{i=1}^{n}
\mu_{0i}\,0.31\,\operatorname{sigmoid}(h_i)
\left[1-\operatorname{sigmoid}(h_i)\right]\beta_j.
$$

This measure captures coefficient size, sigmoid nonlinearity, baseline risk, and the population feature distribution.

### 7.4 Total Marginal Effect When a Feature Affects Both Risk and Response

For a feature that affects both $\mu_{0i}$ and $\mathrm{response}_i$:

$$
\frac{\partial\,\operatorname{ITE}_i}{\partial x_j}
= \mathrm{response}_i\frac{\partial\mu_{0i}}{\partial x_j}
+ \mu_{0i}\frac{\partial\,\mathrm{response}_i}{\partial x_j}.
$$

For a coefficient $\gamma_j$ in the baseline-risk equation $f_{0i}$:

$$
\frac{\partial\mu_{0i}}{\partial x_j}
= \mu_{0i}(1-\mu_{0i})\gamma_j.
$$

Combining the two components gives:

$$
\frac{\partial\,\operatorname{ITE}_i}{\partial x_j}
= \mathrm{response}_i\,\mu_{0i}(1-\mu_{0i})\gamma_j
+ \mu_{0i}\,0.31\,\operatorname{sigmoid}(h_i)
\left[1-\operatorname{sigmoid}(h_i)\right]\beta_j.
$$

If a feature appears only in $f_{0i}$, then $\beta_j=0$. If it appears only in $h_i$, then $\gamma_j=0$. If it appears in both, both terms contribute.

## 8. Binary Features and Practical Importance

For binary features, such as $\mathrm{complex\_care\_program}_i$ or $\mathrm{behavioral\_health\_risk}_i$, an intuitive importance measure is the average counterfactual contrast:

$$
\operatorname{Importance}_j
= \frac{1}{n}\sum_{i=1}^{n}
\left[\operatorname{ITE}_i(x_j=1)-\operatorname{ITE}_i(x_j=0)\right].
$$

This answers: *How much would true treatment benefit change if this feature were switched on versus off, holding the rest of the member profile fixed?*

## 9. How to Validate Model-Recovered Feature Importance

A model can be evaluated by comparing its learned importance ranking with the true importance ranking from the generator:

1. Compute true feature importance from the known DGP.
2. Fit the uplift or causal model using the approved modeling feature set.
3. Compute model-estimated feature importance.
4. Compare the model ranking with the true ranking.

Useful comparison metrics include:

- **Spearman rank correlation:** Does the model rank the true drivers correctly?
- **Top-$k$ overlap:** How many of the true top 5 or top 10 drivers are recovered?
- **Signed agreement:** Does the model learn the correct direction of benefit?
- **Calibration by predicted benefit:** Do members predicted to benefit more actually have higher true ITE?

The cleanest validation target is the known synthetic truth, not observed $Y_i$, because observed outcomes are noisy Bernoulli draws:

$$
\text{true\_treatment\_effect}_i = \mu_{0i} - \mu_{1i}.
$$

## 10. Summary for Managers

The revised synthetic dataset gives every member a known true untreated risk, known true treated risk, and known true treatment benefit:

$$
\begin{aligned}
\text{Untreated risk:}\quad &\mu_{0i} = \operatorname{sigmoid}(\alpha+f_{0i}), \\
\text{Relative treatment response:}\quad &\mathrm{response}_i = 0.04+0.31\,\operatorname{sigmoid}(h_i), \\
\text{Treated risk:}\quad &\mu_{1i} = \mu_{0i}(1-\mathrm{response}_i), \\
\text{True benefit:}\quad &\text{true\_treatment\_effect}_i = \mu_{0i}-\mu_{1i} = \mu_{0i}\,\mathrm{response}_i.
\end{aligned}
$$

This makes the dataset useful for validating causal and uplift models because we can test whether a model recovers the actual synthetic drivers of benefit rather than merely predicting noisy observed outcomes.
