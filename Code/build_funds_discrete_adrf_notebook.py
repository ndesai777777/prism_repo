"""Build the self-contained Funds discrete ADRF notebook; optionally execute it.

Usage: py -3.14 Code/build_funds_discrete_adrf_notebook.py --execute
Requires numpy pandas scipy scikit-learn matplotlib nbformat nbclient ipykernel
threadpoolctl and xgboost (the existing PRISM helper imports xgboost).
"""
from pathlib import Path
import argparse
import textwrap
import nbformat as nbf

HERE = Path(__file__).resolve().parent
DESTINATION = HERE / 'PRISM_Discrete_Dose_Response_Residual_Correction_Funds.ipynb'


def build():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md(r'''
    # Funds combined population: discrete intervention dose-response with residual bias correction

    **Outcome:** any ED visit in the source `outcome_ed_90d` window (binary, matching the existing
    Funds dosage notebook). **X-axis:** exact observed integer intervention count. **Benefit:**
    reduction in ED-event risk relative to count zero, in **percentage points**, not visits prevented.

    This notebook implements the discrete indicator-basis version of Galagate (2016), equation
    (3.7), underlying the R package **causaldrf** residual-correction method. It is a Python
    adaptation of the mathematical estimator, not a claim of bit-for-bit R package parity.
    No EconML or R installation is required.

    The default target is **all eligible rows in the verified Funds training split**, not the
    highest-benefit subset. Set `population='all_eligible'` to deliberately target the complete
    eligible Funds population instead; that consumes the previous holdout. Each source row has
    equal weight. Repeated members are clustered, so this is an average over records, not over
    unique people.

    Run all cells from the repository root or `Code/`. Results go into a new
    `Outputs/Discrete_ADRF_Funds/Python/<population>/` directory. Existing notebooks and model
    outputs are only read. No row-level patient predictions are exported by default.
    ''')
    md(r'''
    ## What is reproduced, and what is adapted?

    The source notebook `PRISM_Dosage_Intervention_Counts_Funds.ipynb` supplies the eligibility
    rules, source-row alignment, seed-123 70/30 treatment/outcome-stratified split, binary ED
    definition, baseline variables, median imputation, missingness indicators, categorical
    encoding, and risk-tier normalization. Numeric imputation, scaling, and category learning
    are moved **inside each fitting fold** to prevent leakage.

    For a discrete count $t$, let $m_t(X)=E[Y\mid T=t,X]$ and $p_t(X)=P(T=t\mid X)$.
    The final risk is

    $$\widehat\mu(t)=\frac1n\sum_i\left[\widehat m_t(X_i)+
    \frac{I(T_i=t)}{\widehat p_t(X_i)}\{Y_i-\widehat m_t(X_i)\}\right].$$

    This is exactly the paper's equation (3.7) with $b_t(T)=I(T=t)$, because
    $E[bb^\top\mid X]=\operatorname{diag}(p_t(X))$. The paper explicitly permits this basis
    in Chapter 3, printed page 50. It estimates a separate mean at each count; it does not assume
    a linear dose effect or interpolate across missing counts.

    The R `aipwee_est` API supports degree 1 or 2 with OLS outcome regression. This notebook
    extends the basis to discrete indicators, uses regularized logistic outcome regressions for
    bounded binary-risk predictions, multinomial treatment probabilities, and member-grouped
    cross-fitting. A ridge linear-probability sensitivity fit is also reported. The inspected
    CRAN source uses an aggregated `Y_hat` in its residual and excludes zero doses when averaging
    slope terms; this notebook follows the dissertation's individual fitted residual equation
    instead. **It intentionally does not duplicate those R implementation details.**

    Corrected row contributions can fall outside [0,1]; they are estimating-function contributions,
    not validated individual counterfactual risks. This analysis estimates the population curve.
    Double robustness requires correct nuisance models in the appropriate limit plus causal
    identification assumptions; regularization and cross-fitting do not guarantee those conditions.

    Sources: [dissertation, Chapter 3](https://drum.lib.umd.edu/bitstreams/584a3d22-fffe-4271-8926-5a30a707da1e/download),
    [causaldrf 0.4.2 documentation](https://search.r-project.org/CRAN/refmans/causaldrf/html/aipwee_est.html),
    [R source inspected](https://github.com/cran/causaldrf/blob/master/R/aipwee_est.R).
    ''')
    code('''
    from pathlib import Path
    import sys, json, hashlib, platform, importlib.metadata
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    from threadpoolctl import threadpool_limits

    CODE_DIR = Path.cwd() if (Path.cwd() / '_prism_model_utils.py').exists() else Path.cwd() / 'Code'
    if not (CODE_DIR / '_prism_model_utils.py').exists():
        raise FileNotFoundError('Run from the repository root or Code folder.')
    ROOT = CODE_DIR.parent
    sys.path.insert(0, str(CODE_DIR))
    from _prism_model_utils import clean_names_simple, split_train_test, to_binary

    CONFIG = {
        'population': 'training',  # or 'all_eligible'; never selects high-benefit members
        'seed': 123, 'folds': 5,
        'minimum_dose_members': 30,  # outcome-blind reporting eligibility
        'minimum_member_ess': 30,
        'low_propensity': 0.01, 'maximum_low_p_fraction': 0.10,
        'maximum_abs_smd': 0.10, 'maximum_member_weight_share': 0.10,
        'propensity_C': 0.2, 'outcome_C': 0.2, 'ridge_alpha': 10.0,
        'multiplier_reps': 1500,
        'small_increment_pp': 1.0,  # illustrative equivalence margin, not a clinical decision rule
        'timing_confirmed': False,
        'run_sensitivities': True, 'write_outputs': True,
    }
    if CONFIG['population'] not in {'training', 'all_eligible'}:
        raise ValueError('Unknown target population.')
    RAW_PATH = ROOT / 'DataSets' / 'Funds_combined.csv'
    SCORE_DIR = ROOT / 'Outputs' / 'Causal-Forests_Funds' / 'Python'
    FULL_PATH = SCORE_DIR / 'causal_forest_scored_output.csv'
    TEST_PATH = SCORE_DIR / 'causal_forest_scored_test_output.csv'
    OUTPUT_DIR = ROOT / 'Outputs' / 'Discrete_ADRF_Funds' / 'Python' / CONFIG['population']
    for path in (RAW_PATH, FULL_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f'Required source-workflow input missing: {path}')
    print('Target:', CONFIG['population'], '| outcome: binary 90-day ED event | dose: intervention count')
    print('Outputs:', OUTPUT_DIR)
    ''')
    md('''
    ## 1. Reconstruct and verify the source population

    Treated eligibility is intervention flag 1 without opt-out 1; control eligibility is opt-out
    1 without intervention flag 1. The raw source count is the dosage even where the binary
    flag disagrees: disagreements are audited and tested separately. Missing counts are **not**
    assigned zero. Missing outcomes are filled with zero for main-analysis parity, with a fully
    refitted complete-outcome sensitivity analysis below.
    ''')
    code('''
    raw = pd.read_csv(RAW_PATH)
    raw.columns = clean_names_simple(raw.columns)
    if raw.columns.duplicated().any():
        raise ValueError('Column cleaning produced duplicate names.')
    scored = pd.read_csv(FULL_PATH, usecols=['analysis_row_id', 'member_id', 'intervention_flag', 'outcome_ed_90d'])
    saved_test = pd.read_csv(TEST_PATH, usecols=['analysis_row_id', 'member_id'])
    required = {'member_id', 'intervention_flag', 'optout_flag', 'outcome_ed_90d', 'interventioncount'}
    if not required.issubset(raw):
        raise KeyError(required - set(raw))
    source_treatment = to_binary(raw['intervention_flag']).where(lambda s: s.isin([0., 1.]))
    source_optout = to_binary(raw['optout_flag']).where(lambda s: s.isin([0., 1.]))
    treated = source_treatment.eq(1.) & ~source_optout.eq(1.)
    control = source_optout.eq(1.) & ~source_treatment.eq(1.)
    eligible = treated | control
    raw_model = raw.loc[eligible].reset_index(drop=True)
    expected_treatment = np.select([treated, control], [1., 0.], default=np.nan)[eligible.to_numpy()]
    expected_outcome = (pd.to_numeric(raw['outcome_ed_90d'], errors='coerce').fillna(0) > 0).astype(float)[eligible].to_numpy()
    checks = {
        'cohort size': len(raw_model) == len(scored),
        'sequential row IDs': scored.analysis_row_id.tolist() == list(range(len(scored))),
        'source member order': raw_model.member_id.astype(str).tolist() == scored.member_id.astype(str).tolist(),
        'treatment alignment': np.array_equal(expected_treatment, scored.intervention_flag.to_numpy()),
        'outcome alignment': np.array_equal(expected_outcome, scored.outcome_ed_90d.to_numpy()),
    }
    if not all(checks.values()):
        raise ValueError(f'Source/scored validation failed: {checks}')
    train, reconstructed_test = split_train_test(scored, train_fraction=.70, seed=123,
                                                stratify_columns=['intervention_flag', 'outcome_ed_90d'])
    if reconstructed_test.analysis_row_id.tolist() != saved_test.analysis_row_id.tolist():
        raise ValueError('Reconstructed split does not match the saved Funds test rows.')
    target = train if CONFIG['population'] == 'training' else scored
    cohort = raw_model.iloc[target.analysis_row_id.to_numpy(dtype=int)].copy().reset_index(drop=True)
    cohort.insert(0, 'analysis_row_id', target.analysis_row_id.to_numpy(dtype=int))
    cohort['received_treatment'] = target.intervention_flag.to_numpy(dtype=int)
    if cohort.member_id.isna().any():
        raise ValueError('Missing member IDs prevent grouped fitting and uncertainty estimation.')
    raw_count = pd.to_numeric(cohort.interventioncount, errors='coerce')
    raw_y = pd.to_numeric(cohort.outcome_ed_90d, errors='coerce')
    if ((raw_count.dropna() < 0) | (raw_count.dropna() % 1 != 0)).any() or np.isinf(raw_count.dropna()).any():
        raise ValueError('Intervention count must be a finite nonnegative integer or missing.')
    if ((raw_y.dropna() < 0) | (raw_y.dropna() % 1 != 0)).any() or np.isinf(raw_y.dropna()).any():
        raise ValueError('ED outcome must be a finite nonnegative count or missing.')
    cohort['dose'] = raw_count
    cohort['outcome_missing_filled_zero'] = raw_y.isna()
    cohort['ed_event_90d'] = (raw_y.fillna(0) > 0).astype(int)
    cohort['flag_count_inconsistent'] = ((cohort.received_treatment.eq(0) & raw_count.gt(0))
                                        | (cohort.received_treatment.eq(1) & raw_count.eq(0)))
    cohort['count_status'] = np.select([raw_count.isna(), raw_count.eq(0)],
                                       ['missing_or_unparseable', 'observed_zero'], default='observed_positive')
    dose_availability = cohort.groupby(['received_treatment', 'count_status']).agg(
        rows=('analysis_row_id', 'size'), unique_members=('member_id', 'nunique'),
        missing_outcomes=('outcome_missing_filled_zero', 'sum'),
    ).reset_index()
    analysis = cohort.loc[cohort.dose.notna()].copy().reset_index(drop=True)
    analysis['dose'] = analysis.dose.astype(int)
    if CONFIG['population'] == 'training' and not set(analysis.analysis_row_id).isdisjoint(saved_test.analysis_row_id):
        raise AssertionError('Holdout rows entered the analysis.')
    flow = pd.DataFrame([
        ('source rows', len(raw)), ('eligible source rows', len(raw_model)),
        ('verified training rows', len(train)), ('verified test rows', len(reconstructed_test)),
        ('chosen target before count exclusions', len(cohort)),
        ('missing count excluded', int(cohort.dose.isna().sum())),
        ('analysis rows', len(analysis)), ('analysis unique members', analysis.member_id.nunique()),
        ('analysis missing outcomes filled zero', int(analysis.outcome_missing_filled_zero.sum())),
        ('analysis flag/count disagreements', int(analysis.flag_count_inconsistent.sum())),
        ('members shared by original train/test', len(set(train.member_id.astype(str)) & set(saved_test.member_id.astype(str)))),
    ], columns=['stage', 'rows_or_members'])
    display(flow)
    display(dose_availability)
    print('Source split verified. Original holdout is row-based; nuisance folds below are member-based.')
    print('Target excludes missing-count rows; generalization to those rows is not established.')
    ''')
    md('''
    ## 2. Inspect count support and time ordering

    Only exact counts with at least the configured number of unique members receive a reported
    causal estimate. This screen does not use ED outcomes. All other known-count rows remain in
    the target; they form an **OTHER category solely in the treatment model**. No plotted count
    means "this many or more." Unobserved and sparse counts get gaps rather than invented estimates.

    The source does not provide intervention-level timestamps linking each counted intervention
    to the ED outcome window. The source engagement duration is audited below but cannot establish
    temporal ordering. Until count accumulation is confirmed to precede the outcome window,
    results are labeled **exploratory adjusted associations**. If ED events trigger later interventions
    in the same window, a longitudinal causal design is required. Counts must also represent
    sufficiently comparable interventions; different types or intensities can change the estimand.
    ''')
    code('''
    count_inventory = analysis.groupby('dose').agg(
        rows=('dose', 'size'), members=('member_id', 'nunique'),
        events=('ed_event_90d', 'sum'), observed_risk=('ed_event_90d', 'mean'),
        missing_outcomes=('outcome_missing_filled_zero', 'sum'),
        flag_disagreements=('flag_count_inconsistent', 'sum'),
    ).reset_index()
    count_inventory['eligible_for_estimation'] = count_inventory.members.ge(CONFIG['minimum_dose_members'])
    levels = count_inventory.loc[count_inventory.eligible_for_estimation, 'dose'].to_numpy(dtype=int)
    if 0 not in levels or len(levels) < 2:
        raise ValueError('Insufficient member support for zero and a positive count.')
    display(count_inventory)
    if 'engagement_length' in cohort:
        days = pd.to_numeric(cohort.engagement_length, errors='coerce')
        print('Source engagement duration >90 days:', int(days.gt(90).sum()), '(does not establish count/outcome ordering)')
    print('Modeled exact counts:', levels.tolist())
    print('Other-count target rows retained:', int((~analysis.dose.isin(levels)).sum()))
    print('Timing confirmed:', CONFIG['timing_confirmed'])
    if not CONFIG['timing_confirmed']:
        print('EXPLORATORY: do not interpret this run as an identified causal dose prescription.')
    ''')
    md('''
    ## 3. Estimator implementation

    Five folds keep every member wholly in either fitting or prediction for each nuisance fit.
    Neither the binary intervention flag, opt-out flag, engagement duration, successful interventions,
    dates after engagement, existing benefit scores, nor any follow-up outcome enters the covariates.
    The baseline names match the source, but their measurement timing still needs domain confirmation.
    Class balancing is not used because it would alter the exposure probabilities.

    The main estimate has **no propensity clipping or trimming**. Very small probabilities are exposed
    through diagnostics. Confidence intervals use member-clustered, cross-fitted estimating scores;
    simultaneous 95% bands use a Gaussian cluster multiplier approximation over the displayed family
    of contrasts. These are asymptotic intervals, not a full nuisance-refitting bootstrap. Their
    reliability requires adequate overlap, enough independent members, and suitable nuisance fits.
    Double robustness of the point estimate alone does **not** guarantee valid confidence intervals
    when only one nuisance model is correct; the influence approximation needs stronger convergence
    conditions. Flagged estimates and intervals are diagnostic, not decision-ready evidence.
    ''')
    # Embed the implementation directly; the notebook remains self-contained.
    source = (HERE / '_funds_discrete_adrf.py').read_text(encoding='utf-8')
    # Break into readable cells at function boundaries without altering source.
    chunks = source.split('\ndef ')
    code(chunks[0])
    for chunk in chunks[1:]:
        code('def ' + chunk)
    md('''
    ## 4. Fit the discrete dose-response model

    For each count, the outcome model predicts risk for the same entire target population.
    Only observed recipients of that exact count contribute that count's weighted residual.
    The final result averages those contributions over all target rows. A single-outcome-class
    training fold uses an explicitly logged smoothed constant; it is not silently dropped.
    ''')
    code('''
    with threadpool_limits(limits=2):
        main = fit_discrete_adrf(analysis, levels, CONFIG)
        curve, adjacent, balance = summarize_result(main, CONFIG)
    display(main['audit'])
    print('Members with rows in more than one prediction fold:',
          int(pd.DataFrame({'member': analysis.member_id, 'fold': main['fold']}).groupby('member').fold.nunique().gt(1).sum()))
    display(curve[['dose', 'observed_rows', 'observed_members', 'outcome_only_risk',
                   'residual_correction', 'ed_risk', 'benefit_pp', 'benefit_ci_low_pp',
                   'benefit_ci_high_pp', 'member_ess', 'max_abs_smd', 'support_flags', 'contrast_pass']])
    ''')
    md('''
    ## 5. Diagnose support before interpreting benefit

    Weighted balance compares each dose's donors to the **same whole target population**, using
    target-population standard deviations. ESS aggregates donor weights by member. The checks
    are conservative screening rules, not a proof of identification. A benefit-versus-zero
    comparison must pass checks at **both** the dose and zero reference. Estimated risks outside
    [0,1] are retained and flagged, never clipped into apparent validity.
    ''')
    code('''
    diagnostic_columns = ['dose', 'propensity_min', 'propensity_p01', 'propensity_median',
                          'low_propensity_fraction', 'member_ess', 'max_member_weight_share',
                          'ipw_weight_sum_over_n', 'max_abs_smd', 'support_pass', 'risk_in_bounds']
    display(curve[diagnostic_columns])
    worst_balance = balance.assign(abs_smd=balance.weighted_smd.abs()).sort_values('abs_smd', ascending=False).head(15)
    display(worst_balance)
    print(f"{int(curve.contrast_pass.sum())}/{len(curve)} dose-versus-zero contrasts pass the configured screens.")
    if not curve.contrast_pass.any():
        print('NO SUPPORTED DOSE CONTRASTS: this run cannot establish a dosage plateau or optimal count.')
    print('Unconstrained risk estimates outside [0,1]:', int((~curve.risk_in_bounds).sum()))
    ''')
    md('''
    ## 6. Sensitivity checks

    Compare the main result with an outcome-only estimate, a regularized linear-probability
    outcome model, stronger nuisance regularization, and a separately refitted complete-outcome
    population. Also refit after excluding disagreements between source count and treatment flag.
    The two exclusions change the target population; their shifts cannot be attributed solely to
    estimator sensitivity. The exact reporting grid is held fixed across these checks.

    Propensity floors below are **sensitivity estimates only**, using the same out-of-fold nuisance
    predictions. Flooring changes the estimating equation and can introduce bias. Main estimates,
    support flags, and intervals always use the unfloored probabilities.
    ''')
    code('''
    sensitivity_tables = []
    for floor in [None, 0.005, 0.01, 0.02]:
        scores = discrete_scores(analysis.ed_event_90d.to_numpy(), analysis.dose.to_numpy(),
                                 levels, main['m'], main['p'], floor=floor)
        risks = scores.mean(axis=0)
        sensitivity_tables.append(pd.DataFrame({
            'scenario': 'main_unfloored' if floor is None else f'propensity_floor_{floor}',
            'dose': levels, 'ed_risk': risks, 'benefit_pp': 100 * (risks[list(levels).index(0)] - risks),
            'target_rows': len(analysis), 'status': 'completed',
        }))
    sensitivity_audits = []
    if CONFIG['run_sensitivities']:
        scenarios = [
            ('linear_probability_ridge', analysis, CONFIG, 'ridge'),
            ('stronger_regularization', analysis, {**CONFIG, 'propensity_C': 0.05, 'outcome_C': 0.05}, 'logistic'),
            ('observed_outcomes_only_changed_target', analysis.loc[~analysis.outcome_missing_filled_zero], CONFIG, 'logistic'),
            ('consistent_flags_only_changed_target', analysis.loc[~analysis.flag_count_inconsistent], CONFIG, 'logistic'),
        ]
        for name, subset, cfg, model in scenarios:
            print('Sensitivity:', name, flush=True)
            try:
                with threadpool_limits(limits=2):
                    fitted = fit_discrete_adrf(subset, levels, cfg, outcome_model=model, verbose=False)
                    summary, _, _ = summarize_result(fitted, cfg)
                summary['scenario'], summary['target_rows'], summary['status'] = name, len(subset), 'completed'
                sensitivity_tables.append(summary)
                sensitivity_audits.append(fitted['audit'].assign(scenario=name))
            except (ValueError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
                sensitivity_tables.append(pd.DataFrame({'scenario': [name], 'status': [f'FAILED: {exc}'],
                                                        'target_rows': [len(subset)]}))
                print('Sensitivity could not be estimated:', exc)
    sensitivity = pd.concat(sensitivity_tables, ignore_index=True)
    display(sensitivity.pivot_table(index='dose', columns='scenario', values='benefit_pp'))
    display(sensitivity[['scenario', 'status', 'target_rows']].drop_duplicates())
    ''')
    md('''
    ## 7. Population curves and incremental benefit

    Positive benefit means lower ED-event risk compared with zero interventions. No lines bridge
    count gaps and no monotonicity constraint forces benefit to increase. Gray crosses show the
    outcome-only standardized estimate; filled blue points pass support screens and orange open
    points are flagged. Error bars are pointwise 95% intervals. Orange/blue vertical bands in the
    benefit panel are wider simultaneous intervals. These diagnostics do not resolve timing or
    unmeasured confounding. Red crosses mark impossible risk point estimates. Shading marks the
    physically possible risk/benefit ranges; estimates are not moved into those ranges.

    A separate **shape-only zoom** below the full-range figure focuses on -25 to +25 percentage
    points. Change `BENEFIT_ZOOM_LIMITS_PP` in the plotting cell to adjust the view. Boundary
    triangles display the actual values of off-screen point estimates. Dotted guides join only
    consecutive integer counts whose estimates are inside the window; they are not a fitted or
    smoothed dose-response curve. Confidence intervals remain in the full-range figure; set
    `SHOW_ZOOM_INTERVALS=True` to also show them in the zoom. All diagnostic flags are retained.
    ''')
    code('''
    BENEFIT_ZOOM_LIMITS_PP = (-25.0, 25.0)  # Display range only; estimates are unchanged.
    SHOW_ZOOM_INTERVALS = False  # Full uncertainty remains visible in the overview above.
    label = 'Adjusted' if CONFIG['timing_confirmed'] else 'Exploratory adjusted'
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1]})
    for passed, color, marker in [(True, '#176b91', 'o'), (False, '#b66c24', 'o')]:
        mask = curve.contrast_pass.eq(passed)
        part = curve.loc[mask]
        if part.empty:
            continue
        name = 'Passes configured screens' if passed else 'Flagged: inspect diagnostics'
        face = color if passed else 'none'
        axes[0].errorbar(part.dose, 100 * part.ed_risk,
                         yerr=100 * np.vstack([part.ed_risk-part.risk_ci_low, part.risk_ci_high-part.ed_risk]),
                         fmt='none', ecolor=color, capsize=3)
        axes[0].scatter(part.dose, 100 * part.ed_risk, facecolors=face, edgecolors=color, label=name)
        axes[1].vlines(part.dose, part.benefit_sim_low_pp, part.benefit_sim_high_pp, color=color, alpha=.22, lw=5)
        axes[1].errorbar(part.dose, part.benefit_pp,
                         yerr=np.vstack([part.benefit_pp-part.benefit_ci_low_pp, part.benefit_ci_high_pp-part.benefit_pp]),
                         fmt='none', ecolor=color, capsize=3)
        axes[1].scatter(part.dose, part.benefit_pp, facecolors=face, edgecolors=color, label=name)
    axes[0].scatter(curve.dose, 100 * curve.outcome_only_risk, marker='x', color='#888888', label='Outcome model only')
    invalid = curve.loc[~curve.risk_in_bounds]
    if len(invalid):
        axes[0].scatter(invalid.dose, 100 * invalid.ed_risk, marker='x', color='#ad2525',
                        s=70, zorder=5, label='Impossible risk estimate: unstable correction')
    axes[0].axhspan(0, 100, color='#527f66', alpha=.06)
    axes[0].set_ylabel('Unconstrained ED-risk estimate (%)')
    screen_label = 'NO SUPPORTED DOSE CONTRASTS' if not curve.contrast_pass.any() else 'Inspect count-specific support flags'
    axes[0].set_title(f'{label} discrete dose-response: Funds {CONFIG["population"]} cohort\\n'
                      f'{len(analysis):,} rows | {analysis.member_id.nunique():,} member clusters | {screen_label}', loc='left')
    axes[0].legend(fontsize=9)
    axes[1].axhline(0, color='#555555', lw=1)
    axes[1].axhspan(-100, 100, color='#527f66', alpha=.06)
    axes[1].set_ylabel('ED risk reduction vs zero (pp)')
    axes[1].legend(fontsize=9)
    axes[2].bar(curve.dose, curve.member_ess, color='#8daebd', width=.7)
    axes[2].axhline(CONFIG['minimum_member_ess'], color='#b66c24', ls='--')
    axes[2].set(xlabel='Exact number of interventions', ylabel='Donor member ESS')
    axes[2].set_xticks(curve.dose)
    axes[2].tick_params(axis='x', rotation=90)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    plt.show()

    # Separate shape view so large residual corrections cannot compress every other point.
    low, high = map(float, BENEFIT_ZOOM_LIMITS_PP)
    if not np.isfinite([low, high]).all() or not low < 0 < high:
        raise ValueError('Zoom limits must be finite and straddle zero.')
    zoom_curve = curve.sort_values('dose').reset_index(drop=True)
    inside = zoom_curve.benefit_pp.between(low, high)
    zoom_fig, zoom_ax = plt.subplots(figsize=(13, 5.5))
    zoom_ax.axhspan(0, high, color='#397d66', alpha=.06)
    zoom_ax.axhline(0, color='#555555', lw=1)
    for j in range(len(zoom_curve) - 1):
        a, b = zoom_curve.iloc[j], zoom_curve.iloc[j + 1]
        if b.dose == a.dose + 1 and inside.iloc[j] and inside.iloc[j + 1]:
            zoom_ax.plot([a.dose, b.dose], [a.benefit_pp, b.benefit_pp],
                         color='#8a8a8a', ls=':', lw=1, zorder=1)
    for passed, color in [(True, '#176b91'), (False, '#b66c24')]:
        part = zoom_curve.loc[zoom_curve.contrast_pass.eq(passed)]
        visible = part.loc[part.benefit_pp.between(low, high)]
        if SHOW_ZOOM_INTERVALS:
            zoom_ax.vlines(part.dose, part.benefit_ci_low_pp, part.benefit_ci_high_pp,
                           color=color, alpha=.3, lw=1)
        if len(visible):
            zoom_ax.scatter(visible.dose, visible.benefit_pp, s=55, zorder=3,
                            facecolors=color if passed else 'none', edgecolors=color,
                            label='Passes screens' if passed else 'Flagged estimate')
            for row in visible.itertuples():
                label_offset = 8 if row.benefit_pp >= 0 else -12
                if row.benefit_pp > high - .12 * (high - low):
                    label_offset = -14
                elif row.benefit_pp < low + .12 * (high - low):
                    label_offset = 10
                zoom_ax.annotate(f'{row.benefit_pp:+.1f}', (row.dose, row.benefit_pp),
                                 xytext=(0, label_offset),
                                 textcoords='offset points', ha='center', fontsize=8, color=color)
    outside = zoom_curve.loc[~inside & np.isfinite(zoom_curve.benefit_pp)]
    padding = .025 * (high - low)
    for j, row in enumerate(outside.itertuples()):
        above = row.benefit_pp > high
        boundary = high - padding if above else low + padding
        zoom_ax.scatter(row.dose, boundary, marker='^' if above else 'v', s=65,
                        color='#ad2525', zorder=4,
                        label='Off-screen estimate (actual value labeled)' if j == 0 else None)
        zoom_ax.annotate(f'{row.benefit_pp:+.1f} pp', (row.dose, boundary),
                         xytext=(0, -15 if above else 10), textcoords='offset points',
                         ha='center', fontsize=8, color='#ad2525')
    zoom_ax.set_ylim(low, high)
    zoom_ax.set_xticks(zoom_curve.dose)
    zoom_ax.set(xlabel='Exact number of interventions', ylabel='ED risk reduction vs zero (pp)')
    zoom_ax.set_title(f'Benefit shape: zoom from {low:g} to {high:g} pp | {screen_label}\\n'
                      'Above zero = lower estimated ED risk; below zero = higher estimated ED risk', loc='left', fontsize=11)
    zoom_ax.grid(axis='y', alpha=.2)
    handles, labels = zoom_ax.get_legend_handles_labels()
    if handles:
        zoom_ax.legend(handles, labels, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3)
    interval_note = ('Pointwise intervals shown; intervals may extend beyond this view.' if SHOW_ZOOM_INTERVALS
                     else 'Shape-only view: confidence intervals are in the full-range figure above.')
    zoom_fig.text(.08, .015, f'{interval_note} Triangles mark off-screen values; no estimates were clipped.', fontsize=9)
    zoom_fig.tight_layout(rect=[0, .08, 1, 1])
    plt.show()

    display(adjacent)
    print('No automated optimal-dose recommendation is made.')
    print('An interval containing zero is inconclusive, not proof that benefit has plateaued.')
    print('The small-increment indicator requires a supported pair and its whole simultaneous interval within',
          f'+/-{CONFIG["small_increment_pp"]} percentage points; the margin needs substantive justification.')
    ''')
    md('''
    ## 8. Reproducible aggregate outputs and interpretation

    Exports contain the cohort audit, all count frequencies, risk/benefit curve, adjacent-count
    contrasts, covariate balance, fold diagnostics, sensitivity results, and input hashes/configuration.
    Intervals are not clipped to probability bounds. Sparse counts remain in the count inventory
    even though no effect is estimated there. An identified causal interpretation additionally
    needs well-defined interventions, consistency, no interference, baseline confounding control,
    valid temporal ordering, and positivity. Missing-to-zero outcomes assume missing records mean
    no ED event; compare the complete-outcome sensitivity before using results.

    Numerical tests live in `Code/test_funds_discrete_adrf.py`. They verify the discrete formula
    against the general matrix correction, both double-robustness identities on a known probability
    distribution, clustered covariance, and leakage-resistant preprocessing. These are Python
    mathematical validation tests; R parity has not been executed.
    ''')
    code('''
    def file_sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    if CONFIG['write_outputs']:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        exports = {
            'cohort_audit.csv': flow, 'count_inventory.csv': count_inventory,
            'dose_availability.csv': dose_availability,
            'discrete_adrf_curve.csv': curve, 'adjacent_count_benefit.csv': adjacent,
            'covariate_balance.csv': balance, 'crossfit_diagnostics.csv': main['audit'],
            'sensitivity_analysis.csv': sensitivity,
        }
        if sensitivity_audits:
            exports['sensitivity_crossfit_diagnostics.csv'] = pd.concat(sensitivity_audits, ignore_index=True)
        for name, table in exports.items():
            table.to_csv(OUTPUT_DIR / name, index=False)
        fig.savefig(OUTPUT_DIR / 'discrete_interventions_vs_ed_benefit.png', dpi=160, bbox_inches='tight')
        zoom_fig.savefig(OUTPUT_DIR / 'discrete_interventions_vs_ed_benefit_zoom.png', dpi=160, bbox_inches='tight')
        metadata = {
            'config': CONFIG, 'input_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in (RAW_PATH, FULL_PATH, TEST_PATH)},
            'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'nbformat']},
            'python': platform.python_version(), 'target_rows': len(analysis),
            'target_unique_members': int(analysis.member_id.nunique()), 'levels': levels.tolist(),
            'baseline_columns': NUMERIC_BASELINE + CATEGORICAL_BASELINE,
            'estimator': 'Galagate equation 3.7, discrete indicator basis, cross-fitted multi-arm AIPW',
            'r_parity': 'Mathematical adaptation; not a literal causaldrf aipwee_est port; R parity not executed',
            'inference': 'Member-cluster influence covariance and Gaussian multiplier simultaneous bands; no nuisance-refit bootstrap',
            'interpretation': label, 'estimand': 'row-average binary ED-event risk; benefit relative to exact count zero',
            'excluded_missing_counts': int(cohort.dose.isna().sum()),
            'supported_contrasts': int(curve.contrast_pass.sum()),
        }
        (OUTPUT_DIR / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('Saved aggregate results to', OUTPUT_DIR)
    run_summary = pd.DataFrame({
        'check': ['source cohort/split validation', 'analysis rows', 'missing-count exclusions',
                  'dose-versus-zero contrasts passing screens', 'unbounded risk estimates',
                  'timing confirmed', 'automatic optimal dose'],
        'result': ['passed', len(analysis), int(cohort.dose.isna().sum()),
                   int(curve.contrast_pass.sum()), int((~curve.risk_in_bounds).sum()),
                   CONFIG['timing_confirmed'], 'not established'],
    })
    display(run_summary)
    print('Complete. Review support flags and sensitivity results before interpreting the curve.')
    ''')
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'},
                         'title': 'Funds discrete intervention dose-response with residual bias correction'}
    nbf.validate(notebook)
    return notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--refresh-plots', action='store_true', help='Refresh saved plots using existing aggregate estimates, without refitting.')
    args = parser.parse_args()
    notebook = build()
    if args.refresh_plots:
        if args.execute:
            parser.error('Choose --execute or --refresh-plots, not both.')
        from nbclient import NotebookClient
        previous = nbf.read(DESTINATION, as_version=4)
        if len(previous.cells) != len(notebook.cells):
            raise ValueError('Notebook structure changed; use a full execution.')
        # Only the plotting discussion, plotting cell, and export cell may differ.
        plot_index = next(i for i, c in enumerate(notebook.cells)
                          if c.cell_type == 'code' and c.source.startswith('BENEFIT_ZOOM_LIMITS_PP'))
        for i, (old, new) in enumerate(zip(previous.cells, notebook.cells)):
            if old.source != new.source and i not in {plot_index - 1, plot_index, len(notebook.cells) - 1}:
                raise ValueError(f'Non-plot cell {i} changed; use full execution.')
        setup = next(c for c in notebook.cells if c.cell_type == 'code' and c.source.startswith('from pathlib import Path'))
        cohort = next(c for c in notebook.cells if c.cell_type == 'code' and c.source.startswith('raw = pd.read_csv'))
        replay = nbf.v4.new_notebook(cells=[
            nbf.v4.new_code_cell(setup.source), nbf.v4.new_code_cell(cohort.source),
            nbf.v4.new_code_cell("curve = pd.read_csv(OUTPUT_DIR / 'discrete_adrf_curve.csv')\nadjacent = pd.read_csv(OUTPUT_DIR / 'adjacent_count_benefit.csv')"),
            nbf.v4.new_code_cell(notebook.cells[plot_index].source),
            nbf.v4.new_code_cell("fig.savefig(OUTPUT_DIR / 'discrete_interventions_vs_ed_benefit.png', dpi=160, bbox_inches='tight')\nzoom_fig.savefig(OUTPUT_DIR / 'discrete_interventions_vs_ed_benefit_zoom.png', dpi=160, bbox_inches='tight')"),
        ])
        NotebookClient(replay, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(HERE)}}).execute()
        previous.cells[plot_index - 1].source = notebook.cells[plot_index - 1].source
        previous.cells[plot_index] = replay.cells[3]
        previous.cells[-1].source = notebook.cells[-1].source
        previous.cells[plot_index].metadata['plot_refresh'] = 'Rendered from saved aggregate estimates; no model refit.'
        nbf.validate(previous)
        nbf.write(previous, DESTINATION)
        print('Refreshed plots without refitting:', DESTINATION, flush=True)
        raise SystemExit(0)
    nbf.write(notebook, DESTINATION)
    print('Built:', DESTINATION, flush=True)
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(notebook, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(HERE)}})
        try:
            client.execute()
        finally:
            nbf.write(notebook, DESTINATION)
        print('Executed successfully:', DESTINATION, flush=True)
