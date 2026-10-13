"""Discrete-basis version of Galagate (2016), equation 3.7.

The notebook builder embeds this source, so the delivered notebook is inspectable
and does not import this module. Tests import it directly. No patient data here.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning

NUMERIC_BASELINE = [
    'age', 'dual_elg', 'diabetes_flag', 'chf_flag', 'copd_flag', 'asthma_flag',
    'depression_flag', 'anxiety_flag', 'substance_use_flag', 'ckd_flag', 'bh_flag',
    'htn_flag', 'hyperlipidemia_flag', 'cad_flag', 'financial_strain_flag',
    'food_insecurity_flag', 'housing_instability_flag', 'transportation_barrier_flag',
    'healthliteracy_challenges_flag', 'op_visits_last_6m', 'ed_visits_last_30d',
    'ed_visits_last_6m', 'admits_last_6m', 'observation_stays_last_6m',
    'total_cost_last_6m', 'rx_count_last_6m', 'opioid_flag', 'polypharmacy_flag',
    'current_risk_score', 'percolator_score',
]
CATEGORICAL_BASELINE = ['client_contract', 'gender', 'county', 'state', 'plan_type', 'risk_tier']


def baseline_frame(frame):
    """Source notebook's numeric conversion and risk-tier normalization."""
    columns = NUMERIC_BASELINE + CATEGORICAL_BASELINE
    missing = sorted(set(columns) - set(frame))
    if missing:
        raise KeyError(f'Missing baseline variables: {missing}')
    x = frame[columns].copy()
    x[NUMERIC_BASELINE] = x[NUMERIC_BASELINE].apply(pd.to_numeric, errors='coerce')
    x[NUMERIC_BASELINE] = x[NUMERIC_BASELINE].replace([np.inf, -np.inf], np.nan)
    for col in CATEGORICAL_BASELINE:
        x[col] = x[col].astype('string').fillna('Missing').astype(str)
    x['risk_tier'] = (
        pd.to_numeric(frame['risk_tier'], errors='coerce')
        .where(lambda s: s.isin([1, 2, 3, 4, 5]))
        .astype('Int64').astype('string').fillna('Missing').astype(str)
    )
    # All numeric missingness indicators, matching the original notebook.
    for col in NUMERIC_BASELINE:
        x[col + '__missing'] = x[col].isna().astype(float)
    return x


def make_preprocessor(numeric=NUMERIC_BASELINE, categorical=CATEGORICAL_BASELINE):
    """Fit only on nuisance-training folds; unseen categories map to all zeros."""
    numeric = list(numeric)
    return ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), numeric + [c + '__missing' for c in numeric]),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(categorical)),
    ], sparse_threshold=0)


def discrete_scores(y, dose, levels, outcome_predictions, propensities, floor=None):
    r"""psi_i(t) = m_i(t) + I(T_i=t)/p_i(t) * (Y_i-m_i(t)).

    This is equation 3.7 with an indicator basis: E[BB'|X] is diagonal.
    Returned rows are estimating contributions, NOT corrected individual risks.
    A floor changes the estimator and is used only in labeled sensitivity runs.
    """
    y, dose, levels = np.asarray(y, float), np.asarray(dose), np.asarray(levels)
    m, p = np.asarray(outcome_predictions, float), np.asarray(propensities, float)
    if y.ndim != 1 or dose.shape != y.shape or m.shape != (len(y), len(levels)) or p.shape != m.shape:
        raise ValueError('Mismatched score dimensions.')
    if not (np.isfinite(y).all() and np.isfinite(m).all() and np.isfinite(p).all()):
        raise ValueError('Nonfinite scores or nuisance predictions.')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('This notebook expects a binary ED-event outcome.')
    if np.any(p <= 0) or np.any(p > 1) or np.any(p.sum(axis=1) > 1 + 1e-8):
        raise ValueError('Invalid treatment probabilities; no silent clipping allowed.')
    if floor is not None and not 0 < floor < 1:
        raise ValueError('Propensity floor must lie in (0, 1).')
    denominator = p if floor is None else np.maximum(p, floor)
    return m + (dose[:, None] == levels[None, :]) * (y[:, None] - m) / denominator


def basis_residual_correction(theta, observed_basis, second_moments, y):
    """General equation 3.7, using linear solves rather than explicit inverses."""
    theta, b, moments, y = map(np.asarray, (theta, observed_basis, second_moments, y))
    if theta.shape != b.shape or moments.shape != (len(y), b.shape[1], b.shape[1]):
        raise ValueError('Invalid basis/moment dimensions.')
    if not all(np.isfinite(v).all() for v in (theta, b, moments, y)):
        raise ValueError('Nonfinite basis inputs.')
    if np.any(np.linalg.eigvalsh(moments) <= 0):
        raise ValueError('Conditional second-moment matrix is not positive definite.')
    residual = y - np.sum(theta * b, axis=1)
    return theta + np.linalg.solve(moments, b[..., None])[..., 0] * residual[:, None]


def cluster_inference(scores, groups, multiplier_reps=1500, seed=123):
    """Row-average estimand, member-cluster covariance and simultaneous bands.

    Cross-fitted influence-function approximation; multipliers do not refit
    nuisance models and do not establish validity under model misspecification.
    """
    scores = np.asarray(scores, float)
    if scores.ndim == 1:
        scores = scores[:, None]
    groups = np.asarray(groups)
    if len(groups) != len(scores) or not np.isfinite(scores).all() or pd.isna(groups).any():
        raise ValueError('Invalid scores/groups for clustered inference.')
    _, code = np.unique(groups, return_inverse=True)
    n, k = scores.shape
    g = code.max() + 1
    if g < 2:
        raise ValueError('Need at least two independent member clusters.')
    estimate = scores.mean(axis=0)
    sums = np.zeros((g, k))
    np.add.at(sums, code, scores - estimate)
    scaled = sums * np.sqrt(g / (g - 1)) / n
    covariance = scaled.T @ scaled
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    critical = norm.ppf(0.975)
    active = se > 1e-14
    if active.any() and multiplier_reps:
        rng = np.random.default_rng(seed)
        maxima = []
        for start in range(0, multiplier_reps, 100):
            size = min(100, multiplier_reps - start)
            draws = rng.standard_normal((size, g)) @ scaled[:, active]
            maxima.extend(np.max(np.abs(draws / se[active]), axis=1))
        critical = max(critical, float(np.quantile(maxima, 0.95)))
    return {
        'estimate': estimate, 'se': se, 'covariance': covariance,
        'lo': estimate - norm.ppf(0.975) * se, 'hi': estimate + norm.ppf(0.975) * se,
        'sim_lo': estimate - critical * se, 'sim_hi': estimate + critical * se,
        'critical': critical, 'clusters': g,
    }


def grouped_folds(x, labels, groups, n_splits, seed):
    """Choose an outcome-blind split with every exposure category in training."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    classes = set(labels)
    for attempt in range(20):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + attempt)
        splits = list(splitter.split(x, labels, groups))
        if all(set(labels[a]) == classes for a, _ in splits):
            for a, b in splits:
                if not set(groups[a]).isdisjoint(groups[b]):
                    raise AssertionError('Member leakage between nuisance folds.')
            return splits
    raise ValueError('Could not find grouped folds containing all exposure classes; increase minimum dose members.')


def fit_discrete_adrf(frame, levels, config, outcome_model='logistic', verbose=True):
    """Out-of-fold multinomial propensity and per-dose outcome regressions."""
    frame = frame.reset_index(drop=True)
    levels = np.asarray(levels, dtype=int)
    if len(levels) < 2 or 0 not in levels or len(np.unique(levels)) != len(levels):
        raise ValueError('Need distinct dose levels including zero and at least one positive count.')
    x = baseline_frame(frame)
    y = frame['ed_event_90d'].to_numpy(dtype=int)
    dose = frame['dose'].to_numpy(dtype=int)
    groups = frame['member_id'].astype(str).to_numpy()
    # Rare counts remain in the target and fitting data as OTHER, not top-coded
    # onto a plotted intervention count. Its probability completes the multinomial.
    labels = np.where(np.isin(dose, levels), dose, -1)
    splits = grouped_folds(x, labels, groups, config['folds'], config['seed'])
    p, m = np.zeros((len(frame), len(levels))), np.zeros((len(frame), len(levels)))
    fold_id = np.full(len(frame), -1, dtype=int)
    audit = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for fold, (a, b) in enumerate(splits):
            prep = make_preprocessor()
            xa, xb = prep.fit_transform(x.iloc[a]), prep.transform(x.iloc[b])
            if not np.isfinite(xa).all() or not np.isfinite(xb).all():
                raise ValueError('Nonfinite preprocessed covariates.')
            treatment = LogisticRegression(C=config['propensity_C'], max_iter=4000, solver='lbfgs')
            treatment.fit(xa, labels[a])
            all_p = treatment.predict_proba(xb)
            for j, level in enumerate(levels):
                p[b, j] = all_p[:, list(treatment.classes_).index(level)]
                local = dose[a] == level
                if local.sum() < 2:
                    raise ValueError(f'Too few training observations for count {level}.')
                ya = y[a][local]
                single_class = len(np.unique(ya)) == 1
                if outcome_model == 'logistic' and not single_class:
                    model = LogisticRegression(C=config['outcome_C'], solver='lbfgs', max_iter=4000)
                    model.fit(xa[local], ya)
                    m[b, j] = model.predict_proba(xb)[:, list(model.classes_).index(1)]
                elif outcome_model == 'logistic':
                    # Explicit fallback for a fold with no events/non-events.
                    m[b, j] = (ya.sum() + 0.5) / (len(ya) + 1)
                elif outcome_model == 'ridge':
                    model = Ridge(alpha=config['ridge_alpha'], solver='lsqr')
                    model.fit(xa[local], ya)
                    m[b, j] = model.predict(xb)
                else:
                    raise ValueError('Unknown outcome model.')
                audit.append({
                    'fold': fold, 'dose': int(level), 'train_rows_at_dose': int(local.sum()),
                    'train_members_at_dose': len(np.unique(groups[a][local])),
                    'train_events': int(ya.sum()), 'single_class_outcome': single_class,
                    'features': xa.shape[1], 'propensity_iterations': int(treatment.n_iter_.max()),
                })
            fold_id[b] = fold
            if verbose:
                print(f'Fold {fold + 1}/{len(splits)}: {len(a):,} fitting rows, {len(b):,} held-out rows', flush=True)
    if (fold_id < 0).any():
        raise AssertionError('Some rows lack out-of-fold predictions.')
    psi = discrete_scores(y, dose, levels, m, p)
    return {'psi': psi, 'm': m, 'p': p, 'fold': fold_id, 'audit': pd.DataFrame(audit),
            'frame': frame, 'levels': levels, 'outcome_model': outcome_model}


def support_diagnostics(result, config):
    """Overlap and balance assessed against the same target at every dose."""
    frame, levels, p, m = (result[k] for k in ('frame', 'levels', 'p', 'm'))
    x = baseline_frame(frame)
    # Full-target encoding used ONLY for reporting balance; never feeds fitting.
    prep = make_preprocessor()
    z = prep.fit_transform(x)
    names = prep.get_feature_names_out()
    sd, mean = z.std(axis=0), z.mean(axis=0)
    varying = sd > 1e-10
    z, sd, mean, names = z[:, varying], sd[varying], mean[varying], names[varying]
    groups = frame['member_id'].astype(str).to_numpy()
    _, code = np.unique(groups, return_inverse=True)
    dose, y = frame['dose'].to_numpy(), frame['ed_event_90d'].to_numpy()
    rows, balances = [], []
    for j, level in enumerate(levels):
        mask = dose == level
        weights = mask / p[:, j]
        member_weights = np.bincount(code, weights=weights)
        ess = weights.sum() ** 2 / np.sum(member_weights ** 2)
        smd = (np.average(z, axis=0, weights=weights) - mean) / sd
        raw_smd = (z[mask].mean(axis=0) - mean) / sd
        max_smd = float(np.abs(smd).max(initial=0))
        low_fraction = float((p[:, j] < config['low_propensity']).mean())
        max_share = float(member_weights.max() / weights.sum())
        reasons = []
        if len(np.unique(groups[mask])) < config['minimum_dose_members']:
            reasons.append('few donor members')
        if ess < config['minimum_member_ess']:
            reasons.append('low member ESS')
        if low_fraction > config['maximum_low_p_fraction']:
            reasons.append('limited target overlap')
        if max_smd > config['maximum_abs_smd']:
            reasons.append('residual covariate imbalance')
        if max_share > config['maximum_member_weight_share']:
            reasons.append('dominant donor member')
        rows.append({
            'dose': int(level), 'observed_rows': int(mask.sum()),
            'observed_members': len(np.unique(groups[mask])), 'observed_ed_risk': float(y[mask].mean()),
            'propensity_min': float(p[:, j].min()), 'propensity_p01': float(np.quantile(p[:, j], .01)),
            'propensity_median': float(np.median(p[:, j])), 'low_propensity_fraction': low_fraction,
            'member_ess': ess, 'max_member_weight_share': max_share,
            'ipw_weight_sum_over_n': float(weights.mean()), 'max_abs_smd': max_smd,
            'dose_propensity_brier': float(np.mean((mask - p[:, j]) ** 2)),
            'outcome_brier_at_observed_dose': float(np.mean((y[mask] - m[mask, j]) ** 2)),
            'outcome_predictions_outside_01': float(((m[:, j] < 0) | (m[:, j] > 1)).mean()),
            'support_pass': not reasons, 'support_flags': '; '.join(reasons) or 'pass',
        })
        balances.extend({'dose': int(level), 'feature': str(name), 'raw_smd': float(raw), 'weighted_smd': float(value)}
                        for name, raw, value in zip(names, raw_smd, smd))
    return pd.DataFrame(rows), pd.DataFrame(balances)


def summarize_result(result, config):
    """Marginal risks, zero-dose benefits, and truly adjacent-dose contrasts."""
    psi, frame, levels = result['psi'], result['frame'], result['levels']
    group = frame['member_id'].astype(str).to_numpy()
    reference = list(levels).index(0)
    risk = cluster_inference(psi, group, config['multiplier_reps'], config['seed'])
    benefit_scores = psi[:, [reference]] - psi
    benefit = cluster_inference(benefit_scores, group, config['multiplier_reps'], config['seed'])
    curve = pd.DataFrame({'dose': levels, 'ed_risk': risk['estimate'],
                          'outcome_only_risk': result['m'].mean(axis=0),
                          'residual_correction': (psi - result['m']).mean(axis=0),
                          'risk_se': risk['se'], 'risk_ci_low': risk['lo'], 'risk_ci_high': risk['hi'],
                          'benefit_pp': 100 * benefit['estimate'],
                          'benefit_ci_low_pp': 100 * benefit['lo'], 'benefit_ci_high_pp': 100 * benefit['hi'],
                          'benefit_sim_low_pp': 100 * benefit['sim_lo'], 'benefit_sim_high_pp': 100 * benefit['sim_hi']})
    diagnostics, balance = support_diagnostics(result, config)
    curve = curve.merge(diagnostics, on='dose', validate='one_to_one')
    curve['risk_in_bounds'] = curve['ed_risk'].between(0, 1)
    base = curve.loc[curve.dose.eq(0)].iloc[0]
    curve['contrast_pass'] = curve['support_pass'] & curve['risk_in_bounds'] & bool(base.support_pass and base.risk_in_bounds)
    pairs = [(j, j + 1) for j in range(len(levels) - 1) if levels[j + 1] == levels[j] + 1]
    if pairs:
        delta = np.column_stack([psi[:, a] - psi[:, b] for a, b in pairs])
        increment = cluster_inference(delta, group, config['multiplier_reps'], config['seed'] + 1)
        adjacent = pd.DataFrame({
            'from_dose': [levels[a] for a, _ in pairs], 'to_dose': [levels[b] for _, b in pairs],
            'additional_reduction_pp': 100 * increment['estimate'],
            'ci_low_pp': 100 * increment['lo'], 'ci_high_pp': 100 * increment['hi'],
            'sim_low_pp': 100 * increment['sim_lo'], 'sim_high_pp': 100 * increment['sim_hi'],
            'pair_support_pass': [bool(curve.iloc[a].support_pass and curve.iloc[b].support_pass
                                       and curve.iloc[a].risk_in_bounds and curve.iloc[b].risk_in_bounds)
                                  for a, b in pairs],
        })
        # Failure to reject zero is NOT evidence of a plateau. Require the full
        # simultaneous interval to fit within a prespecified equivalence band.
        adjacent['small_increment_compatible'] = (
            adjacent['pair_support_pass']
            & adjacent.sim_low_pp.ge(-config['small_increment_pp'])
            & adjacent.sim_high_pp.le(config['small_increment_pp'])
        )
    else:
        adjacent = pd.DataFrame(columns=['from_dose', 'to_dose', 'additional_reduction_pp',
                                          'ci_low_pp', 'ci_high_pp', 'sim_low_pp', 'sim_high_pp',
                                          'pair_support_pass', 'small_increment_compatible'])
    return curve, adjacent, balance
