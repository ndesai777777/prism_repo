"""Numerical/structural tests; no patient data or R installation required.

Run: py -3.14 -m unittest discover -s Code -p test_funds_discrete_adrf.py -v
These verify the dissertation equations, not literal causaldrf package parity.
"""
import ast
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from _funds_discrete_adrf import (
    NUMERIC_BASELINE, CATEGORICAL_BASELINE, baseline_frame, make_preprocessor,
    discrete_scores, basis_residual_correction, cluster_inference,
    grouped_folds, fit_discrete_adrf, summarize_result,
    bin_intervention_counts, INTERVENTION_BIN_LABELS,
)


class DiscreteCorrectionTests(unittest.TestCase):
    def test_fixed_bin_boundaries_preserve_missing_and_index(self):
        counts = pd.Series([0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 500, np.nan],
                           index=np.arange(100, 112))
        original = counts.copy()
        actual = bin_intervention_counts(counts)
        expected = pd.Series([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, pd.NA],
                             index=counts.index, dtype='Int64', name='bin_code')
        pd.testing.assert_series_equal(actual, expected)
        pd.testing.assert_series_equal(counts, original)
        self.assertEqual(INTERVENTION_BIN_LABELS, ('0', '1-5', '6-10', '11-15', '16-20', '21+'))

    def test_binning_rejects_invalid_counts(self):
        for invalid in [-1, .5, np.inf, -np.inf, 'not a count']:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bin_intervention_counts([0, invalid])

    def test_discrete_formula_equals_general_matrix_equation_with_other_category(self):
        rng = np.random.default_rng(132)
        p = rng.dirichlet([2, 3, 4, 5], size=80)
        dose = np.array([rng.choice(4, p=row) for row in p])
        m = rng.uniform(.05, .9, size=p.shape)
        y = rng.binomial(1, .3, size=len(dose))
        basis = np.eye(4)[dose]
        moments = np.array([np.diag(row) for row in p])
        matrix = basis_residual_correction(m, basis, moments, y)
        np.testing.assert_allclose(matrix, discrete_scores(y, dose, np.arange(4), m, p), atol=1e-12)
        # Omitting OTHER from reported coordinates does not omit OTHER target rows.
        selected = [0, 2, 3]
        np.testing.assert_allclose(matrix[:, selected], discrete_scores(y, dose, selected, m[:, selected], p[:, selected]))

    def test_each_double_robustness_branch_exact_population_expectation(self):
        # Enumerate the complete X,T,Y distribution, avoiding Monte Carlo error.
        true_p = np.array([[.6, .3, .1], [.2, .3, .5]])
        true_m = np.array([[.45, .30, .20], [.70, .50, .35]])
        px = np.array([.4, .6])
        records = [(x, t, y) for x in range(2) for t in range(3) for y in range(2)]
        x, t, y = np.array(records).T
        mass = px[x] * true_p[x, t] * np.where(y == 1, true_m[x, t], 1 - true_m[x, t])
        target = px @ true_m
        wrong_m = np.full((len(y), 3), .15)
        wrong_p = np.tile([.4, .4, .2], (len(y), 1))
        for m, p in [(true_m[x], wrong_p), (wrong_m, true_p[x]), (true_m[x], true_p[x])]:
            estimate = mass @ discrete_scores(y, t, np.arange(3), m, p)
            np.testing.assert_allclose(estimate, target, atol=1e-12)
        biased = mass @ discrete_scores(y, t, np.arange(3), wrong_m, wrong_p)
        self.assertGreater(np.max(np.abs(biased - target)), .01)

    def test_zero_residual_gives_no_correction_and_floor_is_explicit(self):
        y, dose = np.array([0, 1]), np.array([0, 1])
        m = np.array([[0., .3], [.5, 1.]])
        p = np.array([[.1, .9], [.8, .2]])
        np.testing.assert_allclose(discrete_scores(y, dose, [0, 1], m, p), m)
        m = np.full((2, 2), .5)
        actual = discrete_scores(y, dose, [0, 1], m, p, floor=.3)
        expected = m.copy()
        expected[0, 0] -= .5 / .3
        expected[1, 1] += .5 / .3
        np.testing.assert_allclose(actual, expected)
        with self.assertRaises(ValueError):
            discrete_scores(y, dose, [0, 1], m, np.zeros((2, 2)))

    def test_linear_basis_closed_form(self):
        # W*[1,t]' = [(m^2+v-m*t)/v, (t-m)/v]'.
        dose = np.array([0., 2., 4.])
        mean, variance = np.array([1., 2., 3.]), np.array([2., 1., 3.])
        b = np.column_stack([np.ones(3), dose])
        moments = np.array([[[1., u], [u, u*u+v]] for u, v in zip(mean, variance)])
        theta = np.array([[.4, -.03], [.5, -.05], [.6, -.08]])
        y = np.array([0., 1., 0.])
        r = y - np.sum(theta*b, axis=1)
        expected = theta + np.column_stack([(mean**2+variance-mean*dose)/variance,
                                            (dose-mean)/variance]) * r[:, None]
        np.testing.assert_allclose(basis_residual_correction(theta, b, moments, y), expected)

    def test_cluster_covariance_matches_independent_hand_calculation(self):
        scores = np.array([[.2, .3], [.4, .1], [.8, .7], [.5, .6], [.1, .2]])
        groups = np.array(['a', 'a', 'b', 'c', 'c'])
        actual = cluster_inference(scores, groups, multiplier_reps=0)
        mean = scores.mean(axis=0)
        totals = np.array([(scores[groups == g] - mean).sum(axis=0) for g in ['a', 'b', 'c']])
        expected = 3/2 * totals.T @ totals / 25
        np.testing.assert_allclose(actual['covariance'], expected)
        # Consistent replication within the same member preserves row-average inference.
        repeated = cluster_inference(np.repeat(scores, 2, axis=0), np.repeat(groups, 2), multiplier_reps=0)
        np.testing.assert_allclose(actual['covariance'], repeated['covariance'])
        zero = cluster_inference(np.zeros(5), groups)
        self.assertEqual(zero['se'][0], 0)

    def test_contrast_accounts_for_pair_covariance_and_benefit_sign(self):
        psi = np.array([[.5, .3], [.6, .1], [.4, .2], [.3, .1]])
        groups = np.arange(4)
        joint = cluster_inference(psi, groups, multiplier_reps=0)
        contrast = cluster_inference(psi[:, 0] - psi[:, 1], groups, multiplier_reps=0)
        self.assertGreater(contrast['estimate'][0], 0)
        v = np.array([1, -1])
        self.assertAlmostEqual(contrast['se'][0]**2, float(v @ joint['covariance'] @ v))

    def test_preprocessing_learns_medians_only_from_training(self):
        train = pd.DataFrame({'x': [1., 3., np.nan], 'x__missing': [0., 0., 1.], 'c': ['a', 'b', 'a']})
        held = pd.DataFrame({'x': [1e9, np.nan], 'x__missing': [0., 1.], 'c': ['new', 'a']})
        prep = make_preprocessor(numeric=['x'], categorical=['c'])
        prep.fit(train)
        result = prep.transform(held)
        self.assertEqual(prep.named_transformers_['numeric']['impute'].statistics_[0], 2.)
        self.assertTrue(np.isfinite(result).all())
        self.assertNotIn('new', prep.named_transformers_['category'].categories_[0])

    def test_grouped_folds_never_split_a_member(self):
        groups = np.repeat(np.arange(90), 2)
        label = np.tile([0, 1, 2], 60)
        splits = grouped_folds(np.zeros((180, 1)), label, groups, 5, 123)
        validation = []
        for a, b in splits:
            self.assertTrue(set(groups[a]).isdisjoint(groups[b]))
            self.assertEqual(set(label[a]), {0, 1, 2})
            validation.extend(b)
        self.assertEqual(sorted(validation), list(range(180)))

    def test_end_to_end_synthetic_data_and_no_bridge_over_gap(self):
        rng = np.random.default_rng(19)
        n = 500
        frame = pd.DataFrame({name: rng.normal(size=n) for name in NUMERIC_BASELINE})
        for col in CATEGORICAL_BASELINE:
            frame[col] = rng.choice(['a', 'b'], size=n)
        frame['risk_tier'] = rng.choice([1, 2, 3], n)
        frame['member_id'] = np.repeat(np.arange(n // 2), 2)
        frame['dose'] = rng.choice([0, 1, 3, 8], n, p=[.3, .3, .3, .1])
        frame['ed_event_90d'] = rng.binomial(1, .4, n)
        cfg = {'folds': 3, 'seed': 123, 'propensity_C': .1, 'outcome_C': .1, 'ridge_alpha': 10.,
               'multiplier_reps': 100, 'minimum_dose_members': 10, 'minimum_member_ess': 10,
               'low_propensity': .01, 'maximum_low_p_fraction': .1, 'maximum_abs_smd': .1,
               'maximum_member_weight_share': .1, 'small_increment_pp': 1.}
        with threadpool_limits(limits=2):
            fitted = fit_discrete_adrf(frame, [0, 1, 3], cfg, verbose=False)
            curve, adjacent, balance = summarize_result(fitted, cfg)
        self.assertEqual(fitted['psi'].shape, (n, 3))
        self.assertEqual(list(curve.dose), [0, 1, 3])
        self.assertEqual(list(zip(adjacent.from_dose, adjacent.to_dose)), [(0, 1)])
        self.assertEqual(curve.loc[curve.dose.eq(0), 'benefit_pp'].iloc[0], 0)
        self.assertEqual(len(fitted['frame']), n)
        self.assertGreater(len(balance), 0)
        self.assertLessEqual(pd.DataFrame({'g': frame.member_id, 'f': fitted['fold']}).groupby('g').f.nunique().max(), 1)

        # Refit six nominal bins, including the open-ended group, on the same rows.
        binned = frame.copy()
        binned['exact_intervention_count'] = rng.choice([0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 50], n)
        binned['dose'] = bin_intervention_counts(binned.exact_intervention_count).astype(int)
        binned['intervention_bin'] = binned.dose.map(dict(enumerate(INTERVENTION_BIN_LABELS)))
        pd.testing.assert_frame_equal(baseline_frame(frame), baseline_frame(binned))
        with threadpool_limits(limits=2):
            bin_fit = fit_discrete_adrf(binned, np.arange(6), cfg, verbose=False)
            bin_curve, bin_adjacent, _ = summarize_result(bin_fit, cfg)
        self.assertEqual(bin_fit['psi'].shape, (n, 6))
        np.testing.assert_allclose(bin_fit['p'].sum(axis=1), 1)
        np.testing.assert_array_equal(bin_fit['frame'].ed_event_90d, frame.ed_event_90d)
        self.assertEqual(list(bin_curve.dose), list(range(6)))
        self.assertEqual(list(zip(bin_adjacent.from_dose, bin_adjacent.to_dose)),
                         [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        self.assertEqual(bin_curve.loc[bin_curve.dose.eq(0), 'benefit_pp'].iloc[0], 0)
        self.assertEqual(pd.DataFrame({'g': binned.member_id, 'f': bin_fit['fold']})
                         .groupby('g').f.nunique().max(), 1)

    def test_notebook_valid_and_embedded_functions_match_module(self):
        here = Path(__file__).resolve().parent
        notebook = nbformat.read(here / 'PRISM_Discrete_Dose_Response_Residual_Correction_Funds.ipynb', as_version=4)
        nbformat.validate(notebook)
        embedded = {}
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                tree = ast.parse(cell.source)
                embedded.update({node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)})
        module = ast.parse((here / '_funds_discrete_adrf.py').read_text(encoding='utf-8'))
        for node in module.body:
            if isinstance(node, ast.FunctionDef):
                self.assertEqual(ast.dump(node), embedded[node.name], node.name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
