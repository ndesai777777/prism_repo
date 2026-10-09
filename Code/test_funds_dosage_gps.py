"""Focused numerical checks for the self-contained dosage notebook.

Run: python Code/test_funds_dosage_gps.py
Only extract functions; do not execute the analysis or read patient data.
These are reference-calculation checks, not cross-language R parity tests.
"""

import ast
from pathlib import Path
import unittest

import nbformat
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def notebook_functions():
    notebook = nbformat.read(
        Path(__file__).with_name('PRISM_Dosage_Intervention_Counts_Funds.ipynb'),
        as_version=4,
    )
    namespace = {'np': np, 'norm': norm}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        parsed = ast.parse(cell.source)
        definitions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), cell.id, 'exec'), namespace)
    return namespace


class GPSNumericalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = notebook_functions()

    def test_positive_density_integrates_to_treatment_probability(self):
        gps = self.functions['positive_gps']
        probability = 0.37
        positive_mass = quad(lambda a: gps(a, probability, np.log(80), 0.5), 0, np.inf)[0]
        self.assertAlmostEqual(positive_mass, probability, places=8)
        self.assertAlmostEqual(positive_mass + (1 - probability), 1.0, places=8)

    def test_r_distance_fixture_boundaries_and_counter_weights(self):
        match = self.functions['match_on_gps']
        # Radius=10: doses at 40 and 60 are included; 61 is excluded.
        # At target 50, scaled dose distances are equal for 40 and 60.
        dose = np.array([40., 60., 61.])
        gps = np.array([0.125, 0.375, 0.25])  # exactly representable tie
        selected, distance, candidates = match(
            dose, gps, np.array([0.125, 0.375, 0.25, 0.125]),
            np.ones(3, dtype=bool), 50., delta_n=20., scale=0.5,
        )
        np.testing.assert_array_equal(candidates, [0, 1])
        np.testing.assert_array_equal(selected, [0, 1, 0, 0])
        np.testing.assert_allclose(distance, [5/21, 5/21, 5/21 + 0.25, 5/21])
        np.testing.assert_array_equal(np.bincount(selected, minlength=3), [3, 1, 0])

    def test_zero_branch_excludes_near_zero_positive_donor(self):
        match = self.functions['match_on_gps']
        dose = np.array([0., 0., 0.1, np.nan])
        selected, _, candidates = match(
            dose, np.array([0.2, 0.8, 0.51, np.nan]), np.array([0.51, 0.21]),
            np.array([True, True, False, False]), 0, delta_n=0, scale=1,
        )
        np.testing.assert_array_equal(candidates, [0, 1])
        np.testing.assert_array_equal(selected, [1, 0])

    def test_empty_and_out_of_range_doses_are_unmatched(self):
        match = self.functions['match_on_gps']
        for target, width in [(50, 2), (110, 100)]:
            selected, distance, candidates = match(
                [20., 100.], [0.01, 0.02], [0.015], [True, True], target, width, 0.5
            )
            self.assertEqual(len(candidates), 0)
            self.assertEqual(selected[0], -1)
            self.assertTrue(np.isnan(distance[0]))

    def test_weighted_local_linear_recovers_linear_curve(self):
        smooth = self.functions['gaussian_local_linear']
        dose = np.array([20., 35., 80., 150., np.nan])
        outcome = 0.6 - 0.002 * dose
        counts = np.array([1, 5, 40, 3, 0])
        targets = np.array([25., 60., 120.])
        np.testing.assert_allclose(smooth(dose, outcome, counts, targets, 30),
                                   0.6 - 0.002 * targets, atol=1e-12)

    def test_count_weights_equal_explicit_replication(self):
        smooth = self.functions['gaussian_local_linear']
        dose = np.array([20., 35., 80., 150.])
        outcome = np.array([0., 1., 1., 0.])
        counts = np.array([1, 5, 4, 3])
        targets = [30., 60., 120.]
        actual = smooth(dose, outcome, counts, targets, 30)
        # Independent weighted least squares on explicitly replicated donors.
        repeated_dose, repeated_y = np.repeat(dose, counts), np.repeat(outcome, counts)
        expected = []
        for target in targets:
            design = np.column_stack([np.ones(len(repeated_dose)), repeated_dose - target])
            root_weights = np.sqrt(norm.pdf((repeated_dose - target) / 30))
            beta = np.linalg.lstsq(design * root_weights[:, None],
                                   repeated_y * root_weights, rcond=None)[0]
            expected.append(beta[0])
        np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
