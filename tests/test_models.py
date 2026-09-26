import unittest

import numpy as np

from state_space import fit_state_space, simulate_state_space
from trial_by_trial import fit_trial_by_trial_update


class ModelTests(unittest.TestCase):
    def test_state_space_parameter_recovery_without_noise(self):
        targets = np.concatenate([np.zeros(5), np.ones(80), np.zeros(20)])
        true_a = 0.90
        true_b = 0.25

        observed = simulate_state_space(
            targets,
            retention=true_a,
            error_sensitivity=true_b,
            random_state=1,
        )["observed"]

        fitted = fit_state_space(targets, observed)

        self.assertAlmostEqual(fitted["retention"], true_a, places=2)
        self.assertAlmostEqual(fitted["error_sensitivity"], true_b, places=2)

    def test_trial_by_trial_returns_finite_gain(self):
        targets = np.concatenate([np.zeros(5), np.ones(40)])
        observed = simulate_state_space(
            targets,
            retention=0.9,
            error_sensitivity=0.2,
            observation_noise_sd=0.01,
            random_state=2,
        )["observed"]

        fitted = fit_trial_by_trial_update(targets, observed)

        self.assertTrue(np.isfinite(fitted["gain"]))
        self.assertTrue(np.isfinite(fitted["r2"]))
        self.assertEqual(fitted["n_pairs"], len(targets) - 1)


if __name__ == "__main__":
    unittest.main()
