import unittest

import numpy as np

from bayesian_fit import grid_posterior
from kalman_filter import kalman_filter_1d
from state_space import simulate_state_space


class BayesianAndKalmanTests(unittest.TestCase):
    def test_bayesian_map_is_close_to_true_parameters(self):
        targets = np.concatenate([np.zeros(5), np.ones(80), np.zeros(20)])
        observed = simulate_state_space(
            targets,
            retention=0.90,
            error_sensitivity=0.25,
            observation_noise_sd=0.02,
            random_state=4,
        )["observed"]

        posterior = grid_posterior(
            targets,
            observed,
            retention_grid=np.linspace(0.80, 0.98, 40),
            error_sensitivity_grid=np.linspace(0.15, 0.35, 40),
            noise_sd=0.02,
        )

        self.assertLess(abs(posterior["map_retention"] - 0.90), 0.05)
        self.assertLess(abs(posterior["map_error_sensitivity"] - 0.25), 0.05)

    def test_kalman_filter_reduces_noise(self):
        rng = np.random.default_rng(5)
        n = 100
        true_state = np.zeros(n)
        for t in range(1, n):
            true_state[t] = 0.95 * true_state[t - 1] + rng.normal(0.0, 0.03)

        observed = true_state + rng.normal(0.0, 0.15, size=n)
        filtered = kalman_filter_1d(
            observed,
            a=0.95,
            process_var=0.03 ** 2,
            observation_var=0.15 ** 2,
        )["state"]

        observed_mse = np.mean((observed - true_state) ** 2)
        filtered_mse = np.mean((filtered - true_state) ** 2)

        self.assertLess(filtered_mse, observed_mse)


if __name__ == "__main__":
    unittest.main()
