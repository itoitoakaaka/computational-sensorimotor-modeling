import numpy as np

from bayesian_fit import credible_interval, grid_posterior, posterior_mean
from state_space import simulate_state_space


def main():
    targets = np.concatenate([np.zeros(10), np.ones(70), np.zeros(20)])

    true_a = 0.92
    true_b = 0.24

    observed = simulate_state_space(
        targets,
        retention=true_a,
        error_sensitivity=true_b,
        observation_noise_sd=0.04,
        random_state=7,
    )["observed"]

    posterior = grid_posterior(
        targets,
        observed,
        noise_sd=0.04,
    )

    a_mean = posterior_mean(
        posterior["retention_grid"],
        posterior["retention_marginal"],
    )
    b_mean = posterior_mean(
        posterior["error_sensitivity_grid"],
        posterior["error_sensitivity_marginal"],
    )

    a_ci = credible_interval(
        posterior["retention_grid"],
        posterior["retention_marginal"],
    )
    b_ci = credible_interval(
        posterior["error_sensitivity_grid"],
        posterior["error_sensitivity_marginal"],
    )

    print(f"True A: {true_a:.3f}")
    print(f"Posterior mean A: {a_mean:.3f}, 95% CI={a_ci}")
    print(f"True B: {true_b:.3f}")
    print(f"Posterior mean B: {b_mean:.3f}, 95% CI={b_ci}")


if __name__ == "__main__":
    main()
