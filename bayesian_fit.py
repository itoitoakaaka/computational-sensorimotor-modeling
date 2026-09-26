from __future__ import annotations

import numpy as np

from state_space import predict_observed


def gaussian_log_likelihood(observed, predicted, noise_sd):
    """Gaussian log likelihood up to an additive constant."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if noise_sd <= 0:
        raise ValueError("noise_sd must be positive")

    residual = observed - predicted
    return float(
        -0.5 * np.sum((residual / noise_sd) ** 2)
        - len(observed) * np.log(noise_sd)
    )


def grid_posterior(
    targets,
    observed,
    retention_grid=None,
    error_sensitivity_grid=None,
    noise_sd=0.05,
):
    """Estimate a transparent posterior over A and B on a regular grid.

    A uniform prior is used over the supplied parameter grid.
    """
    targets = np.asarray(targets, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if targets.shape != observed.shape:
        raise ValueError("targets and observed must have the same shape")

    if retention_grid is None:
        retention_grid = np.linspace(0.50, 0.999, 100)
    if error_sensitivity_grid is None:
        error_sensitivity_grid = np.linspace(0.001, 0.60, 100)

    retention_grid = np.asarray(retention_grid, dtype=float)
    error_sensitivity_grid = np.asarray(error_sensitivity_grid, dtype=float)

    log_posterior = np.empty(
        (len(retention_grid), len(error_sensitivity_grid)),
        dtype=float,
    )

    for i, retention in enumerate(retention_grid):
        for j, error_sensitivity in enumerate(error_sensitivity_grid):
            predicted = predict_observed(
                targets,
                retention=retention,
                error_sensitivity=error_sensitivity,
                initial_state=observed[0],
            )
            log_posterior[i, j] = gaussian_log_likelihood(
                observed,
                predicted,
                noise_sd=noise_sd,
            )

    log_posterior -= np.max(log_posterior)
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()

    p_retention = posterior.sum(axis=1)
    p_error = posterior.sum(axis=0)

    return {
        "retention_grid": retention_grid,
        "error_sensitivity_grid": error_sensitivity_grid,
        "posterior": posterior,
        "retention_marginal": p_retention,
        "error_sensitivity_marginal": p_error,
        "map_retention": float(
            retention_grid[np.unravel_index(np.argmax(posterior), posterior.shape)[0]]
        ),
        "map_error_sensitivity": float(
            error_sensitivity_grid[
                np.unravel_index(np.argmax(posterior), posterior.shape)[1]
            ]
        ),
    }


def posterior_mean(grid, marginal):
    grid = np.asarray(grid, dtype=float)
    marginal = np.asarray(marginal, dtype=float)
    return float(np.sum(grid * marginal))


def credible_interval(grid, marginal, level=0.95):
    """Return an equal-tailed credible interval."""
    grid = np.asarray(grid, dtype=float)
    marginal = np.asarray(marginal, dtype=float)
    cdf = np.cumsum(marginal)
    alpha = (1.0 - level) / 2.0

    lower = grid[np.searchsorted(cdf, alpha)]
    upper = grid[min(np.searchsorted(cdf, 1.0 - alpha), len(grid) - 1)]
    return float(lower), float(upper)
