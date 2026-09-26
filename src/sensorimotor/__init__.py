"""Interpretable models for trial-by-trial sensorimotor behavior."""

from .analysis import fit_participants
from .bayesian import credible_interval, grid_posterior, posterior_mean
from .kalman import kalman_filter_1d
from .state_space import fit_state_space, simulate_state_space
from .trial_by_trial import fit_trial_by_trial_update

__all__ = [
    "credible_interval",
    "fit_participants",
    "fit_state_space",
    "fit_trial_by_trial_update",
    "grid_posterior",
    "kalman_filter_1d",
    "posterior_mean",
    "simulate_state_space",
]
