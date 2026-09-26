import numpy as np

from sensorimotor.state_space import fit_state_space, simulate_state_space
from sensorimotor.trial_by_trial import fit_trial_by_trial_update


def test_state_space_parameter_recovery_without_noise():
    targets = np.concatenate([np.zeros(5), np.ones(80), np.zeros(20)])
    observed = simulate_state_space(
        targets,
        retention=0.90,
        error_sensitivity=0.25,
        random_state=1,
    )["observed"]
    fitted = fit_state_space(targets, observed)

    assert abs(fitted["retention"] - 0.90) < 0.01
    assert abs(fitted["error_sensitivity"] - 0.25) < 0.01


def test_trial_by_trial_returns_finite_gain():
    targets = np.concatenate([np.zeros(5), np.ones(40)])
    observed = simulate_state_space(
        targets,
        retention=0.9,
        error_sensitivity=0.2,
        observation_noise_sd=0.01,
        random_state=2,
    )["observed"]
    fitted = fit_trial_by_trial_update(targets, observed)

    assert np.isfinite(fitted["gain"])
    assert np.isfinite(fitted["r2"])
    assert fitted["n_pairs"] == len(targets) - 1
