from __future__ import annotations

import numpy as np


def kalman_filter_1d(
    observations,
    inputs=None,
    a=1.0,
    b=0.0,
    c=1.0,
    process_var=0.01,
    observation_var=0.05,
    initial_state=0.0,
    initial_var=1.0,
):
    """Run a scalar linear Kalman filter."""
    observations = np.asarray(observations, dtype=float)
    if inputs is None:
        inputs = np.zeros_like(observations)
    inputs = np.asarray(inputs, dtype=float)

    if observations.shape != inputs.shape:
        raise ValueError("observations and inputs must have the same shape")
    if process_var < 0 or observation_var <= 0:
        raise ValueError("invalid noise variance")

    n = len(observations)
    filtered_state = np.zeros(n, dtype=float)
    filtered_var = np.zeros(n, dtype=float)
    kalman_gain = np.zeros(n, dtype=float)
    state = float(initial_state)
    var = float(initial_var)

    for t in range(n):
        state_pred = a * state + b * inputs[t]
        var_pred = (a ** 2) * var + process_var

        innovation = observations[t] - c * state_pred
        innovation_var = (c ** 2) * var_pred + observation_var
        gain = var_pred * c / innovation_var

        state = state_pred + gain * innovation
        var = (1.0 - gain * c) * var_pred

        filtered_state[t] = state
        filtered_var[t] = var
        kalman_gain[t] = gain

    return {
        "state": filtered_state,
        "variance": filtered_var,
        "gain": kalman_gain,
    }
